"""Strong-model check: a Claude model drives APReceiptEnv through the Messages API.

Run before any training, on 2026-10-29 on the version 1 interface (versions A, B and C) and on 2026-10-30 on the final
v2 interface (version V2).

The agent gets what v1's agent got: the same six tools (schemas built from the environment's methods the way TRL
builds them), the same instructions, the same first user message (v1 `USER_PREFIX` + the reset page), the same
observations, the 40-call action limit and the horizon, under version 1's final pipeline settings (surfaces/3,
loop-guard/1). Version B appends rules 1-7 of the visible-rules draft (RULES_FILE, between its markers) to the
instructions. Version C is version B plus rule 8 (business time, its own marked block in the draft), a note on citing a
field's value exactly as shown (`CITATION_NOTE`) and the records the report must cite (`RECORDS_NOTE`). The draft is not
part of this release, so versions B and C run only where it is present. B_RULES_SHA256 pins the text version B ran with.

Where the translation to the Messages API changes what the agent sees or can do (recorded in each episode file):
  - Rendering: Qwen's chat template put the tool schemas in the system text and tool results in <tool_response>
    turns; here they are API tool definitions and tool_result blocks. The schema content is the same.
  - One tool call per turn (`disable_parallel_tool_use`). TRL executed every call in a turn.
  - Claude's tool calls always parse; Qwen could emit an unparseable call. A call TRL could parse but not execute
    (unknown tool, bad arguments) gets TRL's result text, str({"error": ...}), as here.
  - Sampling: the API no longer accepts `temperature`; its default (1.0) applies, as v1's 1.0 did. No seed.
  - Token budget: v1's 69,632 completion tokens (PIPELINE_MAX_COMPLETION_LENGTH, Qwen tokenizer) counted in Claude
    tokens: every token appended after the first request (model turns, thinking included, and tool results).
  - The episode stops when the environment is done; TRL let the model write one more turn, which changes nothing.
  - Refusal fallbacks are not used, and the SDK's own retries are off. A rate-limit or overload error is resent after
    10, 20, 40 ... s (at most 300 s per request, set 2026-10-29), then recorded as api_overload. Any other API error
    or a refusal ends the episode and is recorded. An episode is cut after 1,800 s of wall time (episode_timeout).

Version V2 (run 2026-10-30) is the final v2 interface: APReceiptEnvV2 on surfaces/4, its seven
tools (the six plus coverage; finish requires the report-level discrepancy_type), instructions/5 (SYSTEM_PROMPT_V2, the
rules and report requirements included), the v2 pipeline's limits (training/config.py, checked with require_pipeline:
completion budget PIPELINE_MAX_COMPLETION_LENGTH, here in Claude tokens; tool turns
PIPELINE_MAX_TOOL_CALLING_ITERATIONS), graded at the time by the default grader (g7) and by g8 since 2026-10-31 (see
below). Versions A, B and C are unchanged (v1's interface, surfaces/3, 69,632 Claude tokens) so that the 2026-10-29
sweep stays reproducible.

Run (the SDK is added for the run only; pyproject.toml and uv.lock are unchanged):
    uv run --frozen --with anthropic==2.3.0 python scripts/strong_model_check.py --case L1-2501-69 --version A
    uv run --frozen python scripts/strong_model_check.py --case L1-2501-69 --dry-run-reference   # no API, no cost

The API key is read from ANTHROPIC_API_KEY, or, when that is unset, from the ANTHROPIC_API_KEY line of a git-ignored
.env file in the repository root. It is never printed, logged or written.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import time

from transformers.utils import get_json_schema

from ap_sim.events.clock import ACTION_LIMIT, format_time
from ap_sim.generator.manifests import load
from ap_sim.grader.grader import grade
from ap_sim.grading import g8
from ap_sim.training import rewards as v2_rewards
from ap_sim.training import config as pipeline
from ap_sim.training.environment import AGENT_TOOLS, AGENT_TOOLS_V2, APReceiptEnv, APReceiptEnvV2
from ap_sim.training.loop_guard import LOOP_GUARD
from ap_sim.training.prompt import INSTRUCTIONS_VERSION, INSTRUCTIONS_VERSION_V2, SYSTEM_PROMPT, SYSTEM_PROMPT_V2

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "frozen" / "train_pool.json"   # never a test set
RULES_FILE = ROOT / "docs" / "VISIBLE_RULES_DRAFT.md"
ENV_FILE = ROOT / ".env"                           # git-ignored; read only for ANTHROPIC_API_KEY
USER_PREFIX = "Here is the case. Initial page:\n"   # training/config.py in version 1 of this environment, verbatim
SCHEMA = "surfaces/3"                              # v1 PIPELINE_SCHEMA (version 1's final pipeline settings)
TOKEN_BUDGET = 69_632                              # v1 PIPELINE_MAX_COMPLETION_LENGTH, here in Claude tokens
TURN_CAP = ACTION_LIMIT                            # v1 max_tool_calling_iterations
EXCLUDED = ("token_budget", "token_budget_rollback", "turn_cap", "infrastructure_failure")   # v1 rewards.EXCLUDED
# Since the grader and reward review of 2026-10-31, which adopted g8, the v2 interface (APReceiptEnvV2) is graded by
# g8, and only infrastructure_failure is excluded (v2 rewards.EXCLUDED, with turn_cap and the token-limit stops
# scored), exactly as in training; a call that raises is handled as TRL handles it (execute) and counted as a
# malformed tool call. The token limit is the engine's length stop (stop_reason max_tokens) or the budget, never the
# last token, as in training; an infrastructure failure is re-run once (with_infrastructure_rerun).
API_FAILURES = ("api_error", "api_overload", "api_refusal", "api_empty_response", "episode_timeout")   # no v1 analogue
BACKOFF_LIMIT = 300                                # s of rate-limit/overload backoff per request before api_overload
EPISODE_TIMEOUT = 1800                             # s of wall time; checked between turns
MODEL = "claude-sonnet-5-5"
PRICES = {"claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20}}   # $/MTok
THINKING = {"off": {"type": "between_tools"}, "adaptive": {"type": "adaptive", "display": "summarized"}}
B_RULES_SHA256 = "d39196039f273ef3b1a9764f4e73910d33dc4733e0fc26075f195c4f05f5fb65"   # rules 1-7, as version B ran on 2026-10-29
CITATION_NOTE = ('Citing a field: in an observed evidence entry, value must be the field\'s value exactly as it appears in '
                 'the record on the page, including any object. For example, a receipt record that shows "quantity": '
                 '{"unit": "EA", "value": "129"} is cited with "field": "quantity" and "value": {"unit": "EA", '
                 '"value": "129"}, not "129".')
# Which records the report must cite: the grader's required evidence (grader._required_evidence, the completion
# contract). The example id is neutral (no train-pool case has it).
RECORDS_NOTE = """Which records to cite (the completion contract the grader checks), by the record ids shown on the pages:
- release or keep_blocked: the invoice line of the invoice you are resolving, from the Invoice page (field quantity),
  for example the invoice line INV-100000-1, not the invoice INV-100000 (Invoices on PO line also lists this invoice
  as an invoice record; citing that record does not count); every record on Receipt history (field quantity),
  including any receipt posted during the case; and every other active invoice listed on Invoices on PO line, by its
  invoice record (field quantity).
- release: also every cancelled invoice on the PO line (field state), and, if the recorded receipts did not cover the
  invoice, the answered confirmation (field delivered_total).
- keep_blocked: also the answered confirmation (field delivered_total) and the shortfall correction follow-up the
  action opened (field type).
- pending: the confirmation request (field status) and the confirmation pending follow-up the action opened (field
  type).
Cite each record at its version when you took the deciding action, except the follow-up that action opened: it did not
exist before the action, so cite it as shown in the action's own result (or on a later page)."""


def visible_rules(block: str = "VISIBLE RULES") -> str:
    m = re.search(rf"<!-- BEGIN {block} -->\n(.*?)<!-- END {block} -->", RULES_FILE.read_text(), re.S)
    return m.group(1)


def version_c_additions() -> str:
    """Version C: rule 8 (business time, from the rules draft), the citation note and the records note."""
    return visible_rules("VISIBLE RULES: RULE 8") + "\n" + CITATION_NOTE + "\n\n" + RECORDS_NOTE + "\n"


VERSIONS = ("A", "B", "C", "V2")


def interface(version: str) -> dict:
    """What the agent gets under each version: v1's interface (A, B, C) or the final v2 interface (V2)."""
    if version != "V2":
        return {"env": APReceiptEnv, "schema": SCHEMA, "tools": AGENT_TOOLS, "instructions": INSTRUCTIONS_VERSION,
                "budget": TOKEN_BUDGET, "turn_cap": TURN_CAP}
    v2 = {"env": APReceiptEnvV2, "schema": pipeline.PIPELINE_SCHEMA, "tools": AGENT_TOOLS_V2,
          "instructions": INSTRUCTIONS_VERSION_V2, "budget": pipeline.PIPELINE_MAX_COMPLETION_LENGTH,
          "turn_cap": pipeline.PIPELINE_MAX_TOOL_CALLING_ITERATIONS}
    pipeline.require_pipeline(schema=v2["schema"], instructions=v2["instructions"], max_completion_length=v2["budget"],
                              max_tool_calling_iterations=v2["turn_cap"])   # the v2 runner refuses other settings
    return v2


def system_prompt(version: str) -> str:
    if version == "V2":
        return SYSTEM_PROMPT_V2
    if version == "A":
        return SYSTEM_PROMPT
    rules = visible_rules()
    if sha(rules) != B_RULES_SHA256:   # version B must stay the text it ran with on 2026-10-29
        raise RuntimeError("rules 1-7 in the rules draft changed; version B would no longer be batch 3's version B")
    return SYSTEM_PROMPT + "\n" + rules + (version_c_additions() if version == "C" else "")


def tool_definitions(env: APReceiptEnv) -> list[dict]:
    """TRL's schemas (transformers get_json_schema on the bound methods), in the Messages API's shape, in TRL's
    (alphabetical) order for the v2 environment."""
    out = []
    for name in (sorted(AGENT_TOOLS_V2) if isinstance(env, APReceiptEnvV2) else AGENT_TOOLS):
        fn = get_json_schema(getattr(env, name))["function"]
        out.append({"name": fn["name"], "description": fn["description"], "input_schema": fn["parameters"]})
    return out


def execute(env: APReceiptEnv, name: str, args) -> str:
    """One tool call, as TRL 1.16's tool loop makes it (grpo_trainer._tool_call_loop)."""
    try:
        if name not in (AGENT_TOOLS_V2 if isinstance(env, APReceiptEnvV2) else AGENT_TOOLS):
            raise ValueError(f"Tool {name} not found.")
        result = getattr(env, name)(**args)
    except Exception as e:
        result = {"error": str(e)}
        env.__dict__["_malformed_tool_calls"] = env.__dict__.get("_malformed_tool_calls", 0) + 1   # never an attempt
    return str(result)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def verdict_record(env: APReceiptEnv, stop: str) -> dict:
    scenario, log, _ = env._episode()
    v2 = isinstance(env, APReceiptEnvV2)
    v = g8.grade(scenario, log) if v2 else grade(scenario, log)
    excluded = v2_rewards.EXCLUDED if v2 else EXCLUDED
    reward = None if stop in excluded or stop in API_FAILURES else float(v.reward)
    report = log.get("report")
    return {"stop_reason": stop, "reward": reward, "graded_reward": v.reward, "complete_success": v.reward == 1,
            "expected": v.expected, "reached": v.reached, "critical": v.critical,
            "attempted_violations": v.diagnostics["unsafe_attempts"],
            "business_outcome_correct": v.diagnostics["business_outcome_correct"],
            "finish_accepted": v.diagnostics["finish_accepted"],
            "finish_rejected": sum(1 for s in log["steps"] if s["tool"] == "finish" and not s["result"]["ok"]),
            "citation_problems": v.diagnostics["citation_problems"],
            "required_evidence": [{k: r.get(k) for k in ("record_id", "field", "satisfied", "seen_before_action")}
                                  for r in v.diagnostics["required_evidence"]],
            "calls": log["calls"], "final_clock": format_time(log["final_clock"]),
            "discrepancy_type": v.diagnostics["discrepancy_type"], "grader": v.diagnostics["grader_version"],
            "report": report, "infrastructure": env._infrastructure_failure,
            "malformed_tool_calls": env.__dict__.get("_malformed_tool_calls", 0),
            "post_end_calls": v.diagnostics.get("post_end_calls", 0),
            "superseded_dispositions": v.diagnostics.get("superseded_dispositions", {})}


def new_env(case: str, version: str = "A") -> tuple[APReceiptEnv, str]:
    scenarios = load(MANIFEST)
    i = interface(version)
    env = i["env"]({case: scenarios[case]}, schema=i["schema"], execution_rule=LOOP_GUARD)
    return env, env.reset(scenario=case)


def stop_reason(env: APReceiptEnv, fallback: str) -> str:
    if env._infrastructure_failure:
        return "infrastructure_failure"
    return env._stop_reason or fallback


def api_key() -> str | None:
    """ANTHROPIC_API_KEY from the environment, else from the repo-root .env (git-ignored). Only that one line is
    parsed; nothing from the file is printed, logged or put in an error message."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key or not ENV_FILE.is_file():
        return key
    for line in ENV_FILE.read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.*?)\s*$", line)
        if m:
            value = m.group(1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            return value or None
    return None


def _overloaded(anthropic, exc) -> bool:
    """A rate-limit or overload error (HTTP 429, 529, 503, or an overloaded_error event inside a stream)."""
    if isinstance(exc, (anthropic.RateLimitError, anthropic.OverloadedError, anthropic.ServiceUnavailableError)):
        return True
    return isinstance(exc, anthropic.APIStatusError) and "overloaded_error" in str(exc)


def run_claude(case: str, version: str, model: str, thinking: str, effort: str) -> dict:
    key = api_key()
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set in the environment or in the repo-root .env file.")
    import anthropic
    client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=600.0)   # no SDK retries; backoff below only
    del key

    env, obs = new_env(case, version)
    budget, turn_cap = interface(version)["budget"], interface(version)["turn_cap"]
    system, tools = system_prompt(version), tool_definitions(env)
    messages: list[dict] = [{"role": "user", "content": USER_PREFIX + obs}]
    transcript: list[dict] = [{"role": "user", "content": USER_PREFIX + obs}]
    usage_rows, first_input, used, turns, stop, failure, backoffs = [], None, 0, 0, None, None, []
    started = time.time()
    while stop is None:
        params = dict(model=model, max_tokens=max(1, budget - used), system=system, tools=tools,
                      tool_choice={"type": "auto", "disable_parallel_tool_use": True}, messages=messages,
                      thinking=THINKING[thinking], output_config={"effort": effort},
                      cache_control={"type": "ephemeral"})
        waited, response = 0, None
        while response is None and stop is None:
            try:
                with client.messages.stream(**params) as stream:
                    response = stream.get_final_message()
                    request_id = stream.request_id
            except anthropic.APIError as exc:
                failure = {"type": type(exc).__name__, "status": getattr(exc, "status_code", None),
                           "message": str(exc)[:2000], "request_id": getattr(exc, "request_id", None)}
                # Rate limit / overload only (set 2026-10-29): resend the same request after 10, 20, 40 ... s, at
                # most BACKOFF_LIMIT s in all. Anything else (rejected request, timeout, connection) is never resent.
                if not _overloaded(anthropic, exc):
                    stop = "api_error"
                elif waited >= BACKOFF_LIMIT:
                    stop = "api_overload"
                else:
                    wait = min(10 * 2 ** len([b for b in backoffs if b["turn"] == len(usage_rows)]), BACKOFF_LIMIT - waited)
                    backoffs.append({"turn": len(usage_rows), "wait_s": wait, **failure})
                    time.sleep(wait)
                    waited += wait
        if stop is not None:
            break
        failure = None
        u = response.usage
        total_in = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        first_input = total_in if first_input is None else first_input
        used = total_in - first_input + u.output_tokens
        usage_rows.append({"input": u.input_tokens, "cache_read": u.cache_read_input_tokens or 0,
                           "cache_write": u.cache_creation_input_tokens or 0, "output": u.output_tokens,
                           "stop_reason": response.stop_reason, "request_id": request_id})
        content = [b.to_dict() for b in response.content]
        messages.append({"role": "assistant", "content": content})   # append-only, blocks unchanged
        transcript.append({"role": "assistant", "content": content, "stop_reason": response.stop_reason})
        tool_uses = [b for b in content if b["type"] == "tool_use"]
        if response.stop_reason == "refusal":
            stop, failure = "api_refusal", {"stop_details": response.stop_details.to_dict() if response.stop_details else None}
        elif response.stop_reason == "max_tokens" or used > budget:
            stop = "token_budget"
        elif not content:
            stop = "api_empty_response"
        elif not tool_uses:
            stop = "model_stopped"
        else:
            results = []
            for tu in tool_uses:
                text = execute(env, tu["name"], tu["input"])
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": text})
            messages.append({"role": "user", "content": results})
            transcript.append({"role": "user", "content": results})
            turns += 1
            if env._layer.done or env._infrastructure_failure:
                stop = stop_reason(env, "finish")
            elif turns >= turn_cap:
                stop = "turn_cap"
            elif time.time() - started > EPISODE_TIMEOUT:
                stop = "episode_timeout"
    stop = "infrastructure_failure" if env._infrastructure_failure else stop
    p = PRICES[model]
    tokens = {k: sum(r[k] for r in usage_rows) for k in ("input", "cache_read", "cache_write", "output")}
    cost = (tokens["input"] * p["input"] + tokens["cache_read"] * p["cache_read"]
            + tokens["cache_write"] * p["cache_write"] + tokens["output"] * p["output"]) / 1e6
    thinking_blocks = sum(1 for m in transcript if m["role"] == "assistant" for b in m["content"]
                          if b["type"] in ("thinking", "redacted_thinking"))
    return {"config": config(case, version, model, thinking, effort, system, tools, anthropic.__version__),
            "verdict": verdict_record(env, stop), "failure": failure, "backoffs": backoffs, "tool_calls": env._layer.calls,
            "model_turns": len(usage_rows), "thinking_blocks": thinking_blocks, "completion_tokens_used": used,
            "tokens": tokens, "cost_usd": round(cost, 4), "wall_seconds": round(time.time() - started, 1),
            "transcript": transcript, "episode_log": env._episode()[1]}


def run_reference(case: str, version: str) -> dict:
    """$0 rehearsal: reference/4 (reference/5 under V2) drives the same dispatch, stop and grading path (no API)."""
    from ap_sim.agents.reference import ReferenceAgentV4, ReferenceAgentV5

    env, obs = new_env(case, version)

    class Tools:   # the reference agent's tool interface, routed through execute() like a Claude tool_use block
        initial_observation = json.loads(obs)
        click = staticmethod(lambda element_id: json.loads(execute(env, "click", {"element_id": element_id})))
        fill = staticmethod(lambda element_id, text: json.loads(execute(env, "fill", {"element_id": element_id, "text": text})))
        advance_time = staticmethod(lambda minutes: json.loads(execute(env, "advance_time", {"minutes": minutes})))
        coverage = staticmethod(lambda invoice_id: json.loads(execute(env, "coverage", {"invoice_id": invoice_id})))
        finish = staticmethod(lambda report: json.loads(execute(env, "finish", report)))

    (ReferenceAgentV5 if version == "V2" else ReferenceAgentV4)(Tools()).run()
    system, tools = system_prompt(version), tool_definitions(env)
    return {"config": config(case, version, "reference/5" if version == "V2" else "reference/4", None, None, system,
                             tools, None),
            "verdict": verdict_record(env, stop_reason(env, "model_stopped")), "tool_calls": env._layer.calls,
            "episode_log": env._episode()[1]}


def with_infrastructure_rerun(run, *args) -> dict:
    """An episode that ends in infrastructure_failure (a fault outside the model's control) is re-run once (rule set on
    2026-10-31). If it fails again it counts as a failure (reward 0), and the re-run is reported."""
    rec = run(*args)
    rec["infrastructure_reruns"] = 0
    if rec["verdict"]["stop_reason"] != "infrastructure_failure":
        return rec
    first = {k: rec.get(k) for k in ("verdict", "tool_calls")}
    rec = run(*args)
    rec["infrastructure_reruns"], rec["first_attempt"] = 1, first
    if rec["verdict"]["stop_reason"] == "infrastructure_failure":
        rec["verdict"] = {**rec["verdict"], "reward": 0.0, "counted_as_failure": "infrastructure_failure twice"}
    return rec


def config(case, version, model, thinking, effort, system, tools, sdk) -> dict:
    return {"case": case, "manifest": str(MANIFEST.relative_to(ROOT)), "version": version, "model": model,
            "thinking": THINKING.get(thinking), "effort": effort, "temperature": "API default (not settable)",
            "parallel_tool_use": False, "prompt_caching": "automatic (top-level cache_control, 5-minute TTL)",
            "schema": interface(version)["schema"], "execution_rule": LOOP_GUARD,
            "instructions": interface(version)["instructions"], "environment": interface(version)["env"].__name__,
            "tool_names": [t["name"] for t in tools],
            "system_sha256": sha(system), "rules_sha256": sha(visible_rules()) if version in ("B", "C") else None,
            "version_c_additions_sha256": sha(version_c_additions()) if version == "C" else None,
            "tools_sha256": sha(json.dumps(tools, sort_keys=True)), "user_prefix": USER_PREFIX,
            "token_budget": interface(version)["budget"], "turn_cap": interface(version)["turn_cap"],
            "action_limit": ACTION_LIMIT,
            "anthropic_sdk": sdk, "fallbacks": None, "max_retries": 0,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--version", choices=VERSIONS, default="A")
    ap.add_argument("--model", default=MODEL, choices=sorted(PRICES))
    ap.add_argument("--thinking", choices=sorted(THINKING), default="off")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--out", default=str(ROOT / "runs" / "2026-10-29-strong-model-check"))
    ap.add_argument("--dry-run-reference", action="store_true")
    a = ap.parse_args()
    if a.case not in {e["case_id"] for e in json.loads(MANIFEST.read_text())["entries"]}:
        sys.exit(f"{a.case} is not in {MANIFEST.name}")
    rec = (with_infrastructure_rerun(run_reference, a.case, a.version) if a.dry_run_reference else
           with_infrastructure_rerun(run_claude, a.case, a.version, a.model, a.thinking, a.effort))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{'reference' if a.dry_run_reference else a.model}-{a.case}-{a.version}.json"
    path.write_text(json.dumps(rec, indent=1, default=str))
    v = rec["verdict"]
    print(f"{a.case} {a.version}: stop {v['stop_reason']}; reward {v['reward']}; expected {v['expected']}; reached "
          f"{v['reached']}; critical {v['critical'] or 'none'}; attempts {v['attempted_violations']}; calls "
          f"{rec['tool_calls']}" + (f"; cost ${rec['cost_usd']}" if "cost_usd" in rec else "") + f"\n-> {path}")


if __name__ == "__main__":
    main()
