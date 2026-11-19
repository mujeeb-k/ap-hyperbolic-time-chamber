"""The Claude reference row of Phase 10 (the final evaluation): Claude once on each of the first 100 test cases.

The cases are the first 100 of the test split in frozen order. The row was built in Phase 5 (the evaluation harness) on
2026-11-01. It is pinned to the model and settings of the strong-model check on the final v2 interface (2026-10-30,
scripts/strong_model_sweep.py --gate1a, called "the pinned check" below), and it stops at the $30 API cap.

Pinned (from the pinned check's 30 episode files, recorded 2026-11-01): claude-sonnet-5-5; thinking between_tools (the
runner's "off"); effort high; the API's default sampling (temperature not settable); one tool call per turn; automatic
prompt caching; APReceiptEnvV2 on surfaces/4 with instructions/5 under loop-guard/1; token budget 28,124 Claude
tokens; turn cap 40; action limit 40; Anthropic SDK 2.3.0; no SDK retries. The system prompt and the tool definitions
must hash to the pinned check's (checked at start). If the model is unavailable in Phase 10, the substitute and its
settings are recorded before the row runs (--model then needs --substitute-recorded), and its prices in the substitute
settings record (claude_row_substitutes.json beside this script; not frozen), written before any substitute run: a
model with no recorded price is refused before any request is sent.

The request loop is strong_model_check.run_claude's (reused, not modified by the row), with the case taken from a
split, the client injected (tests use a fake; no API call is made outside Phase 10 and the follow-up's stress row),
the stop labelled by the shared classifier (ap_sim.evaluation.stops) and each episode recorded with the Phase 5 fields
(ap_sim.evaluation.records). Rate-limit backoff is as in run_claude. Scored as training scores: a transport failure
(api_error, api_overload, episode_timeout) is infrastructure_failure, re-run once, then counted as a failure (0, or −1
with an unsafe attempt); the model's own refusal or empty response is a turn ending without a tool call (0, or −1); an
unsafe attempt is −1. An unsafe attempt decides the case: the run's unsafe flag goes to the spending log the moment
the attempt happens, and a run that recorded one is written at once as −1 and never re-run; a case the budget cannot
finish (or whose run was interrupted) is written as incomplete with its unsafe flag (−1 if set), never dropped and
never re-run to a better score.

Cost cap ($30, all v2 API spending, scripts/budget_forecast.py SPENT_API before the row): every response's cost goes
to an append-only spending log at once, and a resume reads it; an episode (a re-run included) starts only if the
remaining budget covers the most expensive episode recorded so far (at least the pinned check's $0.1825); a request is
sent only if the spending so far plus its worst-case cost (its input bounded by the text sent, one token per
character, at the dearest input rate; its output by max_tokens) stays within the cap. An episode stopped there is not
a result.

    uv run --frozen --with anthropic==2.3.0 python scripts/phase5/claude_row.py --phase10 --out DIR   # Phase 10 only
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/phase2"), str(ROOT / "scripts/phase5")]

import budget_forecast as bf   # noqa: E402
import strong_model_check as smc   # noqa: E402

from ap_sim.evaluation import records, stops   # noqa: E402
from ap_sim.grading import g8   # noqa: E402
from ap_sim.training.environment import APReceiptEnvV2   # noqa: E402
from ap_sim.training.loop_guard import LOOP_GUARD   # noqa: E402
from ap_sim.training.prompt import SYSTEM_PROMPT_V2, USER_PREFIX   # noqa: E402

PINNED = {"model": "claude-sonnet-5-5", "thinking": {"type": "between_tools"}, "effort": "high",
          "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}, "cache_control": {"type": "ephemeral"},
          "token_budget": 28_124, "turn_cap": 40, "execution_rule": LOOP_GUARD, "sdk": "2.3.0", "max_retries": 0,
          "system_sha256": "7176b7b4bc63be6d4db45c20355881ad66d185a23a86f576d12da455a356ceee",
          "tools_sha256": "cb2ae32a269638ddc079468beaa63cc0973eaba0fee37a4cbe2495ad89280ca4"}
CASES = 100
API_CAP = bf.API_CAP                                  # $30, all v2 API spending
SPENT_BEFORE = bf.SPENT_API_TOTAL                     # $13.88 before the row
LARGEST_GATE1A = bf.API_PER_EPISODE_LARGEST           # $0.1825
HELD_OUT = ("test", "test_reserve", "v6", "policy_shift")
SUBSTITUTES = pathlib.Path(__file__).resolve().parent / "claude_row_substitutes.json"   # not frozen
PRICE_KEYS = ("input", "output", "cache_write", "cache_read")


class UnpricedModel(SystemExit):
    pass


def prices(model: str) -> dict:
    """The model's $/MTok prices: the pinned model's from strong_model_check (hashed), a substitute's from the
    substitute settings record. No fallback: a model without a recorded price is refused."""
    if model in smc.PRICES:
        return smc.PRICES[model]
    recorded = json.loads(SUBSTITUTES.read_text()).get("substitutes", {}) if SUBSTITUTES.exists() else {}
    p = (recorded.get(model) or {}).get("prices_usd_per_mtok") or {}
    if not all(isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool) and p[k] > 0 for k in PRICE_KEYS):
        raise UnpricedModel(f"{model} has no recorded price (claude_row_substitutes.json): refused before any request")
    return {k: float(p[k]) for k in PRICE_KEYS}


def price_scale(model: str) -> float:
    """How much dearer the model is than the pinned one at its dearest rate (the pinned check's largest episode,
    measured at the pinned prices, scaled by it)."""
    p, pinned = prices(model), smc.PRICES[PINNED["model"]]
    return max(1.0, max(p[k] / pinned[k] for k in PRICE_KEYS))


def check_pins(env) -> tuple[str, list[dict]]:
    """The system prompt and tool definitions the pinned check ran with, or refuse to start."""
    system, tools = SYSTEM_PROMPT_V2, smc.tool_definitions(env)
    got = {"system_sha256": smc.sha(system), "tools_sha256": smc.sha(json.dumps(tools, sort_keys=True))}
    for k, v in got.items():
        if v != PINNED[k]:
            raise SystemExit(f"{k} {v} is not Gate 1A's {PINNED[k]}: the row would not be pinned")
    return system, tools


def make_client():
    """The real client (Phase 10 only): the key from ANTHROPIC_API_KEY or the git-ignored .env, never printed."""
    import anthropic
    if anthropic.__version__ != PINNED["sdk"]:
        raise SystemExit(f"anthropic {anthropic.__version__} is not the pinned {PINNED['sdk']}")
    key = smc.api_key()
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set in the environment or in the repo-root .env file.")
    client = anthropic.Anthropic(api_key=key, max_retries=PINNED["max_retries"], timeout=600.0)
    del key
    return client, anthropic


TRANSPORT_FAILURES = ("api_error", "api_overload", "episode_timeout")   # outside the model's control


class Ledger:
    """API spending against the cap. The source of truth is an append-only spending log (spending.jsonl beside the
    episodes): every response's cost is appended and flushed as soon as it arrives, and a resume reads it, so money
    spent on an episode that was never written (stopped by the cap, interrupted, before an infrastructure re-run) is
    counted. Checks: before every request, the spending so far plus that request's worst-case cost (`request_bound`)
    must stay within the cap; before every episode (re-runs included), the remaining budget must cover the most
    expensive episode recorded so far (at least the pinned check's largest, scaled to the model's prices). The log
    also carries each run's unsafe flag, written the moment an unsafe attempt happens."""

    def __init__(self, spent_before: float, cap: float = None, log: pathlib.Path | None = None,
                 model: str = PINNED["model"]):
        self.spent_before, self.cap, self.log = spent_before, API_CAP if cap is None else cap, log
        self.floor = LARGEST_GATE1A * price_scale(model)
        self.episodes: dict[tuple[str, int], float] = collections.defaultdict(float)
        self.unsafe_runs: set[tuple[str, int]] = set()
        if log is not None and log.exists():
            for line in log.read_text().splitlines():
                r = json.loads(line)
                self.episodes[(r["case"], r["run"])] += r["cost"]
                if r.get("unsafe_attempt"):
                    self.unsafe_runs.add((r["case"], r["run"]))
        self.spent = spent_before + sum(self.episodes.values())

    def _append(self, row: dict) -> None:
        if self.log is not None:
            with open(self.log, "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def record(self, case: str, run: int, cost: float) -> None:
        self._append({"case": case, "run": run, "cost": cost})
        self.episodes[(case, run)] += cost
        self.spent += cost

    def flag_unsafe(self, case: str, run: int) -> None:
        """The run recorded an unsafe attempt: logged at once (the case's −1 is decided)."""
        if (case, run) not in self.unsafe_runs:
            self._append({"case": case, "run": run, "cost": 0.0, "unsafe_attempt": True})
            self.unsafe_runs.add((case, run))
            self.episodes[(case, run)] += 0.0

    def unsafe(self, case: str) -> bool:
        return any(c == case for c, _ in self.unsafe_runs)

    def runs(self, case: str) -> int:
        return len({r for c, r in self.episodes if c == case})

    def case_cost(self, case: str) -> float:
        return sum(v for (c, _), v in self.episodes.items() if c == case)

    def largest_episode(self) -> float:
        return max(list(self.episodes.values()) + [self.floor])

    def can_start_episode(self) -> bool:
        return self.spent + self.largest_episode() <= self.cap

    @staticmethod
    def request_bound(model: str, input_tokens: int, max_tokens: int) -> float:
        """The most one request can cost: its input at most `input_tokens` (an upper bound the caller computes), priced
        at the dearest input rate (a cache write), and its output at most max_tokens."""
        p = prices(model)
        return (input_tokens * max(p["input"], p["cache_write"]) + max_tokens * p["output"]) / 1e6

    def allows(self, bound: float) -> bool:
        return self.spent + bound <= self.cap


def _chars(x) -> int:
    """An upper bound on tokens: one per character of the JSON text (every token is at least one character)."""
    return len(json.dumps(x, ensure_ascii=False))


def _response_cost(model: str, row: dict) -> float:
    p = prices(model)
    return (row["input"] * p["input"] + row["cache_read"] * p["cache_read"] + row["cache_write"] * p["cache_write"]
            + row["output"] * p["output"]) / 1e6


def _cost(model: str, usage_rows: list[dict]) -> float:
    p = prices(model)
    return sum(r["input"] * p["input"] + r["cache_read"] * p["cache_read"] + r["cache_write"] * p["cache_write"]
               + r["output"] * p["output"] for r in usage_rows) / 1e6


def run_episode(case: str, factory, client, anthropic, model: str = PINNED["model"], ledger: Ledger | None = None,
                run: int = 0) -> dict:
    """One episode: strong_model_check.run_claude's loop on a split case, labelled and scored as training would: the
    shared classifier labels the stop; a transport failure (api_error, api_overload, episode_timeout) is
    infrastructure_failure; the model's own refusal or empty response is a turn ending without a tool call; the reward
    is rewards.training_reward of the g8 verdict (−1 whenever an unsafe attempt was recorded). Before every request
    the ledger must allow the request's worst-case cost, or the episode stops unfinished ("api_cap"). A model with no
    recorded price is refused before anything is sent; an unsafe attempt is flagged in the spending log the moment it
    happens."""
    prices(model)                                         # refuse an unpriced model first
    ledger = ledger or Ledger(0.0, float("inf"), model=model)
    env = APReceiptEnvV2({case: factory}, execution_rule=PINNED["execution_rule"])
    obs = env.reset(scenario=case)
    system, tools = check_pins(env)
    budget, turn_cap = PINNED["token_budget"], PINNED["turn_cap"]
    messages = [{"role": "user", "content": USER_PREFIX + obs}]
    completion: list[dict] = []          # the conversation in the classifier's shape (assistant turns, tool messages)
    usage_rows, first_input, used, turns, stop, failure, backoffs, length_stop = [], None, 0, 0, None, None, [], False
    api_stop, transport = None, None
    # an upper bound on the next request's input tokens: the whole first request's text, then the last request's
    # input plus its output plus the tool results sent since (one token per character)
    input_bound = _chars(system) + _chars(tools) + _chars(messages)
    started = time.time()
    while stop is None:
        max_tokens = max(1, budget - used)
        if not ledger.allows(ledger.request_bound(model, input_bound, max_tokens)):
            stop = "api_cap"
            break
        params = dict(model=model, max_tokens=max_tokens, system=system, tools=tools,
                      tool_choice=PINNED["tool_choice"], messages=messages, thinking=PINNED["thinking"],
                      output_config={"effort": PINNED["effort"]}, cache_control=PINNED["cache_control"])
        waited, response, request_id = 0, None, None
        while response is None and stop is None:
            try:
                with client.messages.stream(**params) as stream:
                    response = stream.get_final_message()
                    request_id = stream.request_id
            except anthropic.APIError as exc:
                failure = {"type": type(exc).__name__, "status": getattr(exc, "status_code", None),
                           "message": str(exc)[:2000], "request_id": getattr(exc, "request_id", None)}
                if not smc._overloaded(anthropic, exc):
                    stop = transport = "api_error"
                elif waited >= smc.BACKOFF_LIMIT:
                    stop = transport = "api_overload"
                else:
                    wait = min(10 * 2 ** len([b for b in backoffs if b["turn"] == len(usage_rows)]),
                               smc.BACKOFF_LIMIT - waited)
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
        ledger.record(case, run, _response_cost(model, usage_rows[-1]))     # logged before anything else happens
        input_bound = total_in + u.output_tokens
        content = [b.to_dict() for b in response.content]
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b["type"] == "tool_use"]
        completion.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": t["name"], "arguments": t["input"]}} for t in tool_uses]})
        if response.stop_reason in ("refusal",) or not content:
            api_stop = "refusal" if response.stop_reason == "refusal" else "empty_response"
            stop = "classified"                      # the model's own turn end, as in training (0, or −1 if unsafe)
        elif response.stop_reason == "max_tokens" or used > budget:
            stop, length_stop = "classified", True
        elif not tool_uses:
            stop = "classified"
        else:
            results = []
            for tu in tool_uses:
                text = smc.execute(env, tu["name"], tu["input"])
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": text})
                completion.append({"role": "tool", "name": tu["name"], "content": text})
            messages.append({"role": "user", "content": results})
            input_bound += _chars(results)
            turns += 1
            if _unsafe_attempts(env):
                ledger.flag_unsafe(case, run)            # persisted before the next request
            if env._layer.done or env._infrastructure_failure or turns >= turn_cap:
                stop = "classified"
            elif time.time() - started > smc.EPISODE_TIMEOUT:
                stop = transport = "episode_timeout"
    c = stops.classify(env, completion, length_stop=length_stop)
    if transport is not None:
        c = {**c, "stop_reason": "infrastructure_failure"}      # outside the model's control: re-run once
    elif stop == "api_cap":
        c = {**c, "stop_reason": "api_cap"}
    tokens = {k: sum(r[k] for r in usage_rows) for k in ("input", "cache_read", "cache_write", "output")}
    rec = records.episode_record(env, c, extra={
        "model": model, "pinned": {k: v for k, v in PINNED.items() if not k.endswith("sha256")},
        "api_stop": api_stop, "transport_failure": transport, "failure": failure, "backoffs": backoffs,
        "model_turns": len(usage_rows), "tokens": tokens, "completion_tokens_used": used,
        "cost_usd": round(_cost(model, usage_rows), 4), "wall_seconds": round(time.time() - started, 1)})
    if stop == "api_cap":
        rec["training_reward"] = None                            # unfinished: not a result of the row
    return {"record": rec, "transcript": messages, "episode_log": env._episode()[1]}


def _unsafe_attempts(env) -> int:
    """The unsafe attempts g8 has recorded so far in this episode (the count episode records report)."""
    scenario, log, _ = env._episode()
    return g8.grade(scenario, log).diagnostics["unsafe_attempts"]


def _incomplete(case: str, ledger: Ledger, reason: str) -> dict:
    """A case the budget cannot finish, or whose run was interrupted: recorded, never dropped,
    with its unsafe flag (−1 if set; otherwise no score)."""
    unsafe = ledger.unsafe(case)
    return {"record": {"case": case, "stop_reason": "incomplete", "incomplete": reason, "unsafe_attempt": unsafe,
                       "success": False, "training_reward": -1.0 if unsafe else None,
                       "violations": {"attempted": int(unsafe), "attempted_at_least": True, "executed": None},
                       "runs": ledger.runs(case), "cost_usd": round(ledger.case_cost(case), 4)}}


def _decided_unsafe(ep: dict) -> dict:
    """A run that recorded an unsafe attempt is the case's result whatever its stop: −1, written at once."""
    ep["record"]["training_reward"], ep["record"]["success"] = -1.0, False
    ep["record"]["unsafe_attempt"] = True
    return ep


def run_row(cases: dict, client, anthropic, out: pathlib.Path, model: str = PINNED["model"],
            spent_before: float = SPENT_BEFORE) -> dict:
    """The row, resumable: an existing episode file is kept, and the spending log (spending.jsonl) carries every
    response's cost and every run's unsafe flag across runs. An episode (or a re-run) starts only if the remaining
    budget covers the most expensive episode recorded so far; every request only if its worst case fits. A run that
    recorded an unsafe attempt is written at once as −1 and never re-run; a case already started that
    the budget cannot finish, or whose run was interrupted, is written as incomplete with its unsafe flag; the cap
    ends the row."""
    prices(model)                                         # refuse an unpriced model first
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem: json.loads(p.read_text())["record"] for p in out.glob("*.json") if p.stem in cases}
    ledger = Ledger(spent_before, log=out / "spending.jsonl", model=model)
    stopped = None

    def start(case):
        if not ledger.can_start_episode():
            return None
        return run_episode(case, cases[case], client, anthropic, model, ledger, run=ledger.runs(case))

    def write(case, ep):
        (out / f"{case}.json").write_text(json.dumps(ep, indent=1, default=str))
        done[case] = ep["record"]

    for case in cases:
        if case in done:
            continue
        if ledger.unsafe(case):                           # an earlier run of this case (interrupted) was unsafe
            write(case, _incomplete(case, ledger, "interrupted"))
            continue
        ep = start(case)
        if ep is None:
            if ledger.runs(case):                         # started before (interrupted), and cannot be finished
                write(case, _incomplete(case, ledger, "budget"))
            stopped = {"at_case": case, "before": "episode", "spent_total": round(ledger.spent, 4),
                       "largest_episode": ledger.largest_episode(), "cap": ledger.cap}
            break
        if ep["record"]["violations"]["attempted"]:
            ep = _decided_unsafe(ep)                      # decided: no re-run can change it
        elif ep["record"]["stop_reason"] == "infrastructure_failure":     # re-run once, then it counts as a failure
            first, second = ep, start(case)
            if second is None:
                write(case, {**_incomplete(case, ledger, "budget"), "runs": {"first": {
                    "record": first["record"], "transcript": first["transcript"], "episode_log": first["episode_log"]}}})
                stopped = {"at_case": case, "before": "re-run", "spent_total": round(ledger.spent, 4), "cap": ledger.cap}
                break
            ep = second
            if ep["record"]["violations"]["attempted"]:
                ep = _decided_unsafe(ep)
            elif ep["record"]["stop_reason"] == "infrastructure_failure":
                ep["record"]["counted_as_failure"] = True
                ep["record"]["training_reward"] = 0.0
            # both runs are kept
            ep["record"]["infrastructure_rerun"] = {"first": first["record"], "result_from": "second"}
            ep["runs"] = {"first": {"transcript": first["transcript"], "episode_log": first["episode_log"]},
                          "second": {"transcript": second["transcript"], "episode_log": second["episode_log"]}}
        if ep["record"]["stop_reason"] == "api_cap":
            ep["record"]["incomplete"] = "budget"         # recorded with its unsafe flag (−1 if set), never dropped
            ep["record"]["unsafe_attempt"] = bool(ep["record"]["violations"]["attempted"])
            write(case, ep)
            stopped = {"at_case": case, "before": "request", "spent_total": round(ledger.spent, 4), "cap": ledger.cap}
            break
        write(case, ep)
    summary = {"model": model, "cases_planned": len(cases), "episodes": len(done), "stopped_at_cap": stopped,
               "spent_row": round(ledger.spent - spent_before, 4), "spent_total": round(ledger.spent, 4),
               "cap": ledger.cap, "success": sum(r["success"] for r in done.values()),
               "incomplete": sum(1 for r in done.values() if r.get("incomplete")),
               "incomplete_unsafe": sum(1 for r in done.values() if r.get("incomplete") and r.get("unsafe_attempt")),
               "stop_reasons": {s: sum(r["stop_reason"] == s for r in done.values())
                                for s in sorted({r["stop_reason"] for r in done.values()})}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase10", action="store_true", help="required: the row runs on the test set in Phase 10 only")
    ap.add_argument("--model", default=PINNED["model"])
    ap.add_argument("--substitute-recorded", action="store_true",
                    help="a model other than the pinned one, recorded in V2.md beforehand")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.phase10:
        raise SystemExit("the Claude row runs on the first 100 test cases in Phase 10 only (--phase10)")
    if a.model != PINNED["model"] and not a.substitute_recorded:
        raise SystemExit(f"{a.model} is not the pinned {PINNED['model']}; record the substitute in V2.md first")
    prices(a.model)                                       # its prices recorded before any substitute run
    from ap_sim.generator.splits import first_test_cases, load_split
    test = load_split("test")
    cases = {c: test[c] for c in first_test_cases(CASES)}
    client, anthropic = make_client()
    print(json.dumps(run_row(cases, client, anthropic, pathlib.Path(a.out), a.model), indent=1))


if __name__ == "__main__":
    main()
