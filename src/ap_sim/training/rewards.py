"""The v2 training reward. The environment holds no grading logic: this function calls the independent grader (g8 since
2026-10-31, g7 before) on each rollout's episode log and maps its verdict to the reward. The verdict (0 or 1) stays the
definition of success; the reward is for training only.

    -1    the grader records any attempted unsafe action (diagnostics["unsafe_attempts"] > 0), whatever the stop reason;
    None  otherwise, infrastructure_failure only: a fault outside the model's control (left out of the group
          baseline; TRL 1.16 forces the advantage to 0). Exclusion is reserved for faults the model cannot cause.
          turn_cap is not excluded (a call that raises inside TRL never reaches the environment, so a failing rollout
          could reach the cap below the action limit), and neither are the token-limit stops token_budget and
          token_budget_rollback (a pre-data amendment of the pre-registered exclusions, made on 2026-10-31 before any
          training): all score like action_limit;
    +1    otherwise, a verdict of 1 (0.5 instead if the efficiency-penalty hook is on and the episode reached it through a
          confirmation request on a case already covered at the start);
     0    otherwise (including report-only failures such as discrepancy_type_wrong, and pending_when_resolvable,
          which are critical constraints but not unsafe attempts).

A grader exception: the episode scores 0, or -1 if an unsafe attempt was recorded (counted independently of the report
and of g7's grading: g8.unsafe_attempts, every unsafe attempt among the calls the environment received, executed,
refused or after the end), is logged as grader_error on the trace and in LAST_BATCH, and is never excluded. g8 itself
never raises on a report finish accepts (fuzz-tested); this is the backstop.

Malformed tool calls: a call that raises inside TRL (a wrong argument, an unknown tool) becomes a tool message the
environment never receives. It is not an attempt; it is counted per rollout as `malformed_tool_calls`, and it is not a
delivered result for the rollback check.

An accepted finish whose tool result TRL dropped: TRL executes every call in a message, then drops the results that
would take the completion past max_completion_length. When the environment accepted a finish, the episode is a success
or failure on its own terms, not a token-limit stop: stop_reason is finish, whether or not TRL kept the result message
(TRL's message-dropping is a transport artifact, not model behaviour). The rollout carries finish_result_dropped (True
when the accepted finish's result was dropped) for monitoring in Phases 7 and 8; if it fires often, the completion
budget is too tight. The token-limit stops are only the genuine ones, with no accepted finish.

Ported from version 1 of this environment (training/rewards.py): the stop reasons, the trace saved BEFORE grading; the
tool results counted for the rollback check include coverage.
"""
from __future__ import annotations

import collections
import json
import pathlib
import traceback

from ap_sim.grader import arithmetic as ar
from ap_sim.grading import g8
from ap_sim.grading.g8 import grade
from ap_sim.training.config import PIPELINE_MAX_COMPLETION_LENGTH

IM_END = 151645  # Qwen <|im_end|>, the chat template's end of turn
ENDOFTEXT = 151643  # Qwen <|endoftext|>: also an end-of-sequence token in Qwen3-4B-Instruct-2507's generation config
MAX_COMPLETION_LENGTH = PIPELINE_MAX_COMPLETION_LENGTH   # GRPOConfig.max_completion_length; TRL truncates to it
LAST_BATCH: list[dict] = []  # per-rollout diagnostics of the most recent call (read by the training scripts)
TURN_CAP: int | None = None  # set by the training script to GRPOConfig.max_tool_calling_iterations
SAVE_DIR: pathlib.Path | None = None  # when set, every episode log is written here before grading (bulk; not in git)
EFFICIENCY_PENALTY = False   # off unless Phase 7 profiling adopts it (rate of unnecessary requests > 10%)
# Phase 7 (baselines, profiling, RFT) measured 3 of 192 covered-case episodes (1.56%), so the penalty stayed off.
PENALISED_SUCCESS = 0.5
_saved = 0

EXCLUDED = ("infrastructure_failure",)   # turn_cap and the token-limit stops are graded, never excluded
GRADER_ERROR = "grader_error"


def _trl_error(m: dict) -> bool:
    """A tool message TRL wrote for a call that raised before reaching the environment: TRL stores str({"error": ...}),
    while every environment result is the environment's own JSON."""
    return isinstance(m.get("content"), str) and m["content"].startswith("{'error'")


def delivered_tool_results(completion) -> int:
    from ap_sim.training.environment import AGENT_TOOLS_V2
    return sum(1 for m in (completion or []) if isinstance(m, dict) and m.get("role") == "tool"
               and m.get("name") in AGENT_TOOLS_V2 and not _trl_error(m))


def malformed_tool_calls(completion) -> int:
    """Tool calls that never reached the environment: they raised inside TRL, named no agent tool, or could not be
    parsed (a final unparsed tool call)."""
    from ap_sim.training.environment import AGENT_TOOLS_V2
    return sum(1 for m in (completion or []) if isinstance(m, dict) and m.get("role") == "tool"
               and (m.get("name") not in AGENT_TOOLS_V2 or _trl_error(m))) + int(unparsed_tool_call(completion))


def unparsed_tool_call(completion) -> bool:
    """The last assistant message holds a tool call TRL could not parse (for example cut by <|endoftext|> inside the
    <tool_call> block): a malformed call, not a turn end."""
    last = next((m for m in reversed(completion or []) if isinstance(m, dict) and m.get("role") == "assistant"), None)
    content = last.get("content") if last else None
    return isinstance(content, str) and "<tool_call>" in content and not last.get("tool_calls")


def stop_reason(env_reason: str, completion=None, completion_ids=None, executed_calls: int | None = None,
                length_stop: bool | None = None) -> str:
    """Why a rollout ended: finish / action_limit / loop_detected (environment), token_budget / token_budget_rollback
    (the completion limit), turn_cap (TRL's tool-calling cap), unparsed_tool_call (the last message held a tool call
    TRL could not parse) or model_stopped (the model ended a turn without a tool call and without finish). All are
    graded (EXCLUDED holds only infrastructure_failure). token_budget is decided by the completion reaching
    MAX_COMPLETION_LENGTH (TRL truncates every completion to it) or a length stop the engine reports, never by the
    last token: a turn ending on <|im_end|> or <|endoftext|> is a turn end. An accepted finish is finish even when
    TRL dropped its result (see finish_result_dropped)."""
    if env_reason == "finish":
        return "finish"
    if completion is not None and executed_calls is not None and executed_calls > delivered_tool_results(completion):
        return "token_budget_rollback"
    if env_reason != "no_finish":
        return env_reason
    if length_stop or (completion_ids is not None and len(completion_ids) >= MAX_COMPLETION_LENGTH):
        return "token_budget"
    if unparsed_tool_call(completion):
        return "unparsed_tool_call"
    tool_turns = sum(1 for m in (completion or []) if isinstance(m, dict) and m.get("role") == "tool")
    if TURN_CAP is not None and tool_turns >= TURN_CAP:
        return "turn_cap"
    return "model_stopped"


def finish_result_dropped(log, completion) -> bool:
    """The environment accepted a finish and TRL dropped its tool result. TRL keeps the results
    of the executed calls in order and drops a suffix of them, so the accepted finish's result was dropped exactly when
    fewer results were delivered than the steps up to and including it (post-end calls come after it)."""
    if completion is None:
        return False
    accepted = [i for i, st in enumerate(log["steps"]) if st["tool"] == "finish" and st["result"]["ok"]]
    return bool(accepted) and delivered_tool_results(completion) <= accepted[0]


def covered_at_start(scenario) -> bool:
    """The invoice passed the invoice-vs-receipt rule at the start under the case's policy (the grader's gap at the
    start, computed the same way from the scenario's records, is zero): the cases on which a confirmation request is
    unnecessary (the efficiency-penalty hook), and those whose discrepancy_type is none (POLICY I5)."""
    initial = {}
    for r in scenario.records:
        initial.setdefault(r.kind, []).append(r.model_dump(mode="json"))
    inv_id = scenario.current_invoice
    po_line = next(r["po_line"] for r in initial["invoice_line"] if r["invoice"] == inv_id)
    return ar.gap(initial, po_line, inv_id, getattr(scenario, "policy", "sandbox-v1")) == 0


def requested_confirmation(log) -> bool:
    return any(a["actor"] == "ap_agent" and a["op"] == "request_confirmation" for a in log["attempts"])


def training_reward(verdict, reason: str, scenario=None, log=None) -> float | None:
    """The -1/0/+1 mapping (module docstring). scenario and log are needed only when the efficiency-penalty hook is on."""
    if verdict.diagnostics["unsafe_attempts"] > 0:
        return -1.0
    if reason in EXCLUDED:
        return None
    if verdict.reward != 1:
        return 0.0
    if EFFICIENCY_PENALTY and requested_confirmation(log) and covered_at_start(scenario):
        return PENALISED_SUCCESS
    return 1.0


def _save_trace(scenario, log, reason, completions, completion_ids, i) -> pathlib.Path:
    """The full trace, written BEFORE grading, so a grader exception cannot lose the rollout. The grading outcome
    (diagnostics or the exception) is attached afterwards."""
    global _saved
    _saved += 1
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / f"episode-{_saved:06d}-{scenario.name}.json"
    trace = {"grading": "pending", "termination": reason, "log": log,
             "messages": completions[i] if completions else None,
             "completion_ids": [int(t) for t in completion_ids[i]] if completion_ids is not None else None}
    path.write_text(json.dumps(trace, default=str))
    from ap_sim.training.events import event
    event({"event": "batch_trace_saved", "scenario": scenario.name, "termination": reason, "trace": str(path)})
    return path


def _attach(path: pathlib.Path, **fields) -> None:
    trace = json.loads(path.read_text())
    trace.update(fields)
    path.write_text(json.dumps(trace, default=str))


class InfrastructureStop(RuntimeError):
    """Too many infrastructure failures: the run stops."""


INFRA_PER_STEP = 2         # more than this many in one optimizer step stops the run
INFRA_WINDOW = 500         # the last 500 episodes ...
INFRA_RATE = 0.02          # ... with more than 2% of them (more than 10) infrastructure failures stops the run
_infra_recent: collections.deque = collections.deque(maxlen=INFRA_WINDOW)
_infra_by_step: collections.Counter = collections.Counter()


def monitor_infrastructure(flags: list[bool], step=None) -> None:
    """infrastructure_failure is excluded in training and counted. `step` is the trainer's
    global_step (TRL passes trainer_state to reward functions), so several reward calls of one step add up; without it
    each call is its own step."""
    _infra_recent.extend(bool(f) for f in flags)
    in_step = sum(bool(f) for f in flags)
    if step is not None:
        _infra_by_step[step] += in_step
        in_step = _infra_by_step[step]
    if in_step > INFRA_PER_STEP:
        raise InfrastructureStop(f"{in_step} infrastructure failures in one step (limit {INFRA_PER_STEP})")
    if sum(_infra_recent) > INFRA_RATE * INFRA_WINDOW:
        raise InfrastructureStop(f"{sum(_infra_recent)} infrastructure failures in the last {len(_infra_recent)} "
                                 f"episodes (limit {INFRA_RATE:.0%} of {INFRA_WINDOW})")


def ap_reward_v2(environments, completions=None, completion_ids=None, **kwargs) -> list[float | None]:
    LAST_BATCH.clear()
    rewards = []
    for i, env in enumerate(environments):
        scenario, log, env_reason = env._episode()
        reason = ("infrastructure_failure" if env._infrastructure_failure else
                  stop_reason(env_reason, completions[i] if completions else None,
                              completion_ids[i] if completion_ids is not None else None,
                              executed_calls=len(log["steps"])))
        path = _save_trace(scenario, log, reason, completions, completion_ids, i) if SAVE_DIR is not None else None
        completion = completions[i] if completions else None
        try:
            verdict = grade(scenario, log)
        except Exception as exc:  # 0, or -1 with an unsafe attempt; logged; never excluded
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            try:                 # g8's full attempt collection; it is built never to raise
                unsafe = g8.unsafe_attempts(scenario, log)
            except Exception:    # unreachable by construction; if it happens anyway, unjudgeable counts as unsafe
                unsafe = 1
            rewards.append(-1.0 if unsafe > 0 else 0.0)
            LAST_BATCH.append({"scenario": scenario.name, "reward": rewards[-1], "graded": 0, "excluded": False,
                               "grading": GRADER_ERROR, "grader_error": error, "stop_reason": reason,
                               "infrastructure": env._infrastructure_failure, "unsafe_attempts": unsafe,
                               "calls": log["calls"], "malformed_tool_calls": malformed_tool_calls(completion),
                               "finish_result_dropped": finish_result_dropped(log, completion),
                               "grader": g8.GRADER_VERSION})
            from ap_sim.training.events import event
            event({"event": GRADER_ERROR, "scenario": scenario.name, "stop": reason, "error": error["message"]})
            if path is not None:
                _attach(path, grading=GRADER_ERROR, grading_exception=error, diagnostics=LAST_BATCH[-1])
            continue
        rewards.append(training_reward(verdict, reason, scenario, log))
        d = verdict.diagnostics
        LAST_BATCH.append({"scenario": scenario.name, "reward": rewards[-1], "graded": verdict.reward,
                           "grading": "graded", "malformed_tool_calls": malformed_tool_calls(completion),
                           "refused_unsafe_attempts": d.get("refused_unsafe_attempts", []),
                           "post_end_calls": d.get("post_end_calls", 0),
                           "post_end_unsafe_attempts": d.get("post_end_unsafe_attempts", []),
                           "superseded_dispositions": d.get("superseded_dispositions", {}),
                           "excluded": reason in EXCLUDED and d["unsafe_attempts"] == 0,
                           "infrastructure": env._infrastructure_failure, "stop_reason": reason,
                           "execution_rule": getattr(env, "_execution_rule", None),
                           "loop_stop": getattr(env, "_loop_stop", None),
                           "expected": verdict.expected, "reached": verdict.reached, "critical": verdict.critical,
                           "calls": log["calls"], "final_clock": log["final_clock"],
                           "finish_accepted": d["finish_accepted"],
                           "finish_result_dropped": finish_result_dropped(log, completion),
                           "finish_rejected": sum(1 for st in log["steps"] if st["tool"] == "finish" and not st["result"]["ok"]),
                           "business_outcome_correct": d["business_outcome_correct"], "grader": d["grader_version"],
                           "discrepancy_type": d["discrepancy_type"], "env_seconds": round(env._env_seconds, 4),
                           "predicate": d["evidence_predicate"], "unsafe_attempts": d["unsafe_attempts"],
                           "citation_problems": d["citation_problems"],
                           "efficiency_penalty": EFFICIENCY_PENALTY,
                           "contract_evidence_cited": all(r["satisfied"] for r in d["required_evidence"])
                           if d["required_evidence"] else None,
                           "contract_evidence_seen_before_action": all(
                               r["seen_before_action"] is not False for r in d["required_evidence"])
                           if d["required_evidence"] else None})
        if path is not None:
            _attach(path, grading="graded", diagnostics=LAST_BATCH[-1])
    state = kwargs.get("trainer_state")
    monitor_infrastructure([row.get("stop_reason") == "infrastructure_failure" for row in LAST_BATCH],
                           getattr(state, "global_step", None))
    return rewards
