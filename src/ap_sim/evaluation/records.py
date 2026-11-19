"""Per-episode evaluation records (the fields the pre-registered plan, preregistration_proof/PLAN_V2.md, lists under
Phase 5, the evaluation harness).

Success is g8's verdict. Everything else is descriptive and never feeds the reward:
  success, outcome_correct, report_valid, citation_failure (kinds), violations (attempted, executed), final_action,
  stop_reason, finish_result_dropped, malformed_tool_calls, superseded_dispositions, efficiency (calls, business
  minutes, an unnecessary confirmation on a covered case), resolution time (release and keep-blocked cases: the
  deciding action's time against the grader's T_res), time to the first confirmation request (pending cases),
  discrepancy_type correctness, finish_reason and last token of the final turn, the training reward the same
  episode would get, and the Phase 6 gate's inputs (coverage calls; turns that attempted a tool call and turns parsed).
"""
from __future__ import annotations

from ap_sim.events.clock import format_time
from ap_sim.grader import arithmetic as ar
from ap_sim.grader.grader import REPORT_CRITICAL, _resolution_op
from ap_sim.grading import g8
from ap_sim.training import metrics, rewards

DECIDING = {"release": "release", "keep_blocked": "keep_blocked_shortfall"}   # outcome -> the engine's action name


def _executed_violations(scenario, log: dict) -> int:
    """Unsafe operations the engine accepted (state changed). The engine refuses them, so this is an environment
    guard: it should always be 0."""
    inv_id = scenario.current_invoice
    po_line = next(r.po_line for r in scenario.records if r.kind == "invoice_line" and r.invoice == inv_id)
    profile = getattr(scenario, "policy", "sandbox-v1")
    n = 0
    for op in log["operations"]:
        if not op["attempt"]["accepted"]:
            continue
        if op["action"] == "release" and g8._release_unsafe(op["before"], inv_id, po_line, profile):
            n += 1
        elif op["action"] == "post_receipt" and op["attempt"].get("actor") == "ap_agent":
            n += 1
    return n


def _citation_kinds(verdict, log: dict) -> dict:
    """strong_model_sweep.citation_kinds on this episode (value_format, invoice_not_line, follow_up_version, other)."""
    import strong_model_sweep as sw   # scripts/ (the classifier the strong-model sweep and check used)
    rec = {"verdict": {"report": log.get("report"), "citation_problems": verdict.diagnostics["citation_problems"],
                       "required_evidence": verdict.diagnostics["required_evidence"]}, "episode_log": log}
    return dict(sw.citation_kinds(rec))


def episode_record(env, classification: dict, turns: list[dict] | None = None, extra: dict | None = None) -> dict:
    scenario, log, _ = env._episode()
    v = g8.grade(scenario, log)
    d = v.diagnostics
    reason = classification["stop_reason"]
    reward = rewards.training_reward(v, reason, scenario, log)
    report_valid = bool(d["finish_accepted"]) and not (set(v.critical) & REPORT_CRITICAL)
    resolution = None
    if v.expected in DECIDING and v.reached == v.expected:
        op = _resolution_op(log, DECIDING[v.expected])
        if op is not None:
            at = op["attempt"]["at"]
            resolution = {"at": format_time(at), "minutes": at, "t_res": v.t_res,
                          "late_by_minutes": None if v.t_res == float("inf") else at - v.t_res}
    first_request = metrics.first_confirmation_request_at(log) if v.expected == "pending" else None
    last = (turns or [{}])[-1]
    out = {
        "case": scenario.name, "expected": v.expected, "final_action": v.reached,
        "success": v.reward == 1, "outcome_correct": bool(d["business_outcome_correct"]), "report_valid": report_valid,
        "critical": list(v.critical),
        "citation_failure": _citation_kinds(v, log) if log.get("report") else {},
        "violations": {"attempted": d["unsafe_attempts"], "executed": _executed_violations(scenario, log),
                       "refused": len(d.get("refused_unsafe_attempts", [])),
                       "post_end": len(d.get("post_end_unsafe_attempts", []))},
        "stop_reason": reason, "finish_result_dropped": classification["finish_result_dropped"],
        "malformed_tool_calls": classification["malformed_tool_calls"],
        "superseded_dispositions": d.get("superseded_dispositions", {}),
        "efficiency": {"calls": log["calls"], "business_minutes": log["final_clock"],
                       "unnecessary_confirmation": rewards.covered_at_start(scenario)
                       and rewards.requested_confirmation(log)},
        "resolution_time": resolution,
        "first_confirmation_request_minutes": first_request,
        "discrepancy_type": d.get("discrepancy_type"),
        "discrepancy_type_correct": (d.get("discrepancy_type") or {}).get("correct"),
        "finish_reason": last.get("finish_reason"), "last_token": last.get("last_token"),
        "training_reward": reward, "infrastructure": env._infrastructure_failure,
        "coverage_calls": sum(1 for st in log["steps"] if st["tool"] == "coverage"),
        # each turn's flags from its raw decoded text (monitors.turn_flags)
        "turn_parse": {"attempted": sum(1 for t in turns or [] if t.get("attempted")),
                       "parsed": sum(1 for t in turns or [] if t.get("parsed"))},
        # anomaly counter: the pinned instruct model does not think; a <think> block is investigated
        "think_turns": sum(1 for t in turns or [] if t.get("think_block")),
        "post_end_calls": d.get("post_end_calls", 0),
    }
    return {**out, **(extra or {})}
