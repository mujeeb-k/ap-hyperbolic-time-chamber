"""Shared helpers for the post-hoc analyses: reading stored episodes, replaying them through the frozen environment for g8's own
diagnostics, and the cancelled-invoice citation count. CPU only; no model, no GPU. Nothing here prints case content."""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/gpu")]

R2 = ROOT / "runs" / "session3-phase10" / "run-2"
E1 = ROOT / "runs" / "followup" / "eval-1"
CONTROLS = ROOT / "runs" / "controls" / "eval-1"


def episodes(path: pathlib.Path) -> dict[str, dict]:
    """case id -> stored record, in file order."""
    return {json.loads(line)["case"]: json.loads(line) for line in path.read_text().splitlines()}


def cases(the_set: str) -> dict:
    """The frozen factories, by program (the stress set only as frozen)."""
    if the_set == "test":
        from ap_sim.generator.splits import load_test
        return load_test()
    if the_set == "stress":
        from ap_sim.stress import messiness
        return messiness.load(messiness.MANIFEST, require_frozen=True)
    from ap_sim.generator.splits import load_split
    return load_split(the_set)


def replay(factory, case: str, rec: dict):
    """The episode through the frozen environment (loop-guard/1, purpose evaluation), as the audits did; returns (scenario, log, stop reason)."""
    from ap_sim.evaluation import rollout
    from ap_sim.training.environment import APReceiptEnvV2
    from ap_sim.training.loop_guard import LOOP_GUARD
    env = APReceiptEnvV2({case: factory}, execution_rule=LOOP_GUARD)
    env.reset(scenario=case, purpose="evaluation")
    tools = rollout.trl_tool_dict(env)
    for m in rec["messages"]:
        if m["role"] == "assistant":
            for call in m.get("tool_calls") or []:
                rollout.trl_tool_message(tools, call)
    return env._episode()


def g8_reasons(scenario, log) -> dict:
    """g8's verdict and its own reasons, with numbers and ids replaced by # so no case content is kept."""
    from ap_sim.grading import g8
    v = g8.grade(scenario, log)
    d = v.diagnostics
    return {"success": v.reward == 1, "critical": list(v.critical),
            "citation_problems": [re.sub(r"\S*\d\S*", "#", p) for p in d.get("citation_problems") or []],
            "required_unsatisfied": [f"{r['what']} {r.get('field', '')}".strip() for r in d.get("required_evidence") or [] if not r.get("satisfied")]}


def observed_records(rec: dict) -> list[dict]:
    out = []
    for m in rec["messages"]:
        if m["role"] != "tool":
            continue
        try:
            obs = json.loads(m["content"]).get("observation") or {}
        except (ValueError, AttributeError, TypeError):
            continue
        for region in obs.get("regions") or [] if isinstance(obs, dict) else []:
            out.extend(r for r in region.get("records") or [] if isinstance(r, dict))
    return out


def accepted_report(rec: dict) -> dict | None:
    """The finish call whose result was accepted (ok true), or None."""
    msgs = rec["messages"]
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            if call["function"]["name"] != "finish":
                continue
            reply = next((r for r in msgs[i + 1:] if r["role"] == "tool"), None)
            try:
                ok = reply is not None and json.loads(reply["content"]).get("ok") is True
            except (ValueError, TypeError):
                ok = False
            if ok and isinstance(call["function"]["arguments"], dict):
                return call["function"]["arguments"]
    return None


def evidence_items(report: dict) -> list[dict]:
    items = [e for e in report.get("evidence") or [] if isinstance(e, dict)]
    for f in report.get("findings") or []:
        if isinstance(f, dict):
            items += [e for e in f.get("evidence") or [] if isinstance(e, dict)]
    return items


def cancelled_citation(rec: dict) -> tuple[bool, bool]:
    """(a cancelled invoice appears in the tool results, the accepted report cites each one by a state evidence item): the mechanism check's definition."""
    seen = {r["id"] for r in observed_records(rec) if r.get("type") == "invoice" and (r.get("fields") or {}).get("state") == "cancelled"}
    if not seen:
        return False, False
    report = accepted_report(rec)
    cited = {e.get("record_id") for e in evidence_items(report) if e.get("field") == "state"} if report else set()
    return True, seen <= cited


def tally(values) -> dict:
    return dict(sorted(collections.Counter(values).items()))
