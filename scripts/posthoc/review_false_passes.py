"""False-pass audit: the prevalence of accepted derived figures that are not an accounting quantity of the environment.

Added on 2026-11-12, after the other post-hoc analyses. Post hoc, exploratory, read-only. No model is run and no grader is built: the comparison uses
the grader's own arithmetic and the definition the grader-sensitivity analysis already uses (scripts/posthoc/grader_sensitivity.py).

Definitions:
  accepted derived figure   a derived entry of an episode's report whose operand citations all resolve in observations delivered to the agent, are numeric,
                            and whose expression evaluates exactly to its stated value: g7/g8's derived check raises no problem for it.
  environment quantities    the grader's own values (grader/arithmetic.py, via grader_sensitivity.quantities) of available coverage, net receipts, other
                            active invoices and the gap, for the case's invoice and PO line, at the case's start or at the deciding action.
  not a quantity            an accepted derived figure whose stated value equals none of those values at either time (grader_sensitivity's class (b),
                            applied to accepted entries instead of mismatched ones).
  false pass                an episode g8 scores a success whose report has at least one accepted derived figure that is not a quantity.
A figure that is semantically wrong but whose value happens to equal one of the eight values is not detected, so the counts are a lower bound.

Cells: every finished greedy cell of Phase 10 (run-2), the follow-up (eval-1) and the controls (eval-1) on test, V6, the policy shift, the forgetting set and
stress. The reserve (test_reserve) is excluded: reading its reports would read reserve content. Each replay's verdict must equal the stored one.

Anchor (a figure the paper already reports): derived entries with an incorrect value in failed reports, by the same classifier: 0-3 per model on V6, 0-4 on stress.

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice .venv/bin/python scripts/posthoc/review_false_passes.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import posthoc_common as c   # noqa: E402
import grader_sensitivity as gs   # noqa: E402

OUT = c.ROOT / "runs/posthoc/review_findings/false_passes.json"
SETS = ("test", "v6", "policy_shift", "forgetting_l1", "stress")


def values_at(scenario, log, verdict) -> set[Decimal]:
    from ap_sim.grader import grader as g7
    invoice = scenario.current_invoice
    start = gs.initial_snapshot(scenario)
    po_line = next(l["po_line"] for l in start["invoice_line"] if l["invoice"] == invoice)
    profile = getattr(scenario, "policy", "sandbox-v1")
    snaps = [start]
    key = {"release": "release", "keep_blocked": "keep_blocked_shortfall", "pending": "keep_blocked_pending"}.get(verdict.reached)
    op = g7._resolution_op(log, key) if key else None
    if op is not None:
        snaps.append(op["before"])
    return {value for snap in snaps for value, _ in gs.quantities(snap, po_line, invoice, profile).values()}


def accepted_entries(log) -> list[dict]:
    from ap_sim.grader import grader as g7
    observations = g7._delivered(log)
    report = log.get("report") or {}
    out = []
    for e in [e for f in report.get("findings") or [] for e in f.get("evidence") or []] + list(report.get("evidence") or []):
        if e.get("form") != "derived":
            continue
        values = {}
        for name, o in (e.get("operands") or {}).items():
            found, value = g7._find(observations, o["observation_id"], o["record_id"], o["record_version"], o["field"])
            if not found:
                break
            try:
                values[name] = g7._numeric(value)
            except InvalidOperation:
                break
        else:
            try:
                if g7.evaluate_expression(e["expression"], values) == Decimal(e["value"]):
                    out.append(e)
            except (ValueError, KeyError, InvalidOperation):
                pass
    return out


def describe(scenario, log, verdict, entry) -> dict:
    """Categories only: the figure's shape (sign and kind per term), and the records it omits from, or adds to, the records available coverage needs
    at the deciding action (or the start), by kind and invoice state. No values or ids."""
    import review_item2 as r2
    from ap_sim.grader import grader as g7, arithmetic as ar
    invoice = scenario.current_invoice
    start = gs.initial_snapshot(scenario)
    po_line = next(l["po_line"] for l in start["invoice_line"] if l["invoice"] == invoice)
    key = {"release": "release", "keep_blocked": "keep_blocked_shortfall", "pending": "keep_blocked_pending"}.get(verdict.reached)
    op = g7._resolution_op(log, key) if key else None
    snap = op["before"] if op is not None else start
    _, needed = gs.quantities(snap, po_line, invoice, getattr(scenario, "policy", "sandbox-v1"))["available coverage"]
    states = {i["id"]: i["state"] for i in ar.rows(snap, "invoice")}
    reversals = {r["id"] for r in ar.rows(snap, "receipt") if r.get("movement") == "reversal"}

    def kind(i):
        if i in reversals:
            return "reversal"
        if i.startswith("GR-"):
            return "receipt"
        if i.startswith("INV-"):
            return "current invoice" if i == invoice else f"invoice ({states.get(i, 'unknown')})"
        return "other"
    cited = {gs.header(o["record_id"]) for o in entry["operands"].values()}
    kinds = {n: ("reversal" if gs.header(o["record_id"]) in reversals else "receipt" if o["record_id"].startswith("GR-") else "invoice")
             for n, o in entry["operands"].items()}
    sig = r2.signature(entry["expression"], kinds)
    return {"shape": sorted(f"{s}{k} x{n}" for (s, k), n in sig) if sig else None,
            "omitted_from_available_coverage": dict(collections.Counter(kind(i) for i in needed - cited)),
            "added_beyond_available_coverage": dict(collections.Counter(kind(i) for i in cited - needed)),
            "reached": verdict.reached}


def anchor() -> list[dict]:
    """The paper's 'derived entries with an incorrect value, in failed reports' per model, by grader_sensitivity's own classifier."""
    from ap_sim.grading import g8
    res = []
    cellsets = {k: dict(v) for k, v in gs.CELLS.items()}
    cellsets["stress"].update({name: c.CONTROLS / d for name, d in gs.CONTROL_STRESS.items()})   # the paper's table spans all seven stress models
    for the_set, models in cellsets.items():
        factories = c.cases(the_set)
        per = {}
        for model, path in models.items():
            b = 0
            for case, rec in c.episodes(path / f"{the_set}-greedy" / "episodes.jsonl").items():
                if rec["success"]:
                    continue
                scenario, log, _ = c.replay(factories[case], case, rec)
                b += sum(e["class"] == "b" for e in gs.classify(scenario, log, g8.grade(scenario, log))["entries"])
            per[model] = b
        lo, hi = min(per.values()), max(per.values())
        want = {"v6": (0, 3), "stress": (0, 4)}[the_set]
        res.append({"anchor": f"incorrect-value derived entries in failed {the_set} reports, per model", "v5_location": "Table 10 (and B.4)",
                    "expected": list(want), "reproduced": [lo, hi], "per_model": per, "ok": (lo, hi) == want})
    for r in res:
        print(f"anchor: {r['anchor']} [{r['v5_location']}]: expected {r['expected']}, reproduced {r['reproduced']}: {'ok' if r['ok'] else 'FAILED'}", flush=True)
    if not all(r["ok"] for r in res):
        raise SystemExit("anchor failed; nothing is written")
    return res


def cells() -> dict:
    out = {}
    for root, session in ((c.R2, "phase10"), (c.E1, "followup"), (c.CONTROLS, "controls")):
        for f in sorted(root.glob("*/*-greedy/episodes.jsonl")):
            the_set = f.parent.name[: -len("-greedy")]
            if the_set in SETS and (f.parent / ".done").exists():
                out[f"{session}/{f.parent.parent.name}/{the_set}"] = (the_set, f)
    return out


def main() -> None:
    from ap_sim.grading import g8
    anchors = anchor()
    factories = {}
    rows, totals = {}, collections.Counter()
    false_passes, detail = [], {}
    for key, (the_set, path) in cells().items():
        if the_set not in factories:
            if the_set == "forgetting_l1":
                from ap_sim import forgetting
                factories[the_set] = forgetting.load()
            else:
                factories[the_set] = c.cases(the_set)
        r = collections.Counter()
        fp_cases, fail_cases = [], []
        for case, rec in c.episodes(path).items():
            scenario, log, _ = c.replay(factories[the_set][case], case, rec)
            v = g8.grade(scenario, log)
            if (v.reward == 1) != bool(rec["success"]) or sorted(v.critical) != sorted(rec["critical"]):
                raise SystemExit(f"{key} {case}: the replay disagrees with the stored verdict")
            r["episodes"] += 1
            r["successes"] += v.reward == 1
            acc = accepted_entries(log)
            if not acc:
                continue
            allowed = values_at(scenario, log, v)
            bad = [e for e in acc if Decimal(e["value"]) not in allowed]
            r["episodes_with_accepted_figure"] += 1
            r["accepted_figures"] += len(acc)
            r["accepted_figures_not_a_quantity"] += len(bad)
            if v.reward == 1:
                r["successes_with_accepted_figure"] += 1
                if bad:
                    r["false_passes"] += 1
                    fp_cases.append(case)
                    detail[f"{key}:{case}"] = [describe(scenario, log, v, e) for e in bad]
            elif bad:
                r["failures_with_accepted_non_quantity"] += 1
                fail_cases.append(case)
        rows[key] = {**r, "false_pass_cases": sorted(fp_cases), "failed_episodes_with_accepted_non_quantity": sorted(fail_cases)}
        totals.update(r)
        false_passes += [f"{key}:{x}" for x in fp_cases]
        print(key, dict(r), sorted(fp_cases), flush=True)
    res = {"anchor": anchors, "label": "review addendum: accepted derived figures that are not an environment quantity (post hoc; lower bound)",
           "definition": __doc__.split("Definitions:")[1].split("Cells:")[0].strip(), "cells": rows, "totals": dict(totals), "false_passes": false_passes,
           "false_pass_detail": detail}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print("totals", dict(totals))
    print("wrote", OUT.relative_to(c.ROOT))


if __name__ == "__main__":
    main()
