"""Grader sensitivity: incomplete arithmetic expressions with correct stated values (post hoc and exploratory, 2026-11-10).

Existing V6 and stress episodes only; no GPU; the frozen grader g8 is called, never changed. g8 requires a derived figure's expression to recompute exactly to its stated
value. This asks how success would read if an incomplete expression with a correct stated value were accepted. It is a sensitivity result, not a corrected score.

Each derived entry in a failed report whose expression does not equal its value is classed, mechanically:
  (a) incomplete expression, correct value: the stated value equals a quantity the grader's own arithmetic (grader/arithmetic.py) gives for this invoice and PO line
      (available coverage, net receipts, other active invoices, or the gap) at the case's start or at the deciding action; every record that quantity needs was in an
      observation delivered to the agent; and the expression's operands omit at least one of them;
  (b) incorrect value: the stated value equals none of those quantities;
  (o) correct value, but not (a): the needed records were all cited, or one was never shown. Reported; not accepted.
An episode's report is also (c) unsupported if g8 lists any citation problem other than a derived mismatch. The variant counts an episode as a success only if g8's sole
critical flag is citation_failure, the outcome and the finish were otherwise accepted, every citation problem is a derived mismatch, and every mismatched entry is (a).

    python scripts/posthoc/grader_sensitivity.py --out runs/posthoc/grader_sensitivity
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import posthoc_common as c   # noqa: E402

CELLS = {
    "v6": {"base": c.R2 / "base", "sft-v2": c.R2 / "sft-v2", "grpo-20261101": c.R2 / "grpo", "grpo-20261107": c.E1 / "grpo-20261107", "grpo-20261108": c.E1 / "grpo-20261108"},
    "stress": {"sft-v2": c.E1 / "sft-v2", "rft": c.E1 / "rft", "grpo-20261101": c.E1 / "grpo-20261101", "grpo-20261107": c.E1 / "grpo-20261107",
               "grpo-20261108": c.E1 / "grpo-20261108"},
}
CONTROL_STRESS = {"A: sft-v2 + reminder": "sft-v2-reminder", "B: sft + one epoch": "sft-extra-epoch"}
MISMATCH = "derived result does not equal its expression"
QUANTITIES = ("available coverage", "net receipts", "other active invoices", "gap")


def initial_snapshot(scenario) -> dict:
    snap = {}
    for r in scenario.records:
        snap.setdefault(r.kind, []).append(r.model_dump(mode="json"))
    return snap


def header(record_id: str) -> str:
    """An invoice line's invoice (INV-123-1 -> INV-123); other ids unchanged."""
    parts = record_id.split("-")
    return "-".join(parts[:2]) if record_id.startswith("INV-") and len(parts) == 3 else record_id


def quantities(snap: dict, po_line: str, invoice: str, profile: str) -> dict:
    """Each quantity's value and the records it needs, by the grader's own arithmetic."""
    from ap_sim.grader import arithmetic as ar
    receipts = {r["id"] for r in ar.rows(snap, "receipt") if r["po_line"] == po_line}
    states = {i["id"]: i["state"] for i in ar.rows(snap, "invoice")}
    others = {l["invoice"] for l in ar.rows(snap, "invoice_line") if l["po_line"] == po_line and l["invoice"] != invoice and states.get(l["invoice"]) != "cancelled"}
    line = {l["id"] for l in ar.rows(snap, "invoice_line") if l["invoice"] == invoice and l["po_line"] == po_line}
    return {"available coverage": (ar.coverage(snap, po_line, invoice), receipts | others), "net receipts": (ar.net_receipts(snap, po_line), receipts),
            "other active invoices": (ar.other_active(snap, po_line, invoice), others), "gap": (ar.gap(snap, po_line, invoice, profile), receipts | others | line)}


def classify(scenario, log, verdict) -> dict:
    from ap_sim.grader import grader as g7
    observations = g7._delivered(log)
    shown = {i for _, obs in observations for region in obs["regions"] for rec in region["records"] for i in (rec["id"], header(rec["id"]))}
    invoice = scenario.current_invoice
    start = initial_snapshot(scenario)
    po_line = next(l["po_line"] for l in start["invoice_line"] if l["invoice"] == invoice)
    profile = getattr(scenario, "policy", "sandbox-v1")
    snaps = [("start", start)]
    key = {"release": "release", "keep_blocked": "keep_blocked_shortfall", "pending": "keep_blocked_pending"}.get(verdict.reached)
    op = g7._resolution_op(log, key) if key else None
    if op is not None:
        snaps.insert(0, ("deciding action", op["before"]))
    candidates = [(when, name, value, needed) for when, snap in snaps for name, (value, needed) in quantities(snap, po_line, invoice, profile).items()]
    report = log.get("report") or {}
    entries = [e for f in report.get("findings") or [] for e in f.get("evidence") or []] + list(report.get("evidence") or [])
    classes = []
    for e in entries:
        if e.get("form") != "derived":
            continue
        values, ok = {}, True
        for name, o in (e.get("operands") or {}).items():
            found, value = g7._find(observations, o["observation_id"], o["record_id"], o["record_version"], o["field"])
            try:
                values[name] = g7._numeric(value) if found else None
            except InvalidOperation:
                values[name] = None
            ok = ok and values[name] is not None
        if not ok:
            continue                                        # an unsupported operand: g8 lists it, and the episode is (c)
        try:
            stated = Decimal(e["value"])
            if g7.evaluate_expression(e["expression"], values) == stated:
                continue
        except (ValueError, KeyError, InvalidOperation):
            continue                                        # cannot be evaluated: (c)
        cited = {header(o["record_id"]) for o in e["operands"].values()} | {o["record_id"] for o in e["operands"].values()}
        match = [(when, name, needed) for when, name, value, needed in candidates if value == stated]
        if not match:
            classes.append({"class": "b"})
        elif any(needed <= shown and not needed <= cited for _, _, needed in match):
            when, name, needed = next(m for m in match if m[2] <= shown and not m[2] <= cited)
            observed = {x for ev in entries if ev.get("form") == "observed" for x in (ev.get("record_id"), header(ev.get("record_id") or ""))}
            classes.append({"class": "a", "quantity": name, "at": when, "omitted_cited_elsewhere": (needed - cited) <= observed})
        else:
            classes.append({"class": "o"})
    return {"entries": classes}


def run(cells: dict) -> dict:
    from ap_sim.grading import g8
    out = {"label": "sensitivity, not a corrected score; g8 stays the measure of record (docs/V2.md, 2026-11-10)", "sets": {}}
    for the_set, models in cells.items():
        factories = c.cases(the_set)
        block = {}
        for model, path in models.items():
            eps = c.episodes(path / f"{the_set}-greedy" / "episodes.jsonl")
            row = {"cases": len(eps), "success_g8": 0, "success_variant": 0, "flipped": [], "entries": {"a": 0, "b": 0, "o": 0},
                   "a_by_quantity": {}, "a_omitted_records_cited_elsewhere": 0, "episodes_with_mismatch": 0, "episodes_unsupported": 0, "failures_not_citation_only": 0}
            for case, rec in eps.items():
                scenario, log, _ = c.replay(factories[case], case, rec)
                v = g8.grade(scenario, log)
                if (v.reward == 1) != bool(rec["success"]) or sorted(v.critical) != sorted(rec["critical"]):
                    raise SystemExit(f"{model} {the_set}: a replay disagrees with the stored verdict")
                if v.reward == 1:
                    row["success_g8"] += 1
                    row["success_variant"] += 1
                    continue
                problems = v.diagnostics.get("citation_problems") or []
                mismatches = [p for p in problems if p.startswith(MISMATCH)]
                got = classify(scenario, log, v)["entries"]
                if len(got) != len(mismatches):
                    raise SystemExit(f"{model} {the_set}: {len(got)} mismatched derived entries found, g8 lists {len(mismatches)}")
                for e in got:
                    row["entries"][e["class"]] += 1
                    if e["class"] == "a":
                        k = f"{e['quantity']} at the {e['at']}"
                        row["a_by_quantity"][k] = row["a_by_quantity"].get(k, 0) + 1
                        row["a_omitted_records_cited_elsewhere"] += e["omitted_cited_elsewhere"]
                row["episodes_with_mismatch"] += bool(mismatches)
                row["episodes_unsupported"] += len(problems) > len(mismatches)
                citation_only = list(v.critical) == ["citation_failure"] and v.reached == v.expected and v.diagnostics.get("finish_accepted")
                row["failures_not_citation_only"] += not citation_only
                if citation_only and mismatches and len(problems) == len(mismatches) and all(e["class"] == "a" for e in got):
                    row["success_variant"] += 1
                    row["flipped"].append(case.split("-")[1])     # the case type only
            row["flipped"] = c.tally(row["flipped"])
            block[model] = row
        out["sets"][the_set] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-controls", action="store_true", help="also the controls' stress cells, when they exist")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = {s: dict(m) for s, m in CELLS.items()}
    if a.with_controls:
        for name, model in CONTROL_STRESS.items():
            cell = c.CONTROLS / model / "stress-greedy"
            if (cell / ".done").exists() and not (cell / ".invalid").exists():
                cells["stress"][name] = c.CONTROLS / model
    res = run(cells)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    for s, b in res["sets"].items():
        for m, r in b.items():
            print(s, m, "g8", r["success_g8"], "variant", r["success_variant"], "entries", r["entries"], "a cited elsewhere", r["a_omitted_records_cited_elsewhere"], "unsupported", r["episodes_unsupported"], "not citation-only", r["failures_not_citation_only"], r["a_by_quantity"])


if __name__ == "__main__":
    main()
