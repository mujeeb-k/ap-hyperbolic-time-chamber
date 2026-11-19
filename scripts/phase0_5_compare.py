"""Verifier-versus-grader comparison (read-only, no API, no model): the independent verifier against the grader.

The comparison was run before the v2 splits were built. It needs two inputs that are not part of this release, the
independent verifier and a hand-check worksheet, so it does not run from the release alone. It is kept as the record of
how the comparison was made.

  1. Generates 1,000 fresh cases with the causal generator (seeds FRESH_SEEDS) into runs/phase0_5/fresh_1000.json.
  2. Runs the independent verifier (a separate process) over the train pool, the validation set, the frozen test sets
     (IID, compositional, policy shift) and the fresh cases. On the test sets this is a grader-style check with no model
     output.
  3. Computes the grader's values for every case with the first part of grader.grade (gap at start, unrecorded
     quantity, T_res with the g5/g6 correction, expected outcome) and its coverage arithmetic, and compares coverage,
     gap, T_res and expected outcome. The manifest's stored `expected` is compared too.
  4. Compares the hand-check worksheet's filled-in answers with both.
  5. Lists every case with a negative warehouse answer due between D5 16:41 and 16:55 (the A1 window).

    uv run --frozen python scripts/phase0_5_compare.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import subprocess
import sys
from decimal import Decimal

from ap_sim.events.clock import HORIZON, format_time
from ap_sim.generator import causal
from ap_sim.generator.manifests import load
from ap_sim.grader import arithmetic as ar

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "phase0_5"
FRESH_SEEDS = range(9_100_000, 9_101_000)
SETS = {"train_pool": ROOT / "manifests/frozen/train_pool.json",
        "validation": ROOT / "manifests/frozen/validation.json",
        "iid_test": ROOT / "manifests/frozen/iid_test.json",
        "compositional": ROOT / "manifests/frozen/compositional.json",
        "policy_shift": ROOT / "manifests/frozen_policy_shift/policy_shift.json",
        "fresh_1000": OUT / "fresh_1000.json"}
A1_WINDOW = (2381, 2395)   # D5 16:41 .. D5 16:55


def fresh_manifest() -> dict[str, callable]:
    entries, factories = [], {}
    for seed in FRESH_SEEDS:
        c = causal.generate(f"F-{seed}", seed)
        params = {k: (list(map(list, v)) if k == "reversals" else list(v) if isinstance(v, tuple) else v)
                  for k, v in c.params.items()}
        entries.append({"case_id": c.case_id, "seed": seed, "policy": "sandbox-v1", "variant": c.variant, "params": params})
        factories[c.case_id] = c.scenario
    OUT.mkdir(parents=True, exist_ok=True)
    SETS["fresh_1000"].write_text(json.dumps({"manifest": "fresh_1000 (phase 0.5)", "generator": causal.GENERATOR_VERSION,
                                              "seeds": [FRESH_SEEDS.start, FRESH_SEEDS.stop - 1], "entries": entries}))
    return factories


def grader_side(scenario) -> dict:
    """The first part of grader.grade, verbatim in substance (lines 'Acceptable outcome and T_res'), plus coverage."""
    initial = {}
    for r in scenario.records:
        initial.setdefault(r.kind, []).append(r.model_dump(mode="json"))
    inv = scenario.current_invoice
    po_line = next(l["po_line"] for l in initial["invoice_line"] if l["invoice"] == inv)
    profile = getattr(scenario, "policy", "sandbox-v1")
    gap0 = ar.gap(initial, po_line, inv, profile)
    net0 = ar.net_receipts(initial, po_line)
    delay = scenario.world.confirmation_delay
    unrecorded = None
    if gap0 > 0 and delay is not None:
        content = sum((d.quantity for d in scenario.world.deliveries if d.po_line == po_line and d.at <= delay), Decimal(0))
        unrecorded = content - net0
    try:
        t, branch, calls = ar.t_res(gap0, unrecorded, delay, corrected=True)
        expected = branch if t <= scenario.horizon else "pending"
    except ValueError as exc:
        t, branch, calls, expected = None, "error", None, f"error: {exc}"
    raw_gap = ar.line_qty(initial, inv, po_line) - ar.coverage(initial, po_line, inv)
    return {"coverage": ar.coverage(initial, po_line, inv), "gap": gap0, "raw_gap": max(raw_gap, Decimal(0)),
            "net": net0, "unrecorded": unrecorded, "delay": delay, "t_res": t, "branch": branch, "calls": calls,
            "expected": expected, "profile": profile}


def num(x):
    return None if x is None else (float("inf") if x == float("inf") else Decimal(str(x)))


def verifier_side(c: dict) -> dict:
    a, r, g = c.get("at_start") or {}, c.get("resolution") or {}, c.get("gap_case") or {}
    t = r.get("t_res")   # for a pending case: the hypothetical completion, with completes_by_horizon false
    # The grader's gap is under the active profile (sandbox-v2's tolerance included). The verifier reports that as
    # gap_under_active_profile since its 2026-10-30 update. Its plain "gap" is the 0% excess.
    return {"coverage": a.get("available_coverage"), "gap": a.get("gap_under_active_profile", a.get("gap")), "unrecorded": g.get("unrecorded_qty"),
            "delay": g.get("confirmation_delay"), "t_res": t, "workflow": r.get("workflow"), "calls": r.get("calls"),
            "completes": r.get("completes_by_horizon"), "expected": c.get("expected_outcome"),
            "warnings": c.get("warnings"), "fields_used": c.get("fields_used"), "error": c.get("error")}


def same_t(grader_t, v: dict) -> bool:
    """A finite T_res must be equal; an infinite one (no workflow exists) must be matched by 'does not complete'."""
    if grader_t is None:
        return v["t_res"] is None
    if grader_t == float("inf"):
        return v["completes"] is not True
    return v["t_res"] is not None and num(v["t_res"]) == num(grader_t)


def main() -> None:
    fresh = fresh_manifest()
    results, disagreements, a1 = {}, [], []
    stored_mismatch = collections.Counter()
    hand = {}
    for name, path in SETS.items():
        vout = OUT / f"independent_verifier-{path.stem}.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/independent_verifier.py"), "--manifest", str(path),
                        "--out", str(vout)], check=True, cwd=ROOT)
        vcases = {c["case_id"]: c for c in json.loads(vout.read_text())["cases"]}
        entries = {e["case_id"]: e for e in json.loads(path.read_text())["entries"]}
        factories = fresh if name == "fresh_1000" else load(path)
        tally = collections.Counter()
        for cid, factory in factories.items():
            s = factory()
            g, v = grader_side(s), verifier_side(vcases[cid])
            e = entries[cid]
            tally["cases"] += 1
            if "expected" in e and e["expected"] != g["expected"]:
                stored_mismatch[name] += 1
            diffs = []
            if v["coverage"] is None or num(v["coverage"]) != num(g["coverage"]):
                diffs.append("coverage")
            if v["gap"] is None or num(v["gap"]) != num(g["gap"]):
                diffs.append("gap")
            if not same_t(g["t_res"], v):
                diffs.append("t_res")
            if v["expected"] != g["expected"]:
                diffs.append("expected_outcome")
            tally["agree_all" if not diffs else "disagree"] += 1
            for d in diffs:
                tally[f"disagree:{d}"] += 1
            if diffs:
                disagreements.append({"set": name, "case": cid, "variant": e.get("variant"), "policy": e.get("policy"),
                                      "fields": diffs,
                                      "grader": {k: (str(x) if isinstance(x, Decimal) else x) for k, x in g.items()},
                                      "verifier": {k: v[k] for k in ("coverage", "gap", "unrecorded", "delay", "t_res",
                                                                      "workflow", "calls", "completes", "expected",
                                                                      "warnings", "error")}})
            if g["gap"] > 0 and g["unrecorded"] is not None and g["unrecorded"] <= 0 \
                    and g["delay"] is not None and A1_WINDOW[0] <= g["delay"] <= A1_WINDOW[1]:
                a1.append({"set": name, "case": cid, "delay": g["delay"], "answer_at": format_time(g["delay"])})
            if name == "train_pool":
                hand[cid] = (g, v)
        tally["delays"] = dict(collections.Counter(str(g) for g in (f().world.confirmation_delay for f in factories.values())))
        results[name] = dict(tally)
    # Hand check: the summary table's filled-in rows. The "user" key holds the answers filled in by hand.
    rows = re.findall(r"^\| (\d) \| `([^`]+)` \| ([^|]*) \| ([^|]*) \|$", (ROOT / "docs/HAND_CHECK.md").read_text(), re.M)
    hand_out = []
    for _, cid, cov, outcome in rows:
        g, v = hand[cid]
        hand_out.append({"case": cid, "user": {"coverage": cov.strip(), "expected": outcome.strip()},
                         "verifier": {"coverage": v["coverage"], "expected": v["expected"]},
                         "grader": {"coverage": str(g["coverage"]), "expected": g["expected"]},
                         "all_agree": cov.strip() == str(v["coverage"]) == str(g["coverage"])
                         and outcome.strip() == v["expected"] == g["expected"]})
    by_cause = collections.defaultdict(list)
    for d in disagreements:
        by_cause[(tuple(d["fields"]), d["policy"], d["variant"])].append(d)
    summary = {"fresh_seeds": [FRESH_SEEDS.start, FRESH_SEEDS.stop - 1], "sets": results,
               "stored_expected_vs_grader_mismatches": dict(stored_mismatch),
               "disagreements_by_cause": {" | ".join(map(str, k)): {"n": len(v), "cases": [x["case"] for x in v],
                                                                     "example": v[0]} for k, v in by_cause.items()},
               "hand_check": hand_out, "a1_window": a1}
    (OUT / "phase0_5_compare.json").write_text(json.dumps(summary, indent=1, default=str))
    (OUT / "phase0_5_disagreements.json").write_text(json.dumps(disagreements, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("fresh_seeds", "sets", "stored_expected_vs_grader_mismatches", "hand_check", "a1_window")},
                     indent=1, default=str))
    print("disagreement groups:")
    for k, v in summary["disagreements_by_cause"].items():
        print(f"  {v['n']:4d}  {k}")


if __name__ == "__main__":
    main()
