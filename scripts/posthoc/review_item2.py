"""V6 derived coverage figures, compared with the SFT-v2 demonstrations' expressions and with the grader's own netting (post hoc, 2026-11-12).

Definitions, recorded before this script ran:
  demonstration templates  the `expression` of every derived evidence entry in a `finish` call of a trained assistant turn of the 300 SFT-v2
                           demonstrations (runs/phase4/sft-demos-v2-1/transcripts/*.md, written by the builder with train.jsonl, whose sha256 is
                           checked); planted-mistake turns are not trained and are excluded.
  exact match              the expression equals a template after whitespace is collapsed (operand names included).
  structural match         the multiset of (sign, kind) over the expression's terms, kind = receipt / reversal / invoice by the record the operand names,
                           equals that of some demonstration expression (read the same way on its own SFT scenario); names, order and numbering ignored;
                           an expression with parentheses or '*' has no linear signature and matches only an identical demonstration structure (none exists).
  derived coverage figure  a derived entry of the accepted report with at least one operand naming a receipt or a reversal.
  (a)                      the stated value equals the matching figure of a coverage() result in the same episode: available_coverage if the expression
                           has an invoice operand, else net_posted_receipts; for any coverage() call, and for the last call before finish; episodes without
                           a coverage() call are counted apart.
  (c1) / (c2)              an observed entry of the accepted report names a reversal record / the receipt a reversal reverses.
  equals grader netting    (added 2026-11-12 for the grader-validity note, after the first run) the stated value equals the grader's own net posted receipts
                           or available coverage for this invoice and PO line, at the case's start or at the deciding action (manuscript_counts.stated_values).

Stored episodes are replayed through the frozen environment (posthoc_common.replay), and each replay's g8 verdict must equal the stored record. Counts and
case ids only: no case content is printed or written.

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice .venv/bin/python scripts/posthoc/review_item2.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import posthoc_common as c   # noqa: E402
import manuscript_counts as mc   # noqa: E402  (stated_values, NETTING: the grader's net receipts and available coverage, read-only)
import review_anchor   # noqa: E402  (the anchor rule, 2026-11-12)

DEMOS = c.ROOT / "runs/phase4/sft-demos-v2-1"
TRAIN_SHA = "2e381747645add5976ac8969e557ebf742f515e1cb37804402f6f650991d59da"
OUT = c.ROOT / "runs/posthoc/review_findings/item2.json"
V6 = {"base": c.R2 / "base", "sft-v2": c.R2 / "sft-v2", "grpo-20261101": c.R2 / "grpo",
      "grpo-20261107": c.E1 / "grpo-20261107", "grpo-20261108": c.E1 / "grpo-20261108"}
REPEAT = {"test": {"sft-v2": c.R2 / "sft-v2", "grpo-20261101": c.R2 / "grpo"},
          "stress": {"sft-v2": c.E1 / "sft-v2", "grpo-20261101": c.E1 / "grpo-20261101"}}
SECTION = re.compile(r"^## \[(\d+)\] (\w+)(?: `\w+`)? \((.*)\)\s*$")
TERM = re.compile(r"\s*([+-]?)\s*([a-z][a-z0-9_]*)")


def norm(expr: str) -> str:
    return " ".join(expr.split())


def signature(expr: str, kinds: dict[str, str]):
    """The multiset of (sign, kind) over a linear expression's terms, or None if it is not a plain sum of named operands."""
    if any(ch in expr for ch in "()*"):
        return None   # number literals fail TERM below, so they also give None
    pos, out = 0, collections.Counter()
    text = expr.strip()
    while pos < len(text):
        m = TERM.match(text, pos)
        if not m or m.end() == pos:
            return None
        sign = "-" if m.group(1) == "-" else "+"
        if pos > 0 and not m.group(1):
            return None
        out[(sign, kinds.get(m.group(2), "unknown"))] += 1
        pos = m.end()
    return frozenset(out.items())


def kind_of(record_id: str, reversals: set[str]) -> str:
    if record_id in reversals:
        return "reversal"
    if record_id.startswith("GR-"):
        return "receipt"
    if record_id.startswith("INV-"):
        return "invoice"
    return "other"


def scenario_reversals(scenario) -> tuple[set[str], set[str]]:
    rev = {r.id for r in scenario.records if getattr(r, "movement", None) == "reversal"}
    reversed_ = {r.reverses for r in scenario.records if getattr(r, "movement", None) == "reversal"}
    return rev, reversed_


def demo_templates() -> dict:
    sha = hashlib.sha256((DEMOS / "train.jsonl").read_bytes()).hexdigest()
    if sha != TRAIN_SHA:
        raise SystemExit("train.jsonl does not match the frozen sha256")
    from ap_sim.generator.splits import load_split
    factories = load_split("sft")
    exact, structural, finish_calls, entries = set(), set(), 0, 0
    for path in sorted((DEMOS / "transcripts").glob("sft-*.md")):
        case = path.stem
        scenario = factories[case]()
        rev, _ = scenario_reversals(scenario)
        lines = path.read_text().splitlines()
        trained, i = False, 0
        while i < len(lines):
            m = SECTION.match(lines[i])
            if m:
                trained = m.group(2) == "assistant" and m.group(3) == "trained"
            elif trained and lines[i].strip() == "Call `finish`:":
                j = i + 1
                while not lines[j].startswith("```"):
                    j += 1
                k = j + 1
                while not lines[k].startswith("```"):
                    k += 1
                report = json.loads("\n".join(lines[j + 1:k]))
                finish_calls += 1
                for e in [e for f in report.get("findings") or [] for e in f.get("evidence") or []] + list(report.get("evidence") or []):
                    if e.get("form") != "derived":
                        continue
                    entries += 1
                    exact.add(norm(e["expression"]))
                    kinds = {n: kind_of(o["record_id"], rev) for n, o in (e.get("operands") or {}).items()}
                    sig = signature(e["expression"], kinds)
                    if sig is not None:
                        structural.add(sig)
                i = k
            i += 1
    return {"exact": exact, "structural": structural, "finish_calls": finish_calls, "derived_entries": entries, "train_sha256": sha}


def coverage_results(log: dict) -> list[dict]:
    return [st["result"]["coverage"] for st in log["steps"] if st["tool"] == "coverage" and isinstance(st.get("result"), dict)
            and isinstance(st["result"].get("coverage"), dict)]


def report_entries(report: dict | None) -> list[dict]:
    report = report or {}
    return [e for f in report.get("findings") or [] for e in f.get("evidence") or []] + list(report.get("evidence") or [])


def analyse(the_set: str, path: pathlib.Path, templates: dict, with_bc: bool) -> dict:
    from ap_sim.grading import g8
    factories = c.cases(the_set)
    rows = {"episodes": 0, "with_coverage_figure": 0, "figures": 0, "by_outcome": {}, "cases": {}}
    for case, rec in c.episodes(path / f"{the_set}-greedy" / "episodes.jsonl").items():
        scenario, log, _ = c.replay(factories[case], case, rec)
        v = g8.grade(scenario, log)
        if (v.reward == 1) != bool(rec["success"]) or sorted(v.critical) != sorted(rec["critical"]):
            raise SystemExit(f"{path.name} {the_set} {case}: the replay disagrees with the stored verdict")
        rows["episodes"] += 1
        rev, reversed_ = scenario_reversals(scenario)
        covs = coverage_results(log)
        for cv in covs:
            rev |= {r["id"] for r in cv.get("reversals") or []}
            reversed_ |= {r["reverses"] for r in cv.get("reversals") or [] if r.get("reverses")}
        entries = report_entries(log.get("report"))
        figs = []
        for e in entries:
            if e.get("form") != "derived":
                continue
            kinds = {n: kind_of(o["record_id"], rev) for n, o in (e.get("operands") or {}).items()}
            if not any(k in ("receipt", "reversal") for k in kinds.values()):
                continue
            has_invoice = any(k == "invoice" for k in kinds.values())
            key = "available_coverage" if has_invoice else "net_posted_receipts"
            try:
                value = Decimal(str(e["value"]))
            except (InvalidOperation, KeyError, TypeError):
                value = None
            figures = [Decimal(cv[key]["value"]) for cv in covs]
            fig = {"figure": key, "no_coverage_call": not covs,
                   "a_any_call": value is not None and value in figures,
                   "a_last_call": value is not None and bool(figures) and value == figures[-1]}
            fig["equals_grader_netting"] = value is not None and value in mc.stated_values(scenario, log, v, mc.NETTING)
            if with_bc:
                sig = signature(e["expression"], kinds)
                fig["b_exact"] = norm(e["expression"]) in templates["exact"]
                fig["b_structural"] = sig is not None and sig in templates["structural"]
                fig["has_reversal_operand"] = any(k == "reversal" for k in kinds.values())
                fig["reversal_terms"] = sorted(f"{s}{n[0]}" for n, k in kinds.items() if k == "reversal"
                                               for s in ["-" if re.search(r"-\s*" + re.escape(n) + r"\b", e["expression"]) else "+"])
            figs.append(fig)
        if not figs:
            continue
        observed = {e.get("record_id") for e in entries if e.get("form") == "observed"}
        outcome = "passed" if v.reward == 1 else "failed"
        row = {"outcome": outcome, "figures": figs}
        if with_bc:
            row["c1_reversal_cited_observed"] = bool(rev & observed)
            row["c2_reversed_receipt_cited_observed"] = bool(reversed_ & observed)
            row["reversals_in_case"] = len(rev)
        rows["cases"][case] = row
        rows["with_coverage_figure"] += 1
        rows["figures"] += len(figs)
    tallies = {}
    for outcome in ("passed", "failed"):
        cs = {k: r for k, r in rows["cases"].items() if r["outcome"] == outcome}
        figs = [(k, f) for k, r in cs.items() for f in r["figures"]]
        t = {"episodes": len(cs), "figures": len(figs),
             "no_coverage_call": sorted({k for k, f in figs if f["no_coverage_call"]}),
             "a_any_call_true": sum(f["a_any_call"] for _, f in figs),
             "a_last_call_true": sum(f["a_last_call"] for _, f in figs),
             "a_any_call_false_cases": sorted({k for k, f in figs if not f["a_any_call"] and not f["no_coverage_call"]}),
             "equals_grader_netting_true": sum(f["equals_grader_netting"] for _, f in figs),
             "equals_grader_netting_false_cases": sorted({k for k, f in figs if not f["equals_grader_netting"]})}
        if with_bc:
            t |= {"b_exact_true": sum(f["b_exact"] for _, f in figs), "b_structural_true": sum(f["b_structural"] for _, f in figs),
                  "has_reversal_operand": sum(f["has_reversal_operand"] for _, f in figs),
                  "reversal_terms_sign_and_name_letter": dict(collections.Counter(t for _, f in figs for t in f["reversal_terms"])),
                  "b_structural_false_cases": sorted({k for k, f in figs if not f["b_structural"]}),
                  "c1_true": sum(r["c1_reversal_cited_observed"] for r in cs.values()),
                  "c2_true": sum(r["c2_reversed_receipt_cited_observed"] for r in cs.values()),
                  "c1_false_cases": sorted(k for k, r in cs.items() if not r["c1_reversal_cited_observed"]),
                  "c2_false_cases": sorted(k for k, r in cs.items() if not r["c2_reversed_receipt_cited_observed"])}
        tallies[outcome] = t
    rows["by_outcome"] = tallies
    return rows


def main() -> None:
    anchors = review_anchor.check("item2")
    templates = demo_templates()
    out = {"anchor": anchors, "templates": {"finish_calls": templates["finish_calls"], "derived_entries": templates["derived_entries"],
                         "exact": sorted(templates["exact"]),
                         "structural": sorted(sorted(f"{s}{k}x{n}" for (s, k), n in sig) for sig in templates["structural"]),
                         "train_sha256": templates["train_sha256"]},
           "v6": {m: analyse("v6", p, templates, True) for m, p in V6.items()},
           "repeat_a": {s: {m: analyse(s, p, templates, False) for m, p in d.items()} for s, d in REPEAT.items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(OUT)
    for m, r in out["v6"].items():
        print("v6", m, r["episodes"], r["with_coverage_figure"], json.dumps({k: {kk: vv for kk, vv in t.items() if not isinstance(vv, list)} for k, t in r["by_outcome"].items()}))
    for s, d in out["repeat_a"].items():
        for m, r in d.items():
            print(s, m, r["episodes"], r["with_coverage_figure"], json.dumps({k: {kk: vv for kk, vv in t.items() if not isinstance(vv, list)} for k, t in r["by_outcome"].items()}))


if __name__ == "__main__":
    main()
