"""Counts the paper cites for V6, the policy shift and the stress failures (post hoc, from 2026-11-10).

Stored episodes are replayed in memory through the frozen environment and g8, as the audits did; each replayed verdict and its critical flags must equal the record. g8 is
called, never changed, and nothing stored is rewritten. Counts only: no case content is printed or kept (stress case ids are not kept either).

V6 (SFT-v2 and GRPO 20261101, Phase 10's run-2):
  right action: reached == expected.
  omits the reversal (the paper's count): a failure with the right action that has a derived entry g8 finds mismatched (its expression does not equal its value), whose
      operands leave out a reversal receipt the agent was shown, and whose stated value equals the grader's net receipts or available coverage for this invoice and PO
      line, at the case's start or at the deciding action.
  ... with citation_failure the only critical flag: the same, and g8's sole critical flag is citation_failure (a stricter reading).
  ... matching any of the four quantities: the value may equal any of grader_sensitivity.QUANTITIES (a check that the two-quantity reading loses nothing).
Policy shift (the same two models): the v2-policy cases (ids ending -v2), their failures, and the failures with the right action and discrepancy_type_wrong among the flags.
Stress (the follow-up's five stress cells): failures; those missing a cancelled-invoice citation (a required evidence item about a cancelled invoice unsatisfied); those
  with the wrong decision; those with a derived mismatch among g8's citation problems.
Stress failure breakdown (2026-11-10; the five cells and both controls' cells). Each failure gets every label that applies:
  wrong decision (reached != expected; checked against the stored outcome_correct); missing evidence (g8's required_evidence_missing); cancelled-invoice omission
  (a required item about a cancelled invoice unsatisfied; a subset of missing evidence); derived-figure failure (a derived mismatch among g8's citation problems);
  other citation problem (any other citation problem: a nonexistent citation, a source mismatch...); discrepancy type wrong; and none of these (a loop or an
  unaccepted finish with no flag). It is also given one primary category, in the order wrong decision, cancelled-invoice omission, other missing evidence,
  derived-figure failure, other. The label sets by case id, and each source file's sha256, go to results.json (git-ignored) so that every count traces to its episodes.
Base decisions (test, V6, policy shift, forgetting): episodes whose final action equals the expected one, against the record's outcome_correct, which also
  requires no non-report critical flag; and the flags of the episodes where they differ. For every other model and cell the two counts are equal (checked here).
Seed 20261108 on the forgetting set: each failure's flags and decision, and each observed citation g8 cannot find in the observation it names, classed by whether it
  is the current invoice, its field, and whether that record version appears in a later observation instead.
Decision errors (test and stress; every evaluated model's greedy cell), from the stored records' final_action, expected, stop_reason and violations:
  incorrect release (final action release, expected keep_blocked or pending); incorrect keep-blocked or pending (final action keep_blocked or pending, not the
  expected one); no final action (none recorded: a loop, the token budget, the call cap, a stop, or a report with no business action), by stop reason; and,
  apart from these, episodes with an attempted prohibited action and the counts attempted and executed. On test, the frozen cost-weighted loss per 1,000
  invoices (scripts/gpu/final_analysis.py business_metrics: weights wrong payment 10, wrongful hold 3, keep-blocked/pending confusion 2, one per business day late).
Stress paired comparisons: every pair of the seven stress models, by the frozen metrics.paired_bootstrap (second minus first), with the discordant counts and the
  exact two-sided McNemar p-value (descriptive; not pre-registered).
RFT on the reserve (2026-11-11; exploratory, post hoc): RFT minus SFT-v2 on test_reserve, both from the follow-up evaluation, by the same frozen
  paired bootstrap, with the discordant counts and the exact McNemar p-value. Success flags only. No reserve case content is read beyond them or kept.

    python scripts/posthoc/manuscript_counts.py --out runs/posthoc/manuscript_counts
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import posthoc_common as c   # noqa: E402
import grader_sensitivity as gs   # noqa: E402

V6_AND_SHIFT = {"sft-v2": c.R2 / "sft-v2", "grpo-20261101": c.R2 / "grpo"}
STRESS = {m: c.E1 / m for m in ("sft-v2", "rft", "grpo-20261101", "grpo-20261107", "grpo-20261108")}
STRESS_ALL = {**STRESS, "control-a": c.CONTROLS / "sft-v2-reminder", "control-b": c.CONTROLS / "sft-extra-epoch"}
PRIMARY = ("wrong_decision", "cancelled_invoice_omission", "missing_evidence", "derived_figure", "other")
NETTING = ("net receipts", "available coverage")


def mismatched_entries(log) -> tuple[list[dict], list]:
    """The report's derived entries whose operands all resolve and whose expression evaluates but does not equal the stated value (g8's derived mismatch), and the
    observations delivered to the agent."""
    from ap_sim.grader import grader as g7
    observations = g7._delivered(log)
    report = log.get("report") or {}
    entries = [e for f in report.get("findings") or [] for e in f.get("evidence") or []] + list(report.get("evidence") or [])
    out = []
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
            continue
        try:
            if g7.evaluate_expression(e["expression"], values) == Decimal(e["value"]):
                continue
        except (ValueError, KeyError, InvalidOperation):
            continue
        out.append(e)
    return out, observations


def omits_shown_reversal(entry: dict, reversals: set[str], shown: set[str]) -> bool:
    """The entry's operands leave out at least one reversal receipt the agent was shown."""
    return bool((reversals & shown) - {o["record_id"] for o in (entry.get("operands") or {}).values()})


def stated_values(scenario, log, verdict, names) -> set[Decimal]:
    """The grader's values of the named quantities for this invoice and PO line, at the start and at the deciding action."""
    from ap_sim.grader import grader as g7
    snaps = [gs.initial_snapshot(scenario)]
    key = {"release": "release", "keep_blocked": "keep_blocked_shortfall", "pending": "keep_blocked_pending"}.get(verdict.reached)
    op = g7._resolution_op(log, key) if key else None
    if op is not None:
        snaps.append(op["before"])
    po_line = next(l["po_line"] for l in snaps[0]["invoice_line"] if l["invoice"] == scenario.current_invoice)
    profile = getattr(scenario, "policy", "sandbox-v1")
    return {q[0] for s in snaps for n, q in gs.quantities(s, po_line, scenario.current_invoice, profile).items() if n in names}


def graded(the_set: str, path: pathlib.Path, with_record: bool = False):
    """(case, scenario, log, verdict[, stored record]) for each stored episode, the replay checked against the record."""
    from ap_sim.grading import g8
    factories = c.cases(the_set)
    for case, rec in c.episodes(path / f"{the_set}-greedy" / "episodes.jsonl").items():
        scenario, log, _ = c.replay(factories[case], case, rec)
        v = g8.grade(scenario, log)
        if (v.reward == 1) != bool(rec["success"]) or sorted(v.critical) != sorted(rec["critical"]):
            raise SystemExit(f"{path.name} {the_set}: a replay disagrees with the stored verdict")
        yield (case, scenario, log, v, rec) if with_record else (case, scenario, log, v)


def v6_row(path: pathlib.Path) -> dict:
    row = {"cases": 0, "success": 0, "right_action": 0, "failures": 0, "omits_reversal_value_nets_right_action": 0,
           "same_citation_failure_only": 0, "same_any_of_four_quantities": 0}
    for _, scenario, log, v in graded("v6", path):
        row["cases"] += 1
        right = v.reached == v.expected
        row["right_action"] += right
        if v.reward == 1:
            row["success"] += 1
            continue
        row["failures"] += 1
        if not right:
            continue
        bad, observations = mismatched_entries(log)
        shown = {rec["id"] for _, o in observations for region in o["regions"] for rec in region["records"]}
        reversals = {r.id for r in scenario.records if getattr(r, "movement", None) == "reversal"}
        omit = [e for e in bad if omits_shown_reversal(e, reversals, shown)]
        if not omit:
            continue
        nets = stated_values(scenario, log, v, NETTING)
        four = stated_values(scenario, log, v, gs.QUANTITIES)
        hit = any(Decimal(e["value"]) in nets for e in omit)
        row["omits_reversal_value_nets_right_action"] += hit
        row["same_citation_failure_only"] += hit and list(v.critical) == ["citation_failure"]
        row["same_any_of_four_quantities"] += any(Decimal(e["value"]) in four for e in omit)
    return row


def shift_row(path: pathlib.Path) -> dict:
    row = {"v2_cases": 0, "v2_success": 0, "v2_failures": 0, "v2_fail_discrepancy_type_right_action": 0, "v2_fail_no_final_action": 0}
    for case, _, _, v in graded("policy_shift", path):
        if not case.endswith("-v2"):
            continue
        row["v2_cases"] += 1
        if v.reward == 1:
            row["v2_success"] += 1
            continue
        row["v2_failures"] += 1
        row["v2_fail_discrepancy_type_right_action"] += "discrepancy_type_wrong" in v.critical and v.reached == v.expected
        row["v2_fail_no_final_action"] += v.reached is None
    return row


def stress_row(path: pathlib.Path) -> dict:
    row = {"cases": 0, "failures": 0, "missing_cancelled_citation": 0, "wrong_decision": 0, "derived_mismatch": 0}
    for _, scenario, log, v in graded("stress", path):
        row["cases"] += 1
        if v.reward == 1:
            continue
        r = c.g8_reasons(scenario, log)
        row["failures"] += 1
        row["missing_cancelled_citation"] += any("cancel" in x for x in r["required_unsatisfied"])
        row["wrong_decision"] += v.reached != v.expected
        row["derived_mismatch"] += any(p.startswith(gs.MISMATCH) for p in r["citation_problems"])
    return row


def labels(v, reasons: dict) -> set[str]:
    """Every failure label that applies (the module docstring's definitions)."""
    problems = reasons["citation_problems"]
    got = {"wrong_decision": v.reached != v.expected,
           "missing_evidence": "required_evidence_missing" in v.critical,
           "cancelled_invoice_omission": any("cancel" in x for x in reasons["required_unsatisfied"]),
           "derived_figure": any(p.startswith(gs.MISMATCH) for p in problems),
           "other_citation_problem": any(not p.startswith(gs.MISMATCH) for p in problems),
           "discrepancy_type_wrong": "discrepancy_type_wrong" in v.critical}
    return {k for k, on in got.items() if on} or {"no_flag"}


def primary(got: set[str]) -> str:
    """One category per failure, in PRIMARY's order."""
    return next((k for k in PRIMARY[:-1] if k in got), "other")


def stress_breakdown(path: pathlib.Path) -> dict:
    import collections
    import hashlib
    source = path / "stress-greedy" / "episodes.jsonl"
    row = {"source": str(source.relative_to(c.ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": 0, "success": 0,
           "decision_correct": 0, "report_valid": 0, "failures": 0, "labels": collections.Counter(), "primary": collections.Counter(),
           "combinations": collections.Counter(), "by_case": {}}
    for case, scenario, log, v, rec in graded("stress", path, with_record=True):
        row["cases"] += 1
        row["decision_correct"] += bool(rec["outcome_correct"])
        row["report_valid"] += bool(rec["report_valid"])
        if (v.reached == v.expected) != bool(rec["outcome_correct"]):
            raise SystemExit(f"{path.name}: the replayed decision disagrees with the stored outcome_correct")
        if v.reward == 1:
            row["success"] += 1
            continue
        got = labels(v, c.g8_reasons(scenario, log))
        row["failures"] += 1
        row["labels"].update(got)
        row["primary"][primary(got)] += 1
        row["combinations"]["+".join(sorted(got))] += 1
        row["by_case"][case] = {"labels": sorted(got), "primary": primary(got)}
    return {k: dict(sorted(x.items())) if isinstance(x, collections.Counter) else x for k, x in row.items()}


def mcnemar_exact(gained: int, lost: int) -> float:
    import math
    n, k = gained + lost, min(gained, lost)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def stress_pairs() -> dict:
    """Every pair of the seven stress models: second minus first, by the frozen paired bootstrap."""
    import itertools
    from ap_sim.evaluation import metrics
    success = {m: {k: float(bool(e["success"])) for k, e in c.episodes(p / "stress-greedy" / "episodes.jsonl").items()} for m, p in STRESS_ALL.items()}
    out = {}
    for a, b in itertools.combinations(STRESS_ALL, 2):
        if success[a].keys() != success[b].keys():
            raise SystemExit(f"{a} and {b} have different stress cases")
        r = metrics.paired_bootstrap(success[a], success[b])
        gained = sum(success[b][k] > success[a][k] for k in success[a])
        lost = sum(success[b][k] < success[a][k] for k in success[a])
        out[f"{b} minus {a}"] = {"difference": r["difference"], "low": r["low"], "high": r["high"], "gained": gained, "lost": lost,
                                 "excludes_zero": r["low"] > 0 or r["high"] < 0, "mcnemar_exact_p": round(mcnemar_exact(gained, lost), 4)}
    return out


def reserve_rft() -> dict:
    """RFT minus SFT-v2 on the reserve, both from the follow-up evaluation (the baseline every reserve comparison uses)."""
    from ap_sim.evaluation import metrics
    a, b = ({k: float(bool(e["success"])) for k, e in c.episodes(c.E1 / m / "test_reserve-greedy" / "episodes.jsonl").items()} for m in ("sft-v2", "rft"))
    if a.keys() != b.keys():
        raise SystemExit("RFT and SFT-v2 have different reserve cases")
    r = metrics.paired_bootstrap(a, b)
    gained, lost = sum(b[k] > a[k] for k in a), sum(b[k] < a[k] for k in a)
    return {"cases": r["cases"], "sft_v2": int(sum(a.values())), "rft": int(sum(b.values())), "difference": r["difference"], "low": r["low"],
            "high": r["high"], "gained": gained, "lost": lost, "excludes_zero": r["low"] > 0 or r["high"] < 0,
            "mcnemar_exact_p": round(mcnemar_exact(gained, lost), 4)}


def decision_counts() -> dict:
    """final_action == expected against outcome_correct, in every greedy cell; the base model's differing episodes by flag."""
    import collections
    out = {"cells_where_they_differ": {}, "base_differences": {}}
    for root in (c.R2, c.E1, c.CONTROLS):
        for f in sorted(root.glob("*/*-greedy/episodes.jsonl")):
            if not (f.parent / ".done").exists():
                continue
            eps = list(c.episodes(f).values())
            right = sum(e["final_action"] == e["expected"] for e in eps)
            correct = sum(bool(e["outcome_correct"]) for e in eps)
            if right != correct:
                name = str(f.parent.relative_to(c.ROOT))
                out["cells_where_they_differ"][name] = {"final_action_equals_expected": right, "outcome_correct": correct}
                out["base_differences"][name] = dict(collections.Counter("+".join(sorted(e["critical"])) for e in eps
                                                                         if (e["final_action"] == e["expected"]) != bool(e["outcome_correct"])))
    return out


def forgetting_seed_20261108() -> dict:
    import collections
    from ap_sim import forgetting
    from ap_sim.grading import g8
    from ap_sim.grader import grader as g7
    factories = forgetting.load()
    row = {"cases": 0, "failures": 0, "failures_right_decision": 0, "flags": collections.Counter(), "unfound_citations": collections.Counter()}
    for case, rec in c.episodes(c.E1 / "grpo-20261108" / "forgetting_l1-greedy" / "episodes.jsonl").items():
        row["cases"] += 1
        scenario, log, _ = c.replay(factories[case], case, rec)
        v = g8.grade(scenario, log)
        if (v.reward == 1) != bool(rec["success"]) or sorted(v.critical) != sorted(rec["critical"]):
            raise SystemExit("forgetting: a replay disagrees with the stored verdict")
        if v.reward == 1:
            continue
        row["failures"] += 1
        row["failures_right_decision"] += v.reached == v.expected
        row["flags"]["+".join(sorted(v.critical))] += 1
        observations = g7._delivered(log)
        order = {oid: i for i, (oid, _) in enumerate(observations)}
        report = log.get("report") or {}
        for item in [x for f in report.get("findings") or [] for x in f.get("evidence") or []] + list(report.get("evidence") or []):
            if item.get("form") != "observed":
                continue
            found, _ = g7._find(observations, item["observation_id"], item["record_id"], item["record_version"], item["field"])
            if found:
                continue
            shown_in = [oid for oid, o in observations for region in o["regions"] for r in region["records"]
                        if r["id"] == item["record_id"] and r.get("version") == item["record_version"]]
            later = any(order[oid] > order.get(item["observation_id"], -1) for oid in shown_in)
            who = "current invoice" if item["record_id"].startswith(scenario.current_invoice) else "other record"
            row["unfound_citations"][f"{who}, field {item['field']}, " + ("version shown in a later observation" if later else "version not shown later")] += 1
    return {k: dict(v) if isinstance(v, collections.Counter) else v for k, v in row.items()}


DECISION_CELLS = {
    "test": {"base": c.R2 / "base", "sft-v2": c.R2 / "sft-v2", "rft": c.R2 / "rft", "grpo-20261101": c.R2 / "grpo", "grpo-20261107": c.E1 / "grpo-20261107",
             "grpo-20261108": c.E1 / "grpo-20261108", "control-a": c.CONTROLS / "sft-v2-reminder", "control-b": c.CONTROLS / "sft-extra-epoch",
             "sft-v2-anchor": c.CONTROLS / "sft-v2"},
    "stress": {m: p for m, p in STRESS_ALL.items()},
}


def decision_row(records: list[dict]) -> dict:
    import collections
    holds = ("keep_blocked", "pending")
    none = [r for r in records if r["final_action"] not in ("release",) + holds]
    return {"cases": len(records),
            "right_final_action": sum(r["final_action"] == r["expected"] for r in records),
            "incorrect_release": sum(r["final_action"] == "release" and r["expected"] != "release" for r in records),
            "incorrect_keep_blocked_or_pending": sum(r["final_action"] in holds and r["final_action"] != r["expected"] for r in records),
            "no_final_action": len(none),
            "no_final_action_by_stop": dict(sorted(collections.Counter(r["stop_reason"] for r in none).items())),
            "no_final_action_expected": dict(sorted(collections.Counter(r["expected"] for r in none).items())),
            "episodes_with_attempted_prohibited_action": sum(r["violations"]["attempted"] > 0 for r in records),
            "prohibited_actions_attempted": sum(r["violations"]["attempted"] for r in records),
            "prohibited_actions_executed": sum(r["violations"]["executed"] for r in records)}


def all_greedy_cells() -> dict:
    """Every finished greedy cell of every model, set and form: episodes, incorrect releases, and executed prohibited actions."""
    out = {"cells": 0, "episodes": 0, "incorrect_release": 0, "prohibited_actions_executed": 0}
    for root in (c.R2, c.E1, c.CONTROLS):
        for f in sorted(root.glob("*/*-greedy/episodes.jsonl")):
            if not (f.parent / ".done").exists():
                continue
            records = list(c.episodes(f).values())
            out["cells"] += 1
            out["episodes"] += len(records)
            out["incorrect_release"] += sum(r["final_action"] == "release" and r["expected"] != "release" for r in records)
            out["prohibited_actions_executed"] += sum(r["violations"]["executed"] for r in records)
    return out


def decision_errors() -> dict:
    import final_analysis
    out = {}
    for the_set, models in DECISION_CELLS.items():
        out[the_set] = {}
        for m, path in models.items():
            records = list(c.episodes(path / f"{the_set}-greedy" / "episodes.jsonl").values())
            row = decision_row(records)
            if row["right_final_action"] + row["incorrect_release"] + row["incorrect_keep_blocked_or_pending"] + row["no_final_action"] != row["cases"]:
                raise SystemExit(f"{m} {the_set}: the decision categories do not partition the episodes")
            if the_set == "test":
                row["d5a_loss_per_1000"] = final_analysis.business_metrics(records)["loss_per_1000"]
            out[the_set][m] = row
    return out


def run() -> dict:
    return {"label": "counts the manuscript corrections cite; g8 unchanged (docs/V2.md, 2026-11-10, Manuscript corrections)",
            "v6": {m: v6_row(p) for m, p in V6_AND_SHIFT.items()},
            "policy_shift": {m: shift_row(p) for m, p in V6_AND_SHIFT.items()},
            "stress": {m: stress_row(p) for m, p in STRESS.items()},
            "stress_breakdown": {m: stress_breakdown(p) for m, p in STRESS_ALL.items()},
            "stress_pairs": stress_pairs(),
            "reserve_rft": reserve_rft(),
            "decision_counts": decision_counts(),
            "forgetting_seed_20261108": forgetting_seed_20261108(),
            "decision_errors": decision_errors(),
            "all_greedy_cells": all_greedy_cells()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if (out / "results.json").exists():
        raise SystemExit(f"{out / 'results.json'} exists: write a new directory, never over a stored result")
    res = run()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    shown = {k: v for k, v in res.items() if k != "stress_breakdown"}
    shown["stress_breakdown"] = {m: {k: x for k, x in r.items() if k != "by_case"} for m, r in res["stress_breakdown"].items()}   # no case ids printed
    print(json.dumps(shown, indent=1))


if __name__ == "__main__":
    main()
