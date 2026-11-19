"""The conditional Claude reference on the stress set, specified on 2026-11-09 before any stress result was read.

If the best model's greedy success on the stress set is below 95%, the API cap goes from $30 to $65 and the Phase 10 Claude row runs on all
100 stress cases with exactly its pinned model, prompt, tools and settings, reported descriptively. It changes no pre-registered verdict.

This wrapper changes nothing in the frozen row (scripts/phase5/claude_row.py): it calls the row's own `run_row`,
`make_client` and pins. Only three things differ from the test row, each set here and recorded in the summary:
  - the cases: the frozen stress set, all 100, in manifest order (only as frozen: final_evaluation.check_stress_freeze);
  - the cap: $65 (the module's API_CAP, which the row's Ledger reads; the frozen file is not edited);
  - the spending before the row: the API total so far, read from the test row's own summary.json (spent_total), not the frozen
    script's SPENT_BEFORE constant, which predates the test row.
The model must be the pinned one. If it is unavailable, the row stops on the first request and reports it (no substitute).

    python scripts/phase5/claude_row_stress.py --rule-recorded --best-stress-success 0.NN --out runs/followup/claude-stress
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu"), str(ROOT / "scripts/phase5")]

import claude_row as cr   # noqa: E402

STRESS_CAP = 65.0
THRESHOLD = 0.95
TEST_ROW_SUMMARY = ROOT / "runs/session3-phase10/claude-row/summary.json"


def spent_so_far(summary: pathlib.Path | None = None) -> float:
    """All v2 API spending to date: the test row's own record of the total after it ran (its spending log's sum), refused if absent."""
    summary = summary or TEST_ROW_SUMMARY             # read at call time
    if not summary.exists():
        raise SystemExit(f"{summary} is absent: the API spending so far is unknown, so the row cannot be capped")
    total = json.loads(summary.read_text())["spent_total"]
    if not (isinstance(total, (int, float)) and 0 < total < STRESS_CAP):
        raise SystemExit(f"the recorded API total {total!r} is not a figure below the ${STRESS_CAP} cap")
    return float(total)


def stress_cases() -> dict:
    import final_evaluation
    from ap_sim.stress import messiness
    final_evaluation.check_stress_freeze()
    cases = messiness.load(messiness.MANIFEST, require_frozen=True)
    if len(cases) != 100:
        raise SystemExit(f"{len(cases)} stress cases, not 100")
    return cases


def main(argv: list[str] | None = None, client_factory=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rule-recorded", action="store_true", help="required: the user's rule is in docs/V2.md (2026-11-09)")
    ap.add_argument("--best-stress-success", type=float, required=True,
                    help="the best model's greedy success on the stress set, from the follow-up analysis (stress.success)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not a.rule_recorded:
        raise SystemExit("the stress row runs only under the rule recorded in docs/V2.md (--rule-recorded)")
    if not 0.0 <= a.best_stress_success <= 1.0:
        raise SystemExit(f"--best-stress-success {a.best_stress_success} is not a rate")
    if a.best_stress_success >= THRESHOLD:
        raise SystemExit(f"the best model's stress success {a.best_stress_success} is at least {THRESHOLD}: by the rule the row does not run")
    model = cr.PINNED["model"]
    cr.prices(model)
    spent_before = spent_so_far()
    cases = stress_cases()
    cr.API_CAP = STRESS_CAP                       # the raised cap for this row; the row's Ledger reads it (the frozen file is unchanged)
    client, anthropic = (client_factory or cr.make_client)()
    out = pathlib.Path(a.out)
    summary = cr.run_row(cases, client, anthropic, out, model, spent_before=spent_before)
    summary["stress_rule"] = {"recorded": "docs/V2.md, 2026-11-09, 'A conditional Claude reference on the stress set'",
                              "best_stress_success": a.best_stress_success, "threshold": THRESHOLD, "cap": STRESS_CAP,
                              "spent_before": spent_before, "pinned": cr.PINNED, "cases": "stress, all 100, manifest order"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
