"""Build the release's per-case outcome file (data/outcomes.csv) from the stored evaluation records.

One row per stored greedy episode of every evaluation run the paper reports (39 runs, 4,520 episodes), plus Claude Sonnet 5.5's two runs (200
episodes) and the scripted heuristic (444 rows, right final action only). Inkling 2 Small is not included. The set-aside first
evaluation run (run-1, deviation D59, paper Supplement Table S18) is never read.

Every value comes from an existing record or from the post-hoc code that produced the paper's numbers:
  full_success, right_final_action, grader_flags, attempted_unsafe   the stored record (success; final_action == expected; critical; violations.attempted)
  paper_labels          scripts/posthoc/manuscript_counts.labels on the episode replayed through the frozen environment and g8 (failures only)
  known_false_pass      runs/posthoc/review_findings/false_passes.json (the 34 audited cells; empty elsewhere)
  variant_success       the rule of scripts/posthoc/grader_sensitivity.run, per case (V6 and stress cells it covered; empty elsewhere)
  sampled_successes_of_8  the primary evaluation's test-samples-8 cells (SFT-v2, RFT, GRPO seed 1)
  heuristic rows        runs/posthoc/review_findings/item4.json, cases.*.right
Each replay must reproduce the stored verdict and flags, and every per-cell total is checked against its stored summary or analysis output before
the file is written. Counts only are printed: no case content, and no reserve or stress case id.

The stored records under runs/ are not part of this release, so the script does not run from the release alone. It ran
from release/minimal/ in the development repository, which is why ROOT is two directories up:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice .venv/bin/python release/minimal/build_outcomes.py --out DIR
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/posthoc")]
import posthoc_common as c          # noqa: E402  (also puts src/, scripts/ and scripts/gpu on the path)
import grader_sensitivity as gs     # noqa: E402
import manuscript_counts as mc      # noqa: E402

R2, E1, CT = c.R2, c.E1, c.CONTROLS
COLUMNS = ["model", "evaluation", "set", "case_id", "full_success", "right_final_action", "grader_flags", "paper_labels",
           "known_false_pass", "variant_success", "attempted_unsafe", "sampled_successes_of_8"]

# (model as the paper names it, evaluation, cell directory, false-pass audit key prefix, sets)
RUNS = [
    ("Base", "primary", R2 / "base", "phase10/base", ["test", "v6", "policy_shift", "forgetting_l1"]),
    ("SFT-v2", "primary", R2 / "sft-v2", "phase10/sft-v2", ["test", "v6", "policy_shift", "forgetting_l1"]),
    ("RFT", "primary", R2 / "rft", "phase10/rft", ["test"]),
    ("GRPO seed 1", "primary", R2 / "grpo", "phase10/grpo", ["test", "v6", "policy_shift", "forgetting_l1"]),
    ("GRPO seed 1, merged in bf16", "cross-evaluation", R2 / "grpo-merged", "phase10/grpo-merged", ["test"]),
    ("RFT, merged in bf16", "cross-evaluation", R2 / "rft-merged", "phase10/rft-merged", ["test"]),
    ("SFT-v2 adapter on the base model", "cross-evaluation", R2 / "sft-adapter", "phase10/sft-adapter", ["test"]),
    ("SFT-v2", "follow-up", E1 / "sft-v2", "followup/sft-v2", ["test_reserve", "stress"]),
    ("RFT", "follow-up", E1 / "rft", "followup/rft", ["test_reserve", "stress"]),
    ("GRPO seed 1", "follow-up", E1 / "grpo-20261101", "followup/grpo-20261101", ["test_reserve", "stress"]),
    ("GRPO seed 2", "follow-up", E1 / "grpo-20261107", "followup/grpo-20261107", ["test", "test_reserve", "stress", "v6", "policy_shift", "forgetting_l1"]),
    ("GRPO seed 3", "follow-up", E1 / "grpo-20261108", "followup/grpo-20261108", ["test", "test_reserve", "stress", "v6", "policy_shift", "forgetting_l1"]),
    ("SFT-v2", "controls", CT / "sft-v2", "controls/sft-v2", ["test"]),
    ("SFT-v2 + prompt reminder", "controls", CT / "sft-v2-reminder", "controls/sft-v2-reminder", ["test", "stress"]),
    ("Second SFT run", "controls", CT / "sft-extra-epoch", "controls/sft-extra-epoch", ["test", "stress"]),
]
# grader_sensitivity's cells (its CELLS plus the controls' stress cells), by cell directory
VARIANT = {("v6", p) for p in gs.CELLS["v6"].values()} | {("stress", p) for p in gs.CELLS["stress"].values()} | {("stress", CT / "sft-v2-reminder"), ("stress", CT / "sft-extra-epoch")}
VARIANT_STORED = {("v6", R2 / "base"): ("grader_sensitivity", "v6", "base"), ("v6", R2 / "sft-v2"): ("grader_sensitivity", "v6", "sft-v2"),
                  ("v6", R2 / "grpo"): ("grader_sensitivity", "v6", "grpo-20261101"), ("v6", E1 / "grpo-20261107"): ("grader_sensitivity", "v6", "grpo-20261107"),
                  ("v6", E1 / "grpo-20261108"): ("grader_sensitivity", "v6", "grpo-20261108"),
                  ("stress", E1 / "sft-v2"): ("grader_sensitivity", "stress", "sft-v2"), ("stress", E1 / "rft"): ("grader_sensitivity", "stress", "rft"),
                  ("stress", E1 / "grpo-20261101"): ("grader_sensitivity", "stress", "grpo-20261101"),
                  ("stress", E1 / "grpo-20261107"): ("grader_sensitivity", "stress", "grpo-20261107"),
                  ("stress", E1 / "grpo-20261108"): ("grader_sensitivity", "stress", "grpo-20261108"),
                  ("stress", CT / "sft-v2-reminder"): ("grader_sensitivity_controls", "stress", "A: sft-v2 + reminder"),
                  ("stress", CT / "sft-extra-epoch"): ("grader_sensitivity_controls", "stress", "B: sft + one epoch")}
SAMPLED = {R2 / "sft-v2", R2 / "rft", R2 / "grpo"}
CLAUDE = [("test", ROOT / "runs/session3-phase10/claude-row"), ("stress", ROOT / "runs/followup/claude-stress")]


def J(rel: str):
    return json.loads((ROOT / rel).read_text())


def b(x) -> str:
    return "" if x is None else ("true" if x else "false")


def stop(msg: str):
    raise SystemExit("build_outcomes: " + msg)


def variant_ok(scenario, log, v) -> bool:
    """grader_sensitivity.run's rule for a failure the variant accepts (copied condition, checked against its stored totals)."""
    problems = v.diagnostics.get("citation_problems") or []
    mismatches = [p for p in problems if p.startswith(gs.MISMATCH)]
    got = gs.classify(scenario, log, v)["entries"]
    if len(got) != len(mismatches):
        stop("a classified entry count differs from g8's list")
    citation_only = list(v.critical) == ["citation_failure"] and v.reached == v.expected and v.diagnostics.get("finish_accepted")
    return bool(citation_only and mismatches and len(problems) == len(mismatches) and all(e["class"] == "a" for e in got))


def main() -> None:
    from ap_sim.grading import g8
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if any("run-1" in str(p) for _, _, p, _, _ in RUNS):
        stop("run-1 is never read")
    fp = J("runs/posthoc/review_findings/false_passes.json")["cells"]
    rows, checks = [], []
    factories = {}

    def fac(the_set):
        if the_set not in factories:
            if the_set == "forgetting_l1":
                from ap_sim import forgetting
                factories[the_set] = forgetting.load()
            else:
                factories[the_set] = c.cases(the_set)
        return factories[the_set]

    sampled = {}
    for p in SAMPLED:
        k = collections.Counter()
        n = collections.Counter()
        for line in (p / "test-samples-8" / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            n[e["case"]] += 1
            k[e["case"]] += bool(e["success"])
        if set(n.values()) != {8} or len(n) != 200:
            stop(f"{p.name} samples are not 8 per case on 200 cases")
        sampled[p] = k

    for model, evaluation, path, fpkey, sets in RUNS:
        for the_set in sets:
            cell = path / f"{the_set}-greedy"
            if not (cell / ".done").exists() or (cell / ".invalid").exists():
                stop(f"{model} {the_set}: the cell is not a finished valid cell")
            eps = c.episodes(cell / "episodes.jsonl")
            audit = fp.get(f"{fpkey}/{'forgetting_l1' if the_set == 'forgetting_l1' else the_set}")
            if the_set != "test_reserve" and audit is None:
                stop(f"{model} {the_set}: no false-pass audit entry")
            false_cases = set(audit.get("false_pass_cases") or []) if audit else set()
            do_variant = (the_set, path) in VARIANT
            vcount = 0
            for case, rec in eps.items():
                success = bool(rec["success"])
                labels, variant = "", (True if success else None)
                if not success:
                    scenario, log, _ = c.replay(fac(the_set)[case], case, rec)
                    v = g8.grade(scenario, log)
                    if (v.reward == 1) != success or sorted(v.critical) != sorted(rec["critical"]):
                        stop(f"{model} {the_set}: a replay disagrees with the stored verdict")
                    if (v.reached == v.expected) != (rec["final_action"] == rec["expected"]):
                        stop(f"{model} {the_set}: the replayed decision disagrees with the record")
                    labels = ";".join(sorted(mc.labels(v, c.g8_reasons(scenario, log))))
                    if do_variant:
                        variant = variant_ok(scenario, log, v)
                vcount += bool(variant)
                if success and rec["critical"]:
                    stop(f"{model} {the_set}: a success with critical flags")
                rows.append({"model": model, "evaluation": evaluation, "set": the_set, "case_id": case, "full_success": b(success),
                             "right_final_action": b(rec["final_action"] == rec["expected"]), "grader_flags": ";".join(sorted(rec["critical"])),
                             "paper_labels": labels, "known_false_pass": b(case in false_cases) if audit else "",
                             "variant_success": b(variant) if do_variant else "", "attempted_unsafe": str(int(rec["violations"]["attempted"])),
                             "sampled_successes_of_8": str(sampled[path][case]) if (path in SAMPLED and the_set == "test") else ""})
            summ = json.loads((cell / "summary.json").read_text())
            got = sum(r["full_success"] == "true" for r in rows[-len(eps):])
            checks.append((f"{model} [{evaluation}] {the_set} success vs summary.json", got, summ["success"]["successes"]))
            if audit:
                checks.append((f"{model} [{evaluation}] {the_set} episodes vs audit", len(eps), audit["episodes"]))
                checks.append((f"{model} [{evaluation}] {the_set} successes vs audit", got, audit["successes"]))
            if do_variant:
                src, s, m = VARIANT_STORED[(the_set, path)]
                checks.append((f"{model} [{evaluation}] {the_set} variant success vs {src}", vcount, J(f"runs/posthoc/{src}/results.json")["sets"][s][m]["success_variant"]))

    from ap_sim.training.environment import APReceiptEnvV2
    from ap_sim.training.loop_guard import LOOP_GUARD
    for the_set, d in CLAUDE:
        files = sorted(p for p in d.glob(f"{the_set}-*.json"))
        n_ok = 0
        for p in files:
            r = json.loads(p.read_text())
            rec, case = r["record"], r["record"]["case"]
            if case != p.stem:
                stop("a Claude file name differs from its case")
            success = bool(rec["success"])
            labels = ""
            if not success:
                env = APReceiptEnvV2({case: fac(the_set)[case]}, execution_rule=LOOP_GUARD)
                env.reset(scenario=case, purpose="evaluation")
                scenario = env._episode()[0]
                log = r["episode_log"]
                v = g8.grade(scenario, log)
                if (v.reward == 1) != success or sorted(v.critical) != sorted(rec["critical"]):
                    stop(f"Claude {the_set}: a regrade disagrees with the stored verdict")
                labels = ";".join(sorted(mc.labels(v, c.g8_reasons(scenario, log))))
            n_ok += success
            rows.append({"model": "Claude Sonnet 5.5", "evaluation": "reference", "set": the_set, "case_id": case, "full_success": b(success),
                         "right_final_action": b(rec["final_action"] == rec["expected"]), "grader_flags": ";".join(sorted(rec["critical"])),
                         "paper_labels": labels, "known_false_pass": "", "variant_success": "", "attempted_unsafe": str(int(rec["violations"]["attempted"])),
                         "sampled_successes_of_8": ""})
        checks.append((f"Claude {the_set} episodes", len(files), 100))
        want = J("runs/followup/claude-stress/summary.json")["success"] if the_set == "stress" else 99
        checks.append((f"Claude {the_set} successes", n_ok, want))

    i4 = J("runs/posthoc/review_findings/item4.json")
    for the_set in ("test", "v6", "policy_shift", "stress"):
        for case, x in sorted(i4[the_set]["cases"].items(), key=lambda kv: kv[0]):
            rows.append({"model": "Scripted heuristic", "evaluation": "post hoc", "set": the_set, "case_id": case, "full_success": "",
                         "right_final_action": b(x["right"]), "grader_flags": "", "paper_labels": "", "known_false_pass": "", "variant_success": "",
                         "attempted_unsafe": "", "sampled_successes_of_8": ""})
        checks.append((f"heuristic {the_set} right final action vs item4", sum(x["right"] for x in i4[the_set]["cases"].values()),
                       i4[the_set]["by_family"]["all"]["right_final_action"]))
        checks.append((f"heuristic {the_set} cases vs item4", len(i4[the_set]["cases"]), i4[the_set]["by_family"]["all"]["cases"]))

    keys = [(r["model"], r["evaluation"], r["set"], r["case_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        stop("a (model, evaluation, set, case_id) key repeats")
    total_fp = sum(r["known_false_pass"] == "true" for r in rows)
    checks.append(("known false passes, all audited cells", total_fp, J("runs/posthoc/review_findings/false_passes.json")["totals"]["false_passes"]))
    bad = [(n, g, w) for n, g, w in checks if g != w]
    for n, g, w in checks:
        print(("ok   " if g == w else "FAIL ") + f"{n}: {g} (record {w})")
    if bad:
        stop(f"{len(bad)} checks failed; nothing written")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "outcomes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"rows {len(rows)}, by evaluation {dict(collections.Counter(r['evaluation'] for r in rows))}")
    print("outcomes.csv sha256", hashlib.sha256((out / "outcomes.csv").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
