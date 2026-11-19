"""The follow-up study's analysis by its pre-registered judging rules and their amendments, from saved greedy episodes.
Written on 2026-11-09, before any follow-up cell had run. It reads records; nothing runs a model, and nothing feeds back into what is evaluated.
All of it is exploratory.

    Statistic     For each GRPO seed and set: the paired difference against SFT-v2 in greedy success, paired by case (metrics.paired_bootstrap: 10,000 resamples, 95% percentile,
                  seed 20261101).
    Test          Replicated if seeds 20261107 and 20261108 each have an interval above zero. Seed 20261101's test result is Phase 10's (its run 2 cells), and SFT-v2's test
                  episodes for the new seeds' pairs are Phase 10's sft-v2/test-greedy.
    Reserve       Replicated if all three seeds each have an interval above zero.
    Otherwise     On that set, "not replicated at this seed count"; the original result is not retracted, and is reported as a single-seed finding. Not a refutation.
    Stress set    Descriptive, no verdict: each model's success and each seed's difference with its interval, with the mean and range across seeds; "the same figures"
                  (success, and each seed's difference with its interval) by parameter format, by complication x format and by combination, and each model's per-combination
                  success split by case type (amendments of 2026-11-06, before any model was evaluated on the stress set). The format contrast exists only within M1, M4 and M1+M4.
    Slices        V6, policy_shift and forgetting L1 for the new seeds against Phase 10's SFT-v2 cells: descriptive, no verdict, with the mean and range across seeds.
    Cut cells     Named as cut, with cuts.json's reason; a verdict that needs a cut cell is "not evaluated (cut)". Cells reinstated by the stop rule as amended on
                  2026-11-09 (during the session, before any result was read) are reported with reinstated.json's record.
Additions of 2026-11-09, made before any result was read: the per-group differences, the per-combination split, the stress and slice means and ranges across
seeds, and the reinstatement record.

    python scripts/gpu/followup_analysis.py --run FOLLOWUP_RUN --phase10 PHASE10_RUN --out results.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu")]

import final_analysis   # noqa: E402

SEEDS = (20261101, 20261107, 20261108)
NEW = (20261107, 20261108)
SLICES = ("v6", "policy_shift", "forgetting_l1")
TYPES = ("V1", "V2", "V3a", "V3b", "V4", "V5")
DIFFERENCES_BY = ("format", "complication_x_format", "combination")   # by combination x type, success only (a pre-registered amendment)
NOT_REPLICATED = "not replicated at this seed count"
CUT = "not evaluated (cut)"
NOT_RUN = "not evaluated (reinstated, not run)"
# Phase 10's run 2, whose test cells give seed 20261101's test result and SFT-v2's test and slice episodes (the label is checked, not assumed)
PHASE10_RUN_SHA256 = "4c9db5a2a8e54df58f5efe23e220eec3c90fbed3ea53e3104c4497952d6425d5"


def file_sha256(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def episodes(run_dir: pathlib.Path, name: str) -> list[dict] | None:
    """A finished cell's records, or None if it has no .done (cut, or never run)."""
    d = run_dir / name
    return final_analysis.read_episodes(d) if (d / ".done").exists() else None


def success(records: list[dict]) -> dict:
    by = final_analysis.by_case(records)
    return {"cases": len(by), "successes": sum(bool(r["success"]) for r in by.values()), "rate": round(sum(bool(r["success"]) for r in by.values()) / len(by), 4) if by else None}


def paired(sft: list[dict], grpo: list[dict]) -> dict:
    """GRPO minus SFT-v2, by case: final_analysis.same_cases_paired refuses different case sets."""
    return final_analysis.same_cases_paired(sft, grpo)


def verdict(results: dict[int, dict | None], seeds: tuple[int, ...]) -> str:
    if any(results.get(s) is None for s in seeds):
        return CUT
    return "replicated" if all(results[s]["low"] > 0 for s in seeds) else NOT_REPLICATED


def across(results: dict[int, dict | None]) -> dict:
    d = [r["difference"] for r in results.values() if r]
    return {"mean": round(statistics.fmean(d), 4), "min": min(d), "max": max(d), "seeds": len(d)} if d else {}


def subset(records: list[dict], ids: set[str]) -> list[dict]:
    return [r for r in records if r["case"] in ids]


def stress_groups() -> dict:
    """Case ids by format, by complication x format, and by combination x type, from the frozen manifest (the loader's own format rule), refused unless it is the frozen one."""
    import final_evaluation
    from ap_sim.stress import messiness
    final_evaluation.check_stress_freeze()
    entries = json.loads(messiness.MANIFEST.read_text())["entries"]
    groups: dict[str, dict[str, set[str]]] = {"format": collections.defaultdict(set), "complication_x_format": collections.defaultdict(set),
                                              "combination": collections.defaultdict(set), "combination_x_type": collections.defaultdict(set)}
    for e in entries:
        fmt = "timed" if messiness.is_timed(e["params"]) else "frozen"
        groups["format"][fmt].add(e["case_id"])
        for m in e["complications"]:
            groups["complication_x_format"][f"{m} {fmt}"].add(e["case_id"])
        groups["combination"]["+".join(sorted(e["complications"]))].add(e["case_id"])
        groups["combination_x_type"][f"{'+'.join(sorted(e['complications']))} {e['group']}"].add(e["case_id"])
    return {k: dict(v) for k, v in groups.items()}


def stress_report(models: dict[str, list[dict] | None]) -> dict:
    groups = stress_groups()
    sft = models.get("sft-v2")
    out = {"note": "Descriptive, no verdict. The frozen-against-timed contrast exists only within M1, M4 and M1+M4; elsewhere format cannot be separated from complication "
                   "(red-team round 17, finding 1). Per-combination figures partly reflect the type mix, so they are also split by type (finding 2).",
           "success": {m: success(r) if r else CUT for m, r in models.items()},
           "difference_vs_sft_v2": {m: paired(sft, r) if r and sft else CUT for m, r in models.items() if m.startswith("grpo")}}
    seeds = {int(m.split("-")[1]): d for m, d in out["difference_vs_sft_v2"].items() if isinstance(d, dict)}
    out["across_seeds"] = across(seeds)
    for kind, by in groups.items():
        out[f"by_{kind}"] = {g: {m: success(subset(r, ids)) if r else CUT for m, r in models.items()} for g, ids in sorted(by.items())}
        if kind in DIFFERENCES_BY:      # "the same figures" by group: each seed's paired difference with its interval (a pre-registered amendment)
            out[f"difference_by_{kind}"] = {g: {m: paired(subset(sft, ids), subset(r, ids)) if r and sft else CUT for m, r in models.items() if m.startswith("grpo")}
                                            for g, ids in sorted(by.items())}
    return out


def analyse(run_dir: pathlib.Path, phase10: pathlib.Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    cuts = json.loads((run_dir / "cuts.json").read_text()) if (run_dir / "cuts.json").exists() else []
    reinstated = json.loads((run_dir / "reinstated.json").read_text()) if (run_dir / "reinstated.json").exists() else None
    planned = [c["name"] for c in run["cells"]["followup"]]
    missing = [n for n in planned if not (run_dir / n / ".done").exists()]
    cut = {n for c in cuts for n in c["cells"]}
    # A reinstated cell that never finished (it failed, or the pod stopped) is reported as not run, not waited for; any other missing cell is waited for.
    # Only an applied reinstatement counts, and only cut 1's own cells can be excused: the new seeds' slices.
    applied = bool(reinstated and reinstated.get("applied"))
    cut1_cells = {c["name"] for c in run["cells"]["followup"] if c["model"] in {f"grpo-{s}" for s in NEW} and c["set"] in SLICES}
    not_run = {n for n in (reinstated or {}).get("reinstated", []) if applied and n in cut1_cells and n in missing and n not in cut}
    if set(missing) - cut - not_run:
        raise SystemExit(f"cells neither done nor cut: {sorted(set(missing) - cut - not_run)}: the analysis waits for them")
    if file_sha256(phase10 / "run.json") != PHASE10_RUN_SHA256:
        raise SystemExit(f"{phase10} is not Phase 10's run-2 (its run.json's sha256 is not {PHASE10_RUN_SHA256[:16]}...)")
    p10 = {"sft-v2/test": episodes(phase10, "sft-v2/test-greedy"), "grpo/test": episodes(phase10, "grpo/test-greedy"),
           **{f"sft-v2/{s}": episodes(phase10, f"sft-v2/{s}-greedy") for s in SLICES}}
    if any(v is None for v in p10.values()):
        raise SystemExit(f"Phase 10's cells are not all in {phase10}")
    grpo = {s: {x: episodes(run_dir, f"grpo-{s}/{x}-greedy") for x in ("test", "test_reserve", "stress", *SLICES)} for s in SEEDS}
    sft_reserve = episodes(run_dir, "sft-v2/test_reserve-greedy")

    test = {20261101: paired(p10["sft-v2/test"], p10["grpo/test"])}
    test.update({s: paired(p10["sft-v2/test"], grpo[s]["test"]) if grpo[s]["test"] else None for s in NEW})
    reserve = {s: paired(sft_reserve, grpo[s]["test_reserve"]) if grpo[s]["test_reserve"] and sft_reserve else None for s in SEEDS}
    res = {
        "labels": "exploratory (FOLLOWUP_PREREG §6)",
        "cuts": cuts,
        "test": {"per_seed": {str(s): v for s, v in test.items()}, "seed_20261101_from": "Phase 10, run-2", "verdict": verdict(test, NEW), "across_seeds": across(test),
                 "rule": "replicated if seeds 20261107 and 20261108 each have an interval above zero"},
        "reserve": {"per_seed": {str(s): v for s, v in reserve.items()}, "verdict": verdict(reserve, SEEDS), "across_seeds": across(reserve),
                    "rule": "replicated if all three seeds each have an interval above zero", "sft_v2": success(sft_reserve) if sft_reserve else None},
        "stress": stress_report({"sft-v2": episodes(run_dir, "sft-v2/stress-greedy"), "rft": episodes(run_dir, "rft/stress-greedy"),
                                 **{f"grpo-{s}": grpo[s]["stress"] for s in SEEDS}}),
        "rft_reserve": success(r) if (r := episodes(run_dir, "rft/test_reserve-greedy")) else CUT,
        "slices_descriptive": {x: {str(s): (paired(p10[f"sft-v2/{x}"], grpo[s][x]) if grpo[s][x] else (NOT_RUN if f"grpo-{s}/{x}-greedy" in not_run else CUT)) for s in NEW}
                               for x in SLICES},
        "slices_across_seeds": {x: across({s: paired(p10[f"sft-v2/{x}"], grpo[s][x]) if grpo[s][x] else None for s in NEW}) for x in SLICES},
        "reinstated": reinstated,
        "reinstated_fate": {n: ("reinstatement not applied" if not applied else "ran" if (run_dir / n / ".done").exists() else "cut again by the frozen check" if n in cut
                                else NOT_RUN) for n in (reinstated or {}).get("reinstated", [])},
        "if_not_replicated": "the original result is not retracted; it is reported as a single-seed finding. Not replicated is not a refutation (§6, power).",
        "same_seed_control": "seed 20261108's step 1 did not repeat its gate run (docs/V2.md, 2026-11-07): sampling is not exactly reproducible from the seed",
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase10", required=True, help="Phase 10's run directory (run-2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = analyse(pathlib.Path(a.run), pathlib.Path(a.phase10))
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k]["verdict"] for k in ("test", "reserve")}))


if __name__ == "__main__":
    main()
