"""Anchor checks for the post-hoc analyses of 2026-11-12, each reproducing a number the paper already reports.

Before an analysis reports anything new, its script reproduces such a number from the same records and states it. A failed anchor stops the
analysis (SystemExit) before anything is written.

Each check returns {"anchor", "v5_location", "expected", "reproduced", "ok"}. `v5_location` is where the paper reports the number, in the
section and table numbering of the 2026-11-12 draft. Counts only: no case content is printed or kept.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import posthoc_common as c   # noqa: E402


def _successes(path: pathlib.Path) -> int:
    return sum(bool(r["success"]) for r in c.episodes(path / "episodes.jsonl").values())


def _result(anchor: str, where: str, expected, reproduced) -> dict:
    return {"anchor": anchor, "v5_location": where, "expected": expected, "reproduced": reproduced, "ok": expected == reproduced}


def item1() -> list[dict]:
    return [_result("greedy successes on test and the policy shift, SFT-v2 and GRPO seed 1", "Table 7 and Appendix B.1 (SFT-v2 185 / 23, GRPO seed 1 200 / 22)",
                    {"sft-v2": [185, 23], "grpo-20261101": [200, 22]},
                    {"sft-v2": [_successes(c.R2 / "sft-v2/test-greedy"), _successes(c.R2 / "sft-v2/policy_shift-greedy")],
                     "grpo-20261101": [_successes(c.R2 / "grpo/test-greedy"), _successes(c.R2 / "grpo/policy_shift-greedy")]})]


def item2() -> list[dict]:
    cells = {"base": c.R2 / "base", "sft-v2": c.R2 / "sft-v2", "grpo-20261101": c.R2 / "grpo",
             "grpo-20261107": c.E1 / "grpo-20261107", "grpo-20261108": c.E1 / "grpo-20261108"}
    return [_result("greedy V6 successes", "Appendix B.1 (base 0, SFT-v2 43, GRPO seeds 1-3 44 / 44 / 38)",
                    {"base": 0, "sft-v2": 43, "grpo-20261101": 44, "grpo-20261107": 44, "grpo-20261108": 38},
                    {m: _successes(p / "v6-greedy") for m, p in cells.items()})]


def item3() -> list[dict]:
    """12 of SFT-v2's 15 test failures are cancelled-invoice omissions (g8's own diagnostics, as manuscript_counts labels them);
    and GRPO seed 1's 27 passing V5 reports with a cancelled invoice cite it (27 of 27)."""
    import manuscript_counts as mc
    from ap_sim.grading import g8
    factories = c.cases("test")
    recs = c.episodes(c.R2 / "sft-v2/test-greedy/episodes.jsonl")
    failures = {k: r for k, r in recs.items() if not r["success"]}
    omissions = 0
    for case, rec in failures.items():
        scenario, log, _ = c.replay(factories[case], case, rec)
        v = g8.grade(scenario, log)
        omissions += "cancelled_invoice_omission" in mc.labels(v, c.g8_reasons(scenario, log))
    grpo = c.episodes(c.R2 / "grpo/test-greedy/episodes.jsonl")
    cited = [c.cancelled_citation(r) for k, r in grpo.items() if k.startswith("test-V5-")]
    with_cancelled = [x for x in cited if x[0]]
    return [_result("SFT-v2's test failures and its cancelled-invoice omissions among them", "§4.4 (12 of SFT-v2's 15 test failures)",
                    [12, 15], [omissions, len(failures)]),
            _result("GRPO seed 1's V5 test reports with a cancelled invoice that cite it", "Appendix D, grading checks (27 of 27)",
                    [27, 27], [sum(1 for x in with_cancelled if x[1]), len(with_cancelled)])]


def item4() -> list[dict]:
    """The baseline's metrics (g8's reached against expected, its unsafe attempts) through the same replay and grader path, on SFT-v2's stored test episodes."""
    from ap_sim.grading import g8
    factories = c.cases("test")
    right = success = attempts = 0
    for case, rec in c.episodes(c.R2 / "sft-v2/test-greedy/episodes.jsonl").items():
        scenario, log, _ = c.replay(factories[case], case, rec)
        v = g8.grade(scenario, log)
        right += v.reached == v.expected
        success += v.reward == 1
        attempts += int(v.diagnostics.get("unsafe_attempts") or 0)
    return [_result("SFT-v2 on test, regraded: successes, right final actions, attempted unsafe actions",
                    "Appendix B.1 (185 / 200) and B.7 (0 attempted unsafe actions)", [185, 200, 0], [success, right, attempts])]


def item7() -> list[dict]:
    """Profiling SFT-v2 on 120 RL cases x 8 samples: no case at 0 of 8, 78 at 8 of 8, 42 at 1-7."""
    path = c.ROOT / "runs/session1-pod/session1/profiling/episodes.jsonl"
    per = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        per[r["case"].split("#")[0]].append(bool(r["success"]))
    dist = collections.Counter("8" if all(v) else "0" if not any(v) else "1-7" for v in per.values())
    return [_result("profiling: RL cases by successes of 8", "Appendix A.2 (0 at 0/8, 42 at 1-7, 78 at 8/8, of 120)",
                    {"cases": 120, "0": 0, "1-7": 42, "8": 78}, {"cases": len(per), "0": dist["0"], "1-7": dist["1-7"], "8": dist["8"]})]


def item8() -> list[dict]:
    cell = c.E1 / "grpo-20261108"
    return [_result("GRPO seed 3's greedy successes on test and the forgetting set", "Appendix B.1 (193 / 200 on test, 32 / 40 on forgetting)",
                    [193, 32], [_successes(cell / "test-greedy"), _successes(cell / "forgetting_l1-greedy")])]


def check(item: str) -> list[dict]:
    results = globals()[item]()
    for r in results:
        print(f"anchor ({item}): {r['anchor']} [{r['v5_location']}]: expected {r['expected']}, reproduced {r['reproduced']}: {'ok' if r['ok'] else 'FAILED'}",
              flush=True)
    if not all(r["ok"] for r in results):
        raise SystemExit(f"{item}: anchor failed; the item is stopped and nothing is written")
    return results
