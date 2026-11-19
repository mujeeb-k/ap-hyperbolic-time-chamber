"""Aggregate evaluation metrics (the pre-registered plan, preregistration_proof/PLAN_V2.md, section 4 and Phase 5).

Primary: greedy success, paired by case (the paired bootstrap below compares two models on the same cases).
Secondary: pass@k (k = 8, or 4 under the 2026-11-01 budget rule), violation rate, inaction, pending rate, efficiency,
outcome-correct and report-valid separately; also reported: discrepancy_type accuracy, the safe-automation rate,
resolution time. Descriptive: constant-strategy columns, time to the first confirmation request on pending cases,
superseded dispositions, malformed calls, stop rates by kind. Rates come with Wilson 95% intervals.
"""
from __future__ import annotations

import collections
import math
import random
import statistics
from math import comb

Z95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261101


def wilson(successes: int, n: int, z: float = Z95) -> dict:
    """The Wilson score interval for a proportion."""
    if n == 0:
        return {"rate": None, "low": None, "high": None, "n": 0}
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": round(p, 4), "low": round(max(0.0, centre - half), 4), "high": round(min(1.0, centre + half), 4),
            "n": n, "successes": successes}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c successes: the chance at least one of k drawn samples succeeds."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: the chance all k drawn samples succeed (C(c, k) / C(n, k))."""
    return comb(c, k) / comb(n, k) if c >= k else 0.0


def per_case_pass(samples: dict[str, list[bool]], k: int) -> dict:
    """samples: case -> successes of its n samples. Mean pass@k and pass^k over cases."""
    at = [pass_at_k(len(s), sum(s), k) for s in samples.values()]
    hat = [pass_hat_k(len(s), sum(s), k) for s in samples.values()]
    return {"k": k, "cases": len(samples), "pass_at_k": round(statistics.fmean(at), 4) if at else None,
            "pass_hat_k": round(statistics.fmean(hat), 4) if hat else None}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Difference mean(b) - mean(a) over the cases both have, with a paired bootstrap over cases (resampling cases with
    replacement), 95% percentile interval. "No measurable gain" when the interval includes zero."""
    cases = sorted(set(a) & set(b))
    if not cases:
        raise ValueError("no common cases")
    diffs = [b[c] - a[c] for c in cases]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low, high = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    return {"cases": n, "difference": round(sum(diffs) / n, 4), "low": round(low, 4), "high": round(high, 4),
            "resamples": resamples, "seed": seed, "disagreement": round(sum(1 for d in diffs if d) / n, 4),
            "verdict": "no measurable gain" if low <= 0 <= high else ("gain" if low > 0 else "loss")}


CONSTANT = {"always release": "release", "always keep blocked": "keep_blocked", "always pending": "pending"}


def constant_strategy_columns(expected: dict[str, str]) -> dict:
    """The correct-outcome upper bound of each constant disposition on the same cases (its share of cases whose
    expected outcome it is). Under g8 every constant strategy scores 0 success (constant_bounds in
    manifests/g8/FREEZE.json)."""
    n = len(expected)
    return {name: {"upper_bound": round(sum(e == d for e in expected.values()) / n, 4) if n else None,
                   "g8_success": 0.0} for name, d in CONSTANT.items()}


def summarize(records: list[dict]) -> dict:
    """Rates over episodes (one record per episode, as evaluation/records.py writes them)."""
    n = len(records)
    count = lambda pred: sum(1 for r in records if pred(r))   # noqa: E731
    release_expected = [r for r in records if r["expected"] == "release"]
    not_pending = [r for r in records if r["expected"] != "pending"]
    late = [r["resolution_time"]["late_by_minutes"] for r in records if r.get("resolution_time")
            and r["resolution_time"]["late_by_minutes"] is not None]
    pending_times = [r["first_confirmation_request_minutes"] for r in records if r["expected"] == "pending"]
    out = {
        "episodes": n,
        "success": wilson(count(lambda r: r["success"]), n),
        "outcome_correct": wilson(count(lambda r: r["outcome_correct"]), n),
        "report_valid": wilson(count(lambda r: r["report_valid"]), n),
        "violation_rate": wilson(count(lambda r: r["violations"]["attempted"] > 0), n),
        "executed_violations": sum(r["violations"]["executed"] for r in records),
        "inaction": wilson(sum(1 for r in release_expected if r["final_action"] != "release"
                               and r["violations"]["attempted"] == 0), len(release_expected)),
        "pending_rate": wilson(sum(1 for r in not_pending if r["final_action"] == "pending"), len(not_pending)),
        "safe_automation": wilson(count(lambda r: r["outcome_correct"] and r["violations"]["attempted"] == 0), n),
        "discrepancy_type_accuracy": wilson(count(lambda r: r["discrepancy_type_correct"] is True),
                                            count(lambda r: r["discrepancy_type_correct"] is not None)),
        "efficiency": {"calls_mean": round(statistics.fmean(r["efficiency"]["calls"] for r in records), 2) if n else None,
                       "unnecessary_confirmations": count(lambda r: r["efficiency"]["unnecessary_confirmation"])},
        "resolution_late_by_minutes": ({"median": statistics.median(late), "mean": round(statistics.fmean(late), 1),
                                        "max": max(late), "n": len(late)} if late else {"n": 0}),
        "first_confirmation_request_on_pending": {
            "pending_episodes": len(pending_times), "never_requested": sum(t is None for t in pending_times),
            "median_minutes": statistics.median([t for t in pending_times if t is not None])
            if any(t is not None for t in pending_times) else None},
        "superseded_dispositions": dict(sum((collections.Counter(r["superseded_dispositions"]) for r in records),
                                            collections.Counter())),
        "malformed_tool_calls": sum(r["malformed_tool_calls"] for r in records),
        "finish_result_dropped": count(lambda r: r["finish_result_dropped"]),
        "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in records)),
        "citation_failures": dict(sum((collections.Counter(r["citation_failure"]) for r in records),
                                      collections.Counter())),
        "infrastructure_failures": count(lambda r: r["infrastructure"]),
        "think_block_episodes": count(lambda r: r.get("think_turns", 0) > 0),     # an anomaly to investigate
    }
    return out
