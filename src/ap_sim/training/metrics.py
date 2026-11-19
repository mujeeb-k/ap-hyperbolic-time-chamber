"""Descriptive secondary metrics for the v2 evaluation reports. None is a success condition: success stays the grader's
verdict, and nothing here feeds the training reward.

Time to the first confirmation request on pending cases: on the episodes whose expected outcome is
pending, the business minutes from the start of the case (D1 09:00 = 0) to the agent's first accepted delivery-
confirmation request (a request on record, as Workflow rule 7 requires for pending). Episodes that never made one are
counted separately, not given a time.
"""
from __future__ import annotations

import statistics


def first_confirmation_request_at(log: dict) -> int | None:
    """Business minutes from the start to the agent's first accepted confirmation request; None if it made none."""
    times = [a["at"] for a in log["attempts"]
             if a["actor"] == "ap_agent" and a["op"] == "request_confirmation" and a["accepted"]]
    return min(times) if times else None


def pending_request_timing(episodes) -> dict:
    """episodes: (expected outcome, episode log) pairs, e.g. (verdict.expected, log). Only expected == "pending" counts."""
    times, never = [], 0
    for expected, log in episodes:
        if expected != "pending":
            continue
        t = first_confirmation_request_at(log)
        if t is None:
            never += 1
        else:
            times.append(t)
    out = {"pending_episodes": len(times) + never, "with_request": len(times), "never_requested": never}
    if times:
        q = statistics.quantiles(times, n=4) if len(times) > 1 else [times[0]] * 3
        out["minutes_to_first_request"] = {"median": statistics.median(times), "q1": q[0], "q3": q[2],
                                           "mean": round(statistics.fmean(times), 1), "min": min(times),
                                           "max": max(times)}
    return out
