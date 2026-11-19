"""Difficulty ladder. Levels are assigned from a generated case's STRUCTURE (`causal.Case.variables`), not from
tool-call counts:

  L1  short cases: at most 2 decisions, at most 1 wait, simple records (ladder/2, below)
  L2  covered cases whose records are not simple (history: a reversal, a cancellation, 3 receipts or 2 other
      invoices); no confirmation request is needed
  L3  the full investigation: a delivered-unrecorded shortfall, a later answer, or pending

L1 BY EFFORT (ladder/2). L1 is defined by the effort the shortest valid workflow needs from
an agent that only observes, computed from the case's variables by `effort()`:
  decisions  the operational actions that commit the case, counted in that workflow: request confirmation, request
             receipt posting, reprocess, release, keep blocked (views, fills and waits are not decisions)
  waits      advance_time calls at the 480-minute maximum until the answer can be seen (ceil((d - 15) / 480)), or,
             when no answer can arrive in time, until no resolution remains possible (to H - 30), since an
             observing agent cannot know in advance that the answer will not come
  simple     at most 2 receipts, at most 1 other active invoice, no reversal, no cancellation
L1 = decisions <= 2 and waits <= 1 and simple records: covered stale cases (reprocess, release; no wait) and early
non-delivery cases (request, keep blocked; one wait; 7-8 calls). L2 = covered cases with history (no wait). L3 =
everything else (delivered-unrecorded: 4 decisions; later answers; pending). The former `L1_blocked_candidate`
cases are L1 under ladder/2; the name is kept as an alias so older reports still resolve.

Measured reference-agent calls (600 generated cases): covered 5 (L1 and L2 alike); non-delivery 8-10;
delivered-but-unrecorded 11-13; late answers 14-17; pending 16. A covered case with history still takes 5 calls, so
call counts do not separate L2 from L1.
"""
from __future__ import annotations

from ap_sim.generator.causal import Case, generate

L1_BLOCKED_CANDIDATE = "L1_blocked_candidate"   # ladder/1 name; under ladder/2 these cases are L1


LADDER_VERSION = "ladder/2"


def effort(case: Case) -> dict:
    """Decisions and waits of the shortest valid workflow for an observing agent, and record simplicity."""
    from ap_sim.events.clock import ADVANCE_MAX, HORIZON
    v, d = case.variables, case.params["delay"]
    simple = (not v["reversal"] and v["cancellation"] == "none" and len(v["receipts"]) <= 2
              and len(v["other_active"]) <= 1)
    if v["gap"] == 0:
        decisions, waits = 2, 0                                  # reprocess, release
    elif d is not None and d + 25 <= HORIZON:                    # an answer that leaves time to act on it
        decisions = 4 if v["unrecorded"] in ("equal", "above") else 2   # +posting, reprocess, release | keep blocked
        waits = -(-max(0, d - 15) // ADVANCE_MAX)
    else:                                                        # pending: wait until no resolution is possible
        decisions, waits = 2, -(-(HORIZON - 30 - 15) // ADVANCE_MAX)
    return {"decisions": decisions, "waits": waits, "simple": simple}


def level(case: Case) -> str:
    e = effort(case)
    if e["decisions"] <= 2 and e["waits"] <= 1 and e["simple"]:
        return "L1"
    if case.variables["gap"] == 0:
        return "L2"
    return "L3"


def sample(target: str, n: int, seed: int) -> list[Case]:
    """`n` cases of one level (L1, L2, L3 or the L1 blocked candidate), deterministic in `seed`, drawn from the
    causal generator with structural nudges for the rarer levels; every case keeps its own derived variant."""
    force = {"L1": {"gap": 0, "reversal": False, "cancellation": "none", "stale_receipt": True, "n_other": 1},
             "L2": {"gap": 0},
             L1_BLOCKED_CANDIDATE: {"unrecorded": "zero", "delay": "early", "reversal": False, "cancellation": "none",
                                    "n_receipts": 2, "n_other": 1},
             "L3": {}}[target]
    out, k = [], 0
    while len(out) < n:
        k += 1
        if k > 50 * n:
            raise ValueError(f"could not sample {n} cases for {target}")
        case = generate(f"{target}-{seed}-{k}", seed * 1000 + k, force or None)
        lv = level(case)
        blocked_l1 = lv == "L1" and case.variables["gap"] > 0
        if (target == L1_BLOCKED_CANDIDATE and blocked_l1) or (target == lv and target != L1_BLOCKED_CANDIDATE):
            out.append(case)
    return out
