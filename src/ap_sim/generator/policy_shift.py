"""Policy-shift cases. The same recorded case under two visible policy profiles: sandbox-v1 (A2 disabled) and
sandbox-v2 (A2 enabled: a quantity tolerance on invoice vs receipt, POLICY A2). The sandbox-v2 values (5% and EUR 500)
were fixed when the policy-shift split was frozen, and that split was evaluated under them. The case's cause is
coherent: the block was evaluated under v1, and v2 took effect after it (shown on the Active policy page and in the
event history), so a fresh check under v2 passes when the invoice's excess over coverage is within the tolerance.

A shift case is kept only when the grader's expected outcome DIFFERS between the profiles: the hidden world has
nothing unrecorded, so v1 requires the confirmation and keeps the invoice blocked, while v2 releases it after a
fresh check with no confirmation. Boundaries: the excess exactly at the tolerance (passes under v2) and one unit over
(fails: v2 then follows the v1 workflow, so the decision does not flip and the case is a negative control).
"""
from __future__ import annotations

import dataclasses
from decimal import Decimal

from ap_sim.generator.causal import Case, check_scope, generate
from ap_sim.grader import arithmetic as ar
from ap_sim.grader.grader import grade
from ap_sim.interfaces.structured import StructuredSurface
from ap_sim.interfaces.tools import ToolLayer


def with_policy(case: Case, profile: str, suffix: str) -> Case:
    """The case under a profile. Asserts (check_scope) that it is in scope at t = 0 under that profile and, for
    sandbox-v2, that the profile took effect at or before t = 0 (POLICY I2, I5)."""
    params = {**case.params, "policy": profile}
    check_scope(params)
    return dataclasses.replace(case, case_id=f"{case.case_id}-{suffix}", params=params)


def expected(case: Case) -> str:
    s = case.scenario()
    return grade(s, ToolLayer(s, StructuredSurface(), "surfaces/2").episode_log()).expected


def tolerance(case: Case) -> int:
    """The largest whole excess A2 (sandbox-v2) allows for this case's coverage and PO price."""
    p = case.params
    net = sum(p["receipts"]) - sum(q for _, q in p.get("reversals", ()))
    cover = Decimal(net - sum(p["other_invoices"]))
    pct, absolute = ar.A2["sandbox-v2"]
    return int(min(cover * pct / 100, absolute / Decimal(p["price"])))


def pair(seed: int, excess: int | None = None) -> tuple[Case, Case] | None:
    """(v1 case, v2 case) for one seed; None when no excess within the tolerance exists for its records."""
    for k in range(40):
        base = generate(f"PS{seed}-{k}", seed * 100 + k, {"unrecorded": "zero", "delay": "early", "gap": 1,
                                                          "reversal": False})
        tol = tolerance(base)
        if tol < 1:
            continue
        gap = excess if excess is not None else 1 + (seed + k) % tol
        case = generate(f"PS{seed}-{k}", seed * 100 + k, {"unrecorded": "zero", "delay": "early", "gap": gap,
                                                          "reversal": False})
        if tolerance(case) < gap and excess is None:
            continue
        return with_policy(case, "sandbox-v1", "v1"), with_policy(case, "sandbox-v2", "v2")
    return None


def shift_cases(n: int, seed: int = 4700, max_seeds: int = 400) -> list[tuple[Case, Case]]:
    out, s = [], seed
    while len(out) < n:
        if s - seed >= max_seeds:   # bounded: a generator that cannot find flipping pairs must fail, not hang
            raise ValueError(f"only {len(out)} flipping pairs in {max_seeds} seeds")
        p = pair(s)
        s += 1
        if p and expected(p[0]) == "keep_blocked" and expected(p[1]) == "release":
            out.append(p)
    return out


def boundary(seed: int = 4800) -> list[tuple[Case, Case, str]]:
    """(v1, v2, label): excess exactly at the tolerance (flips) and one unit over (does not flip)."""
    out = []
    for k in range(60):
        base = generate(f"PB{seed}-{k}", seed * 100 + k, {"unrecorded": "zero", "delay": "early", "gap": 1,
                                                          "reversal": False})
        tol = tolerance(base)
        if tol < 2:
            continue
        for gap, label in ((tol, "at_limit"), (tol + 1, "one_over")):
            c = generate(f"PB{seed}-{k}", seed * 100 + k, {"unrecorded": "zero", "delay": "early", "gap": gap,
                                                           "reversal": False})
            if tolerance(c) == tol:   # the label is part of the id: the two boundary cases share a base case id
                out.append((with_policy(c, "sandbox-v1", f"{label}-v1"), with_policy(c, "sandbox-v2", f"{label}-v2"),
                            label))
        if {lab for *_, lab in out} == {"at_limit", "one_over"}:
            return out
        out = []
    raise ValueError("no boundary pair found")
