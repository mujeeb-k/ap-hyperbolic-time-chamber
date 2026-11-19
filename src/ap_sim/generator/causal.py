"""Causal scenario generator. Samples the underlying situation, checks its constraints,
and builds the scenario with `fixtures.build`. The VARIANT is derived from the sampled variables, never chosen as a
label, and the acceptable outcome is decided by the grader alone. Everything here is evaluator-only: the agent sees
only what the engine renders.

Sampled variables (the causal situation):
  receipts            1-3 posted goods receipts on the PO line (quantities)
  reversal            optional reversal of part of one receipt (it corrected an over-posting; V6)
  other_active        0-2 other active invoices on the PO line
  cancellation        none, or one other invoice cancelled BEFORE the block (no effect) or AFTER it (stale block)
  stale_receipt       whether the last receipt was posted after the block was evaluated (stale block)
  gap                 invoice quantity minus current coverage: 0 (covered; V5) or 1-25 (V1-V4, V6)
  unrecorded          goods delivered but not recorded: 0, exactly the gap, or above it (a partial delivery, between 0
                      and the gap, is out of scope: POLICY I3)
  delay               the warehouse's answer: early, mid, late but within the horizon (V3), beyond it, or never (V4)
Constraints (checked, else resampled): the invoice was blocked when evaluated (coverage then below its quantity);
a covered case (gap 0) is blocked only because of a stale cause; quantities are positive; the PO quantity exceeds
receipts and invoiced totals (no unintended PO comparison failure).
Assertions (checked, never resampled, so the random stream and every existing case are unchanged): `check_scope`
raises OutOfScopeCase for a case POLICY I3 puts out of scope at t = 0, or (I2, I5) a sandbox-v2 case whose profile
takes effect after t = 0.

Each case records its provenance (generator version, seed, variables, derived variant, constraint results) and a
STRUCTURAL SIGNATURE: the abstract shape of the case, independent of ids and quantities, used to measure diversity.

Counterfactual siblings share every recorded (agent-visible) record and differ only in the hidden world, which
changes the correct outcome: the warehouse's delivered total (delivered vs not) or its answer time (within vs
beyond the horizon).
"""
from __future__ import annotations

import dataclasses
import random
from decimal import Decimal

from ap_sim.generator.fixtures import Scenario, build

GENERATOR_VERSION = "causal/2"   # provenance; causal/2: no invoice is cancelled before it was posted
RNG_STREAM = "causal/1"          # the random stream stays causal/1, so every seed that was coherent yields the same case
PRICES = ["4.80", "7.35", "9.99", "12.50", "16.75", "18.40", "20.00", "31.25", "45.00"]
DELAYS = {"early": 240, "mid": 720, "late": 1920, "beyond": 2880, "never": None}  # H = 2400 (POLICY C1)


class OutOfScopeCase(ValueError):
    """A case the generator must never produce: out of scope at t = 0 (POLICY I3), or a sandbox-v2 case whose profile
    takes effect after t = 0 (POLICY I2, I5)."""


def check_scope(params: dict) -> None:
    """Assert, on the records at t = 0 under the case's profile: sandbox-v2 (if active) took effect at or before
    t = 0 (C1: an event at t <= the clock has happened), net posted receipts are positive (no A4-DW), no blocking
    comparison other than invoice vs receipt (A4) fails (no A1, A5, A6 or A7 failure), and a confirmation can only
    be positive or negative (unrecorded quantity 0 or at least the gap; no partial confirmation). I3's "after a
    resolving action" is left to the reachability checks, which run every resolving action."""
    from ap_sim.domain.policy import SANDBOX_V2_EFFECT_MINUTES
    from ap_sim.grader import arithmetic as ar
    profile = params.get("policy", "sandbox-v1")
    problems = []
    if profile == "sandbox-v2" and params["blocked_at"] + SANDBOX_V2_EFFECT_MINUTES > 0:
        problems.append(f"sandbox-v2 takes effect at t = {params['blocked_at'] + SANDBOX_V2_EFFECT_MINUTES}, after the start")
    s = build("scope", 0, **params)
    snap = {}
    for r in s.records:
        snap.setdefault(r.kind, []).append(r.model_dump(mode="json"))
    po_line = next(l["po_line"] for l in snap["invoice_line"] if l["invoice"] == s.current_invoice)
    net = ar.net_receipts(snap, po_line)
    if net <= 0:
        problems.append("zero net posted receipts (A4-DW)")
    other = sorted(set(ar.failing_blocking(snap, s.current_invoice, profile)) - {"invoice_vs_receipt"})
    if other:
        problems.append(f"blocking comparisons fail at the start: {other}")
    gap0, unrecorded = ar.gap(snap, po_line, s.current_invoice, profile), Decimal(params["physical_total"]) - net
    if gap0 > 0 and 0 < unrecorded < gap0:
        problems.append(f"partial confirmation: unrecorded {unrecorded} is between 0 and the gap {gap0}")
    if problems:
        raise OutOfScopeCase("; ".join(problems))


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    seed: int
    variables: dict
    variant: str
    signature: tuple
    params: dict

    def scenario(self) -> Scenario:
        return build(self.case_id, self.seed, **self.params)

    def provenance(self) -> dict:
        return {"generator": GENERATOR_VERSION, "case_id": self.case_id, "seed": self.seed,
                "variables": self.variables, "variant": self.variant, "signature": list(self.signature),
                "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params.items()}}


def derive_variant(v: dict) -> str:
    """The variant (V1-V6) follows from the variables, not from a label."""
    if v["gap"] == 0:
        return "V5_stale_covered"
    answered = v["delay"] in ("early", "mid", "late")
    delivered = v["unrecorded"] in ("equal", "above")
    base = ("V4_pending" if not answered else
            ("V3a_late_delivered" if delivered else "V3b_late_not_delivered") if v["delay"] == "late" else
            "V1_delivered_unrecorded" if delivered else "V2_not_delivered")
    return f"V6_reversal+{base}" if v["reversal"] else base


def signature(v: dict) -> tuple:
    return (derive_variant(v), len(v["receipts"]), v["reversal"], len(v["other_active"]), v["cancellation"],
            v["stale_receipt"], "gap0" if v["gap"] == 0 else "gap>0", v["unrecorded"], v["delay"])


def _sample(rng: random.Random, force: dict | None = None) -> dict:
    force = force or {}
    n = force.get("n_receipts", rng.choice([1, 2, 2, 3]))
    receipts = [rng.randint(15, 90) for _ in range(n)]
    reversal = force.get("reversal", rng.random() < 0.2)
    rev = (rng.randrange(n), rng.randint(3, max(3, receipts[0] // 3))) if reversal else None
    if rev and rev[1] >= receipts[rev[0]]:
        rev = (rev[0], receipts[rev[0]] - 1)
    other_active = [rng.randint(5, 30) for _ in range(force.get("n_other", rng.choice([0, 0, 1, 1, 2])))]
    cancellation = force.get("cancellation", rng.choice(["none", "none", "before_block", "after_block"]))
    stale_receipt = force.get("stale_receipt", n >= 2 and rng.random() < 0.35)
    gap = force.get("gap", 0 if rng.random() < 0.3 else rng.randint(1, 25))
    unrecorded = force.get("unrecorded", "none" if gap == 0 else rng.choice(["zero", "equal", "above"]))
    delay = force.get("delay", rng.choice(["early", "mid", "mid", "late", "beyond", "never"]))
    return {"receipts": receipts, "reversal": bool(rev), "reversal_spec": rev, "other_active": other_active,
            "cancellation": cancellation, "cancelled_qty": rng.randint(10, 40) if cancellation != "none" else 0,
            "stale_receipt": stale_receipt, "gap": gap, "unrecorded": unrecorded, "delay": delay,
            "price": rng.choice(PRICES)}


def _params(v: dict, rng: random.Random) -> tuple[dict | None, list[str]]:
    receipts, rev = v["receipts"], v["reversal_spec"]
    net = sum(receipts) - (rev[1] if rev else 0)
    coverage = net - sum(v["other_active"])
    invoice_qty = coverage + v["gap"]
    problems = []
    if coverage <= 0 or invoice_qty <= 0:
        problems.append("coverage or invoice quantity not positive")
    gap = v["gap"]
    extra = {"none": 0, "zero": 0, "equal": gap, "above": gap + rng.randint(1, 10)}[v["unrecorded"]]
    # Receipt i is posted at -2400 + 400 (i + 1); a stale receipt means the block came before the last one.
    times = [-2400 + 400 * (i + 1) for i in range(len(receipts))]
    blocked_at = (times[-1] - 100) if v["stale_receipt"] else -160
    cancelled_at = None
    if v["cancellation"] != "none":
        cancelled_at = -80 if v["cancellation"] == "after_block" else blocked_at - 30
    if gap == 0 and not (v["stale_receipt"] or v["cancellation"] == "after_block"):
        problems.append("a covered invoice needs a stale cause to be blocked")
    # The builder posts other invoices after the last receipt, the cancelled one first (fixtures.build), so a
    # cancellation must come after -2400 + 400 x len(receipts) + 120. causal/1 compared with -2400 + 120 instead and
    # admitted a stale block with a "before the block" cancellation, cancelled 250 minutes before it was posted (97 of
    # version 1's frozen and fresh cases, found in an audit before the v2 splits were built). Such a combination is
    # now resampled.
    if cancelled_at is not None and cancelled_at <= -2400 + 400 * len(receipts) + 120:
        problems.append("cancellation before the invoice was posted")
    invoiced = sum(v["other_active"]) + v["cancelled_qty"] + invoice_qty
    po_qty = max(net + extra, invoiced) + rng.randint(10, 60)
    if problems:
        return None, problems
    params = dict(po_qty=po_qty, price=v["price"], receipts=receipts, other_invoices=v["other_active"],
                  invoice_qty=invoice_qty, physical_total=net + extra, delay=DELAYS[v["delay"]],
                  cancelled_invoices=(v["cancelled_qty"],) if v["cancellation"] != "none" else (),
                  blocked_at=blocked_at, cancelled_at=cancelled_at, reversals=(tuple(rev),) if rev else ())
    # Admissibility by the engine's own load-time evaluation (the records as they stood when the block was
    # evaluated): blocked on receipt coverage, and on nothing else.
    from ap_sim.engine.engine import Engine
    s = build("probe", 0, **params)
    e = Engine(list(s.records), s.world, s.seed, s.current_invoice)
    results = {c.comparison: (c.effect, c.result) for c in e.store.comparison_results(s.current_invoice)}
    if results.get("invoice_vs_receipt") != ("blocking", "fail"):
        return None, ["not blocked on receipt coverage when evaluated"]
    if any(eff == "blocking" and res == "fail" for name, (eff, res) in results.items() if name != "invoice_vs_receipt"):
        return None, ["another blocking comparison fails"]
    check_scope(params)
    return params, []


def generate(case_id: str, seed: int, force: dict | None = None, max_tries: int = 200) -> Case:
    rng = random.Random(f"{RNG_STREAM}:{seed}")
    for attempt in range(max_tries):
        v = _sample(rng, force)
        params, problems = _params(v, rng)
        if params is not None:
            v = {**v, "constraint_resamples": attempt}
            return Case(case_id, seed, {k: x for k, x in v.items() if k != "reversal_spec"} | {
                "reversal_spec": list(v["reversal_spec"]) if v["reversal_spec"] else None},
                derive_variant(v), signature(v), params)
    raise ValueError(f"no admissible case for seed {seed} with {force} in {max_tries} tries")


def counterfactual(case: Case, change: str) -> Case:
    """A sibling with the SAME recorded records and a different hidden world: `delivered` flips the warehouse's
    delivered total across the gap (V1 <-> V2), `timing` moves its answer beyond the horizon (-> V4)."""
    v, p = dict(case.variables), dict(case.params)
    if change == "delivered":
        v["unrecorded"] = "zero" if v["unrecorded"] in ("equal", "above") else "equal"
        net = sum(v["receipts"]) - (v["reversal_spec"][1] if v["reversal_spec"] else 0)
        p["physical_total"] = net + (0 if v["unrecorded"] == "zero" else v["gap"])
    elif change == "timing":
        v["delay"] = "beyond"
        p["delay"] = DELAYS["beyond"]
    else:
        raise ValueError(change)
    check_scope(p)
    return Case(case.case_id + f"-cf-{change}", case.seed, v, derive_variant(v), signature(v), p)


def boundary_cases(seed: int) -> list[Case]:
    """The boundary cases: unrecorded exactly equal to the gap; a covered invoice at equality; answers at
    the edge of the horizon (late within, beyond)."""
    out = [generate(f"B{seed}-eq-unrec", seed, {"unrecorded": "equal", "gap": 7, "delay": "mid"}),
           generate(f"B{seed}-late-in", seed + 1, {"unrecorded": "equal", "gap": 5, "delay": "late"}),
           generate(f"B{seed}-beyond", seed + 2, {"unrecorded": "equal", "gap": 5, "delay": "beyond"}),
           generate(f"B{seed}-stale-eq", seed + 3, {"gap": 0, "stale_receipt": True, "n_receipts": 2,
                                                     "cancellation": "none"})]
    return out


def as_decimal(x) -> Decimal:
    return Decimal(str(x))
