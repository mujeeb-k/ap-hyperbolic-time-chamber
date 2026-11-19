"""The messiness stress set (docs/FOLLOWUP_PREREG.md section 4; rulings P and J): generator, builder and loader.

A new package outside the hash-frozen src/ap_sim/generator/ (the pre-registration named src/ap_sim/generator/stress.py;
moved here so no unrecorded file sits in a hashed directory): no frozen file is edited. Cases are built from the frozen record classes (`domain/records.py`), the
frozen hidden-world classes (`events/world.py`) and the frozen `fixtures.Scenario`, exactly as `fixtures.build` does,
in two parameter formats (manifests/stress/SPEC.md):
  frozen  exactly the frozen parameters, built by fixtures.build itself, whenever the case fits its fixed timing: at
          most 3 receipts, at most one cancellation and no M3 (so M1 with 2-3 receipts and M4 are data, as section 4
          says)
  timed   the cases fixtures.build cannot express (several cancellations at different times, 4-5 receipts, or a
          re-issue posted after its original's cancellation): every
          pre-episode time stored explicitly (`timed_receipts`, `timed_other_invoices`), built by `build_timed`, which
          mirrors fixtures.build record for record (tests/test_stress_set.py checks it on the frozen timing)

Complications (each case carries one or two; detected from the parameters by `complications`, never from a label):
  M1  several recorded receipts: 2 to 5 (never 6: a sixth would post at t = 0)
  M2  several cancelled invoices on the PO line (2 or 3), each cancelled at a different time
  M3  a re-issued invoice: an active invoice of the same quantity as a cancelled one, posted after that one was
      cancelled (user ruling 2026-11-06; the content predicate below still matches any same-quantity active invoice
      posted after the cancelled one was posted, and timing_problems refuses one posted before the cancellation)
  M4  a look-alike duplicate: another active invoice of the same quantity as the invoice under review

The type is DERIVED after the complications are added: from the built scenario's own records, through the frozen
`causal.derive_variant`, and the outcome from the frozen grader (g7) on an unacted episode log.
Evaluator side only: the training environment never imports this module.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import pathlib
import random
from collections.abc import Callable
from decimal import Decimal

from ap_sim.domain.records import Invoice, InvoiceLine, Money, POLine, PurchaseOrder, Qty, Receipt, Record
from ap_sim.events.world import HiddenWorld, PhysicalDelivery
from ap_sim.generator.causal import DELAYS, PRICES, derive_variant
from ap_sim.generator.fixtures import CENT, Scenario

ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "manifests" / "stress" / "messiness.json"
SPLIT = "stress_messiness"

GENERATOR_VERSION = "stress/2"   # stress/2: frozen parameters (fixtures.build) wherever they can express the case
RNG_STREAM = "stress/1"
SEED_BASE, SPLIT_INDEX = 11_000_000, 8              # seed = 11,000,000 + 100,000 x 8 + 10,000 x group + k
GROUPS = ("V1", "V2", "V3a", "V3b", "V4", "V5")
COUNTS = {"V1": 20, "V2": 18, "V3a": 10, "V3b": 10, "V4": 22, "V5": 20}   # the test mix (pre-registration)
VARIANT = {"V1": "V1_delivered_unrecorded", "V2": "V2_not_delivered", "V3a": "V3a_late_delivered",
           "V3b": "V3b_late_not_delivered", "V4": "V4_pending", "V5": "V5_stale_covered"}
EXPECTED = {"V1": "release", "V2": "keep_blocked", "V3a": "release", "V3b": "keep_blocked", "V4": "pending",
            "V5": "release"}
COMPLICATIONS = ("M1", "M2", "M3", "M4")
SINGLES, PAIRS = 7, 12                                # 4 x 7 + 6 x 12 = 100 cases; each complication in 7 + 3 x 12 = 43

# Timing (business minutes; t = 0 is the start of the episode). See SPEC.md.
RECEIPT_BASE, RECEIPT_STEP, MAX_RECEIPTS = -2400, 400, 5   # receipt i (1-based) at -2400 + 400 i: -2000 ... -400
BLOCK_AT = -160                                       # the block's evaluation, as in fixtures.build
STALE_RECEIPT_OFFSET = 100                            # a stale receipt: the block was evaluated 100 minutes before it
UNRECORDED_DELIVERY_AT = -300                         # delivered, not recorded (as in fixtures.build)
MIN_GAP_MINUTES = 20                                  # every recorded event is at least 20 minutes from the block and t = 0
HIDDEN = ("physical_total", "delay")


# --- the builder ---------------------------------------------------------------------------------------------------
FROZEN_KEYS = ("po_qty", "price", "receipts", "other_invoices", "invoice_qty", "physical_total", "delay",
               "cancelled_invoices", "blocked_at", "cancelled_at", "reversals")
TIMED_KEYS = ("po_qty", "price", "invoice_qty", "blocked_at", "physical_total", "delay", "timed_receipts",
              "timed_other_invoices")
INVOICE_STEP = 120                                    # fixtures.build: other invoices every 120 minutes


def is_timed(params: dict) -> bool:
    return "timed_receipts" in params


def build(name: str, seed: int, **params) -> Scenario:
    """Either format: frozen parameters through the frozen fixtures.build, timed ones through build_timed."""
    if is_timed(params):
        return build_timed(name, seed, **params)
    from ap_sim.generator.fixtures import build as frozen_build
    from ap_sim.generator.manifests import _tuples
    return frozen_build(name, seed, **_tuples(params))


def timed_view(params: dict) -> dict:
    """Any case in the timed representation: [quantity, posted_at] per receipt, [quantity, posted_at, cancelled_at]
    per other invoice in posting order. A frozen case gets fixtures.build's timing: receipt i at -2400 + 400 i; other
    invoices every 120 minutes after the last receipt, the cancelled ones first, sharing cancelled_at."""
    if is_timed(params):
        return {"receipts": params["timed_receipts"], "other_invoices": params["timed_other_invoices"],
                "invoice_qty": params["invoice_qty"], "blocked_at": params["blocked_at"]}
    if params.get("reversals"):
        raise ValueError("a reversal is out of the stress set's scope")
    rt = [RECEIPT_BASE + RECEIPT_STEP * (i + 1) for i in range(len(params["receipts"]))]
    t, others = rt[-1], []
    for q, cancelled in ([(q, True) for q in params.get("cancelled_invoices", ())]
                         + [(q, False) for q in params["other_invoices"]]):
        t += INVOICE_STEP
        others.append([q, t, params.get("cancelled_at") if cancelled else None])
    return {"receipts": [[q, x] for q, x in zip(params["receipts"], rt)], "other_invoices": others,
            "invoice_qty": params["invoice_qty"], "blocked_at": params.get("blocked_at", -160)}


def build_timed(name: str, seed: int, *, po_qty: int, price: str, invoice_qty: int, blocked_at: int,
                physical_total: int, delay: int | None, timed_receipts: list, timed_other_invoices: list,
                policy: str = "sandbox-v1") -> Scenario:
    """`timed_receipts`: [quantity, posted_at] per recorded goods receipt. `timed_other_invoices`: [quantity, posted_at,
    cancelled_at or None] per other invoice on the PO line (cancelled when cancelled_at is set). The invoice under
    review is posted, evaluated and blocked at `blocked_at`. Record ids come from the same seeded stream and in the
    same order as fixtures.build (PO, receipts, other invoices, the invoice under review)."""
    rng = random.Random(f"{seed}:initial-ids")
    used: set[str] = set()

    def rid(prefix: str) -> str:
        while (v := f"{prefix}-{rng.randrange(100000, 1000000)}") in used:
            pass
        used.add(v)
        return v

    unit = Decimal(price)
    po = PurchaseOrder(id=rid("PO"), supplier="Supplier A", status="open")
    pl = POLine(id=f"{po.id}-1", po=po.id, item="Item 1", ordered_quantity=Qty(value=Decimal(po_qty)),
                unit_price=Money(value=unit), line_amount=Money(value=(po_qty * unit).quantize(CENT)))
    records: list[Record] = [po, pl]
    receipts, other_invoices = timed_receipts, timed_other_invoices
    for q, t in receipts:
        records.append(Receipt(id=rid("GR"), po_line=pl.id, posted_at=t, movement="receipt",
                               quantity=Qty(value=Decimal(q))))
    for q, posted, cancelled in other_invoices:
        inv = Invoice(id=rid("INV"), supplier="Supplier A", po=po.id, posted_at=posted, eligibility="eligible",
                      evaluated_at=posted,
                      **({"state": "cancelled", "cancelled_at": cancelled, "version": 2} if cancelled is not None
                         else {"state": "posted_released"}))    # cancelled: posted (v1), then cancelled (v2)
        records += [inv, InvoiceLine(id=f"{inv.id}-1", invoice=inv.id, po_line=pl.id, quantity=Qty(value=Decimal(q)),
                                     unit_price=Money(value=unit), amount=Money(value=(q * unit).quantize(CENT)))]
    current = Invoice(id=rid("INV"), supplier="Supplier A", po=po.id, state="posted_blocked", posted_at=blocked_at,
                      evaluated_at=blocked_at)
    records += [current, InvoiceLine(id=f"{current.id}-1", invoice=current.id, po_line=pl.id,
                                     quantity=Qty(value=Decimal(invoice_qty)), unit_price=Money(value=unit),
                                     amount=Money(value=(invoice_qty * unit).quantize(CENT)))]
    recorded = sum(q for q, _ in receipts)
    deliveries = [PhysicalDelivery(po_line=pl.id, at=t, quantity=Decimal(q)) for q, t in receipts]
    if physical_total > recorded:      # delivered but not yet recorded (posting lag)
        deliveries.append(PhysicalDelivery(po_line=pl.id, at=UNRECORDED_DELIVERY_AT,
                                           quantity=Decimal(physical_total - recorded)))
    world = HiddenWorld(deliveries=tuple(deliveries), confirmation_delay=delay)
    return Scenario(name=name, seed=seed, records=tuple(records), world=world, current_invoice=current.id,
                    policy=policy)


# --- content predicates --------------------------------------------------------------------------------------------
def complications(params: dict) -> list[str]:
    """The complications a case carries, from its parameters alone (definitions in SPEC.md)."""
    view = timed_view(params)
    others = view["other_invoices"]
    cancelled = [(q, p, c) for q, p, c in others if c is not None]
    active = [(q, p) for q, p, c in others if c is None]
    out = []
    if len(view["receipts"]) >= 2:
        out.append("M1")
    if len(cancelled) >= 2 and len({c for _, _, c in cancelled}) == len(cancelled):
        out.append("M2")
    if any(qa == qc and pa > pc for qa, pa in active for qc, pc, _ in cancelled):
        out.append("M3")
    if any(qa == params["invoice_qty"] for qa, _ in active):
        out.append("M4")
    return out


def snapshot(s: Scenario) -> dict:
    snap: dict = {}
    for r in s.records:
        snap.setdefault(r.kind, []).append(r.model_dump(mode="json"))
    return snap


def timing_problems(params: dict, reissue_rule: bool = True, frozen_spacing: bool = True) -> list[str]:
    """Structural rules of the stress timing (SPEC.md): everything before t = 0, receipts on the fixed grid.
    `reissue_rule=False` leaves out the M3 rule and `frozen_spacing=False` the 20-minute spacing for a frozen-format
    case (both only for _params' screen of a frozen-shaped M3 draw, which applies the rules from before the rulings)."""
    out = []
    view = timed_view(params)
    rs, others, b = view["receipts"], view["other_invoices"], view["blocked_at"]
    if not 1 <= len(rs) <= MAX_RECEIPTS:
        out.append(f"{len(rs)} receipts (1 to {MAX_RECEIPTS})")
    if [t for _, t in rs] != [RECEIPT_BASE + RECEIPT_STEP * (i + 1) for i in range(len(rs))]:
        out.append("receipts are not on the grid -2400 + 400 i")
    if b not in (BLOCK_AT, rs[-1][1] - STALE_RECEIPT_OFFSET):
        out.append("blocked_at is neither -160 nor 100 minutes before the last receipt")
    times = [t for _, t in rs] + [b]
    for q, p, c in others:
        times.append(p)
        if c is not None:
            times.append(c)
            if c <= p:
                out.append("an invoice cancelled before it was posted")
        if p < rs[0][1]:
            out.append("an invoice posted before the first receipt")
    if any(t > -MIN_GAP_MINUTES for t in times):
        out.append("an event at or after t = -20")
    if len(times) != len(set(times)):
        out.append("two events at the same time")
    # SPEC.md section 3, rule 5 (user ruling 2026-11-06 on the spacing): every event at least 20 minutes from every
    # other, in both formats (a frozen case is read with its frozen times). Enforced here, so a draw that breaks it is
    # resampled by `generate` and refused by `load` (red-team round 17, finding 3: frozen cases were not checked).
    if is_timed(params) or frozen_spacing:
        ts = sorted(times)
        if any(y - x < MIN_GAP_MINUTES for x, y in zip(ts, ts[1:])):
            out.append(f"two events less than {MIN_GAP_MINUTES} minutes apart")
    if is_timed(params):
        # SPEC.md section 3, rule 3: no other invoice posted after the block (timed format only; frozen stale-receipt
        # cases post them after the block, as the test split does: red-team round 16, finding 3)
        if any(p > b for _, p, _ in others):
            out.append("an other invoice posted after the block")
    # M3 (user ruling 2026-11-06): an active invoice of a cancelled invoice's quantity, posted after that one was
    # posted, is a re-issue and must be posted after the original's cancellation (never while it is still active)
    for qa, pa, ca in (others if reissue_rule else ()):
        if ca is None and any(qc == qa and pc < pa < cc for qc, pc, cc in others if cc is not None):
            out.append("a re-issue posted while its original was still active")
            break
    if any(q <= 0 for q, _ in rs) or any(q <= 0 for q, _, _ in others) or params["invoice_qty"] <= 0:
        out.append("a quantity not positive")
    return out


def scope_problems(params: dict) -> list[str]:
    """A frozen-format case: the frozen causal.check_scope itself. A timed case: its three assertions on the timed
    builder's records at t = 0 (the frozen function builds with fixtures.build and cannot take timed parameters): net
    posted receipts positive; no blocking comparison other than invoice vs receipt fails; a confirmation can only be
    positive or negative (no partial delivery)."""
    from ap_sim.grader import arithmetic as ar
    if not is_timed(params):
        from ap_sim.generator.causal import OutOfScopeCase, check_scope
        from ap_sim.generator.manifests import _tuples
        try:
            check_scope(_tuples(params))
        except OutOfScopeCase as exc:
            return [str(exc)]
        return []
    s = build("scope", 0, **params)
    snap = snapshot(s)
    po_line = next(l["po_line"] for l in snap["invoice_line"] if l["invoice"] == s.current_invoice)
    profile = params.get("policy", "sandbox-v1")
    problems = []
    net = ar.net_receipts(snap, po_line)
    if net <= 0:
        problems.append("zero net posted receipts (A4-DW)")
    other = sorted(set(ar.failing_blocking(snap, s.current_invoice, profile)) - {"invoice_vs_receipt"})
    if other:
        problems.append(f"blocking comparisons fail at the start: {other}")
    gap0, unrecorded = ar.gap(snap, po_line, s.current_invoice, profile), Decimal(params["physical_total"]) - net
    if gap0 > 0 and 0 < unrecorded < gap0:
        problems.append(f"partial confirmation: unrecorded {unrecorded} is between 0 and the gap {gap0}")
    return problems


def admissibility_problems(params: dict) -> list[str]:
    """`causal._params`' load-time admissibility: the engine's own evaluation of the records as they stood when the
    block was evaluated finds invoice vs receipt failing, and no other blocking comparison failing."""
    from ap_sim.engine.engine import Engine
    s = build("probe", 0, **params)
    e = Engine(list(s.records), s.world, s.seed, s.current_invoice)
    results = {c.comparison: (c.effect, c.result) for c in e.store.comparison_results(s.current_invoice)}
    if results.get("invoice_vs_receipt") != ("blocking", "fail"):
        return ["not blocked on receipt coverage when evaluated"]
    if any(eff == "blocking" and res == "fail" for n, (eff, res) in results.items() if n != "invoice_vs_receipt"):
        return ["another blocking comparison fails when evaluated"]
    return []


def derived_variables(s: Scenario) -> dict:
    """The causal variables `causal.derive_variant` reads, measured on the built scenario (never the intent)."""
    from ap_sim.grader import arithmetic as ar
    snap = snapshot(s)
    po_line = next(l["po_line"] for l in snap["invoice_line"] if l["invoice"] == s.current_invoice)
    gap0 = ar.gap(snap, po_line, s.current_invoice, getattr(s, "policy", "sandbox-v1"))
    net = ar.net_receipts(snap, po_line)
    unrec = sum((d.quantity for d in s.world.deliveries if d.po_line == po_line), Decimal(0)) - net
    cls = ("none" if gap0 == 0 else "zero" if unrec == 0 else "equal" if unrec == gap0 else
           "above" if unrec > gap0 else "partial")
    delay = next((k for k, d in DELAYS.items() if d == s.world.confirmation_delay), f"other:{s.world.confirmation_delay}")
    return {"gap": int(gap0), "unrecorded": cls, "unrecorded_quantity": int(unrec), "delay": delay, "reversal": False,
            "net_receipts": int(net), "coverage": int(ar.coverage(snap, po_line, s.current_invoice))}


def label(s: Scenario) -> dict:
    """Expected outcome and discrepancy_type from the frozen grader (g7) on an unacted surfaces/4 episode log, as
    scripts/phase2/build_splits.label does."""
    from ap_sim.grader.grader import GRADER_VERSION, grade
    from ap_sim.interfaces.structured import StructuredSurface
    from ap_sim.interfaces.tools import ToolLayer
    v = grade(s, ToolLayer(s, StructuredSurface(), "surfaces/4").episode_log())
    line = next(r for r in s.records if r.kind == "invoice_line" and r.invoice == s.current_invoice)
    return {"expected": v.expected, "discrepancy_type": v.diagnostics["discrepancy_type"]["required"],
            "grader": GRADER_VERSION, "amount": str(line.amount.value), "currency": line.amount.currency}


# --- the balanced design -------------------------------------------------------------------------------------------
def cells() -> list[tuple[str, ...]]:
    return [(c,) for c in COMPLICATIONS] + list(itertools.combinations(COMPLICATIONS, 2))


DESIGN_SEED = "stress-design/1"


def allocation() -> dict[str, dict[tuple[str, ...], int]]:
    """Cases per (type stratum, combination cell), a controlled rounding of the proportional share
    n_type x n_cell / 100: every entry is the floor or the ceiling of its share, every stratum sums to its size and
    every cell to its total (single 7, pair 12). Floors first; the remaining units go to the entries with the largest
    fractional share, ties in an order drawn from random.Random("stress-design/1"), retried until rows and columns
    close. So the complications are spread over the types in proportion (independent of the outcome)."""
    cs = cells()
    total = {c: SINGLES if len(c) == 1 else PAIRS for c in cs}
    rng = random.Random(DESIGN_SEED)
    for _ in range(10_000):
        share = {(g, c): COUNTS[g] * total[c] / 100 for g in GROUPS for c in cs}
        alloc = {k: int(v) for k, v in share.items()}
        row = {g: COUNTS[g] - sum(alloc[g, c] for c in cs) for g in GROUPS}
        col = {c: total[c] - sum(alloc[g, c] for g in GROUPS) for c in cs}
        for g, c in sorted(share, key=lambda k: (-(share[k] - alloc[k]), rng.random())):
            if share[g, c] > alloc[g, c] and row[g] > 0 and col[c] > 0:
                alloc[g, c] += 1
                row[g] -= 1
                col[c] -= 1
        if not any(row.values()) and not any(col.values()):
            return {g: {c: alloc[g, c] for c in cs} for g in GROUPS}
    raise RuntimeError("no controlled rounding found")


# Slot swaps (user ruling 2026-11-06, M3 re-issues posted after the original's cancellation). A V5 case without M1 or
# M2 has no stale cause left once its M3 original must be cancelled before the block (the re-issue is never posted
# after it), so V5 cannot hold M3 or M3+M4. Each such V5 slot swaps its combination with a slot of a shortfall
# stratum whose case was regenerated under the same ruling; every cell total, so every complication and pair count,
# is unchanged. ((stratum, slot), (stratum, slot)).
DESIGN_SWAPS = ((("V5", 2), ("V4", 11)), (("V5", 12), ("V2", 3)), (("V5", 8), ("V4", 13)), (("V5", 18), ("V1", 14)))


def design() -> dict[str, list[tuple[str, ...]]]:
    """The combination of the j-th accepted case of each type stratum, fixed before any draw (so a redraw or a
    withdrawal never changes the mix): the stratum's allocation, ordered by a low-discrepancy rule (slot i of a cell
    with n slots in the stratum sits at (i + 0.5) / n; ties in cell order), then DESIGN_SWAPS."""
    out = {}
    for g, alloc in allocation().items():
        keyed = [((i + 0.5) / n, ci, c) for ci, (c, n) in enumerate(alloc.items()) for i in range(n)]
        out[g] = [c for *_, c in sorted(keyed)]
    for (ga, ja), (gb, jb) in DESIGN_SWAPS:
        out[ga][ja], out[gb][jb] = out[gb][jb], out[ga][ja]
    return out


def design_allocation() -> dict[str, dict[tuple[str, ...], int]]:
    """Cases per (stratum, cell) as the design (with its swaps) actually assigns them."""
    import collections
    return {g: {c: collections.Counter(d)[c] for c in cells()} for g, d in design().items()}


# --- sampling ------------------------------------------------------------------------------------------------------
def force(group: str, j: int, seed: int) -> dict:
    """The forced variables of the j-th accepted case of a stratum: build_splits.force's cells, with its own stream.
    V5 alternates its stale cause: a receipt posted after the block (only possible with M1) or a cancellation after it."""
    rng = random.Random(f"stress-strata:{seed}")
    if group == "V5":
        return {"gap": 0, "stale": "receipt" if j % 2 == 0 else "cancellation"}
    f = {"gap": 1 if j % 10 == 9 else rng.randint(2, 25)}
    if group == "V1":
        f |= {"delay": ("early", "early", "mid", "mid")[j % 4], "unrecorded": ("equal", "above")[j % 2]}
    elif group == "V2":
        f |= {"delay": ("early", "mid")[j % 2], "unrecorded": "zero"}
    elif group == "V3a":
        f |= {"delay": "late", "unrecorded": ("equal", "above")[j % 2]}
    elif group == "V3b":
        f |= {"delay": "late", "unrecorded": "zero"}
    elif group == "V4":
        f |= {"delay": ("beyond", "never")[j % 2], "unrecorded": rng.choice(["zero", "equal", "above"])}
    return f


def _sample(rng: random.Random, combo: tuple[str, ...], f: dict) -> dict:
    m1, m2, m3, m4 = (c in combo for c in COMPLICATIONS)
    v5 = f["gap"] == 0
    n = rng.choice([2, 3, 4, 5]) if m1 else 1
    receipts = [rng.randint(10, 60) if m1 else rng.randint(30, 120) for _ in range(n)]
    if v5:   # M3 without M2 (user ruling 2026-11-06): the re-issued original is cancelled before the block, so the
        # stale cause can only be a later receipt
        stale_receipt = (f["stale"] == "receipt" or (m3 and not m2)) and m1
    else:
        stale_receipt = n >= 2 and rng.random() < 0.35
    others = [rng.randint(5, 30) for _ in range(rng.choice([0, 0, 1, 1, 2]))]
    if m2:
        timings = [rng.choice(["before_block", "after_block"]) for _ in range(rng.choice([2, 2, 3]))]
    elif m3:
        timings = [rng.choice(["before_block", "after_block"])]
    else:
        base = rng.choice(["none", "none", "before_block", "after_block"])
        timings = [] if base == "none" else [base]
    if v5 and not stale_receipt:          # the stale cause must be a cancellation after the block
        if not timings:
            timings = ["after_block"]
        elif "after_block" not in timings:
            timings[rng.randrange(len(timings))] = "after_block"
    cancelled = [[rng.randint(10, 40), t] for t in timings]
    reissue = None
    if m3:
        k = rng.randrange(len(cancelled))
        if v5 and not stale_receipt and [t for _, t in cancelled].count("after_block") == 1 \
                and cancelled[k][1] == "after_block":
            when = "before_cancel"         # the re-issue precedes the late cancellation, which is the stale cause
        elif cancelled[k][1] == "after_block":
            when = "before_cancel"         # no invoice is posted after the block (SPEC.md)
        else:
            when = rng.choice(["after_cancel", "after_cancel", "before_cancel"])
        reissue = {"of": k, "when": when}
    gap = f["gap"]
    unrecorded = f.get("unrecorded", "none")
    delay = f.get("delay") or rng.choice(["early", "mid", "mid", "late", "beyond", "never"])
    return {"combo": list(combo), "receipts": receipts, "stale_receipt": stale_receipt, "other_active": others,
            "cancelled": cancelled, "reissue": reissue, "lookalike": m4, "gap": gap, "unrecorded": unrecorded,
            "delay": delay, "price": rng.choice(PRICES)}


def _timeline(v: dict, invoice_qty: int, rng: random.Random) -> tuple[list, list, int] | None:
    """Times for every recorded event (SPEC.md): receipts on the fixed grid; the block at -160, or 100 minutes before
    the last receipt (stale receipt); each other invoice posted after the first receipt and before the block, at a
    moment when the receipts then posted covered it and every invoice then active (as a released invoice would have
    been); cancellations after their invoice's posting, before or after the block as sampled; nothing after -20."""
    n = len(v["receipts"])
    rt = [RECEIPT_BASE + RECEIPT_STEP * (i + 1) for i in range(n)]
    blocked_at = rt[-1] - STALE_RECEIPT_OFFSET if v["stale_receipt"] else BLOCK_AT
    # invoices: (qty, role); cancelled ones carry their block timing
    inv = [{"q": q, "role": "other"} for q in v["other_active"]]
    if v["lookalike"]:
        inv.append({"q": invoice_qty, "role": "lookalike"})
    for i, (q, when) in enumerate(v["cancelled"]):
        inv.append({"q": q, "role": "cancelled", "k": i, "when": when})
    order = [i for i, x in enumerate(inv)]
    rng.shuffle(order)
    seq = [("post", i) for i in order]
    for i, x in enumerate(inv):
        if x["role"] == "cancelled" and x["when"] == "before_block":
            p = seq.index(("post", i))
            seq.insert(rng.randint(p + 1, len(seq)), ("cancel", i))
    if v["reissue"]:
        k = v["reissue"]["of"]
        orig = next(i for i, x in enumerate(inv) if x["role"] == "cancelled" and x["k"] == k)
        inv.append({"q": inv[orig]["q"], "role": "reissue", "of": orig})
        r = len(inv) - 1
        p = seq.index(("post", orig))
        if v["reissue"]["when"] == "after_cancel":
            c = seq.index(("cancel", orig))
            seq.insert(rng.randint(c + 1, len(seq)), ("post", r))
        else:
            c = seq.index(("cancel", orig)) if ("cancel", orig) in seq else len(seq)
            seq.insert(rng.randint(p + 1, c), ("post", r))
    # assign times before the block
    window_end = blocked_at - MIN_GAP_MINUTES
    posted, cancelled_at = {}, {}
    t = rt[0]
    for idx, (kind, i) in enumerate(seq):
        left = len(seq) - idx
        room = (window_end - t) // left
        step_hi = max(MIN_GAP_MINUTES, min(150, room))
        t += rng.randrange(MIN_GAP_MINUTES, step_hi + 1, 5) if step_hi > MIN_GAP_MINUTES else MIN_GAP_MINUTES
        if kind == "post":
            q = inv[i]["q"]
            while True:   # wait for the receipts that cover it (receipts so far minus invoices active so far)
                cover = sum(qr for qr, tr in zip(v["receipts"], rt) if tr <= t) - sum(
                    inv[j]["q"] for j, pt in posted.items() if not (j in cancelled_at and cancelled_at[j] <= t))
                if cover >= q:
                    break
                later = [tr for tr in rt if tr > t and tr < window_end]
                if not later:
                    return None
                t = later[0] + MIN_GAP_MINUTES
            posted[i] = t
        else:
            cancelled_at[i] = t
        if t in rt:
            t += 5
            if kind == "post":
                posted[i] = t
            else:
                cancelled_at[i] = t
        if t > window_end:
            return None
    # cancellations after the block, in random order, between the block and -20
    late = [i for i, x in enumerate(inv) if x["role"] == "cancelled" and x["when"] == "after_block"]
    rng.shuffle(late)
    t = blocked_at
    for idx, i in enumerate(late):
        room = (-MIN_GAP_MINUTES - t) // (len(late) - idx)
        if room < MIN_GAP_MINUTES:
            return None
        t += rng.randrange(MIN_GAP_MINUTES, min(room, 90) + 1, 5)
        if t in rt:
            t += 5
        if t <= posted[i]:
            return None
        cancelled_at[i] = t
    receipts = [[q, tr] for q, tr in zip(v["receipts"], rt)]
    others = sorted(([inv[i]["q"], posted[i], cancelled_at.get(i)] for i in range(len(inv))), key=lambda x: x[1])
    return receipts, others, blocked_at


def _params(v: dict, rng: random.Random) -> tuple[dict | None, list[str]]:
    receipts = list(v["receipts"])
    reissue_q = v["cancelled"][v["reissue"]["of"]][0] if v["reissue"] else 0
    active_others = sum(v["other_active"]) + reissue_q
    gap = v["gap"]
    if v["lookalike"]:     # the look-alike has the invoice's quantity: 2 x invoice = net - others + gap
        if (sum(receipts) - active_others + gap) % 2:
            receipts[-1] += 1
        invoice_qty = (sum(receipts) - active_others + gap) // 2
    else:
        invoice_qty = sum(receipts) - active_others + gap
    v = {**v, "receipts": receipts}
    net = sum(receipts)
    if invoice_qty - gap <= 0 or invoice_qty <= 0:
        return None, ["coverage or invoice quantity not positive"]
    extra = {"none": 0, "zero": 0, "equal": gap, "above": gap + rng.randint(1, 10)}[v["unrecorded"]]
    invoiced = (sum(v["other_active"]) + sum(q for q, _ in v["cancelled"]) + reissue_q
                + (invoice_qty if v["lookalike"] else 0) + invoice_qty)
    po_qty = max(net + extra, invoiced) + rng.randint(10, 60)
    timed = not (len(receipts) <= 3 and len(v["cancelled"]) <= 1)
    if not timed:                                              # fixtures.build expresses it
        params, problems = _frozen_params(v, invoice_qty, po_qty, net + extra, reissue_q, rng)
        if params is None:
            return None, problems
        if v["reissue"]:
            # M3 (user ruling 2026-11-06): the frozen timing cannot post a re-issue after its original's cancellation
            # (the shared cancellation follows every posting), so an M3 draw is always timed. A frozen-shaped M3 draw is
            # first screened by the rules it met before the ruling, so every draw the earlier build rejected stays
            # rejected and every other case keeps its draw; it is then timed from the same sampled variables.
            old = timing_problems(params, reissue_rule=False, frozen_spacing=False) or admissibility_problems(params) or \
                scope_problems(params)
            if not old and sorted(complications(params)) != sorted(v["combo"]):
                old = ["content differs"]
            if old:
                return None, old
            timed = True
    if timed:
        tl = _timeline(v, invoice_qty, rng)
        if tl is None:
            return None, ["no timeline fits"]
        rts, others, blocked_at = tl
        params = dict(po_qty=po_qty, price=v["price"], invoice_qty=invoice_qty, blocked_at=blocked_at,
                      physical_total=net + extra, delay=DELAYS[v["delay"]], timed_receipts=rts,
                      timed_other_invoices=others)
    problems = timing_problems(params)
    if not problems:
        problems = admissibility_problems(params) or scope_problems(params)
    if not problems and sorted(complications(params)) != sorted(v["combo"]):
        problems = [f"content shows {complications(params)}, not {v['combo']}"]
    return (None, problems) if problems else (params, [])


def _frozen_params(v: dict, invoice_qty: int, po_qty: int, physical_total: int, reissue_q: int,
                   rng: random.Random) -> tuple[dict | None, list[str]]:
    """The frozen parameters, as causal._params writes them: receipts on the grid, the block at -160 or 100 minutes
    before the last receipt, the one cancellation (if any) at -80 (after the block) or 30 minutes before the block,
    and it must come after its invoice's posting (the last receipt + 120). The active other invoices (ordinary ones,
    the look-alike, the re-issue) follow the cancelled one in a random order."""
    rt = [RECEIPT_BASE + RECEIPT_STEP * (i + 1) for i in range(len(v["receipts"]))]
    blocked_at = rt[-1] - STALE_RECEIPT_OFFSET if v["stale_receipt"] else BLOCK_AT
    cancelled_at = None
    if v["cancelled"]:
        cancelled_at = -80 if v["cancelled"][0][1] == "after_block" else blocked_at - 30
        if cancelled_at <= rt[-1] + INVOICE_STEP:
            return None, ["cancellation before the invoice was posted"]
    actives = list(v["other_active"]) + ([invoice_qty] if v["lookalike"] else []) + ([reissue_q] if v["reissue"] else [])
    rng.shuffle(actives)
    return dict(po_qty=po_qty, price=v["price"], receipts=list(v["receipts"]), other_invoices=actives,
                invoice_qty=invoice_qty, physical_total=physical_total, delay=DELAYS[v["delay"]],
                cancelled_invoices=[q for q, _ in v["cancelled"]], blocked_at=blocked_at, cancelled_at=cancelled_at,
                reversals=[]), []


@dataclasses.dataclass(frozen=True)
class StressCase:
    case_id: str
    seed: int
    group: str                 # the stratum being filled (the derived type must equal it)
    combo: tuple[str, ...]     # the design's combination for this slot; equals the content's
    variables: dict            # the sampled variables (intent; provenance only)
    params: dict
    resamples: int

    def scenario(self) -> Scenario:
        return build(self.case_id, self.seed, **self.params)


def generate(case_id: str, seed: int, group: str, j: int, combo: tuple[str, ...], max_tries: int = 500) -> StressCase:
    rng = random.Random(f"{RNG_STREAM}:{seed}")
    f = force(group, j, seed)
    for attempt in range(max_tries):
        v = _sample(rng, combo, f)
        params, _ = _params(v, rng)
        if params is not None:
            return StressCase(case_id, seed, group, tuple(combo), v, params, attempt)
    raise ValueError(f"no admissible case for seed {seed}, {group}, {combo} in {max_tries} tries")


def derive(s: Scenario) -> dict:
    """Derived type and labels of a built case: causal.derive_variant on the measured variables, g7's labels."""
    dv = derived_variables(s)
    return {"variant": derive_variant(dv), "derived_variables": dv, **label(s)}


# --- keys ----------------------------------------------------------------------------------------------------------
def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def visible_content(s: Scenario) -> dict:
    """What the agent can see of a case at the start, with record ids removed (ids come from the seed), comparable
    across the frozen and the stress parameter formats: the PO line, the invoice under review, and every receipt and
    other invoice on the PO line with its times and state, in time order."""
    snap = snapshot(s)
    cur = next(i for i in snap["invoice"] if i["id"] == s.current_invoice)
    line = next(l for l in snap["invoice_line"] if l["invoice"] == s.current_invoice)
    qty = {l["invoice"]: l["quantity"]["value"] for l in snap["invoice_line"]}
    (pl,) = snap["po_line"]
    return {"po_line": [pl["ordered_quantity"]["value"], pl["unit_price"]["value"]],
            "invoice": [line["quantity"]["value"], cur["posted_at"], cur["evaluated_at"]],
            "receipts": sorted([r["posted_at"], r["movement"], r["quantity"]["value"]] for r in snap.get("receipt", [])),
            "other_invoices": sorted([i["posted_at"], i["state"], i.get("cancelled_at"), qty[i["id"]]]
                                     for i in snap["invoice"] if i["id"] != s.current_invoice),
            "policy": getattr(s, "policy", "sandbox-v1")}


def visible_keys(seed: int, s: Scenario) -> dict:
    """exact: seed and visible content and hidden world; near: visible content and hidden world; group: visible content
    alone (the agent's view at the start). Computed from a BUILT scenario, so every parameter format compares."""
    vis = visible_content(s)
    hidden = {"delivered_total": str(sum((d.quantity for d in s.world.deliveries), Decimal(0))),
              "delay": s.world.confirmation_delay}
    return {"exact": _sha([seed, vis, hidden]), "near": _sha([vis, hidden]), "group": _sha(vis)}


def figures(s: Scenario) -> dict:
    """Business figures of a built case, the same whatever its parameter format (from the records)."""
    snap = snapshot(s)
    qty = {l["invoice"]: str(Decimal(l["quantity"]["value"])) for l in snap["invoice_line"]}
    others = [i for i in snap["invoice"] if i["id"] != s.current_invoice]
    rec = [str(Decimal(r["quantity"]["value"]) * (1 if r["movement"] == "receipt" else -1))
           for r in snap.get("receipt", [])]
    (pl,) = snap["po_line"]
    return {"invoice_qty": qty[s.current_invoice], "receipts": sorted(rec, key=Decimal),
            "active_others": sorted((qty[i["id"]] for i in others if i["state"] != "cancelled"), key=Decimal),
            "cancelled": sorted((qty[i["id"]] for i in others if i["state"] == "cancelled"), key=Decimal),
            "po_qty": str(Decimal(pl["ordered_quantity"]["value"])), "price": str(Decimal(pl["unit_price"]["value"]))}


def figure_key(s: Scenario) -> str:
    """Format-independent figures: the invoice, PO, receipt, active and cancelled quantities and the price (red-team
    round 16, finding 1: a timed re-encoding of a frozen case has other parameter keys but the same figures)."""
    return _sha(figures(s))


def existing_cases():
    """(source, seed, scenario) for every existing case: the nine splits (sft, rl, dev, regression, test,
    test_reserve, v6, policy_shift, forgetting_l1) and the earlier sources (the v1 frozen sets, the fresh 1,000 as
    stored when runs/phase0_5/fresh_1000.json is present, and as regenerated by causal, the 55 dev cases, the boundary
    cases). Built by program; nothing is printed (the reserve's cases are never shown)."""
    from ap_sim.generator import causal
    from ap_sim.generator.fixtures import build as frozen_build
    from ap_sim.generator.manifests import _tuples
    from ap_sim.generator.splits import DIR, SPLITS
    paths = [(sp, DIR / f"{sp}.json") for sp in SPLITS]
    paths.append(("forgetting_l1", ROOT / "manifests" / "forgetting_l1" / "forgetting_l1.json"))
    paths += [(f"frozen:{f.stem}", f) for f in sorted((ROOT / "manifests").glob("frozen*/*.json"))
              if f.name != "FREEZE.json"]
    fresh = ROOT / "runs" / "phase0_5" / "fresh_1000.json"
    if fresh.exists():
        paths.append(("fresh_1000", fresh))
    for src, p in paths:
        for e in json.loads(p.read_text())["entries"]:
            prm = _tuples(e["params"])
            yield src, e["seed"], frozen_build(e.get("case_id", "x"), e["seed"], **prm), prm
    for e in json.loads((ROOT / "manifests" / "dev_cases.json").read_text())["cases"]:
        prm = _tuples(e["params"])
        yield "dev_cases", e["seed"], frozen_build(e["id"], e["seed"], **prm), prm
    for seed in range(9_100_000, 9_101_000):
        c = causal.generate(f"F-{seed}", seed)
        yield "fresh_1000_regenerated", seed, c.scenario(), c.params
    for seed in (4242, 5100, 5200, 5300):
        for c in causal.boundary_cases(seed):
            yield "boundary", c.seed, c.scenario(), c.params


_INDEX: dict | None = None


def existing_index() -> dict:
    """{family: {key: set of sources}} over existing_cases(): seed; param near and group (generator/splits.keys);
    visible near and group (visible_keys); figures (figure_key). Cached per process."""
    global _INDEX
    if _INDEX is None:
        from ap_sim.generator.splits import keys
        idx = {f: {} for f in ("seed", "param near", "param group", "visible near", "visible group", "figures")}
        counts = {}
        for src, seed, s, prm in existing_cases():
            counts[src] = counts.get(src, 0) + 1
            pk, vk = keys(seed, prm), visible_keys(seed, s)
            for fam, k in (("seed", seed), ("param near", pk["near"]), ("param group", pk["group"]),
                           ("visible near", vk["near"]), ("visible group", vk["group"]), ("figures", figure_key(s))):
                idx[fam].setdefault(k, set()).add(src)
        _INDEX = {"index": idx, "cases_by_source": counts}
    return _INDEX


def leakage_matches(seed: int, params: dict, s: Scenario | None = None) -> list[str]:
    """Where a stress case's seed, parameter keys, visible keys or figures occur among the existing cases:
    ['<family>: <source>', ...] (source names only)."""
    from ap_sim.generator.splits import keys
    s = s or build("x", seed, **params)
    pk, vk = keys(seed, params), visible_keys(seed, s)
    idx = existing_index()["index"]
    mine = {"seed": seed, "param near": pk["near"], "param group": pk["group"], "visible near": vk["near"],
            "visible group": vk["group"], "figures": figure_key(s)}
    return [f"{fam}: {src}" for fam, k in mine.items() for src in sorted(idx[fam].get(k, ()))]


def param_keys(seed: int, params: dict) -> dict:
    """The frozen key definitions (generator/splits.keys) applied to the stress parameters."""
    from ap_sim.generator.splits import keys
    return keys(seed, params)


def params_sha(params: dict) -> str:
    return _sha(params)


# --- loader --------------------------------------------------------------------------------------------------------
class StressRefused(ValueError):
    """The loader was asked for something that is not a stress case of this manifest."""


def load(path: str | pathlib.Path = MANIFEST, *, require_frozen: bool = False) -> dict[str, Callable[[], Scenario]]:
    """case_id -> scenario factory, in manifest order. Refuses: a manifest of another split; a parameter sha256
    mismatch; a case whose seed, parameter near or group key, visible near or group key, or format-independent
    figures match any existing case (existing_cases: the nine splits and the earlier sources); a case whose
    complications by content differ from its stored ones or number other than 1 or 2; a case that fails the timing,
    scope or admissibility rules; stored variant, expected or discrepancy_type that differ from a recompute. With require_frozen, the manifest's sha256 must equal manifests/stress/FREEZE.json's."""
    path = pathlib.Path(path)
    if require_frozen:
        freeze = path.parent / "FREEZE.json"
        if not freeze.exists():
            raise StressRefused("no FREEZE.json: the stress set is not frozen")
        want = json.loads(freeze.read_text())["files"].get(path.name)
        if want != hashlib.sha256(path.read_bytes()).hexdigest():
            raise StressRefused(f"{path.name} is not the frozen manifest")
    data = json.loads(path.read_text())
    if data.get("split") != SPLIT:
        raise StressRefused(f"the stress loader was given the {data.get('split')!r} manifest")
    out = {}
    for e in data["entries"]:
        cid, params = e["case_id"], e["params"]
        if e.get("split") != SPLIT:
            raise StressRefused(f"{cid} belongs to {e.get('split')!r}")
        if params_sha(params) != e["params_sha256"]:
            raise StressRefused(f"{cid}: parameters do not match their sha256")
        s = build(cid, e["seed"], **params)
        hits = leakage_matches(e["seed"], params, s)
        if hits:
            raise StressRefused(f"{cid}: matches an existing case ({'; '.join(hits)})")
        content = complications(params)
        if sorted(content) != sorted(e["complications"]) or not 1 <= len(content) <= 2:
            raise StressRefused(f"{cid}: complications by content {content}, stored {e['complications']}")
        why = timing_problems(params) or scope_problems(params) or admissibility_problems(params)
        if why:
            raise StressRefused(f"{cid}: {why}")
        d = derive(s)        # red-team round 16, finding 7: the stored labels against a recompute
        for k in ("variant", "expected", "discrepancy_type"):
            if e.get(k) != d[k]:
                raise StressRefused(f"{cid}: stored {k} {e.get(k)!r}, recomputed {d[k]!r}")
        out[cid] = (lambda cid=cid, seed=e["seed"], p=params: build(cid, seed, **p))
    return out
