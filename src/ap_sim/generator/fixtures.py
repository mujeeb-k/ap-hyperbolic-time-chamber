"""Scenario fixtures: the two branches with a few value variations (not the generator).

Inputs are the reference and golden cases of POLICY D1 and E1: V1/V2 (a counterfactual pair sharing a seed and
an identical recorded state), G4 (delivered but unrecorded), G5 and G3 (not delivered).
Initial record ids come from a dedicated seeded stream, so twins share ids.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from ap_sim.domain.records import Invoice, InvoiceLine, Money, POLine, PurchaseOrder, Qty, Receipt, Record
from ap_sim.events.clock import HORIZON
from ap_sim.events.world import HiddenWorld, PhysicalDelivery

CENT = Decimal("0.01")


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    records: tuple[Record, ...]
    world: HiddenWorld
    current_invoice: str
    horizon: int = HORIZON
    policy: str = "sandbox-v1"   # the active policy profile; sandbox-v2 only in the policy-shift set


def build(name: str, seed: int, *, po_qty: int, price: str, receipts: list[int], other_invoices: list[int],
          invoice_qty: int, physical_total: int, delay: int | None, cancelled_invoices: tuple[int, ...] = (),
          blocked_at: int = -160, cancelled_at: int | None = None,
          reversals: tuple[tuple[int, int], ...] = (), policy: str = "sandbox-v1") -> Scenario:
    """`reversals` (variant V6): (receipt index, quantity) pairs; each posts a reversal movement of that
    receipt 200 minutes after it, correcting an over-posting, so the physical delivery is the receipt minus the
    reversal. Empty by default: every existing fixture and manifest case builds exactly as before."""
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
    t = -2400  # history before the episode, in business minutes
    posted = []
    for q in receipts:
        t += 400
        posted.append(Receipt(id=rid("GR"), po_line=pl.id, posted_at=t, movement="receipt", quantity=Qty(value=Decimal(q))))
        records.append(posted[-1])
    reversed_by = {}
    for i, q in reversals:
        r = posted[i]
        reversed_by[i] = reversed_by.get(i, 0) + q
        records.append(Receipt(id=rid("GR"), po_line=pl.id, posted_at=r.posted_at + 200, movement="reversal",
                               quantity=Qty(value=Decimal(q)), reverses=r.id))
    for q, cancelled in [(q, True) for q in cancelled_invoices] + [(q, False) for q in other_invoices]:
        t += 120
        inv = Invoice(id=rid("INV"), supplier="Supplier A", po=po.id, posted_at=t, eligibility="eligible", evaluated_at=t,
                      **({"state": "cancelled", "cancelled_at": cancelled_at, "version": 2} if cancelled
                         else {"state": "posted_released"}))  # cancelled: posted (v1), then cancelled (v2)
        records += [inv, InvoiceLine(id=f"{inv.id}-1", invoice=inv.id, po_line=pl.id, quantity=Qty(value=Decimal(q)),
                                     unit_price=Money(value=unit), amount=Money(value=(q * unit).quantize(CENT)))]
    current = Invoice(id=rid("INV"), supplier="Supplier A", po=po.id, state="posted_blocked", posted_at=blocked_at,
                      evaluated_at=blocked_at)
    records += [current, InvoiceLine(id=f"{current.id}-1", invoice=current.id, po_line=pl.id,
                                     quantity=Qty(value=Decimal(invoice_qty)), unit_price=Money(value=unit),
                                     amount=Money(value=(invoice_qty * unit).quantize(CENT)))]
    recorded = sum(receipts) - sum(q for _, q in reversals)
    deliveries = [PhysicalDelivery(po_line=pl.id, at=-2400 + 400 * (i + 1), quantity=Decimal(q - reversed_by.get(i, 0)))
                  for i, q in enumerate(receipts) if q - reversed_by.get(i, 0) > 0]
    if physical_total > recorded:  # delivered but not yet recorded (posting lag)
        deliveries.append(PhysicalDelivery(po_line=pl.id, at=-300, quantity=Decimal(physical_total - recorded)))
    world = HiddenWorld(deliveries=tuple(deliveries), confirmation_delay=delay)
    return Scenario(name=name, seed=seed, records=tuple(records), world=world, current_invoice=current.id,
                    policy=policy)


# POLICY D1 / E1 inputs (fixed 2026-10-21).
SCENARIOS = {
    "V1": lambda: build("V1", 101, po_qty=100, price="20.00", receipts=[40, 30], other_invoices=[50],
                        invoice_qty=30, physical_total=80, delay=480),
    "V2": lambda: build("V2", 101, po_qty=100, price="20.00", receipts=[40, 30], other_invoices=[50],
                        invoice_qty=30, physical_total=70, delay=480),
    "G4": lambda: build("G4", 202, po_qty=80, price="31.25", receipts=[30, 25], other_invoices=[20],
                        invoice_qty=40, physical_total=60, delay=960),
    "G5": lambda: build("G5", 303, po_qty=150, price="9.99", receipts=[90], other_invoices=[60],
                        invoice_qty=45, physical_total=90, delay=1440),
    "G3": lambda: build("G3", 404, po_qty=250, price="4.80", receipts=[100, 75], other_invoices=[120],
                        invoice_qty=56, physical_total=175, delay=240),
}
# Development-only diagnostic cases: the short no-gap goldens G1, G2, G7 (POLICY E1), for the integration
# diagnostics run before training. Never in held-out evaluation; their results establish integration only.
# Stale-block causes (POLICY E1): G1 and G2 were blocked between their two receipts; G7 was blocked while its
# 40-unit invoice was active, which was cancelled later. The engine shows the comparisons of that earlier
# evaluation, so release requires an explicit reprocess first.
DEV_ONLY = {
    "G1": lambda: build("G1", 505, po_qty=120, price="12.50", receipts=[80, 40], other_invoices=[],
                        invoice_qty=120, physical_total=120, delay=480, blocked_at=-1800),
    "G2": lambda: build("G2", 606, po_qty=200, price="7.35", receipts=[25, 35], other_invoices=[],
                        invoice_qty=45, physical_total=60, delay=480, blocked_at=-1800),
    "G7": lambda: build("G7", 707, po_qty=90, price="18.40", receipts=[50, 25], other_invoices=[20],
                        invoice_qty=50, physical_total=75, delay=480, cancelled_invoices=(40,), cancelled_at=-80),
}
def _load_dev_cases() -> dict[str, dict]:
    """Designated development cases (`manifests/dev_cases.json`): variations of the implemented workflows
    for the SFT warm-up (dev_train) and the baseline/post-SFT evaluation (dev_eval). Never validation or test."""
    import json
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[3] / "manifests" / "dev_cases.json"
    if not path.exists():
        return {}
    out = {}
    for c in json.loads(path.read_text())["cases"]:
        p = dict(c["params"])
        p["cancelled_invoices"] = tuple(p.get("cancelled_invoices", ()))
        out[c["id"]] = {"split": c["split"], "workflow": c["workflow"],
                        "factory": (lambda c=c, p=p: build(c["id"], c["seed"], **p))}
    return out


DEV_CASES = _load_dev_cases()
DEV_TRAIN = {k: v["factory"] for k, v in DEV_CASES.items() if v["split"] == "dev_train"}
DEV_EVAL = {k: v["factory"] for k, v in DEV_CASES.items() if v["split"] == "dev_eval"}
WITH_DEV_ONLY = {**SCENARIOS, **DEV_ONLY, **DEV_TRAIN, **DEV_EVAL}  # integration diagnostics, SFT warm-up, dev evaluation
DELIVERED = ("V1", "G4")
NOT_DELIVERED = ("V2", "G5", "G3")
