"""Sandbox policy v1 (POLICY.md section A). Engine-side evaluation of the applicable comparisons.

Every limit fails only when strictly exceeded; equality passes (POLICY A, boundary note).
A2 (quantity tolerance) is disabled in the default profile and is therefore not listed.
"""
from __future__ import annotations

from decimal import Decimal

from ap_sim.domain import calculations as calc
from ap_sim.domain.store import RecordStore

POLICY_VERSION = "sandbox-v1"

# Policy profiles. sandbox-v1 is the default everywhere and the baseline. sandbox-v2 exists only for the policy-shift
# set. Its values were fixed when that set was frozen and were not changed afterwards. It enables A2, a quantity
# tolerance on the invoice-vs-receipt comparison (POLICY A2). The excess of the invoice quantity over available
# coverage passes when excess <= 5% of available coverage AND excess x PO unit price <= EUR 500. Either exceeded
# fails. Equality passes. A profile is visible to the agent on the Active policy page (surfaces/3), with the time it
# took effect.
# sandbox-v2 takes effect this many business minutes after the block's evaluation (POLICY I2; app/session.py).
SANDBOX_V2_EFFECT_MINUTES = 100

PROFILES = {
    "sandbox-v1": {"a2": None},
    "sandbox-v2": {"a2": {"pct": Decimal("5"), "abs": Decimal("500")}},
}

PRICE_PCT = Decimal("10")
PRICE_ABS = Decimal("1000")
RECEIPT_VS_PO_PCT = Decimal("5")
INVOICE_VS_PO_QTY_PCT = Decimal("10")
INVOICE_VS_PO_AMOUNT_PCT = Decimal("10")
INVOICE_VS_PO_AMOUNT_ABS = Decimal("1000")

# Catalog order and effect of each comparison.
EFFECT = {
    "price": "blocking",
    "receipt_vs_po": "informational",
    "invoice_vs_receipt": "blocking",
    "invoice_no_receipt": "blocking",
    "invoice_vs_po_quantity": "blocking",
    "invoice_vs_po_amount": "blocking",
    "closed_po": "blocking",
}
ORDER = ["price", "quantity", "receipt_vs_po", "invoice_vs_receipt", "invoice_no_receipt",
         "invoice_vs_po_quantity", "invoice_vs_po_amount", "closed_po"]


def _pct(limit_pct: Decimal) -> Decimal:
    return 1 + limit_pct / 100


def a2_passes(excess: Decimal, cover: Decimal, po_unit_price: Decimal, profile: str) -> bool:
    """A2 (sandbox-v2 only): may the invoice exceed available coverage by `excess`? False when A2 is disabled."""
    a2 = PROFILES[profile]["a2"]
    if a2 is None or excess <= 0:
        return excess <= 0
    return excess <= cover * a2["pct"] / 100 and excess * po_unit_price <= a2["abs"]


def evaluate(store: RecordStore, invoice_id: str, profile: str = POLICY_VERSION) -> list[tuple[str, str, str]]:
    """Return [(comparison, effect, result)] for every applicable comparison, in catalog order."""
    invoice = store.get(invoice_id)
    po = store.po(invoice.po)
    lines = store.invoice_lines(invoice_id)
    po_lines = {l.id: l for l in store.po_lines(po.id)}
    results: dict[str, str] = {}

    rank = {"pass": 0, "flag": 1, "fail": 2}

    def worst(name: str, outcome: str) -> None:
        if name not in results or rank[outcome] > rank[results[name]]:
            results[name] = outcome

    for po_line_id in sorted({l.po_line for l in lines}):
        pl = po_lines[po_line_id]
        # A1 price, per invoice line on this PO line.
        for l in (x for x in lines if x.po_line == po_line_id):
            diff = abs(l.unit_price.value - pl.unit_price.value)
            pct = diff / pl.unit_price.value * 100
            effect = diff * l.quantity.value
            worst("price", "fail" if pct > PRICE_PCT or effect > PRICE_ABS else "pass")
        net = calc.net_posted(store, po_line_id)
        # A3 receipt vs PO (informational).
        worst("receipt_vs_po", "flag" if net > pl.ordered_quantity.value * _pct(RECEIPT_VS_PO_PCT) else "pass")
        # A4 invoice vs receipt (DQ-style), or the DW-style rule when no goods-receipt quantity exists.
        joint = calc.joint_invoice_quantity(store, invoice_id, po_line_id)
        if net == 0:
            worst("invoice_no_receipt", "fail" if joint > 0 else "pass")
        else:
            cover = calc.available_coverage(store, po_line_id, invoice_id)
            ok = joint <= cover or a2_passes(joint - cover, cover, pl.unit_price.value, profile)
            worst("invoice_vs_receipt", "pass" if ok else "fail")
        # A5 cumulative active invoiced quantity (including the current invoice) vs PO quantity.
        cumulative = calc.other_active_invoiced(store, po_line_id, invoice_id) + joint
        worst("invoice_vs_po_quantity",
              "fail" if cumulative > pl.ordered_quantity.value * _pct(INVOICE_VS_PO_QTY_PCT) else "pass")

    # A6 cumulative active invoiced amount vs PO amount (PO level).
    active = calc.active_invoice_ids(store)
    po_line_ids = set(po_lines)
    cum_amount = sum((l.amount.value for l in store.invoice_lines()
                      if l.invoice in active and l.po_line in po_line_ids), Decimal(0))
    po_amount = sum((l.line_amount.value for l in po_lines.values()), Decimal(0))
    limit = min(po_amount * _pct(INVOICE_VS_PO_AMOUNT_PCT), po_amount + INVOICE_VS_PO_AMOUNT_ABS)
    worst("invoice_vs_po_amount", "fail" if cum_amount > limit else "pass")
    # A7 closed PO.
    if po.status == "closed":
        worst("closed_po", "fail")

    return [(name, EFFECT[name], results[name]) for name in ORDER if name in results]


def eligible(evaluation: list[tuple[str, str, str]]) -> bool:
    return all(result == "pass" for _, effect, result in evaluation if effect == "blocking")
