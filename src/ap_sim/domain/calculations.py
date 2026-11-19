"""Coverage arithmetic (POLICY A4).

available coverage = (posted receipts - reversals) - other active invoiced quantity
Other active invoiced quantity excludes the current invoice and every cancelled invoice.
All current-invoice lines on the same PO line are evaluated jointly.
"""
from __future__ import annotations

from decimal import Decimal

from ap_sim.domain.store import RecordStore


def net_posted(store: RecordStore, po_line: str) -> Decimal:
    total = Decimal(0)
    for r in store.receipts(po_line):
        total += r.quantity.value if r.movement == "receipt" else -r.quantity.value
    return total


def active_invoice_ids(store: RecordStore) -> set[str]:
    return {i.id for i in store.invoices() if i.state != "cancelled"}


def other_active_invoiced(store: RecordStore, po_line: str, current_invoice: str) -> Decimal:
    active = active_invoice_ids(store) - {current_invoice}
    return sum((l.quantity.value for l in store.invoice_lines(po_line=po_line) if l.invoice in active), Decimal(0))


def joint_invoice_quantity(store: RecordStore, invoice_id: str, po_line: str) -> Decimal:
    return sum((l.quantity.value for l in store.invoice_lines(invoice_id, po_line)), Decimal(0))


def available_coverage(store: RecordStore, po_line: str, current_invoice: str) -> Decimal:
    return net_posted(store, po_line) - other_active_invoiced(store, po_line, current_invoice)


def coverage_gap(store: RecordStore, po_line: str, current_invoice: str) -> Decimal:
    gap = joint_invoice_quantity(store, current_invoice, po_line) - available_coverage(store, po_line, current_invoice)
    return max(gap, Decimal(0))
