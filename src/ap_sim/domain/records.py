"""Recorded-layer business records. Validated, with stable ids and versions.

Money and quantities are Decimal with explicit units; floats are rejected everywhere.
Times are virtual business minutes from episode start (negative = before the episode).
"""
from __future__ import annotations

from decimal import Decimal
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CENT = Decimal("0.01")


def _no_float(v):
    if isinstance(v, float):
        raise TypeError("floats are not allowed in the accounting path; use Decimal or str")
    return v


class Qty(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: Decimal
    unit: Literal["EA"] = "EA"

    _reject_float = field_validator("value", mode="before")(_no_float)


class Money(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: Decimal
    currency: Literal["EUR"] = "EUR"

    _reject_float = field_validator("value", mode="before")(_no_float)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: ClassVar[str] = "record"
    id: str
    version: int = 1


InvoiceState = Literal["received_unposted", "posted_blocked", "posted_released", "cancelled"]
Comparison = Literal["price", "quantity", "receipt_vs_po", "invoice_vs_receipt", "invoice_no_receipt",
                     "invoice_vs_po_quantity", "invoice_vs_po_amount", "closed_po"]
FollowUpType = Literal["shortfall_correction", "confirmation_pending", "buyer_review"]


class PurchaseOrder(Record):
    kind: ClassVar[str] = "po"
    supplier: str
    status: Literal["open", "closed"]


class POLine(Record):
    kind: ClassVar[str] = "po_line"
    po: str
    item: str
    ordered_quantity: Qty
    unit_price: Money
    line_amount: Money

    @model_validator(mode="after")
    def _amount(self):
        if self.line_amount.value != (self.ordered_quantity.value * self.unit_price.value).quantize(CENT):
            raise ValueError("line_amount must equal ordered_quantity x unit_price")
        return self


class Receipt(Record):
    kind: ClassVar[str] = "receipt"
    po_line: str
    posted_at: int
    movement: Literal["receipt", "reversal"]
    quantity: Qty
    reverses: str | None = None

    @model_validator(mode="after")
    def _reversal_ref(self):
        if (self.movement == "reversal") != (self.reverses is not None):
            raise ValueError("a reversal must reference the receipt it reverses, and only a reversal may")
        if self.quantity.value <= 0:
            raise ValueError("quantity must be positive; the movement carries the sign")
        return self


class Invoice(Record):
    kind: ClassVar[str] = "invoice"
    supplier: str
    po: str
    state: InvoiceState
    posted_at: int | None = None
    cancelled_at: int | None = None
    eligibility: Literal["eligible", "not_eligible"] = "not_eligible"
    evaluated_at: int | None = None
    evaluated_inputs: str = ""  # digest of the record versions the last evaluation used


class InvoiceLine(Record):
    kind: ClassVar[str] = "invoice_line"
    invoice: str
    po_line: str
    quantity: Qty
    unit_price: Money
    amount: Money

    @model_validator(mode="after")
    def _amount(self):
        if self.amount.value != (self.quantity.value * self.unit_price.value).quantize(CENT):
            raise ValueError("amount must equal quantity x unit_price")
        return self


class ComparisonResult(Record):
    kind: ClassVar[str] = "comparison_result"
    invoice: str
    comparison: Comparison
    effect: Literal["blocking", "informational"]
    result: Literal["pass", "fail", "flag", "not_applicable"]
    evaluated_at: int


class ConfirmationRequest(Record):
    kind: ClassVar[str] = "confirmation_request"
    invoice: str
    po_line: str
    requested_at: int
    status: Literal["open", "answered"] = "open"
    answered_at: int | None = None
    delivered_total: Qty | None = None


class PostingRequest(Record):
    kind: ClassVar[str] = "posting_request"
    invoice: str
    po_line: str
    requested_at: int
    quantity: Qty
    confirmation: str
    status: Literal["requested", "posted", "rejected"] = "requested"
    receipt: str | None = None


class FollowUp(Record):
    kind: ClassVar[str] = "follow_up"
    invoice: str
    created_at: int
    type: FollowUpType
    owner: Literal["ap", "buyer"]
    status: Literal["open", "closed"] = "open"


RECORD_TYPES: dict[str, type[Record]] = {
    cls.kind: cls
    for cls in (PurchaseOrder, POLine, Receipt, Invoice, InvoiceLine, ComparisonResult,
                ConfirmationRequest, PostingRequest, FollowUp)
}
