"""Hidden layer of the three-layer world. Held only by the event simulator and the grader.

Never rendered, never passed to agent-facing code. Physical deliveries can exist without a posted
receipt; the warehouse answers a confirmation request after a fixed delay, or never.
"""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class PhysicalDelivery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    po_line: str
    at: int
    quantity: Decimal


class HiddenWorld(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    deliveries: tuple[PhysicalDelivery, ...]
    confirmation_delay: int | None  # minutes after the request; None = the warehouse never answers

    def delivered_by(self, po_line: str, t: int) -> Decimal:
        return sum((d.quantity for d in self.deliveries if d.po_line == po_line and d.at <= t), Decimal(0))
