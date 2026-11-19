"""The single authoritative business engine.

Every business mutation passes through `execute` (agent and role operations) or a scheduled event
(warehouse response, receiving-role posting). Each transition checks the actor's authority and its
prerequisites, applies its effects atomically, and is logged; a rejection changes no record.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from decimal import Decimal

from ap_sim.domain import calculations as calc
from ap_sim.domain import policy
from ap_sim.domain.records import (ComparisonResult, ConfirmationRequest, FollowUp, Invoice, PostingRequest,
                                   Qty, Receipt, Record)
from ap_sim.domain.store import RecordStore
from ap_sim.events.scheduler import Scheduler
from ap_sim.events.world import HiddenWorld

PERMISSIONS = {
    "ap_agent": {"request_confirmation", "request_receipt_posting", "reprocess", "release", "keep_blocked"},
    "receiving": {"post_receipt"},
}
FOLLOW_UP_OWNER = {"shortfall_correction": "ap", "confirmation_pending": "ap", "buyer_review": "buyer"}
_WHOLE = re.compile(r"^[1-9][0-9]{0,8}$")


@dataclass
class Attempt:
    seq: int
    at: int
    actor: str
    op: str
    args: dict
    accepted: bool = False
    message: str = ""
    changed: list[str] = field(default_factory=list)


class Rejected(Exception):
    pass


def _as_of(records: list[Record], t: int | None) -> RecordStore:
    """The recorded state at time t: receipts and invoices posted later are absent, and an invoice cancelled
    later is shown as it was before the cancellation (active, one version earlier)."""
    if t is None:
        return RecordStore(list(records))
    later = {r.id for r in records if isinstance(r, Invoice) and r.posted_at is not None and r.posted_at > t}
    out = []
    for r in records:
        if (isinstance(r, Receipt) and r.posted_at > t) or r.id in later or getattr(r, "invoice", None) in later:
            continue
        if isinstance(r, Invoice) and r.state == "cancelled" and r.cancelled_at is not None and r.cancelled_at > t:
            r = r.model_copy(update={"state": "posted_released", "cancelled_at": None, "version": r.version - 1})
        out.append(r)
    return RecordStore(out)


class Engine:
    def __init__(self, records: list[Record], world: HiddenWorld, seed: int, current_invoice: str,
                 policy_profile: str = policy.POLICY_VERSION):
        self.world = world
        self.policy_profile = policy_profile   # active now; the recorded block was evaluated under sandbox-v1
        self.current_invoice = current_invoice
        self.clock = 0
        self.state_version = 1
        self.scheduler = Scheduler()
        self.attempts: list[Attempt] = []
        self._ids = random.Random(f"{seed}:runtime-ids")
        self.store = RecordStore([r for r in records if r.id != current_invoice])
        invoice = next(r for r in records if r.id == current_invoice)
        self.store.add(invoice)
        # Load-time evaluation: the comparison records shown at reset are those of the invoice's recorded
        # evaluation, computed on the records as they stood at its evaluated_at. When records changed
        # after it (a stale block: a later receipt, a later cancellation), the shown results are the old ones
        # and the input digest no longer matches, so release requires an explicit reprocess.
        as_of = _as_of(records, invoice.evaluated_at)
        evaluation = policy.evaluate(as_of, current_invoice)
        for name, effect, result in evaluation:
            self.store.add(ComparisonResult(id=self._new_id("CMP"), invoice=current_invoice, comparison=name,
                                            effect=effect, result=result, evaluated_at=invoice.evaluated_at))
        self.store._by_id[current_invoice] = invoice.model_copy(update={
            "eligibility": "eligible" if policy.eligible(evaluation) else "not_eligible",
            "evaluated_inputs": self._inputs_digest(as_of)})

    # Identifiers ------------------------------------------------------------------------------
    def _new_id(self, prefix: str) -> str:
        while True:
            rid = f"{prefix}-{self._ids.randrange(100000, 1000000)}"
            if not self.store.has(rid):
                return rid

    # Public API -------------------------------------------------------------------------------
    @property
    def invoice(self) -> Invoice:
        return self.store.get(self.current_invoice)

    @property
    def po_line(self) -> str:
        return self.store.invoice_lines(self.current_invoice)[0].po_line

    def business_hash(self) -> str:
        return self.store.business_hash()

    def changed_by(self, seq: int) -> list[str]:
        return list(self.attempts[seq - 1].changed)

    def execute(self, actor: str, op: str, **args) -> Attempt:
        attempt = Attempt(seq=len(self.attempts) + 1, at=self.clock, actor=actor, op=op, args=dict(args))
        self.attempts.append(attempt)
        try:
            if op not in PERMISSIONS.get(actor, set()):
                raise Rejected(f"Not authorized: the {actor} role cannot perform {op}.")
            getattr(self, f"_op_{op}")(attempt, **args)
        except Rejected as exc:
            attempt.accepted, attempt.message = False, str(exc)
            attempt.changed.clear()
            return attempt
        attempt.accepted = True
        self.state_version += 1
        return attempt

    def advance(self, minutes: int) -> None:
        target = self.clock + minutes
        while (event := self.scheduler.pop_due(target)) is not None:
            self.clock = max(self.clock, event.at)
            getattr(self, f"_event_{event.kind}")(event)
        self.clock = target

    # Mutation helpers (atomic per operation: all checks happen before the first write) ----------
    def _create(self, record: Record, attempt: Attempt | None) -> None:
        self.store.add(record)
        if attempt is not None:
            attempt.changed.append(record.id)

    def _update(self, record: Record, attempt: Attempt | None, **changes) -> Record:
        new = record.model_copy(update={**changes, "version": record.version + 1})
        self.store.replace(type(record).model_validate(new.model_dump()))
        if attempt is not None and record.id not in attempt.changed:
            attempt.changed.append(record.id)
        return new

    def _require_blocked(self) -> Invoice:
        inv = self.invoice
        if inv.state != "posted_blocked":
            raise Rejected(f"Rejected: invoice {inv.id} is not blocked (state {inv.state}).")
        return inv

    def _inputs_digest(self, store: RecordStore | None = None) -> str:
        """Versions of every record that feeds the current invoice's comparisons."""
        store = store or self.store
        inv = self.invoice
        parts = [f"{r.id}:{r.version}" for r in store.invoice_lines(inv.id)]
        parts.append(f"{inv.po}:{store.po(inv.po).version}")
        for pl in store.po_lines(inv.po):
            parts.append(f"{pl.id}:{pl.version}")
            parts += [f"{r.id}:{r.version}" for r in store.receipts(pl.id)]
            for line in store.invoice_lines(po_line=pl.id):
                if line.invoice != inv.id:
                    other = store.get(line.invoice)
                    parts += [f"{line.id}:{line.version}", f"{other.id}:{other.version}"]
        return hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()[:16]

    # Operations ---------------------------------------------------------------------------------
    def _op_request_confirmation(self, attempt: Attempt) -> None:
        inv = self._require_blocked()
        coverage = [c for c in self.store.comparison_results(inv.id)
                    if c.comparison in ("invoice_vs_receipt", "invoice_no_receipt") and c.result == "fail"]
        if not coverage:
            raise Rejected("Rejected: the invoice is not blocked on receipt coverage.")
        if any(c.status == "open" for c in self.store.confirmations(self.po_line)):
            raise Rejected("Rejected: a delivery confirmation request is already open for this PO line.")
        request = ConfirmationRequest(id=self._new_id("CR"), invoice=inv.id, po_line=self.po_line,
                                      requested_at=self.clock)
        self._create(request, attempt)
        if self.world.confirmation_delay is not None:
            self.scheduler.schedule(self.clock + self.world.confirmation_delay, "confirmation_response",
                                    {"request": request.id})

    def _op_request_receipt_posting(self, attempt: Attempt, quantity: str = "") -> None:
        inv = self._require_blocked()
        if not _WHOLE.match(str(quantity).strip()):
            raise Rejected("Rejected: the quantity to post must be a positive whole number.")
        qty = Decimal(str(quantity).strip())
        answered = [c for c in self.store.confirmations(self.po_line) if c.status == "answered"]
        if not answered:
            raise Rejected("Rejected: no delivery confirmation is on record for this PO line.")
        latest = max(answered, key=lambda c: (c.answered_at, c.id))
        confirmed_unposted = latest.delivered_total.value - calc.net_posted(self.store, self.po_line)
        if qty > confirmed_unposted:
            raise Rejected("Rejected: the delivery confirmation on record does not cover the requested quantity.")
        request = PostingRequest(id=self._new_id("PR"), invoice=inv.id, po_line=self.po_line,
                                 requested_at=self.clock, quantity=Qty(value=qty), confirmation=latest.id)
        self._create(request, attempt)
        self.scheduler.schedule(self.clock, "receipt_posting", {"request": request.id}, cause=attempt.seq)

    def _op_post_receipt(self, attempt: Attempt, quantity: str = "") -> None:  # receiving role only
        raise Rejected("Rejected: goods receipts are posted by the receiving role from a posting request.")

    def _op_reprocess(self, attempt: Attempt) -> None:
        inv = self._require_blocked()
        evaluation = policy.evaluate(self.store, inv.id, self.policy_profile)
        current = {c.comparison: c for c in self.store.comparison_results(inv.id)}
        seen = set()
        for name, effect, result in evaluation:
            seen.add(name)
            if name in current:
                self._update(current[name], attempt, effect=effect, result=result, evaluated_at=self.clock)
            else:
                self._create(ComparisonResult(id=self._new_id("CMP"), invoice=inv.id, comparison=name,
                                              effect=effect, result=result, evaluated_at=self.clock), attempt)
        for name, record in current.items():
            if name not in seen:
                self._update(record, attempt, result="not_applicable", evaluated_at=self.clock)
        self._update(inv, attempt, eligibility="eligible" if policy.eligible(evaluation) else "not_eligible",
                     evaluated_at=self.clock, evaluated_inputs=self._inputs_digest())

    def _op_release(self, attempt: Attempt) -> None:
        inv = self._require_blocked()
        # Release needs an explicit reprocess in this episode; the load-time evaluation
        # (the comparison records shown at reset) never substitutes for it, even when it is current and passing.
        if not any(a.op == "reprocess" and a.accepted for a in self.attempts):
            raise Rejected("Rejected: the invoice has not been reprocessed; reprocess the invoice first.")
        if inv.evaluated_inputs != self._inputs_digest():
            raise Rejected("Rejected: records changed since the last evaluation; reprocess the invoice first.")
        if inv.eligibility != "eligible":
            raise Rejected("Rejected: a blocking comparison fails; the invoice cannot be released.")
        self._update(inv, attempt, state="posted_released")

    def _op_keep_blocked(self, attempt: Attempt, follow_up_type: str = "") -> None:
        inv = self._require_blocked()
        if follow_up_type not in FOLLOW_UP_OWNER:
            raise Rejected("Rejected: unknown follow-up type.")
        if any(f.type == follow_up_type and f.status == "open" for f in self.store.follow_ups(inv.id)):
            raise Rejected("Rejected: an open follow-up of this type already exists for the invoice.")
        self._create(FollowUp(id=self._new_id("WI"), invoice=inv.id, created_at=self.clock, type=follow_up_type,
                              owner=FOLLOW_UP_OWNER[follow_up_type]), attempt)

    # Scheduled events (hidden schedule; simulated external actors) -------------------------------
    def _event_confirmation_response(self, event) -> None:
        request = self.store.get(event.payload["request"])
        delivered = self.world.delivered_by(request.po_line, event.at)
        self._update(request, None, status="answered", answered_at=event.at, delivered_total=Qty(value=delivered))
        self.state_version += 1

    def _event_receipt_posting(self, event) -> None:
        cause = self.attempts[event.cause - 1]
        request = self.store.get(event.payload["request"])
        log = Attempt(seq=len(self.attempts) + 1, at=event.at, actor="receiving", op="post_receipt",
                      args={"request": request.id, "quantity": str(request.quantity.value)})
        self.attempts.append(log)
        net = calc.net_posted(self.store, request.po_line)
        if self.world.delivered_by(request.po_line, event.at) < net + request.quantity.value:
            self._update(request, cause, status="rejected")
            log.message = "Rejected by receiving: the goods are not physically on hand."
        else:
            receipt = Receipt(id=self._new_id("GR"), po_line=request.po_line, posted_at=event.at,
                              movement="receipt", quantity=request.quantity)
            self._create(receipt, cause)
            self._update(request, cause, status="posted", receipt=receipt.id)
            log.accepted, log.message = True, "Posted by receiving."
        self.state_version += 1
