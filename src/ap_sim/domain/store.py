"""In-memory container for recorded state, with the queries the calculations need."""
from __future__ import annotations

import hashlib
import json

from ap_sim.domain.records import (ComparisonResult, ConfirmationRequest, FollowUp, Invoice, InvoiceLine,
                                   POLine, PostingRequest, PurchaseOrder, Receipt, Record)


class RecordStore:
    def __init__(self, records: list[Record] = ()):
        self._by_id: dict[str, Record] = {}
        for r in records:
            self.add(r)

    def add(self, record: Record) -> None:
        if record.id in self._by_id:
            raise ValueError(f"duplicate record id {record.id}")
        self._by_id[record.id] = record

    def replace(self, record: Record) -> None:
        old = self._by_id[record.id]
        if type(old) is not type(record) or record.version != old.version + 1:
            raise ValueError(f"{record.id}: replacement must keep the type and bump the version by one")
        self._by_id[record.id] = record

    def get(self, record_id: str) -> Record:
        return self._by_id[record_id]

    def has(self, record_id: str) -> bool:
        return record_id in self._by_id

    def all(self, cls: type[Record] | None = None) -> list[Record]:
        out = [r for r in self._by_id.values() if cls is None or isinstance(r, cls)]
        return sorted(out, key=lambda r: r.id)

    # Typed queries -----------------------------------------------------------------------------
    def po(self, po_id: str) -> PurchaseOrder:
        return self.get(po_id)

    def po_lines(self, po_id: str) -> list[POLine]:
        return [l for l in self.all(POLine) if l.po == po_id]

    def receipts(self, po_line: str) -> list[Receipt]:
        return [r for r in self.all(Receipt) if r.po_line == po_line]

    def invoices(self) -> list[Invoice]:
        return self.all(Invoice)

    def invoice_lines(self, invoice_id: str | None = None, po_line: str | None = None) -> list[InvoiceLine]:
        return [l for l in self.all(InvoiceLine)
                if (invoice_id is None or l.invoice == invoice_id) and (po_line is None or l.po_line == po_line)]

    def comparison_results(self, invoice_id: str) -> list[ComparisonResult]:
        return [c for c in self.all(ComparisonResult) if c.invoice == invoice_id]

    def confirmations(self, po_line: str | None = None) -> list[ConfirmationRequest]:
        return [c for c in self.all(ConfirmationRequest) if po_line is None or c.po_line == po_line]

    def posting_requests(self) -> list[PostingRequest]:
        return self.all(PostingRequest)

    def follow_ups(self, invoice_id: str | None = None) -> list[FollowUp]:
        return [f for f in self.all(FollowUp) if invoice_id is None or f.invoice == invoice_id]

    # Snapshot and hash -------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Plain-JSON view of every record, grouped by kind and sorted by id (Decimals as strings)."""
        out: dict[str, list[dict]] = {}
        for r in self.all():
            out.setdefault(r.kind, []).append(r.model_dump(mode="json"))
        return dict(sorted(out.items()))

    def business_hash(self) -> str:
        blob = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
