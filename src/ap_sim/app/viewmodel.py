"""Page view models built from recorded state and UI state only (never the hidden world).

Values are catalog-typed: str (text, enum code, ref), int (int, time in business minutes),
Qty, Money, or None. The HTML renderer and the page-model producer both consume these objects.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal

from ap_sim.app import catalog
from ap_sim.app.session import AppSession
from ap_sim.domain.records import CENT, Money, Qty, Record


@dataclass
class RecordVM:
    id: str
    version: int
    type: str
    fields: dict


@dataclass
class ControlVM:
    key: str
    role: str
    name: str
    value: str | None = None


@dataclass
class RegionVM:
    role: str
    title: str
    records: list[RecordVM] = field(default_factory=list)
    controls: list[ControlVM] = field(default_factory=list)


@dataclass
class PageVM:
    page: str
    title: str
    state_version: int
    clock: int
    role: str
    regions: list[RegionVM]
    case: dict  # persistent case header: invoice, supplier, po


def _row(region_role: str, record: Record, **overrides) -> RecordVM:
    region = catalog.REGION[region_role]
    data = {c.field: overrides.get(c.field, getattr(record, c.field, None)) for c in region.columns}
    return RecordVM(record.id, record.version, region.record_type, data)


def _records(session: AppSession, role: str, only: set[str] | None = None) -> list[RecordVM]:
    store, eng = session.engine.store, session.engine
    inv = eng.invoice
    keep = (lambda r: r.id in only) if only is not None else (lambda r: True)
    if role == "invoice_header":
        rows = [_row(role, inv)] if keep(inv) else []
    elif role == "invoice_lines":
        rows = [_row(role, l) for l in store.invoice_lines(inv.id) if keep(l)]
    elif role == "blocking_comparisons":
        rs = sorted(store.comparison_results(inv.id), key=lambda c: (catalog.COMPARISON_ORDER.index(c.comparison), c.id))
        rows = [_row(role, c) for c in rs if keep(c)]
    elif role == "po_header":
        po = store.po(inv.po)
        rows = [_row(role, po)] if keep(po) else []
    elif role == "po_lines":
        rows = [_row(role, l) for l in store.po_lines(inv.po) if keep(l)]
    elif role == "receipts":
        rs = sorted(store.receipts(eng.po_line), key=lambda r: (r.posted_at, r.id))
        rows = [_row(role, r) for r in rs if keep(r)]
    elif role == "po_line_invoices":
        rows = []
        for i in sorted(store.invoices(), key=lambda i: (i.posted_at if i.posted_at is not None else 0, i.id)):
            lines = store.invoice_lines(i.id, eng.po_line)
            if lines and keep(i):
                qty = sum((l.quantity.value for l in lines), Decimal(0))
                amount = sum((l.amount.value for l in lines), Decimal(0)).quantize(CENT)
                rows.append(_row(role, i, quantity=Qty(value=qty), amount=Money(value=amount)))
    elif role == "confirmations":
        rs = sorted(store.confirmations(eng.po_line), key=lambda c: (c.requested_at, c.id))
        rows = [_row(role, c) for c in rs if keep(c)]
    elif role == "posting_requests":
        rs = sorted((p for p in store.posting_requests() if p.invoice == inv.id), key=lambda p: (p.requested_at, p.id))
        rows = [_row(role, p) for p in rs if keep(p)]
    elif role == "follow_ups":
        rs = sorted(store.follow_ups(inv.id), key=lambda f: (f.created_at, f.id))
        rows = [_row(role, f) for f in rs if keep(f)]
    elif role == "events":
        rows = [r for r in _events(session) if keep(r)]
    elif role == "policy_rules":
        rows = [r for r in _policy_rules(session) if keep(r)]
    elif role in ("workflow_rules", "report_requirements"):   # surfaces/4: the same text as the prompt
        from ap_sim.domain import rules
        items = rules.WORKFLOW_RULES if role == "workflow_rules" else rules.REPORT_REQUIREMENTS
        prefix, kind, key = (("WR", "workflow_rule", "rule") if role == "workflow_rules"
                             else ("RR", "report_requirement", "requirement"))
        rows = [r for r in (RecordVM(f"{prefix}-{i + 1:02d}", 1, kind, {key: title, "text": text})
                            for i, (title, text) in enumerate(items)) if keep(r)]
    elif role == "permissions":
        buttons = [key for key, r, _, _, _ in catalog.ACTIONS if r == "button"]
        rows = [RecordVM(f"PERM-{i + 1:02d}", 1, "permission",
                         {"action": key, "allowed": "no" if key == "post_receipt" else "yes"})
                for i, key in enumerate(buttons)]
    else:
        raise KeyError(role)
    return rows


def _event(kind: str, time: int, related: str, summary: str) -> RecordVM:
    digest = hashlib.sha256(f"{kind}|{related}|{time}".encode()).hexdigest()[:6].upper()
    return RecordVM(f"EV-{digest}", 1, "event", {"time": time, "type": kind, "related": related, "summary": summary})


def _events(session: AppSession) -> list[RecordVM]:
    """Events that have ALREADY happened (timestamps at or before the clock), derived from recorded state and the
    agent's own accepted operations. Never a scheduled event, an answer time before it fires, or the hidden world."""
    store, eng = session.engine.store, session.engine
    inv, out = eng.invoice, []
    qty = lambda q: format(q.value.normalize(), "f") + " EA"  # noqa: E731
    for i in store.invoices():
        if store.invoice_lines(i.id, eng.po_line):
            if i.posted_at is not None:
                out.append(_event("invoice_posted", i.posted_at, i.id, f"Invoice {i.id} posted."))
            if i.state == "cancelled" and i.cancelled_at is not None:
                out.append(_event("invoice_cancelled", i.cancelled_at, i.id, f"Invoice {i.id} cancelled."))
    out.append(_event("invoice_checked", session.initial_check[0], inv.id,
                      f"Invoice {inv.id} checked: {catalog.ENUMS['eligibility'][session.initial_check[1]]}."))
    for r in store.receipts(eng.po_line):
        kind = "receipt_reversed" if r.movement == "reversal" else "receipt_posted"
        out.append(_event(kind, r.posted_at, r.id, f"{'Reversal' if r.movement == 'reversal' else 'Receipt'} "
                                                   f"{r.id}: {qty(r.quantity)}."))
    for c in store.confirmations(eng.po_line):
        out.append(_event("confirmation_requested", c.requested_at, c.id, f"Confirmation {c.id} requested."))
        if c.status == "answered":
            out.append(_event("confirmation_answered", c.answered_at, c.id,
                              f"Warehouse answered {c.id}: delivered {qty(c.delivered_total)} in total."))
    for pr in (p for p in store.posting_requests() if p.invoice == inv.id):
        out.append(_event("posting_requested", pr.requested_at, pr.id, f"Receipt posting {pr.id} requested: "
                                                                       f"{qty(pr.quantity)}."))
    for f in store.follow_ups(inv.id):
        out.append(_event("follow_up_opened", f.created_at, f.id,
                          f"Follow-up {f.id} opened: {catalog.ENUMS['follow_up_type'][f.type]}."))
    for h in session.history:
        if h["action"] == "reprocess" and h["attempt"]["accepted"]:
            out.append(_event("invoice_checked", h["attempt"]["at"], inv.id, f"Invoice {inv.id} reprocessed."))
        if h["action"] == "release" and h["attempt"]["accepted"]:
            out.append(_event("invoice_released", h["attempt"]["at"], inv.id, f"Invoice {inv.id} released."))
    if session.policy_since is not None:
        out.append(_event("policy_effective", session.policy_since, f"POL-{session.policy_profile}",
                          f"Policy {session.policy_profile} took effect."))
    return sorted((e for e in out if e.fields["time"] <= eng.clock), key=lambda e: (e.fields["time"], e.id))


def _policy_rules(session: AppSession) -> list[RecordVM]:
    from ap_sim.domain import policy as pol
    a2 = pol.PROFILES[session.policy_profile]["a2"]
    since = session.policy_since
    rules = [("Policy version", session.policy_profile, "informational"),
             ("Price tolerance", f"{pol.PRICE_PCT}% of PO unit price, or EUR {pol.PRICE_ABS} line effect", "blocking"),
             ("Invoice vs receipt", "Invoice quantity must not exceed available coverage (net receipts minus other "
                                    "active invoices)", "blocking"),
             ("Quantity tolerance on invoice vs receipt",
              "Disabled" if a2 is None else f"Excess over coverage passes if at most {a2['pct']}% of available "
                                            f"coverage and at most EUR {a2['abs']} (excess x PO unit price)",
              "disabled" if a2 is None else "blocking"),
             ("Invoice without receipt", "No invoice quantity without a posted receipt", "blocking"),
             ("Receipt vs PO", f"Net receipts above {100 + pol.RECEIPT_VS_PO_PCT}% of PO quantity are flagged",
              "informational"),
             ("Invoice vs PO quantity", f"Cumulative active invoiced quantity at most {100 + pol.INVOICE_VS_PO_QTY_PCT}% "
                                        "of PO quantity", "blocking"),
             ("Invoice vs PO amount", f"Cumulative active invoiced amount at most {100 + pol.INVOICE_VS_PO_AMOUNT_PCT}% "
                                      f"of PO amount or PO amount + EUR {pol.INVOICE_VS_PO_AMOUNT_ABS}", "blocking"),
             ("Closed PO", "No release against a closed PO", "blocking")]
    return [RecordVM(f"POL-{i + 1:02d}", 1, "policy_rule",
                     {"rule": r, "limit": lim, "effect": eff, "effective_since": since}) for i, (r, lim, eff) in
            enumerate(rules)]


def build_page(session: AppSession) -> PageVM:
    eng = session.engine
    regions = [RegionVM(r, catalog.REGION[r].title, _records(session, r)) for r in catalog.PAGE_REGIONS[session.page]]
    if session.last_result is not None:
        res, attempt = session.last_result, session.last_result.attempt
        message = attempt.message if not attempt.accepted else _accepted_message(session, res.action, attempt)
        regions.append(RegionVM("result", catalog.REGION["result"].title, [RecordVM(
            res.id, 1, "action_result",
            {"action": res.action, "outcome": "accepted" if attempt.accepted else "rejected", "message": message})]))
        changed = set(attempt.changed)
        kinds = {eng.store.get(rid).kind for rid in changed}
        for base in (r.role for r in catalog.REGIONS):
            if base in {catalog.CHANGED_ROLE_FOR_KIND.get(k) for k in kinds}:
                rows = _records(session, base, only=changed)
                if rows:
                    regions.append(RegionVM(f"changed:{base}", catalog.CHANGED_PREFIX + catalog.REGION[base].title, rows))
    regions.append(RegionVM("actions", catalog.REGION["actions"].title, controls=[
        ControlVM(f"act:{key}", role, name, session.draft if role == "textbox" else None)
        for key, role, name, _, _ in catalog.ACTIONS]))
    regions.append(RegionVM("navigation", catalog.REGION["navigation"].title, controls=[
        ControlVM(f"nav:{p}", "link", title) for p, title in catalog.pages_for(session.schema).items()]))
    inv = eng.invoice
    case = {"invoice": inv.id, "supplier": inv.supplier, "po": inv.po}
    return PageVM(session.page, catalog.PAGE_TITLE[session.page], eng.state_version, eng.clock, "ap_agent", regions,
                  case)


def _accepted_message(session: AppSession, action: str, attempt) -> str:
    """What an accepted action actually did, composed from the records it changed."""
    store = session.engine.store
    changed = [store.get(rid) for rid in attempt.changed]
    first = lambda kind: next((r for r in changed if r.kind == kind), None)  # noqa: E731
    if action == "request_confirmation":
        return f"Confirmation request {first('confirmation_request').id} sent to the warehouse; awaiting its answer."
    if action == "request_receipt_posting":
        pr = first("posting_request")
        head = f"Posting request {pr.id} for {format(pr.quantity.value.normalize(), 'f')} EA accepted; "
        return head + (f"receiving posted receipt {pr.receipt}." if pr.status == "posted" else "receiving rejected it.")
    if action == "reprocess":
        label = catalog.ENUMS["eligibility"][session.engine.invoice.eligibility]
        return f"Invoice reprocessed; eligibility: {label}."
    if action == "release":
        return "Invoice released."
    fu = first("follow_up")
    return f"Invoice kept blocked; follow-up {fu.id} ({catalog.ENUMS['follow_up_type'][fu.type]}) opened."
