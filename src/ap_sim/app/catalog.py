"""The catalog: the only artifact both producers share.

Pages, regions, columns (visible header -> field, type), enumerations (code -> visible label),
and the case-action and navigation controls.
"""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA = "surfaces/4"   # the default: the v2 pipeline schema (training/config.PIPELINE_SCHEMA)
PRESENTATION_PROFILE = "ap-workbench/2"  # recorded in every episode log
APP_TITLE = "AP Workbench"               # appears only in the document <title>
EMPTY_MESSAGE = "No records"             # the only body row of a table without records


@dataclass(frozen=True)
class Column:
    header: str
    field: str
    type: str  # text | enum:<name> | ref | int | qty | money | time
    nullable: bool = False


@dataclass(frozen=True)
class Region:
    role: str
    title: str
    record_type: str | None
    columns: tuple[Column, ...] = ()


C = Column
REGIONS: tuple[Region, ...] = (
    Region("invoice_header", "Invoice", "invoice", (
        C("Supplier", "supplier", "text"), C("PO", "po", "ref"), C("State", "state", "enum:invoice_state"),
        C("Posted at", "posted_at", "time"), C("Eligibility", "eligibility", "enum:eligibility"),
        C("Evaluated at", "evaluated_at", "time"))),
    Region("invoice_lines", "Invoice lines", "invoice_line", (
        C("PO line", "po_line", "ref"), C("Quantity", "quantity", "qty"), C("Unit price", "unit_price", "money"),
        C("Amount", "amount", "money"))),
    Region("blocking_comparisons", "Blocking comparisons", "comparison_result", (
        C("Comparison", "comparison", "enum:comparison"), C("Effect", "effect", "enum:effect"),
        C("Result", "result", "enum:result"), C("Evaluated at", "evaluated_at", "time"))),
    Region("po_header", "Purchase order", "po", (
        C("Supplier", "supplier", "text"), C("Status", "status", "enum:po_status"))),
    Region("po_lines", "PO lines", "po_line", (
        C("Item", "item", "text"), C("Ordered quantity", "ordered_quantity", "qty"),
        C("Unit price", "unit_price", "money"), C("Line amount", "line_amount", "money"))),
    Region("receipts", "Receipt history", "receipt", (
        C("PO line", "po_line", "ref"), C("Posted at", "posted_at", "time"), C("Movement", "movement", "enum:movement"),
        C("Quantity", "quantity", "qty"), C("Reverses", "reverses", "ref", True))),
    Region("po_line_invoices", "Invoices on PO line", "invoice", (
        C("State", "state", "enum:invoice_state"), C("Quantity", "quantity", "qty"), C("Amount", "amount", "money"),
        C("Posted at", "posted_at", "time"), C("Cancelled at", "cancelled_at", "time", True))),
    Region("confirmations", "Delivery confirmations", "confirmation_request", (
        C("PO line", "po_line", "ref"), C("Requested at", "requested_at", "time"),
        C("Status", "status", "enum:request_status"), C("Answered at", "answered_at", "time", True),
        C("Delivered quantity (total)", "delivered_total", "qty", True))),
    Region("posting_requests", "Receipt-posting requests", "posting_request", (
        C("Requested at", "requested_at", "time"), C("Quantity", "quantity", "qty"),
        C("Confirmation", "confirmation", "ref"), C("Status", "status", "enum:posting_status"),
        C("Receipt", "receipt", "ref", True))),
    Region("follow_ups", "Follow-ups", "follow_up", (
        C("Created at", "created_at", "time"), C("Type", "type", "enum:follow_up_type"),
        C("Owner", "owner", "enum:owner"), C("Invoice", "invoice", "ref"),
        C("Status", "status", "enum:follow_up_status"))),
    # surfaces/3: event history and the active policy with the role's permissions.
    Region("events", "Event history", "event", (
        C("Time", "time", "time"), C("Type", "type", "enum:event_type"), C("Related record", "related", "ref"),
        C("Summary", "summary", "text"))),
    Region("policy_rules", "Active policy", "policy_rule", (
        C("Rule", "rule", "text"), C("Limit", "limit", "text"), C("Effect", "effect", "enum:rule_effect"),
        C("Effective since", "effective_since", "time", True))),
    Region("permissions", "Your permissions", "permission", (
        C("Action", "action", "enum:action"), C("Allowed", "allowed", "enum:allowed"))),
    # surfaces/4: the workflow rules and the report requirements (domain/rules.py).
    Region("workflow_rules", "Workflow rules", "workflow_rule", (
        C("Rule", "rule", "text"), C("Text", "text", "text"))),
    Region("report_requirements", "Report requirements", "report_requirement", (
        C("Requirement", "requirement", "text"), C("Text", "text", "text"))),
    Region("result", "Last action result", "action_result", (
        C("Action", "action", "enum:action"), C("Outcome", "outcome", "enum:outcome"), C("Message", "message", "text"))),
    Region("actions", "Case actions", None),
    Region("navigation", "Navigation", None),
)
REGION = {r.role: r for r in REGIONS}
REGION_BY_TITLE = {r.title: r for r in REGIONS}
CHANGED_PREFIX = "Changed: "

# Which base region shows a changed record of a given kind (for "Changed: <title>" regions).
CHANGED_ROLE_FOR_KIND = {
    "invoice": "invoice_header", "comparison_result": "blocking_comparisons", "receipt": "receipts",
    "confirmation_request": "confirmations", "posting_request": "posting_requests", "follow_up": "follow_ups",
}

PAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("invoice", "Invoice", ("invoice_header", "invoice_lines", "blocking_comparisons")),
    ("purchase_order", "Purchase order", ("po_header", "po_lines")),
    ("receipts", "Receipt history", ("receipts",)),
    ("po_line_invoices", "Invoices on PO line", ("po_line_invoices",)),
    ("confirmations", "Delivery confirmations", ("confirmations",)),
    ("posting_requests", "Receipt-posting requests", ("posting_requests",)),
    ("follow_ups", "Follow-ups", ("follow_ups",)),
)
# surfaces/3 adds two pages. Every recorded run replays on the schema it recorded.
PAGES_V3 = PAGES + (
    ("events", "Event history", ("events",)),
    ("policy", "Active policy", ("policy_rules", "permissions")),
)
# surfaces/4: surfaces/3 plus the Workflow rules and Report requirements pages, which carry
# the same text as the prompt (domain/rules.py). surfaces/3 is unchanged.
PAGES_V4 = PAGES_V3 + (
    ("workflow_rules", "Workflow rules", ("workflow_rules",)),
    ("report_requirements", "Report requirements", ("report_requirements",)),
)
SCHEMAS = {"surfaces/2": PAGES, "surfaces/3": PAGES_V3, "surfaces/4": PAGES_V4}


def pages_for(schema: str) -> dict[str, str]:
    """Page key -> title, in navigation order, for one observation schema."""
    return {p: t for p, t, _ in SCHEMAS[schema]}


PAGE_TITLE = {p: t for p, t, _ in PAGES_V4}   # every page of every schema (a superset)
PAGE_BY_TITLE = {t: p for p, t, _ in PAGES_V4}
PAGE_REGIONS = {p: rs for p, _, rs in PAGES_V4}

ROLE_LABEL = {"ap_agent": "AP agent"}

# Persistent case header: banner label -> key of the observation's `case` object.
CASE_ITEMS = (("Invoice", "invoice"), ("Supplier", "supplier"), ("PO", "po"))

# Case actions, in display order. key -> (role, accessible name, engine op, engine args)
ACTIONS: tuple[tuple[str, str, str, str | None, dict], ...] = (
    ("request_confirmation", "button", "Request delivery confirmation", "request_confirmation", {}),
    ("quantity", "textbox", "Quantity (EA)", None, {}),
    ("request_receipt_posting", "button", "Request receipt posting", "request_receipt_posting", {}),
    ("post_receipt", "button", "Post goods receipt (receiving role)", "post_receipt", {}),
    ("reprocess", "button", "Reprocess invoice", "reprocess", {}),
    ("release", "button", "Release invoice", "release", {}),
    ("keep_blocked_shortfall", "button", "Keep blocked: shortfall correction", "keep_blocked",
     {"follow_up_type": "shortfall_correction"}),
    ("keep_blocked_pending", "button", "Keep blocked: confirmation pending", "keep_blocked",
     {"follow_up_type": "confirmation_pending"}),
    ("keep_blocked_buyer_review", "button", "Keep blocked: buyer review", "keep_blocked",
     {"follow_up_type": "buyer_review"}),
)
ACTION = {a[0]: a for a in ACTIONS}
QUANTITY_ACTIONS = ("request_receipt_posting", "post_receipt")

ENUMS: dict[str, dict[str, str]] = {
    "invoice_state": {"received_unposted": "Received, not posted", "posted_blocked": "Posted, blocked",
                      "posted_released": "Posted, released", "cancelled": "Cancelled"},
    "eligibility": {"eligible": "Eligible", "not_eligible": "Not eligible"},
    "comparison": {"price": "Price tolerance", "quantity": "Quantity tolerance", "receipt_vs_po": "Receipt vs PO",
                   "invoice_vs_receipt": "Invoice vs receipt", "invoice_no_receipt": "Invoice without receipt",
                   "invoice_vs_po_quantity": "Invoice vs PO quantity", "invoice_vs_po_amount": "Invoice vs PO amount",
                   "closed_po": "Closed PO"},
    "effect": {"blocking": "Blocking", "informational": "Informational"},
    "result": {"pass": "Pass", "fail": "Fail", "flag": "Flagged", "not_applicable": "Not applicable"},
    "po_status": {"open": "Open", "closed": "Closed"},
    "movement": {"receipt": "Receipt", "reversal": "Reversal"},
    "request_status": {"open": "Open", "answered": "Answered"},
    "posting_status": {"requested": "Requested", "posted": "Posted", "rejected": "Rejected"},
    "follow_up_type": {"shortfall_correction": "Shortfall correction", "confirmation_pending": "Confirmation pending",
                       "buyer_review": "Buyer review"},
    "owner": {"ap": "AP", "buyer": "Buyer"},
    "follow_up_status": {"open": "Open", "closed": "Closed"},
    "action": {key: name for key, role, name, _, _ in ACTIONS if role == "button"},
    "outcome": {"accepted": "Accepted", "rejected": "Rejected"},
    "event_type": {"invoice_posted": "Invoice posted", "invoice_checked": "Invoice checked",
                   "receipt_posted": "Receipt posted", "receipt_reversed": "Receipt reversed",
                   "invoice_cancelled": "Invoice cancelled", "policy_effective": "Policy took effect",
                   "confirmation_requested": "Confirmation requested", "confirmation_answered": "Confirmation answered",
                   "posting_requested": "Receipt posting requested", "follow_up_opened": "Follow-up opened",
                   "invoice_released": "Invoice released"},
    "rule_effect": {"blocking": "Blocking", "informational": "Informational", "disabled": "Disabled"},
    "allowed": {"yes": "Allowed", "no": "Not allowed"},
}
ENUM_BY_LABEL = {name: {label: code for code, label in m.items()} for name, m in ENUMS.items()}
COMPARISON_ORDER = list(ENUMS["comparison"])
