"""The visible workflow rules and report requirements (surfaces/4, instructions/5): ONE source for the
prompt, the Workflow rules page and the Report requirements page. All three must stay consistent with each other,
with the grader (grader._required_evidence) and with POLICY.

The wording comes from version C of the strong-model check run before training (scripts/strong_model_check.py):
rules 1-8 plus its citation and records notes. It contains no evaluator-only information: no confirmation delays,
no hidden quantities, no case types, no outcomes of any case.
"""
from __future__ import annotations

WORKFLOW_RULES: tuple[tuple[str, str], ...] = (
    # Stable know-how only (since 2026-10-30): policy-specific values (comparison limits, tolerances, when a
    # policy took effect) appear only on the Active policy page, never here; rule 1 sends the agent there.
    ("1. Read the active policy",
     "The comparison limits and tolerances, and the time the current policy took effect, are on the Active policy page. "
     "They can differ between cases, so read them there for each case. These rules say how to work; the Active policy "
     "page says which limits apply."),
    ("2. Coverage",
     "Available coverage = net posted receipts on the PO line (receipts minus reversals) minus the quantity of every "
     "other active invoice on the same PO line. Cancelled invoices do not count. Whether the invoice quantity may exceed "
     "available coverage, and by how much, is set by the invoice-vs-receipt rule and any quantity tolerance on the "
     "Active policy page. A limit fails only when it is strictly exceeded; equality passes."),
    ("3. Blocks can be stale",
     "The comparison results on the invoice were evaluated at the time shown. Receipts posted, invoices cancelled, or a "
     "policy that took effect after that time are not reflected. Before deciding, view Receipt history and Invoices on "
     "PO line and check the invoice against the current records and the current policy."),
    ("4. Release needs a reprocess first",
     "Reprocess the invoice to re-evaluate it on the current records, then release. Release only when the latest "
     "reprocess found the invoice eligible and no receipt or invoice on the PO line has changed since; otherwise "
     "reprocess again first. Asking to release while a blocking comparison fails, or before eligibility is established "
     "on the current records, is a policy violation even when the application refuses it."),
    ("5. When a confirmation is needed",
     "If the current records already satisfy the invoice-vs-receipt rule on the Active policy page, no confirmation is "
     "needed: reprocess, then release. If they do not, the gap is the invoice quantity minus available coverage. The "
     "records cannot tell whether the goods arrived and were not recorded, or never arrived. Request a delivery "
     "confirmation from the warehouse and wait for its answer before deciding. A missing answer is not evidence that "
     "the goods did not arrive. Never release while a gap remains."),
    ("6. How a receipt gets posted",
     "AP cannot post goods receipts. \"Post goods receipt (receiving role)\" is not permitted for your role; using it is "
     "a policy violation even though it is refused. After a positive confirmation is on record, type the quantity into "
     "\"Quantity (EA)\" and click \"Request receipt posting\". Request the full confirmed unrecorded quantity: the "
     "confirmed delivered total minus the net posted receipts. The receiving role posts it at once."),
    ("7. Which outcome and follow-up",
     "If the confirmation covers the gap: request the receipt posting, reprocess, release; no follow-up. If it does not "
     "(the confirmed delivered total minus net posted receipts is less than the gap): \"Keep blocked: shortfall "
     "correction\". If no answer can arrive in time to finish the resolution before the horizon: \"Keep blocked: "
     "confirmation pending\"; this needs a confirmation request on record. \"Keep blocked: buyer review\" is only for an "
     "invoice against a closed PO."),
    ("8. Waiting, and when pending is correct",
     "Business time passes only when you act or call advance_time. A confirmation answer appears on the Delivery "
     "confirmations page; advance_time does not show it, so view that page after waiting. You are not told when an "
     "answer will arrive. Confirmation pending is correct only when the answer cannot arrive in time for you to finish "
     "the resolution before the horizon (D5 17:00); choosing it while a resolution was still possible is a policy "
     "violation. So after requesting a confirmation, wait (advance_time, up to 480 minutes per call) and check the "
     "Delivery confirmations page after each wait. When an answer arrives, act on it. Keep blocked as confirmation "
     "pending only if your check of the Delivery confirmations page at D5 16:35 or later still shows no answer."),
    ("9. Business time",
     "Business time runs only from 09:00 to 17:00 (480 minutes per business day, so D1 17:00 is the same moment as D2 "
     "09:00), and the horizon is D5 17:00, 2,400 business minutes after the start at D1 09:00."),
)

REPORT_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("1. Citing a value",
     "In an observed evidence entry, value must be the field's value exactly as it appears in the record on the page, "
     "including any object. For example, a receipt record that shows \"quantity\": {\"unit\": \"EA\", \"value\": \"129\"} "
     "is cited with \"field\": \"quantity\" and \"value\": {\"unit\": \"EA\", \"value\": \"129\"}, not \"129\"."),
    ("2. Citing an observation",
     "Cite each record with the observation_id and state_version of an observation in which you saw it, and the "
     "record_version shown there. A citation to an observation that does not show that record at that version does not "
     "count."),
    ("3. Which version",
     "Cite each record at its version just before you took the deciding action (release, keep blocked, or keep blocked "
     "as confirmation pending), except the follow-up that action opened: it did not exist before the action, so cite it "
     "as shown in the action's own result or on a later page."),
    ("4. Records for a release",
     "The invoice line of the invoice you are resolving, from the Invoice page (field quantity), for example the invoice "
     "line INV-100000-1, not the invoice INV-100000 (Invoices on PO line also lists this invoice as an invoice record; "
     "citing that record does not count); every record on Receipt history (field quantity), including any receipt "
     "posted during the case; every other active invoice listed on Invoices on PO line, by its invoice record (field "
     "quantity); every cancelled invoice on the PO line (field state); and, if the recorded receipts did not cover the "
     "invoice when you started, the answered confirmation (field delivered_total)."),
    ("5. Records for keep blocked",
     "The invoice line (as for a release), every record on Receipt history (field quantity), every other active "
     "invoice by its invoice record (field quantity), the answered confirmation (field delivered_total), and the "
     "shortfall correction follow-up the action opened (field type)."),
    ("6. Records for confirmation pending",
     "The confirmation request (field status) and the confirmation pending follow-up the action opened (field type)."),
    ("7. Discrepancy type",
     "Every report has a top-level discrepancy_type: receipt_coverage_gap if, at the start of the case, the invoice "
     "failed the invoice-vs-receipt rule under the policy then in effect (see the Active policy page); none if it "
     "passed, including when its excess over coverage was within a tolerance that policy allows. The discrepancy_type "
     "inside a finding is separate and optional to get right; the top-level field is the one that counts."),
)

# The same requirements as data, in the grader's terms (grader._required_evidence "what" labels and fields), for the
# consistency test: outcome -> {(what, field)}. Conditional items: the confirmation for a release only when the
# recorded receipts did not cover the invoice at the start; cancelled invoices only when there are any.
REQUIRED_RECORDS: dict[str, set[tuple[str, str]]] = {
    "release": {("invoice line", "quantity"), ("receipt", "quantity"), ("other active invoice", "quantity"),
                ("cancelled invoice", "state"), ("positive confirmation", "delivered_total")},
    "keep_blocked": {("invoice line", "quantity"), ("receipt", "quantity"), ("other active invoice", "quantity"),
                     ("negative confirmation", "delivered_total"), ("shortfall_correction follow-up", "type")},
    "pending": {("confirmation request status", "status"), ("confirmation_pending follow-up", "type")},
}

RULES_HEADING = "Workflow rules (also on the Workflow rules page):"
REQUIREMENTS_HEADING = "Report requirements (also on the Report requirements page):"


def prompt_section() -> str:
    """The text appended to the instructions (instructions/5)."""
    rules = "\n".join(f"{title}. {text}" for title, text in WORKFLOW_RULES)
    reqs = "\n".join(f"{title}. {text}" for title, text in REPORT_REQUIREMENTS)
    return f"{RULES_HEADING}\n{rules}\n\n{REQUIREMENTS_HEADING}\n{reqs}\n"
