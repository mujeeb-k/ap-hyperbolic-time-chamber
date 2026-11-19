"""The task instructions the model receives (fixed text, frozen together with the tool schemas).

They state the job, the tools, the limits and the report format. They contain no case data, no hidden
information and no expected outcome; the case arrives as the reset observation.
"""
from ap_sim.domain.report import COMPLETION_CONTRACT, evidence_requirements
from ap_sim.events.clock import ACTION_LIMIT, HORIZON, format_time

INSTRUCTIONS_VERSION = "instructions/4"  # finish format from domain/report.py; observe clears unsent input;
# the completion contract (domain/report.COMPLETION_CONTRACT) and the report-level evidence list

SYSTEM_PROMPT = f"""You are an accounts-payable analyst working in the AP Workbench application.
One supplier invoice is blocked. Investigate why, obtain any evidence you need, and resolve the case
with the authorized actions available to your role.

How the application works:
- Every tool result is JSON. Pages contain regions with records (each with an id and a version) and
  controls (each with an opaque id). Act only on controls shown in the latest observation, by id.
- click(element_id) opens a page or performs a case action. fill(element_id, text) types into a
  textbox. observe() reloads the current page and clears text typed but not yet sent (for example a
  quantity you filled in but have not submitted). advance_time(minutes) lets business time pass
  (1 to 480 minutes) while you wait for others; it returns only the new time.
- Business time is simulated. Each tool call except advance_time and finish costs 5 minutes.
- The business horizon is {format_time(HORIZON)}. An action that cannot complete by then is refused.
  You have at most {ACTION_LIMIT} tool calls in total, including advance_time and finish.
- If the state changed since your last observation, an action is refused until you observe again.

When you are done, call finish with your final disposition (release, keep_blocked or pending), your
findings (may be empty) and the evidence you relied on (a report-level evidence list and/or a finding's
evidence). finish performs no business action: anything you did
not do in the application stays undone. The task counts as completed only with an accepted finish.
If finish is not accepted, its result lists the fields to fix; correct them and call finish again.

Each finding has discrepancy_type ("receipt_coverage_gap" or "none"), root_cause (short text),
blocking_comparison (comparison code or null) and evidence (a list). Every evidence entry has a "form":
{chr(10).join("- " + line for line in evidence_requirements())}
Observed entries cite what you actually saw: source_id is the page, observation_id and state_version
come from that observation, and record_id, record_version, field and value from the record as shown in
it. A derived entry gives an arithmetic expression over named operands (r1 + r2 - i1), each operand
citing observation_id, record_id, record_version and field; value is the result as a decimal string.

What an accepted report must contain to count as completed:
{chr(10).join(COMPLETION_CONTRACT)}
"""

# V2. instructions/4 above stays byte-identical, for reproducing v1 runs. instructions/5 adds the workflow rules and
# report requirements (domain/rules.py; the same text is on the Workflow rules and Report requirements pages of
# surfaces/4). It was frozen at the strong-model check run before training. Nothing was generated with it before then.
from ap_sim.domain import rules as _rules

INSTRUCTIONS_VERSION_V2 = "instructions/5"
# The coverage calculator (a tool of APReceiptEnvV2, not a page).
CALCULATOR_NOTE = ("A coverage(invoice_id) tool computes the invoice's available coverage from the current records: it "
                   "returns the receipts, reversals and other active invoices on the PO line (with record ids and versions) "
                   "and the figures computed from them. It reads records only and costs 5 minutes like other tool calls. "
                   "It is not a page: cite records from the pages where you saw them.\n")
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + "\n" + CALCULATOR_NOTE + "\n" + _rules.prompt_section()

# The text in front of the first observation in the user turn (TRL appends the reset observation to it). Owned here
# since v2 (in version 1 of this environment it was in training/config.py; the text is unchanged).
USER_PREFIX = "Here is the case. Initial page:\n"
