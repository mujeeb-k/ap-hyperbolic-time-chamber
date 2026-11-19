"""The `finish` report. It performs no business action.

Evidence forms: observed (a cited value from a delivered observation), derived (a restricted
arithmetic expression over named observed operands, evaluated by the grader, never executed),
unsupported (an honest assertion without support).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(_Strict):
    observation_id: str
    record_id: str
    record_version: int
    field: str


class ObservedEvidence(_Strict):
    form: Literal["observed"]
    source_kind: Literal["page"] = "page"
    source_id: str  # page id
    observation_id: str
    state_version: int
    record_id: str
    record_version: int
    field: str
    value: Any


class DerivedEvidence(_Strict):
    form: Literal["derived"]
    expression: str = Field(max_length=200)
    operands: dict[str, Citation]
    source_observation_ids: list[str]
    state_version: int
    value: str  # decimal string


class UnsupportedEvidence(_Strict):
    form: Literal["unsupported"]
    claim: str = Field(max_length=500)


Evidence = Annotated[Union[ObservedEvidence, DerivedEvidence, UnsupportedEvidence], Field(discriminator="form")]


class Finding(_Strict):
    discrepancy_type: Literal["receipt_coverage_gap", "none"]
    root_cause: str = Field(max_length=500)
    blocking_comparison: str | None = None
    evidence: list[Evidence] = []


class FinishReport(_Strict):
    findings: list[Finding] = []
    final_disposition: Literal["release", "keep_blocked", "pending"]
    # Supporting evidence for the disposition, independent of findings: findings may stay empty when there is
    # no discrepancy to describe, while the required source records are still cited here (or in a finding).
    evidence: list[Evidence] = []



class FinishReportV2(FinishReport):
    """The instructions/5 report (surfaces/4; POLICY I5): FinishReport plus a REQUIRED top-level
    discrepancy_type, graded by g7. FinishReport itself is unchanged, so surfaces/3 reports validate exactly as before.
    The per-finding discrepancy_type stays as it is, ungraded."""
    discrepancy_type: Literal["receipt_coverage_gap", "none"]


def report_model(schema: str) -> type[FinishReport]:
    """The report model a schema's finish validates against."""
    return FinishReportV2 if schema == "surfaces/4" else FinishReport

# The completion contract the grader enforces from g4, in agent-facing words: what must be cited, per
# disposition. It names record kinds and fields only; nothing about any case or its hidden schedule. The grader
# (grader._required_evidence) checks exactly these, against the records current when the deciding action was taken.
COMPLETION_CONTRACT = [
    "final_disposition must be the outcome you actually completed in the application (released, kept blocked "
    "with a shortfall follow-up, or kept blocked as confirmation pending); a report never replaces an action.",
    "Cite, as observed entries (in the report's evidence list or in a finding's evidence), each record at its "
    "version when you took the deciding action (release / keep blocked / keep blocked pending):",
    "- release: the invoice line (quantity), every receipt on the PO line including any posted during the case "
    "(quantity), every other active invoice on the PO line (quantity), every cancelled invoice on the PO line "
    "(state), and, if the recorded receipts did not cover the invoice, the answered confirmation (delivered_total);",
    "- keep_blocked: the invoice line, every receipt and every other active invoice on the PO line (quantity), and "
    "the answered confirmation (delivered_total);",
    "- pending: the confirmation request (status).",
    "For keep_blocked and pending, also cite the follow-up the action opened (type); it may come from the action's "
    "own result.",
    "Findings may be empty when there is nothing more to explain; derived entries are optional and, if given, must "
    "be correct.",
]


# Agent-facing descriptions, generated from the models above so the tool schema, the instructions and the
# validator cannot drift apart. They describe the format only: nothing about the case, the
# expected decision or how the report is graded.
_FORMS = {"observed": ObservedEvidence, "derived": DerivedEvidence, "unsupported": UnsupportedEvidence}


def _strip_titles(node):
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def findings_json_schema() -> dict:
    """JSON Schema of the `findings` argument (a list of Finding), with every nested evidence form."""
    from pydantic import TypeAdapter
    return _strip_titles(TypeAdapter(list[Finding]).json_schema())


def evidence_json_schema() -> dict:
    """JSON Schema of the report-level `evidence` argument (a list of evidence entries)."""
    from pydantic import TypeAdapter
    return _strip_titles(TypeAdapter(list[Evidence]).json_schema())


def evidence_requirements() -> list[str]:
    """One line per evidence form: its `form` value and its required and optional fields."""
    lines = []
    for form, model in _FORMS.items():
        required = [n for n, f in model.model_fields.items() if f.is_required() and n != "form"]
        optional = [n for n, f in model.model_fields.items() if not f.is_required()]
        line = f'form "{form}": required ' + (", ".join(required) if required else "nothing else")
        lines.append(line + (f"; optional {', '.join(optional)}" if optional else ""))
    return lines


_PROBLEMS = {"missing": "required field is missing", "extra_forbidden": "unknown field (not allowed)",
             "union_tag_not_found": 'evidence entry needs "form": "observed", "derived" or "unsupported"',
             "union_tag_invalid": 'evidence "form" must be "observed", "derived" or "unsupported"'}


def describe_errors(exc, limit: int = 12) -> list[dict]:
    """Field-level problems of a failed validation, as {field, problem}. Input values are never echoed."""
    out = []
    for e in exc.errors(include_url=False, include_input=False):
        path, prev = "", None
        for part in e["loc"]:
            if part in _FORMS and isinstance(prev, int):  # pydantic's union-branch tag, not a field the agent wrote
                prev = part
                continue
            path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
            prev = part
        out.append({"field": path or "(report)", "problem": _PROBLEMS.get(e["type"], e["msg"])})
    return out[:limit]
