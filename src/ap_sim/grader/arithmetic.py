"""The grader's own reference arithmetic, written independently of the engine.

Works on plain JSON snapshots ({kind: [record dicts]}); never imports the engine, the engine's
policy or calculations, the application or the surfaces.
"""
from __future__ import annotations

from decimal import Decimal

# Own copies of the policy constants (POLICY A, C1).
ACTION_COST, HORIZON, ADVANCE_MAX, ACTION_LIMIT = 5, 2400, 480, 40


def _d(v) -> Decimal:
    return Decimal(str(v))


def rows(snap: dict, kind: str) -> list[dict]:
    return snap.get(kind, [])


def by_id(snap: dict) -> dict[str, dict]:
    return {r["id"]: r for recs in snap.values() for r in recs}


def net_receipts(snap: dict, po_line: str) -> Decimal:
    total = Decimal(0)
    for r in rows(snap, "receipt"):
        if r["po_line"] == po_line:
            q = _d(r["quantity"]["value"])
            total += q if r["movement"] == "receipt" else -q
    return total


def _invoice_states(snap: dict) -> dict[str, str]:
    return {i["id"]: i["state"] for i in rows(snap, "invoice")}


def line_qty(snap: dict, invoice: str, po_line: str) -> Decimal:
    return sum((_d(l["quantity"]["value"]) for l in rows(snap, "invoice_line")
                if l["invoice"] == invoice and l["po_line"] == po_line), Decimal(0))


def other_active(snap: dict, po_line: str, invoice: str) -> Decimal:
    states = _invoice_states(snap)
    return sum((_d(l["quantity"]["value"]) for l in rows(snap, "invoice_line")
                if l["po_line"] == po_line and l["invoice"] != invoice and states[l["invoice"]] != "cancelled"),
               Decimal(0))


def coverage(snap: dict, po_line: str, invoice: str) -> Decimal:
    return net_receipts(snap, po_line) - other_active(snap, po_line, invoice)


# The grader's own copy of the policy profiles' quantity tolerance (POLICY A2). It is enabled only in sandbox-v2, a
# synthetic profile used only for the policy-shift set.
A2 = {"sandbox-v1": None, "sandbox-v2": (Decimal("5"), Decimal("500"))}


def po_unit_price(snap: dict, po_line: str) -> Decimal:
    return next(_d(l["unit_price"]["value"]) for l in rows(snap, "po_line") if l["id"] == po_line)


def within_a2(excess: Decimal, cover: Decimal, unit_price: Decimal, profile: str) -> bool:
    if excess <= 0:
        return True
    if A2[profile] is None:
        return False
    pct, absolute = A2[profile]
    return excess <= cover * pct / 100 and excess * unit_price <= absolute


def gap(snap: dict, po_line: str, invoice: str, profile: str = "sandbox-v1") -> Decimal:
    """The quantity still to be established by delivery evidence: zero when covered, or (sandbox-v2) when the
    excess over coverage is within the A2 tolerance."""
    excess = line_qty(snap, invoice, po_line) - coverage(snap, po_line, invoice)
    if within_a2(excess, coverage(snap, po_line, invoice), po_unit_price(snap, po_line), profile):
        return Decimal(0)
    return max(excess, Decimal(0))


def failing_blocking(snap: dict, invoice: str, profile: str = "sandbox-v1") -> list[str]:
    """Blocking comparisons that fail on this snapshot (POLICY A1, A4/DW, A5, A6, A7; equality passes)."""
    ids = by_id(snap)
    inv = ids[invoice]
    po = ids[inv["po"]]
    po_lines = [l for l in rows(snap, "po_line") if l["po"] == po["id"]]
    lines = [l for l in rows(snap, "invoice_line") if l["invoice"] == invoice]
    states = _invoice_states(snap)
    failing: list[str] = []
    for pl in po_lines:
        mine = [l for l in lines if l["po_line"] == pl["id"]]
        if not mine:
            continue
        po_price, ordered = _d(pl["unit_price"]["value"]), _d(pl["ordered_quantity"]["value"])
        for l in mine:
            diff = abs(_d(l["unit_price"]["value"]) - po_price)
            if diff / po_price * 100 > 10 or diff * _d(l["quantity"]["value"]) > 1000:
                failing.append("price")
        joint = line_qty(snap, invoice, pl["id"])
        net = net_receipts(snap, pl["id"])
        if net == 0:
            if joint > 0:
                failing.append("invoice_no_receipt")
        elif not within_a2(joint - (net - other_active(snap, pl["id"], invoice)),
                           net - other_active(snap, pl["id"], invoice), _d(pl["unit_price"]["value"]), profile):
            failing.append("invoice_vs_receipt")
        if other_active(snap, pl["id"], invoice) + joint > ordered * Decimal("1.10"):
            failing.append("invoice_vs_po_quantity")
    po_amount = sum((_d(pl["line_amount"]["value"]) for pl in po_lines), Decimal(0))
    cum = sum((_d(l["amount"]["value"]) for l in rows(snap, "invoice_line")
               if l["po_line"] in {pl["id"] for pl in po_lines} and states[l["invoice"]] != "cancelled"), Decimal(0))
    if cum > min(po_amount * Decimal("1.10"), po_amount + 1000):
        failing.append("invoice_vs_po_amount")
    if po["status"] == "closed":
        failing.append("closed_po")
    return sorted(set(failing))


def eligibility_current(snap: dict, invoice: str, po_line: str) -> bool:
    """Added in g6: did the invoice's latest recorded evaluation establish eligibility on the CURRENT
    records? False when that evaluation did not find it eligible, or when a receipt or an invoice on the PO line was
    posted or cancelled after it (the evaluation is stale). Decided from the recorded state alone."""
    inv = by_id(snap)[invoice]
    if inv.get("eligibility") != "eligible":
        return False
    t = inv["evaluated_at"]
    if any(r["po_line"] == po_line and r["posted_at"] > t for r in rows(snap, "receipt")):
        return False
    others = {l["invoice"] for l in rows(snap, "invoice_line") if l["po_line"] == po_line and l["invoice"] != invoice}
    for i in rows(snap, "invoice"):
        if i["id"] in others and ((i.get("posted_at") is not None and i["posted_at"] > t)
                                  or (i.get("cancelled_at") is not None and i["cancelled_at"] > t)):
            return False
    return True


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def t_res(gap0: Decimal, unrecorded: Decimal | None, delay: int | None, corrected: bool = False) -> tuple[float, str, int]:
    """T_res from the POLICY C3 workflow templates with surfaces/1 call counts.

    The templates are hand-built schedules, not proven globally optimal. A bounded search over valid workflows was
    built later as src/ap_sim/verification/reachability.py. Returns (T_res, resolving branch, calls); infinite when no
    valid workflow exists within the action limit.
    """
    if gap0 == 0:
        return 4 * ACTION_COST, "release", 5
    if delay is None:
        return float("inf"), "none", 0
    pre = 3 * ACTION_COST
    if corrected:
        # Added in g5: POLICY C1 event firing lets the confirmations view start one action before the
        # answer and still show it, so O_conf runs max(15, d - 5) -> max(20, d). The horizon is unchanged.
        k = ceil_div(max(0, delay - ACTION_COST - pre), ADVANCE_MAX)
        seen = max(pre + ACTION_COST, delay)
        if unrecorded >= gap0:
            t, branch, calls = seen + 4 * ACTION_COST, "release", 9 + k
        elif unrecorded <= 0:
            t, branch, calls = seen + ACTION_COST, "keep_blocked", 6 + k
        else:
            raise ValueError("partial delivery: excluded from v1 (P15)")
        return (float("inf"), "none", calls) if calls > ACTION_LIMIT else (t, branch, calls)
    k = ceil_div(max(0, delay - pre), ADVANCE_MAX)
    if unrecorded >= gap0:
        t, branch, calls = max(pre, delay) + 5 * ACTION_COST, "release", 9 + k
    elif unrecorded <= 0:
        t, branch, calls = max(pre, delay) + 2 * ACTION_COST, "keep_blocked", 6 + k
    else:
        raise ValueError("partial delivery: excluded from v1 (P15)")
    if calls > ACTION_LIMIT:
        return float("inf"), "none", calls
    return t, branch, calls
