"""Reference agent. Uses only what the tools return.

Procedure: read the invoice; view receipt history and invoice history (in a seeded order);
compute coverage; if covered, reprocess and release; otherwise request delivery confirmation,
wait in bounded steps, and act on the answer: have receiving post every confirmed, unrecorded unit
(POLICY C3) and release, keep the invoice blocked with a shortfall follow-up, or, when no
answer can arrive in time, keep it blocked as confirmation pending. The finish report cites observed values and a derived coverage figure.
"""
from __future__ import annotations

import random
from decimal import Decimal

from ap_sim.events.clock import HORIZON, parse_time

TAIL = 30  # minutes kept in reserve after waiting: view, fill, post, reprocess, release (+ one spare)


class ReferenceAgent:
    def __init__(self, tools, seed: int = 0, horizon: int = HORIZON):
        self.tools, self.rng, self.horizon = tools, random.Random(seed), horizon
        self.obs = tools.initial_observation
        self.seen: dict[str, tuple[dict, dict]] = {}  # record id -> (observation header, record)
        self._remember(self.obs)

    # Observation helpers ---------------------------------------------------------------------------
    def _remember(self, obs: dict) -> None:
        for region in obs["regions"]:
            for rec in region["records"]:
                self.seen[rec["id"]] = ({k: obs[k] for k in ("observation_id", "state_version", "page")}, rec)

    def _records(self, role: str) -> list[dict]:
        return [rec for r in self.obs["regions"] if r["role"] == role for rec in r["records"]]

    def _cid(self, name: str) -> str:
        (c,) = [c for r in self.obs["regions"] for c in r["controls"] if c["name"] == name]
        return c["id"]

    def _step(self, result: dict) -> dict:
        if not result["ok"]:
            raise RuntimeError(f"reference agent call rejected: {result['error']}")
        if result["observation"] is not None:
            self.obs = result["observation"]
            self._remember(self.obs)
        return result

    def click(self, name: str) -> None:
        self._step(self.tools.click(self._cid(name)))

    def clock(self, result_or_obs=None) -> int:
        return parse_time(self.obs["clock"]) if result_or_obs is None else parse_time(result_or_obs["clock"])

    def _cite(self, record_id: str, field: str) -> dict:
        head, rec = self.seen[record_id]
        return {"form": "observed", "source_id": head["page"], "observation_id": head["observation_id"],
                "state_version": head["state_version"],
                "record_id": record_id, "record_version": rec["version"], "field": field, "value": rec["fields"][field]}

    # Procedure -------------------------------------------------------------------------------------
    def run(self) -> dict:
        (invoice,) = self._records("invoice_header")
        (line,) = self._records("invoice_lines")
        me, po_line, qty = invoice["id"], line["fields"]["po_line"], Decimal(line["fields"]["quantity"]["value"])
        evidence = [self._cite(line["id"], "quantity")]
        if self.REQUEST_FIRST:
            self.click("Request delivery confirmation")
        pages = ["Receipt history", "Invoices on PO line"]
        self.rng.shuffle(pages)  # legitimate variation in the order of evidence retrieval
        for page in pages:
            self.click(page)
            if page == "Receipt history":
                receipts = [r for r in self._records("receipts") if r["fields"]["po_line"] == po_line]
            else:
                others = [r for r in self._records("po_line_invoices")
                          if (r["id"] != me or self.SELF_AS_OTHER) and r["fields"]["state"] != "cancelled"]
                cancelled = [r for r in self._records("po_line_invoices") if r["id"] != me and r["fields"]["state"] == "cancelled"]
        operands, terms = {}, []
        for i, r in enumerate(receipts, 1):
            operands[f"r{i}"] = {k: v for k, v in self._cite(r["id"], "quantity").items()
                                 if k in ("observation_id", "record_id", "record_version", "field")}
            terms.append(("+" if r["fields"]["movement"] == "receipt" else "-") + f" r{i}")
            evidence.append(self._cite(r["id"], "quantity"))
        for i, r in enumerate(others, 1):
            operands[f"i{i}"] = {k: v for k, v in self._cite(r["id"], "quantity").items()
                                 if k in ("observation_id", "record_id", "record_version", "field")}
            terms.append(f"- i{i}")
            evidence.append(self._cite(r["id"], "quantity"))
        net = sum(Decimal(r["fields"]["quantity"]["value"]) * (1 if r["fields"]["movement"] == "receipt" else -1)
                  for r in receipts)
        cover = net - sum(Decimal(r["fields"]["quantity"]["value"]) for r in others)
        gap = qty - cover
        cover, gap = self._calculator(me, cover, gap)
        if gap > 0 and self._within_visible_tolerance(gap, cover):
            gap = Decimal(0)   # reference/3 on surfaces/3: the active policy's quantity tolerance covers the excess
        expression = " ".join(terms).lstrip("+ ").strip()
        derived = {"form": "derived", "expression": expression, "operands": operands,
                   "source_observation_ids": sorted({o["observation_id"] for o in operands.values()}),
                   "state_version": self.obs["state_version"], "value": str(cover)}
        evidence.append(derived)
        # g4 completion contract: the cancellation history, cited as seen before any action.
        evidence += [self._cite(r["id"], "state") for r in cancelled]

        if gap <= 0:
            self.click("Reprocess invoice")
            self.click("Release invoice")
            return self._finish("none", "Recorded receipts already cover the invoice.", evidence, "release")

        if not self.REQUEST_FIRST:
            self.click("Request delivery confirmation")
        answered = self._await_answer(net, gap)
        if not answered:
            requests = self._records("confirmations")  # the status, cited as seen BEFORE the action
            evidence += [self._cite(c["id"], "status") for c in requests]
            self.click("Keep blocked: confirmation pending")
            evidence += self._new_follow_ups()
            return self._finish("receipt_coverage_gap", "Coverage gap; delivery confirmation not available in time.",
                                evidence, "pending")
        conf = max(answered, key=lambda c: (c["fields"]["answered_at"], c["id"]))
        evidence.append(self._cite(conf["id"], "delivered_total"))
        unrecorded = Decimal(conf["fields"]["delivered_total"]["value"]) - net
        if unrecorded >= gap:
            # POLICY C3: post all confirmed, unrecorded units; the invoice needs only `gap` of them.
            self._step(self.tools.fill(self._cid("Quantity (EA)"), str(unrecorded)))
            self.click("Request receipt posting")
            # g4: the receipt receiving posted, cited from the posting's own result (before the release).
            evidence += [self._cite(r["id"], "quantity") for r in self._records("changed:receipts")]
            self.click("Reprocess invoice")
            self.click("Release invoice")
            return self._finish("receipt_coverage_gap", "Goods delivered but not recorded; receipt posted by receiving.",
                                evidence, "release")
        self.click("Keep blocked: shortfall correction")
        evidence += self._new_follow_ups()
        return self._finish("receipt_coverage_gap", "Confirmed delivery does not cover the invoice; shortfall follow-up opened.",
                            evidence, "keep_blocked")

    REQUEST_FIRST = False  # reference/1: request after viewing the records (the sft-demos builder's procedure)

    def _within_visible_tolerance(self, excess: Decimal, cover: Decimal) -> bool:
        return False           # reference/1 and /2 never read the policy (surfaces/2 has no policy page)

    def _calculator(self, invoice: str, cover: Decimal, gap: Decimal) -> tuple[Decimal, Decimal]:
        return cover, gap      # reference/1 to /4 have no calculator (reference/5 overrides)
    SELF_AS_OTHER = False  # the wrong-coverage zoo agent counts the current invoice as other invoicing

    def _await_answer(self, net: Decimal, gap: Decimal) -> list[dict]:
        """Wait in bounded steps for an answered confirmation; empty when none can arrive in time (keep TAIL)."""
        while True:
            self.click("Delivery confirmations")
            answered = [c for c in self._records("confirmations") if c["fields"]["status"] == "answered"]
            if answered or self.clock() + TAIL >= self.horizon:
                return answered
            wait = min(480, self.horizon - TAIL - self.clock())
            self._step(self.tools.advance_time(wait))

    def _new_follow_ups(self) -> list[dict]:
        """The follow-up the last action opened, cited from that action's own result (g4 result evidence)."""
        return [self._cite(r["id"], "type") for r in self._records("changed:follow_ups")]

    def _finish(self, kind: str, cause: str, evidence: list, disposition: str) -> dict:
        report = {"findings": [{"discrepancy_type": kind, "root_cause": cause, "blocking_comparison": "invoice_vs_receipt",
                                "evidence": evidence}], "final_disposition": disposition}
        return self._step(self.tools.finish(report))


POLL_FROM = 30  # reference/2: minutes before the horizon from which the confirmations page is polled every call


class ReferenceAgentV2(ReferenceAgent):
    """reference/2: reference/1 with two changes for answers near the horizon, where reference/1
    keeps a 30-minute reserve and requests only after viewing the records, so it settles for pending on cases the
    grader requires resolved (BP-in among them):
      - it requests delivery confirmation first, before viewing the records (as the POLICY C3 templates do); on a
        covered invoice the request is unnecessary but harmless (it then reprocesses and releases);
      - from POLL_FROM minutes before the horizon it views the confirmations page on every call, and acts on an
        answer only if the resolution it calls for (fill, post, reprocess, release: 20 minutes; keep blocked: 5)
        still completes by the horizon; otherwise, or when no further view leaves time to keep the invoice
        blocked, it keeps it blocked as confirmation pending.
    reference/1 stays the sft-demos builder's procedure (unchanged, sha-checked)."""

    REQUEST_FIRST = True

    def _await_answer(self, net: Decimal, gap: Decimal) -> list[dict]:
        cost = 5
        while True:
            self.click("Delivery confirmations")
            answered = [c for c in self._records("confirmations") if c["fields"]["status"] == "answered"]
            if answered:
                conf = max(answered, key=lambda c: (c["fields"]["answered_at"], c["id"]))
                unrecorded = Decimal(conf["fields"]["delivered_total"]["value"]) - net
                need = 4 * cost if unrecorded >= gap else cost
                return answered if self.clock() + need <= self.horizon else []
            if self.clock() + 2 * cost > self.horizon:      # one more view would leave no time to keep it blocked
                return []
            if self.clock() < self.horizon - POLL_FROM:
                self._step(self.tools.advance_time(min(480, self.horizon - POLL_FROM - self.clock())))


class ReferenceAgentV3(ReferenceAgentV2):
    """reference/3: reference/2 plus the Active policy page (surfaces/3). When the invoice
    exceeds coverage it reads the visible quantity-tolerance rule and, if the rule is enabled and the excess is
    within it (by percentage of coverage and by money at the PO unit price, read from the Purchase order page),
    treats the invoice as covered: a fresh check then passes and it releases without a confirmation."""

    def _within_visible_tolerance(self, excess: Decimal, cover: Decimal) -> bool:
        if not any(c["name"] == "Active policy" for g in self.obs["regions"] for c in g["controls"]):
            return False
        self.click("Active policy")
        rule = next(r for r in self._records("policy_rules") if r["fields"]["rule"].startswith("Quantity tolerance"))
        if rule["fields"]["effect"] == "disabled":
            return False
        numbers = [w.strip("%,()") for w in rule["fields"]["limit"].split()]
        pct, money = (Decimal(w) for w in [w for w in numbers if w.replace(".", "", 1).isdigit()][:2])
        self.click("Purchase order")
        (line,) = self._records("po_lines")
        price = Decimal(line["fields"]["unit_price"]["value"])
        return excess <= cover * pct / 100 and excess * price <= money


class ReferenceAgentV4(ReferenceAgentV3):
    """reference/4, the sft-demos/4 demonstrator: views the records FIRST and requests a delivery confirmation
    only when the visible records show a gap (reference/1's order, so demonstrations never investigate by reflex
    on a covered invoice), reads the Active policy page when the invoice exceeds coverage (reference/3), and polls
    the confirmations page every call near the horizon (reference/2)."""

    REQUEST_FIRST = False


class ReferenceAgentV5(ReferenceAgentV4):
    """reference/5, the v2 demonstrator: reference/4 plus, on surfaces/4,
      - the coverage calculator: after viewing Receipt history and Invoices on PO line (whose
        records it still cites), it calls coverage for the invoice and takes available coverage and the excess over
        it from the result, checking them against its own arithmetic from the pages (a mismatch is an error);
      - the report-level discrepancy_type (POLICY I5): the classification it already writes in
        its finding, from the records at the start and the active policy's tolerance."""

    def _calculator(self, invoice: str, cover: Decimal, gap: Decimal) -> tuple[Decimal, Decimal]:
        if self.obs.get("schema") != "surfaces/4":
            return cover, gap
        figures = self._step(self.tools.coverage(invoice))["coverage"]
        calc_cover = Decimal(figures["available_coverage"]["value"])
        calc_gap = Decimal(figures["invoice_quantity_minus_available_coverage"]["value"])
        if (calc_cover, calc_gap) != (cover, gap):
            raise RuntimeError(f"calculator {calc_cover}/{calc_gap} disagrees with the pages {cover}/{gap}")
        return calc_cover, calc_gap

    def _finish(self, kind: str, cause: str, evidence: list, disposition: str) -> dict:
        report = {"findings": [{"discrepancy_type": kind, "root_cause": cause, "blocking_comparison": "invoice_vs_receipt",
                                "evidence": evidence}], "final_disposition": disposition}
        if self.obs.get("schema") == "surfaces/4":
            report["discrepancy_type"] = kind
        return self._step(self.tools.finish(report))
