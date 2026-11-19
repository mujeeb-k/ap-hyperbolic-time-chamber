"""The tool layer shared by both surfaces: validation, stale rule, horizon, call limit,
virtual-time charges, observation ids, and the episode record used by replay and the grader.

Agents see only the tool results. Grader-only data (snapshots, attempts) goes to `self.log`.
"""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal

from pydantic import ValidationError

from ap_sim.app import catalog
from ap_sim.app.controls import control_ids
from ap_sim.app.session import AppSession
from ap_sim.domain.report import FinishReport, describe_errors, report_model
from ap_sim.events.clock import ACTION_COST, ACTION_LIMIT, ADVANCE_MAX, format_time
from ap_sim.interfaces.canonical import dumps, encode

# Characters of canonical JSON. Raised from 8000 to 20000 on 2026-10-31, above the enumerated maximum over every
# frozen case (11,930 reached, 13,968 bound over the 40-call budget and the widest rows). Nothing under 8000 changes.
SIZE_LIMIT = 20000
TRUNCATION_NOTICE = ("This page is too long to show in full: the last {n} record(s) of the page are not shown. The "
                     "action result, the changed records, the actions and the navigation are complete.")


class EnvironmentFailure(RuntimeError):
    """An infrastructure failure (producer error): counted, never the agent's fault."""


def truncate(obs: dict, limit: int) -> dict:
    """The backstop for an observation over the cap: drop the page body's records from the
    end, one at a time, until it fits, and say so visibly (a `truncated` notice; `omitted_records` on each cut region).
    The result region, the changed records, the actions and the navigation are never cut. Never raises: if the kept
    parts alone are over the cap, the observation is delivered with every page record omitted."""
    body = set(catalog.PAGE_REGIONS[obs["page"]])
    regions = [dict(r, records=list(r["records"])) for r in obs["regions"]]
    cut = [r for r in regions if r["role"] in body and r["records"]]
    omitted = 0
    out = {**obs, "regions": regions}
    while cut and len(dumps(out)) > limit:
        region = cut[-1]
        region["records"].pop()
        region["omitted_records"] = region.get("omitted_records", 0) + 1
        omitted += 1
        out["truncated"] = TRUNCATION_NOTICE.format(n=omitted)
        if not region["records"]:
            cut.pop()
    return out


class ToolLayer:
    def __init__(self, scenario, surface, schema: str = catalog.SCHEMA):
        self.session = AppSession(scenario, schema)
        self.schema = schema
        self.surface = surface
        self.ids = control_ids(scenario.seed)
        self.horizon = scenario.horizon
        self.calls = 0
        self.done = False
        self.termination: str | None = None
        self.report: FinishReport | None = None
        self._obs_n = 0
        self.last_obs: dict | None = None
        self.log: dict = {"scenario": scenario.name, "seed": scenario.seed, "surface": surface.name,
                          "schema": schema, "presentation_profile": catalog.PRESENTATION_PROFILE, "steps": [],
                          "policy": self.session.policy_profile}
        surface.attach(self.session, self.ids)
        self.initial_observation = self._deliver()
        self.log["initial_observation"] = self.initial_observation
        self.log["initial_hash"] = self.session.engine.business_hash()

    # Internals ------------------------------------------------------------------------------------
    @property
    def clock(self) -> int:
        return self.session.engine.clock

    def _deliver(self) -> dict:
        obs = self.surface.produce()
        self._obs_n += 1
        obs = {"schema": self.schema, "observation_id": f"o{self._obs_n}", **obs}
        if len(dumps(obs)) > SIZE_LIMIT:
            obs = truncate(obs, SIZE_LIMIT)
        self.last_obs = obs
        return obs

    def _executed(self, tool: str, args: dict) -> dict:
        """An accepted call that shows a page: the step and its state change are recorded first, then the page is
        rendered into that same step, so a rendering failure can never drop an executed action from the log."""
        env = self._result(tool, args, True)
        env["observation"] = self._deliver()
        return env

    def _result(self, tool: str, args: dict, ok: bool, error=None, observation=None) -> dict:
        env = {"ok": ok, "clock": format_time(self.clock), "error": error, "observation": observation}
        eng = self.session.engine
        self.log["steps"].append({
            "i": len(self.log["steps"]) + 1, "tool": tool, "args": args, "result": env,
            "clock": self.clock, "state_version": eng.state_version, "business_hash": eng.business_hash(),
            "snapshot": eng.store.snapshot()})
        if self.calls >= ACTION_LIMIT and not self.done:
            self.done, self.termination = True, "action_limit"
        return env

    def _reject(self, tool: str, args: dict, code: str, message: str) -> dict:
        self.surface.charge(min(ACTION_COST, self.horizon - self.clock))
        return self._result(tool, args, False, {"code": code, "message": message})

    def _begin(self, tool: str, args: dict, cost: int) -> dict | None:
        """Returns an error envelope if the call cannot proceed; otherwise counts the call."""
        if self.done:
            return {"ok": False, "clock": format_time(self.clock),
                    "error": {"code": "episode_over", "message": "The episode has ended."}, "observation": None}
        self.calls += 1
        if self.clock + cost > self.horizon:
            return self._reject(tool, args, "horizon", "This action cannot complete before the business horizon.")
        return None

    def _control(self, element_id) -> dict | None:
        for region in self.last_obs["regions"]:
            for c in region["controls"]:
                if c["id"] == element_id:
                    return c
        return None

    def _key(self, element_id: str) -> str:
        return {v: k for k, v in self.ids.items()}[element_id]

    # Tools ----------------------------------------------------------------------------------------
    def observe(self) -> dict:
        if (err := self._begin("observe", {}, ACTION_COST)) is not None:
            return err
        self.surface.observe()
        return self._executed("observe", {})

    def click(self, element_id: str) -> dict:
        args = {"element_id": element_id}
        if (err := self._begin("click", args, ACTION_COST)) is not None:
            return err
        if not isinstance(element_id, str):
            return self._reject("click", args, "malformed", "element_id must be a string.")
        control = self._control(element_id)
        if control is None:
            return self._reject("click", args, "unknown_element", "No such control on the current page.")
        if control["role"] not in ("button", "link"):
            return self._reject("click", args, "wrong_control_kind", "Use fill for a textbox.")
        if control["role"] == "button" and self.session.engine.state_version != self.last_obs["state_version"]:
            return self._reject("click", args, "stale_state",
                                "The state changed since your last observation; observe again before acting.")
        n_ops = len(self.session.history)
        self.surface.click(self._key(element_id), element_id)
        if len(self.session.history) > n_ops:  # tag the operation with the step that submitted it
            self.session.history[-1]["step"] = len(self.log["steps"]) + 1
        return self._executed("click", args)

    def fill(self, element_id: str, text: str) -> dict:
        args = {"element_id": element_id, "text": text}
        if (err := self._begin("fill", args, ACTION_COST)) is not None:
            return err
        if not isinstance(element_id, str) or not isinstance(text, str) or len(text) > 100:
            return self._reject("fill", args, "malformed", "fill needs an element_id and a text of at most 100 characters.")
        control = self._control(element_id)
        if control is None:
            return self._reject("fill", args, "unknown_element", "No such control on the current page.")
        if control["role"] != "textbox":
            return self._reject("fill", args, "wrong_control_kind", "Only a textbox can be filled.")
        self.surface.fill(self._key(element_id), element_id, text)
        return self._executed("fill", args)

    def select(self, element_id: str, option: str) -> dict:
        args = {"element_id": element_id, "option": option}
        if (err := self._begin("select", args, ACTION_COST)) is not None:
            return err
        control = self._control(element_id) if isinstance(element_id, str) else None
        if control is None:
            return self._reject("select", args, "unknown_element", "No such control on the current page.")
        return self._reject("select", args, "wrong_control_kind", "Only a combobox supports select.")

    def advance_time(self, minutes: int) -> dict:
        args = {"minutes": minutes}
        if self.done:
            return self._begin("advance_time", args, 0)
        if not isinstance(minutes, int) or isinstance(minutes, bool) or not 1 <= minutes <= ADVANCE_MAX:
            self.calls += 1
            return self._reject("advance_time", args, "malformed", f"minutes must be an integer from 1 to {ADVANCE_MAX}.")
        if self.clock >= self.horizon:
            self.calls += 1
            return self._result("advance_time", args, False, {"code": "horizon", "message": "The business horizon is reached."})
        self.calls += 1
        self.surface.advance(min(minutes, self.horizon - self.clock))
        return self._result("advance_time", args, True)

    def finish(self, report: dict) -> dict:
        args = {"report": report}
        if self.done:
            return self._begin("finish", args, 0)
        self.calls += 1
        try:
            parsed = report_model(self.schema).model_validate(report)   # surfaces/4: FinishReportV2
        except ValidationError as exc:  # rejected: counts as a call (C1) but costs no time, like any finish (C1)
            problems = describe_errors(exc)
            return self._result("finish", args, False, {
                "code": "invalid_report", "problems": problems,
                "message": f"The report was not accepted ({exc.error_count()} problem(s)); fix the listed fields "
                           "and call finish again. Nothing was recorded."})
        self.report, self.done, self.termination = parsed, True, "finish"
        return self._result("finish", args, True)

    def coverage(self, invoice_id: str) -> dict:
        """The coverage calculator: the records that determine the invoice's available coverage, with
        ids and versions, and the figures computed from them, on the current records. Read-only; it costs ACTION_COST
        and one call like any tool. Its result is not an observation (no observation_id), so it cannot be cited.
        It reads engine state, not a page, so it is the same on the structured and the browser surface."""
        args = {"invoice_id": invoice_id}
        if (err := self._begin("coverage", args, ACTION_COST)) is not None:
            return err
        eng = self.session.engine
        if not isinstance(invoice_id, str) or invoice_id != eng.current_invoice:
            return self._reject("coverage", args, "unknown_invoice", "No such invoice in this case.")
        self.surface.charge(ACTION_COST)
        payload = coverage_payload(eng)
        result = self._result("coverage", args, True)
        result["coverage"] = payload   # the logged step holds the same dict
        return result

    # Episode record ---------------------------------------------------------------------------------
    def episode_log(self) -> dict:
        eng = self.session.engine
        return {**self.log, "termination": self.termination, "calls": self.calls,
                "report": self.report.model_dump(mode="json") if self.report else None,
                "operations": self.session.history, "attempts": [asdict(a) for a in eng.attempts],
                "final_snapshot": eng.store.snapshot(), "final_hash": eng.business_hash(), "final_clock": eng.clock,
                "current_invoice": eng.current_invoice}


def coverage_payload(eng) -> dict:
    """What the calculator returns: inputs (receipts, reversals, other active invoices; ids and versions as on the pages)
    and figures. Numbers and ids only: no decision, no action, nothing that is not in the recorded state."""
    from ap_sim.domain import calculations as calc
    store, inv_id, po_line = eng.store, eng.current_invoice, eng.po_line

    def qty(d):
        return {"value": format(d.normalize(), "f"), "unit": "EA"}

    movements = sorted(store.receipts(po_line), key=lambda r: (r.posted_at, r.id))
    active = calc.active_invoice_ids(store)
    others, cancelled = [], []
    for i in sorted(store.invoices(), key=lambda i: (i.posted_at if i.posted_at is not None else 0, i.id)):
        lines = store.invoice_lines(i.id, po_line)
        if i.id == inv_id or not lines:
            continue
        if i.id in active:
            others.append({"id": i.id, "version": i.version,
                           "quantity": qty(sum((l.quantity.value for l in lines), Decimal(0)))})
        else:
            cancelled.append({"id": i.id, "version": i.version})
    line = store.invoice_lines(inv_id, po_line)[0]
    invoice_qty = calc.joint_invoice_quantity(store, inv_id, po_line)
    cover = calc.available_coverage(store, po_line, inv_id)
    return {
        "invoice": inv_id, "po_line": po_line,
        "invoice_line": {"id": line.id, "version": line.version, "quantity": qty(invoice_qty)},
        "receipts": [{"id": r.id, "version": r.version, "posted_at": encode("time", r.posted_at),
                      "quantity": qty(r.quantity.value)} for r in movements if r.movement == "receipt"],
        "reversals": [{"id": r.id, "version": r.version, "posted_at": encode("time", r.posted_at), "reverses": r.reverses,
                       "quantity": qty(r.quantity.value)} for r in movements if r.movement == "reversal"],
        "other_active_invoices": others,
        "cancelled_invoices_not_counted": cancelled,
        "net_posted_receipts": qty(calc.net_posted(store, po_line)),
        "other_active_quantity": qty(calc.other_active_invoiced(store, po_line, inv_id)),
        "available_coverage": qty(cover),
        "invoice_quantity": qty(invoice_qty),
        "invoice_quantity_minus_available_coverage": qty(invoice_qty - cover),
    }


class AgentTools:
    """The only object an agent receives: the tools and the reset observation. Nothing else. (`coverage` is the v2
    calculator; agents on surfaces/3 and earlier never call it.)"""

    def __init__(self, layer: ToolLayer):
        self._layer = layer
        self.initial_observation = layer.initial_observation

    def observe(self) -> dict:
        return self._layer.observe()

    def click(self, element_id: str) -> dict:
        return self._layer.click(element_id)

    def fill(self, element_id: str, text: str) -> dict:
        return self._layer.fill(element_id, text)

    def select(self, element_id: str, option: str) -> dict:
        return self._layer.select(element_id, option)

    def advance_time(self, minutes: int) -> dict:
        return self._layer.advance_time(minutes)

    def finish(self, report: dict) -> dict:
        return self._layer.finish(report)

    def coverage(self, invoice_id: str) -> dict:
        return self._layer.coverage(invoice_id)

