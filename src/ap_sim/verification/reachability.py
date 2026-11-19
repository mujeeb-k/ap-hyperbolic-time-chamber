"""Full-knowledge reachability checker. VERIFICATION ONLY: it reads the hidden world (the warehouse's answer delay
and the physical deliveries), so it is never a baseline, never an agent, and never reachable through any agent tool.

For one scenario it searches a bounded family of plans and runs every plan through the agent's ACTUAL interface
(`ToolLayer` with the structured surface, the same six tools and limits), then grades it with the current grader.
A plan is built from these steps, in three phases around the confirmation request Q and the wait W:

  A (before Q)   any of R (Receipt history), I (Invoices on PO line), X (Reprocess, only without Q)
  Q              Request delivery confirmation (optional)
  B (Q..W)       any of R, I, C (Delivery confirmations), F (fill the quantity: the hidden unrecorded units);
                 X here too when there is no wait
  W              advance time to the answer (from the hidden delay), to one action cost before it (the next view's
                 observation, delivered when its cost has elapsed, then shows the answer), or to the last moment
                 before the horizon
  D (after W)    any of C, R, I, F, P (Request receipt posting), X
  T              the terminal action: L (Release), S (Keep blocked: shortfall) or N (Keep blocked: pending)
  finish         a report citing every observed record version of the roles the completion contract names

Views may be skipped and the request, views and fill may sit in any phase, so the search can beat the grader's
hand-built T_res templates (`grader.arithmetic.t_res`).
Reachability is judged from the outcome REACHED, not from reward (reward is defined against the grader's own
expected outcome):
  reachable      some plan reaches the expected outcome with reward 1
  no rival       no plan reaches a different RESOLVING outcome (release or keep blocked) with no critical constraint
                 and an accepted report; a pending case where a release completes cleanly before the horizon would be
                 a grader bug. Business-only rivals (report-invalid) are listed as a diagnostic.
  earliest       the earliest resolution (the deciding action's end) among reward-1 plans, against the template T_res

    python -m ap_sim.verification.reachability
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from ap_sim.events.clock import parse_time
from ap_sim.grader import arithmetic as ar
from ap_sim.grader.grader import REPORT_CRITICAL, grade
from ap_sim.interfaces.structured import StructuredSurface
from ap_sim.interfaces.tools import AgentTools, ToolLayer

NAMES = {"R": "Receipt history", "I": "Invoices on PO line", "C": "Delivery confirmations", "Q": "Request delivery confirmation",
         "P": "Request receipt posting", "X": "Reprocess invoice", "L": "Release invoice",
         "S": "Keep blocked: shortfall correction", "N": "Keep blocked: confirmation pending"}
TERMINAL = {"L": ("release", "release"), "S": ("keep_blocked", "keep_blocked_shortfall"),
            "N": ("pending", "keep_blocked_pending")}
CITE = {"invoice_lines": ("quantity",), "receipts": ("quantity",), "changed:receipts": ("quantity",),
        "po_line_invoices": ("quantity", "state"), "confirmations": ("status", "delivered_total"),
        "changed:confirmations": ("status", "delivered_total"), "follow_ups": ("type",), "changed:follow_ups": ("type",)}


@dataclass(frozen=True)
class Plan:
    a: tuple[str, ...]
    q: bool
    b: tuple[str, ...]
    w: str | None          # None, "answer", "before_answer" or "tail"
    d: tuple[str, ...]
    t: str

    def __str__(self) -> str:
        w = {"answer": "W(answer)", "before_answer": "W(answer-5)", "tail": "W(tail)", None: ""}[self.w]
        return " ".join([*self.a, *(["Q"] if self.q else []), *self.b, *([w] if w else []), *self.d, self.t, "finish"])


def plans() -> list[Plan]:
    """The bounded plan family, in a canonical order inside each phase (order within a phase changes neither the
    clock nor what is seen before the deciding action)."""
    out = []
    for t in "LSN":
        for x_a in ((), ("X",)) if t == "L" else ((),):
            for ri in itertools.product((None, "A"), repeat=2):           # no request: only the A phase exists
                a = tuple(n for n, g in zip("RI", ri) if g == "A") + x_a
                out.append(Plan(a, False, (), None, (), t))
        for w in (None, "answer", "before_answer", "tail"):
            for ri in itertools.product((None, "A", "B", "D"), repeat=2):
                for c in (None, "B", "D"):
                    fp = ((None, None), ("B", "D"), ("D", "D")) if t == "L" else ((None, None),)
                    for f, p in fp:
                        for x in ((None, "B" if w is None else "D") if t == "L" else (None,)):
                            g = {"R": ri[0], "I": ri[1], "C": c, "F": f, "P": p, "X": x}
                            if w is None and any(v == "D" for v in g.values()):
                                continue                                      # no wait: B and D are one phase
                            def pick(ph, g=g):
                                return tuple(n for n in "CRIFPX" if g[n] == ph)
                            out.append(Plan(tuple(n for n in "RI" if g[n] == "A"), True, pick("B"), w, pick("D"), t))
    return out


class _Run:
    def __init__(self, scenario):
        self.scenario = scenario
        self.layer = ToolLayer(scenario, StructuredSurface(), "surfaces/2")
        self.tools = AgentTools(self.layer)
        self.obs = self.tools.initial_observation
        self.seen: list[tuple[dict, dict, str]] = []
        self._remember(self.obs)
        self.requested_at = None
        self.blind_fill = False

    def answer_seen(self) -> bool:
        return any(role.endswith("confirmations") and rec["fields"].get("status") == "answered"
                   for _, rec, role in self.seen)

    def _remember(self, obs):
        for g in obs["regions"]:
            for rec in g["records"]:
                self.seen.append((obs, rec, g["role"]))

    def _take(self, result):
        if result.get("observation"):
            self.obs = result["observation"]
            self._remember(self.obs)

    def click(self, step: str):
        if step == "Q":
            self.requested_at = parse_time(self.obs["clock"])
        cid = next(c["id"] for g in self.obs["regions"] for c in g["controls"] if c["name"] == NAMES[step])
        self._take(self.tools.click(cid))

    def fill(self, qty: str):
        cid = next(c["id"] for g in self.obs["regions"] for c in g["controls"] if c["name"] == "Quantity (EA)")
        self._take(self.tools.fill(cid, qty))

    def wait_until(self, target: int):
        while self.layer.clock < target and not self.layer.done:
            r = self.tools.advance_time(min(ar.ADVANCE_MAX, target - self.layer.clock))
            if not r["ok"]:
                break

    def report(self, disposition: str, kind: str) -> dict:
        evidence, done = [], set()
        for obs, rec, role in self.seen:
            for fld in CITE.get(role, ()):
                key = (obs["observation_id"], rec["id"], rec["version"], fld)
                if key in done or rec["fields"].get(fld) is None:
                    continue
                done.add(key)
                evidence.append({"form": "observed", "source_id": obs["page"], "observation_id": obs["observation_id"],
                                 "state_version": obs["state_version"], "record_id": rec["id"],
                                 "record_version": rec["version"], "field": fld, "value": rec["fields"][fld]})
        return {"findings": [{"discrepancy_type": kind, "root_cause": "Reachability check.",
                              "blocking_comparison": "invoice_vs_receipt", "evidence": evidence}],
                "final_disposition": disposition}


def _facts(scenario) -> tuple:
    initial = {}
    for r in scenario.records:
        initial.setdefault(r.kind, []).append(r.model_dump(mode="json"))
    inv = scenario.current_invoice
    po_line = next(l["po_line"] for l in initial["invoice_line"] if l["invoice"] == inv)
    return ar.gap(initial, po_line, inv), ar.net_receipts(initial, po_line), po_line


def execute(scenario, plan: Plan) -> dict:
    gap0, net0, po_line = _facts(scenario)
    delay = scenario.world.confirmation_delay
    run = _Run(scenario)
    disposition, key = TERMINAL[plan.t]

    def do(step):
        if step == "F":   # full knowledge: every delivered, unrecorded unit (POLICY C3), known before any answer
            run.blind_fill |= not run.answer_seen()
            seen_by = run.requested_at + delay if run.requested_at is not None and delay is not None else 0
            run.fill(str(scenario.world.delivered_by(po_line, seen_by) - net0))   # what the answer will report
        else:
            run.click(step)

    for s in plan.a:
        do(s)
    if plan.q:
        do("Q")
    for s in plan.b:
        do(s)
    if plan.w == "answer" and delay is not None:
        run.wait_until(run.requested_at + delay)
    elif plan.w == "before_answer" and delay is not None:
        run.wait_until(run.requested_at + delay - ar.ACTION_COST)
    elif plan.w == "tail":
        run.wait_until(scenario.horizon - ar.ACTION_COST * (len(plan.d) + 1))
    for s in plan.d:
        do(s)
    do(plan.t)
    run.tools.finish(run.report(disposition, "none" if gap0 == 0 else "receipt_coverage_gap"))
    log = run.layer.episode_log()
    v = grade(scenario, log)
    op = next((o for o in log["operations"] if o["action"] == key and o["attempt"]["accepted"]), None)
    return {"plan": str(plan), "reached": v.reached, "reward": v.reward, "critical": v.critical,
            "business_critical": sorted(set(v.critical) - REPORT_CRITICAL), "expected": v.expected, "t_res": v.t_res,
            "resolved_at": op["attempt"]["at"] + ar.ACTION_COST if op else None, "calls": log["calls"],
            "finish_accepted": v.diagnostics["finish_accepted"], "blind_fill": run.blind_fill}


def check(scenario) -> dict:
    """Run every plan; return the verdict on reachability, rivals and the earliest resolution."""
    results = [execute(scenario, p) for p in plans()]
    expected, t_res = results[0]["expected"], results[0]["t_res"]
    good = [r for r in results if r["reward"] == 1 and not r["blind_fill"]]
    # A rival: a RESOLVING outcome other than the expected one, reached with no critical constraint at all and an
    # accepted report (pending is never a rival: it is acceptable only when no resolution is reachable, which is
    # exactly what this checks for a pending case). Business-only rivals (valid business, report-invalid) are a
    # diagnostic: an agent cannot evidence them, e.g. a posting made without viewing the answer it relies on.
    resolving = [r for r in results if r["reached"] in ("release", "keep_blocked") and r["reached"] != expected]
    # A plan that filled the quantity before any answer was observed needs the hidden truth: such a rival is
    # listed separately (an agent could only have guessed it).
    valid = [r for r in resolving if not r["critical"] and r["finish_accepted"]]
    rivals = [r for r in valid if not r["blind_fill"]]
    blind_rivals = [r for r in valid if r["blind_fill"]]
    business_rivals = [r for r in resolving if r not in valid and not r["business_critical"]]
    earliest = min(good, key=lambda r: (r["resolved_at"], r["calls"])) if good else None
    fewest = min(good, key=lambda r: (r["calls"], r["resolved_at"])) if good else None
    return {"scenario": scenario.name, "expected": expected, "template_t_res": t_res, "plans": len(results),
            "reachable": bool(good), "reward_1_plans": len(good), "rivals": [r["plan"] + " -> " + r["reached"] for r in rivals],
            "rivals_needing_hidden_knowledge": [r["plan"] + " -> " + r["reached"] for r in blind_rivals],
            "business_only_rivals": [f'{r["plan"]} -> {r["reached"]} at {r["resolved_at"]} {r["critical"]}' for r in business_rivals],
            "earliest": earliest and {k: earliest[k] for k in ("plan", "resolved_at", "calls")},
            "fewest_calls": fewest and {k: fewest[k] for k in ("plan", "resolved_at", "calls")},
            "beats_template": bool(earliest and expected != "pending" and earliest["resolved_at"] < t_res),
            "later_than_template": bool(earliest and expected != "pending" and earliest["resolved_at"] > t_res),
            "reward_1_plan_list": [r["plan"] for r in good]}


if __name__ == "__main__":
    from ap_sim.generator.fixtures import SCENARIOS

    for name in ("V1", "V2"):
        out = check(SCENARIOS[name]())
        out.pop("reward_1_plan_list")
        print(json.dumps(out, indent=1, default=str))
