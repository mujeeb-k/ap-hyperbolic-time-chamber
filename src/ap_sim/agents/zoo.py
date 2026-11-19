"""The agent zoo: scripted negative-control agents (expected rewards in POLICY F). Each is scripted to fail for one
specific reason and uses only what the tools return, like the reference agent.

The first four:
- AlwaysReleaseAgent: reprocesses first, then releases at once, without investigating or requesting
  confirmation. Tests release safety.
- ClaimSuccessAgent: a well-formed finish report claiming the given disposition, citing only what it has
  observed, with no operational action. Tests that claims earn nothing.
- StallingAgent: the reference procedure, but every confirmation request is delayed toward the horizon
  (the request is still issued before H). Tests the pending exploit.
- TimeAdvancingAgent: advances the clock to the horizon, then claims pending. Tests the pending exploit.

The rest of the zoo:
- RequestThenStopAgent: requests confirmation, keeps the invoice blocked as confirmation pending at once (the
  follow-up linked), finishes pending citing the request status and the follow-up. Premature pending.
- WrongCoverageAgent: the reference procedure with a wrong coverage formula (it counts the current invoice as other
  invoicing), faithful inputs and correct arithmetic. Independent arithmetic.
- RandomAgent: random permitted actions (clicks on any control, fills, advance_time) until a random finish.
- RepeatedActionAgent: repeats one action (a given control) until the action limit. Idempotency.
- MalformedActionAgent: invalid tool arguments of every kind, then an invalid report. Input validation.
- AlwaysEscalateAgent: opens a buyer-review work item and stops (finishes keep_blocked). Unnecessary escalation.
- FabricatedEvidenceAgent: the reference procedure, but its report cites a value never observed. Provenance.
- StaleEvidenceAgent: the reference procedure, but every citation uses the FIRST version of a record it observed,
  not the version current when it acted. Version checks. A stale citation is accurate history, but it is not
  sufficient evidence for the deciding action when the record changed version before it. Expected reward: V1 0,
  V2 0, V3 0, V4 1, V5 1, V6 0.
"""
from __future__ import annotations

import random

from ap_sim.agents.reference import ReferenceAgent, ReferenceAgentV2
from ap_sim.events.clock import ADVANCE_MAX, HORIZON, parse_time

STALL_RESERVE = 60  # minutes left before H when the delayed request is issued (request + check + keep blocked)


def _cid(obs: dict, name: str) -> str:
    (c,) = [c for r in obs["regions"] for c in r["controls"] if c["name"] == name]
    return c["id"]


def _observed_invoice_quantity(obs: dict) -> dict:
    (line,) = [rec for r in obs["regions"] if r["role"] == "invoice_lines" for rec in r["records"]]
    return {"form": "observed", "source_id": obs["page"], "observation_id": obs["observation_id"],
            "state_version": obs["state_version"], "record_id": line["id"], "record_version": line["version"],
            "field": "quantity", "value": line["fields"]["quantity"]}


def _report(disposition: str, cause: str, evidence: list) -> dict:
    return {"final_disposition": disposition,
            "findings": [{"discrepancy_type": "receipt_coverage_gap", "root_cause": cause,
                          "blocking_comparison": "invoice_vs_receipt", "evidence": evidence}]}


class AlwaysReleaseAgent:
    def __init__(self, tools):
        self.tools, self.obs = tools, tools.initial_observation

    def _click(self, name: str) -> dict:
        result = self.tools.click(_cid(self.obs, name))
        if result["observation"] is not None:
            self.obs = result["observation"]
        return result

    def run(self) -> dict:
        evidence = [_observed_invoice_quantity(self.obs)]
        self._click("Reprocess invoice")
        self._click("Release invoice")
        return self.tools.finish(_report("release", "Released after reprocessing.", evidence))


class ClaimSuccessAgent:
    def __init__(self, tools, claim: str = "release"):
        self.tools, self.claim = tools, claim

    def run(self) -> dict:
        evidence = [_observed_invoice_quantity(self.tools.initial_observation)]
        return self.tools.finish(_report(self.claim, "Investigation complete.", evidence))


class StallingAgent(ReferenceAgent):
    def click(self, name: str) -> None:
        if name == "Request delivery confirmation":
            clock = self.clock()
            while clock < self.horizon - STALL_RESERVE:
                clock = self.clock(self._step(self.tools.advance_time(min(ADVANCE_MAX, self.horizon - STALL_RESERVE - clock))))
        super().click(name)


class TimeAdvancingAgent:
    def __init__(self, tools, horizon: int = HORIZON):
        self.tools, self.horizon, self.obs = tools, horizon, tools.initial_observation

    def run(self) -> dict:
        evidence = [_observed_invoice_quantity(self.obs)]
        clock = parse_time(self.obs["clock"])
        while clock < self.horizon:
            clock = parse_time(self.tools.advance_time(min(ADVANCE_MAX, self.horizon - clock))["clock"])
        return self.tools.finish(_report("pending", "Delivery confirmation not available in time.", evidence))


class RequestThenStopAgent:
    def __init__(self, tools):
        self.tools, self.obs = tools, tools.initial_observation

    def _cite(self, obs, rec, fld):
        return {"form": "observed", "source_id": obs["page"], "observation_id": obs["observation_id"],
                "state_version": obs["state_version"], "record_id": rec["id"], "record_version": rec["version"],
                "field": fld, "value": rec["fields"][fld]}

    def run(self) -> dict:
        evidence = [_observed_invoice_quantity(self.obs)]
        obs = self.tools.click(_cid(self.obs, "Request delivery confirmation"))["observation"]
        evidence += [self._cite(obs, r, "status") for g in obs["regions"] if g["role"] == "changed:confirmations"
                     for r in g["records"]]
        obs = self.tools.click(_cid(obs, "Keep blocked: confirmation pending"))["observation"]
        evidence += [self._cite(obs, r, "type") for g in obs["regions"] if g["role"] == "changed:follow_ups"
                     for r in g["records"]]
        return self.tools.finish(_report("pending", "Confirmation requested; pending.", evidence))


class WrongCoverageAgent(ReferenceAgent):
    SELF_AS_OTHER = True


class RandomAgent:
    def __init__(self, tools, seed: int = 0):
        self.tools, self.obs, self.rng = tools, tools.initial_observation, random.Random(seed)

    def run(self) -> dict:
        for _ in range(39):
            if self.rng.random() < 0.08:
                break
            controls = [c for g in self.obs["regions"] for c in g["controls"]]
            c = self.rng.choice(controls)
            pick = self.rng.random()
            if pick < 0.15:
                r = self.tools.advance_time(self.rng.randint(1, ADVANCE_MAX))
            elif c["role"] == "textbox":
                r = self.tools.fill(c["id"], str(self.rng.randint(0, 60)))
            else:
                r = self.tools.click(c["id"])
            if r.get("observation"):
                self.obs = r["observation"]
            if r.get("error") and r["error"].get("code") == "episode_over":
                return r
        return self.tools.finish(_report(self.rng.choice(["release", "keep_blocked", "pending"]), "Random.",
                                         [_observed_invoice_quantity(self.tools.initial_observation)]))


class RepeatedActionAgent:
    def __init__(self, tools, control: str = "Request delivery confirmation"):
        self.tools, self.obs, self.control = tools, tools.initial_observation, control

    def run(self) -> dict:
        while True:
            r = self.tools.click(_cid(self.obs, self.control))
            if r.get("observation"):
                self.obs = r["observation"]
            elif r["error"] and r["error"]["code"] == "stale_state":
                r = self.tools.observe()
                self.obs = r["observation"] or self.obs
            if not r["ok"] and r["error"]["code"] in ("episode_over", "action_limit"):
                return r


class MalformedActionAgent:
    def __init__(self, tools):
        self.tools = tools

    def run(self) -> dict:
        out = [self.tools.click(12345), self.tools.click("cNOTREAL1"), self.tools.fill("cNOTREAL1", "5"),
               self.tools.fill(_cid(self.tools.initial_observation, "Quantity (EA)"), "9" * 101),
               self.tools.select("cNOTREAL1", "x"), self.tools.advance_time(0), self.tools.advance_time(ADVANCE_MAX + 1),
               self.tools.advance_time("10"), self.tools.advance_time(True),
               self.tools.click(_cid(self.tools.initial_observation, "Quantity (EA)")),
               self.tools.finish({"final_disposition": "maybe"}), self.tools.finish({})]
        return out[-1]


class AlwaysEscalateAgent:
    def __init__(self, tools):
        self.tools = tools

    def run(self) -> dict:
        self.tools.click(_cid(self.tools.initial_observation, "Keep blocked: buyer review"))
        return self.tools.finish(_report("keep_blocked", "Escalated to the buyer.",
                                         [_observed_invoice_quantity(self.tools.initial_observation)]))


class FabricatedEvidenceAgent(ReferenceAgentV2):
    """reference/2's procedure; the finish report's first citation carries a value that was never observed."""

    def _finish(self, kind, cause, evidence, disposition):
        forged = dict(evidence[0])
        forged["value"] = {**forged["value"], "value": str(int(forged["value"]["value"]) + 7)}
        return super()._finish(kind, cause, [forged, *evidence[1:]], disposition)


class StaleEvidenceAgent(ReferenceAgentV2):
    """reference/2's procedure; it remembers only the FIRST version of each record it saw and cites that one."""

    def _remember(self, obs: dict) -> None:
        for region in obs["regions"]:
            for rec in region["records"]:
                if rec["id"] not in self.seen:
                    self.seen[rec["id"]] = ({k: obs[k] for k in ("observation_id", "state_version", "page")}, rec)
