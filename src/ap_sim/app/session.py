"""Application session: one case workspace over one engine (shared by both surfaces).

UI state: the current page, the typed quantity, and the last operational result. Every UI request
charges virtual time (POLICY C1) and lets due events fire before the next render.
Per-operation snapshots of the recorded state are kept for the grader's action log.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from ap_sim.app import catalog
from ap_sim.engine.engine import Attempt, Engine
from ap_sim.events.clock import ACTION_COST


@dataclass
class LastResult:
    id: str
    action: str
    attempt: Attempt


class AppSession:
    def __init__(self, scenario, schema: str = catalog.SCHEMA):
        self.scenario_name = scenario.name
        self.seed = scenario.seed
        self.engine = Engine(list(scenario.records), scenario.world, scenario.seed, scenario.current_invoice,
                             getattr(scenario, "policy", "sandbox-v1"))
        self.schema = schema
        self.policy_profile = getattr(scenario, "policy", "sandbox-v1")
        # The policy-shift profile took effect after the recorded block (POLICY I2): 100 minutes after its evaluation.
        inv = self.engine.invoice
        from ap_sim.domain.policy import SANDBOX_V2_EFFECT_MINUTES
        self.policy_since = (None if self.policy_profile == "sandbox-v1"
                             else inv.evaluated_at + SANDBOX_V2_EFFECT_MINUTES)   # POLICY I2
        self.initial_check = (inv.evaluated_at, inv.eligibility)
        self.page = "invoice"
        self.draft = ""
        self.last_result: LastResult | None = None
        self.history: list[dict] = []  # one entry per agent operation attempt: attempt + recorded state before it
        self._result_ids = random.Random(f"{scenario.seed}:result-ids")

    # UI requests ----------------------------------------------------------------------------------
    def charge(self, minutes: int) -> None:
        self.engine.advance(minutes)

    def navigate(self, page: str) -> None:
        if page not in catalog.pages_for(self.schema):
            raise KeyError(page)
        self.charge(ACTION_COST)
        self.page, self.draft, self.last_result = page, "", None

    def observe(self) -> None:
        self.charge(ACTION_COST)
        self.draft, self.last_result = "", None

    def fill(self, value: str) -> None:
        self.charge(ACTION_COST)
        self.draft = value

    def advance(self, minutes: int) -> None:
        self.charge(minutes)
        self.draft, self.last_result = "", None

    def submit_with_quantity(self, key: str, quantity: str) -> Attempt:
        """A person types the quantity and clicks a posting button: the typed value travels with the
        click. Typing is not a tool call and costs no virtual time."""
        self.draft = quantity
        return self.operate(key)

    def operate(self, key: str) -> Attempt:
        _, role, _, op, args = catalog.ACTION[key]
        if role != "button":
            raise KeyError(key)
        if key in catalog.QUANTITY_ACTIONS:
            args = {**args, "quantity": self.draft}
        before = self.engine.store.snapshot()
        attempt = self.engine.execute("ap_agent", op, **args)
        self.charge(ACTION_COST)  # direct-consequence events (receiving posting) fire here, before the render
        self.history.append({"attempt": asdict(attempt), "action": key, "before": before})
        self.last_result = LastResult(id=f"ACT-{self._result_ids.randrange(100000, 1000000)}", action=key,
                                      attempt=attempt)
        self.draft = ""
        return attempt
