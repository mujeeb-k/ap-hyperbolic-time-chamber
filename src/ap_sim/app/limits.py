"""Episode limits for manual play: the same termination rules the agent's tool layer applies (POLICY C1).
An action may start only if it completes by the horizon; advance_time is clamped at the
horizon; the action limit counts every request, refused ones included; a refused request costs one action
of time, capped at the horizon."""
from __future__ import annotations

from ap_sim.events.clock import ACTION_COST, ACTION_LIMIT


class EpisodeLimits:
    def __init__(self, horizon: int, action_limit: int = ACTION_LIMIT):
        self.horizon, self.action_limit, self.calls = horizon, action_limit, 0

    def admit(self, clock: int, cost: int) -> str | None:
        """Count one request; return a refusal reason, or None when the request may proceed."""
        if self.calls >= self.action_limit:
            return "action_limit"
        self.calls += 1
        if clock + cost > self.horizon:
            return "horizon"
        return None

    def admit_advance(self, clock: int, minutes: int) -> tuple[str | None, int]:
        """Count one advance request; return (refusal, minutes actually advanced)."""
        if self.calls >= self.action_limit:
            return "action_limit", 0
        self.calls += 1
        if clock >= self.horizon:
            return "horizon", 0
        return None, min(minutes, self.horizon - clock)

    def refusal_cost(self, clock: int) -> int:
        return max(0, min(ACTION_COST, self.horizon - clock))
