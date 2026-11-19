"""Deterministic event queue (POLICY C1): by timestamp, then type priority, then creation sequence."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

PRIORITY = {"receipt_posting": 1, "confirmation_response": 2}
OTHER_PRIORITY = 3


@dataclass(order=True)
class ScheduledEvent:
    at: int
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False)
    cause: int | None = field(compare=False, default=None)  # attempt sequence number that caused it


class Scheduler:
    def __init__(self):
        self._heap: list[ScheduledEvent] = []
        self._seq = 0

    def schedule(self, at: int, kind: str, payload: dict, cause: int | None = None) -> None:
        self._seq += 1
        heapq.heappush(self._heap, ScheduledEvent(at, PRIORITY.get(kind, OTHER_PRIORITY), self._seq, kind, payload, cause))

    def pop_due(self, until: int) -> ScheduledEvent | None:
        if self._heap and self._heap[0].at <= until:
            return heapq.heappop(self._heap)
        return None
