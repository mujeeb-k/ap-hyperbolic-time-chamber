"""Virtual business clock (POLICY C1).

Time is business minutes from episode start: t = 0 is D1 09:00, one business day is 480 minutes.
Display: "D<n> HH:MM"; a positive multiple of 480 is the end of that day ("D1 17:00").
Negative t is history before the episode (n <= 0).
"""
from __future__ import annotations

import re

MINUTES_PER_DAY = 480
DAY_START = 9 * 60
ACTION_COST = 5          # POLICY C1: every tool call except advance_time and finish
ADVANCE_MAX = 480        # POLICY C1: advance_time bound per call
HORIZON = 2400           # POLICY C1: 5 business days
ACTION_LIMIT = 40        # POLICY C1: every tool call counts

_TIME = re.compile(r"^D(-?\d+) (\d{2}):(\d{2})$")


def format_time(t: int) -> str:
    if t > 0 and t % MINUTES_PER_DAY == 0:
        day, minute = t // MINUTES_PER_DAY, MINUTES_PER_DAY
    else:
        day, minute = t // MINUTES_PER_DAY + 1, t % MINUTES_PER_DAY
    clock = DAY_START + minute
    return f"D{day} {clock // 60:02d}:{clock % 60:02d}"


def parse_time(text: str) -> int:
    m = _TIME.match(text)
    if not m:
        raise ValueError(f"not a virtual time: {text!r}")
    day, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3))
    minute = hh * 60 + mm - DAY_START
    if not 0 <= minute <= MINUTES_PER_DAY:
        raise ValueError(f"outside business hours: {text!r}")
    t = (day - 1) * MINUTES_PER_DAY + minute
    if format_time(t) != text:
        raise ValueError(f"non-canonical virtual time: {text!r}")
    return t
