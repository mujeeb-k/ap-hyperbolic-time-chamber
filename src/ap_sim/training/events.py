"""Incremental run evidence (evaluator-side only, never shown to the agent). When HTC_AP_EVENT_LOG is set, each event
is appended as one JSON line and flushed at once, so a run that is cut keeps everything before the cut. When
HTC_AP_TRACE_DIR is set, an episode's log is written the moment the episode ends, independently of when its generation
batch is graded. Both are no-ops when unset."""
from __future__ import annotations

import json
import os
import pathlib
import time


def event(row: dict) -> None:
    path = os.environ.get("HTC_AP_EVENT_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 2), **row}, default=str) + "\n")
            f.flush()


def save_episode(name: str, payload: dict) -> str | None:
    folder = os.environ.get("HTC_AP_TRACE_DIR")
    if not folder:
        return None
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"{name}.json"
    path.write_text(json.dumps(payload, default=str))
    return str(path)
