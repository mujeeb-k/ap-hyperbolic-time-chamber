"""Deterministic replay: re-execute a recorded tool-call sequence without
any model and compare every step's tool result, clock and business-state hash."""
from __future__ import annotations

from ap_sim.interfaces.canonical import dumps
from ap_sim.interfaces.tools import ToolLayer


def replay_calls(scenario, surface, steps: list[dict], schema: str) -> ToolLayer:
    """Replay under the recorded episode's schema (its log's \"schema\"); the schema is never defaulted."""
    env = ToolLayer(scenario, surface, schema)
    for step in steps:
        getattr(env, step["tool"])(**step["args"])
    return env


def compare(log: dict, env: ToolLayer) -> list[str]:
    """Differences between a recorded episode and its replay (empty list = identical)."""
    replayed = env.episode_log()
    diffs = []
    if dumps(log["initial_observation"]) != dumps(replayed["initial_observation"]):
        diffs.append("initial observation")
    if len(log["steps"]) != len(replayed["steps"]):
        diffs.append(f"step count {len(log['steps'])} vs {len(replayed['steps'])}")
    for a, b in zip(log["steps"], replayed["steps"]):
        for key in ("business_hash", "clock", "state_version"):
            if a[key] != b[key]:
                diffs.append(f"step {a['i']}: {key}")
        if dumps(a["result"]) != dumps(b["result"]):
            diffs.append(f"step {a['i']}: tool result")
    if log["final_hash"] != replayed["final_hash"]:
        diffs.append("final business-state hash")
    return diffs
