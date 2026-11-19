"""Manual play: one scenario served to a person, under the same horizon and action-limit rules as the
agent's tool layer, with every rendered page and operation recorded so the run can be graded and
inspected afterwards."""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from datetime import datetime, timezone

from ap_sim.app import catalog
from ap_sim.app.controls import control_ids
from ap_sim.app.limits import EpisodeLimits
from ap_sim.app.server import AppServer
from ap_sim.app.session import AppSession
from ap_sim.interfaces.structured import page_model_observation


class ManualPlay:
    def __init__(self, scenario, schema: str = catalog.SCHEMA):
        self.scenario = scenario
        self.session = AppSession(scenario, schema)
        self.ids = control_ids(scenario.seed)
        self.initial = {"observation_id": "m0", **page_model_observation(self.session, self.ids)}
        self.seen: list[dict] = []  # every page rendered to the person, in order (indices 1, 2, ...)
        operate = self.session.operate

        def logged_operate(key):  # tag each operation with its place in the sequence of rendered pages
            attempt = operate(key)
            self.session.history[-1]["step"] = len(self.seen) + 1
            return attempt

        self.session.operate = logged_operate
        self.server = AppServer(self.session)
        self.server.limits = EpisodeLimits(scenario.horizon)
        self.server.on_render = lambda s: self.seen.append(
            {"observation_id": f"m{len(self.seen) + 1}", **page_model_observation(s, self.ids)})

    def start(self) -> "ManualPlay":
        self.server.start()
        return self

    def stop(self) -> None:
        self.server.stop()

    def log(self) -> dict:
        eng = self.session.engine
        return {"scenario": self.scenario.name, "seed": self.scenario.seed, "surface": "manual",
                "schema": self.session.schema, "presentation_profile": catalog.PRESENTATION_PROFILE,
                "horizon": self.scenario.horizon, "initial_observation": self.initial,
                "steps": [{"i": i, "result": {"observation": o}} for i, o in enumerate(self.seen, 1)],
                "operations": self.session.history, "attempts": [asdict(a) for a in eng.attempts],
                "refusals": self.server.refusals, "requests": self.server.limits.calls,
                "final_snapshot": eng.store.snapshot(), "final_clock": eng.clock, "report": None,
                "termination": "manual", "calls": None}

    def save(self, directory: str = "trajectories/manual") -> pathlib.Path:
        path = pathlib.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = path / f"{self.scenario.name}-{stamp}.json"
        out.write_text(json.dumps(self.log(), indent=1, default=str))
        return out
