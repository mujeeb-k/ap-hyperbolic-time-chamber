"""Structured surface: page-model producer and direct controller execution."""
from __future__ import annotations

from ap_sim.app import catalog
from ap_sim.app.session import AppSession
from ap_sim.app.viewmodel import build_page
from ap_sim.events.clock import format_time
from ap_sim.interfaces.canonical import encode


def _columns(role: str) -> tuple[catalog.Column, ...]:
    base = role.split(":", 1)[1] if role.startswith("changed:") else role
    return catalog.REGION[base].columns


def page_model_observation(session: AppSession, ids: dict[str, str]) -> dict:
    vm = build_page(session)
    regions = []
    for r in vm.regions:
        cols = {c.field: c for c in _columns(r.role)}
        records = [{"id": rec.id, "type": rec.type, "version": rec.version,
                    "fields": {f: encode(cols[f].type, v) for f, v in rec.fields.items()}} for rec in r.records]
        controls = []
        for c in r.controls:
            ctl = {"id": ids[c.key], "role": c.role, "name": c.name, "enabled": True}
            if c.role == "textbox":
                ctl["value"] = c.value or ""
            controls.append(ctl)
        regions.append({"role": r.role, "records": records, "controls": controls})
    return {"state_version": vm.state_version, "clock": format_time(vm.clock), "role": vm.role,
            "page": vm.page, "case": {k: encode("text", v) for k, v in vm.case.items()}, "regions": regions}


class StructuredSurface:
    name = "structured"

    def attach(self, session: AppSession, ids: dict[str, str]) -> None:
        self.session, self.ids = session, ids

    def produce(self) -> dict:
        return page_model_observation(self.session, self.ids)

    def click(self, key: str, element_id: str) -> None:
        kind, name = key.split(":", 1)
        if kind == "nav":
            self.session.navigate(name)
        else:
            self.session.operate(name)

    def fill(self, key: str, element_id: str, text: str) -> None:
        self.session.fill(text)

    def observe(self) -> None:
        self.session.observe()

    def advance(self, minutes: int) -> None:
        self.session.advance(minutes)

    def charge(self, minutes: int) -> None:
        self.session.charge(minutes)

    def close(self) -> None:
        pass
