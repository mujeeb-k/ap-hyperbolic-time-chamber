"""Manifest loading: every case of a frozen (or candidate) manifest is rebuilt from its STORED parameters and
checked against its params sha256 before any use. A seed alone cannot rebuild the training pool (it was drawn with
structural forcing), so parameters are the record. Evaluator-only: the training environment receives the resulting
scenario factories by injection and never imports this module."""
from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Callable

from ap_sim.generator.fixtures import Scenario, build


def params_sha(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


def _tuples(params: dict) -> dict:
    p = dict(params)
    for k in ("cancelled_invoices",):
        if k in p:
            p[k] = tuple(p[k])
    if "reversals" in p:
        p["reversals"] = tuple(tuple(r) for r in p["reversals"])
    return p


def load(path: str | pathlib.Path) -> dict[str, Callable[[], Scenario]]:
    """case_id -> zero-argument scenario factory, for one manifest file; raises on any parameter mismatch."""
    data = json.loads(pathlib.Path(path).read_text())
    out = {}
    for e in data["entries"]:
        params = _tuples(e["params"])
        if params_sha(params) != e["params_sha256"]:
            raise ValueError(f"{path}: parameters of {e['case_id']} do not match their sha256")
        out[e["case_id"]] = (lambda cid=e["case_id"], seed=e["seed"], p=params: build(cid, seed, **p))
    return out
