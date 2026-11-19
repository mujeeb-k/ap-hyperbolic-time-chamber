"""The v2 case splits (SFT, RL, dev, regression, test, test reserve, V6, policy shift), their keys, and their
loaders. The design was fixed on 2026-10-30, before any case existed.

Keys (computed from a case's parameters, never trusted from a stored field):
  exact     sha256 of (seed, parameters): the identical case
  near      sha256 of the parameters: the same numbers and timings (record ids come from the seed, so a case that
            differs only in names or ids has the same near key)
  group     sha256 of the parameters without the hidden world (`physical_total`, `delay`) and the policy: what the
            agent sees at the start, so twins, counterfactual siblings and policy-shift copies share it

Loaders. Each split has its own loader, which accepts only its own frozen manifest and refuses:
  - any case whose near or group key occurs in another split's manifest;
  - for SFT, RL, dev, regression and test: any case with a reversal record and any sandbox-v2 case;
  - for V6: any case without a reversal and a gap at the start (identified by content, not by its label).
Evaluator side only: the training environment receives the factories by injection and never imports this module.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Callable

from ap_sim.generator.fixtures import Scenario, build
from ap_sim.generator.manifests import _tuples, params_sha

ROOT = pathlib.Path(__file__).resolve().parents[3]
DIR = ROOT / "manifests" / "v2"
FREEZE = DIR / "FREEZE.json"

SPLITS = ("sft", "rl", "dev", "regression", "test", "test_reserve", "v6", "policy_shift")
TRAINING = ("sft", "rl", "dev", "regression")          # never a reversal, never sandbox-v2
HIDDEN = ("physical_total", "delay")


class SplitRefused(ValueError):
    """A loader was asked for a case that does not belong to its split."""


# --- keys and content predicates -------------------------------------------------------------------------------------
def _canonical(params: dict) -> dict:
    p = {k: (list(map(list, v)) if k == "reversals" else list(v) if isinstance(v, tuple) else v)
         for k, v in params.items()}
    if p.get("policy", "sandbox-v1") == "sandbox-v1":
        p.pop("policy", None)
    return p


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def keys(seed: int, params: dict) -> dict:
    p = _canonical(params)
    visible = {k: v for k, v in p.items() if k not in HIDDEN and k != "policy"}
    return {"exact": _sha([seed, p]), "near": _sha(p), "group": _sha(visible)}


def has_reversal(params: dict) -> bool:
    return bool(params.get("reversals"))


def gap_at_start(params: dict) -> int:
    """Invoice quantity minus coverage under sandbox-v1 (net receipts less other active invoices), floored at 0."""
    net = sum(params["receipts"]) - sum(q for _, q in params.get("reversals", ()))
    return max(0, params["invoice_qty"] - (net - sum(params["other_invoices"])))


def is_v6(params: dict) -> bool:
    """V6 by content: a reversal and a gap at the start (a reversal with no gap is a covered case, labelled V5)."""
    return has_reversal(params) and gap_at_start(params) > 0


def sandbox_v2(params: dict) -> bool:
    return params.get("policy", "sandbox-v1") == "sandbox-v2"


def invoice_amount(s: Scenario) -> dict:
    """The current invoice's line amount and currency (the invoice header carries none)."""
    line = next(r for r in s.records if r.kind == "invoice_line" and r.invoice == s.current_invoice)
    return {"amount": str(line.amount.value), "currency": line.amount.currency}


# --- loaders ---------------------------------------------------------------------------------------------------------
def path(split: str) -> pathlib.Path:
    return DIR / f"{split}.json"


def _frozen_hashes() -> dict:
    return json.loads(FREEZE.read_text())["files"] if FREEZE.exists() else {}


def file_sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _other_keys(split: str, directory: pathlib.Path) -> dict[str, str]:
    out = {}
    for other in SPLITS:
        p = directory / f"{other}.json"
        if other == split or not p.exists():
            continue
        for e in json.loads(p.read_text())["entries"]:
            k = keys(e["seed"], _tuples(e["params"]))
            out[k["near"]] = out[k["group"]] = f"{other}/{e['case_id']}"
    return out


def refusal(split: str, entry: dict) -> str | None:
    """Why `split`'s loader refuses this entry on its content (None if it does not)."""
    p = entry["params"]
    if split in TRAINING + ("test", "test_reserve"):
        if has_reversal(p):
            return "has a reversal (D4)"
        if sandbox_v2(p):
            return "is a sandbox-v2 case"
    if split == "v6" and not is_v6(p):
        return "is not V6 by content (needs a reversal and a gap at the start)"
    if split == "policy_shift" and has_reversal(p):
        return "has a reversal"
    return None


def load_split(split: str, *, directory: pathlib.Path | None = None, require_frozen: bool = True,
               manifest: dict | None = None) -> dict[str, Callable[[], Scenario]]:
    """case_id -> scenario factory, in the manifest's frozen order. Raises SplitRefused on anything that is not this
    split's own case (see the module docstring). `manifest` substitutes the file's content (tests only)."""
    if split not in SPLITS:
        raise SplitRefused(f"unknown split {split!r}")
    directory = directory or DIR
    p = directory / f"{split}.json"
    if manifest is None:
        if require_frozen:
            want = _frozen_hashes().get(f"{split}.json")
            if want is None or file_sha(p) != want:
                raise SplitRefused(f"{split}: the manifest is not the frozen one")
        manifest = json.loads(p.read_text())
    if manifest.get("split") != split:
        raise SplitRefused(f"{split}'s loader was given the {manifest.get('split')!r} manifest")
    others = _other_keys(split, directory)
    out = {}
    for e in manifest["entries"]:
        cid = e["case_id"]
        if e.get("split") != split:
            raise SplitRefused(f"{cid} belongs to {e.get('split')!r}, not {split!r}")
        params = _tuples(e["params"])
        if params_sha(params) != e["params_sha256"]:
            raise SplitRefused(f"{cid}: parameters do not match their sha256")
        why = refusal(split, e)
        if why:
            raise SplitRefused(f"{split} refuses {cid}: it {why}")
        k = keys(e["seed"], params)
        for kind in ("near", "group"):
            if k[kind] in others:
                raise SplitRefused(f"{split} refuses {cid}: its {kind} key belongs to {others[k[kind]]}")
        out[cid] = (lambda cid=cid, seed=e["seed"], p=params: build(cid, seed, **p))
    return out


def load_test(*, extended: bool = False, **kw) -> dict[str, Callable[[], Scenario]]:
    """The test set in frozen order: 200 cases, or 300 with the reserve appended in its frozen order (only when the
    test-extension trigger of the pre-registered plan, preregistration_proof/PLAN_V2.md section 4, has fired)."""
    out = load_split("test", **kw)
    if extended:
        out.update(load_split("test_reserve", **kw))
    return out


def first_test_cases(n: int = 100, **kw) -> list[str]:
    """The first n test cases in frozen order (the 100 of the Claude reference row)."""
    return list(load_split("test", **kw))[:n]
