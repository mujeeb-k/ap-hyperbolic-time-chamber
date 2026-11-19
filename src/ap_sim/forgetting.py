"""The L1 forgetting set (defined 2026-10-31): 40 ladder/2 L1 cases, frozen in manifests/forgetting_l1 with their
own freeze record, apart from the v2 splits (generator/splits.py). It takes the place of the regression split, which
is unused: checkpoint scoring in Phases 6-9 (training) and the regression evaluation in Phase 10, the final
evaluation (phases of the pre-registered plan, preregistration_proof/PLAN_V2.md).

This loader lives outside every directory manifests/v2/FREEZE.json hashes, so adding it changes none of them.

The loader accepts only the frozen manifest and refuses, by content, any case that has a reversal, runs under a policy
other than sandbox-v1, is not L1 under ladder/2, or shares a key (exact, near-duplicate or grouping) with any v2
split.

The regression rule (pre-declared, unconditional): a checkpoint is disqualified if its greedy success on this set is
more than 2 standard errors below SFT-v2's, the standard error being that of the per-case paired difference.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections.abc import Callable, Mapping

from ap_sim.generator import causal, ladder
from ap_sim.generator.manifests import _tuples, params_sha
from ap_sim.generator.splits import DIR as V2_DIR, SPLITS, SplitRefused, keys

ROOT = pathlib.Path(__file__).resolve().parents[2]
NAME = "forgetting_l1"
DIR = ROOT / "manifests" / NAME
FREEZE = DIR / "FREEZE.json"


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _v2_keys() -> dict[str, str]:
    out = {}
    for s in SPLITS:
        for e in json.loads((V2_DIR / f"{s}.json").read_text())["entries"]:
            for k in e["keys"].values():
                out[k] = s
    return out


def load(*, manifest: dict | None = None, require_frozen: bool = True) -> dict[str, Callable[[], object]]:
    """case_id -> scenario factory, in frozen order. Raises SplitRefused on anything outside the frozen set."""
    path = DIR / f"{NAME}.json"
    if manifest is None:
        if require_frozen:
            want = json.loads(FREEZE.read_text())["files"][f"{NAME}.json"]
            if _sha(path) != want:
                raise SplitRefused(f"{NAME}: the manifest is not the frozen one")
        manifest = json.loads(path.read_text())
    if manifest.get("split") != NAME:
        raise SplitRefused(f"{NAME}: not this set's manifest ({manifest.get('split')!r})")
    others = _v2_keys()
    out = {}
    for e in manifest["entries"]:
        cid = e.get("case_id")
        if e.get("split") != NAME:
            raise SplitRefused(f"{NAME}: {cid} is labelled {e.get('split')!r}")
        params = _tuples(e["params"])
        if params_sha(params) != e.get("params_sha256"):
            raise SplitRefused(f"{NAME}: {cid}: parameters do not match their sha256")
        if params.get("reversals"):
            raise SplitRefused(f"{NAME}: {cid} has a reversal")
        if params.get("policy", "sandbox-v1") != "sandbox-v1":
            raise SplitRefused(f"{NAME}: {cid} runs under {params.get('policy')}")
        c = causal.generate(cid, e["seed"], e.get("force"))
        if params_sha(c.params) != e["params_sha256"]:
            raise SplitRefused(f"{NAME}: {cid} does not regenerate from its seed and forcing")
        if ladder.level(c) != "L1":
            raise SplitRefused(f"{NAME}: {cid} is {ladder.level(c)} under ladder/2, not L1")
        for kind, k in keys(e["seed"], params).items():
            if k in others:
                raise SplitRefused(f"{NAME}: {cid} shares its {kind} key with {others[k]}/")
        out[cid] = c.scenario
    return out


def regression_check(sft_v2: Mapping[str, bool], checkpoint: Mapping[str, bool]) -> dict:
    """The pre-declared regression rule on this set's greedy successes (case id -> success), paired by case: the
    checkpoint is disqualified if mean(checkpoint - SFT-v2) < -2 SE, SE = sample sd of the per-case differences
    / sqrt(n). A fixed points threshold is not used: on 40 cases its noise depends on how often the two disagree."""
    if set(sft_v2) != set(checkpoint):
        raise ValueError("the two results must cover the same cases")
    d = [int(checkpoint[c]) - int(sft_v2[c]) for c in sorted(sft_v2)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return {"cases": n, "sft_v2_success": sum(map(int, sft_v2.values())) / n,
            "checkpoint_success": sum(map(int, checkpoint.values())) / n, "mean_difference": mean,
            "standard_error": se, "threshold": -2 * se, "disqualified": mean < -2 * se}
