"""Shared helpers for the stress-set build and validation (the set is specified in manifests/stress/SPEC.md).

The existing cases are read from their manifests and built by program. test_reserve: its keys, seeds and scan figures
are computed here like every other split's, and only COUNTS of matches are ever written or printed, so the reserve
stays unread until its evaluation. Nothing here prints or returns a reserve case's id, parameters or content.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for d in ("scripts", "scripts/phase2", "scripts/step5", "scripts/step9"):
    if str(ROOT / d) not in sys.path:
        sys.path.insert(0, str(ROOT / d))

from ap_sim.stress import messiness as st   # noqa: E402
from ap_sim.generator.fixtures import build as frozen_build   # noqa: E402
from ap_sim.generator.manifests import _tuples   # noqa: E402
from ap_sim.generator.splits import DIR as V2_DIR, keys   # noqa: E402

SPLITS = ("sft", "rl", "dev", "regression", "test", "test_reserve", "v6", "policy_shift", "forgetting_l1")
RESERVE = "test_reserve"
OUT = ROOT / "runs" / "stress"
MANIFEST_DIR = ROOT / "manifests" / "stress"


def sha_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def split_path(split: str) -> pathlib.Path:
    return (ROOT / "manifests" / "forgetting_l1" / "forgetting_l1.json") if split == "forgetting_l1" else V2_DIR / f"{split}.json"


def split_entries(split: str) -> list[dict]:
    return json.loads(split_path(split).read_text())["entries"]


def frozen_versions() -> dict:
    """The frozen code the stress set depends on, checked against manifests/v2/FREEZE.json and manifests/g8/FREEZE.json
    (generator, engine, g7, reference/5, surfaces/4, policy; g8, reward, evaluation harness)."""
    out, bad = {}, []
    for f in ("manifests/v2/FREEZE.json", "manifests/g8/FREEZE.json"):
        for group, files in json.loads((ROOT / f).read_text())["versions"].items():
            if isinstance(files, str):
                continue
            for p, h in files.items():
                ok = sha_file(ROOT / p) == h
                out[p] = ok
                if not ok:
                    bad.append(p)
    return {"files_checked": len(out), "mismatches": bad}


class Existing:
    """Seeds and keys of every existing split, computed by program from the manifests (counts only for the reserve).
    Key families: `param` (generator/splits.keys on the stored parameters, the frozen definition) and `visible`
    (stress.visible_keys on the built scenario: comparable across parameter formats)."""

    def __init__(self):
        self.seeds: dict[int, set[str]] = {}
        self.param: dict[str, dict[str, set[str]]] = {k: {} for k in ("exact", "near", "group")}
        self.visible: dict[str, dict[str, set[str]]] = {k: {} for k in ("exact", "near", "group")}
        self.cases: dict[str, int] = {}
        self.scenarios: dict[str, list] = {}      # split -> built scenarios (for the parameter scan); never printed
        for split in SPLITS:
            es = split_entries(split)
            self.cases[split] = len(es)
            built = []
            for e in es:
                p = _tuples(e["params"])
                self.seeds.setdefault(e["seed"], set()).add(split)
                for kind, k in keys(e["seed"], p).items():
                    self.param[kind].setdefault(k, set()).add(split)
                s = frozen_build(e["case_id"], e["seed"], **p)
                for kind, k in st.visible_keys(e["seed"], s).items():
                    self.visible[kind].setdefault(k, set()).add(split)
                built.append(s)
            self.scenarios[split] = built

    def add_earlier(self) -> dict:
        """The sources used before the v2 splits were built (build_splits.earlier_keys: the five v1 sets, the fresh
        1,000 and their regeneration, the 55 dev cases, the boundary cases), param keys only, under the name
        'earlier'."""
        import build_splits as bs
        earlier, counts = bs.earlier_keys()
        for k in earlier:
            for kind in ("exact", "near", "group"):
                self.param[kind].setdefault(k, set()).add("earlier")
        return counts

    def clash(self, seed: int, pk: dict, vk: dict) -> list[str]:
        """Where a new case's seed or keys already occur: ['<family> <kind> key: <split>', ...] (split names only)."""
        out = [f"seed: {s}" for s in sorted(self.seeds.get(seed, ()))]
        for fam, ks, table in (("param", pk, self.param), ("visible", vk, self.visible)):
            for kind in ("exact", "near", "group"):
                out += [f"{fam} {kind} key: {s}" for s in sorted(table[kind].get(ks[kind], ()))]
        return out

    def take(self, seed: int, pk: dict, vk: dict, owner: str = "stress") -> None:
        self.seeds.setdefault(seed, set()).add(owner)
        for kind in ("exact", "near", "group"):
            self.param[kind].setdefault(pk[kind], set()).add(owner)
            self.visible[kind].setdefault(vk[kind], set()).add(owner)


def reference_episode(case_id: str, factory) -> dict:
    """reference/5 in APReceiptEnvV2 (surfaces/4), graded by g7 (as runs/phase2/checks/reference.json) and by g8 (as
    the evaluation grades), with its observations and calls (the episode-limits check's procedure, with the log kept)."""
    from ap_sim.agents.reference import ReferenceAgentV5
    from ap_sim.grader.grader import grade as g7
    from ap_sim.grading import g8
    from ap_sim.training.environment import APReceiptEnvV2
    env = APReceiptEnvV2({case_id: factory})
    obs = json.loads(env.reset(scenario=case_id))
    calls = []

    def call(name, args):
        calls.append({"name": name, "arguments": args})
        return json.loads(getattr(env, name)(**args))

    class Tools:
        initial_observation = obs
        click = staticmethod(lambda element_id: call("click", {"element_id": element_id}))
        fill = staticmethod(lambda element_id, text: call("fill", {"element_id": element_id, "text": text}))
        advance_time = staticmethod(lambda minutes: call("advance_time", {"minutes": minutes}))
        coverage = staticmethod(lambda invoice_id: call("coverage", {"invoice_id": invoice_id}))
        finish = staticmethod(lambda report: call("finish", report))

    error = None
    try:
        ReferenceAgentV5(Tools()).run()
    except Exception as exc:          # a rejected call or a calculator mismatch: the reference cannot solve it
        error = f"{type(exc).__name__}: {exc}"
    scenario, log, _ = env._episode()
    v7, v8 = g7(scenario, log), g8.grade(scenario, log)
    return {"error": error, "calls": calls, "log": log,
            "g7": {"reward": v7.reward, "reached": v7.reached, "expected": v7.expected, "critical": v7.critical,
                   "t_res": None if v7.t_res == float("inf") else float(v7.t_res)},
            "g8": {"reward": v8.reward, "reached": v8.reached, "expected": v8.expected, "critical": v8.critical}}


def tokenizer():
    from qwen_check import MODEL, REVISION
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema, get_training_chat_template
    tok = add_response_schema(AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True))
    return tok, get_training_chat_template(tok), {"model": MODEL, "revision": REVISION}


def limits() -> dict:
    from ap_sim.training.config import (PIPELINE_MAX_COMPLETION_LENGTH, PIPELINE_MAX_CONTEXT_LENGTH,
                                        PIPELINE_MAX_TOOL_CALLING_ITERATIONS)
    return {"tool_calls": PIPELINE_MAX_TOOL_CALLING_ITERATIONS, "completion_tokens": PIPELINE_MAX_COMPLETION_LENGTH,
            "context_tokens": PIPELINE_MAX_CONTEXT_LENGTH}
