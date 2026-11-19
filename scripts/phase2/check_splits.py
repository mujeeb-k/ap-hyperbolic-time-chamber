"""The checks on every v2 split before freezing (the splits were frozen on 2026-10-30). No model is run.

Before the final evaluation, held-out cases may be read only by checks that view no model output. Here only the
grader, the reference agents (reference/5, the scripted baselines and the zoo), the independent verifier, the
reachability checker and the leakage audit touch the cases.

  scope         causal.check_scope on every case; every case loads through its own split's loader (unfrozen)
  reference     reference/5 in APReceiptEnvV2 (surfaces/4, instructions/5) at reward 1, measured as TRL renders it
                with the pinned Qwen tokenizer; closest margins to 40 tool calls, 28,354 completion and 40,094
                context tokens; closest margin of the grader's T_res to the horizon
  verifier      the independent verifier (a separate process that reads POLICY.md alone) against the grader: coverage,
                gap, T_res, expected outcome and discrepancy_type; the stored labels against the grader
  reachability  ap_sim.verification.reachability.check: the expected outcome reachable at reward 1, no rival
  leakage       the leakage audit's probes on the shortfall cases, per split and pooled
  baselines     always release, always keep blocked, always pending, always request confirmation, and the zoo,
                on surfaces/4 graded by g7: success (verdict 1) and outcome-correct rates per split
  bounds        from baselines.json, no agent rerun: the constant-strategy baselines with both columns, g7 success
                and the correct-outcome rate labelled "upper bound for a constant strategy" (2026-10-31)

The independent verifier and the helper modules that the reference, leakage and baselines parts import (episode_limits,
leakage_audit, zoo_matrix) are not part of this release. Only the scope and reachability parts run from the release
alone. The script is kept as the record of the checks.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/phase2/check_splits.py <part|all> [--out runs/phase2/checks]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for d in ("scripts", "scripts/step5", "scripts/step9"):
    sys.path.insert(0, str(ROOT / d))

from ap_sim.events.clock import HORIZON   # noqa: E402
from ap_sim.generator.causal import check_scope   # noqa: E402
from ap_sim.generator.manifests import _tuples   # noqa: E402
from ap_sim.generator.splits import DIR, SPLITS, load_split   # noqa: E402
from ap_sim.grader.grader import grade   # noqa: E402
from ap_sim.interfaces.structured import StructuredSurface   # noqa: E402
from ap_sim.interfaces.tools import ToolLayer   # noqa: E402
from ap_sim.training.config import (PIPELINE_MAX_COMPLETION_LENGTH, PIPELINE_MAX_CONTEXT_LENGTH,   # noqa: E402
                                    PIPELINE_MAX_TOOL_CALLING_ITERATIONS)

LIMITS = {"tool_calls": PIPELINE_MAX_TOOL_CALLING_ITERATIONS, "completion_tokens": PIPELINE_MAX_COMPLETION_LENGTH,
          "context_tokens": PIPELINE_MAX_CONTEXT_LENGTH}


def entries(split: str) -> list[dict]:
    return json.loads((DIR / f"{split}.json").read_text())["entries"]


def factories(split: str) -> dict:
    return load_split(split, require_frozen=False)


class _Case:
    """A manifest entry with the interface the leakage audit and the zoo use (scenario(), variant, variables)."""

    def __init__(self, e: dict, factory):
        self.case_id, self.variant, self._f = e["case_id"], e["variant"], factory
        self.variables = {"gap": 1 if e["discrepancy_type"] == "receipt_coverage_gap" else 0}

    def scenario(self):
        return self._f()


# --- parts ---------------------------------------------------------------------------------------------------------
def scope() -> dict:
    out = {}
    for split in SPLITS:
        es, f = entries(split), factories(split)          # the loader itself refuses anything out of its split
        for e in es:
            check_scope(_tuples(e["params"]))
        out[split] = {"cases": len(es), "loaded": len(f), "scope_failures": 0}
    return out


def reference() -> dict:
    from episode_limits import measure
    from qwen_check import MODEL, REVISION
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema, get_training_chat_template
    from phase0_5_compare import grader_side
    tok = add_response_schema(AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True))
    template = get_training_chat_template(tok)
    out, failures = {}, []
    for split in SPLITS:
        rows = []
        for cid, f in factories(split).items():
            r = measure(cid, f, tok, template)
            g = grader_side(f())
            r["t_res"] = None if g["t_res"] in (None, float("inf")) else float(g["t_res"])
            r["expected"], r["delay"] = g["expected"], g["delay"]
            rows.append(r)
            if r["verdict_reward"] != 1:
                failures.append({"split": split, "case": cid, "critical": r["verdict_critical"]})
        resolving = [r for r in rows if r["expected"] != "pending"]
        pending_late = [r for r in rows if r["expected"] == "pending" and r["delay"] is not None]
        out[split] = {
            "cases": len(rows), "reward_1": sum(r["verdict_reward"] == 1 for r in rows),
            "largest": {k: max(r[k] for r in rows) for k in LIMITS},
            "closest_margin_to_limit": {k: LIMITS[k] - max(r[k] for r in rows) for k in LIMITS},
            "worst_case": {k: max(rows, key=lambda r: r[k])["case"] for k in LIMITS},
            "horizon": {"H": HORIZON,
                        "resolving_closest_margin": min(HORIZON - r["t_res"] for r in resolving) if resolving else None,
                        "resolving_latest_t_res": max(r["t_res"] for r in resolving) if resolving else None,
                        "pending_answer_closest_after_H": (min(r["delay"] for r in pending_late) - HORIZON)
                        if pending_late else None}}
    return {"limits": LIMITS, "by_split": out, "failures": failures}


def verifier(out_dir: pathlib.Path) -> dict:
    from phase0_5_compare import grader_side, num, same_t, verifier_side
    result, disagreements = {}, []
    for split in SPLITS:
        vout = out_dir / f"verifier-{split}.json"
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "independent_verifier.py"), "--manifest",
                        str(DIR / f"{split}.json"), "--out", str(vout), "--policy", str(ROOT / "docs" / "POLICY.md")],
                       check=True, capture_output=True)
        vcases = {c["case_id"]: c for c in json.loads(vout.read_text())["cases"]}
        tally = collections.Counter()
        stored = {e["case_id"]: e for e in entries(split)}
        for cid, f in factories(split).items():
            s = f()
            g, raw = grader_side(s), vcases[cid]
            v = verifier_side(raw)
            g7 = grade(s, ToolLayer(s, StructuredSurface(), "surfaces/4").episode_log())
            want_dt = g7.diagnostics["discrepancy_type"]["required"]
            diffs = []
            if v["coverage"] is None or num(v["coverage"]) != num(g["coverage"]):
                diffs.append("coverage")
            if v["gap"] is None or num(v["gap"]) != num(g["gap"]):
                diffs.append("gap")
            if not same_t(g["t_res"], v):
                diffs.append("t_res")
            if v["expected"] != g["expected"]:
                diffs.append("expected_outcome")
            if raw.get("expected_discrepancy_type") != want_dt:
                diffs.append("discrepancy_type")
            if raw.get("error") or raw.get("expected_discrepancy_type") is None:
                diffs.append("verifier_error_or_out_of_scope")
            if stored[cid]["expected"] != g7.expected or stored[cid]["discrepancy_type"] != want_dt:
                diffs.append("stored_label")
            tally["cases"] += 1
            tally["agree_all" if not diffs else "disagree"] += 1
            for d in diffs:
                tally[f"disagree:{d}"] += 1
            if raw.get("warnings"):
                tally["verifier_warnings"] += 1
            if diffs:
                disagreements.append({"split": split, "case": cid, "fields": diffs,
                                      "grader": {k: str(x) for k, x in g.items()}, "g7_discrepancy_type": want_dt,
                                      "verifier": {k: v[k] for k in ("coverage", "gap", "t_res", "expected", "error")}
                                      | {"discrepancy_type": raw.get("expected_discrepancy_type")}})
        result[split] = dict(tally)
    return {"by_split": result, "disagreements": disagreements}


def reachability() -> dict:
    from ap_sim.verification.reachability import check
    out, problems = {}, []
    for split in SPLITS:
        t = collections.Counter()
        for cid, f in factories(split).items():
            r = check(f())
            t["cases"] += 1
            t["reachable"] += r["reachable"]
            t["rivals"] += bool(r["rivals"])
            t["rivals_needing_hidden_knowledge"] += bool(r["rivals_needing_hidden_knowledge"])
            t["beats_template"] += r["beats_template"]
            t["later_than_template"] += r["later_than_template"]
            if not r["reachable"] or r["rivals"] or r["later_than_template"]:
                problems.append({"split": split, "case": cid, "reachable": r["reachable"], "rivals": r["rivals"],
                                 "later_than_template": r["later_than_template"], "expected": r["expected"]})
        out[split] = dict(t)
    return {"by_split": out, "problems": problems}


def leakage() -> dict:
    import leakage_audit as la
    la.SCHEMA = "surfaces/4"
    cache = {}

    def features(split):
        if split not in cache:
            fs, es = factories(split), entries(split)
            rows = []
            for e in es:
                if e["discrepancy_type"] != "receipt_coverage_gap" or e["policy"] != "sandbox-v1":
                    continue      # the outcome is hidden only when there is a gap
                c = _Case(e, fs[e["case_id"]])
                _, obs = la.pages(c)
                rows.append((la.forbidden_features(obs), la.forbidden_features(la.pre_answer(c), "wait"),
                             la.business_features(obs), la.OUTCOMES.index(e["expected"])))
            cache[split] = rows
        return cache[split]

    def block(rows):
        y = np.array([r[3] for r in rows])
        Xf, fn = la._matrix([r[0] for r in rows])
        Xw, wn = la._matrix([r[1] for r in rows])
        Xb = np.array([r[2] for r in rows])
        return {"shortfall_cases": len(rows), "outcome_mix": {la.OUTCOMES[i]: int((y == i).sum()) for i in range(3)},
                "forbidden_before_acting": {"channels": la.informative(Xf, fn), **la.probes(Xf, y)},
                "forbidden_while_waiting": {"channels": la.informative(Xw, wn), **la.probes(Xw, y)},
                "forbidden_all": la.probes(np.c_[Xf, Xw], y),
                "visible_business_evidence": la.probes(Xb, y),
                "positive_control_planted_outcome": la.probes(np.c_[Xf, y + np.random.default_rng(2).normal(0, 0.8, len(y))], y)}

    groups = {"test+test_reserve": ("test", "test_reserve"), "rl": ("rl",), "sft": ("sft",), "v6": ("v6",),
              "pooled non-V6 (sft, rl, dev, regression, test, reserve)": ("sft", "rl", "dev", "regression", "test",
                                                                          "test_reserve")}
    out = {name: block([r for s in ss for r in features(s)]) for name, ss in groups.items()}

    def flags(b):
        return [f"{blk}/{probe}" for blk in ("forbidden_before_acting", "forbidden_while_waiting", "forbidden_all",
                                             "visible_business_evidence")
                for probe in ("logistic", "boosted_stumps") if not b[blk][probe]["at_chance"]]
    return {"blocks": out, "flags": {n: flags(b) for n, b in out.items()},
            "positive_controls_detected": {n: [p for p in ("logistic", "boosted_stumps")
                                               if not b["positive_control_planted_outcome"][p]["at_chance"]]
                                           for n, b in out.items()}}


def baselines() -> dict:
    import zoo_matrix as zm
    zm.SCHEMA = "surfaces/4"
    named = {"always release": "always-release", "always keep blocked": "claim-success: keep_blocked",
             "always pending": "claim-success: pending", "always request confirmation": "request-then-stop"}
    out = {}
    for split in SPLITS:
        fs, es = factories(split), entries(split)
        cases = [(e, _Case(e, fs[e["case_id"]])) for e in es]
        rows = {}
        for name, (cls, kw) in zm.AGENTS.items():
            succ = correct = 0
            for e, c in cases:
                r = zm.run(cls, kw, c, 0)
                succ += r["reward"] == 1
                correct += r["reached"] == e["expected"]
            rows[name] = {"success": round(succ / len(cases), 4), "outcome_correct": round(correct / len(cases), 4)}
        out[split] = {"cases": len(cases), "named_baselines": {k: rows[v] for k, v in named.items()},
                      "constant_strategies": constant_bounds(es, {k: rows[v] for k, v in named.items()}),
                      "zoo": rows}
    return {"agents": list(zm.AGENTS), "named": named, "by_split": out}


CLAIM_ONLY = {"always keep blocked": "keep_blocked", "always pending": "pending"}
UPPER = "correct outcome (upper bound for a constant strategy)"


def constant_bounds(es: list, named_rows: dict) -> dict:
    """Both columns for each constant strategy. Always release and always request confirmation act, so their correct-
    outcome rate is measured (outcome reached = expected). Always keep blocked and always pending are the zoo's
    claim-success agents (unchanged): they act on nothing and reach no outcome, so theirs is the share of cases whose
    expected outcome is that disposition, the most the strategy can reach. The best of these is the bound."""
    n = len(es)
    rows = {}
    for name, r in named_rows.items():
        if name in CLAIM_ONLY:
            upper, basis = sum(e["expected"] == CLAIM_ONLY[name] for e in es) / n, "share of expected outcome"
        else:
            upper, basis = r["outcome_correct"], "measured"
        rows[name] = {"g7 success": r["success"], UPPER: round(upper, 4), "basis": basis}
    best = max(rows, key=lambda k: rows[k][UPPER])
    return {"rows": rows, "best_upper_bound": {"strategy": best, "rate": rows[best][UPPER]}}


def bounds(out: pathlib.Path) -> dict:
    b = json.loads((out / "baselines.json").read_text())["by_split"]
    return {"by_split": {s: constant_bounds(entries(s), b[s]["named_baselines"]) for s in SPLITS}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["scope", "reference", "verifier", "reachability", "leakage", "baselines", "bounds",
                                     "all"])
    ap.add_argument("--out", default=str(ROOT / "runs" / "phase2" / "checks"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = ["scope", "verifier", "reachability", "baselines", "leakage", "reference"] if a.part == "all" else [a.part]
    for p in parts:
        t0 = time.time()
        r = verifier(out) if p == "verifier" else bounds(out) if p == "bounds" else globals()[p]()
        r["seconds"] = round(time.time() - t0, 1)
        (out / f"{p}.json").write_text(json.dumps(r, indent=1, default=str))
        brief = {k: v for k, v in r.items() if k not in ("disagreements", "problems", "blocks")}
        print(f"== {p}", json.dumps(brief, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
