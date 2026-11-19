"""The metrics of the Phase 6 gate (Phase 6 of the pre-registered plan, SFT and its gate, with pre-data decisions of
2026-11-01): twin flip 90%, parse rate 98%, calculator use 95%, report-valid 90% on dev greedy (failure path: fix the
demonstrations before any RL); forced recovery reported without a threshold. All on dev cases:

  parse rate       of the turns that attempted a tool call (a parsed call, or <tool_call> text TRL could not parse,
                   in the turn's text or in a <think> block TRL moved into reasoning_content), the share TRL's parser
                   turned into a tool call with no unparsed call left in the turn;
  calculator use   the share of episodes that called coverage at least once;
  twin flip        on newly generated counterfactual siblings of the dev cases (the same visible records, another
                   hidden world: `delivered` and `timing`; checked to be no case of any frozen split, test included)
                   whose expected outcomes differ, the share of pairs where both siblings reach the correct outcome
                   (outcome correctness; also reported: the share where the two final actions merely differ);
  forced recovery  episodes started from a planted mistake (the Phase 4 builder's mistakes, on dev cases: a malformed
                   call, a rejected report, a stale click; the conversation up to and including the mistake and its
                   result is given): the share whose next call reaches the environment and is accepted (recovered)
                   and the share that succeed; reported, no threshold;
  report-valid     on dev greedy: an accepted finish with no report-critical flag.
"""
from __future__ import annotations

import collections
import random

from ap_sim.evaluation import metrics

THRESHOLDS = {"twin_flip": 0.90, "parse_rate": 0.98, "calculator_use": 0.95, "report_valid": 0.90}
FAILURE_PATHS = {"twin_flip": "below 90% after two fix attempts: stop and reconsider the data",
                 "parse_rate": "below 98%: proceeding needs a reason recorded in V2.md",
                 "calculator_use": "below 95%: proceeding needs a reason recorded in V2.md",
                 "report_valid": "below 90% on dev greedy: fix the demonstrations before any RL"}
FORCED_SEED = 20261104       # the forced-recovery plans' seed (recorded with the gate's definitions)


def case_type(case_id: str) -> str:
    return case_id.split("-")[1]


def siblings(entries: dict) -> tuple[dict, list[tuple[str, str, str]]]:
    """Counterfactual siblings of the given split entries: ({sibling id: scenario factory}, [(case, change, sibling)])."""
    from ap_sim.generator.causal import counterfactual, generate
    from ap_sim.generator.splits import keys
    frozen = frozen_keys()
    factories, pairs = {}, []
    for cid, e in entries.items():
        case = generate(cid, e["seed"], e["force"])
        for change in ("delivered", "timing"):
            try:
                sib = counterfactual(case, change)
            except Exception:          # out of scope for this case (e.g. a covered invoice): no sibling
                continue
            k = keys(sib.seed, sib.params)
            if k["near"] == keys(case.seed, case.params)["near"]:
                continue                       # no change (e.g. timing on a case already answered too late)
            if k["exact"] in frozen or k["near"] in frozen:
                raise ValueError(f"sibling {sib.case_id} is a case of a frozen split")
            scenario = sib.scenario()
            factories[scenario.name] = (lambda s=sib: s.scenario())
            pairs.append((cid, change, scenario.name))
    return factories, pairs


def frozen_keys() -> set[str]:
    """The exact and near keys of every case in every frozen split and the forgetting set (siblings must be none)."""
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3] / "manifests"
    out = set()
    for f in list((root / "v2").glob("*.json")) + list((root / "forgetting_l1").glob("*.json")):
        data = json.loads(f.read_text())
        for e in data.get("entries", []) if isinstance(data, dict) else []:
            k = e.get("keys") or {}
            out |= {k.get("exact"), k.get("near")} - {None}
    return out


def forced_plans(entries: dict, seed: int = FORCED_SEED) -> list[dict]:
    """One forced-recovery start per dev case and mistake kind that applies (stale only on V1, V2, V3a, V3b), built
    with the Phase 4 builder: [{case, kind, plan, prefix (messages through the mistake's result), continuation}]."""
    import demos
    from ap_sim.generator.splits import load_split
    split = load_split("dev")
    rng = random.Random(seed)
    out = []
    for i, (cid, e) in enumerate(entries.items()):
        group = e["keys"]["group"]
        kinds = [("malformed", {"variant": ("name", "type")[i % 2]}),
                 ("report_defect", {"defect": demos.DEFECTS[i % len(demos.DEFECTS)]})]
        if case_type(cid) in demos.STALE_TYPES:
            kinds.append(("stale", {}))
        for kind, extra in kinds:
            plan = {"mistake": kind, **extra}
            if kind == "malformed":
                n = demos.demonstrate(split[cid](), group, {})["calls"]
                for _ in range(20):
                    plan["at_call"] = rng.randint(1, n)
                    try:
                        d = demos.demonstrate(split[cid](), group, plan)
                        break
                    except Exception:
                        continue
            else:
                d = demos.demonstrate(split[cid](), group, plan)
            cut = d["mistake_at"] + 1
            out.append({"case": cid, "kind": kind, "plan": dict(plan), "prefix": d["messages"][:cut],
                        "continuation": [m for m in d["messages"][cut:] if m["role"] == "assistant"
                                         and m.get("tool_calls")]})
    return out


def parse_rate(records: list[dict]) -> dict:
    attempted = sum(r["turn_parse"]["attempted"] for r in records)
    return metrics.wilson(sum(r["turn_parse"]["parsed"] for r in records), attempted)


def calculator_use(records: list[dict]) -> dict:
    return metrics.wilson(sum(1 for r in records if r["coverage_calls"] > 0), len(records))


def twin_flip(by_case: dict, pairs: list[tuple[str, str, str]]) -> dict:
    """by_case: case or sibling id -> its record (greedy). Pairs whose expected outcomes differ count."""
    usable = [(a, ch, b) for a, ch, b in pairs if a in by_case and b in by_case
              and by_case[a]["expected"] != by_case[b]["expected"]]
    correct = sum(1 for a, _, b in usable if by_case[a]["outcome_correct"] and by_case[b]["outcome_correct"])
    differ = sum(1 for a, _, b in usable if by_case[a]["final_action"] != by_case[b]["final_action"])
    return {"pairs": len(usable), "flip": metrics.wilson(correct, len(usable)),
            "decision_differs": metrics.wilson(differ, len(usable)),
            "by_change": dict(collections.Counter(ch for _, ch, _ in usable))}


def forced_recovery(records: list[dict]) -> dict:
    """records of forced episodes carry `forced`: {kind, recovered (the first call after the prefix was accepted)}."""
    by_kind = collections.defaultdict(list)
    for r in records:
        by_kind[r["forced"]["kind"]].append(r)
    summary = lambda rs: {"episodes": len(rs),   # noqa: E731
                          "recovered": metrics.wilson(sum(r["forced"]["recovered"] for r in rs), len(rs)),
                          "success": metrics.wilson(sum(r["success"] for r in rs), len(rs))}
    return {"all": summary(records), "by_kind": {k: summary(v) for k, v in sorted(by_kind.items())}}


def success_by_type(records: list[dict]) -> dict:
    by = collections.defaultdict(list)
    for r in records:
        by[case_type(r["case"])].append(r["success"])
    return {t: metrics.wilson(sum(v), len(v)) for t, v in sorted(by.items())}


def verdict(summary: dict) -> dict:
    """Each threshold met or not (point estimates), with its failure path; forced recovery has no threshold."""
    got = {"twin_flip": summary["twin_flip"]["flip"]["rate"], "parse_rate": summary["parse_rate"]["rate"],
           "calculator_use": summary["calculator_use"]["rate"],
           "report_valid": summary["greedy"]["report_valid"]["rate"]}
    return {k: {"value": v, "threshold": THRESHOLDS[k], "met": v is not None and v >= THRESHOLDS[k],
                "failure_path": FAILURE_PATHS[k]} for k, v in got.items()}
