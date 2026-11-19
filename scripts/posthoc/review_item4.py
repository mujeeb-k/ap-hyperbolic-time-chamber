"""A scripted heuristic baseline, run once with no tuning (post hoc, 2026-11-12).

The design, recorded before this script ran. The agent uses only what the tools return and only click, fill, advance_time and finish.
It never calls coverage, never reads the Active policy page and does no coverage arithmetic: the system's own reprocessing decides eligibility.
  1. Reprocess the invoice and read its eligibility on the returned page.
  2. Eligible: release, then finish.
  3. Not eligible: request delivery confirmation; then repeatedly view Delivery confirmations. If a request is answered, act on it (4). If the page's
     clock is D5 16:35 or later and nothing is answered, keep blocked as confirmation pending, then finish. Otherwise advance by
     min(480, minutes to D5 16:30) and view again.
  4. Act on the latest answer: view Receipt history and take the net of the PO line's receipts shown (receipts minus reversals). If the answer's
     delivered_total exceeds that net, fill the difference, request receipt posting and reprocess. If the invoice is then eligible, release it.
     Otherwise keep blocked with a shortfall correction. Then finish.
  5. Finish with a minimal report (disposition, the invoice line quantity cited as observed, discrepancy type). The report is not scored here.
Environment: APReceiptEnvV2 with loop-guard/1, purpose evaluation; graded by the unchanged g8. A rejected call stops the agent where it is.
Reported per set and family: right final action (g8 reached == expected), attempted unsafe actions (g8 diagnostics unsafe_attempts), executed
unsafe actions (evaluation.records._executed_violations), and agent stops. Counts and case ids only; no case content is printed or written.

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice .venv/bin/python scripts/posthoc/review_item4.py [--sets dev] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from decimal import Decimal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import posthoc_common as c   # noqa: E402
import review_anchor   # noqa: E402  (the anchor rule, 2026-11-12)

from ap_sim.events.clock import parse_time   # noqa: E402

OUT = c.ROOT / "runs/posthoc/review_findings/item4.json"
CHECK_AT = parse_time("D5 16:35")
VIEW_AT = parse_time("D5 16:30")   # a view costs 5 minutes, so a view started here shows D5 16:35


class Stop(Exception):
    pass


class HeuristicAgent:
    def __init__(self, tools):
        self.tools = tools
        self.obs = tools.initial_observation
        self.seen: dict[str, tuple[dict, dict]] = {}
        self._remember(self.obs)

    def _remember(self, obs: dict) -> None:
        for region in obs["regions"]:
            for rec in region["records"]:
                self.seen[rec["id"]] = ({k: obs[k] for k in ("observation_id", "state_version", "page")}, rec)

    def _records(self, role: str) -> list[dict]:
        return [rec for r in self.obs["regions"] if r["role"] == role for rec in r["records"]]

    def _step(self, result: dict) -> dict:
        if not result["ok"]:
            raise Stop(result["error"]["code"] if isinstance(result.get("error"), dict) else "rejected")
        if result.get("observation") is not None:
            self.obs = result["observation"]
            self._remember(self.obs)
        return result

    def click(self, name: str) -> None:
        ids = [ctl["id"] for r in self.obs["regions"] for ctl in r["controls"] if ctl["name"] == name]
        if len(ids) != 1:
            raise Stop(f"control not on page: {name}")
        self._step(self.tools.click(ids[0]))

    def clock(self) -> int:
        return parse_time(self.obs["clock"])

    def eligible(self) -> bool:
        heads = self._records("invoice_header")
        if not heads:                    # the action returned another page: open the Invoice page to read the header
            self.click("Invoice")
            heads = self._records("invoice_header")
        if not heads:
            raise Stop("no invoice header on the page")
        return heads[0]["fields"].get("eligibility") == "eligible"

    def cite(self, record_id: str, field: str) -> dict:
        head, rec = self.seen[record_id]
        return {"form": "observed", "source_id": head["page"], "observation_id": head["observation_id"], "state_version": head["state_version"],
                "record_id": record_id, "record_version": rec["version"], "field": field, "value": rec["fields"][field]}

    def finish(self, disposition: str, kind: str) -> None:
        report = {"findings": [{"discrepancy_type": kind, "root_cause": "Heuristic baseline.", "blocking_comparison": "invoice_vs_receipt",
                                "evidence": [self.cite(self.line_id, "quantity")]}],
                  "final_disposition": disposition, "discrepancy_type": kind}
        self.tools.finish(report)   # not scored here; a rejected report ends the episode as it stands

    def run(self) -> str:
        (line,) = self._records("invoice_lines")
        self.line_id, po_line = line["id"], line["fields"]["po_line"]
        self.click("Reprocess invoice")
        if self.eligible():
            self.click("Release invoice")
            self.finish("release", "none")
            return "release"
        self.click("Request delivery confirmation")
        while True:
            self.click("Delivery confirmations")
            answered = [r for r in self._records("confirmations") if r["fields"]["status"] == "answered"]
            if answered:
                break
            if self.clock() >= CHECK_AT:
                self.click("Keep blocked: confirmation pending")
                self.finish("pending", "receipt_coverage_gap")
                return "pending"
            self._step(self.tools.advance_time(max(5, min(480, VIEW_AT - self.clock()))))
        conf = max(answered, key=lambda r: (r["fields"]["answered_at"], r["id"]))
        delivered = Decimal(conf["fields"]["delivered_total"]["value"])
        self.click("Receipt history")
        net = sum(Decimal(r["fields"]["quantity"]["value"]) * (1 if r["fields"]["movement"] == "receipt" else -1)
                  for r in self._records("receipts") if r["fields"]["po_line"] == po_line)
        if delivered > net:
            ids = [ctl["id"] for r in self.obs["regions"] for ctl in r["controls"] if ctl["name"] == "Quantity (EA)"]
            if len(ids) != 1:
                raise Stop("control not on page: Quantity (EA)")
            self._step(self.tools.fill(ids[0], format((delivered - net).normalize(), "f")))
            self.click("Request receipt posting")
            self.click("Reprocess invoice")
            if self.eligible():
                self.click("Release invoice")
                self.finish("release", "receipt_coverage_gap")
                return "release"
        self.click("Keep blocked: shortfall correction")
        self.finish("keep_blocked", "receipt_coverage_gap")
        return "keep_blocked"


def family(case: str) -> str:
    """The case family: V1..V6 from 'set-V1-3'; on the policy shift, the program (PS shift, PB boundary) and the policy version (v1, v2)."""
    if case.startswith(("PS", "PB")):
        return f"{case[:2]}-{case.rsplit('-', 1)[1]}"
    return case.split("-")[1]


def tally(rows: dict) -> dict:
    fam = collections.defaultdict(lambda: {"cases": 0, "right_final_action": 0, "attempted_unsafe_episodes": 0, "attempted_unsafe": 0,
                                           "executed_unsafe": 0, "agent_stops": 0, "wrong_cases": []})
    for case, r in rows.items():
        for key in (family(case), "all"):
            f = fam[key]
            f["cases"] += 1
            f["right_final_action"] += r["right"]
            f["attempted_unsafe_episodes"] += r["attempted_unsafe"] > 0
            f["attempted_unsafe"] += r["attempted_unsafe"]
            f["executed_unsafe"] += r["executed_unsafe"]
            f["agent_stops"] += r["stop"] is not None
            if not r["right"]:
                f["wrong_cases"].append(case)
    return dict(sorted(fam.items()))


def episode(case: str, factory) -> dict:
    from ap_sim.evaluation.records import _executed_violations
    from ap_sim.grading import g8
    from ap_sim.training.environment import APReceiptEnvV2
    from ap_sim.training.loop_guard import LOOP_GUARD
    env = APReceiptEnvV2({case: factory}, execution_rule=LOOP_GUARD)
    obs = json.loads(env.reset(scenario=case, purpose="evaluation"))

    def call(name, args):
        return json.loads(getattr(env, name)(**args))

    class Tools:
        initial_observation = obs
        click = staticmethod(lambda element_id: call("click", {"element_id": element_id}))
        fill = staticmethod(lambda element_id, text: call("fill", {"element_id": element_id, "text": text}))
        advance_time = staticmethod(lambda minutes: call("advance_time", {"minutes": minutes}))
        finish = staticmethod(lambda report: call("finish", report))

    stop = None
    try:
        HeuristicAgent(Tools()).run()
    except Stop as exc:
        stop = str(exc)
    scenario, log, _ = env._episode()
    v = g8.grade(scenario, log)
    return {"family": family(case), "expected": v.expected, "reached": v.reached, "right": v.reached == v.expected,
            "attempted_unsafe": int(v.diagnostics.get("unsafe_attempts") or 0), "executed_unsafe": _executed_violations(scenario, log),
            "stop": stop, "calls": len(log["steps"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["test", "v6", "policy_shift", "stress"])
    ap.add_argument("--limit", type=int, default=None, help="debugging on dev only")
    ap.add_argument("--regroup", action="store_true", help="recompute the family tallies of the stored item4.json, without re-running the agent")
    a = ap.parse_args()
    anchors = None if a.limit else review_anchor.check("item4")
    if a.regroup:
        out = json.loads(OUT.read_text())
        for s, d in out.items():
            if not isinstance(d, dict) or "cases" not in d:
                continue
            for case, r in d["cases"].items():
                r["family"] = family(case)
            d["by_family"] = tally(d["cases"])
        out["anchor"] = anchors
        OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(OUT, "regrouped")
        return
    out = {}
    for s in a.sets:
        factories = c.cases(s)
        ids = list(factories)[: a.limit] if a.limit else list(factories)
        rows = {case: episode(case, factories[case]) for case in ids}
        out[s] = {"by_family": tally(rows), "cases": rows,
                  "reached_vs_expected": dict(collections.Counter(f"{r['expected']}->{r['reached']}" for r in rows.values())),
                  "stops": dict(collections.Counter(r["stop"] for r in rows.values() if r["stop"]))}
        print(s, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "wrong_cases"} for k, v in out[s]["by_family"].items()}))
        print(" ", out[s]["reached_vs_expected"], out[s]["stops"])
    if not a.limit:
        out["anchor"] = anchors
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(OUT)


if __name__ == "__main__":
    main()
