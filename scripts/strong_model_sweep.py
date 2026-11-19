"""Strong-model check, full sweep: 30 train-pool cases x versions A, B, C through strong_model_check.

Run on 2026-10-29, before any training.

Cases: 5 per type (V1, V2, V3a, V3b, V4, V5), drawn with a fixed seed from manifests/frozen/train_pool.json. V6 is
v2's held-out type and is never drawn. Order: round-robin over the types, each case run as A, B, C in turn, so a stop
leaves the versions and types balanced.

Guards (set 2026-10-29):
  - cost cap: before each case, stop if the spend so far plus 3 x the largest episode cost seen (at least $0.30) would
    exceed the cap;
  - every episode is written to disk (its own file plus one line of sweep.jsonl) as soon as it ends;
  - a crash or timeout is recorded as a failure and the sweep moves on; no episode is ever re-run (an existing file is
    skipped, so a restarted sweep resumes);
  - the same failure on three consecutive episodes stops the sweep;
  - rate-limit/overload backoff lives in strong_model_check.run_claude.

    uv run --frozen --with anthropic==2.3.0 python scripts/strong_model_sweep.py run
    uv run --frozen python scripts/strong_model_sweep.py run --dry-run-reference   # $0, reference/4
    uv run --frozen python scripts/strong_model_sweep.py report

--gate1a (2026-10-30): the strong-model check on the final v2 interface, the same 30 cases as version V2, at most $5:
    uv run --frozen --with anthropic==2.3.0 python scripts/strong_model_sweep.py run --gate1a
    uv run --frozen python scripts/strong_model_sweep.py run --gate1a --dry-run-reference   # $0, reference/5
    uv run --frozen python scripts/strong_model_sweep.py report --gate1a
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import strong_model_check as smc   # noqa: E402

TYPES = ("V1_delivered_unrecorded", "V2_not_delivered", "V3a_late_delivered", "V3b_late_not_delivered", "V4_pending",
         "V5_stale_covered")
PER_TYPE = 5
SEED = 20261029
VERSIONS = ("A", "B", "C")
COST_CAP = 15.00
MIN_EPISODE_ESTIMATE = 0.30
OUT = smc.ROOT / "runs" / "2026-10-29-strong-model-sweep"


def select_cases() -> list[tuple[str, str]]:
    """(case_id, variant) in run order: round-robin over the types."""
    entries = json.loads(smc.MANIFEST.read_text())["entries"]
    rng = random.Random(SEED)
    picks = {}
    for t in TYPES:
        pool = sorted(e["case_id"] for e in entries if e["variant"] == t)
        picks[t] = rng.sample(pool, PER_TYPE)
    assert not any(e["variant"].startswith("V6") for e in entries if e["case_id"] in sum(picks.values(), []))
    return [(picks[t][i], t) for i in range(PER_TYPE) for t in TYPES]


def failure_signature(rec: dict) -> str | None:
    if "crash" in rec:
        return f"crash:{rec['crash']['type']}"
    stop = rec["verdict"]["stop_reason"]
    if stop in smc.API_FAILURES:
        f = rec.get("failure") or {}
        return f"{stop}:{f.get('type')}:{f.get('status')}"
    return None


# --gate1a runs the strong-model check on the final v2 interface (2026-10-30).
GATE1A = {"versions": ("V2",), "cap": 5.00, "out": smc.ROOT / "runs" / "2026-10-30-gate1a-strong-model"}


def run(dry: bool, gate1a: bool = False) -> None:
    global VERSIONS, COST_CAP, OUT
    if gate1a:
        VERSIONS, COST_CAP, OUT = GATE1A["versions"], GATE1A["cap"], GATE1A["out"]
    out = OUT / ("dry-run" if dry else "episodes")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "sweep.jsonl"
    cases = select_cases()
    (out / "selection.json").write_text(json.dumps({"seed": SEED, "per_type": PER_TYPE, "cases": cases}, indent=1))
    done = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    spent = sum(d["cost_usd"] for d in done)
    largest = max([d["cost_usd"] for d in done] + [MIN_EPISODE_ESTIMATE])
    recent: list[str | None] = [d.get("failure_signature") for d in done][-3:]
    for case, variant in cases:
        if not dry and spent + len(VERSIONS) * largest > COST_CAP:
            print(f"STOP (cost cap): spent ${spent:.4f}; case {case} could cost up to ${len(VERSIONS) * largest:.2f}")
            return
        for version in VERSIONS:
            path = out / f"{case}-{version}.json"
            if path.exists():
                continue
            started = time.time()
            try:
                rec = (smc.run_reference(case, version) if dry
                       else smc.run_claude(case, version, smc.MODEL, "off", "high"))
            except Exception as exc:   # a runner crash: recorded as this episode's outcome, never re-run
                rec = {"config": {"case": case, "version": version},
                       "crash": {"type": type(exc).__name__, "message": str(exc)[:2000],
                                 "traceback": traceback.format_exc()[-4000:]},
                       "wall_seconds": round(time.time() - started, 1)}
            rec.setdefault("cost_usd", 0.0)
            path.write_text(json.dumps(rec, indent=1, default=str))
            sig = failure_signature(rec)
            v = rec.get("verdict", {})
            row = {"case": case, "variant": variant, "version": version, "cost_usd": rec["cost_usd"],
                   "stop_reason": v.get("stop_reason", "crash"), "reward": v.get("reward"), "expected": v.get("expected"),
                   "reached": v.get("reached"), "critical": v.get("critical"), "failure_signature": sig,
                   "backoffs": len(rec.get("backoffs", [])), "file": path.name}
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            spent += rec["cost_usd"]
            largest = max(largest, rec["cost_usd"])
            print(f"{case} {version}: {row['stop_reason']} reward {row['reward']} reached {row['reached']} "
                  f"(expected {row['expected']}) ${rec['cost_usd']} total ${spent:.4f}", flush=True)
            recent = (recent + [sig])[-3:]
            if len(recent) == 3 and recent[0] is not None and len(set(recent)) == 1:
                print(f"STOP (same failure on three consecutive episodes): {sig}")
                return
    print(f"DONE: spent ${spent:.4f}")


# Report -------------------------------------------------------------------------------------------------------------
def _observed_values(log: dict) -> dict:
    """(observation_id, record_id, record_version, field) -> value, over every observation delivered to the agent."""
    out = {}
    observations = [log["initial_observation"]] + [s["result"].get("observation") for s in log["steps"]]
    for obs in observations:
        for region in (obs or {}).get("regions", []):
            for rec in region["records"]:
                for k, val in rec["fields"].items():
                    out[(obs["observation_id"], rec["id"], rec["version"], k)] = val
    return out


def citation_kinds(rec: dict) -> collections.Counter:
    """Citation and required-evidence problems of one episode's accepted report, by kind (each problem counted once).
    value_format: a cited value that equals the shown field's "value" but omits the object around it;
    invoice_not_line: the invoice line required, the resolved invoice's own invoice record cited instead;
    follow_up_version: the opened follow-up cited, but not at a (observation, version) where the agent saw it;
    other: everything else (a wrong value, a citation that does not exist, a required record not cited)."""
    kinds = collections.Counter()
    v, log = rec["verdict"], rec["episode_log"]
    report = v.get("report")
    if not report:
        return kinds
    seen = _observed_values(log)
    evidence = (report.get("evidence") or []) + [e for f in report["findings"] for e in (f.get("evidence") or [])]
    observed = [e for e in evidence if e["form"] == "observed"]
    explained = set()   # (record_id, field) whose requirement failure a classified citation problem explains
    for e in observed:
        key = (e["observation_id"], e["record_id"], e["record_version"], e["field"])
        if key in seen and seen[key] != e["value"]:
            shown = seen[key]
            if isinstance(shown, dict) and "value" in shown and str(e["value"]) == str(shown["value"]):
                kinds["value_format"] += 1
            else:
                kinds["other"] += 1
            explained.add((e["record_id"], e["field"]))
    for p in v["citation_problems"]:
        if not p.startswith("value mismatch"):
            kinds["other"] += 1
    inv_id = log["current_invoice"]
    for r in v["required_evidence"]:
        if r["satisfied"] or (r["record_id"], r["field"]) in explained:
            continue
        cited = [e for e in observed if e["record_id"] == r["record_id"] and e["field"] == r["field"]]
        if r["record_id"].startswith("WI-"):
            kinds["follow_up_version" if cited else "other"] += 1
        elif r["record_id"] == f"{inv_id}-1" and any(e["record_id"] == inv_id for e in observed):
            kinds["invoice_not_line"] += 1
        else:
            kinds["other"] += 1
    return kinds


def report(dry: bool, folder: pathlib.Path | None = None, versions: tuple[str, ...] = VERSIONS) -> dict:
    folder = folder or OUT / ("dry-run" if dry else "episodes")
    rows = [json.loads(l) for l in (folder / "sweep.jsonl").read_text().splitlines()]
    by_version = collections.defaultdict(list)
    for row in rows:
        by_version[row["version"]].append((row, json.loads((folder / row["file"]).read_text())))
    summary, mismatches = {}, []
    for version in versions:
        eps = by_version.get(version, [])
        s = collections.Counter()
        kinds = collections.Counter()
        kind_episodes = collections.Counter()
        for row, rec in eps:
            s["episodes"] += 1
            s["cost"] += rec.get("cost_usd", 0.0)
            s["backoff_events"] += len(rec.get("backoffs", []))
            s["backoff_episodes"] += bool(rec.get("backoffs"))
            if "crash" in rec:
                s["crash"] += 1
                continue
            v = rec["verdict"]
            s[f"stop:{v['stop_reason']}"] += 1
            if v["stop_reason"] in smc.API_FAILURES:
                s["api_failure"] += 1
            s["reached_expected"] += v["reached"] == v["expected"]
            s["reward_1"] += v["reward"] == 1.0
            s["reward_none"] += v["reward"] is None
            s["attempt_episodes"] += v["attempted_violations"] > 0
            s["attempts"] += v["attempted_violations"]
            s["finish_accepted"] += bool(v["finish_accepted"])
            s["citation_failure_episodes"] += bool({"citation_failure", "required_evidence_missing"} & set(v["critical"]))
            for c in v["critical"]:
                s[f"critical:{c}"] += 1
            k = citation_kinds(rec)
            kinds.update(k)
            kind_episodes.update(set(k))
            if v["reached"] != v["expected"]:
                mismatches.append({"case": row["case"], "variant": row["variant"], "version": version,
                                   "expected": v["expected"], "reached": v["reached"], "stop": v["stop_reason"],
                                   "critical": v["critical"], "calls": v["calls"], "final_clock": v["final_clock"]})
        s["cost"] = round(s["cost"], 4)
        summary[version] = {"counts": dict(s), "citation_problems": dict(kinds), "citation_problem_episodes": dict(kind_episodes)}
    by_type = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        t = by_type[(row["variant"], row["version"])]
        t["n"] += 1
        t["reached"] += row["reached"] == row["expected"]
        t["reward_1"] += row["reward"] == 1.0
    result = {"summary": summary, "mismatches": mismatches,
              "by_type": {f"{k[0]}|{k[1]}": dict(v) for k, v in sorted(by_type.items())},
              "total_cost": round(sum(r["cost_usd"] for r in rows), 4), "episodes": len(rows)}
    (folder / "report.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("select", "run", "report"))
    ap.add_argument("--dry-run-reference", action="store_true")
    ap.add_argument("--gate1a", action="store_true", help="version V2 (the final v2 interface), $5 cap, own folder")
    a = ap.parse_args()
    if a.command == "select":
        for case, variant in select_cases():
            print(case, variant)
    elif a.command == "run":
        run(a.dry_run_reference, a.gate1a)
    elif a.gate1a:
        print(json.dumps(report(a.dry_run_reference, GATE1A["out"] / ("dry-run" if a.dry_run_reference else "episodes"),
                                GATE1A["versions"]), indent=1, default=str))
    else:
        report(a.dry_run_reference)


if __name__ == "__main__":
    main()
