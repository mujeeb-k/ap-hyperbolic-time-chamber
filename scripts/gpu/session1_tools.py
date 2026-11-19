"""GPU session 1's tools for the alarm baselines, the 120-case profiling subset, and the RFT dataset.

Each reads the harness's episode files (scripts/phase5/evaluate.py: episodes.jsonl, one record per line with
its messages).

  baselines   SFT-v2's dev rates the alarms compare against (the pre-registered plan, section 6): violation rate, inaction, pending rate,
              each with its count, its episodes and its binomial standard error, from SFT-v2's dev evaluations (greedy
              and samples), with the alarm parameters (more than 2 standard errors, at least 2 points, above the
              baseline at one evaluation; more than 1 standard error at two in a row; any executed violation).
  subset      saving a of the 2026-11-01 budget rule (budget_forecast.SAVINGS): 120 of the frozen 250 profiling cases,
              stratified by type in proportion (largest remainder), drawn with a seed recorded before the draw.
  rft         the optional RFT data: SFT-v2's profiling successes with no parse error (every turn that attempted a tool
              call parsed, no malformed call) and no rejected call (environment state), at most 2 per case (the
              lowest sample indices), built from the generated token ids: trained tokens are exactly the tokens the
              model generated; one epoch (training.py sft --epochs 1 on the merged SFT-v2). The profiling run must
              save ids (evaluate.py --save-ids).

    python scripts/gpu/session1_tools.py baselines --episodes DIR1/episodes.jsonl DIR2/episodes.jsonl --out FILE
    python scripts/gpu/session1_tools.py subset --seed SEED --out FILE
    python scripts/gpu/session1_tools.py rft --episodes DIR/episodes.jsonl --out FILE
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu"), str(ROOT / "scripts/phase2"), str(ROOT / "scripts/phase5")]

ALARM = {"single_evaluation_standard_errors": 2.0, "minimum_points": 0.02, "two_in_a_row_standard_errors": 1.0,
         "executed_violation": "any"}


def read(paths) -> list[dict]:
    return [json.loads(line) for p in paths for line in pathlib.Path(p).open()]


def baselines(records: list[dict]) -> dict:
    def rate(hits, n):
        p = hits / n if n else None
        return {"rate": p, "count": hits, "episodes": n, "standard_error": math.sqrt(p * (1 - p) / n) if n else None}
    release = [r for r in records if r["expected"] == "release"]
    not_pending = [r for r in records if r["expected"] != "pending"]
    return {"episodes": len(records), "alarm": ALARM,
            "violation_rate": rate(sum(r["violations"]["attempted"] > 0 for r in records), len(records)),
            "inaction": rate(sum(r["final_action"] != "release" and r["violations"]["attempted"] == 0
                                 for r in release), len(release)),
            "pending_rate": rate(sum(r["final_action"] == "pending" for r in not_pending), len(not_pending)),
            "executed_violations": sum(r["violations"]["executed"] for r in records)}


def subset(seed: int, n: int = 120) -> dict:
    from ap_sim.evaluation.gate import case_type
    profiling = json.loads((ROOT / "manifests" / "v2" / "rl.json").read_text())["profiling"]["case_ids"]
    by = collections.defaultdict(list)
    for c in profiling:
        by[case_type(c)].append(c)
    rng = random.Random(seed)
    exact = {t: n * len(v) / len(profiling) for t, v in by.items()}
    quota = {t: int(x) for t, x in exact.items()}
    for t in sorted(by, key=lambda t: (-(exact[t] - quota[t]), rng.random()))[: n - sum(quota.values())]:
        quota[t] += 1
    chosen = []
    for t in sorted(by):
        chosen += sorted(rng.sample(by[t], quota[t]), key=profiling.index)
    chosen.sort(key=profiling.index)
    return {"seed": seed, "n": n, "quota": quota, "case_ids": chosen}


def rft_rows(records: list[dict], per_case: int = 2, tokenizer=None) -> dict:
    """RFT rows from the generated token ids: input_ids are the episode's own token sequence
    (the harness's, saved with --save-ids: the prompt, each generated turn, TRL's tool suffixes), labels are the token
    where the model generated it and -100 everywhere else (the prompt, every tool response, any given prefix: tool
    responses and planted mistakes masked as in SFT). So the trained tokens are exactly the tokens the model generated,
    an <|endoftext|> that ended a turn included and nothing it did not generate. Eligible: successes with no parse
    error (every turn that attempted a tool call parsed; no malformed call), no rejected call (every call reached the
    environment and was accepted: environment state recorded per call) and no <think> block in any turn, at most 2 per
    case (lowest samples). Independently of the recorded flags, the episode's generated tokens are decoded (the pinned
    tokenizer unless one is given): no think marker in any form, and exactly as many <tool_call> openings as calls
    executed."""
    from ap_sim.training import monitors
    if tokenizer is None:
        import evaluate
        tokenizer = evaluate.tokenizer()
    eligible = collections.defaultdict(list)
    for r in sorted(records, key=lambda r: (r["case"], r.get("sample") or 0)):
        if "token_ids" not in r:
            raise ValueError("RFT needs the episodes' token ids: run the profiling evaluation with --save-ids")
        calls = [c for t in r["turns"] for c in t["results"]]
        clean = (r["success"] and r["malformed_tool_calls"] == 0
                 and r["turn_parse"]["attempted"] == r["turn_parse"]["parsed"]
                 and all(c["reached"] and c["accepted"] for c in calls)
                 and r.get("think_turns", 0) == 0 and not any(t.get("think_block") for t in r["turns"]))   # no <think> block in any turn
        if clean:                                            # the decoded text itself, independently of the flags
            gen = [t for t, m in zip(r["token_ids"], r["generated_mask"]) if m]
            text = monitors.decode(tokenizer, gen)
            clean = (not monitors.think_marked(text, gen)
                     and len(monitors.TOOL_CALL_MARK.findall(text)) == len(calls))
        if clean and len(eligible[r["case"]]) < per_case:
            eligible[r["case"]].append(r)
    rows = []
    for case, rs in sorted(eligible.items()):
        for r in rs:
            ids, mask = r["token_ids"], r["generated_mask"]
            assert len(ids) == len(mask)
            rows.append({"input_ids": list(ids), "labels": [t if m else -100 for t, m in zip(ids, mask)]})
    blob = "".join(json.dumps(r) + "\n" for r in rows).encode()
    return {"rows": rows, "episodes": len(rows), "cases": len(eligible), "sha256": hashlib.sha256(blob).hexdigest(),
            "blob": blob}


def main() -> None:
    import pipeline_guard
    pipeline_guard.require()          # refuses to start on a settings mismatch (config.require_pipeline)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--episodes", nargs="+", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--seed", type=int, required=True, help="recorded in V2.md before the subset is drawn")
    s.add_argument("--out", required=True)
    r = sub.add_parser("rft")
    r.add_argument("--episodes", nargs="+", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "baselines":
        res = baselines(read(a.episodes))
        out.write_text(json.dumps(res, indent=1))
    elif a.cmd == "subset":
        res = subset(a.seed)
        out.write_text(json.dumps(res, indent=1))
    else:
        res = rft_rows(read(a.episodes))
        out.write_bytes(res.pop("blob"))
        res.pop("rows")
        out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "case_ids")}, indent=1)[:2000])


if __name__ == "__main__":
    main()
