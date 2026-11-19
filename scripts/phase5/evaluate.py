"""The Qwen evaluation harness (Phase 5): a model (vLLM) or a scripted fake on a frozen split, with the training
rollout path, limits and stop classifier (ap_sim.evaluation), one record per episode and a summary.

Held-out splits (test, test_reserve, v6, policy_shift) are refused unless --phase10 is given. No model output on them
is produced or viewed before Phase 10, the final evaluation. Development of the harness ran on dev only. Greedy by
default; --samples k runs k samples per case at T 0.8, top_p 1.0, seeds 0..k-1 (identical across models), for pass@k
and profiling. Phase 10's seeds are keyed by case as well, by the backend its driver passes in
(scripts/gpu/final_evaluation.py).

The command line makes the pipeline check first (scripts/gpu/pipeline_guard.py, added 2026-11-02), so a direct call is
guarded. An infrastructure failure re-runs once, as the pre-registered plan (preregistration_proof/PLAN_V2.md) says in
section 1, under the Claude row's rules (`evaluate`).

    python scripts/phase5/evaluate.py --backend vllm --model PATH --split dev [--samples 8] --out DIR
    HF_HUB_OFFLINE=1 python scripts/phase5/evaluate.py --backend reference --split dev --out DIR    # $0, harness check
    HF_HUB_OFFLINE=1 python scripts/phase5/evaluate.py --backend reckless --split dev --out DIR     # $0, harness check
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/phase2"), str(ROOT / "scripts/gpu")]

from ap_sim.evaluation import metrics, records, rollout   # noqa: E402
from ap_sim.training.loop_guard import LOOP_GUARD   # noqa: E402  (the v2 decision, 2026-11-01: always loop-guard/1)

HELD_OUT = ("test", "test_reserve", "v6", "policy_shift")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SAMPLE_TEMPERATURE, SAMPLE_TOP_P = 0.8, 1.0
INFRASTRUCTURE_FIRST_RUNS = "infrastructure_first_runs.jsonl"
SERVING_FAULTS = ("abort", "error")     # finish reasons of a request the engine gave up on: outside the model's control


def tokenizer(name: str = MODEL, revision: str | None = REVISION, local_only: bool = True):
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema
    return add_response_schema(AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local_only))


def profiling_cases() -> list[str]:
    """Phase 7's frozen stratified 250 RL cases (manifests/v2/rl.json, seed 20261102, recorded before profiling)."""
    return json.loads((ROOT / "manifests" / "v2" / "rl.json").read_text())["profiling"]["case_ids"]


def load_cases(split: str, phase10: bool = False, limit: int | None = None, only: list[str] | None = None) -> dict:
    if split in HELD_OUT and not phase10:
        raise SystemExit(f"{split} is held out until Phase 10 (D-v2-1); pass --phase10 only then")
    if split == "forgetting_l1":
        from ap_sim.forgetting import load
        cases = load()
    else:
        from ap_sim.generator.splits import load_split
        cases = load_split(split)
    if only is not None:
        cases = {c: cases[c] for c in only}
    return dict(list(cases.items())[:limit]) if limit else cases


def episode_line(rec: dict, messages: list, extra: dict) -> str:
    """One episodes.jsonl line. The parser accepts a tool call's JSON nested up to the interpreter's recursion limit
    (about 9,990 levels), and the line's own levels then take the encoder past it (found in an audit of model-written
    text); such an episode's messages are written as a note (the record, its scores, and with --save-ids
    the token ids, which hold the turn, are written in full)."""
    try:
        return json.dumps({**rec, "messages": messages, **extra}, default=str)
    except RecursionError:
        return json.dumps({**rec, "messages": "not written: nested deeper than the JSON encoder allows", **extra},
                          default=str)


def evaluate(cases: dict, backend_name: str, tok, *, samples: int = 0, execution_rule: str = LOOP_GUARD,
             model: str | None = None, revision: str | None = None, budget: int | None = None,
             out: pathlib.Path | None = None, gpu_memory_utilization: float = 0.85, save_ids: bool = False,
             backend=None) -> dict:
    """One evaluation. `backend`: an engine already built (Phase 10's driver), used in place of the one `backend_name` names.

    The infrastructure re-run (the pre-registered plan, section 1), under the rules of the Claude row (claude_row.run_row). An episode
    that ends in infrastructure_failure with no unsafe attempt goes aside, to infrastructure_first_runs.jsonl, and runs once more when the
    others are done, with the same case and sample, so a seeded backend repeats its seeds: a repetition, not a fresh draw. The second run is
    the result. If it fails the same way it counts as a failure (success false, reward 0, or -1 with an unsafe attempt). An unsafe attempt
    decides the case: -1, written at once, never re-run. episodes.jsonl and the records hold exactly one row per (case, sample), and each
    re-run's row carries its first run's record. A turn the engine ended with a SERVING_FAULTS finish reason makes the episode an infrastructure
    failure, under the same rules: it is the serving layer's fault and not the model's, whatever the classifier would call the turn."""
    template = rollout.grpo_template(tok)
    sample_ids = list(range(samples)) if samples else [None]
    kw = {"budget": budget} if budget else {}

    def episode(case, sample):
        return rollout.Episode(case, cases[case], tok, template, execution_rule=execution_rule, sample=sample, **kw)

    episodes = [episode(case, s) for case in cases for s in sample_ids]
    if backend is None and backend_name == "vllm":
        backend = rollout.VLLMBackend(model, revision, gpu_memory_utilization=gpu_memory_utilization,
                                      temperature=SAMPLE_TEMPERATURE if samples else 0.0, top_p=SAMPLE_TOP_P)
    elif backend is None:
        make = rollout.reference_plan if backend_name == "reference" else rollout.reckless_plan
        plans = {case: make(case, f) for case, f in cases.items()}
        backend = rollout.ScriptedBackend(tok, {(c, s): plans[c] for c in cases for s in sample_ids})
    rows, first_runs = [], {}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "episodes.jsonl").write_text("")
        (out / INFRASTRUCTURE_FIRST_RUNS).unlink(missing_ok=True)      # this evaluation's first runs only, never an earlier one's
    started = time.time()

    def record(ep) -> dict:
        rec = records.episode_record(ep.env, ep.classification(), ep.turns,
                                     {"sample": ep.sample, "tool_turns": ep.tool_turns,
                                      "prompt_tokens": ep.n_prompt, "completion_tokens": len(ep.seq) - ep.n_prompt,
                                      # the wall clock from the start of the run, batched with the rest; generation_seconds is its share of each batch
                                      "wall_seconds": round(time.time() - started, 3),
                                      "generation_seconds": round(ep.generation_seconds, 3)})
        fault = next((t["finish_reason"] for t in ep.turns if t.get("finish_reason") in SERVING_FAULTS), None)
        if fault and rec["stop_reason"] != "infrastructure_failure":
            rec = {**rec, "stop_reason": "infrastructure_failure", "infrastructure": f"the engine ended a request: {fault}",
                   "training_reward": -1.0 if rec["violations"]["attempted"] else None}
        return rec

    def extra(ep) -> dict:
        return ({"token_ids": ep.seq, "generated_mask": [int(g) for g in ep.generated], "turns": ep.turns}
                if save_ids else {})                              # RFT rows are built from these

    def keep(rec, ep) -> None:
        rows.append(rec)
        if out:
            with open(out / "episodes.jsonl", "a") as f:
                f.write(episode_line(rec, ep.messages, extra(ep)) + "\n")

    def unsafe_decided(rec) -> dict:
        return {**rec, "success": False, "training_reward": -1.0, "unsafe_attempt": True}

    def done(ep):
        rec = record(ep)
        if rec["stop_reason"] != "infrastructure_failure":
            keep(rec, ep)
        elif rec["violations"]["attempted"]:
            keep(unsafe_decided(rec), ep)
        else:
            first_runs[(ep.case, ep.sample)] = rec
            if out:
                with open(out / INFRASTRUCTURE_FIRST_RUNS, "a") as f:
                    f.write(episode_line(rec, ep.messages, extra(ep)) + "\n")

    def done_again(ep):
        rec = {**record(ep), "infrastructure_rerun": {"first": first_runs[(ep.case, ep.sample)], "result_from": "second"}}
        if rec["violations"]["attempted"]:
            rec = unsafe_decided(rec)
        elif rec["stop_reason"] == "infrastructure_failure":
            rec = {**rec, "success": False, "training_reward": 0.0, "counted_as_failure": True}
        keep(rec, ep)

    timing = rollout.run(episodes, backend, on_done=done)
    if first_runs:
        again = rollout.run([episode(case, sample) for case, sample in first_runs], backend, on_done=done_again)
        timing = {**timing, "rounds": timing["rounds"] + again["rounds"], "wall_seconds": round(time.time() - started, 1)}
    timing["infrastructure_reruns"] = len(first_runs)
    if len(rows) != len(episodes):
        raise RuntimeError(f"{len(rows)} rows for {len(episodes)} episodes")
    by_case = collections.defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r["success"])
    summary = {"backend": backend.info(), "execution_rule": execution_rule, "samples": samples or "greedy",
               "timing": timing, **metrics.summarize(rows),
               "infrastructure": {"reruns": len(first_runs), "counted_as_failure": sum(bool(r.get("counted_as_failure")) for r in rows),
                                  "decided_by_an_unsafe_attempt": sum(bool(r.get("unsafe_attempt")) for r in rows)},
               "constant_strategy": metrics.constant_strategy_columns({r["case"]: r["expected"] for r in rows})}
    if samples:
        summary["pass"] = [metrics.per_case_pass(by_case, k) for k in sorted({1, min(4, samples), samples})]
    if out:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return {"summary": summary, "records": rows}


def main() -> None:
    import pipeline_guard
    pipeline_guard.require()          # refuses to start on a settings mismatch, before anything else (2026-11-02)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=("vllm", "reference", "reckless"), required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--phase10", action="store_true", help="allow the held-out splits (Phase 10 only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--profiling", action="store_true", help="Phase 7: the frozen 250 profiling cases (split rl)")
    ap.add_argument("--cases-file", default=None, help="a JSON list of case ids (e.g. the 120-case subset)")
    ap.add_argument("--save-ids", action="store_true", help="also write each episode's token ids and generated mask")
    ap.add_argument("--samples", type=int, default=0, help="0: greedy; k: k samples per case (seeds 0..k-1)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = tokenizer(local_only=a.backend != "vllm")
    only = profiling_cases() if a.profiling else (json.loads(pathlib.Path(a.cases_file).read_text())
                                                   if a.cases_file else None)
    res = evaluate(load_cases("rl" if a.profiling else a.split, a.phase10, a.limit, only), a.backend, tok,
                   samples=a.samples, save_ids=a.save_ids,
                   model=a.model, revision=a.revision if a.model == MODEL else None,
                   out=pathlib.Path(a.out), gpu_memory_utilization=a.gpu_memory_utilization)
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("episodes", "success", "violation_rate", "stop_reasons", "timing")}, indent=1))


if __name__ == "__main__":
    main()
