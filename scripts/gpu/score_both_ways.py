"""The selected checkpoint scored on dev both ways, bf16-merged and runtime LoRA.

Added on 2026-11-03. A descriptive measurement of how much the bf16 merge costs, and the first rate measured for runtime LoRA. A probe, not the frozen harness: it uses the harness's episodes and
scoring and only swaps the backend from here, changing no frozen file. Nothing about selection or the record depends on it. Each way
and sampling is its own process, because a vLLM engine does not give its memory back.

    python scripts/gpu/score_both_ways.py score --way lora --samples 8 --base $PERSIST/sft/merged --merged DIR --adapter CKPT --out DIR
    python scripts/gpu/score_both_ways.py compare --dir DIR --out gap.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu"), str(ROOT / "scripts/phase2"), str(ROOT / "scripts/phase5")]

LORA_NAME = "adapter"
WAYS, SAMPLINGS = ("merged", "lora"), (0, 8)


def lora_backend(adapter: str):
    """The harness's vLLM backend with the adapter applied at runtime: the same engine settings and sampling, plus LoRA."""
    from ap_sim.evaluation import rollout
    import training

    class WithLoRA:
        def __init__(self, llm, request):
            self.llm, self.request = llm, request

        def generate(self, *args, **kwargs):
            return self.llm.generate(*args, lora_request=self.request, **kwargs)

    class RuntimeLoRA(rollout.VLLMBackend):
        name = "vllm-runtime-lora"

        def __init__(self, model, revision, max_model_len=rollout.PIPELINE_MAX_CONTEXT_LENGTH, gpu_memory_utilization=0.85, temperature=0.0, top_p=1.0):
            from vllm import LLM
            from vllm.lora.request import LoRARequest
            self.temperature, self.top_p = temperature, top_p
            llm = LLM(model=model, revision=revision, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                      enable_prefix_caching=True, seed=0, enable_lora=True, max_lora_rank=training.LORA_RANK, max_loras=1)
            self.llm = WithLoRA(llm, LoRARequest(LORA_NAME, 1, adapter))

        def info(self):
            return {**super().info(), "adapter": adapter}
    return RuntimeLoRA


def score(way: str, samples: int, base: str, merged: str, adapter: str, out: pathlib.Path) -> dict:
    import evaluate
    from ap_sim.evaluation import rollout
    if way == "lora":
        rollout.VLLMBackend = lora_backend(adapter)      # this process only; the frozen file is unchanged
    tok = evaluate.tokenizer(local_only=False)
    cases = evaluate.load_cases("dev", False, None, None)
    return evaluate.evaluate(cases, "vllm", tok, samples=samples, model=base if way == "lora" else merged, revision=None, out=out)["summary"]


def compare(directory: pathlib.Path) -> dict:
    """Success and the safety counts, each way, and the difference per case, paired on the same cases and seeds."""
    res: dict = {"descriptive_only": True}
    for samples in SAMPLINGS:
        rows = {}
        for way in WAYS:
            with open(directory / f"{way}-{samples}" / "episodes.jsonl") as f:
                rows[way] = {(r["case"], r.get("sample")): r for r in map(json.loads, f)}
        if rows["merged"].keys() != rows["lora"].keys():
            raise SystemExit(f"the two ways differ in episodes at samples {samples}")
        n = len(rows["merged"])
        hits = {w: sum(bool(r["success"]) for r in rows[w].values()) for w in WAYS}
        rates = {w: json.loads((directory / f"{w}-{samples}" / "summary.json").read_text())["timing"].get("episodes_per_hour") for w in WAYS}
        res["greedy" if not samples else f"samples_{samples}"] = {
            "episodes": n, "success": {w: hits[w] / n for w in WAYS}, "lora_minus_merged": (hits["lora"] - hits["merged"]) / n,
            "disagreements": sum(bool(rows["merged"][k]["success"]) != bool(rows["lora"][k]["success"]) for k in rows["merged"]),
            "attempted_violations": {w: sum(r["violations"]["attempted"] > 0 for r in rows[w].values()) for w in WAYS},
            "executed_violations": {w: sum(r["violations"]["executed"] for r in rows[w].values()) for w in WAYS},
            "episodes_per_hour": rates}
    return res


def main() -> None:
    import pipeline_guard
    pipeline_guard.require()          # refuses to start on a settings mismatch (config.require_pipeline)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--way", choices=WAYS, required=True)
    s.add_argument("--samples", type=int, choices=SAMPLINGS, required=True)
    s.add_argument("--base", required=True, help="the merged SFT-v2")
    s.add_argument("--merged", required=True, help="the checkpoint merged in bf16")
    s.add_argument("--adapter", required=True, help="the checkpoint's adapter")
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dir", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "score":
        print(json.dumps(score(a.way, a.samples, a.base, a.merged, a.adapter, pathlib.Path(a.out)), indent=1, default=str)[:2000])
    else:
        res = compare(pathlib.Path(a.dir))
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
