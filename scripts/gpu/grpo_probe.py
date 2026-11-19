"""GRPO's real configuration, measured before any budget cap was confirmed (added after the measurement block, 2026-11-02).

The measurement block (benchmark.py) replayed demonstrations through a vLLM engine set up for evaluation (85% of the GPU), and timed a plain
forward-backward. GRPO runs differently: TRL's colocated engine (35%, max_num_batched_tokens 4096, 64 sequences), its
fused log-probabilities, and three passes per rollout (with gradients, the old log-probabilities, the KL reference). This
script measures that path, in one process per configuration, on the base model:

  config  --utilization U [--sleep]   TRL's real trainer and engine. Peak GPU memory and time of the loss passes at
          micro-batches 1, 2, 4, 8 until one does not fit, on the longest rows and on rows spaced through the lengths;
          then a teacher-forced replay of 64 demonstrations through that engine. With --sleep, each round does what TRL's
          sleep mode does: the weights pushed back, the cache woken, generation, sleep level 2.
  step    --utilization U [--sleep] --micro-batch M   One real GRPO step, 8 cases x 8 rollouts with gradient accumulation
          keeping 64 rollouts per update, through the real hooks on the first pilot cases. Timings, token counts and memory
          are kept; rollouts, rewards and any model text are not (no model output on any case is viewed).
  eval    Evaluation's engine (85%) at batches 16, 32 and 100.
  all     All of them, each in its own process, stopping at --minutes, then one more step without sleep mode at the largest
          share for vLLM that the measured update leaves room for. Results accumulate in OUT/probe.json.

Probe only: the micro-batch and sleep mode are overridden in this process. training.py and its recorded settings are unchanged.

    python scripts/gpu/grpo_probe.py all --data PERSIST/sft/train.jsonl --out DIR
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu"), str(ROOT / "scripts/phase2")]

import gpu_cleanup   # noqa: E402

MICRO_BATCHES = (1, 2, 4, 8)
ROLLOUTS_PER_UPDATE = 64
REPLAY_ROWS = 64
EVAL_BATCHES = (16, 32, 100)
REPLAY_MINUTES, STEP_MINUTES, EVAL_MINUTES = 10, 12, 6
UTILIZATION_PLAIN, UTILIZATION_SLEEP = 0.35, 0.80
GPU_GB = 79.25


class GpuMemory:
    """The whole GPU's memory.used, sampled once a second: vLLM's engine is not in torch's own counters."""

    def __init__(self):
        self.peak_mib, self._stop = 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            return int(out.split()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._read())
            self._stop.wait(1.0)

    def __enter__(self):
        self.peak_mib = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)

    def now_gb(self) -> float:
        return round(self._read() / 1024, 2)


def configure(micro_batch: int, sleep: bool) -> None:
    """Probe only: this process's GRPO settings, with 64 rollouts per update kept by gradient accumulation."""
    import training
    steps = ROLLOUTS_PER_UPDATE // micro_batch
    training.GRPO_SETTINGS = {**training.GRPO_SETTINGS, "per_device_train_batch_size": micro_batch,
                              "gradient_accumulation_steps": steps, "steps_per_generation": steps}
    original = training.grpo_args
    training.grpo_args = lambda *a, **k: {**original(*a, **k), "vllm_enable_sleep_mode": sleep}


def split_row(tok, row: dict) -> tuple[list[int], list[int]]:
    """(prompt, completion) of a demonstration: the completion starts at the first assistant turn, as in a rollout, and
    holds the tool responses."""
    import benchmark
    start = len(benchmark.turn_plan(tok, row)[0][0])
    return row["input_ids"][:start], row["input_ids"][start:]


def padded(tok, rows: list[dict]) -> tuple[list[list[int]], list[list[int]], int]:
    """As TRL lays a micro-batch out: prompts left-padded, completions right-padded. (input_ids, attention_mask, logits_to_keep)."""
    pad = tok.pad_token_id
    parts = [split_row(tok, r) for r in rows]
    p_max, c_max = max(len(p) for p, _ in parts), max(len(c) for _, c in parts)
    ids = [[pad] * (p_max - len(p)) + p + c + [pad] * (c_max - len(c)) for p, c in parts]
    mask = [[0] * (p_max - len(p)) + [1] * (len(p) + len(c)) + [0] * (c_max - len(c)) for p, c in parts]
    return ids, mask, c_max


def loss_pass(trainer, tok, rows: list[dict], micro_batch: int) -> dict:
    """The three passes per rollout, on `rows`, through TRL's own log-probability function."""
    import torch
    model = trainer.model
    device = next(model.parameters()).device
    ids, mask, keep = padded(tok, rows)
    ids, mask = torch.tensor(ids, device=device), torch.tensor(mask, device=device)
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    logps, _, _ = trainer._get_per_token_logps_and_entropies(model, ids, mask, keep, batch_size=micro_batch, compute_entropy=True)
    scored = mask[:, -keep:]
    loss = -(logps * scored).sum() / scored.sum()
    loss.backward()
    model.zero_grad(set_to_none=True)
    sync()
    t1 = time.perf_counter()
    with torch.no_grad():
        trainer._get_per_token_logps_and_entropies(model, ids, mask, keep, batch_size=micro_batch)
    sync()
    t2 = time.perf_counter()
    with torch.no_grad(), model.disable_adapter():
        trainer._get_per_token_logps_and_entropies(model, ids, mask, keep, batch_size=micro_batch)
    sync()
    t3 = time.perf_counter()
    n = len(rows)
    out = {"rows": n, "tokens_max": ids.size(1), "seconds_per_rollout_gradient": round((t1 - t0) / n, 3),
           "seconds_per_rollout_old_logps": round((t2 - t1) / n, 3), "seconds_per_rollout_reference": round((t3 - t2) / n, 3),
           "seconds_per_rollout": round((t3 - t0) / n, 3)}
    if cuda:
        out["peak_torch_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return out


def spaced(data: list[dict], n: int) -> list[dict]:
    """n rows evenly spaced through the lengths."""
    order = sorted(data, key=lambda r: len(r["input_ids"]))
    return [order[int((i + 0.5) * len(order) / n)] for i in range(n)]


def longest(data: list[dict], n: int) -> list[dict]:
    return sorted(data, key=lambda r: len(r["input_ids"]), reverse=True)[:n]


def memory_ladder(trainer, tok, data: list[dict], memory: GpuMemory, batches=MICRO_BATCHES) -> dict:
    """Micro-batches smallest first, until one does not fit; the longest rows (the worst case: a step's micro-batch can
    hold eight rollouts of one long case) and rows spaced through the lengths."""
    import torch
    model = trainer.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    loss_pass(trainer, tok, spaced(data, 1), 1)      # warm-up, not reported
    out = {}
    for kind, pick in (("longest", longest), ("spaced", spaced)):
        out[kind] = {}
        for m in batches:
            try:
                with memory:
                    res = loss_pass(trainer, tok, pick(data, m), m)
                out[kind][str(m)] = {"fits": True, **res, "peak_gpu_gb": round(memory.peak_mib / 1024, 2)}
            except torch.OutOfMemoryError as exc:
                out[kind][str(m)] = {"fits": False, "error": str(exc)[:200], "peak_gpu_gb": round(memory.peak_mib / 1024, 2)}
                gc.collect()
                torch.cuda.empty_cache()
                break
    fitting = [int(m) for m, r in out["longest"].items() if r["fits"]]
    out["largest_micro_batch_fitting_longest"] = max(fitting) if fitting else 0
    fitting = [int(m) for m, r in out["spaced"].items() if r["fits"]]
    out["largest_micro_batch_fitting_spaced"] = max(fitting) if fitting else 0
    return out


def replay(generate, plans: list[list[tuple[list[int], int]]], deadline: float, before=None, after=None, clock=time.perf_counter) -> dict:
    """Teacher-forced lockstep rounds until the plans end or the deadline. before/after run around each round's generation
    (the sleep cycle). A run cut short says so, and how much of the tokens it reached."""
    total = sum(n for plan in plans for _, n in plan)
    done, rounds, around = 0, 0, 0.0
    t0 = clock()
    finished = True
    while True:
        active = [p[rounds] for p in plans if rounds < len(p)]
        if not active:
            break
        if clock() > deadline:
            finished = False
            break
        t1 = clock()
        if before:
            before()
        t2 = clock()
        generate(active)
        t3 = clock()
        if after:
            after()
        around += (t2 - t1) + (clock() - t3)
        done += sum(n for _, n in active)
        rounds += 1
    wall = clock() - t0
    out = {"episodes": len(plans), "finished": finished, "rounds": rounds, "wall_seconds": round(wall, 2),
           "seconds_around_generation": round(around, 2), "generated_tokens": done, "generated_tokens_total": total,
           "generated_tokens_per_second": round(done / wall, 1) if wall else None}
    if finished and wall:
        out["episodes_per_hour"] = round(len(plans) / wall * 3600, 1)
    else:
        out["episodes_per_hour_extrapolated_by_tokens"] = round(len(plans) * (done / total) / wall * 3600, 1) if wall and total else None
    return out


def replay_rows(data: list[dict], n: int) -> list[dict]:
    import benchmark
    return [data[i] for i in benchmark.allocate(data, benchmark.row_types(len(data)), benchmark.rl_mix(), (n,))["batches"][n]]


def run_config(a) -> dict:
    import benchmark
    import training
    configure(8, a.sleep)
    tok = training.tokenizer()
    data = [json.loads(line) for line in open(a.data)]
    memory = GpuMemory()
    trainer = training.build_grpo(pathlib.Path(tempfile.mkdtemp()), model=training.MODEL, role="pilot", lr=5e-6, steps=1, cases=8,
                                  vllm_gpu_memory_utilization=a.utilization)
    trainer.model.to("cuda")
    out = {"utilization": a.utilization, "sleep": a.sleep, "gpu_gb_after_build": memory.now_gb(),
           "ladder": memory_ladder(trainer, tok, data, memory)}
    print(json.dumps(out["ladder"]), flush=True)
    import torch
    torch.cuda.empty_cache()
    vg = trainer.vllm_generation
    llm = vg.llm
    engine = types.SimpleNamespace(llm=llm)
    before = after = None
    if a.sleep:
        def before():
            if vg._llm_weights_sleeping:
                vg.sync_weights()
            llm.wake_up(tags=["kv_cache"])

        def after():
            llm.sleep(level=2)
            vg._llm_weights_sleeping = True
    plans = [benchmark.turn_plan(tok, r) for r in replay_rows(data, REPLAY_ROWS)]
    with memory:
        res = replay(lambda turns: benchmark.VLLMEngine.generate_turns(engine, turns), plans,
                     time.perf_counter() + REPLAY_MINUTES * 60, before, after)
    out["replay"] = {**res, "peak_gpu_gb": round(memory.peak_mib / 1024, 2)}
    return out


def token_counts(inputs: dict) -> dict:
    lengths = inputs["completion_mask"].sum(dim=1).tolist() if "completion_mask" in inputs else []
    return {"completion_tokens_scored_mean": round(statistics.fmean(lengths), 1) if lengths else None,
            "completion_tokens_scored_max": max(lengths) if lengths else None,
            "rollouts": inputs["completion_ids"].size(0) if "completion_ids" in inputs else None,
            "completion_width": inputs["completion_ids"].size(1) if "completion_ids" in inputs else None,
            "prompt_width": inputs["prompt_ids"].size(1) if "prompt_ids" in inputs else None}


def on_policy_check(args, timers: dict, global_step: int, micro_batch: int) -> dict:
    """Exactly one optimizer update per generation of 64 rollouts, with no rollout reuse: the settings, and what the step did."""
    steps = ROLLOUTS_PER_UPDATE // micro_batch
    checks = {"micro_batch": args.per_device_train_batch_size == micro_batch,
              "generation_batch": args.per_device_train_batch_size * args.steps_per_generation == ROLLOUTS_PER_UPDATE,
              "accumulation_equals_steps_per_generation": args.gradient_accumulation_steps == args.steps_per_generation == steps,
              "no_reuse": args.num_iterations == 1, "one_generation": timers["generations"] == 1,
              "micro_steps": timers["micro_steps"] == steps, "one_update": global_step == 1,
              "rollouts_scored": (timers.get("tokens") or {}).get("rollouts") == ROLLOUTS_PER_UPDATE}
    return {**checks, "ok": all(checks.values()), "generations": timers["generations"], "micro_steps_run": timers["micro_steps"]}


def run_step(a) -> dict:
    import torch
    import training
    configure(a.micro_batch, a.sleep)
    memory = GpuMemory()
    out = {"utilization": a.utilization, "sleep": a.sleep, "micro_batch": a.micro_batch,
           "accumulation": ROLLOUTS_PER_UPDATE // a.micro_batch, "rollouts_per_update": ROLLOUTS_PER_UPDATE}
    trainer = training.build_grpo(pathlib.Path(tempfile.mkdtemp()), model=training.MODEL, role="pilot", lr=5e-6, steps=1, cases=8,
                                  vllm_gpu_memory_utilization=a.utilization)
    timers = {"generation_seconds": 0.0, "generation_calls": 0, "scoring_prepare_seconds": 0.0, "generations": 0, "micro_steps": 0}
    single, score, micro_step = trainer._generate_single_turn, trainer._generate_and_score_completions, trainer.training_step

    def counted_micro_step(*args, **kw):
        timers["micro_steps"] += 1
        return micro_step(*args, **kw)

    def timed_single(*args, **kw):
        t0 = time.perf_counter()
        try:
            return single(*args, **kw)
        finally:
            timers["generation_seconds"] += time.perf_counter() - t0
            timers["generation_calls"] += 1

    def timed_score(*args, **kw):
        t0 = time.perf_counter()
        result = score(*args, **kw)
        timers["scoring_prepare_seconds"] += time.perf_counter() - t0
        timers["generations"] += 1
        timers["tokens"] = token_counts(result)
        return result
    trainer._generate_single_turn, trainer._generate_and_score_completions = timed_single, timed_score
    trainer.training_step = counted_micro_step
    t0 = time.perf_counter()
    try:
        with memory:
            trainer.train()
        out["ok"] = trainer.state.global_step == 1
    except torch.OutOfMemoryError as exc:
        out.update(ok=False, error=f"out of memory: {str(exc)[:200]}")
    except Exception as exc:      # recorded: a failing configuration is a result here
        out.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
    wall = time.perf_counter() - t0
    prepare = timers["scoring_prepare_seconds"]
    out.update({"wall_seconds": round(wall, 1), "generation_seconds": round(timers["generation_seconds"], 1),
                "generation_calls": timers["generation_calls"],
                "old_and_reference_passes_seconds": round(prepare - timers["generation_seconds"], 1),
                "update_seconds": round(wall - prepare, 1), "tokens": timers.get("tokens"),
                "peak_gpu_gb": round(memory.peak_mib / 1024, 2),
                "peak_torch_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)})
    out["on_policy"] = on_policy_check(trainer.args, timers, trainer.state.global_step, a.micro_batch)
    return out


def run_eval(a) -> dict:
    import benchmark
    import training
    tok = training.tokenizer()
    data = [json.loads(line) for line in open(a.data)]
    chosen = benchmark.allocate(data, benchmark.row_types(len(data)), benchmark.rl_mix(), EVAL_BATCHES)["batches"]
    engine = benchmark.VLLMEngine()
    memory = GpuMemory()
    out = {}
    for b in EVAL_BATCHES:
        plans = [benchmark.turn_plan(tok, data[i]) for i in chosen[b]]
        with memory:
            res = replay(engine.generate_turns, plans, time.perf_counter() + EVAL_MINUTES * 60)
        out[str(b)] = {**res, "peak_gpu_gb": round(memory.peak_mib / 1024, 2)}
        print(json.dumps(out[str(b)]), flush=True)
    return out


def child(block: str, a, timeout_minutes: float, *extra: str) -> dict:
    """A block in its own process, so a colocated engine's memory is really gone afterwards and a failure stays in its block."""
    cmd = [sys.executable, __file__, block, "--data", a.data, "--out", a.out, *extra]
    t0 = time.perf_counter()
    record = gpu_cleanup.Record(str(pathlib.Path(a.out).parent / "gpu_cleanup.jsonl"))
    code = gpu_cleanup.run_group(cmd, record, timeout_minutes * 60)     # its vLLM engine is ended with it
    if not gpu_cleanup.free_gpu(record)["free"]:
        sys.exit(gpu_cleanup.NOT_FREED)
    name = pathlib.Path(a.out) / (block + "".join(x.replace("--", "-") for x in extra).replace(".", "") + ".json")
    result = json.loads(name.read_text()) if name.exists() else {}
    return {"exit": code, "seconds": round(time.perf_counter() - t0, 1), **result}


def largest_fit(config: dict) -> int:
    ladder = config.get("ladder") or {}
    return ladder.get("largest_micro_batch_fitting_longest") or ladder.get("largest_micro_batch_fitting_spaced") or 0


def raised_utilization(config: dict, total_gb: float = GPU_GB, margin_gb: float = 4.0) -> float | None:
    """vLLM's share if the update's own peak at the fitting micro-batch, on the longest rows, plus a margin, is all that
    must stay free; None unless that is clearly above the 35% tested."""
    ladder = config.get("ladder") or {}
    m = ladder.get("largest_micro_batch_fitting_longest")
    peak = (ladder.get("longest", {}).get(str(m)) or {}).get("peak_torch_gb") if m else None
    if peak is None:
        return None
    share = round((total_gb - peak - margin_gb) / total_gb, 2)
    return share if share >= UTILIZATION_PLAIN + 0.1 else None


def run_all(a) -> dict:
    out_dir = pathlib.Path(a.out)
    results, deadline = {}, time.time() + a.minutes * 60

    def save():
        (out_dir / "probe.json").write_text(json.dumps(results, indent=1, default=str))

    def left() -> float:
        return (deadline - time.time()) / 60
    for name, util, sleep in (("plain", UTILIZATION_PLAIN, False), ("sleep", UTILIZATION_SLEEP, True)):
        flags = ["--utilization", str(util), *(["--sleep"] if sleep else [])]
        if left() < 5:
            results[f"{name}_config"] = {"skipped": "out of time"}
            continue
        results[f"{name}_config"] = child("config", a, min(left(), 25), *flags)
        save()
        m = largest_fit(results[f"{name}_config"])
        if not m or left() < 5:
            results[f"{name}_step"] = {"skipped": "no micro-batch fits" if not m else "out of time"}
            continue
        results[f"{name}_step"] = child("step", a, min(left(), STEP_MINUTES), *flags, "--micro-batch", str(m))
        save()
    if left() >= 3:
        results["eval"] = child("eval", a, min(left(), 3 * EVAL_MINUTES + 3))
    else:
        results["eval"] = {"skipped": "out of time"}
    save()
    # Without sleep mode: all the memory the update leaves free given to vLLM, at the micro-batch that fits.
    raised = raised_utilization(results.get("plain_config", {}))
    if raised and left() >= 5:
        m = largest_fit(results["plain_config"])
        results["raised_step"] = child("step", a, min(left(), STEP_MINUTES), "--utilization", str(raised), "--micro-batch", str(m))
    else:
        results["raised_step"] = {"skipped": "no room above 35%" if not raised else "out of time"}
    save()
    return results


def main() -> None:
    import pipeline_guard
    pipeline_guard.require()          # refuses to start on a settings mismatch (config.require_pipeline)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("block", choices=("all", "config", "step", "eval"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--utilization", type=float, default=UTILIZATION_PLAIN)
    ap.add_argument("--sleep", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=70)
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.block == "all":
        print(json.dumps(run_all(a), indent=1, default=str))
        return
    extra = {"config": ["--utilization", str(a.utilization), *(["--sleep"] if a.sleep else [])],
             "step": ["--utilization", str(a.utilization), *(["--sleep"] if a.sleep else []), "--micro-batch", str(a.micro_batch)],
             "eval": []}[a.block]
    result = {"config": run_config, "step": run_step, "eval": run_eval}[a.block](a)
    name = out_dir / (a.block + "".join(x.replace("--", "-") for x in extra).replace(".", "") + ".json")
    name.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
