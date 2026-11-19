"""GPU session 1's measurement block, then the budget forecast re-run with the 2026-11-01 decision rule applied.

The block takes about 30 minutes and runs before any SFT training (set on 2026-11-01). The decision rule is
scripts/budget_forecast.py `decide`.

(a) Update time per sequence on demonstration-length sequences: the frozen SFT rows (sft-demos/v2-1; mean context
    16,560, longest 22,007 tokens) through the model with LoRA rank 32 on all linear layers, bf16, gradient
    checkpointing: one forward-backward with the loss on the trained tokens, plus the KL reference pass (one forward
    with the adapter disabled, no gradient), timed per sequence after warm-up; peak memory recorded. The mean comes from
    a sample stratified as in `allocate`. The longest row runs last and is reported on its own, so it reaches the peak
    memory without pulling up the mean.
(b) Generation throughput on long multi-turn prompts, with vLLM prefix caching on and realistic batch sizes (64, a GRPO
    step; and a larger evaluation batch): the demonstrations replayed turn by turn (teacher forcing). Round k sends each
    active demonstration's real context before its assistant turn k as the prompt and generates exactly that turn's
    length (ignore_eos), so the prefill and decode work, and the prefix reuse across turns, are a real episode's.
    Reported: episodes per hour per batch size, and a 64-episode batch's wall time (the longest episode bounds a step).
    Rows (set on 2026-11-02; see `allocate`): each batch has the RL split's case mix, includes the longest demonstrations,
    and shares no row with another batch or with (c). 64 + 256 rows would not fit in 300, so the larger batch is 200.
(c) Single-episode latency (a pre-data decision of 2026-11-02): demonstrations replayed the same
    way, one at a time with nothing else in the engine: the shortest, the longest and the three quartile rows by length,
    so the row choice uses token counts only. These rows are in no batch, so their own turns are not in the prefix cache;
    the very longest rows go to the batches, so "longest" here is the longest left. One further row is replayed first as a
    warm-up and not reported. Generation only: no environment, parsing or grading time.
Every row choice uses case type and token length only, never a model's output. measured.json also holds `composition`:
each batch's and sample's case mix and episode-length distribution, next to the SFT data's and the RL split's mix
(--describe prints it without a model or GPU).
Writes measured.json and decision.json into --out.

    python scripts/gpu/benchmark.py --data /workspace/persist/sft/train.jsonl --out /workspace/persist/benchmark
    python scripts/gpu/benchmark.py --tiny --data ... --out ...       # CPU smoke test (tiny model, fake engine)
    python scripts/gpu/benchmark.py --describe --data ... --out ...   # the rows each part uses; no model, no GPU
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu"), str(ROOT / "scripts/phase4")]

import training   # noqa: E402

ASSISTANT_HEADER = "<|im_start|>assistant\n"
BATCH_SIZES = (64, 200)
TINY_BATCH_SIZES = (4, 8)
MANIFESTS = ROOT / "manifests" / "v2"


def rows(path: pathlib.Path, n: int | None = None, longest_first: bool = False) -> list[dict]:
    data = [json.loads(line) for line in path.open()]
    if longest_first:
        data.sort(key=lambda r: -len(r["input_ids"]))
    return data[:n] if n else data


def row_types(n: int) -> list[str]:
    """The case type of each row: the frozen rows follow manifests/v2/sft.json's order (a test checks it on the sidecar)."""
    entries = json.loads((MANIFESTS / "sft.json").read_text())["entries"]
    if len(entries) != n:
        raise ValueError(f"{n} rows, but {len(entries)} SFT cases: not the frozen dataset")
    return [e["group"] for e in entries]


def rl_mix() -> dict[str, int]:
    """Cases per type in the RL split (its manifest's type labels only)."""
    counts: dict[str, int] = {}
    for e in json.loads((MANIFESTS / "rl.json").read_text())["entries"]:
        counts[e["group"]] = counts.get(e["group"], 0) + 1
    return dict(sorted(counts.items()))


def quota(mix: dict[str, int], n: int) -> dict[str, int]:
    """n divided across the types in the mix's proportions, by largest remainder (ties in type order)."""
    total = sum(mix.values())
    out = {t: n * c // total for t, c in mix.items()}
    by_remainder = sorted(mix, key=lambda t: (-(n * mix[t] % total), t))
    for t in by_remainder[:n - sum(out.values())]:
        out[t] += 1
    return out


def allocate(data: list[dict], types: list[str], mix: dict[str, int], batch_sizes=BATCH_SIZES) -> dict:
    """Which rows each generation measurement uses: disjoint, so no batch starts with another's turns in the prefix cache.

    Batch k takes the k-th longest row of each type, so the batches hold the longest demonstrations. The single-episode
    rows are then the warm-up and quantile rows by length of what remains. The rest of each type is dealt in length order
    to the batches and to `unused`, each receiving its share evenly, so each batch's lengths follow the type's."""
    length = [len(r["input_ids"]) for r in data]
    by_type = {t: sorted((i for i, x in enumerate(types) if x == t), key=lambda i: (-length[i], i)) for t in mix}
    needs = {b: quota(mix, b) for b in batch_sizes}
    batches: dict[int, list[int]] = {b: [] for b in batch_sizes}
    for k, b in enumerate(batch_sizes):
        for t in mix:
            if needs[b][t]:
                batches[b].append(by_type[t][k])
    taken = {i for rows in batches.values() for i in rows}
    remaining = sorted((i for i in range(len(data)) if i not in taken), key=lambda i: (length[i], i))
    at = lambda q: remaining[round(q * (len(remaining) - 1))]                                  # noqa: E731
    warmup = at(SINGLE_EPISODE_WARMUP_QUANTILE)
    single = [at(q) for q in SINGLE_EPISODE_QUANTILES]
    if warmup in single or len(set(single)) != len(single):
        raise ValueError("too few rows for the single-episode measurement")
    taken |= {warmup, *single}
    unused: list[int] = []
    for t in mix:
        pool = [i for i in by_type[t] if i not in taken]
        want = {b: needs[b][t] - sum(types[i] == t for i in batches[b]) for b in batch_sizes}
        spare = len(pool) - sum(want.values())
        if spare < 0:
            raise ValueError(f"not enough {t} rows for batches of {batch_sizes} with no row shared")
        buckets = [*((want[b], batches[b]) for b in batch_sizes), (spare, unused)]
        got = [0] * len(buckets)
        for p, i in enumerate(pool):
            k = max(range(len(buckets)), key=lambda j: buckets[j][0] * (p + 1) / len(pool) - got[j])
            buckets[k][1].append(i)
            got[k] += 1
    return {"batches": {b: sorted(rows) for b, rows in batches.items()}, "warmup": warmup, "single": single,
            "unused": sorted(unused)}


def update_sample(data: list[dict], types: list[str], mix: dict[str, int], n: int) -> dict:
    """(a)'s rows: n stratified as `quota` says, evenly spaced through each type's lengths (the mean is meant to be typical);
    the longest row separately; two warm-up rows, the shortest not otherwise used."""
    length = [len(r["input_ids"]) for r in data]
    want = quota(mix, n)
    sample = []
    for t in mix:
        ranked = sorted((i for i, x in enumerate(types) if x == t), key=lambda i: (length[i], i))
        sample += [ranked[(2 * k + 1) * len(ranked) // (2 * want[t])] for k in range(want[t])]
    longest = max(range(len(data)), key=lambda i: (length[i], -i))
    used = {*sample, longest}
    warmup = sorted((i for i in range(len(data)) if i not in used), key=lambda i: (length[i], i))[:2]
    return {"sample": sorted(sample), "longest": longest, "warmup": warmup}


def describe_rows(tok, data: list[dict], types: list[str], indices: list[int]) -> dict:
    """One set of rows: its case mix, and its episode lengths in assistant turns, context tokens and generated tokens."""
    from sft_description import distribution
    plans = [turn_plan(tok, data[i]) for i in indices]
    mix: dict[str, int] = {}
    for i in indices:
        mix[types[i]] = mix.get(types[i], 0) + 1
    return {"rows": len(indices), "case_mix": dict(sorted(mix.items())),
            "turns": distribution([len(p) for p in plans]),
            "context_tokens": distribution([len(data[i]["input_ids"]) for i in indices]),
            "generated_tokens": distribution([sum(n for _, n in p) for p in plans])}


def composition(tok, data: list[dict], types: list[str], mix: dict[str, int], rows: dict, sample: dict) -> dict:
    """Each batch's and sample's mix and length distribution, next to the SFT data's and the RL split's mix."""
    out = {"rl_split_case_mix": mix, "sft_data": describe_rows(tok, data, types, list(range(len(data))))}
    for b, indices in rows["batches"].items():
        out[f"batch {b}"] = describe_rows(tok, data, types, indices)
    out["single_episode"] = describe_rows(tok, data, types, rows["single"])
    out["update_sample"] = describe_rows(tok, data, types, sample["sample"])
    out["update_longest_context_tokens"] = len(data[sample["longest"]]["input_ids"])
    return out


def update_time(data: list[dict], tiny: bool, warmup: int = 2, longest: int = 0) -> dict:
    """(a): seconds per sequence for a forward-backward plus the KL reference forward; peak memory. `data` is the warm-up
    rows, the sample, then `longest` rows reported on their own."""
    import torch
    from peft import get_peft_model
    tok = training.tokenizer()
    cuda = torch.cuda.is_available() and not tiny
    model = training.load_model(training.MODEL, tiny, tok)
    model = get_peft_model(model, training.lora_config())
    if cuda:
        model = model.cuda()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        torch.cuda.reset_peak_memory_stats()
    model.train()       # as the trainers do: gradient checkpointing applies only in training mode (GPU session 1: out of memory)
    device = next(model.parameters()).device
    timings, ref_timings = [], []
    for i, r in enumerate(data):
        ids = torch.tensor([r["input_ids"]], device=device)
        labels = torch.tensor([r["labels"]], device=device)
        sync = torch.cuda.synchronize if cuda else (lambda: None)
        sync()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        sync()
        t1 = time.perf_counter()
        with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            model(input_ids=ids, use_cache=False)
        sync()
        t2 = time.perf_counter()
        if i >= warmup:
            timings.append(t1 - t0)
            ref_timings.append(t2 - t1)
    if longest:
        extra = [a + b for a, b in zip(timings[-longest:], ref_timings[-longest:])]
        timings, ref_timings = timings[:-longest], ref_timings[:-longest]
    measured = data[warmup:len(data) - longest]
    out = {"sequences": len(timings), "tokens_mean": round(statistics.fmean(len(r["input_ids"]) for r in measured)),
           "seconds_forward_backward": round(statistics.fmean(timings), 3),
           "seconds_reference_forward": round(statistics.fmean(ref_timings), 3)}
    out["seconds_per_sequence_update_with_kl"] = round(out["seconds_forward_backward"] + out["seconds_reference_forward"], 3)
    if longest:
        out["longest_sequences"] = [{"tokens": len(r["input_ids"]), "seconds_update_with_kl": round(s, 3)}
                                    for r, s in zip(data[-longest:], extra)]
    if cuda:
        out["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        # vLLM starts in this process next and refuses to start if this memory is still held by torch's cache
        del model, loss, ids, labels
        gc.collect()      # the PEFT modules hold reference cycles
        torch.cuda.empty_cache()
    return out


def turn_plan(tok, row: dict) -> list[tuple[list[int], int]]:
    """(context before assistant turn k, that turn's length) for each assistant turn of one demonstration."""
    ids = row["input_ids"]
    header = tok.encode(ASSISTANT_HEADER, add_special_tokens=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    plan, i = [], 0
    while True:
        j = next((k for k in range(i, len(ids) - len(header) + 1) if ids[k:k + len(header)] == header), None)
        if j is None:
            return plan
        start = j + len(header)
        end = ids.index(im_end, start) + 1
        plan.append((ids[:start], end - start))
        i = end


def generation_throughput(engine, tok, data: list[dict], batches: dict[int, list[int]]) -> dict:
    """(b): teacher-forced multi-turn replay in lockstep rounds per batch; `batches` is each size's rows (`allocate`)."""
    out = {}
    for b, indices in batches.items():
        batch = [turn_plan(tok, data[i]) for i in indices]
        t0, rounds = time.perf_counter(), 0
        while True:
            active = [p[rounds] for p in batch if rounds < len(p)]
            if not active:
                break
            engine.generate_turns(active)
            rounds += 1
        wall = time.perf_counter() - t0
        out[str(b)] = {"episodes": b, "rounds": rounds, "wall_seconds": round(wall, 2),
                       "episodes_per_hour": round(b / wall * 3600, 1)}
    return out


SINGLE_EPISODE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
SINGLE_EPISODE_WARMUP_QUANTILE = 0.1


def single_episode_latency(engine, tok, warmup: dict, rows: list[dict], clock=time.perf_counter) -> dict:
    """(c): one episode at a time, turn by turn, shortest first. Seconds per episode and per turn; a warm-up row first."""
    for turn in turn_plan(tok, warmup):
        engine.generate_turns([turn])
    episodes = []
    for q, row in zip(SINGLE_EPISODE_QUANTILES, rows):
        plan = turn_plan(tok, row)
        turns = []
        for turn in plan:
            t0 = clock()
            engine.generate_turns([turn])
            turns.append(clock() - t0)
        wall, generated = sum(turns), sum(n for _, n in plan)
        episodes.append({"length_quantile": q, "context_tokens": len(row["input_ids"]), "turns": len(plan),
                         "generated_tokens": generated, "wall_seconds": round(wall, 2),
                         "seconds_per_turn_mean": round(wall / len(plan), 3),
                         "seconds_per_turn_max": round(max(turns), 3),
                         "generated_tokens_per_second": round(generated / wall, 1) if wall else None})
    walls = [e["wall_seconds"] for e in episodes]
    return {"episodes": episodes, "wall_seconds_median": statistics.median(walls), "wall_seconds_max": max(walls),
            "scope": "generation only, teacher-forced replay, one episode in the engine"}


class VLLMEngine:
    def __init__(self):
        from vllm import LLM
        from ap_sim.training.config import PIPELINE_MAX_CONTEXT_LENGTH
        self.llm = LLM(model=training.MODEL, revision=training.REVISION, dtype="bfloat16",
                       max_model_len=PIPELINE_MAX_CONTEXT_LENGTH, gpu_memory_utilization=0.85,
                       enable_prefix_caching=True, seed=0)

    def prefix_caching(self):
        try:
            return self.llm.llm_engine.vllm_config.cache_config.enable_prefix_caching
        except AttributeError:
            return "unknown"

    def generate_turns(self, turns):
        from vllm import SamplingParams
        params = [SamplingParams(temperature=0.8, top_p=1.0, max_tokens=n, min_tokens=n, ignore_eos=True, seed=0)
                  for _, n in turns]
        self.llm.generate([{"prompt_token_ids": ctx} for ctx, _ in turns], sampling_params=params, use_tqdm=False)


class FakeEngine:
    """CPU smoke test: records the requests; no generation."""

    def __init__(self):
        self.requests = []

    def prefix_caching(self):
        return "fake"

    def generate_turns(self, turns):
        self.requests.append([(len(c), n) for c, n in turns])


def main() -> None:
    import pipeline_guard
    pipeline_guard.require()          # refuses to start on a settings mismatch (config.require_pipeline)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--update-sequences", type=int, default=12)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--describe", action="store_true", help="print the rows each part uses, and stop (no model, no GPU)")
    ap.add_argument("--gpu", choices=("A100", "H100"), help="the family of this session's GPU; checked against the device")
    ap.add_argument("--price-per-hour", type=float, help="this session's actual GPU price, $/h")
    ap.add_argument("--volume-gb", type=float, help="this session's network volume, counted against the GPU budget")
    ap.add_argument("--volume-per-gb-month", type=float)
    ap.add_argument("--volume-months", type=float, default=1.0, help="how long the volume is budgeted for")
    a = ap.parse_args()
    if not a.describe and None in (a.gpu, a.price_per_hour, a.volume_gb, a.volume_per_gb_month):
        ap.error("--gpu, --price-per-hour, --volume-gb and --volume-per-gb-month are required: the rule uses the actual figures")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = rows(pathlib.Path(a.data))
    tok = training.tokenizer()
    types, mix = row_types(len(data)), rl_mix()
    sizes = TINY_BATCH_SIZES if a.tiny else BATCH_SIZES
    chosen = allocate(data, types, mix, sizes)
    sample = update_sample(data, types, mix, 2 if a.tiny else a.update_sequences)
    described = composition(tok, data, types, mix, chosen, sample)
    if a.describe:
        print(json.dumps(described, indent=1))
        return
    import torch
    device = "tiny" if a.tiny else torch.cuda.get_device_name(0)
    if not a.tiny and a.gpu not in device:
        sys.exit(f"--gpu {a.gpu} does not match this device: {device}")
    order = [*sample["warmup"], *sample["sample"], sample["longest"]]
    update_rows = [data[i] for i in order]
    if a.tiny:
        update_rows = [{"input_ids": r["input_ids"][-400:], "labels": r["labels"][-400:]} for r in update_rows]
    upd = update_time(update_rows, a.tiny, longest=1)
    engine = FakeEngine() if a.tiny else VLLMEngine()
    single = single_episode_latency(engine, tok, data[chosen["warmup"]], [data[i] for i in chosen["single"]])
    gen = generation_throughput(engine, tok, data, chosen["batches"])
    step_batch, largest = gen[str(sizes[0])], gen[str(sizes[-1])]
    measured = {"gpu": device, "price_per_hour": a.price_per_hour,
                "update": upd, "generation": gen, "single_episode": single, "composition": described,
                "prefix_caching": engine.prefix_caching(), "episodes_per_hour": largest["episodes_per_hour"],
                "step_generation_minutes": round(step_batch["wall_seconds"] / 60, 2),
                "seconds_per_sequence_update_with_kl": upd["seconds_per_sequence_update_with_kl"]}
    (out / "measured.json").write_text(json.dumps(measured, indent=1))
    import budget_forecast
    volume = budget_forecast.volume_dollars(a.volume_gb, a.volume_per_gb_month, a.volume_months)
    decision = {f"test {n}": budget_forecast.decide_with_scenarios(measured, n, a.gpu, a.price_per_hour, volume)
                for n in (200, 300)}
    (out / "decision.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps({"measured": measured, "decision_200": decision["test 200"]["decision"]}, indent=1))


if __name__ == "__main__":
    main()
