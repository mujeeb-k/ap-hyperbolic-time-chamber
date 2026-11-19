"""Parse and scoring time per optimizer step (added on 2026-11-03 for Phase 8 of the pre-registered plan, GRPO plumbing): TRL's
parser and the reward function, with a warning if either takes more than 60 s in a step. The evaluation harness audit (Phase 5)
bounds the parser at about 5.5 s per whole-budget turn, with an unknown-tool-name echo about 5.6 s and history re-decoding
0.2-0.8 s per rollout.

Separate from `monitors`, which was frozen with the evaluation and runner code at the end of Phase 5. Timing changes nothing:
the parser and the reward function return, and raise, exactly what they would. Each step's row in timing.jsonl also holds what
the reward function returned, as counts (Phase 8's step-1 check reads them), the token weights TRL's importance correction gave
the tokens in the loss (TRL logs only their min, mean and max), with a warning when too many are at the cap against Phase 8's
baseline, and how much of the LoRA update survives the bf16 merge that vLLM receives. `parse_response` is the frozen hook's,
`rollout_hooks.parse_response`; only GRPOTrainer's call goes through the timer, and the evaluation harness calls the hook
directly.
"""
from __future__ import annotations

import functools
import hashlib
import json
import pathlib
import time
import warnings

WARNING_SECONDS = 60.0
WARNING_SHARE_AT_CAP = 0.01         # without a baseline
# (2026-11-03) Against the baseline Phase 8's first real step records (the adapter is zero there, so it holds only the difference between
# vLLM and the trainer): at most twice it, and never below 0.1%, since a baseline of 0 would otherwise fail on one token.
BASELINE_MULTIPLE, SHARE_AT_CAP_FLOOR = 2.0, 0.001
# (2026-11-03) After Phase 8's first real step these are drift monitors, not fault detectors. A flag never stops a run.
# A flag is reported for review. The outcome alarms stay the automatic stops. The share at the cap is flagged above its
# limit on more than half the updates. The share below 1/3 is flagged above the same limit on its own baseline for this many updates in a row.
SCORED_SHARE_MIN, SUSTAINED_UPDATES = 0.99, 3
TIMING_FILE = "timing.jsonl"
_blank = {"parse_seconds": 0.0, "parse_calls": 0, "scoring_seconds": 0.0, "scoring_calls": 0, "rollouts": 0, "rewarded": 0,
          "failed": 0, "penalized": 0, "excluded": 0, "importance": None, "rollout_digest": None}
_state = dict(_blank)
_baseline: dict = {"share_at_cap": None}


def reset() -> None:
    _state.clear()
    _state.update(_blank)


def timed_parser(parser):
    @functools.wraps(parser)
    def parse(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return parser(*args, **kwargs)
        finally:
            _state["parse_seconds"] += time.perf_counter() - t0
            _state["parse_calls"] += 1
    return parse


def timed_reward(reward_fn):
    """The reward function, unchanged in its values and name, timed; its values counted by kind."""
    @functools.wraps(reward_fn)
    def scored(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = reward_fn(*args, **kwargs)
        finally:
            _state["scoring_seconds"] += time.perf_counter() - t0
            _state["scoring_calls"] += 1
        for r in out:
            _state["rollouts"] += 1
            key = "excluded" if r is None else "rewarded" if r == 1.0 else "failed" if r == 0.0 else "penalized"
            _state[key] += 1
        return out
    return scored


def importance_figures(ratio, mask, cap: float, sampling=None) -> dict | None:
    """One generation's token weights (TRL's token_truncate: the exponential of trainer minus vLLM log-probability, clamped above at
    `cap`), over the tokens in the loss that vLLM scored. `mask`: completion mask times tool mask; everywhere else TRL's weight is
    exactly 1, as it is where `sampling`, vLLM's log-probabilities, is NaN, so counting those would hide what the weights say. None
    where TRL computes no ratio (no vLLM). Without a floor no weight is 0: the low end is descriptive. A ratio that is not per token
    has no figures: nothing here may raise into training, and Phase 8's check fails on it."""
    if ratio is None:
        return None
    import torch
    in_loss = mask.bool()
    scored = in_loss if sampling is None else in_loss & ~torch.isnan(sampling)
    w = ratio.detach().float()[scored].cpu() if ratio.shape == mask.shape else torch.empty(0)
    figures = {"shape": list(ratio.shape), "tokens_in_loss": int(in_loss.sum()), "tokens_scored": w.numel()}
    if not w.numel():
        return {**figures, "share_at_cap": None, "mean_weight": None, "share_below_one_third": None, "quantiles": None}
    quantiles = torch.quantile(w, torch.tensor([0.0, 0.001, 0.5, 0.999, 1.0])).tolist()
    return {**figures, "share_at_cap": round(float((w >= cap).float().mean()), 6), "mean_weight": round(float(w.mean()), 5),
            "share_below_one_third": round(float((w < 1 / 3).float().mean()), 6),
            "quantiles": {k: round(v, 4) for k, v in zip(("0%", "0.1%", "50%", "99.9%", "100%"), quantiles)}}


def share_at_cap_limit(baseline: float | None) -> float:
    return WARNING_SHARE_AT_CAP if baseline is None else max(BASELINE_MULTIPLE * baseline, SHARE_AT_CAP_FLOOR)


def baselines_from_step(step_json: pathlib.Path) -> dict | None:
    """What Phase 8's first real step recorded for its token weights, from step.json. None without it."""
    if not step_json.exists():
        return None
    weights = (json.loads(step_json.read_text())["checks"].get("inference_difference") or {}).get("weights") or {}
    return {k: weights.get(k) for k in ("share_at_cap", "share_below_one_third")}


def monitors(rows: list[dict], baselines: dict | None) -> dict:
    """The drift monitors over a run's rows in timing.jsonl, reported for review. Reads only, and stops nothing. A re-run's row
    for a step replaces the earlier one."""
    by_step = {r.get("step", i): r for i, r in enumerate(rows)}
    updates = [by_step[s].get("importance") or {} for s in sorted(by_step)]
    base = baselines or {}
    cap_limit = share_at_cap_limit(base.get("share_at_cap"))
    third_limit = share_at_cap_limit(base.get("share_below_one_third")) if base.get("share_below_one_third") is not None else SHARE_AT_CAP_FLOOR
    at_cap = [u.get("share_at_cap") for u in updates]
    below = [u.get("share_below_one_third") for u in updates]
    longest = run = 0
    for b in below:
        run = run + 1 if b is not None and b > third_limit else 0
        longest = max(longest, run)
    above_cap = sum(c is not None and c > cap_limit for c in at_cap)
    unscored = sum(not u.get("tokens_scored") or u["tokens_scored"] < SCORED_SHARE_MIN * u["tokens_in_loss"] for u in updates)
    flags = []
    if baselines is None:
        flags.append("no baseline from Phase 8's first step: the limits below are the fallbacks")
    if above_cap > len(updates) / 2:
        flags.append(f"share at the cap above {cap_limit} on {above_cap} of {len(updates)} updates")
    if longest >= SUSTAINED_UPDATES:
        flags.append(f"share below 1/3 above {third_limit} for {longest} updates in a row")
    if unscored:
        flags.append(f"fewer than {SCORED_SHARE_MIN:.0%} of the tokens scored on {unscored} of {len(updates)} updates")
    return {"flags": flags, "updates": len(updates), "baselines": baselines, "share_at_cap_limit": cap_limit, "share_below_one_third_limit": third_limit,
            "share_at_cap_by_step": at_cap, "share_below_one_third_by_step": below,
            "mean_weight_by_step": [u.get("mean_weight") for u in updates],
            "lora_retained_by_step": [(by_step[s].get("lora_retained") or {}).get("share") for s in sorted(by_step)]}


def retained_at_step(rows: list[dict], step: int) -> dict | None:
    """The retained share in the row for this step, the update just made. A re-run's row replaces the earlier one."""
    found = [r for r in rows if r.get("step") == step]
    return found[-1].get("lora_retained") if found else None


def set_baseline(share_at_cap: float | None) -> None:
    """Phase 8's first real step's share of tokens at the cap, which later steps' warnings compare against."""
    _baseline["share_at_cap"] = share_at_cap


def lora_retained(model, adapter: str = "default") -> dict:
    """How much of the LoRA update survives TRL's merge at the next sync: PEFT adds each layer's delta to its bf16 base weight in place,
    and what is smaller than half a step rounds away. `share`: the update's own direction, applied . true / true . true, over every
    LoRA layer (1 is all of it). `norm_ratio`: |applied| / |true|. None before any update. Reads only: nothing is written, merged or drawn."""
    import torch
    dot, true, applied = 0.0, 0.0, 0.0
    layers = 0
    with torch.no_grad():
        for m in model.modules():
            if not (hasattr(m, "get_delta_weight") and adapter in getattr(m, "lora_A", {})):
                continue
            delta = m.get_delta_weight(adapter).float()
            weight = m.get_base_layer().weight
            gained = (weight.float() + delta).to(weight.dtype).float() - weight.float()
            dot += float((gained * delta).sum(dtype=torch.float64))
            true += float((delta * delta).sum(dtype=torch.float64))
            applied += float((gained * gained).sum(dtype=torch.float64))
            layers += 1
            del delta, gained
    return {"layers": layers, "share": round(dot / true, 5) if true else None,
            "norm_ratio": round((applied / true) ** 0.5, 5) if true else None, "update_norm": round(true ** 0.5, 6)}


def loss_mask(inputs: dict):
    """The tokens the loss reads, as TRL computes it: completion mask times tool mask."""
    tools = inputs.get("tool_mask")
    return inputs["completion_mask"] if tools is None else inputs["completion_mask"] * tools


def rollout_digest(completion_ids, completion_mask=None, before: str | None = None) -> str:
    """sha256 of the step's generated token ids, each rollout's own tokens in order (padding left out), chained onto the step's earlier
    generations: equal digests mean the same rollouts. The follow-up seed runs used it to check that each new seed's first step
    differs from the original's. Reads only; returns hex."""
    h = hashlib.sha256((before or "").encode())
    ids = completion_ids.tolist() if hasattr(completion_ids, "tolist") else completion_ids
    mask = completion_mask.tolist() if hasattr(completion_mask, "tolist") else completion_mask
    for i, row_ids in enumerate(ids):
        keep = row_ids if mask is None else [t for t, m in zip(row_ids, mask[i]) if m]
        h.update(json.dumps(keep).encode() + b"\n")
    return h.hexdigest()


def install(trainer) -> None:
    """Over the frozen hook's parser, after rollout_hooks.install has put it where GRPOTrainer calls it. Each generation's weights
    are read from what TRL returns, which is passed on untouched."""
    from trl.trainer import grpo_trainer
    grpo_trainer.parse_response = timed_parser(grpo_trainer.parse_response)
    score = trainer._generate_and_score_completions

    def scored(*args, **kwargs):
        out = score(*args, **kwargs)
        if out.get("completion_ids") is not None:
            _state["rollout_digest"] = rollout_digest(out["completion_ids"], out.get("completion_mask"), _state["rollout_digest"])
        if out.get("importance_sampling_ratio") is not None:
            _state["importance"] = importance_figures(out["importance_sampling_ratio"], loss_mask(out),
                                                      trainer.vllm_importance_sampling_clip_max, out.get("sampling_per_token_logps"))
        return out
    trainer._generate_and_score_completions = scored
    reset()


def row(step: int) -> dict:
    """This step's row; the counters start again."""
    out = {"step": step, **{k: round(v, 3) if isinstance(v, float) else v for k, v in _state.items()}}
    reset()
    return out


def callback():
    from transformers import TrainerCallback

    class StepLog(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            entry = row(state.global_step)
            try:                    # a log: nothing here may stop training
                entry["lora_retained"] = lora_retained(kw["model"]) if kw.get("model") is not None else None
            except Exception as e:  # noqa: BLE001
                entry["lora_retained"] = {"error": repr(e)[:200]}
                warnings.warn(f"step {state.global_step}: the retained share could not be read: {e!r}", stacklevel=1)
            with open(pathlib.Path(args.output_dir) / TIMING_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            for what in ("parse_seconds", "scoring_seconds"):
                if entry[what] > WARNING_SECONDS:
                    warnings.warn(f"step {state.global_step}: {what} {entry[what]} above {WARNING_SECONDS} s: investigate",
                                  stacklevel=1)
            weights = entry["importance"]
            limit = share_at_cap_limit(_baseline["share_at_cap"])
            if weights and (weights["share_at_cap"] or 0) > limit:
                warnings.warn(f"step {state.global_step}: {weights['share_at_cap']} of the tokens in the loss at the importance cap, "
                              f"above {limit}: vLLM and the trainer differ", stacklevel=1)
            return control

    return StepLog()


__all__ = ["BASELINE_MULTIPLE", "SCORED_SHARE_MIN", "SHARE_AT_CAP_FLOOR", "SUSTAINED_UPDATES", "WARNING_SHARE_AT_CAP", "WARNING_SECONDS", "TIMING_FILE",
           "baselines_from_step", "callback", "importance_figures", "install", "lora_retained", "loss_mask", "monitors", "reset", "retained_at_step", "rollout_digest", "row",
           "set_baseline", "share_at_cap_limit", "timed_parser", "timed_reward"]
