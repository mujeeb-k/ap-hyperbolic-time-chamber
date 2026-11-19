"""Padding trimmed per micro-batch for TRL 1.16's GRPOTrainer (added on 2026-11-02 and 2026-11-03).

TRL pads all 64 rollouts of a generation to one width (`_generate_and_score_completions`) and keeps that padding in every
micro-batch, and `GRPOConfig` has no option that removes it (only `pad_to_multiple_of`). A padded token costs a forward and a
backward pass and changes nothing: padded rollouts ran about 3 times slower than unpadded ones in the GPU session 1 measurement.

All three passes per rollout (the gradient pass in `_compute_loss`; the old log-probabilities and the KL reference in
`_generate_and_score_completions`) go through `_get_per_token_logps_and_entropies`, so that is the one place wrapped here.
For each chunk of `batch_size` rollouts:

  - the trailing columns where no rollout of the chunk attends are dropped before the model runs;
  - a chunk that would then be all content gets one masked padding column at the end (right-trim/2). `transformers` drops an
    all-ones mask and sends the pass to another attention path (`masking_utils._ignore_causal_mask_sdpa`, SDPA's `is_causal`). On
    the GPU in bf16 that moved log-probabilities by up to 1.17 on the widest prompt's rollouts, which have no left padding (the first
    real step of GPU session 2); with one masked column they matched the untrimmed pass exactly. So every chunk takes the masked
    path, at the cost of one column;
  - the result is padded back to the width TRL expects with zeros, which carry no gradient. So everything downstream sees
    the shapes it always saw. TRL only reads these positions through the completion mask, which is 0 on them.

Rollouts stay separate rows: nothing is flattened into one sequence, and each keeps its own attention mask.

Only the right side is trimmed. The prompts' left padding (at most 26 tokens: prompts are 5,853 to 5,879) stays, because
removing it shifts every token's position: in bf16 that moved logits by a rounding step (0.0078) in a test on a small random
Qwen3, where removing right padding moved them by 0.0. Across a real pass the two widths can still round differently: one
float32 rounding step was seen on CPU (9.5e-7 on log-probabilities near -12).

The loss and gradients at micro-batch 1 and larger were tested against the untrimmed pass, within those rounding limits.
Phase 8 (GRPO plumbing, GPU session 2) makes the same comparison on the GPU on real rollouts.
"""
from __future__ import annotations

import torch

TRIM_VERSION = "right-trim/2"
# Positions from the attention mask in all three passes. Adopted on 2026-11-04 because the Phase 8 diagnostic showed that
# they meet the vLLM limits. It did not show them closer to vLLM.
MASK_POSITIONS = True
# Set False, in a commit of its own, only if the Phase 8 diagnostic found that the trim fails the vLLM limits and the
# untrimmed pass meets them. It did not, so the trim stays on.
TRIM_ENABLED = True


def attended_width(attention_mask: torch.Tensor) -> int:
    """How many columns the chunk needs: through the last column where any rollout attends. 0 if none does."""
    columns = attention_mask.any(dim=0).nonzero()
    return int(columns[-1]) + 1 if columns.numel() else 0


def mask_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Each token's position counted from the row's first attended token: what vLLM uses, whose prompts have no left padding. Without it Qwen3 uses `arange(width)`, so
    a left-padded row's positions are shifted by its padding, at any width."""
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


def decoder_of(model):
    """The module that turns position IDs into rotary embeddings, under whatever PEFT and accelerate wrap it in."""
    return next(m for _, m in model.named_modules() if hasattr(m, "rotary_emb") and hasattr(m, "embed_tokens"))


def use_mask_positions(model):
    """Every forward pass through this model takes `mask_positions`: the policy, old and reference passes alike, trimmed or not, since all of them reach this module, the
    reference with the adapter disabled. Returns the hook's handle, which removes it. A caller's own position IDs stand."""
    def with_positions(_module, args, kwargs):
        if kwargs.get("position_ids") is None and kwargs.get("attention_mask") is not None:
            kwargs["position_ids"] = mask_positions(kwargs["attention_mask"])
        return args, kwargs
    return decoder_of(model).register_forward_pre_hook(with_positions, with_kwargs=True)


def install(trainer) -> None:
    """Wrap the trainer's per-token pass. `trainer.padding_columns` counts the columns the passes ran on and what they would
    have run on untrimmed, as [trimmed, full], row-weighted; `trainer.padding_columns_by_pass` the same for the gradient pass and
    the two without gradients (Phase 8 reads it to show the trim is active on each)."""
    original = trainer._get_per_token_logps_and_entropies
    trainer.padding_columns = [0, 0]
    trainer.padding_columns_by_pass = {"gradient": [0, 0], "no_gradient": [0, 0]}
    trainer.padding_trim = TRIM_VERSION if TRIM_ENABLED else None
    if MASK_POSITIONS:
        use_mask_positions(trainer.model)
    if not TRIM_ENABLED:
        return

    def per_token(model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False,
                  compute_aux_loss=False, **kwargs):
        if compute_aux_loss or any(v is not None for v in kwargs.values()):
            raise RuntimeError(f"padding trim: only text-only passes without an auxiliary loss are supported: {sorted(kwargs)}")
        rows, full = input_ids.size(0), input_ids.size(1)
        prompt = full - logits_to_keep
        logps, entropies = [], []
        size = batch_size or rows
        for start in range(0, rows, size):
            ids, mask = input_ids[start:start + size], attention_mask[start:start + size]
            # At least one completion column: TRL slices labels[:, :-keep], and keep 0 would select every label.
            keep = max(attended_width(mask) - prompt, 1)
            cut_ids, cut_mask = ids[:, :prompt + keep], mask[:, :prompt + keep]
            extra = int(bool(cut_mask.all()))     # all content: one masked column keeps the chunk on the masked attention path
            if extra:
                cut_ids = torch.cat([cut_ids, cut_ids.new_full((cut_ids.size(0), 1), trainer.processing_class.pad_token_id)], dim=1)
                cut_mask = torch.cat([cut_mask, cut_mask.new_zeros((cut_mask.size(0), 1))], dim=1)
            lp, ent, _ = original(model, cut_ids, cut_mask, keep + extra, batch_size=ids.size(0),
                                  compute_entropy=compute_entropy, compute_aux_loss=False)
            pad = (0, logits_to_keep - keep)
            logps.append(torch.nn.functional.pad(lp[:, :keep], pad))      # the added column's own result goes: it is padding
            if compute_entropy:
                entropies.append(torch.nn.functional.pad(ent[:, :keep], pad))
            for count in (trainer.padding_columns, trainer.padding_columns_by_pass["gradient" if torch.is_grad_enabled() else "no_gradient"]):
                count[0] += ids.size(0) * (prompt + keep + extra)
                count[1] += ids.size(0) * full
        return torch.cat(logps, dim=0), torch.cat(entropies, dim=0) if compute_entropy else None, None

    trainer._get_per_token_logps_and_entropies = per_token


__all__ = ["MASK_POSITIONS", "TRIM_ENABLED", "TRIM_VERSION", "attended_width", "decoder_of", "install", "mask_positions", "use_mask_positions"]
