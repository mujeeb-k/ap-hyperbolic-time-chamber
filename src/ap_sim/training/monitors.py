r"""Training monitors outside the frozen reward code, added on 2026-11-01 with the evaluation harness (Phase 5 of the
pre-registered plan).

  - Think-block anomaly counter. The pinned model (Qwen3-4B-Instruct-2507) does not think; TRL's parser moves a
    <think>...</think> block into reasoning_content, where a tool call is neither parsed nor executed, and the frozen
    rewards.unparsed_tool_call does not see it (a known gap affecting only a label and the malformed-call count, never
    the reward; recorded, not re-pinned). `count_think_blocks(reward_fn)` wraps the reward function: the rewards pass
    through unchanged (the same values, the same function name for TRL's logging), and each call counts the episodes
    whose completion holds a <think> block. `AnomalyLog`, a trainer callback, writes the count per optimizer step to
    anomalies.jsonl in the output directory and warns on any occurrence (a warning to investigate).
  - The decoded-text check. TRL's parser keeps only a turn's last <think> block and drops earlier ones (with any tool
    call inside them) from every parsed field, and it recognises the tags however they are tokenized; so the flags are
    decided from the turn's raw decoded text, never from the parsed fields: any think marker in any form (special tokens or
    plain characters, any spacing or case) marks a think-block anomaly, and every <tool_call> in the text must match a
    call that was parsed and executed (`turn_flags`). The harness, RFT eligibility and the training anomaly counter all
    use this check.
  - Linear time on any text. Every pattern here runs on model-written text, so each is written without nested or
    adjacent quantifiers over the same characters. An earlier pattern, `<\s*/?\s*think`, split a whitespace
    run between its two `\s*` in every way (quadratic in the run; a `<` and 40,000 spaces took 8.4 s). The counter decodes
    only the tokens the model generated (`install` keeps TRL's tool mask for the reward call), never the tool results,
    which echo model-written field values.
  - The infrastructure circuit breaker across a resume. rewards.monitor_infrastructure keeps the last 500
    episodes' infrastructure flags and the per-step counts in module state, which a resumed run (a new process) would
    start empty. `MonitorCheckpoint`, a trainer callback, writes that state into every checkpoint
    (infra_monitor.json), and `restore_infrastructure(checkpoint)` puts it back before training resumes (or clears it on a
    fresh start). rewards.py is unchanged.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re
import warnings

from ap_sim.training import rewards

THINK_TOKEN_IDS = (151667, 151668)        # <think>, </think> (the pinned tokenizer)
INFRA_FILE = "infra_monitor.json"
_think_counts: list[dict] = []            # one entry per reward call, drained by AnomalyLog at each step
_tool_masks: list | None = None           # TRL's tool mask for the latest generation batch (install), taken by the counter
# any think marker: "<" then an optional "/" and "think", with any spacing and case (<think>, </think>, < THINK >,
# </ think>, <thinking ...); deliberately wide: the pinned model does not think, so a false alarm costs only a look.
# Linear time: each whitespace run is matched by one quantifier only; after a "<" the scan reads at most
# the run that follows it, and those runs are disjoint ("<" is not whitespace).
THINK_MARK = re.compile(r"<\s*(?:/\s*)?think", re.IGNORECASE)
TOOL_CALL_MARK = re.compile(r"<\s*tool_call\s*>", re.IGNORECASE)      # each opening tag (not </tool_call>); linear


def decode(tokenizer, ids) -> str:
    """The raw text of generated ids, special tokens included (a <think> special token decodes to "<think>")."""
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def think_marked(text: str = "", ids=()) -> bool:
    """A think marker in the raw text, or a think token among the ids."""
    return bool(THINK_MARK.search(text or "")) or any(t in THINK_TOKEN_IDS for t in ids or ())


def turn_flags(text: str, ids, parsed_calls: int, executed_calls: int) -> dict:
    """A turn's parse flags from its raw decoded text. Attempted: a parsed call, a <tool_call>
    in the text, or a think marker. Parsed: at least one parsed call, every <tool_call> in the text matched by a call
    that was parsed and executed, and no think marker (a think-block turn counts as a parse failure, conservatively)."""
    marks = len(TOOL_CALL_MARK.findall(text or ""))
    think = think_marked(text, ids)
    attempted = bool(parsed_calls or marks or think)
    parsed = bool(parsed_calls and marks == parsed_calls == executed_calls and not think)
    return {"tool_call_marks": marks, "think_block": think, "attempted": attempted, "parsed": parsed,
            "unparsed_tool_call": attempted and not parsed}


def generated_segments(ids, mask=None) -> list[list[int]]:
    """The runs of ids the model generated (mask 1; TRL's tool mask: 0 on tool results). Without a mask, or with one
    that does not match the ids, all the ids as one run (conservative: a tool result's echo may then raise a false
    alarm, but nothing generated is missed)."""
    ids = list(ids)
    if mask is None or len(mask) != len(ids):
        return [ids]
    runs, run = [], []
    for t, m in zip(ids, mask):
        if m:
            run.append(t)
        elif run:
            runs.append(run)
            run = []
    return runs + [run] if run else runs


def think_episode(completion, completion_ids=None, tokenizer=None, mask=None) -> bool:
    """True when the completion holds a think marker in what the model generated: in its decoded ids (with a
    tokenizer; only the generated runs when TRL's tool mask is given), a think token among them, or a marker in an
    assistant message (content, reasoning_content). Tool messages are the environment's (they echo model-written field
    values) and are not read."""
    if completion_ids is not None and any(think_marked(decode(tokenizer, run) if tokenizer else "", run)
                                          for run in generated_segments(completion_ids, mask)):
        return True
    return any(isinstance(m, dict) and m.get("role", "assistant") == "assistant"
               and (m.get("reasoning_content") or think_marked(str(m.get("content") or "")))
               for m in completion or [])


def count_think_blocks(reward_fn, tokenizer):
    """The reward function, unchanged in its values and name, counting think-block episodes per call (the completion's
    ids decoded with the tokenizer, required: without the decoded text, plain-character think tags are missed)."""
    if tokenizer is None:
        raise ValueError("the think-block counter needs the trainer's tokenizer")
    @functools.wraps(reward_fn)
    def wrapped(*args, completions=None, completion_ids=None, **kwargs):
        out = reward_fn(*args, completions=completions, completion_ids=completion_ids, **kwargs)
        comps = completions or []
        ids = completion_ids if completion_ids is not None else [None] * len(comps)
        masks = take_tool_masks()
        if masks is None or len(masks) != len(ids):
            masks = [None] * len(ids)                 # no mask for this batch: every id decoded (conservative)
        state = kwargs.get("trainer_state")
        _think_counts.append({"step": getattr(state, "global_step", None), "episodes": len(comps),
                              "think_block_episodes": sum(think_episode(c, i, tokenizer, m)
                                                          for c, i, m in zip(comps, ids, masks))})
        return out
    return wrapped


def take_tool_masks() -> list | None:
    """The tool mask kept for the latest generation batch, once (a later reward call without a new batch gets None)."""
    global _tool_masks
    masks, _tool_masks = _tool_masks, None
    return masks


def install(trainer) -> None:
    """Keep TRL's tool mask (1 on generated tokens, 0 on tool results) for the reward call that follows each generation
    batch: GRPOTrainer._generate returns it from _tool_call_loop, then scores that same batch's completion_ids, which
    the reward function receives without the mask."""
    loop = trainer._tool_call_loop

    @functools.wraps(loop)
    def tool_call_loop(*args, **kwargs):
        global _tool_masks
        out = loop(*args, **kwargs)
        _tool_masks = [list(m) for m in out[0]]
        return out
    trainer._tool_call_loop = tool_call_loop


def drain_think_counts() -> dict:
    episodes = sum(c["episodes"] for c in _think_counts)
    think = sum(c["think_block_episodes"] for c in _think_counts)
    _think_counts.clear()
    return {"episodes": episodes, "think_block_episodes": think}


def save_infrastructure(directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / INFRA_FILE).write_text(json.dumps({
        "recent": [bool(x) for x in rewards._infra_recent],
        "by_step": {str(k): v for k, v in rewards._infra_by_step.items()}}))


def restore_infrastructure(checkpoint: str | pathlib.Path | None) -> bool:
    """Restore the circuit breaker's window from a checkpoint (True), or clear it for a fresh start (False)."""
    rewards._infra_recent.clear()
    rewards._infra_by_step.clear()
    if checkpoint is None:
        return False
    path = pathlib.Path(checkpoint) / INFRA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: a checkpoint without the circuit breaker's window cannot be resumed safely")
    data = json.loads(path.read_text())
    rewards._infra_recent.extend(data["recent"])
    rewards._infra_by_step.update({int(k): v for k, v in data["by_step"].items()})
    return True


def callbacks() -> list:
    from transformers import TrainerCallback

    class MonitorCheckpoint(TrainerCallback):
        """Writes the circuit breaker's window into each checkpoint the trainer saves."""

        def on_save(self, args, state, control, **kw):
            save_infrastructure(pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}")
            return control

    class AnomalyLog(TrainerCallback):
        """Per optimizer step: the think-block episodes counted by the wrapped reward, to anomalies.jsonl; a warning on
        any occurrence."""

        def on_step_end(self, args, state, control, **kw):
            counts = drain_think_counts()
            row = {"step": state.global_step, **counts}
            with open(pathlib.Path(args.output_dir) / "anomalies.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if counts["think_block_episodes"]:
                warnings.warn(f"step {state.global_step}: {counts['think_block_episodes']} episode(s) with a <think> "
                              "block (the pinned model does not think): investigate", stacklevel=1)
            return control

    return [MonitorCheckpoint(), AnomalyLog()]


__all__ = ["count_think_blocks", "callbacks", "decode", "generated_segments", "install", "restore_infrastructure",
           "save_infrastructure", "take_tool_masks", "think_episode", "think_marked", "turn_flags"]
