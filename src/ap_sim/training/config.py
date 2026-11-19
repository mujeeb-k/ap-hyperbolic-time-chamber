"""The v2 pipeline settings: the observation schema, the instructions version and the episode limits every v2 run uses.
A v2 runner (the strong-model check run before training, SFT, GRPO and evaluation) calls `require_pipeline` before it
starts, and refuses to start without these settings.

The hard caps are fixed at 69,632 tokens (v1's PIPELINE_MAX_COMPLETION_LENGTH, Qwen tokenizer, tool results included)
and 40 tool calls (POLICY C1). The limits a runner uses are measured under them. Each limit is
min(2 x the largest value observed, its hard cap), where the largest is taken across reference/5 on the 280 non-V6
train-pool cases and Claude's 30 episodes in the strong-model check, both counted with the pinned Qwen tokenizer and
TRL's chat template. Claude's episodes raised the two token limits over the limits from reference/5 alone (28,124 and
39,874). The tool-call limit is unchanged.

In version 1 of this environment the equivalents were in `training/config.py`. Historical reproductions pass their own
schema explicitly and do not go through this module.
"""
from __future__ import annotations

PIPELINE_SCHEMA = "surfaces/4"
PIPELINE_INSTRUCTIONS = "instructions/5"

TOKEN_HARD_CAP = 69_632          # a ceiling only; no token limit may exceed it
TOOL_CALL_HARD_CAP = 40          # POLICY C1 (events/clock.ACTION_LIMIT)

# reference/5 on the 280 non-V6 train-pool cases
MEASURED_REFERENCE5 = {"tool_calls": 22, "completion_tokens": 14_062, "context_tokens": 19_937}
# Claude's 30 episodes in the strong-model check, thinking text included
MEASURED_CLAUDE = {"tool_calls": 22, "completion_tokens": 14_177, "context_tokens": 20_047}
MEASURED_LARGEST = {k: max(MEASURED_REFERENCE5[k], MEASURED_CLAUDE[k]) for k in MEASURED_REFERENCE5}
MEASURED_PROMPT_TOKENS = (5_853, 5_879)   # min, max over both: system prompt + tools + first page

PIPELINE_MAX_TOOL_CALLING_ITERATIONS = min(2 * MEASURED_LARGEST["tool_calls"], TOOL_CALL_HARD_CAP)     # 40 (cap)
PIPELINE_MAX_COMPLETION_LENGTH = min(2 * MEASURED_LARGEST["completion_tokens"], TOKEN_HARD_CAP)         # 28,354
PIPELINE_MAX_CONTEXT_LENGTH = min(2 * MEASURED_LARGEST["context_tokens"], TOKEN_HARD_CAP)               # 40,094

assert PIPELINE_MAX_TOOL_CALLING_ITERATIONS <= TOOL_CALL_HARD_CAP
assert PIPELINE_MAX_COMPLETION_LENGTH <= TOKEN_HARD_CAP and PIPELINE_MAX_CONTEXT_LENGTH <= TOKEN_HARD_CAP
# The longest prompt plus a full completion must fit the context limit.
assert MEASURED_PROMPT_TOKENS[1] + PIPELINE_MAX_COMPLETION_LENGTH <= PIPELINE_MAX_CONTEXT_LENGTH

_SETTINGS = {"schema": PIPELINE_SCHEMA, "instructions": PIPELINE_INSTRUCTIONS,
             "max_completion_length": PIPELINE_MAX_COMPLETION_LENGTH,
             "max_tool_calling_iterations": PIPELINE_MAX_TOOL_CALLING_ITERATIONS}


class PipelineConfigError(ValueError):
    """A v2 run was started without the pipeline settings, or with different ones."""


def require_pipeline(*, schema: str | None, instructions: str | None, max_completion_length: int | None,
                     max_tool_calling_iterations: int | None) -> None:
    """Refuse to start (PipelineConfigError) unless every setting is given and equals the pipeline's."""
    given = {"schema": schema, "instructions": instructions, "max_completion_length": max_completion_length,
             "max_tool_calling_iterations": max_tool_calling_iterations}
    problems = []
    for name, want in _SETTINGS.items():
        got = given[name]
        if got is None:
            problems.append(f"{name} is not set (the v2 pipeline uses {want!r})")
        elif got != want:
            problems.append(f"{name} is {got!r}, but the v2 pipeline uses {want!r}")
    if problems:
        raise PipelineConfigError("refusing to start a v2 run: " + "; ".join(problems)
                                  + ". Pass the values from ap_sim.training.config.")
