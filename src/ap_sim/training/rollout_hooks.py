"""Training-loop hooks for TRL 1.16's GRPOTrainer, installed together (`install`), added on 2026-11-01 with the
evaluation harness (Phase 5 of the pre-registered plan). The evaluation harness (ap_sim.evaluation.rollout) applies the
same rules, and a differential test checked that both give the same rollouts.

  - Tool-name guard. A tool call whose name is not valid Unicode (a lone surrogate from a JSON escape) is a malformed
    call: TRL cannot execute it and answers with its error text, which never reaches the environment. TRL then renders
    the tool-result suffix from a dummy tool call carrying the raw name, which the tokenizer cannot encode, and the
    whole step fails. The hook writes the name with the surrogate escaped (`safe_name`) into the tool message before the
    suffix is rendered (the same message object the completion holds), so the call stays a malformed call and the step
    goes on.
  - Episode end (pre-registered plan, Phase 8: rollouts end at episode end, as in evaluation). Once the environment has
    ended the episode during a tool block (an accepted finish, the action limit, the loop guard, which the environment
    applies: APReceiptEnv's execution rule) or failed (an infrastructure failure), that block ends at its <|im_end|> and
    the model generates nothing more for the rollout. The environments are EpisodeEndEnvV2, which remember the result
    string of the call that ended the episode; TRL's tool message holds that very string object. (loop_guard.install,
    which found the loop notice by matching text in the tool messages, is no longer used.)
  - Parser guard: the one decision point for parsed calls. TRL's parse_response falls back to plain text on a malformed
    tool call but catches only ValueError and TypeError; a tool call whose JSON nests past the interpreter's recursion
    limit (about 10,000 levels, a 5,000-token turn) raised RecursionError out of training and the harness, and where
    that limit falls moves with the caller's stack. An earlier guard measured bracket nesting in the text, which `]`
    inside a JSON string could lower (calls nested to ~9,990 then parsed, the loop guard's key overflowed, the
    environment counted one call twice, and training and evaluation scored the same turn differently). `parse_response`
    now decides after parsing, from what was parsed: `call_depth`, an iterative count of each call's nesting (strings
    are values, so nothing in them counts; it sees what transformers re-parses from a string argument), and a call
    deeper than MAX_JSON_DEPTH (4,096) gets TRL's own malformed-call fallback (the turn decoded as plain text). A parse
    that raises RecursionError gets the same fallback; that happens only past the cap (the limit is far above it), so a
    call deeper than the cap is unparsed and one within it is TRL's result, whatever the stack. `install` puts it where
    GRPOTrainer calls the parser; the harness (and so scripts/qwen_check.py), the smoke test and the TRL drivers call
    it. The rule: nothing recurses over model-supplied nested data before this check (TRL's own parse is inside it);
    everything after it is iterative or bounded by the cap (the environment's input limits, depth 16, are iterative; the
    loop guard keys a rejected call without serializing it; C-level JSON encoding is bounded far above 4,096). The one
    change from TRL: a call nested deeper than 4,096, which TRL parsed and the environment rejected (deeper than 16), is
    an unparsed call that ends the rollout; both are malformed.
"""
from __future__ import annotations

from ap_sim.training import loop_guard
from ap_sim.training.environment import APReceiptEnvV2

IM_END_TOKEN = "<|im_end|>"


MAX_JSON_DEPTH = 4_096   # levels in one call's JSON; the parser's recursion limit is about 9,990 and moves with the stack


def call_depth(value) -> int:
    """The nesting of a parsed value: a scalar 0, an object or a list one more than its deepest member. Iterative (an
    explicit stack, no recursion), so any parsed value is measured whatever its depth; linear in its size."""
    deepest, stack = 0, [(value, 1)]
    while stack:
        v, d = stack.pop()
        if isinstance(v, dict):
            deepest = max(deepest, d)
            stack.extend((x, d + 1) for x in v.values())
        elif isinstance(v, (list, tuple)):
            deepest = max(deepest, d)
            stack.extend((x, d + 1) for x in v)
    return deepest


def parse_response(tokenizer, ids, prefix=None, tools=None) -> dict:
    """trl.chat_template_utils.parse_response, except that a parsed call nested deeper than MAX_JSON_DEPTH (the model's
    call object: {"name", "arguments"}, counted by `call_depth` after parsing) or a RecursionError gets TRL's own
    fallback for a malformed call (the turn decoded as plain text, skipping special tokens): no call is executed."""
    from trl.chat_template_utils import parse_response as trl_parse_response
    fallback = lambda: {"role": "assistant", "content": tokenizer.decode(ids, skip_special_tokens=True)}   # noqa: E731
    try:
        parsed = trl_parse_response(tokenizer, ids, prefix=prefix, tools=tools)
    except RecursionError:       # only past the cap: the recursion limit is far above it
        return fallback()
    calls = parsed.get("tool_calls") or []
    if any(call_depth(c.get("function", c) if isinstance(c, dict) else c) > MAX_JSON_DEPTH for c in calls):
        return fallback()
    return parsed


def guard_parser() -> None:
    """GRPOTrainer calls the parser by the name it imported (in _generate and _tool_call_loop): point that name at the
    guarded parser."""
    from trl.trainer import grpo_trainer
    grpo_trainer.parse_response = parse_response


def safe_name(name):
    """The name with anything that is not valid Unicode escaped (\\udxxx); a valid name is returned unchanged."""
    if not isinstance(name, str):
        return name
    try:
        name.encode("utf-8")
        return name
    except UnicodeEncodeError:
        return name.encode("utf-8", "backslashreplace").decode("utf-8")


def sanitize(tool_messages: list[dict]) -> list[dict]:
    """Escape invalid tool names in place (the completion holds the same message objects)."""
    for m in tool_messages:
        if "name" in m:
            m["name"] = safe_name(m["name"])
    return tool_messages


class EpisodeEndEnvV2(APReceiptEnvV2):
    """APReceiptEnvV2 that remembers the result string of the call that ended the episode (or failed it). Its tools,
    schemas and results are APReceiptEnvV2's own (only private attributes are added)."""

    def reset(self, scenario: str = "V1", **kwargs) -> str:
        self._end_result = None
        return super().reset(scenario, **kwargs)

    def _call(self, tool: str, *args) -> str:
        before = self._layer is not None and (self._layer.done or self._infrastructure_failure is not None)
        out = super()._call(tool, *args)
        after = self._layer is not None and (self._layer.done or self._infrastructure_failure is not None)
        if after and not before:
            self._end_result = out
        return out


def make_environment_factory(scenarios, execution_rule: str = loop_guard.LOOP_GUARD):
    """The training environment factory: EpisodeEndEnvV2 (surfaces/4) under loop-guard/1 (the v2 execution rule)."""
    return lambda: EpisodeEndEnvV2(scenarios, execution_rule=execution_rule)


def ended(environments, tool_messages: list[dict]) -> bool:
    """True when one of these tool messages is the result of the call that ended (or failed) its episode."""
    ends = [getattr(e, "_end_result", None) for e in environments or []]
    return any(m.get("content") is r for m in tool_messages for r in ends if r is not None)


def install(trainer) -> None:
    """Loop guard, tool-name guard and episode end on a GRPOTrainer (always together, for training and evaluation).

    Two of TRL's internals are wrapped; each wrapper returns exactly the fields, types and shapes TRL's own returns:
      - `_get_tool_suffix_ids(tool_messages) -> list[int]`: invalid tool names escaped first; when the environment has
        ended (or failed) the episode in this tool block, the ids stop at the block's <|im_end|> (no assistant header).
        Whether the episode ended is environment state (`ended`: the environment's own record of the result that ended
        it), never a match on text: a loop-guard stop is an episode end (the environment sets it), so no tool result is
        searched for the loop notice.
      - `_generate_single_turn(prompt_ids, images, multimodal_fields, num_generations, has_tool_images) ->
        (ids, logprobs)`: a prompt whose ids end at <|im_end|> is a stopped rollout (a live prompt ends with the
        assistant header the suffix adds); it gets an empty turn and, when logprobs are produced, an empty logprob list.
        On the vLLM path TRL produces per-token logprobs, so when no prompt is live the result is empty lists, never
        None (TRL indexes them).
    """
    guard_parser()                                  # RecursionError in the parser falls back, never raises
    eos = trainer.processing_class.convert_tokens_to_ids(IM_END_TOKEN)
    suffix_ids, generate = trainer._get_tool_suffix_ids, trainer._generate_single_turn

    def tool_suffix(tool_messages):
        ids = suffix_ids(sanitize(tool_messages))
        if ended(getattr(trainer, "environments", None), tool_messages) and eos in ids and ids[-1] != eos:
            ids = ids[: len(ids) - ids[::-1].index(eos)]   # through the tool block's <|im_end|>, no assistant header
        return ids

    def generate_single_turn(prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        stopped = [bool(p) and p[-1] == eos for p in prompt_ids]
        if not any(stopped):
            return generate(prompt_ids, images, multimodal_fields, num_generations, has_tool_images)
        if images or multimodal_fields or num_generations != 1:
            raise RuntimeError("rollout hooks: only text-only tool turns are supported")
        live = [p for p, s in zip(prompt_ids, stopped) if not s]
        if live:
            ids, logprobs = generate(live, images, multimodal_fields, num_generations, has_tool_images)
        else:   # what TRL's own call returns for no prompts: no ids, and logprobs exactly when it produces them
            ids, logprobs = [], ([] if getattr(trainer, "use_vllm", False) else None)
        it = iter(ids)
        out = [[] if s else next(it) for s in stopped]
        if logprobs is not None:
            lt = iter(logprobs)
            logprobs = [[] if s else next(lt) for s in stopped]
        return out, logprobs

    trainer._get_tool_suffix_ids, trainer._generate_single_turn = tool_suffix, generate_single_turn
    trainer.execution_rule = loop_guard.LOOP_GUARD
    trainer.rollout_hooks = ("loop-guard/1", "tool-name guard", "episode end")
