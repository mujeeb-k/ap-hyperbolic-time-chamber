"""The Qwen evaluation harness's rollout: one episode of a model on APReceiptEnvV2, rendered, parsed,
executed and limited exactly as TRL 1.16's GRPOTrainer does in training, so an evaluation sees what a rollout sees.

Transcribed from TRL 1.16.0 (grpo_trainer.py), which takes precedence:
  - template: GRPOTrainer keeps the model's own chat template when it is prefix-preserving with tools (the pinned
    Qwen3-4B-Instruct-2507's is) and swaps in TRL's training template otherwise (grpo_trainer.py 776-779);
  - tools: the environment's public methods in inspect.getmembers order, as GRPOTrainer builds them;
  - prompt: [system SYSTEM_PROMPT_V2, user USER_PREFIX + reset page], rendered once with the generation prompt;
  - ids are concatenated, never re-rendered: prompt + each generated turn + _get_tool_suffix_ids(tool messages);
  - parsing: trl.chat_template_utils.parse_response(tokenizer, ids, prefix=..., tools=...);
  - execution: TRL's own code transcribed (trl_tool_message): the tool called with **arguments, an exception or an
    unknown name answered with str({"error": ...}), never reaching the environment;
  - limits: each turn may generate at most the remaining completion budget (max_completion_length 28,354); a tool
    result whose suffix would push the completion past the budget, or the sequence to the context limit (40,094), is
    dropped and the rollout ends (TRL's rollback); at most max_tool_calling_iterations (40) tool turns.
  - the training hooks (training/rollout_hooks.install): loop-guard/1 (the v2 execution rule), the tool-name guard, and
    rollouts ending at episode end (an accepted finish, the action limit, the loop guard, an infrastructure failure).
A differential test ran varied and adversarial model outputs through TRL's own rollout code and this module and required
the same training reward, success and stop label.
The stop reason, finish_result_dropped and malformed calls come from the shared classifier (evaluation/stops.py).
"""
from __future__ import annotations

import inspect
import json
import time

from ap_sim.evaluation import stops
from ap_sim.training.config import (PIPELINE_MAX_COMPLETION_LENGTH, PIPELINE_MAX_CONTEXT_LENGTH,
                                    PIPELINE_MAX_TOOL_CALLING_ITERATIONS)
from ap_sim.training.environment import APReceiptEnvV2
from ap_sim.training.loop_guard import LOOP_GUARD
from ap_sim.training.monitors import THINK_TOKEN_IDS, decode, turn_flags   # noqa: F401 (re-exported)
from ap_sim.training.prompt import SYSTEM_PROMPT_V2, USER_PREFIX
from ap_sim.training.rollout_hooks import sanitize

IM_END, ENDOFTEXT = 151645, 151643     # both end a turn (rewards.IM_END, rewards.ENDOFTEXT)


def execute(env, name: str, args) -> str:
    """One tool call by name and arguments, as TRL 1.16's tool loop makes it (the form of `execute` in
    scripts/strong_model_check.py)."""
    return trl_tool_message(dict(trl_tool_dict(env)), {"type": "function", "function": {"name": name,
                                                                                       "arguments": args}})["content"]


def trl_tool_dict(env) -> dict:
    """GRPOTrainer's per-rollout sync tool dict: the environment's tools by __name__ (TRL 1.16.0, _generate)."""
    return {tool.__name__: tool for tool in trl_tools(env) if not inspect.iscoroutinefunction(tool)}


def trl_tool_message(tool_dict: dict, tool_call: dict) -> dict:
    """One parsed tool call executed exactly as GRPOTrainer._tool_call_loop executes it (TRL 1.16.0, transcribed): the
    tool called with ``**function["arguments"]`` (whatever they are; a non-object raises), an unknown name or any
    exception answered with str({"error": ...}) (never reaching the environment), the content str(result)."""
    if tool_call["type"] == "function":
        function = tool_call["function"]
        name = function["name"]
        try:
            if name in tool_dict:
                result = tool_dict[name](**function["arguments"])
            else:
                raise ValueError(f"Tool {name} not found.")
        except Exception as e:
            result = {"error": str(e)}
    else:
        name = tool_call.get("name", "unknown")
        result = {"error": f"Unsupported tool call type: {tool_call['type']}"}
    content = result if isinstance(result, list) else str(result)
    return {"role": "tool", "name": name, "content": content}


def trl_tools(env) -> list:
    """The environment's tools exactly as GRPOTrainer lists them (bound methods, inspect.getmembers order)."""
    return [getattr(env, name) for name, _ in inspect.getmembers(type(env), predicate=inspect.isfunction)
            if name not in ("reset", "get_reward") and not name.startswith("_")]


def grpo_template(tokenizer) -> str | None:
    """The template GRPOTrainer renders with: None (the model's own) when it is prefix-preserving, else TRL's."""
    from trl.chat_template_utils import get_training_chat_template, is_chat_template_prefix_preserving
    return None if is_chat_template_prefix_preserving(tokenizer) else get_training_chat_template(tokenizer)


def tool_suffix_ids(tokenizer, chat_template, tool_messages: list[dict]) -> list[int]:
    """GRPOTrainer._get_tool_suffix_ids (TRL 1.16.0), transcribed (text-only)."""
    dummy = [{"role": "user", "content": "dummy"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]}]
    prefix = tokenizer.apply_chat_template(dummy, add_generation_prompt=False, tokenize=True,
                                           chat_template=chat_template, return_dict=False)
    full = tokenizer.apply_chat_template(dummy + tool_messages, add_generation_prompt=True, tokenize=True,
                                         chat_template=chat_template, return_dict=False)
    eos = [i for i, t in enumerate(prefix) if t == tokenizer.eos_token_id]
    if eos:
        prefix = prefix[: eos[-1] + 1]
    if full[: len(prefix)] != prefix:
        raise ValueError("Unexpected tokenization: the EOS-trimmed prefix ids are not a prefix of the full ids.")
    return full[len(prefix):]


class Episode:
    """One rollout, following GRPOTrainer._tool_call_loop with the training hooks (training/rollout_hooks.install).
    `step` takes the next generated turn (token ids and the engine's finish reason):
      - the first turn is always parsed; a later turn that the budget left empty is skipped (TRL: no message) and ends
        the rollout;
      - a turn's tool calls are executed only while fewer than turn_cap tool blocks have run (TRL checks its iteration
        cap before executing, so after the 40th block one more turn is generated and its calls are not executed);
      - every call goes through trl_tool_message; invalid tool names are escaped before the suffix is rendered (the
        tool-name guard);
      - the suffix stops at the tool block's <|im_end|> when the environment ended (or failed) the episode in this block
        (a loop-guard stop included: the environment applies it), before the overlong check (TRL's rollback), and the
        rollout then ends. No decision matches text in a tool result."""

    def __init__(self, case: str, factory, tokenizer, chat_template, *, execution_rule: str = LOOP_GUARD,
                 budget: int = PIPELINE_MAX_COMPLETION_LENGTH, max_model_len: int = PIPELINE_MAX_CONTEXT_LENGTH,
                 turn_cap: int = PIPELINE_MAX_TOOL_CALLING_ITERATIONS, sample: int | None = None, env=None,
                 system: str = SYSTEM_PROMPT_V2, prefix: list[dict] | None = None):
        from ap_sim.training.rollout_hooks import parse_response   # TRL's, RecursionError guarded
        self.parse_response = parse_response
        self.case, self.sample, self.tok, self.template = case, sample, tokenizer, chat_template
        self.budget, self.max_model_len, self.turn_cap, self.execution_rule = budget, max_model_len, turn_cap, execution_rule
        self.env = env if env is not None else APReceiptEnvV2({case: factory}, execution_rule=execution_rule)
        first = self.env.reset(scenario=case, purpose="evaluation")
        self.tools = trl_tools(self.env)
        self.tool_dict = trl_tool_dict(self.env)
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": USER_PREFIX + first}]
        ids = tokenizer.apply_chat_template(conversation=[self.messages], tools=self.tools, chat_template=chat_template,
                                            add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"][0]
        self.seq = list(ids.tolist() if hasattr(ids, "tolist") else ids)
        self.n_prompt = len(self.seq)
        self.generated = [False] * len(self.seq)            # True on the tokens the model generated (RFT rows)
        self.turns, self.tool_turns, self.done, self.length_stop = [], 0, False, False
        self.started, self.generation_seconds = time.time(), 0.0
        if prefix is not None:
            self._force(prefix)

    def _force(self, prefix: list[dict]) -> None:
        """Start from a given conversation (the Phase 6 forced-recovery episodes: a demonstration up to and including a
        planted mistake and its result). Its calls are replayed into the environment, which must answer each exactly
        as the prefix records; the prefix's tokens are its rendering (identical to TRL's concatenation, as for the
        Phase 4 SFT demonstrations)."""
        if prefix[:2] != self.messages:
            raise ValueError("the prefix does not start with this case's prompt")
        k = 2
        while k < len(prefix):
            m = prefix[k]
            if m["role"] == "assistant" and m.get("tool_calls"):
                got = [trl_tool_message(self.tool_dict, c) for c in m["tool_calls"]]
                want = prefix[k + 1:k + 1 + len(got)]
                if [g["content"] for g in got] != [w["content"] for w in want]:
                    raise ValueError(f"the environment answers the prefix's call at message {k} differently")
                self.tool_turns += 1
                k += 1 + len(got)
            else:
                k += 1
        self.messages = [dict(m) for m in prefix]
        ids = self.tok.apply_chat_template(conversation=[self.messages], tools=self.tools, chat_template=self.template,
                                           add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"][0]
        self.seq = list(ids.tolist() if hasattr(ids, "tolist") else ids)
        self.generated = [False] * len(self.seq)            # the given prefix, the planted mistake included, is not
        self.forced = len(prefix)

    @property
    def remaining(self) -> int:
        return self.budget - (len(self.seq) - self.n_prompt)

    def _execute(self, call: dict, turn: dict) -> dict:
        """One call through trl_tool_message, with what the environment did (its own state, never the result text):
        whether the call reached it (its call count rose) and whether it accepted it (the logged step's ok)."""
        layer = self.env._layer
        before = layer.calls
        message = trl_tool_message(self.tool_dict, call)
        reached = layer.calls > before
        ok = bool(reached and layer.log["steps"] and layer.log["steps"][-1]["result"].get("ok"))
        turn["results"].append({"tool": message["name"], "chars": len(str(message["content"])), "reached": reached,
                                "accepted": ok})
        return message

    def _ended(self) -> bool:
        return self.env._layer.done or self.env._infrastructure_failure is not None

    def step(self, ids: list[int], finish_reason: str | None = None, seconds: float = 0.0) -> None:
        self.generation_seconds += seconds
        ids = list(ids[: max(self.remaining, 0)])          # TRL truncates the completion to max_completion_length
        if not ids and self.turns:                          # TRL: an empty post-tool generation adds no message
            self.turns.append({"generated_tokens": 0, "finish_reason": finish_reason, "last_token": None,
                               "skipped_empty": True, "tool_calls": [], "results": []})
            self.done = True
            return
        parsed = self.parse_response(self.tok, ids, prefix=self.seq, tools=self.tools)
        self.seq += ids
        self.generated += [True] * len(ids)
        self.messages.append(parsed)
        calls = parsed.get("tool_calls") or []
        turn = {"generated_tokens": len(ids), "finish_reason": finish_reason, "last_token": ids[-1] if ids else None,
                "tool_calls": [c.get("function", {}).get("name") if isinstance(c, dict) else None for c in calls],
                "results": []}
        self.turns.append(turn)
        self._advance(calls, turn)
        # The flags come from the turn's raw decoded text, never from the parsed fields (TRL's parser drops all but the
        # last <think> block, a tool call inside one included, from every field)
        turn.update(turn_flags(decode(self.tok, ids), ids, len(calls), len(turn["results"])))

    def _advance(self, calls: list, turn: dict) -> None:
        """The rest of TRL's tool round for a parsed turn: execute its calls, append the tool suffix, or end."""
        if len(self.seq) - self.n_prompt >= self.budget:
            self.length_stop = True
        if not calls or self.tool_turns >= self.turn_cap:
            self.done = True                                 # no call, or past TRL's iteration cap (not executed)
            return
        tool_messages = [self._execute(call, turn) for call in calls]
        suffix = tool_suffix_ids(self.tok, self.template, sanitize(tool_messages))
        ended = self._ended()                                 # environment state (a loop-guard stop ends the episode)
        if ended and IM_END in suffix:                        # rollout_hooks: through the block's <|im_end|>
            suffix = suffix[: len(suffix) - suffix[::-1].index(IM_END)]
        pct = len(self.seq) + len(suffix)
        if pct - self.n_prompt > self.budget or pct >= self.max_model_len:
            self.done = True                                 # TRL drops these tool results and ends the rollout
            turn["rolled_back"] = True
            return
        self.seq += suffix
        self.generated += [False] * len(suffix)
        self.messages.extend(tool_messages)
        self.tool_turns += 1
        if ended:
            self.done = True

    def classification(self) -> dict:
        from ap_sim.training import rewards
        length = self.length_stop if self.budget != rewards.MAX_COMPLETION_LENGTH else None   # training's criterion
        return stops.classify(self.env, self.messages[2:], self.seq[self.n_prompt:], length_stop=length)


# Scripted backends (no model): they write Qwen tool-call text, tokenized with the pinned tokenizer -------------------
def tool_call_text(name: str, arguments: dict, content: str = "") -> str:
    return (content + "\n" if content else "") + "<tool_call>\n" + json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


class ScriptedBackend:
    """Replays a fixed list of turns per episode: each turn is (text, end token) or (token ids, None)."""
    name = "scripted"

    def __init__(self, tokenizer, plans: dict):
        self.tok, self.plans = tokenizer, plans

    def info(self) -> dict:
        return {"backend": self.name}

    def generate(self, episodes) -> list[tuple[list[int], str]]:
        out = []
        for ep in episodes:
            plan = self.plans[(ep.case, ep.sample)]
            k = len(ep.turns)
            if k >= len(plan):
                out.append(([IM_END], "stop"))
                continue
            text, end = plan[k]
            ids = list(text) if end is None else self.tok.encode(text, add_special_tokens=False) + [end]
            out.append((ids, "stop"))
        return out


def reference_plan(case: str, factory, seed: int = 0) -> list[tuple[str, int]]:
    """reference/5's calls on this case (run once in its own environment: deterministic), as Qwen tool-call turns."""
    from ap_sim.agents.reference import ReferenceAgentV5
    env = APReceiptEnvV2({case: factory}, execution_rule=LOOP_GUARD)
    obs = json.loads(env.reset(scenario=case))
    calls = []

    def call(name, args):
        calls.append((name, args))
        return json.loads(execute(env, name, args))

    class Tools:
        initial_observation = obs
        click = staticmethod(lambda element_id: call("click", {"element_id": element_id}))
        fill = staticmethod(lambda element_id, text: call("fill", {"element_id": element_id, "text": text}))
        advance_time = staticmethod(lambda minutes: call("advance_time", {"minutes": minutes}))
        coverage = staticmethod(lambda invoice_id: call("coverage", {"invoice_id": invoice_id}))
        finish = staticmethod(lambda report: call("finish", {k: report[k] for k in
                                                            ("final_disposition", "discrepancy_type", "findings",
                                                             "evidence") if k in report}))

    ReferenceAgentV5(Tools(), seed=seed).run()
    return [(tool_call_text(n, a), IM_END) for n, a in calls]


def reckless_plan(case: str, factory) -> list[tuple[str, int]]:
    """The reckless fake (Gate 5 in Phase 5 of the pre-registered plan): reprocess, release, finish release with no
    evidence. On a case whose recorded receipts do not cover the invoice, the release is an unsafe attempt (refused by
    the engine, counted by g8)."""
    env = APReceiptEnvV2({case: factory}, execution_rule=LOOP_GUARD)
    obs = json.loads(env.reset(scenario=case))
    ids = {c["name"]: c["id"] for g in obs["regions"] for c in g["controls"]}
    return [(tool_call_text("click", {"element_id": ids["Reprocess invoice"]}), IM_END),
            (tool_call_text("click", {"element_id": ids["Release invoice"]}), IM_END),
            (tool_call_text("finish", {"final_disposition": "release", "discrepancy_type": "none", "findings": []}),
             IM_END)]


class VLLMBackend:
    """vLLM offline (GPU sessions): prompts as token ids, one SamplingParams per request, both end-of-turn tokens as
    stop tokens, prefix caching on (as the pre-registered plan requires). Greedy, or sampled at T 0.8 / top_p 1.0 with a
    per-request seed (pass@k: seeds 0..k-1, identical across models)."""
    name = "vllm"

    def __init__(self, model: str, revision: str | None, max_model_len: int = PIPELINE_MAX_CONTEXT_LENGTH,
                 gpu_memory_utilization: float = 0.85, temperature: float = 0.0, top_p: float = 1.0):
        from vllm import LLM
        self.temperature, self.top_p = temperature, top_p
        self.llm = LLM(model=model, revision=revision, dtype="bfloat16", max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=0)

    def info(self) -> dict:
        import torch, transformers, vllm
        return {"backend": self.name, "vllm": vllm.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__, "temperature": self.temperature, "top_p": self.top_p,
                "prefix_caching": True}

    def generate(self, episodes) -> list[tuple[list[int], str]]:
        from vllm import SamplingParams
        params = [SamplingParams(n=1, temperature=self.temperature, top_p=self.top_p, top_k=0, min_p=0.0,
                                 repetition_penalty=1.0, max_tokens=max(ep.remaining, 1),
                                 stop_token_ids=[IM_END, ENDOFTEXT],
                                 seed=None if self.temperature == 0 else ep.sample * 1_000_003 + len(ep.turns))
                  for ep in episodes]
        outs = self.llm.generate([{"prompt_token_ids": ep.seq} for ep in episodes], sampling_params=params, use_tqdm=False)
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


def run(episodes: list[Episode], backend, on_done=None) -> dict:
    """Drive the episodes in lockstep rounds until all are done; on_done(episode) is called as each one ends."""
    active, rounds, started = list(episodes), 0, time.time()
    while active:
        t0 = time.time()
        generated = backend.generate(active)
        seconds = (time.time() - t0) / max(len(active), 1)
        for ep, (ids, finish_reason) in zip(active, generated, strict=True):
            ep.step(ids, finish_reason, seconds)
            if ep.done and on_done:
                on_done(ep)
        active = [ep for ep in active if not ep.done]
        rounds += 1
    wall = time.time() - started
    return {"episodes": len(episodes), "rounds": rounds, "wall_seconds": round(wall, 1),
            "episodes_per_hour": round(len(episodes) / wall * 3600, 1) if wall else None}
