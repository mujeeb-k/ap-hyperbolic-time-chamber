"""Qwen base-model check: Qwen3-4B-Instruct-2507 through APReceiptEnv on the strong-model sweep's 30 cases.

Run on one A100 pod on 2026-10-30, before any training. Versions A and C ran, one greedy episode each, rendered and
parsed exactly as TRL 1.16's tool loop does.

Reference: training/rollout.py in version 1 of this environment, with TRL 1.16's own source taking precedence where they
differ (grpo_trainer.py: _tokenize_prompts, _tool_call_loop, _get_tool_suffix_ids):
  - chat template: the one GRPOTrainer uses (the model's own when prefix-preserving with tools, else TRL's training
    template; 2026-11-01, was always the training template), after add_response_schema;
  - stops: the shared classifier (ap_sim.evaluation.stops; 2026-11-01): by completion length and the environment,
    never by the last token; finish_result_dropped recorded; version V2 runs the v2 interface;
  - tools: the environment's public methods in inspect.getmembers order (alphabetical), as TRL builds _env_tools;
    rollout.py and the strong-model runner used AGENT_TOOLS order, which renders a different tool block;
  - prompt: [system, user = USER_PREFIX + reset page], rendered once with add_generation_prompt;
  - token ids are concatenated, never re-rendered: prompt + generated ids + tool suffix ids (_get_tool_suffix_ids);
  - parsing: trl.chat_template_utils.parse_response(tokenizer, ids, prefix=..., tools=...);
  - a tool call TRL can parse but not execute gets str({"error": ...}) (strong_model_check.execute, TRL's text);
  - budget: 69,632 completion tokens (v1 PIPELINE_MAX_COMPLETION_LENGTH, surfaces/3), tool results included; an
    overlong tool result is rolled back (token_budget_rollback); 40 tool turns (turn_cap);
  - conditions: surfaces/3, loop-guard/1 (version 1's final pipeline settings). A loop-guard stop ends the rollout,
    as rollout_hooks.install does.
Deviation, recorded in every episode: the episode stops as soon as the environment is done (finish, action limit).
TRL lets the model write one more turn there (no stop tool); that turn changes nothing graded.

Decoding is greedy (temperature 0 / do_sample False). Greedy paths can differ between backends and batch shapes
(bf16), so the backend, batch size and library versions are recorded per episode and backends are never mixed in a
comparison; --cases N runs the first N cases (round-robin order) for a parity subset.

Backends:
  vllm          vLLM offline LLM (colocated), prompts as token ids, one SamplingParams per request;
  transformers  plain transformers: continuous batching (generate_batch) when available, else left-padded generate;
  reference     $0 dry run: reference/4's tool calls written as Qwen <tool_call> text, tokenized with the pinned
                tokenizer and sent through the same parse, suffix, budget, dispatch and grading path (no model).

    python scripts/qwen_check.py --backend vllm
    python scripts/qwen_check.py --backend transformers [--no-continuous-batching] [--batch-size 16]
    HF_HUB_OFFLINE=1 python scripts/qwen_check.py --backend reference
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import json
import os
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import strong_model_check as smc   # noqa: E402
import strong_model_sweep as sw    # noqa: E402

from ap_sim.generator.manifests import load   # noqa: E402
from ap_sim.evaluation import rollout, stops   # noqa: E402,F401
from ap_sim.training import rewards   # noqa: E402
from ap_sim.training.config import PIPELINE_MAX_COMPLETION_LENGTH   # noqa: E402
from ap_sim.training.environment import APReceiptEnv, APReceiptEnvV2   # noqa: E402
from ap_sim.training.loop_guard import LOOP_GUARD   # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"   # v1 training/config.py MODEL_REVISION
BUDGET = smc.TOKEN_BUDGET        # 69,632
TURN_CAP = smc.TURN_CAP          # 40
VERSIONS = ("A", "C")       # the 2026-10-30 check. "V2" (the v2 interface) is also accepted
V2_BUDGET = PIPELINE_MAX_COMPLETION_LENGTH   # 28,354
OUT = smc.ROOT / "runs" / "qwen-check"


def trl_tools(env) -> list:
    """The environment tools exactly as TRL 1.16 lists them (grpo_trainer, environment_factory probe)."""
    return [getattr(env, name) for name, _ in inspect.getmembers(type(env), predicate=inspect.isfunction)
            if name not in ("reset", "get_reward") and not name.startswith("_")]


def tool_suffix_ids(tokenizer, chat_template, tool_messages: list[dict]) -> list[int]:
    """GRPOTrainer._get_tool_suffix_ids (TRL 1.16), transcribed."""
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


class Episode(rollout.Episode):
    """One rollout: the evaluation harness's rollout (ap_sim.evaluation.rollout.Episode: TRL 1.16's tool loop with the
    training hooks, the shared stop classifier) on this check's interface. Version V2 runs the v2 interface
    (APReceiptEnvV2, surfaces/4, instructions/5, the v2 budget); A and C v1's (APReceiptEnv, surfaces/3, 69,632). Both
    under loop-guard/1. Stops are labelled by the shared classifier: by completion length and the environment, never by
    the last token; an accepted finish whose result is rolled back is `finish` with finish_result_dropped (2026-11-01, when
    this check began to share the harness's rollout code)."""

    def __init__(self, case: str, version: str, factory, tokenizer, chat_template, max_model_len: int):
        self.version = version
        env = (APReceiptEnvV2({case: factory}, execution_rule=LOOP_GUARD) if version == "V2" else
               APReceiptEnv({case: factory}, schema=smc.SCHEMA, execution_rule=LOOP_GUARD))
        super().__init__(case, factory, tokenizer, chat_template, execution_rule=LOOP_GUARD,
                         budget=V2_BUDGET if version == "V2" else BUDGET, max_model_len=max_model_len,
                         turn_cap=TURN_CAP, env=env, system=smc.system_prompt(version))
        self.eos = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def step(self, ids: list[int], seconds: float = 0.0) -> None:   # the backends here return ids only
        super().step(ids, None, seconds)

    def record(self, backend_info: dict) -> dict:
        from ap_sim.grading import g8
        c = self.classification()
        stop = c["stop_reason"]
        scenario, log, _ = self.env._episode()
        verdict = g8.grade(scenario, log) if self.version == "V2" else None
        return {"config": {"case": self.case, "version": self.version, "model": MODEL, "revision": REVISION,
                           "decoding": "greedy", "budget": self.budget, "turn_cap": TURN_CAP,
                           "schema": "surfaces/4" if self.version == "V2" else smc.SCHEMA,
                           "execution_rule": LOOP_GUARD, "system_sha256": smc.sha(self.messages[0]["content"]),
                           "tool_order": [t.__name__ for t in self.tools], "deviation": "stops when the environment is done",
                           **backend_info},
                "verdict": smc.verdict_record(self.env, stop),
                # the training reward of the same rollout: verdict.reward above is the v1-style
                # graded figure (the verdict with exclusions), which is not the -1/0/+1 training reward
                "training_reward": None if verdict is None else rewards.training_reward(verdict, stop, scenario, log),
                "finish_result_dropped": c["finish_result_dropped"],
                "tool_calls": self.env._layer.calls, "tool_turns": self.tool_turns,
                "malformed_tool_calls": c["malformed_tool_calls"],
                "prompt_tokens": self.n_prompt, "completion_tokens": len(self.seq) - self.n_prompt,
                "generation_seconds": round(self.generation_seconds, 2),
                "wall_seconds": round(time.time() - self.started, 1), "cost_usd": 0.0,
                "turns": self.turns, "messages": self.messages, "episode_log": self.env._episode()[1]}


# Backends -----------------------------------------------------------------------------------------------------------
END_TOKENS = (151645, 151643)   # <|im_end|>, <|endoftext|>: both end a turn, as in training


def cut_at_end(row: list[int]) -> list[int]:
    """A generated row up to and including its first end-of-turn token (padding and anything after it dropped)."""
    k = next((i for i, t in enumerate(row) if t in END_TOKENS), None)
    return row if k is None else row[: k + 1]

class ReferenceBackend:
    """$0: replays reference/4's tool calls for each episode as Qwen <tool_call> text (no model)."""
    name = "reference"

    def __init__(self, tokenizer, episodes):
        from ap_sim.agents.reference import ReferenceAgentV4
        self.tok, self.plans = tokenizer, {}
        for ep in episodes:   # the same case in a separate environment: deterministic, so the calls replay exactly
            if ep.version == "V2":   # reference/5 on the v2 interface
                from ap_sim.evaluation.rollout import reference_plan
                self.plans[(ep.case, ep.version)] = [{"text": t} for t, _ in
                                                     reference_plan(ep.case, load(smc.MANIFEST)[ep.case])]
                continue
            env = APReceiptEnv({ep.case: load(smc.MANIFEST)[ep.case]}, schema=smc.SCHEMA, execution_rule=LOOP_GUARD)
            obs = json.loads(env.reset(scenario=ep.case))
            calls = []

            def call(name, args, env=env, calls=calls):
                calls.append({"name": name, "arguments": args})
                return json.loads(smc.execute(env, name, args))

            class Tools:
                initial_observation = obs
                click = staticmethod(lambda element_id: call("click", {"element_id": element_id}))
                fill = staticmethod(lambda element_id, text: call("fill", {"element_id": element_id, "text": text}))
                advance_time = staticmethod(lambda minutes: call("advance_time", {"minutes": minutes}))
                finish = staticmethod(lambda report: call("finish", report))

            ReferenceAgentV4(Tools()).run()
            self.plans[(ep.case, ep.version)] = calls

    def info(self) -> dict:
        return {"backend": self.name}

    def generate(self, episodes) -> list[list[int]]:
        out = []
        for ep in episodes:
            c = self.plans[(ep.case, ep.version)][ep.tool_turns]
            text = c["text"] if "text" in c else "<tool_call>\n" + json.dumps(c, ensure_ascii=False) + "\n</tool_call>"
            out.append(self.tok.encode(text, add_special_tokens=False) + [ep.eos])
        return out


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id, revision, tokenizer, continuous: bool, batch_size: int, local_only: bool):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch, self.tok, self.batch_size = torch, tokenizer, batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                          local_files_only=local_only).to(device).eval()
        self.continuous = continuous and hasattr(self.model, "generate_batch")
        self.mode = "continuous_batching" if self.continuous else "padded_generate"

    def info(self) -> dict:
        import torch, transformers
        return {"backend": self.name, "mode": self.mode, "batch_size": self.batch_size, "torch": torch.__version__,
                "transformers": transformers.__version__, "device": str(self.model.device)}

    def _config(self, max_new_tokens: int):
        from transformers import GenerationConfig
        return GenerationConfig(do_sample=False, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new_tokens,
                                pad_token_id=self.tok.pad_token_id, eos_token_id=list(END_TOKENS))

    def generate(self, episodes) -> list[list[int]]:
        out = []
        for i in range(0, len(episodes), self.batch_size):
            chunk = episodes[i:i + self.batch_size]
            cfg = self._config(max(ep.remaining for ep in chunk))
            with self.torch.no_grad():
                if self.continuous:
                    res = self.model.generate_batch([ep.seq for ep in chunk], generation_config=cfg, progress_bar=False)
                    out += [cut_at_end(list(r.generated_tokens)) for r in res.values()]
                else:   # TRL's regular path: left-padded batch
                    width = max(len(ep.seq) for ep in chunk)
                    pad = self.tok.pad_token_id
                    ids = self.torch.tensor([[pad] * (width - len(ep.seq)) + ep.seq for ep in chunk], device=self.model.device)
                    mask = self.torch.tensor([[0] * (width - len(ep.seq)) + [1] * len(ep.seq) for ep in chunk],
                                             device=self.model.device)
                    gen = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
                    for row in gen[:, width:].tolist():
                        out.append(cut_at_end(row))   # the turn ends at its first end token; nothing after it
        return out


class VLLMBackend:
    name = "vllm"

    def __init__(self, model_id, revision, max_model_len: int, gpu_memory_utilization: float):
        from vllm import LLM
        self.llm = LLM(model=model_id, revision=revision, dtype="bfloat16", max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=0)

    def info(self) -> dict:
        import torch, transformers, vllm
        return {"backend": self.name, "vllm": vllm.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__}

    def generate(self, episodes) -> list[list[int]]:
        from vllm import SamplingParams
        params = [SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                                 max_tokens=max(ep.remaining, 1), stop_token_ids=list(END_TOKENS)) for ep in episodes]
        outs = self.llm.generate([{"prompt_token_ids": ep.seq} for ep in episodes], sampling_params=params, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]


# Driver -------------------------------------------------------------------------------------------------------------
def main() -> None:
    global BUDGET
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=("vllm", "transformers", "reference"), required=True)
    ap.add_argument("--versions", default="A,C")
    ap.add_argument("--cases", type=int, default=30, help="first N cases of the sweep's round-robin list")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--tokenizer", default=MODEL, help="smoke tests only (a tiny model with the pinned tokenizer)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-continuous-batching", action="store_true")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=int, default=BUDGET, help="smoke tests only; the check uses 69,632")
    a = ap.parse_args()
    BUDGET = a.budget
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema

    local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    revision = a.revision if a.model == MODEL else None
    tokenizer = add_response_schema(AutoTokenizer.from_pretrained(
        a.tokenizer, revision=REVISION if a.tokenizer == MODEL else None, local_files_only=local_only))
    from ap_sim.evaluation.rollout import grpo_template
    template = grpo_template(tokenizer)   # as GRPOTrainer chooses it (2026-11-01; was always the training template)
    cases = sw.select_cases()[: a.cases]
    versions = tuple(a.versions.split(","))
    scenarios = load(smc.MANIFEST)
    max_model_len = 8192 + BUDGET   # the longest prompt (about 4.0k tokens for A, 5.2k for C) plus the whole budget
    episodes = [Episode(case, v, scenarios[case], tokenizer, template, max_model_len) for case, _ in cases for v in versions]
    if a.backend == "reference":
        backend = ReferenceBackend(tokenizer, episodes)
    elif a.backend == "transformers":
        backend = TransformersBackend(a.model, revision, tokenizer, not a.no_continuous_batching, a.batch_size, local_only)
    else:
        backend = VLLMBackend(a.model, revision, max_model_len, a.gpu_memory_utilization)
    tag = a.backend if a.backend != "transformers" else f"transformers-{backend.mode}"
    folder = pathlib.Path(a.out) if a.out else OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "sweep.jsonl"
    log.write_text("")
    info = {**backend.info(), "python": platform.python_version(), "host": platform.node(),
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    variant = dict(cases)
    started, rounds = time.time(), 0
    active = list(episodes)
    while active:
        t0 = time.time()
        generated = backend.generate(active)
        seconds = (time.time() - t0) / max(len(active), 1)
        for ep, ids in zip(active, generated, strict=True):
            ep.step(ids, seconds)
            if ep.done:   # written to disk the moment it ends
                rec = ep.record(info)
                path = folder / f"{ep.case}-{ep.version}.json"
                path.write_text(json.dumps(rec, indent=1, default=str))
                v = rec["verdict"]
                with log.open("a") as f:
                    f.write(json.dumps({"case": ep.case, "variant": variant[ep.case], "version": ep.version,
                                        "cost_usd": 0.0, "stop_reason": v["stop_reason"], "reward": v["reward"],
                                        "expected": v["expected"], "reached": v["reached"], "critical": v["critical"],
                                        "failure_signature": None, "backoffs": 0, "file": path.name}) + "\n")
        active = [ep for ep in active if not ep.done]
        rounds += 1
        print(f"round {rounds}: {len(active)} active, {time.time() - started:.0f} s", flush=True)
    wall = time.time() - started
    summary = {**info, "episodes": len(episodes), "rounds": rounds, "wall_seconds": round(wall, 1),
               "episodes_per_hour": round(len(episodes) / wall * 3600, 1) if wall else None}
    (folder / "timing.json").write_text(json.dumps(summary, indent=1))
    result = sw.report(False, folder=folder, versions=versions)
    print(json.dumps(summary, indent=1))
    return result


if __name__ == "__main__":
    main()
