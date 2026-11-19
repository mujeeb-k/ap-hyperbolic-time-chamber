"""The one stop classifier every runner uses (added on 2026-11-01 with the evaluation harness): training
(rewards.ap_reward_v2), the evaluation harness (evaluation/rollout.py), scripts/qwen_check.py and the Claude reference
row.

Training computes its stop inside ap_reward_v2 from rewards.stop_reason and rewards.finish_result_dropped (rewards.py
is hashed in the g8 record, so it is not edited). `classify` is that computation, unchanged and in the same order, so
the other runners call exactly the code training calls. A parity test checked that they agree on the same episode.
"""
from __future__ import annotations

from ap_sim.training import rewards
from ap_sim.training.config import PIPELINE_MAX_TOOL_CALLING_ITERATIONS

# rewards.py says the training script sets TURN_CAP to max_tool_calling_iterations, but at first none did.
# Every runner imports this module, and scripts/gpu/training.py sets it too; classify sets it when unset and refuses any
# other value.
if rewards.TURN_CAP is None:
    rewards.TURN_CAP = PIPELINE_MAX_TOOL_CALLING_ITERATIONS


def classify(env, completion: list[dict] | None, completion_ids: list[int] | None = None,
             length_stop: bool | None = None) -> dict:
    """The stop reason, finish_result_dropped and the malformed-call count of one finished rollout, as training
    computes them. completion: the messages after the prompt (assistant turns and the tool messages delivered);
    completion_ids: the tokens after the prompt; length_stop: the engine reported a length stop (runners whose budget
    differs from training's pass it instead of completion_ids)."""
    if rewards.TURN_CAP is None:
        rewards.TURN_CAP = PIPELINE_MAX_TOOL_CALLING_ITERATIONS
    if rewards.TURN_CAP != PIPELINE_MAX_TOOL_CALLING_ITERATIONS:
        raise RuntimeError(f"rewards.TURN_CAP is {rewards.TURN_CAP}, not the pipeline's {PIPELINE_MAX_TOOL_CALLING_ITERATIONS}")
    _, log, env_reason = env._episode()
    reason = ("infrastructure_failure" if env._infrastructure_failure else
              rewards.stop_reason(env_reason, completion, completion_ids, executed_calls=len(log["steps"]),
                                  length_stop=length_stop))
    return {"stop_reason": reason, "finish_result_dropped": rewards.finish_result_dropped(log, completion),
            "malformed_tool_calls": rewards.malformed_tool_calls(completion)}
