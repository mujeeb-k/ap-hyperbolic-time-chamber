"""The pipeline check every GPU entry point makes first (added on 2026-11-01, after a documentation audit).

The settings this code would actually use, read from the code that uses them, must equal the v2 pipeline's
(ap_sim.training.config.require_pipeline), or the entry point refuses to start.

The evaluation harness (`scripts/phase5/evaluate.py`) and the Phase 6 gate runner make this check first themselves (since
2026-11-02), so a direct call is guarded too.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu")]


def effective() -> dict[str, dict]:
    """The settings each path would run with, by path: training (GRPOConfig's values, the reward's token limit and turn
    cap) and the evaluation harness (Episode's defaults); schema from the environment, instructions from the prompt."""
    import training
    from ap_sim.evaluation import rollout, stops  # noqa: F401  (stops sets rewards.TURN_CAP, as every runner imports it)
    from ap_sim.training import prompt, rewards
    from ap_sim.training.config import PIPELINE_MAX_TOOL_CALLING_ITERATIONS
    from ap_sim.training.environment import APReceiptEnvV2
    schema = inspect.signature(APReceiptEnvV2.__init__).parameters["schema"].default
    grpo = training.grpo_args(pathlib.Path("unused"), 1e-6, 1, cuda=False)
    episode = inspect.signature(rollout.Episode.__init__).parameters
    common = {"schema": schema, "instructions": prompt.INSTRUCTIONS_VERSION_V2}
    return {"training": {**common, "max_completion_length": grpo["max_completion_length"],
                         "max_tool_calling_iterations": grpo["max_tool_calling_iterations"]},
            "reward": {**common, "max_completion_length": rewards.MAX_COMPLETION_LENGTH,
                       # None means not set yet: stops.classify and build_grpo set it, and refuse only another value
                       "max_tool_calling_iterations": PIPELINE_MAX_TOOL_CALLING_ITERATIONS if rewards.TURN_CAP is None
                       else rewards.TURN_CAP},
            "harness": {**common, "max_completion_length": episode["budget"].default,
                        "max_tool_calling_iterations": episode["turn_cap"].default}}


def require() -> None:
    """Refuse to start (PipelineConfigError) unless every path's settings are the pipeline's."""
    from ap_sim.training.config import PipelineConfigError, require_pipeline
    for path, settings in effective().items():
        try:
            require_pipeline(**settings)
        except PipelineConfigError as e:
            raise PipelineConfigError(f"{path}: {e}") from None
