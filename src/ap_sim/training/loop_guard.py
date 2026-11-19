"""Execution rule loop-guard/1: end an episode that repeats one rejected request.

Repetition is the same tool, the same normalized arguments (canonical JSON) and the same rejection (the complete
error object the tool returned), on consecutive calls. A request that differs in any of the three restarts the count at
one; an accepted call resets it to zero. The fifth such call ends the episode with stop reason `loop_detected`: a
model-caused failure, graded by the unchanged grader (no accepted report, so reward 0), kept in every denominator, never
excluded as infrastructure. It is not by itself an unsafe attempt: the grader's safety findings are unchanged.

Two parts, always installed together, for training and both comparison arms alike:
  - `RepeatCounter`, held by the environment (one per episode; reset with the episode);
  - the trainer hook: TRL 1.16's tool loop has no environment-side stop. In v2 it is
    ap_sim.training.rollout_hooks.install (an ended episode, a loop-guard stop included, ends its tool block at the
    <|im_end|> and generates nothing more, decided from the environment's state); this module's own `install`, which
    matched the notice's text in the tool messages, is retired.

This is not a runtime bound: a rollout that varies its request, or repeats accepted calls, is limited only by the
existing action limit, the token budget and the hard session shutdown.
"""
from __future__ import annotations

import hashlib
import json

from ap_sim.interfaces.canonical import dumps

LOOP_GUARD = "loop-guard/1"
NO_GUARD = "none"            # the version 1 screening run: no loop guard
EXECUTION_RULES = (LOOP_GUARD, NO_GUARD)
THRESHOLD = 5
STOP_CODE = "loop_detected"
STOP_MARK = f'"code":"{STOP_CODE}"'   # in the canonical tool result; only the environment writes tool results
NOTICE = {"code": STOP_CODE,
          "message": f"The same request was rejected {THRESHOLD} times in a row with the same error. The episode has ended."}


class RepeatCounter:
    """Consecutive identical rejected requests within one episode."""

    def __init__(self, threshold: int = THRESHOLD):
        self.threshold, self.key, self.count = threshold, None, 0

    def update(self, tool: str, args: tuple, result: dict) -> bool:
        """Record one executed call and its result; True when this call is the threshold-th repetition."""
        if result.get("ok"):
            self.key, self.count = None, 0
            return False
        error = result.get("error") or {}
        rejected = error.get("code") == "invalid_arguments" and LIMITS_REJECTION in str(error.get("message", ""))
        key = (tool, fingerprint(list(args)) if rejected else dumps(list(args)), dumps(result.get("error")))
        self.count = self.count + 1 if key == self.key else 1
        self.key = key
        return self.count >= self.threshold


# The environment's rejection of arguments past its input limits (environment._reject_arguments): such arguments may be
# nested up to the parser's cap (4,096), and are never serialized here.
LIMITS_REJECTION = "its arguments exceed the tool input limits"


def fingerprint(value) -> str:
    """A digest of the value's canonical JSON form (sorted keys), built iteratively: equal for equal values, so a
    repeated rejected request is still a repetition; no recursion and no serialization of the nested value."""
    h, stack = hashlib.sha256(), [value]
    while stack:
        v = stack.pop()
        if isinstance(v, _Mark):
            h.update(v.text)
        elif isinstance(v, dict):
            h.update(b"{")
            stack.append(_Mark(b"}"))
            for k in sorted(v, key=str, reverse=True):
                stack += [_Mark(b","), v[k], _Mark(json.dumps(str(k), ensure_ascii=False).encode() + b":")]
        elif isinstance(v, (list, tuple)):
            h.update(b"[")
            stack.append(_Mark(b"]"))
            for x in reversed(v):
                stack += [_Mark(b","), x]
        else:
            h.update(json.dumps(v, ensure_ascii=False, default=repr).encode())
    return h.hexdigest()


class _Mark:
    __slots__ = ("text",)

    def __init__(self, text: bytes):
        self.text = text


def install(trainer) -> None:
    """Retired, because no control flow may depend on model-influenced text. This hook found the loop notice by
    matching text in the tool messages, which a model can write into TRL's own error text, and returned None logprobs
    where TRL's vLLM path indexes them. Use ap_sim.training.rollout_hooks.install, which decides from the
    environment's state and keeps TRL's output shapes."""
    raise RuntimeError("loop_guard.install is retired: use ap_sim.training.rollout_hooks.install (round 7, F1/F5)")
