"""The environment class exposed to TRL.

Its public methods are exactly the six agent tools plus the reserved `reset`. TRL turns every other
public method into a tool, so everything else is private. It holds no grading logic: the reward comes
from `training/rewards.py`, which calls the independent grader on the episode log.

Training uses the structured surface. Scenarios are injected by the training script
(`make_environment_factory`), so this module imports no generator, grader or hidden-world code.
"""
from __future__ import annotations

import json
import os
import time
import uuid

from collections.abc import Callable, Mapping
from typing import Literal

from ap_sim.events.clock import ACTION_COST
from ap_sim.domain.report import COMPLETION_CONTRACT, evidence_json_schema, findings_json_schema
from ap_sim.interfaces.canonical import dumps
from ap_sim.training.events import event, save_episode
from ap_sim.training.loop_guard import LOOP_GUARD, NO_GUARD, NOTICE, RepeatCounter
from ap_sim.interfaces.structured import StructuredSurface
from ap_sim.interfaces.tools import ToolLayer
from ap_sim.training.config import PIPELINE_SCHEMA, PipelineConfigError

AGENT_TOOLS = ("observe", "click", "fill", "select", "advance_time", "finish")


# Tool input limits, checked on every tool's arguments before any validation, so model input can never reach a depth,
# size or number that the validators or serializers fail on. The largest arguments in Claude's 1,859 tool calls in the
# strong-model check run before training were 2,581 characters, nesting depth 5 and strings of 309 characters.
MAX_ARGUMENT_DEPTH = 16          # nested lists and objects, counting the argument list itself as level 1
MAX_ARGUMENT_CHARS = 32_000      # the arguments serialized as JSON
MAX_STRING_CHARS = 4_000         # any one string, object keys included
MAX_INTEGER = 10 ** 18           # any one integer, in absolute value
# Every string, object keys included, must be valid Unicode. TRL parses a JSON escape such as \ud800 into a lone
# surrogate, which no tokenizer can encode; echoed in a tool result it would stop the whole training step inside TRL.
# Such text is the model's own invalid argument: a rejection of the call.
NOT_UNICODE = "text that is not valid Unicode"


def _unicode(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def argument_problem(args) -> str | None:
    """What makes the arguments exceed the limits, in words that reveal nothing about the case; None within them.
    Iterative (no recursion), and it stops at the first problem, so any input is checked in bounded time."""
    stack, seen = [(args, 1)], 0
    while stack:
        value, depth = stack.pop()
        seen += 1
        if seen > MAX_ARGUMENT_CHARS:
            return f"more than {MAX_ARGUMENT_CHARS} values"
        if isinstance(value, dict):
            if depth >= MAX_ARGUMENT_DEPTH:
                return f"nesting deeper than {MAX_ARGUMENT_DEPTH} levels"
            for k, v in value.items():
                if isinstance(k, str) and len(k) > MAX_STRING_CHARS:
                    return f"a key longer than {MAX_STRING_CHARS} characters"
                if isinstance(k, str) and not _unicode(k):
                    return NOT_UNICODE
                stack.append((v, depth + 1))
        elif isinstance(value, (list, tuple)):
            if depth >= MAX_ARGUMENT_DEPTH:
                return f"nesting deeper than {MAX_ARGUMENT_DEPTH} levels"
            stack.extend((v, depth + 1) for v in value)
        elif isinstance(value, str) and len(value) > MAX_STRING_CHARS:
            return f"a string longer than {MAX_STRING_CHARS} characters"
        elif isinstance(value, str) and not _unicode(value):
            return NOT_UNICODE
        elif isinstance(value, int) and not isinstance(value, bool) and abs(value) > MAX_INTEGER:
            return "an integer out of range"
    if len(json.dumps(args, default=repr)) > MAX_ARGUMENT_CHARS:
        return f"more than {MAX_ARGUMENT_CHARS} characters in all"
    return None


def _jsonable(args: tuple) -> list:
    """A post-end call's arguments as the model sent them, kept only as far as they survive JSON (the log is JSON)."""
    return json.loads(json.dumps(list(args), default=repr))


def _check_schema(cls, schema: str | None) -> None:
    """Each class runs only the schemas its finish can serve: APReceiptEnv's finish has no
    discrepancy_type, which surfaces/4's report model requires; APReceiptEnvV2's always sends it, which the surfaces/2
    and surfaces/3 report model rejects as an unknown field."""
    from ap_sim.app.catalog import SCHEMAS
    name, other = cls.__name__, ("APReceiptEnvV2" if cls is APReceiptEnv else "APReceiptEnv")
    if schema is None:
        raise PipelineConfigError(f"{name} needs an explicit schema: one of {', '.join(map(repr, cls.SCHEMAS))} "
                                  "(v1 reproductions). The v2 pipeline uses APReceiptEnvV2 on 'surfaces/4'.")
    if schema not in SCHEMAS:
        raise PipelineConfigError(f"{name}: unknown schema {schema!r}; known: {', '.join(map(repr, SCHEMAS))}.")
    if schema not in cls.SCHEMAS:
        raise PipelineConfigError(f"{name} cannot run {schema!r}: its report model and finish tool do not match "
                                  f"(it runs {', '.join(map(repr, cls.SCHEMAS))}). Use {other}.")


class APReceiptEnv:
    """One AP receipt-investigation episode on the structured surface (version 1's six tools; surfaces/2 or surfaces/3, set
    explicitly: there is no default schema)."""

    SCHEMAS = ("surfaces/2", "surfaces/3")

    def __init__(self, scenarios: Mapping[str, Callable[[], object]] | None = None, surface_factory=None,
                 schema: str | None = None, execution_rule: str = NO_GUARD):
        _check_schema(type(self), schema)
        self._scenarios = scenarios or {}
        self._schema = schema   # the observation schema; the frozen version 1 pipeline passes surfaces/3
        self._surface_factory = surface_factory or StructuredSurface  # browser only for measurement, never training
        self._layer: ToolLayer | None = None
        self._scenario = None
        self._stop_reason: str | None = None
        self._env_seconds = 0.0  # wall time spent inside the environment (to measure its overhead)
        self._infrastructure_failure: str | None = None  # not the model's fault: excluded from the loss, counted
        assert execution_rule in (LOOP_GUARD, NO_GUARD), execution_rule
        self._execution_rule = execution_rule   # loop-guard/1 only together with loop_guard.install(trainer)
        self._repeats = RepeatCounter()
        self._loop_stop: dict | None = None
        self._post_end: list[dict] = []

    def reset(self, scenario: str = "V1", **kwargs) -> str:
        """Start a new episode for the named scenario and return the initial observation (JSON).

        The scenario name and seed come from the dataset row, never from the prompt text.
        """
        self._scenario = self._scenarios[scenario]()
        self._layer = ToolLayer(self._scenario, self._surface_factory(), self._schema)
        self._stop_reason = None
        self._env_seconds = 0.0
        self._infrastructure_failure = None
        self._repeats, self._loop_stop = RepeatCounter(), None   # never carried across episodes
        self._post_end = []
        self._episode_saved = False
        self._episode_id = uuid.uuid4().hex[:12]   # unique per episode (TRL reuses environment objects)
        event({"event": "episode_start", "episode": self._episode_id, "scenario": getattr(self._scenario, "name", scenario),
               **({"purpose": kwargs["purpose"]} if "purpose" in kwargs else {})})   # a check's reset is not a rollout
        return dumps(self._layer.initial_observation)

    # Agent tools (docstrings and type hints become the tool schemas) ------------------------------
    def observe(self) -> str:
        """Reload the current page and return it. Text typed but not yet sent (such as a quantity) is cleared.

        Returns:
            The tool result as JSON: ok, clock, error and the page observation.
        """
        return self._call("observe")

    def click(self, element_id: str) -> str:
        """Activate a link or button shown in the latest observation.

        Args:
            element_id: The opaque id of a link or button in the latest observation.

        Returns:
            The tool result as JSON: ok, clock, error and the resulting page observation.
        """
        return self._call("click", element_id)

    def fill(self, element_id: str, text: str) -> str:
        """Type text into a textbox shown in the latest observation.

        Args:
            element_id: The opaque id of a textbox in the latest observation.
            text: The text to enter (at most 100 characters).

        Returns:
            The tool result as JSON: ok, clock, error and the page observation showing the value.
        """
        return self._call("fill", element_id, text)

    def select(self, element_id: str, option: str) -> str:
        """Choose an option in a select box shown in the latest observation.

        Args:
            element_id: The opaque id of a select box in the latest observation.
            option: The visible label of the option to choose.

        Returns:
            The tool result as JSON: ok, clock, error and the page observation.
        """
        return self._call("select", element_id, option)

    def advance_time(self, minutes: int) -> str:
        """Let business time pass while waiting for others. Returns only the new time.

        Args:
            minutes: Minutes of business time to let pass, from 1 to 480.

        Returns:
            The tool result as JSON: ok, clock and error (no page content).
        """
        return self._call("advance_time", minutes)

    def finish(self, final_disposition: Literal["release", "keep_blocked", "pending"], findings: list,
               evidence: list | None = None) -> str:
        # The docstring (the agent-facing schema) is generated below from domain/report.py.
        report = {"final_disposition": final_disposition, "findings": findings}
        if evidence is not None:
            report["evidence"] = evidence
        return self._call("finish", report)

    # Private helpers (not tools) --------------------------------------------------------------------
    def _call(self, tool: str, *args) -> str:
        if self._layer is None:
            return json.dumps({"ok": False, "error": {"code": "not_started", "message": "No episode."}})
        started = time.perf_counter()
        result = None
        try:
            problem = argument_problem(args)
        except Exception:   # unreachable by construction (iterative, JSON values only); a rejection, never a crash
            problem = "arguments that cannot be read"
        if self._layer.done:
            # A call received after the episode ended (an accepted finish, the action limit, the loop guard) is an
            # attempt. The tool layer answers it episode_over and logs nothing (a hashed file), so it is recorded here,
            # with the step count and stop at that moment, for the grader (g8) to judge.
            self._post_end.append({"tool": tool, "args": [f"not recorded: {problem}"] if problem else _jsonable(args),
                                   "after_step": len(self._layer.log["steps"]), "stop": self._stop_reason})
        try:
            result = self._dispatch(tool, *args, problem=problem)
            return result
        except RecursionError:   # backstop only: the input limits make it unreachable from model input
            result = dumps(self._reject_arguments(tool, "arguments that could not be processed"))
            return result
        except Exception as exc:
            # infrastructure_failure is reserved for faults outside the model's control (the observation producer,
            # I/O, the serving layer). Model arguments are checked first (input limits, the finish pre-check) and
            # rejected as calls. A tool-code bug that a model input can trigger is fixed in code.
            if self._infrastructure_failure is None:
                self._infrastructure_failure = f"{type(exc).__name__}: {exc}"
            return json.dumps({"ok": False, "error": {"code": "infrastructure",
                                                      "message": "The environment failed; this episode is void."}})
        finally:
            self._env_seconds += time.perf_counter() - started
            # incremental event log for the evaluator (no-op unless the environment variables are set)
            event({"event": "tool", "episode": getattr(self, "_episode_id", None), "scenario": getattr(self._scenario, "name", None),
                   "tool": tool, "ok": None if result is None else json.loads(result).get("ok"),
                   "calls": self._layer.calls, "done": self._layer.done, "stop": self._stop_reason,
                   "seconds": round(time.perf_counter() - started, 4)})
            if self._layer.done and not self._episode_saved:
                self._episode_saved = True
                path = save_episode(f"episode-{getattr(self._scenario, 'name', 'x')}-{self._episode_id}",
                                    {"grading": "not yet (saved at episode end)", "termination": self._stop_reason,
                                     "execution_rule": self._execution_rule, "loop_stop": self._loop_stop,
                                     "episode": self._episode_id, "scenario": getattr(self._scenario, "name", None),
                                     "log": self._layer.episode_log()})
                event({"event": "episode_end", "episode": self._episode_id, "stop": self._stop_reason, "trace": path})

    def _reject_arguments(self, tool: str, problem: str) -> dict:
        """A call whose arguments exceed the input limits (or could not be processed): rejected and logged as a call
        (C1), like the tool layer's own malformed-argument rejections; a finish costs no time, other tools 5 minutes."""
        layer, logged = self._layer, {"arguments": f"not recorded: {problem}"}
        error = {"code": "invalid_arguments",
                 "message": f"The call was not carried out: its arguments exceed the tool input limits ({problem}). "
                            "Nothing was done."}
        if tool == "finish" and not layer.done:
            layer.calls += 1
            return layer._result(tool, logged, False, error)
        if (err := layer._begin(tool, logged, 0 if tool in ("finish", "advance_time") else ACTION_COST)) is not None:
            return err
        return layer._reject(tool, logged, error["code"], error["message"])

    def _dispatch(self, tool: str, *args, problem: str | None = None) -> str:
        was_done = self._layer.done
        if problem is not None:
            result = self._reject_arguments(tool, problem)
        elif tool == "finish":
            result = self._layer.finish(args[0])
        elif tool == "advance_time":
            result = self._layer.advance_time(args[0])
        else:
            result = getattr(self._layer, tool)(*args)
        if (self._execution_rule == LOOP_GUARD and not was_done and self._repeats.update(tool, args, result)
                and not self._layer.done):   # the action limit, reached on the same call, keeps precedence
            # the rejected call stays in the log as executed; the episode ends here, graded unchanged
            self._layer.done, self._layer.termination = True, "loop_detected"
            self._loop_stop = {"rule": LOOP_GUARD, "step": len(self._layer.log["steps"]), "tool": tool,
                               "repeats": self._repeats.count, "error": result.get("error")}
            event({"event": "loop_detected", "episode": self._episode_id, **self._loop_stop})
            result = {**result, "error": dict(NOTICE), "observation": None}
        if self._layer.done and self._stop_reason is None:
            self._stop_reason = self._layer.termination  # "finish", "action_limit" or "loop_detected"
        return dumps(result)

    def _episode(self) -> tuple[object, dict, str]:
        """For the reward function and the recorder only: (scenario, episode log, stop reason). The log carries the
        calls received after the episode ended (`post_end_calls`)."""
        reason = self._stop_reason or "no_finish"  # the model stopped calling tools, or the trainer cut it off
        return self._scenario, {**self._layer.episode_log(), "post_end_calls": list(self._post_end)}, reason


# The `finish` schema the model sees: the enum comes from the type hint; the full nested findings schema, with
# every evidence form and its required fields, is generated from the same models the validator uses.
APReceiptEnv.finish.__doc__ = f"""End the episode with a structured report. Performs no business action.

A rejected report returns ok false with the fields to fix; nothing is recorded and you may call finish
again. An accepted report ends the episode.

{chr(10).join(COMPLETION_CONTRACT)}

Args:
    final_disposition: Your final disposition for the invoice: the outcome you actually completed.
    findings: A list of findings (may be empty). JSON Schema: {json.dumps(findings_json_schema(), separators=(",", ":"))}
    evidence: Supporting evidence for the disposition, independent of findings: a list of evidence entries in the same forms as a finding's evidence. JSON Schema: {json.dumps(evidence_json_schema(), separators=(",", ":"))}

Returns:
    The tool result as JSON: ok, clock and error (with the problems to fix when not accepted).
"""


def make_environment_factory(scenarios: Mapping[str, Callable[[], object]], surface_factory=None,
                             schema: str | None = None, execution_rule: str = NO_GUARD) -> Callable[[], APReceiptEnv]:
    """The zero-argument factory TRL calls once per rollout slot (structured surface unless measuring)."""
    _check_schema(APReceiptEnv, schema)
    return lambda: APReceiptEnv(scenarios, surface_factory, schema, execution_rule)



AGENT_TOOLS_V2 = AGENT_TOOLS + ("coverage",)


class APReceiptEnvV2(APReceiptEnv):
    """The v2 environment: APReceiptEnv plus the coverage calculator, on surfaces/4 by default.
    A separate class because TRL exposes every public method as a tool: APReceiptEnv keeps exactly the six v1 tools,
    so surfaces/3 and v1 reproductions see what they saw before."""

    SCHEMAS = ("surfaces/4",)

    def __init__(self, scenarios=None, surface_factory=None, schema: str = PIPELINE_SCHEMA,
                 execution_rule: str = NO_GUARD):
        super().__init__(scenarios, surface_factory, schema, execution_rule)

    def reset(self, scenario: str = "V1", **kwargs) -> str:
        out = super().reset(scenario, **kwargs)
        self._layer.finish = self._checked_finish(self._layer)
        return out

    @staticmethod
    def _checked_finish(layer):
        """Finish rejects a derived evidence entry whose expression is malformed (outside the grader's
        arithmetic, or naming a name that is not one of its operands) as a normal, retryable report rejection that
        names the entry and says what is wrong; nothing is recorded, and like any rejected finish it counts as a call
        (C1). The check runs on a report the report model accepts; otherwise the tool layer's own rejection stands.
        The tool layer (a hashed file) is not edited: the layer's finish is wrapped per episode."""
        from pydantic import ValidationError
        from ap_sim.domain.report import report_model
        from ap_sim.grading.g8 import malformed_derived
        finish = layer.finish

        def checked(report):
            if not layer.done:
                try:   # the same validation and serialization the tool layer and the log will apply to the report
                    parsed = report_model(layer.schema).model_validate(report).model_dump(mode="json")
                    json.dumps(parsed)
                except ValidationError:
                    parsed = None    # the tool layer rejects it, naming the fields
                except Exception as exc:   # the model's argument, so a rejection, never a crash
                    layer.calls += 1
                    return layer._result("finish", {"report": "not recorded: it could not be processed"}, False, {
                        "code": "invalid_report", "problems": [{"field": "report", "problem": type(exc).__name__}],
                        "message": "The report could not be processed; fix it and call finish again. Nothing was "
                                   "recorded."})
                bad = malformed_derived(parsed) if parsed else []
                if bad:
                    layer.calls += 1
                    problems = [{"field": f"{path}[{k}].expression", "problem": p} for path, k, p in bad]
                    return layer._result("finish", {"report": report}, False, {
                        "code": "invalid_report", "problems": problems,
                        "message": f"The report was not accepted ({len(problems)} problem(s)); fix the listed fields "
                                   "and call finish again. Nothing was recorded."})
            return finish(report)
        return checked

    def coverage(self, invoice_id: str) -> str:
        """Compute the invoice's available coverage from the current records. Returns the receipts, reversals and other
        active invoices on its PO line (with record ids and versions) and the figures computed from them: net posted
        receipts, other active quantity, available coverage, the invoice quantity and the invoice quantity minus
        available coverage. Read-only; it costs 5 minutes like other tool calls. It is not a page: cite records from
        the pages where you saw them.

        Args:
            invoice_id: The id of the invoice you are resolving, as shown on the Invoice page.

        Returns:
            The tool result as JSON: ok, clock, error and coverage.
        """
        return self._call("coverage", invoice_id)

    def finish(self, final_disposition: Literal["release", "keep_blocked", "pending"],
               discrepancy_type: Literal["receipt_coverage_gap", "none"], findings: list,
               evidence: list | None = None) -> str:
        # The docstring (the agent-facing schema) is generated below, like APReceiptEnv.finish's.
        report = {"final_disposition": final_disposition, "discrepancy_type": discrepancy_type, "findings": findings}
        if evidence is not None:
            report["evidence"] = evidence
        return self._call("finish", report)


def make_environment_factory_v2(scenarios: Mapping[str, Callable[[], object]], surface_factory=None,
                                schema: str = PIPELINE_SCHEMA, execution_rule: str = NO_GUARD) -> Callable[[], APReceiptEnvV2]:
    """The zero-argument factory for the v2 environment (structured surface unless measuring)."""
    _check_schema(APReceiptEnvV2, schema)
    return lambda: APReceiptEnvV2(scenarios, surface_factory, schema, execution_rule)


# The v2 finish schema: the v1 text plus the required report-level discrepancy_type (defined by Report requirement 7
# in domain/rules.py).
from ap_sim.domain.rules import REPORT_REQUIREMENTS as _REQS  # noqa: E402

APReceiptEnvV2.finish.__doc__ = APReceiptEnv.finish.__doc__.replace(
    "    final_disposition: Your final disposition for the invoice: the outcome you actually completed.\n",
    "    final_disposition: Your final disposition for the invoice: the outcome you actually completed.\n"
    "    discrepancy_type: " + dict(_REQS)["7. Discrepancy type"] + "\n")
assert "discrepancy_type: Every report" in APReceiptEnvV2.finish.__doc__
