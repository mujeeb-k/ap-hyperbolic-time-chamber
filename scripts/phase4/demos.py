"""Phase 4, the SFT demonstrations (sft-demos/v2-1), built from reference/5 episodes on the frozen sft split.

Each demonstration is a reference/5 episode driven through APReceiptEnvV2 exactly as TRL 1.16's tool loop drives it
(strong_model_check.execute: the environment's own method with keyword arguments; an exception there becomes TRL's
str({"error": ...}) tool message and never reaches the environment). The recorder between reference/5 and the
environment adds what reference/5 does not write:

  - reasoning text: 1-3 sentences in the assistant message before each tool call. `narrate` builds it from the
    messages already in the conversation (the reset page and earlier tool results) and the call being made, nothing
    else: no case id, label, split field or hidden-world value reaches it, so counterfactual twins get identical text
    until a tool result differs (the twin check);
  - at most one planted mistake (plan), followed by the recovery:
      malformed      a call with a wrong argument name (it raises inside TRL: not an attempt; finish without
                     discrepancy_type is one) or a wrong argument type TRL's parser keeps (it reaches the environment,
                     which rejects it); the recovery repeats the call correctly;
      report_defect  a finish the environment rejects as invalid_report (discrepancy_type null, a missing required
                     field, a derived expression naming an undefined operand, a derived value given as a number);
                     the recovery corrects that field. Defects finish accepts and g8 then fails (a wrong source_id, a
                     value given as a bare string, a missing citation) cannot be recovered from and are not used;
      stale          after a wait during which the state changed, a click on "Reprocess invoice" (safe: g8 judges only
                     Release invoice and Post goods receipt as unsafe controls) without observing; the tool layer
                     refuses it as stale_state; the recovery observes, which reloads the Delivery confirmations page,
                     and replaces reference/5's own click on that page.
    Never planted: an unsafe attempt, a disposition later superseded, an unnecessary confirmation.
  - a closing message after the accepted finish; no call after it.

Masking (fixed 2026-11-01, before any training): explicit per-token labels in a pre-tokenized row {input_ids, labels}.
Every assistant turn is trained, before and after the planted mistake, except the planted mistake turn; no system, user
or tool token is trained. TRL 1.16.0's SFTTrainer uses such labels as they are.

Recoveries (fixed 2026-11-01): 135 of the 300, 45 of each kind, allocated with ALLOCATION_SEED (`allocate`).

The identifier check imports EVALUATOR_ONLY from a test module that is not part of this release, and the demonstrations
themselves are not released, so the script does not run from the release alone.

    .venv/bin/python scripts/phase4/demos.py review       # the five REVIEW demonstrations, with fixed plans
    .venv/bin/python scripts/phase4/demos.py all [DIR]    # all 300 and the acceptance summary (default runs/phase4/...)
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import pathlib
import random
import re
import sys
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/phase2"), str(ROOT / "scripts/phase3"),
                str(ROOT / "scripts/step5")]

import strong_model_check as smc   # noqa: E402

from ap_sim.agents.reference import ReferenceAgentV5   # noqa: E402
from ap_sim.events.clock import HORIZON, format_time, parse_time   # noqa: E402
from ap_sim.training.environment import APReceiptEnvV2   # noqa: E402
from ap_sim.training.loop_guard import NO_GUARD   # noqa: E402
from ap_sim.training.prompt import SYSTEM_PROMPT_V2, USER_PREFIX   # noqa: E402

DATASET_VERSION = "sft-demos/v2-1"
MISTAKES = ("malformed", "report_defect", "stale")
DEFECTS = ("discrepancy_type_null", "missing_state_version", "undefined_operand", "numeric_derived_value")
FINAL_CHECK = "D5 16:35"   # Workflow rule 8


# --- what the conversation has shown so far ------------------------------------------------------------------------
def records(obs: dict | None, role: str) -> list[dict]:
    return [r for g in (obs or {}).get("regions", []) if g["role"] == role for r in g["records"]]


def q(rec: dict, field: str = "quantity") -> Decimal:
    return Decimal(rec["fields"][field]["value"])


def ea(x: Decimal) -> str:
    return f"{x} EA"


class Seen:
    """Everything the agent has been shown, read from the conversation alone (system, user, then the turns)."""

    def __init__(self, messages: list[dict]):
        self.first = json.loads(messages[1]["content"][len(USER_PREFIX):])
        self.pages, self.coverage, self.clock = {self.first["page"]: self.first}, None, self.first["clock"]
        self.controls = {c["id"]: c["name"] for g in self.first["regions"] for c in g["controls"]}
        self.last_tool, self.last_result, self.results, self.current = None, None, [], self.first
        self.calls = []
        for m in messages[2:]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                f = m["tool_calls"][0]["function"]
                self.calls.append((f["name"], f["arguments"]))
            if m["role"] != "tool":
                continue
            self.last_tool = m["name"]
            if m["content"].startswith("{'error'"):          # TRL's text for a call that raised
                self.last_result = {"trl_error": ast.literal_eval(m["content"])["error"]}
                continue
            r = json.loads(m["content"])
            self.last_result = r
            self.clock = r.get("clock") or self.clock
            if r.get("coverage"):
                self.coverage = r["coverage"]
            o = r.get("observation")
            if o:
                self.current = o
                self.pages[o["page"]] = o
                self.controls.update({c["id"]: c["name"] for g in o["regions"] for c in g["controls"]})
                self.results += [x["fields"] for x in records(o, "result")]

    # The case as the agent sees it -----------------------------------------------------------------------------
    @property
    def invoice(self) -> dict:
        return records(self.first, "invoice_header")[0]

    @property
    def line(self) -> dict:
        return records(self.first, "invoice_lines")[0]

    @property
    def po_line(self) -> str:
        return self.line["fields"]["po_line"]

    def receipts(self) -> list[dict]:
        return [r for r in records(self.pages.get("receipts"), "receipts") if r["fields"]["po_line"] == self.po_line]

    def net(self) -> Decimal:
        return sum((q(r) if r["fields"]["movement"] == "receipt" else -q(r) for r in self.receipts()), Decimal(0))

    def others(self) -> list[dict]:
        return [r for r in records(self.pages.get("po_line_invoices"), "po_line_invoices")
                if r["id"] != self.invoice["id"] and r["fields"]["state"] != "cancelled"]

    def cancelled(self) -> list[dict]:
        return [r for r in records(self.pages.get("po_line_invoices"), "po_line_invoices")
                if r["id"] != self.invoice["id"] and r["fields"]["state"] == "cancelled"]

    def figures(self) -> tuple[Decimal, Decimal, Decimal]:
        """(available coverage, invoice quantity, invoice quantity minus coverage) from the calculator's result."""
        c = self.coverage
        return (Decimal(c["available_coverage"]["value"]), Decimal(c["invoice_quantity"]["value"]),
                Decimal(c["invoice_quantity_minus_available_coverage"]["value"]))

    def tolerance(self) -> dict:
        return next(r for r in records(self.pages["policy"], "policy_rules")
                    if r["fields"]["rule"].startswith("Quantity tolerance"))["fields"]

    def confirmation(self) -> dict | None:
        """The latest confirmation request as last shown on the Delivery confirmations page."""
        conf = [r for r in records(self.pages.get("confirmations"), "confirmations")
                if r["fields"]["po_line"] == self.po_line]
        answered = [r for r in conf if r["fields"]["status"] == "answered"]
        if answered:
            return max(answered, key=lambda r: (r["fields"]["answered_at"], r["id"]))
        return conf[-1] if conf else None

    def result(self, action: str) -> dict | None:
        return next((r for r in reversed(self.results) if r["action"] == action), None)


# --- reasoning text --------------------------------------------------------------------------------------------------
def _receipts_sentence(s: Seen) -> str:
    rs = s.receipts()
    terms = " + ".join(str(q(r)) if r["fields"]["movement"] == "receipt" else f"(reversal) -{q(r)}" for r in rs)
    if not rs:
        return f"Receipt history shows no receipts on {s.po_line}: net posted receipts 0 EA."
    many = len(rs) > 1
    return (f"Receipt history shows {'receipts' if many else 'one receipt'} on {s.po_line}"
            f"{' of ' + terms + ' EA' if many else ' of ' + terms + ' EA'}: net posted receipts {ea(s.net())}.")


def _invoices_sentence(s: Seen) -> str:
    oth, can = s.others(), s.cancelled()
    head = ("Other active invoices on the PO line: " + ", ".join(f"{r['id']} for {ea(q(r))}" for r in oth)
            if oth else "No other active invoice on the PO line")
    if len(can) == 1:
        head += f"; {can[0]['id']} is cancelled and does not count"
    elif can:
        head += "; " + ", ".join(r["id"] for r in can) + " are cancelled and do not count"
    return head + "."


def _intro(s: Seen) -> str:
    failing = [r["fields"]["comparison"] for r in records(s.first, "blocking_comparisons")
               if r["fields"]["effect"] == "blocking" and r["fields"]["result"] == "fail"]
    return (f"Invoice {s.invoice['id']} for {ea(q(s.line))} on PO line {s.po_line} is blocked: the "
            f"{' and '.join(failing)} comparison failed when it was evaluated at {s.invoice['fields']['evaluated_at']}.")


def narrate(messages: list[dict], name: str, args: dict) -> str:
    """The reasoning before a call: built from the conversation so far and the call itself, nothing else."""
    s = Seen(messages)
    target = s.controls.get(args.get("element_id")) if name in ("click", "fill") else None
    if name == "click" and target in ("Receipt history", "Invoices on PO line"):
        if "receipts" not in s.pages and "po_line_invoices" not in s.pages:
            return _intro(s) + " The block may be stale, so check it against the current records, starting with " + \
                ("the receipts on the PO line." if target == "Receipt history" else "the other invoices on the PO line.")
        done = _receipts_sentence(s) if "receipts" in s.pages else _invoices_sentence(s)
        return done + (" Next, the receipts on the PO line." if target == "Receipt history"
                       else " Next, the other invoices on the PO line.")
    if name == "coverage":
        last = _receipts_sentence(s) if s.calls[-1:] and s.controls.get(s.calls[-1][1].get("element_id")) == \
            "Receipt history" else _invoices_sentence(s)
        oth = sum((q(r) for r in s.others()), Decimal(0))
        cover, qty = s.net() - oth, q(s.line)
        verdict = (f"covers the invoice quantity of {ea(qty)}" if qty <= cover
                   else f"is {ea(qty - cover)} short of the invoice quantity of {ea(qty)}")
        return (f"{last} By my count, available coverage is {s.net()} - {oth} = {ea(cover)}, which {verdict}. "
                f"Check the figures with the coverage tool.")
    if s.last_tool == "coverage" and name == "click":
        cover, qty, gap = s.figures()
        if gap <= 0:
            assert target == "Reprocess invoice", target
            return (f"The calculator agrees: available coverage {ea(cover)} against the invoice quantity {ea(qty)}, so "
                    "the invoice-vs-receipt rule is met on the current records and the block is stale. Reprocess the "
                    "invoice so it is evaluated again.")
        assert target == "Active policy", target
        return (f"The calculator agrees: the invoice quantity {ea(qty)} exceeds available coverage {ea(cover)} by "
                f"{ea(gap)}. Check the Active policy page for a quantity tolerance before acting on the gap.")
    if target == "Request delivery confirmation":
        tol, (cover, qty, gap) = s.tolerance(), s.figures()
        if tol["effect"] != "disabled":
            raise NotImplementedError("a quantity tolerance in effect (not in the sft split)")
        return (f"The quantity tolerance on invoice vs receipt is disabled, so the gap of {ea(gap)} stands. The records "
                "cannot tell whether the goods arrived unrecorded or never arrived: request a delivery confirmation "
                "and wait for the answer.")
    if target == "Delivery confirmations":
        req = s.result("request_confirmation")
        if s.last_tool == "click" and s.controls.get(s.calls[-1][1].get("element_id")) == "Request delivery confirmation":
            return f"{req['message']} Open the Delivery confirmations page to follow it."
        if s.last_tool == "advance_time":
            return f"It is now {s.clock}. Check the Delivery confirmations page for an answer."
        c, now = s.confirmation(), parse_time(s.clock)
        assert now >= parse_time(FINAL_CHECK), s.clock     # reference/5 polls only after waiting to D5 16:30
        return (f"Request {c['id']} is still {c['fields']['status']} at {s.clock}. Another check, done by "
                f"{format_time(now + 5)}, still leaves time to keep the invoice blocked before the horizon "
                f"{format_time(HORIZON)}, and an answer may still arrive, so check the page again.")
    if name == "advance_time":
        c, until = s.confirmation(), format_time(parse_time(s.clock) + args["minutes"])
        return (f"Request {c['id']} is still {c['fields']['status']} at {s.clock}, with no answer yet. Wait "
                f"{args['minutes']} minutes, until {until}, then check the page again.")
    if name == "fill":
        c, (cover, qty, gap) = s.confirmation(), s.figures()
        d = q(c, "delivered_total")
        return (f"Request {c['id']} is answered: delivered total {ea(d)} against net posted receipts {ea(s.net())}, so "
                f"{ea(d - s.net())} were delivered but not recorded, which covers the gap of {ea(gap)}. Enter the full "
                f"unrecorded quantity, {d - s.net()}, for the receipt posting.")
    if target == "Request receipt posting":
        return f"Quantity {args_fill(s)} entered. Ask receiving to post the receipt."
    if target == "Reprocess invoice":
        post = s.result("request_receipt_posting")
        cover, qty, gap = s.figures()
        added = Decimal(args_fill(s))
        return (f"{post['message']} Coverage is now {cover} + {added} = {ea(cover + added)} against the invoice "
                f"quantity {ea(qty)}. Reprocess the invoice so it is evaluated on the current records.")
    if target == "Release invoice":
        res = s.result("reprocess")
        comps = records(s.current, "changed:blocking_comparisons")
        assert comps and all(r["fields"]["result"] == "pass" for r in comps if r["fields"]["effect"] == "blocking")
        return f"{res['message']} Every blocking comparison now passes, so release the invoice."
    if target == "Keep blocked: shortfall correction":
        c, (cover, qty, gap) = s.confirmation(), s.figures()
        d = q(c, "delivered_total")
        unrec = d - s.net()
        return (f"Request {c['id']} is answered: delivered total {ea(d)} against net posted receipts {ea(s.net())}, so "
                f"{ea(max(unrec, Decimal(0)))} {'is' if unrec == 1 else 'are'} unrecorded, less than the gap of "
                f"{ea(gap)}. The shortfall is confirmed: keep the invoice blocked for shortfall correction.")
    if target == "Keep blocked: confirmation pending":
        c = s.confirmation()
        assert parse_time(s.clock) >= parse_time(FINAL_CHECK) and s.current["page"] == "confirmations"
        return (f"My check of the Delivery confirmations page at {s.clock}, no earlier than {FINAL_CHECK}, still shows "
                f"request {c['id']} {c['fields']['status']} with no answer, and the horizon is {format_time(HORIZON)}: "
                "no answer can arrive in time to finish a resolution. Keep the invoice blocked with the confirmation "
                "pending.")
    if name == "finish":
        cover = Decimal(s.coverage["available_coverage"]["value"])
        qty = q(s.line)
        kind = args.get("discrepancy_type")
        why = (f"at the start the invoice quantity {ea(qty)} exceeded available coverage {ea(cover)}" if kind ==
               "receipt_coverage_gap" else f"at the start available coverage {ea(cover)} already covered the invoice "
               f"quantity {ea(qty)}")
        done = {"release": "The invoice is released", "keep_blocked": "The invoice is kept blocked for shortfall "
                "correction", "pending": "The invoice is kept blocked with the confirmation pending"}[
            args["final_disposition"]]
        return (f"{done}. Report it with discrepancy type {kind}, since {why}, citing each record as I saw it.")
    raise NotImplementedError(f"no reasoning text for {name} {target}")


def args_fill(s: Seen) -> str:
    return next(a["text"] for n, a in reversed(s.calls) if n == "fill")


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=\.)\s+(?=[A-Z])", text)


def recovery_text(messages: list[dict], name: str, args: dict, before: list[dict]) -> str:
    """The reasoning before the call that recovers from the planted mistake (the last tool message): what went wrong,
    then the reasoning for the call itself (the last two sentences of what `narrate` writes for it on the conversation
    before the mistake), so the comparison stays in a trained turn while the mistake turn is masked. 1-3 sentences."""
    s = Seen(messages)
    r = s.last_result
    if "error" in r and isinstance(r.get("error"), dict) and r["error"]["code"] == "stale_state":
        return (f"The click was refused because the state changed since my last observation (it is now {s.clock}). "
                "Observe the page again before acting.")
    why = " ".join(_sentences(narrate(before, name, args))[-2:])
    if "trl_error" in r:
        return f"That call failed before it reached the application: {r['trl_error']}. {why}"
    if r["error"]["code"] == "invalid_report":
        probs = "; ".join(f"{p['field']}: {p['problem']}" for p in r["error"]["problems"])
        return f"The report was not accepted ({probs}). {why}"
    return f"The call was rejected ({r['error']['message'].rstrip('.')}). {why}"


CLOSING = {"release": "Invoice {inv} is released and the report is submitted.",
           "keep_blocked": "Invoice {inv} stays blocked for shortfall correction, and the report is submitted.",
           "pending": "Invoice {inv} stays blocked with the delivery confirmation pending, and the report is submitted."}


# --- planted mistakes --------------------------------------------------------------------------------------------------
# A wrong argument NAME raises inside TRL (the method gets an unexpected keyword or misses a required one): never
# received, so not an attempt. A wrong argument TYPE survives TRL's parser (which types arguments by the schema and
# would turn "480" into 480, so only types it keeps are used) and reaches the environment, which rejects the call.
WRONG_NAME = {"click": ("element_id", "id"), "fill": ("element_id", "id"), "coverage": ("invoice_id", "invoice"),
              "advance_time": ("minutes", "duration")}
WRONG_TYPE = {"click": lambda a: {**a, "element_id": [a["element_id"]]},
              "fill": lambda a: {**a, "text": int(a["text"])},
              "coverage": lambda a: {**a, "invoice_id": [a["invoice_id"]]},
              "advance_time": lambda a: {**a, "minutes": float(a["minutes"])}}


def malformed(name: str, args: dict, kind: str) -> dict:
    """The call with one argument wrong. finish is always the name kind: without discrepancy_type (a malformed call,
    rejected by TRL before the environment; fixed 2026-11-01)."""
    if name == "finish":
        return {k: v for k, v in args.items() if k != "discrepancy_type"}
    if kind == "type":
        return WRONG_TYPE[name](args)
    right, wrong = WRONG_NAME[name]
    return {(wrong if k == right else k): v for k, v in args.items()}


def defective(report: dict, defect: str) -> dict:
    """A report the environment rejects as invalid_report, with one field wrong."""
    bad = copy.deepcopy(report)
    evidence = [e for f in bad["findings"] for e in f["evidence"]] + bad.get("evidence", [])
    if defect == "discrepancy_type_null":
        bad["discrepancy_type"] = None
    elif defect == "missing_state_version":
        del next(e for e in evidence if e["form"] == "observed")["state_version"]
    elif defect == "undefined_operand":
        d = next(e for e in evidence if e["form"] == "derived")
        names = sorted(d["operands"])
        d["expression"] = d["expression"].replace(names[0], names[0][0] + str(len(names) + 1), 1)
    elif defect == "numeric_derived_value":
        d = next(e for e in evidence if e["form"] == "derived")
        d["value"] = int(Decimal(d["value"]))
    else:
        raise ValueError(defect)
    return bad


# --- the recorder ------------------------------------------------------------------------------------------------------
class Recorder:
    """reference/5's tools interface (dict results, finish(report)), backed by APReceiptEnvV2 through TRL's call path.
    Records the conversation, writes the reasoning text, plants the planned mistake and its recovery."""

    def __init__(self, env: APReceiptEnvV2, scenario_name: str, plan: dict):
        self.env, self.plan = env, plan
        first = env.reset(scenario=scenario_name)
        self.initial_observation = json.loads(first)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT_V2}, {"role": "user", "content": USER_PREFIX + first}]
        self.mistake_at: int | None = None     # index of the planted mistake's tool message
        self.n_calls = 0                       # reference/5's calls so far (the planted mistake not counted)
        self.before: list[dict] | None = None  # the conversation before the planted mistake

    def _send(self, name: str, args: dict, text: str) -> str:
        self.messages.append({"role": "assistant", "content": text,
                              "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]})
        out = smc.execute(self.env, name, args)
        self.messages.append({"role": "tool", "name": name, "content": out})
        return out

    def _plant(self, name: str, args: dict, text: str) -> None:
        self.before = list(self.messages)
        self._send(name, args, text)
        self.mistake_at = len(self.messages) - 1

    def _call(self, name: str, args: dict) -> dict:
        self.n_calls += 1
        kind = self.plan.get("mistake")
        recover = False
        if self.mistake_at is None and kind == "malformed" and self.n_calls == self.plan["at_call"]:
            self._plant(name, malformed(name, args, self.plan["variant"]), narrate(self.messages, name, args))
            recover = True
        if self.mistake_at is None and kind == "stale" and self._stale_moment(name, args):
            s = Seen(self.messages)
            reprocess = next(i for i, n in s.controls.items() if n == "Reprocess invoice")
            self._plant("click", {"element_id": reprocess},
                        f"It is now {s.clock}. Reprocess the invoice to see whether the block still applies.")
            out = json.loads(self.messages[-1]["content"])
            assert out["error"]["code"] == "stale_state", out
            name, args, recover = "observe", {}, True   # the recovery: observe reloads the Delivery confirmations page
        text = recovery_text(self.messages, name, args, self.before) if recover else narrate(self.messages, name, args)
        return json.loads(self._send(name, args, text))

    def _stale_moment(self, name: str, args: dict) -> bool:
        """reference/5 is about to view Delivery confirmations right after the first wait during which the state changed
        (the builder looks at the environment for this; the agent cannot, which is why its click is refused)."""
        s = Seen(self.messages)
        layer = self.env._layer
        return (name == "click" and s.controls.get(args["element_id"]) == "Delivery confirmations"
                and s.last_tool == "advance_time" and s.current["page"] == "confirmations"
                and layer.session.engine.state_version != layer.last_obs["state_version"])

    # The AgentTools interface --------------------------------------------------------------------------------------
    def observe(self):
        return self._call("observe", {})

    def click(self, element_id):
        return self._call("click", {"element_id": element_id})

    def fill(self, element_id, text):
        return self._call("fill", {"element_id": element_id, "text": text})

    def select(self, element_id, option):
        return self._call("select", {"element_id": element_id, "option": option})

    def advance_time(self, minutes):
        return self._call("advance_time", {"minutes": minutes})

    def coverage(self, invoice_id):
        return self._call("coverage", {"invoice_id": invoice_id})

    def finish(self, report):
        args = {"final_disposition": report["final_disposition"], "discrepancy_type": report["discrepancy_type"],
                "findings": report["findings"]}
        if report.get("evidence"):
            args["evidence"] = report["evidence"]
        if self.plan.get("mistake") == "report_defect" and self.mistake_at is None:
            self._plant("finish", defective(args, self.plan["defect"]), narrate(self.messages, "finish", args))
            out = json.loads(self.messages[-1]["content"])
            assert out["ok"] is False and out["error"]["code"] == "invalid_report", out
            self.n_calls += 1
            return json.loads(self._send("finish", args, recovery_text(self.messages, "finish", args, self.before)))
        return self._call("finish", args)


def seed_of(group_key: str) -> int:
    """reference/5's seed (the order of the two record pages) from the grouping key, so twins share it."""
    return int(group_key[:8], 16)


def demonstrate(scenario, group_key: str, plan: dict, require: bool = True) -> dict:
    """One demonstration: the conversation, where the planted mistake is, and the episode for grading. With
    require=False a plan that cannot be realised (a stale click needs a state change during a wait) leaves the
    demonstration clean (the twin check's siblings)."""
    env = APReceiptEnvV2({scenario.name: lambda: scenario}, execution_rule=NO_GUARD)
    rec = Recorder(env, scenario.name, plan)
    ReferenceAgentV5(rec, seed=seed_of(group_key)).run()
    if require and plan.get("mistake") and rec.mistake_at is None:
        raise ValueError(f"plan {plan} not realised on {scenario.name}")
    finish = rec.messages[-2]["tool_calls"][0]["function"]["arguments"]
    inv = Seen(rec.messages).invoice["id"]
    messages = rec.messages + [{"role": "assistant", "content": CLOSING[finish["final_disposition"]].format(inv=inv)}]
    return {"env": env, "messages": messages, "plan": plan, "mistake_at": rec.mistake_at, "calls": rec.n_calls}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --- checks (the acceptance checks before sft-demos/v2-1 was frozen) ---------------------------------------------------
def replay(scenario, messages: list[dict], mistake_at: int | None) -> dict:
    """Replay the stored conversation through a fresh APReceiptEnvV2 by TRL's call path: every tool message must come
    back byte-identical; then grade it as training does (g8, ap_reward_v2 with the conversation as the completion)."""
    from ap_sim.training import rewards
    env = APReceiptEnvV2({scenario.name: lambda: scenario}, execution_rule=NO_GUARD)
    first = env.reset(scenario=scenario.name)
    problems = [] if messages[1]["content"] == USER_PREFIX + first else ["reset page differs"]
    rejections = []
    for k, m in enumerate(messages):
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        f = m["tool_calls"][0]["function"]
        out = smc.execute(env, f["name"], f["arguments"])
        if out != messages[k + 1]["content"]:
            problems.append(f"tool message {k + 1} differs on replay")
        if out.startswith("{'error'") or not json.loads(out)["ok"]:
            rejections.append(k + 1)
    (reward,) = rewards.ap_reward_v2([env], completions=[messages[2:]])
    row = rewards.LAST_BATCH[0]
    planted = [] if mistake_at is None else [mistake_at]
    if rejections != planted:
        problems.append(f"rejections at {rejections}, planted {planted}")
    if env._post_end:
        problems.append(f"{len(env._post_end)} post-end calls")
    if reward != 1 or row["stop_reason"] != "finish" or row["finish_result_dropped"]:
        problems.append(f"reward {reward}, stop {row['stop_reason']}, dropped {row['finish_result_dropped']}")
    tail = messages[-1]
    if tail["role"] != "assistant" or tail.get("tool_calls") or messages[-2]["role"] != "tool":
        problems.append("does not end with one closing message after the finish result")
    scenario_, log, _ = env._episode()
    return {"reward": reward, "stop_reason": row["stop_reason"], "finish_result_dropped": row["finish_result_dropped"],
            "malformed_tool_calls": row["malformed_tool_calls"], "calls": env._layer.calls,
            "final": log["report"]["final_disposition"] if log.get("report") else None,
            "rejections": rejections, "problems": problems}


def masked_turn(demo: dict) -> int | None:
    """The message index of the planted mistake's assistant turn (the one assistant turn that is not trained)."""
    return None if demo["mistake_at"] is None else demo["mistake_at"] - 1


def twin_check(entry: dict, plan: dict) -> list[dict]:
    """Counterfactual siblings (the same agent-visible records, another hidden world; in no split) built with the same
    plan and seed. The check applies to trained turns (fixed 2026-11-01). The first message that differs must be
    a tool message (the hidden fact becoming visible) or the masked planted-mistake turn, which may be placed using
    environment state; every trained turn after it reacts to the tool results before it."""
    from ap_sim.generator.causal import counterfactual, generate
    case = generate(entry["case_id"], entry["seed"], entry["force"])
    a = demonstrate(case.scenario(), entry["keys"]["group"], plan)
    out = []
    for change in ("delivered", "timing"):
        try:
            sib = counterfactual(case, change)
        except Exception as exc:          # out of scope for this case
            out.append({"change": change, "skipped": f"{type(exc).__name__}: {exc}", "ok": True})
            continue
        b = demonstrate(sib.scenario(), entry["keys"]["group"], plan, require=False)
        i = next((k for k, (x, y) in enumerate(zip(a["messages"], b["messages"])) if x != y), None)
        if i is None and len(a["messages"]) != len(b["messages"]):
            i = min(len(a["messages"]), len(b["messages"]))
        role = None if i is None else a["messages"][i]["role"]
        masked = i is not None and i in (masked_turn(a), masked_turn(b))
        out.append({"change": change, "sibling_planted": b["mistake_at"] is not None, "first_difference": i,
                    "role": role, "masked_turn": masked, "ok": i is None or role == "tool" or masked})
    return out


def forbidden_strings() -> set[str]:
    """Case, split and group identifiers (every split's case ids and keys): none may appear in a training row."""
    from check_splits import SPLITS, entries as split_entries
    out = set()
    for split in SPLITS:
        for e in split_entries(split):
            out |= {e["case_id"], e["keys"]["exact"], e["keys"]["near"], e["keys"]["group"], e["params_sha256"]}
            out |= {e["keys"]["group"][:12], e["keys"]["near"][:12], e["keys"]["exact"][:12]}
    return out


IDENTIFIER_PATTERNS = ("sft-", "rl-", "dev-", "test-", "v6-", "policy_shift", "test_reserve", "forgetting")


def identifier_problems(messages: list[dict], row: dict, forbidden: set[str], decoded: str) -> list[str]:
    """The conversation and the row's decoded text: no identifier, no evaluator-only term; the row holds only
    input_ids and labels."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_agent_visibility import EVALUATOR_ONLY
    text = json.dumps(messages) + decoded
    found = [s for s in forbidden if s in text] + [p for p in IDENTIFIER_PATTERNS if p in text]
    if set(row) != {"input_ids", "labels"}:
        found.append(f"row keys {sorted(row)}")
    texts = json.dumps([m["content"] for m in messages if m["role"] != "system"])
    # TRL's own error text for a misnamed argument ("got an unexpected keyword argument"), and the recovery that quotes
    # it, contain "expected"; it is TRL's wording about the call, not a case fact. Nothing else is exempt.
    hit = EVALUATOR_ONLY.search(texts.replace("unexpected keyword argument", ""))
    return found + ([f"evaluator-only term {hit.group(0)!r}"] if hit else [])


class Renderer:
    """The pinned tokenizer and the two templates TRL 1.16 uses: GRPOTrainer renders with the model's own template
    (chat_template None: it is prefix-preserving with tools, grpo_trainer.py 776-779); the SFT tokens here come from
    TRL's training template (the one SFTTrainer would swap in for assistant-only loss, sft_trainer.py 941-944), whose
    generation markers give the assistant mask."""

    def __init__(self):
        from transformers import AutoTokenizer
        from trl.chat_template_utils import add_response_schema, get_training_chat_template, parse_response
        from qwen_check import MODEL, REVISION
        self.tok = add_response_schema(AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True))
        self.template = get_training_chat_template(self.tok)
        self.parse_response = parse_response
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.newline = self.tok.encode("\n", add_special_tokens=False)

    def ids(self, messages, tools, gen=False, mask=False, grpo=False):
        out = self.tok.apply_chat_template(messages, tools=tools, chat_template=None if grpo else self.template,
                                           add_generation_prompt=gen, tokenize=True, return_dict=True,
                                           return_assistant_tokens_mask=mask)
        return out if mask else list(out["input_ids"])


def tool_schemas() -> tuple[list, list[dict]]:
    """The tools exactly as TRL lists them for APReceiptEnvV2 (bound methods, inspect.getmembers order), and their JSON
    schemas."""
    from transformers.utils import get_json_schema
    from qwen_check import trl_tools
    tools = trl_tools(APReceiptEnvV2())
    return tools, [get_json_schema(t) for t in tools]


def spans(mask: list[int]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, t in enumerate(list(mask) + [0]):
        if t and start is None:
            start = i
        if not t and start is not None:
            out.append((start, i))
            start = None
    return out


def training_row(r: Renderer, demo: dict, schemas: list[dict]) -> dict:
    """The pre-tokenized training row (fixed 2026-11-01): input_ids are the whole conversation as the training
    template renders it; labels train every assistant turn (its tokens through <|im_end|> and the newline the template
    marks after it) except the planted mistake turn, and nothing else (no system, user or tool token)."""
    out = r.ids(demo["messages"], schemas, mask=True)
    ids, amask = list(out["input_ids"]), list(out["assistant_masks"])
    turns = [k for k, m in enumerate(demo["messages"]) if m["role"] == "assistant"]
    sp = spans(amask)
    assert len(sp) == len(turns), (len(sp), len(turns))
    labels = [-100] * len(ids)
    for (s0, e0), k in zip(sp, turns):
        if k != masked_turn(demo):
            labels[s0:e0] = ids[s0:e0]
    return {"input_ids": ids, "labels": labels}


def render_check(r: Renderer, demo: dict, row: dict, tools: list, schemas: list[dict]) -> dict:
    """Rendering and masking, on CPU:
      1. the stored JSON schemas render exactly as the environment's methods (GRPO passes the methods);
      2. a rollout built as GRPOTrainer builds it, in its template (the model's own): the prompt (system, USER_PREFIX +
         reset page, generation prompt), each assistant turn's tokens through <|im_end|>, then _get_tool_suffix_ids
         for its tool message, equals the row's input_ids token for token (the SFT rendering), and every assistant
         turn parses back (parse_response) to the stored message;
      3. the labels, against an expectation built from the messages alone: every assistant turn except the planted
         mistake is trained, as exactly that turn's generated tokens through <|im_end|> plus the one newline token
         the training template adds; no other token (system, user, tool response, generation header, the mistake
         turn) is trained;
      4. completion and context lengths against the limits."""
    from qwen_check import tool_suffix_ids
    from ap_sim.training.config import PIPELINE_MAX_COMPLETION_LENGTH, PIPELINE_MAX_CONTEXT_LENGTH
    msgs, problems, ids = demo["messages"], [], row["input_ids"]
    if r.ids(msgs, tools) != r.ids(msgs, schemas):
        problems.append("schemas render differently from the methods")
    grpo_prompt = r.ids(msgs[:2], tools, gen=True, grpo=True)
    seq, turns = list(grpo_prompt), {}
    for k in range(2, len(msgs)):
        m = msgs[k]
        if m["role"] == "assistant":
            before = r.ids(msgs[:k], tools, gen=True, grpo=True)
            turn = r.ids(msgs[:k + 1], tools, grpo=True)[len(before):]
            turn = turn[:turn.index(r.im_end) + 1]
            parsed = r.parse_response(r.tok, turn, prefix=seq, tools=schemas)
            want = {"role": "assistant", "content": m["content"]}
            if m.get("tool_calls"):
                want["tool_calls"] = m["tool_calls"]
            if {k2: parsed.get(k2) for k2 in want} != want:
                problems.append(f"assistant turn {k} parses back differently")
            turns[k] = (len(seq), turn)
            seq += turn
        else:
            seq += tool_suffix_ids(r.tok, None, [m])
    if ids[:len(seq)] != seq or r.tok.decode(ids[len(seq):]) not in ("", "\n"):
        problems.append("the GRPO rollout tokens differ from the SFT rendering")
    want = [-100] * len(ids)
    for k, (start, turn) in turns.items():
        if k != masked_turn(demo):
            span = turn + r.newline
            want[start:start + len(span)] = span
    if row["labels"] != want:
        problems.append("labels differ from the expectation built from the messages")
    trained = sum(x != -100 for x in row["labels"])
    completion_tokens, context = len(ids) - len(grpo_prompt), len(ids)
    if completion_tokens > PIPELINE_MAX_COMPLETION_LENGTH or context > PIPELINE_MAX_CONTEXT_LENGTH:
        problems.append(f"over a limit: completion {completion_tokens}, context {context}")
    assistant_tokens = sum(len(t) for _, t in turns.values())
    return {"prompt_tokens": len(grpo_prompt), "completion_tokens": completion_tokens, "context_tokens": context,
            "trained_tokens": trained, "assistant_tokens": assistant_tokens,
            "trained_turns": sum(1 for k in turns if k != masked_turn(demo)), "problems": problems}


# --- allocation (fixed 2026-11-01) -------------------------------------------------------------------------------------
ALLOCATION_SEED = 20261101
PER_TYPE = 45                          # 135 of 300 (45%): 45 malformed, 45 report_defect, 45 stale
STALE_TYPES = ("V1", "V2", "V3a", "V3b")


def case_type(case_id: str) -> str:
    return case_id.split("-")[1]


def quotas(counts: dict[str, int], total: int, rng: random.Random) -> dict[str, int]:
    """`total` spread over the types in proportion to their counts: largest remainders, ties broken by rng."""
    n = sum(counts.values())
    exact = {t: total * c / n for t, c in counts.items()}
    out = {t: int(x) for t, x in exact.items()}
    order = sorted(counts, key=lambda t: (-(exact[t] - out[t]), rng.random()))
    for t in order[:total - sum(out.values())]:
        out[t] += 1
    return out


def allocate(case_ids: list[str], seed: int = ALLOCATION_SEED) -> tuple[dict[str, dict], dict]:
    """Which cases get which planted mistake (case id -> plan; clean cases are absent), and the quotas. Stale goes to
    V1, V2, V3a and V3b only; malformed and report defects to all types in proportion to the split. Per type, the cases
    are shuffled once and taken in order: stale first, then malformed, then report defects. Report defects cycle
    through the four kinds in a shuffled order; malformed variants (name or type) alternate the same way. The turn of a
    malformed call is drawn later, uniformly over the demonstration's own calls (`draw_turn`)."""
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = {}
    for cid in case_ids:
        by_type.setdefault(case_type(cid), []).append(cid)
    counts = {t: len(v) for t, v in by_type.items()}
    q = {"stale": quotas({t: counts[t] for t in STALE_TYPES}, PER_TYPE, rng),
         "malformed": quotas(counts, PER_TYPE, rng), "report_defect": quotas(counts, PER_TYPE, rng)}
    defects = [DEFECTS[i % len(DEFECTS)] for i in range(PER_TYPE)]
    variants = ["name" if i % 2 == 0 else "type" for i in range(PER_TYPE)]
    rng.shuffle(defects)
    rng.shuffle(variants)
    plans = {}
    for t in sorted(by_type):
        pool = list(by_type[t])
        rng.shuffle(pool)
        for kind in ("stale", "malformed", "report_defect"):
            for _ in range(q[kind].get(t, 0)):
                cid = pool.pop(0)
                plans[cid] = {"mistake": kind}
    for kind, extra in (("report_defect", defects), ("malformed", variants)):
        chosen = sorted((c for c, p in plans.items() if p["mistake"] == kind), key=case_ids.index)
        for cid, x in zip(chosen, extra, strict=True):
            plans[cid]["defect" if kind == "report_defect" else "variant"] = x
    return plans, q


def draw_turn(case_id: str, n_calls: int, attempt: int) -> int:
    """The call before which a malformed call is planted: uniform over the clean demonstration's calls, from a seed
    recorded per case (ALLOCATION_SEED, the case and the attempt). A later attempt is drawn only when the planted call
    would break the episode (a wrong-type call reaches the environment and costs a call and 5 minutes)."""
    return random.Random(f"{ALLOCATION_SEED}:{case_id}:{attempt}").randint(1, n_calls)


def plan_for(scenario_factory, entry: dict, plan: dict | None) -> tuple[dict, dict, list]:
    """The demonstration for one case under its plan (with the malformed turn drawn), and the attempts made."""
    group = entry["keys"]["group"]
    if not plan:
        return demonstrate(scenario_factory(), group, {}), {}, []
    if plan["mistake"] != "malformed" or "at_call" in plan:      # the REVIEW cases' fixed plans give the turn
        return demonstrate(scenario_factory(), group, plan), plan, []
    n = demonstrate(scenario_factory(), group, {})["calls"]
    tried = []
    for attempt in range(50):
        p = {**plan, "at_call": draw_turn(entry["case_id"], n, attempt), "attempt": attempt}
        try:
            d = demonstrate(scenario_factory(), group, p)
        except Exception as exc:      # reference/5 met a refusal it does not expect (e.g. the horizon)
            tried.append({"at_call": p["at_call"], "failed": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        rep = replay(scenario_factory(), d["messages"], d["mistake_at"])
        if not rep["problems"]:
            return d, p, tried
        tried.append({"at_call": p["at_call"], "failed": rep["problems"][:2]})
    raise RuntimeError(f"no workable malformed turn for {entry['case_id']}: {tried[:3]}")


# --- transcripts -------------------------------------------------------------------------------------------------------
def transcript(cid: str, entry: dict, demo: dict, checks: dict) -> str:
    """A readable transcript of one demonstration, for review."""
    rc, rp = checks["render"], checks["replay"]
    mt = masked_turn(demo)
    lines = [f"# {cid}: {entry['expected']} ({entry['variant']})", "",
             f"- Plan: {demo['plan'] or 'clean (no planted mistake)'}",
             f"- Masking: every assistant turn is trained except "
             f"{'the planted mistake (message ' + str(mt) + ')' if mt is not None else 'none (no mistake)'}; system, "
             "user and tool messages are not trained.",
             f"- Replay: reward {rp['reward']}, stop {rp['stop_reason']}, finish_result_dropped "
             f"{rp['finish_result_dropped']}, {rp['calls']} environment calls, rejections at messages {rp['rejections']}, "
             f"malformed calls (TRL) {rp['malformed_tool_calls']}, problems {rp['problems'] or 'none'}",
             f"- Tokens: prompt {rc['prompt_tokens']}, completion {rc['completion_tokens']} (limit 28,354), context "
             f"{rc['context_tokens']} (limit 40,094); trained {rc['trained_tokens']} "
             f"({100 * rc['trained_tokens'] / rc['completion_tokens']:.1f}% of the completion) in {rc['trained_turns']} "
             f"assistant turns; render problems {rc['problems'] or 'none'}",
             f"- Twin check: {checks['twins']}", f"- Identifiers: {checks['identifiers'] or 'none'}", ""]
    for k, m in enumerate(demo["messages"]):
        if m["role"] == "system":
            lines += [f"## [{k}] system (not trained)", "", f"SYSTEM_PROMPT_V2 ({len(m['content'])} characters; "
                      "instructions/5, unchanged)", ""]
        elif m["role"] == "user":
            lines += [f"## [{k}] user (not trained)", "", "```", m["content"], "```", ""]
        elif m["role"] == "assistant":
            tag = "PLANTED MISTAKE, not trained" if k == mt else "trained"
            lines += [f"## [{k}] assistant ({tag})", "", m["content"], ""]
            if m.get("tool_calls"):
                f = m["tool_calls"][0]["function"]
                body = json.dumps(f["arguments"], indent=1) if f["name"] == "finish" else json.dumps(f["arguments"])
                lines += [f"Call `{f['name']}`:", "", "```json", body, "```", ""]
        else:
            lines += [f"## [{k}] tool `{m['name']}` (not trained)", "", "```", m["content"], "```", ""]
    return "\n".join(lines)


# --- building and the acceptance summary -------------------------------------------------------------------------------
_W: dict = {}


def _init_worker() -> None:
    from check_splits import entries as split_entries
    from ap_sim.generator.splits import load_split
    _W["cases"] = load_split("sft")
    _W["entries"] = {e["case_id"]: e for e in split_entries("sft")}
    _W["r"] = Renderer()
    _W["tools"], _W["schemas"] = tool_schemas()
    _W["forbidden"] = forbidden_strings()


def build_one(job: tuple[str, dict | None]) -> dict:
    """Build, replay, render-check, twin-check and identifier-check one demonstration."""
    cid, plan = job
    cases, e, r = _W["cases"], _W["entries"][cid], _W["r"]
    demo, used, tried = plan_for(cases[cid], e, plan)
    row = training_row(r, demo, _W["schemas"])
    checks = {"replay": replay(cases[cid](), demo["messages"], demo["mistake_at"]),
              "render": render_check(r, demo, row, _W["tools"], _W["schemas"]),
              "twins": twin_check(e, used),
              "identifiers": identifier_problems(demo["messages"], row, _W["forbidden"], r.tok.decode(row["input_ids"]))}
    ok = (not checks["replay"]["problems"] and not checks["render"]["problems"] and not checks["identifiers"]
          and all(t["ok"] for t in checks["twins"]))
    mistake = None
    if demo["mistake_at"] is not None:
        f = demo["messages"][demo["mistake_at"] - 1]["tool_calls"][0]["function"]
        mistake = {"tool": f["name"], "result": demo["messages"][demo["mistake_at"]]["content"][:160]}
    return {"case_id": cid, "type": case_type(cid), "expected": e["expected"], "plan": used, "redraws": tried,
            "mistake": mistake, "ok": ok, "checks": checks, "row": row,
            "messages": demo["messages"], "transcript": transcript(cid, e, demo, checks)}


REVIEW = {"sft-V1-0": {"mistake": "stale"}, "sft-V2-0": {"mistake": "report_defect", "defect": "undefined_operand"},
          "sft-V4-0": {"mistake": "malformed", "variant": "name", "at_call": 3}, "sft-V5-0": None, "sft-V3a-0": None}


def build_rows(workers: int = 6) -> bytes:
    """train.jsonl alone, rebuilt from the builder and ALLOCATION_SEED (no checks): what the freeze test compares."""
    from concurrent.futures import ProcessPoolExecutor
    from check_splits import entries as split_entries
    case_ids = [e["case_id"] for e in split_entries("sft")]
    plans, _ = allocate(case_ids)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        rows = list(pool.map(_row_one, [(c, plans.get(c)) for c in case_ids], chunksize=2))
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _row_one(job: tuple[str, dict | None]) -> dict:
    cid, plan = job
    demo, _, _ = plan_for(_W["cases"][cid], _W["entries"][cid], plan)
    return training_row(_W["r"], demo, _W["schemas"])


def main(mode: str, out: pathlib.Path | None = None) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from check_splits import entries as split_entries
    case_ids = [e["case_id"] for e in split_entries("sft")]
    plans, q = allocate(case_ids)
    out = out or ROOT / "runs" / "phase4" / ("review" if mode == "review" else "sft-demos-v2-1")
    out.mkdir(parents=True, exist_ok=True)
    if mode == "review":
        jobs = list(REVIEW.items())     # the five REVIEW cases, with their fixed plans
    else:
        jobs = [(c, plans.get(c)) for c in case_ids]
    with ProcessPoolExecutor(max_workers=6, initializer=_init_worker) as pool:
        results = list(pool.map(build_one, jobs, chunksize=2))
    for res in results:
        (out / "transcripts").mkdir(exist_ok=True)
        (out / "transcripts" / f"{res['case_id']}.md").write_text(res.pop("transcript"))
        print(res["case_id"], res["plan"], "OK" if res["ok"] else "PROBLEMS", res["checks"]["render"]["completion_tokens"],
              flush=True)
    rows = "".join(json.dumps(res["row"]) + "\n" for res in results)
    (out / "train.jsonl").write_text(rows)
    with open(out / "sidecar.jsonl", "w") as f:   # not training data: case ids, plans and conversations for audit
        for res in results:
            f.write(json.dumps({k: v for k, v in res.items() if k != "row"}) + "\n")
    summary = gate4(results, q, hashlib.sha256(rows.encode()).hexdigest()) if mode == "all" else \
        {"review": [{"case": r_["case_id"], "plan": r_["plan"], "ok": r_["ok"]} for r_ in results]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:6000])


# The acceptance summary of every check on all 300 demonstrations, made before the set was frozen.
def gate4(results: list[dict], q: dict, sha256: str) -> dict:
    import collections
    from ap_sim.training.config import PIPELINE_MAX_COMPLETION_LENGTH
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for res in results:
        by_type[res["type"]].append(res)

    def stats(rs: list[dict]) -> dict:
        comp = [x["checks"]["render"]["completion_tokens"] for x in rs]
        trained = [x["checks"]["render"]["trained_tokens"] for x in rs]
        rec = [x for x in rs if x["plan"]]
        return {"demonstrations": len(rs),
                "final_actions": dict(collections.Counter(x["checks"]["replay"]["final"] for x in rs)),
                "recoveries": len(rec), "recovery_share": round(len(rec) / len(rs), 3),
                "mistakes": dict(collections.Counter(x["plan"]["mistake"] for x in rec)),
                "completion_tokens": {"min": min(comp), "mean": round(sum(comp) / len(comp)), "max": max(comp)},
                "trained_share_of_completion": round(sum(trained) / sum(comp), 4),
                "trained_share_clean": round(sum(x["checks"]["render"]["trained_tokens"] for x in rs if not x["plan"]) /
                                             max(1, sum(x["checks"]["render"]["completion_tokens"] for x in rs
                                                        if not x["plan"])), 4),
                "trained_share_recovery": round(sum(x["checks"]["render"]["trained_tokens"] for x in rec) /
                                                max(1, sum(x["checks"]["render"]["completion_tokens"] for x in rec)), 4),
                "env_calls": {"min": min(x["checks"]["replay"]["calls"] for x in rs),
                              "max": max(x["checks"]["replay"]["calls"] for x in rs)}}

    rec = [x for x in results if x["plan"]]
    longest = max(results, key=lambda x: x["checks"]["render"]["completion_tokens"])
    failures = [{"case": x["case_id"], "replay": x["checks"]["replay"]["problems"],
                 "render": x["checks"]["render"]["problems"], "twins": [t for t in x["checks"]["twins"] if not t["ok"]],
                 "identifiers": x["checks"]["identifiers"]} for x in results if not x["ok"]]
    return {
        "dataset": DATASET_VERSION, "train_jsonl_sha256": sha256, "allocation_seed": ALLOCATION_SEED,
        "demonstrations": len(results), "passed_all_checks": sum(x["ok"] for x in results), "failures": failures,
        "replay_reward_1_stop_finish": sum(x["checks"]["replay"]["reward"] == 1 and
                                           x["checks"]["replay"]["stop_reason"] == "finish" for x in results),
        "only_rejection_is_planted": sum(not any("rejections" in p for p in x["checks"]["replay"]["problems"])
                                         for x in results),
        "post_end_calls": sum(any("post-end" in p for p in x["checks"]["replay"]["problems"]) for x in results),
        "identifier_hits": sum(bool(x["checks"]["identifiers"]) for x in results),
        "twin_pairs": sum(1 for x in results for t in x["checks"]["twins"] if "skipped" not in t),
        "twin_pairs_skipped": sum(1 for x in results for t in x["checks"]["twins"] if "skipped" in t),
        "twin_pairs_first_difference_masked_turn": sum(1 for x in results for t in x["checks"]["twins"]
                                                       if t.get("masked_turn")),
        "twin_pairs_failing": sum(1 for x in results for t in x["checks"]["twins"] if not t["ok"]),
        "longest": {"case": longest["case_id"], "completion_tokens": longest["checks"]["render"]["completion_tokens"],
                    "limit": PIPELINE_MAX_COMPLETION_LENGTH,
                    "context_tokens": longest["checks"]["render"]["context_tokens"]},
        "quotas": q,
        "recoveries": {"total": len(rec), "share": round(len(rec) / len(results), 3),
                       "by_kind": dict(collections.Counter(x["plan"]["mistake"] for x in rec)),
                       "by_kind_and_type": {k: dict(collections.Counter(x["type"] for x in rec
                                                                        if x["plan"]["mistake"] == k))
                                            for k in MISTAKES},
                       "report_defects": dict(collections.Counter(x["plan"]["defect"] for x in rec
                                                                  if x["plan"]["mistake"] == "report_defect")),
                       "malformed_variants": dict(collections.Counter(x["plan"]["variant"] for x in rec
                                                                      if x["plan"]["mistake"] == "malformed")),
                       "malformed_tools": dict(collections.Counter(x["mistake"]["tool"] for x in rec
                                                                   if x["plan"]["mistake"] == "malformed")),
                       "malformed_redrawn": {x["case_id"]: x["redraws"] for x in rec if x["redraws"]}},
        "by_type": {t: stats(rs) for t, rs in sorted(by_type.items())},
        "all": stats(results),
    }


if __name__ == "__main__":
    if sys.argv[1:2] not in (["review"], ["all"]):
        raise SystemExit(__doc__)
    main(sys.argv[1], pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None)
