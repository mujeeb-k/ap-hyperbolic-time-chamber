"""Phase 10's analyses (preregistration_proof/PLAN_V2.md, sections 4 and 5), from saved episodes.

Everything here reads records; nothing runs a model, and nothing feeds back into what is evaluated. scripts/gpu/final_evaluation.py calls `assemble` when every cell is done.

    unseen_analysis     The pre-registered secondary analysis (declared before any model output on the v2 splits): within type and
                        reweighted to the test mix, with the seen-signature cases on the same mix.
    business_metrics    The confusion matrix, wrong-payment and wrongful-hold rates, the expected weighted loss per 1,000 invoices (10/3/2/1).
    failure_catalogue   Failures by grader flag with counts per model, one transcript per flag and model, chosen by frozen case order.
    recovery            Recovery from a model's own errors: the third of the hypothesis's metrics, which no record holds as such.
    claude_row          Claude's row on the first 100 test cases against SFT-v2 and GRPO on those same 100, and its measured cost per case.

Where the plan leaves a definition open, the reading made here is written into the output under "definitions" (a reader's call, not a result).
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu"), str(ROOT / "scripts/phase2"), str(ROOT / "scripts/phase5")]

from ap_sim.evaluation import gate, metrics   # noqa: E402
from ap_sim.events.clock import MINUTES_PER_DAY   # noqa: E402

TEST_MIX = {"V1": 0.20, "V2": 0.18, "V3a": 0.10, "V3b": 0.10, "V4": 0.22, "V5": 0.20}     # pre-declared before any model output on the v2 splits
WEIGHTS = {"wrong_payment": 10, "wrongful_hold": 3, "pending_keep_blocked_confusion": 2, "business_day_late": 1}     # analysis only
OUTCOMES = ("release", "keep_blocked", "pending")
NO_ACTION = "no final action"
NO_FLAG = "no grader flag"
PRIMARY_PAIR = ("sft-v2", "grpo")


def read_episodes(directory: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in (pathlib.Path(directory) / "episodes.jsonl").read_text().splitlines() if line.strip()]


def by_case(records: list[dict]) -> dict[str, dict]:
    """One record per case (a greedy evaluation). Two rows for a case would otherwise silently keep one."""
    out: dict[str, dict] = {}
    for r in records:
        if r["case"] in out:
            raise ValueError(f"{r['case']} appears twice")
        out[r["case"]] = r
    return out


def samples_by_case(records: list[dict]) -> dict[str, dict]:
    """case -> sample -> record (a sampled evaluation), refusing a repeated (case, sample)."""
    out: dict[str, dict] = collections.defaultdict(dict)
    for r in records:
        if r.get("sample") in out[r["case"]]:
            raise ValueError(f"{r['case']} sample {r.get('sample')} appears twice")
        out[r["case"]][r.get("sample")] = r
    return dict(out)


# Unseen signatures ---------------------------------------------------------------------------------------------------------
def manifest_entries(split: str) -> list[dict]:
    return json.loads((ROOT / "manifests" / "v2" / f"{split}.json").read_text())["entries"]


def training_signatures() -> set[str]:
    """The structural signatures of SFT and RL together (the secondary analysis's definition of training)."""
    return {json.dumps(e["signature"]) for split in ("sft", "rl") for e in manifest_entries(split)}


def signature_table(splits: tuple[str, ...] = ("test", "test_reserve")) -> dict[str, dict]:
    """case -> its type and whether its signature occurs in training, read from the frozen manifests (no model output)."""
    seen = training_signatures()
    return {e["case_id"]: {"type": e["group"], "unseen": json.dumps(e["signature"]) not in seen} for split in splits for e in manifest_entries(split)}


def percentile_interval(values: list[float]) -> tuple[float, float]:
    """As metrics.paired_bootstrap takes it: the 2.5th and 97.5th of the sorted resamples."""
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def reweighted(by_type: dict[str, list[float]], weights: dict[str, float], resamples: int = metrics.BOOTSTRAP_RESAMPLES,
               seed: int = metrics.BOOTSTRAP_SEED) -> dict:
    """Sum over types of weight x the within-type mean, with an interval from resampling cases within each type. None if a type in the
    weights has no case here: the mix cannot be reached without it, and nothing is renormalised to hide that."""
    missing = sorted(t for t in weights if not by_type.get(t))
    if missing:
        return {"rate": None, "low": None, "high": None, "types_without_a_case": missing}
    rng = random.Random(seed)
    types = sorted(weights)
    rate = sum(weights[t] * statistics.fmean(by_type[t]) for t in types)
    draws = []
    for _ in range(resamples):
        total = 0.0
        for t in types:
            v = by_type[t]
            total += weights[t] * sum(v[rng.randrange(len(v))] for _ in v) / len(v)
        draws.append(total)
    low, high = percentile_interval(draws)
    return {"rate": round(rate, 4), "low": round(low, 4), "high": round(high, 4), "resamples": resamples, "seed": seed}


def unseen_analysis(greedy: dict[str, list[dict]], table: dict[str, dict], weights: dict[str, float] = TEST_MIX,
                    pair: tuple[str, str] = PRIMARY_PAIR, check_mix: bool = True, resamples: int = metrics.BOOTSTRAP_RESAMPLES) -> dict:
    """Greedy success per model on the unseen-signature cases and on the seen-signature cases, each within type (with its count and Wilson
    interval) and reweighted to the test mix; and the paired difference between the pair on each. Secondary; counts per type are small."""
    rows = {m: by_case(r) for m, r in greedy.items()}
    case_sets = {m: set(r) for m, r in rows.items()}
    cases = next(iter(case_sets.values()), set())
    if any(s != cases for s in case_sets.values()):
        raise ValueError("the models were not evaluated on the same cases")
    unknown = sorted(cases - set(table))
    if unknown:
        raise ValueError(f"no signature for {unknown[:5]}")
    types = collections.Counter(table[c]["type"] for c in cases)
    if check_mix and {t: round(n / len(cases), 2) for t, n in types.items()} != weights:
        raise ValueError(f"the cases' mix {dict(types)} is not the pre-declared test mix")
    out: dict = {"definition": "signature not in SFT and RL; within type, and reweighted to the test mix (V2.md, pre-declared)", "weights": weights,
                 "test_cases": len(cases), "pair": list(pair)}
    for name, wanted in (("unseen", True), ("seen", False)):
        chosen = sorted(c for c in cases if table[c]["unseen"] is wanted)
        group = collections.defaultdict(list)
        for c in chosen:
            group[table[c]["type"]].append(c)
        block: dict = {"cases": len(chosen), "cases_by_type": {t: len(v) for t, v in sorted(group.items())}, "models": {}}
        for model, r in rows.items():
            success = {t: [float(r[c]["success"]) for c in v] for t, v in group.items()}
            block["models"][model] = {"within_type": {t: metrics.wilson(int(sum(s)), len(s)) for t, s in sorted(success.items())},
                                      "reweighted": reweighted(success, weights, resamples)}
        if all(m in rows for m in pair):
            difference = {t: [float(rows[pair[1]][c]["success"]) - float(rows[pair[0]][c]["success"]) for c in v] for t, v in group.items()}
            block["difference"] = reweighted(difference, weights, resamples)
        out[name] = block
    return out


# Business metrics ----------------------------------------------------------------------------------------------------------
BUSINESS_DEFINITIONS = {
    "predicted": "the outcome reached (the grader's `reached`, the record's final_action); none is its own column",
    "wrong_payment_rate": "reached release, expected not release, over the episodes expected not release. The engine refuses an unsafe release, so an attempt is not a payment: attempted violations are beside it",
    "wrongful_hold_rate": "reached keep_blocked or pending, expected release, over the episodes expected release",
    "no_final_action_on_release": "expected release, no outcome reached, over the same episodes. Its own line, and charged as a hold in the loss: the invoice stays blocked. A reading, not in the plan",
    "pending_keep_blocked_confusion": "reached one of the two, expected the other",
    "business_days_late": "late_by_minutes / 480 where positive, on an outcome reached correctly (the record's resolution time), not rounded. A reading: the plan says one day costs 1",
    "loss_per_1000": "1,000 x the mean over episodes of the weights charged; analysis only, never a metric of success (D5a)",
    "weights": WEIGHTS,
}


def confusion_matrix(records: list[dict]) -> dict:
    table = {e: {p: 0 for p in OUTCOMES + (NO_ACTION,)} for e in OUTCOMES}
    for r in records:
        table[r["expected"]][r["final_action"] if r["final_action"] in OUTCOMES else NO_ACTION] += 1
    return table


def episode_loss(r: dict) -> float:
    expected, reached = r["expected"], r["final_action"]
    if expected != "release" and reached == "release":
        return WEIGHTS["wrong_payment"]
    if expected == "release" and reached != "release":
        return WEIGHTS["wrongful_hold"]
    if {expected, reached} == {"keep_blocked", "pending"}:
        return WEIGHTS["pending_keep_blocked_confusion"]
    late = (r.get("resolution_time") or {}).get("late_by_minutes")
    return WEIGHTS["business_day_late"] * late / MINUTES_PER_DAY if late and late > 0 else 0.0


def business_metrics(records: list[dict]) -> dict:
    n = len(records)
    paying = [r for r in records if r["expected"] != "release"]
    release = [r for r in records if r["expected"] == "release"]
    return {"episodes": n, "confusion": confusion_matrix(records),
            "wrong_payment_rate": metrics.wilson(sum(r["final_action"] == "release" for r in paying), len(paying)),
            "attempted_violation_rate": metrics.wilson(sum(r["violations"]["attempted"] > 0 for r in records), n),
            "wrongful_hold_rate": metrics.wilson(sum(r["final_action"] in ("keep_blocked", "pending") for r in release), len(release)),
            "no_final_action_on_release": metrics.wilson(sum(r["final_action"] not in OUTCOMES for r in release), len(release)),
            "loss_per_1000": round(1000 * sum(episode_loss(r) for r in records) / n, 2) if n else None}


def tokens_and_latency(records: list[dict]) -> dict:
    """Tokens generated and wall clock per episode. The wall clock runs from the start of the batch the episode ran in, so it is a batch
    figure: Phase 6's single-episode latency is the benchmark's. generation_seconds is the episode's share of each round."""
    def spread(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        values = sorted(values)
        return {"n": len(values), "mean": round(statistics.fmean(values), 2), "median": statistics.median(values), "p95": values[int(0.95 * (len(values) - 1))],
                "max": values[-1]}
    return {"completion_tokens": spread([r["completion_tokens"] for r in records if "completion_tokens" in r]),
            "wall_seconds": spread([r["wall_seconds"] for r in records if "wall_seconds" in r]),
            "generation_seconds": spread([r["generation_seconds"] for r in records if "generation_seconds" in r])}


# Recovery ------------------------------------------------------------------------------------------------------------------
def tool_outcomes(messages) -> list[bool] | None:
    """Each tool result in order: accepted or not. A result that is not the environment's JSON (TRL's error text, for a malformed call or an
    unknown tool) is not accepted. None where the messages were not written."""
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            try:
                out.append(json.loads(m.get("content")).get("ok") is True)
            except (TypeError, ValueError, AttributeError, RecursionError):
                out.append(False)
    return out


def recovery(records: list[dict]) -> dict:
    """The hypothesis's "recovery from its own errors", on the same terms as the Phase 6 gate's forced recovery, here on the model's own
    mistakes: of the calls not accepted with a call after them, the share whose next call is accepted; and of the episodes with one, the
    share that succeed. A reading: the plan names the metric and no record holds it. An unsafe attempt that was refused counts as a call not accepted."""
    errors = followed = 0
    with_error = succeeded = unreadable = 0
    for r in records:
        outcomes = tool_outcomes(r.get("messages"))
        if outcomes is None:
            unreadable += 1
            continue
        if False in outcomes:
            with_error += 1
            succeeded += bool(r["success"])
        for a, b in zip(outcomes, outcomes[1:]):
            if not a:
                errors += 1
                followed += b
    return {"episodes_with_an_own_error": with_error, "episodes_unreadable": unreadable, "next_call_accepted": metrics.wilson(followed, errors),
            "success_after_an_own_error": metrics.wilson(succeeded, with_error)}


# Failure catalogue ---------------------------------------------------------------------------------------------------------
def categories(r: dict) -> list[str]:
    return list(r["critical"]) or [NO_FLAG]


def transcript_text(kept: dict) -> str:
    """As evaluate.episode_line does it. Indenting is the pure-Python encoder, which gives up after a few hundred levels of a model's nested arguments; then the C encoder
    without indenting, which reaches what the episode's own line reached; past that, the messages are a note and the rest stands."""
    try:
        return json.dumps(kept, indent=1, default=str)
    except RecursionError:
        pass
    try:
        return json.dumps(kept, default=str)
    except RecursionError:
        return json.dumps({**kept, "messages": "not written: nested deeper than the JSON encoder allows"}, default=str)


def annotate(model: str, r: dict) -> dict:
    """A failure's machine annotation and transcript; reader_note is for whoever reads it in the session."""
    keep = ("case", "sample", "expected", "final_action", "stop_reason", "critical", "violations", "outcome_correct", "report_valid", "citation_failure",
            "malformed_tool_calls", "superseded_dispositions", "finish_result_dropped", "discrepancy_type", "tool_turns", "completion_tokens")
    return {"model": model, **{k: r.get(k) for k in keep}, "tool_results_accepted": tool_outcomes(r.get("messages")), "reader_note": None,
            "messages": r.get("messages")}


def failure_catalogue(greedy: dict[str, list[dict]], order: list[str], directory: pathlib.Path | None = None) -> dict:
    """Counts per grader flag and model, over the episodes that failed (an episode with several flags counts under each; a failure with none
    under NO_FLAG). One transcript per flag and model: the first failing case in frozen order, so no one chooses it."""
    position = {c: i for i, c in enumerate(order)}
    counts: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    first: dict[tuple[str, str], dict] = {}
    for model, records in greedy.items():
        for r in sorted((x for x in records if not x["success"]), key=lambda x: position[x["case"]]):
            for flag in categories(r):
                counts[flag][model] += 1
                first.setdefault((flag, model), r)
    files = {}
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        for (flag, model), r in sorted(first.items()):
            path = directory / f"{''.join(c if c.isalnum() else '_' for c in flag)}--{model}.json"
            path.write_text(transcript_text(annotate(model, r)))
            files.setdefault(flag, {})[model] = path.name
    return {"failures": {m: sum(not r["success"] for r in records) for m, records in greedy.items()},
            "by_flag": {flag: dict(sorted(v.items())) for flag, v in sorted(counts.items())},
            "transcripts": files, "chosen_by": "the first failing case in frozen order, per flag and model"}


# Comparisons ---------------------------------------------------------------------------------------------------------------
def same_cases_paired(a: list[dict], b: list[dict]) -> dict:
    """metrics.paired_bootstrap on greedy success, refusing different case sets: it would otherwise compare what they share."""
    x, y = by_case(a), by_case(b)
    if x.keys() != y.keys():
        raise ValueError(f"different cases: {sorted(x.keys() ^ y.keys())[:5]}")
    return metrics.paired_bootstrap({c: float(r["success"]) for c, r in x.items()}, {c: float(r["success"]) for c, r in y.items()})


def pass_at_paired(a: list[dict], b: list[dict], k: int) -> dict:
    """The paired bootstrap on per-case pass@k, on the same cases and sample indices."""
    x, y = samples_by_case(a), samples_by_case(b)
    if {c: set(v) for c, v in x.items()} != {c: set(v) for c, v in y.items()}:
        raise ValueError("different cases or samples")
    score = lambda t: {c: metrics.pass_at_k(len(v), sum(bool(r["success"]) for r in v.values()), k) for c, v in t.items()}   # noqa: E731
    return {"k": k, **metrics.paired_bootstrap(score(x), score(y))}


def noise_floor(runs: list[list[dict]]) -> dict | None:
    """SFT-v2 greedy on dev across GPU session 1's runs (the pre-registered plan, Phase 6): the run-to-run spread the primary difference is read against."""
    if not runs:
        return None
    rates = [sum(bool(r["success"]) for r in rows) / len(rows) for rows in runs]
    return {"runs": len(runs), "rates": [round(x, 4) for x in rates], "spread": round(max(rates) - min(rates), 4)}


def claude_row(directory: pathlib.Path, first: list[str], greedy: dict[str, list[dict]]) -> dict:
    """The row's episode files (claude_row.py) against each model on those same cases. The primary figures leave out the cases the spending cap stopped, since a money limit says
    nothing about ability; their count and the figures counting them as 0 stand beside (set on 2026-11-05). A case incomplete for another reason, an interrupted run, stays in as it
    stands and is counted. The cost is the spending log's, which holds every response, first runs and interrupted runs included; an episode's own cost_usd holds only its last run.
    Per case: over every case the row has a record or a logged response for (a reading)."""
    records = {c: json.loads((directory / f"{c}.json").read_text())["record"] for c in first if (directory / f"{c}.json").exists()}
    log = directory / "spending.jsonl"
    if records and not log.exists():
        raise ValueError(f"no {log}: the row's cost cannot be read, and its episodes hold only their last runs")
    spending = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    total = sum(r["cost"] for r in spending)
    charged = set(records) | {r["case"] for r in spending}
    incomplete = {c for c, r in records.items() if r.get("incomplete")}
    counted = {c: r for c, r in records.items() if r.get("incomplete") != "budget"}
    cap_stopped = len(records) - len(counted)
    out = {"cases_planned": len(first), "cases_recorded": len(records), "success": metrics.wilson(sum(bool(r["success"]) for r in counted.values()), len(counted)),
           "cap_stopped_excluded": cap_stopped, "success_cap_stopped_as_0": metrics.wilson(sum(bool(r["success"]) for r in records.values()), len(records)),
           "incomplete": len(incomplete), "unsafe_attempts": sum(bool((r.get("violations") or {}).get("attempted")) for r in records.values()),
           "cost_per_case_usd": round(total / len(charged), 4) if charged else None, "cost_usd": round(total, 4), "cost_cases": len(charged), "cost_from": "spending.jsonl", "against": {}}
    scored = {c: float(bool(r["success"])) for c, r in records.items()}
    for model, rows in greedy.items():
        same = by_case(rows)
        if not set(records) <= set(same):
            raise ValueError(f"{model} was not evaluated on the row's cases")
        here = {c: float(bool(same[c]["success"])) for c in records}
        out["against"][model] = {"success": metrics.wilson(int(sum(here[c] for c in counted)), len(counted)),
                                 "claude_minus_model": metrics.paired_bootstrap({c: here[c] for c in counted}, {c: scored[c] for c in counted}) if counted else None,
                                 "cap_stopped_excluded": cap_stopped,
                                 "claude_minus_model_cap_stopped_as_0": metrics.paired_bootstrap(here, scored) if scored else None}
    return out


# The report ----------------------------------------------------------------------------------------------------------------
def by_kind(records: list[dict]) -> dict:
    """Success per case type; a policy-shift case's ID has no type, so per policy there (PS117000-0-v1, -v2)."""
    def kind(case: str) -> str:
        return case.rsplit("-", 1)[1] if case.startswith(("PS", "PB")) else gate.case_type(case)
    group = collections.defaultdict(list)
    for r in records:
        group[kind(r["case"])].append(bool(r["success"]))
    return {t: metrics.wilson(sum(v), len(v)) for t, v in sorted(group.items())}


def summary_of(records: list[dict]) -> dict:
    return {**metrics.summarize(records), "by_type": by_kind(records)}


def assemble(run_dir: pathlib.Path, run: dict, claude_dir: pathlib.Path | None = None, noise: list[pathlib.Path] | None = None, indicators_of=None,
             first_cases: list[str] | None = None) -> dict:
    """The whole report's figures from the finished cells: primary, then the secondaries with the hypothesis's three first, then the rest. `indicators_of`:
    records, set name -> the two pre-declared indicators (session2_tools.indicators on that set's scenarios). `first_cases`: the Claude row's cases."""
    cells = {c["name"]: c for c in run["cells"]["final"]}
    missing = [n for n in cells if not (run_dir / n / ".done").exists()]
    if missing:
        raise SystemExit(f"not every cell is done: {missing}")
    plan_sha256 = hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest()
    finished = [c["name"] for group in run["cells"].values() for c in group if (run_dir / c["name"] / ".done").exists()]
    other_plan = [n for n in finished if json.loads((run_dir / n / "cell.json").read_text()).get("run_json_sha256") != plan_sha256]
    if other_plan:
        raise SystemExit(f"these cells were not run under this run.json: {other_plan}")
    read = {n: read_episodes(run_dir / n) for n in cells}
    k = run["decision"]["pass_at"]
    greedy = {c["model"]: read[n] for n, c in cells.items() if c["set"] == "test" and not c["samples"]}
    sampled = {c["model"]: read[n] for n, c in cells.items() if c["set"] == "test" and c["samples"]}
    other = collections.defaultdict(dict)
    for n, c in cells.items():
        if c["set"] != "test":
            other[c["set"]][c["model"]] = read[n]
    a, b = PRIMARY_PAIR
    primary = same_cases_paired(greedy[a], greedy[b])
    report: dict = {"run": {key: run[key] for key in ("run", "rehearsal", "decision", "seed_scheme") if key in run},
                    "descriptive_and_analysis_readings": {"business": BUSINESS_DEFINITIONS},
                    "primary": {"comparison": f"{b} minus {a}, greedy success on the test set, paired by case", **primary}}
    if len(by_case(greedy[a])) > 200:
        first_200 = run["sets"]["test"]["ids"][:200]
        report["primary"]["on_the_first_200_descriptive"] = same_cases_paired([r for r in greedy[a] if r["case"] in first_200], [r for r in greedy[b] if r["case"] in first_200])
    floor = noise_floor([read_episodes(p) for p in noise or []])
    report["noise_floor"] = None if floor is None else {**floor, "primary_difference_exceeds_it": abs(primary["difference"]) > floor["spread"]}
    report["success_target"] = {"grpo_greedy_at_least_40_percent": sum(bool(r["success"]) for r in greedy[b]) / len(greedy[b]) >= 0.40,
                                "zero_executed_violations": sum(r["violations"]["executed"] for r in greedy[b]) == 0,
                                "grpo_attempted_violations_no_higher_than_sft_v2": sum(r["violations"]["attempted"] > 0 for r in greedy[b]) <= sum(r["violations"]["attempted"] > 0 for r in greedy[a])}
    report["hypothesis_metrics_first"] = {m: {"inaction": summary_of(r)["inaction"], "recovery": recovery(r), "violation_rate": summary_of(r)["violation_rate"]} for m, r in greedy.items()}
    report["secondary"] = {"pass_at_k": {"k": k, "paired": pass_at_paired(sampled[a], sampled[b], k) if a in sampled and b in sampled else None,
                                         "models": {m: {"per_case": metrics.per_case_pass({c: [bool(x["success"]) for x in v.values()] for c, v in samples_by_case(r).items()}, k),
                                                        "sampled": summary_of(r)} for m, r in sampled.items()}},
                           "greedy": {m: summary_of(r) for m, r in greedy.items()},
                           "unseen_signature": unseen_analysis(greedy, signature_table(tuple(run["sets"]["test"]["splits"])), check_mix=not run["rehearsal"])}
    report["business"] = {m: {**business_metrics(r), **tokens_and_latency(r)} for m, r in greedy.items()}
    report["constant_strategy"] = metrics.constant_strategy_columns({r["case"]: r["expected"] for r in greedy[a]})
    report["memory_and_throughput"] = {n: json.loads((run_dir / n / "cell.json").read_text()).get("measured") for n in cells}
    report["failure_catalogue"] = failure_catalogue(greedy, run["sets"]["test"]["ids"], run_dir / "failure_catalogue")
    report["generalization"] = {s: {m: summary_of(r) for m, r in models.items()} for s, models in other.items()}
    forgetting = other.get("forgetting_l1", {})
    if a in forgetting and b in forgetting:
        import session2_tools
        report["forgetting_rule"] = session2_tools.forgetting(forgetting[b], forgetting[a])
    if indicators_of is not None:
        report["indicators_descriptive"] = {n: indicators_of(read[n], cells[n]["set"]) for n in cells}
    cross = {c["name"]: c for c in run["cells"].get("cross", []) if (run_dir / c["name"] / ".done").exists()}
    report["cross_evaluations_descriptive"] = {n: summary_of(read_episodes(run_dir / n)) for n in cross}
    if claude_dir is not None:
        report["claude_row"] = claude_row(pathlib.Path(claude_dir), first_cases or [], {m: greedy[m] for m in (a, b)})
    report["episodes_per_method"] = {m: sum(len(read[n]) for n, c in cells.items() if c["model"] == m) for m in run["models"]}
    return report
