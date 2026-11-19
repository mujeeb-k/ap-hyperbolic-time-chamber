"""Phase 10 (final evaluation) driver for every model x set x greedy/pass@k cell, each in its own process.

It ran in GPU session 3 (preregistration_proof/PLAN_V2.md, Phase 10). Each cell is its own process because a vLLM engine does not give its memory back.
The final cells and the cross-evaluations are separate parts.

The configuration comes from Phase 8's decision.json (pass_at, test_cases, rft_test, rft_samples), never from here; the selected GRPO checkpoint is an input.

    Primary forms (pre-data amendment, 2026-11-03): SFT-v2 as its merged bf16 model; GRPO and RFT as the merged SFT-v2 plus the adapter at runtime
    through vLLM's LoRA support. Base greedy only. SFT-v2, GRPO and RFT greedy and pass@k, the same k. V6, policy_shift and forgetting_l1 greedy only, for
    base, SFT-v2 and GRPO. Cross-evaluations (GRPO and RFT bf16-merged; SFT-v2's adapter on the base model, if saved) are their own opt-in part, descriptive.

    Seeds (set on 2026-11-04): every sampled turn's seed is keyed by case ID, sample index and turn (`sample_seed`), so it is the same for every model and does
    not depend on which cases run beside it. Greedy has no seed. The frozen backend's rule (sample and turn only) stays as it is for what it already ran.

    One sitting: `prepare` writes run.json once, with the decision, the inputs' sha256 and the code's. Every cell checks them first, so all cells run the same
    code on the same weights. A cell with .done is not repeated; an interrupted cell's files are kept aside and the cell runs again whole. A bug found after
    these runs: fix it, run every model again in a new run directory (FINAL_RUN=2), and report both (the pre-registered plan, Phase 10).

    What a change costs. `code_files` is what decides results: all of src/ap_sim and the scripts a result passes through. A change there refuses every later step,
    and the routes are to restore it or to start a new run, which repeats every cell. final_analysis.py only reads finished episodes, so it is outside that
    list: a fix to it needs no paid cell again. `report` checks everything else and writes both hashes of that file into results.json, so the record shows it.

    python scripts/gpu/final_evaluation.py prepare --phase10 --decision D --selected S --baselines B --grpo-checkpoint CKPT --sft-merged M --rft-dir R --out RUN
    python scripts/gpu/final_evaluation.py preflight --dir RUN --form grpo    # dev only: the adapter really applied, and the throughput
    python scripts/gpu/final_evaluation.py budget --dir RUN --part final --spent X --price P --volume V
    python scripts/gpu/final_evaluation.py verify-code --dir RUN
    python scripts/gpu/final_evaluation.py cell --dir RUN --name grpo/test-greedy
    python scripts/gpu/final_evaluation.py report --dir RUN [--claude DIR] [--noise DIR ...]
    python scripts/gpu/final_evaluation.py claude-row --out DIR               # prints the row's command; the row runs locally, by hand
    python scripts/gpu/final_evaluation.py prepare --rehearsal --backend reference --decision D --out RUN    # dev only, scripted: no model at all

    The follow-up study, all greedy. It was pre-registered on its own on 2026-11-06 and is exploratory. It has its own plan, pinned inputs and stop rule
    (the section above main).
    python scripts/gpu/final_evaluation.py prepare-followup --sft-merged M --rft-dir R --grpo-20261101 CKPT --seeds-dir FOLLOWUP --out RUN
    python scripts/gpu/final_evaluation.py preflight --dir RUN --form grpo-20261107 --adapter-only
    python scripts/gpu/final_evaluation.py followup-budget --dir RUN --spent X --price P       # may cut cells not yet run, in the pre-registered order
    python scripts/gpu/final_evaluation.py followup-cells --dir RUN                           # the plan's cells, less any cut
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/gpu"), str(ROOT / "scripts/phase2"), str(ROOT / "scripts/phase4"), str(ROOT / "scripts/phase5")]

import budget_forecast   # noqa: E402
import final_analysis   # noqa: E402
import session2_tools   # noqa: E402

SEED_SCHEME = "sha256 of the JSON text of [case, sample, turn], first 8 bytes big-endian, top bit cleared"
SEED_MASK = 2 ** 63 - 1
TRAINED = ("sft-v2", "grpo", "rft")          # greedy and pass@k, at the same k
GREEDY_ONLY = ("base",)
SLICES = ("v6", "policy_shift", "forgetting_l1")
SLICE_MODELS = ("sft-v2", "grpo", "base")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TEST_SIZES = (200, 300)
PASS_AT = (4, 8)
REHEARSAL_CASES = {"test": 12, "v6": 2, "policy_shift": 2, "forgetting_l1": 2, "test_reserve": 2, "stress": 2}
PREFLIGHT_GREEDY_CASES = 30
CHECK_MAX_TOKENS, CHECK_PROMPTS, SEED_CHECK_MAX_TOKENS = 8, 4, 32
SEED_SPREAD_TEMPERATURE, SEED_SPREAD_DRAWS = 2.0, 3      # hot enough that seeds must diverge however peaked the model is: the check is of the engine's seeds, not of the model
# Episodes in one cell that failed in the infrastructure twice, above which the cell stops before .done: a fault that hits one model's cell would read as that model's failures.
# Set on 2026-11-05: more than 10 still failing after their re-run, and the cell runs again whole. It looks only at infrastructure failures, never at outcomes.
SYSTEMATIC_FAULT_LIMIT = 10
# Scripts a result passes through. With all of src/ap_sim (`code_files`) this is what decides a result: checked before every cell and by the report, against what prepare recorded.
# The code that trained is checked too by session2_tools.verify.
EVALUATION_SCRIPTS = ("scripts/gpu/final_evaluation.py", "scripts/gpu/score_both_ways.py", "scripts/gpu/grpo_probe.py", "scripts/gpu/gpu_cleanup.py", "scripts/gpu/session1_tools.py",
                      "scripts/phase5/evaluate.py", "scripts/phase5/claude_row.py", "scripts/phase6/gate.py", "scripts/strong_model_sweep.py", "scripts/strong_model_check.py",
                      "scripts/budget_forecast.py", "scripts/gpu/followup_eval.sh", *session2_tools.FROZEN_CODE)   # followup_eval.sh holds half the follow-up's stop rule
ANALYSIS_ONLY = "scripts/gpu/final_analysis.py"     # reads finished episodes: outside the check, its hash in results.json (the module docstring)


def sample_seed(case: str, sample: int, turn: int) -> int:
    """A sampled turn's seed: from the case ID, the sample index and the turn, nothing else. Not Python's hash(), which changes between processes."""
    text = json.dumps([case, sample, turn], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & SEED_MASK


class KeyedSeeds:
    """The engine, as the backend sees it: a sampled request's seed replaced by `sample_seed` just before it goes in. Every other setting stays the frozen backend's
    own, from its own code. A greedy request has no seed and stays without one."""

    def __init__(self, engine, owner):
        self.engine, self.owner = engine, owner

    def generate(self, prompts, sampling_params=None, **kwargs):
        for episode, params in zip(self.owner.episodes_in_round, sampling_params, strict=True):
            if params.seed is not None:
                params.seed = sample_seed(episode.case, episode.sample, len(episode.turns))
        return self.engine.generate(prompts, sampling_params=sampling_params, **kwargs)


def backend_class(adapter: str | None):
    """The frozen vLLM backend, or the runtime-LoRA backend of GPU session 2's gap probe (the same engine settings plus LoRA), with the seeds keyed by case."""
    from ap_sim.evaluation import rollout
    import score_both_ways
    parent = rollout.VLLMBackend if adapter is None else score_both_ways.lora_backend(adapter)

    class Phase10Backend(parent):
        name = f"{parent.name}-keyed-seeds"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.llm = KeyedSeeds(self.llm, self)

        def generate(self, episodes):
            self.episodes_in_round = episodes
            return super().generate(episodes)

        def info(self):
            return {**super().info(), "seeds": SEED_SCHEME}
    return Phase10Backend


def make_backend(form: dict, samples: int, gpu_memory_utilization: float = 0.85):
    import evaluate
    return backend_class(form.get("adapter"))(form["model"], form.get("revision"), gpu_memory_utilization=gpu_memory_utilization,
                                              temperature=evaluate.SAMPLE_TEMPERATURE if samples else 0.0, top_p=evaluate.SAMPLE_TOP_P)


def innermost(llm):
    while hasattr(llm, "engine") or hasattr(llm, "llm"):
        llm = getattr(llm, "engine", None) or llm.llm
    return llm


def engine_figures(llm) -> dict:
    """What memory per concurrent episode needs from the engine. vLLM reserves its share of the GPU at the start, so whole-GPU memory says little by itself."""
    try:
        return {"max_num_seqs": innermost(llm).llm_engine.vllm_config.scheduler_config.max_num_seqs}
    except AttributeError:
        return {"max_num_seqs": None}


# The decision and the plan -------------------------------------------------------------------------------------------------
def read_decision(path: pathlib.Path) -> dict:
    """Phase 8's decision, refused unless it says exactly what the driver can run."""
    d = json.loads(pathlib.Path(path).read_text())
    if d.get("decision") != "proceed":
        raise SystemExit(f"decision.json says {d.get('decision')!r}, not proceed")
    missing = [k for k in ("pass_at", "test_cases", "rft_test", "rft_samples", "approved_cap") if k not in d]
    if missing:
        raise SystemExit(f"decision.json lacks {missing}")
    if d["pass_at"] not in PASS_AT or isinstance(d["pass_at"], bool):
        raise SystemExit(f"pass_at {d['pass_at']!r} is not one of {PASS_AT}")
    if d["test_cases"] not in TEST_SIZES or isinstance(d["test_cases"], bool):
        raise SystemExit(f"test_cases {d['test_cases']!r} is not one of {TEST_SIZES}: the test set has only those frozen forms")
    if not isinstance(d["rft_test"], bool):
        raise SystemExit(f"rft_test {d['rft_test']!r} is not true or false")
    if d["rft_test"] and d["rft_samples"] != d["pass_at"]:
        raise SystemExit(f"rft_samples {d['rft_samples']!r} is not pass_at {d['pass_at']}: the trained models are sampled at the same k")
    return d


def cell(model: str, the_set: str, samples: int = 0, descriptive: bool = False) -> dict:
    return {"name": f"{model}/{the_set}-{f'samples-{samples}' if samples else 'greedy'}", "model": model, "set": the_set, "samples": samples, "descriptive": descriptive}


def plan_cells(decision: dict, sft_adapter_saved: bool = False) -> dict:
    """The matrix, the primary comparison first. The order is fixed before anything runs and nothing about it depends on a result."""
    k = decision["pass_at"]
    rft = decision["rft_test"]
    final = [cell("sft-v2", "test"), cell("grpo", "test"), cell("base", "test")] + ([cell("rft", "test")] if rft else [])
    final += [cell("sft-v2", "test", k), cell("grpo", "test", k)] + ([cell("rft", "test", k)] if rft else [])
    final += [cell(m, s) for s in SLICES for m in SLICE_MODELS]
    cross = [cell("grpo-merged", "test", descriptive=True)] + ([cell("rft-merged", "test", descriptive=True)] if rft else [])
    cross += [cell("sft-adapter", "test", descriptive=True)] if sft_adapter_saved else []
    return {"final": final, "cross": cross}


def forecast_episodes(decision: dict) -> int:
    """The episodes Phase 8's rule priced (budget_forecast): the plan's models and sets at the decision's k and test size, and RFT's beside them."""
    p, n, k = budget_forecast.PLAN, decision["test_cases"], decision["pass_at"]
    test = n * (p["final_models_greedy"] + p["final_models_sampled"] * k)
    rft = budget_forecast.RFT_TEST_MODELS * n * (1 + k) if decision["rft_test"] else 0
    slices = p["slice_models"] * sum(p["slice_cases"].values())
    return test + rft + slices + 3 * p["regression_cases"]         # 3: phase10_regression_models in budget_forecast.after_session_1


def set_ids(the_set: str, decision: dict, rehearsal: bool) -> tuple[list[str], list[str]]:
    """(the set's case IDs in frozen order, the splits they come from). Read from the frozen loaders (no model output involved). In a rehearsal every set is dev."""
    if rehearsal:
        from ap_sim.generator.splits import load_split
        dev, n = list(load_split("dev")), REHEARSAL_CASES[the_set]
        return dev[::len(dev) // n][:n], ["dev"]            # spread over dev's order, which runs type by type
    return list(load_set(the_set, decision)), (["test", "test_reserve"] if the_set == "test" and decision["test_cases"] == 300 else [the_set])


def load_set(the_set: str, decision: dict) -> dict:
    import evaluate
    if the_set == "test":
        from ap_sim.generator.splits import load_test
        return load_test(extended=decision["test_cases"] == 300)
    if the_set == "stress":       # the follow-up's messiness set, only as frozen, and only the freeze pinned here
        from ap_sim.stress import messiness
        check_stress_freeze()
        return messiness.load(messiness.MANIFEST, require_frozen=True)
    return evaluate.load_cases(the_set, phase10=True)


def load_for_cell(run: dict, the_set: str) -> dict:
    """The cell's cases, the same ones prepare recorded, in the same order."""
    if run["rehearsal"]:
        from ap_sim.generator.splits import load_split
        dev = load_split("dev")
        cases = {c: dev[c] for c in run["sets"][the_set]["ids"]}
    else:
        cases = load_set(the_set, run["decision"])
    if list(cases) != run["sets"][the_set]["ids"]:
        raise SystemExit(f"{the_set}: the cases are not the ones prepare recorded")
    return cases


# Inputs --------------------------------------------------------------------------------------------------------------------
def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2 ** 22), b""):
            h.update(block)
    return h.hexdigest()


def adapter_hashes(directory: pathlib.Path) -> dict:
    """The two files vLLM reads of an adapter, refused unless they are this experiment's LoRA (rank 32, alpha 64)."""
    missing = [f for f in ADAPTER_FILES if not (directory / f).is_file()]
    if missing:
        raise SystemExit(f"{directory} is not a whole adapter: no {missing}")
    config = json.loads((directory / "adapter_config.json").read_text())
    if (config.get("r"), config.get("lora_alpha")) != (32, 64):
        raise SystemExit(f"{directory}: rank {config.get('r')}, alpha {config.get('lora_alpha')}, not this experiment's 32 and 64")
    return {f: file_sha256(directory / f) for f in ADAPTER_FILES}


def model_hashes(directory: pathlib.Path) -> dict:
    """Every file of a merged model directory: name to sha256."""
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not any(p.suffix == ".safetensors" for p in files):
        raise SystemExit(f"{directory} holds no weights")
    return {str(p.relative_to(directory)): file_sha256(p) for p in files}


def code_files() -> list[str]:
    source = [str(p.relative_to(ROOT)) for p in (ROOT / "src" / "ap_sim").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return sorted({*source, *EVALUATION_SCRIPTS} - {ANALYSIS_ONLY})


def code_hashes() -> dict:
    return {f: file_sha256(ROOT / f) for f in code_files()}


def adapter_trained(directory: pathlib.Path) -> bool:
    """A nonzero LoRA B matrix, as the merge script's check asks: an adapter of zeros would evaluate SFT-v2 under another name."""
    from safetensors import safe_open
    with safe_open(str(directory / ADAPTER_FILES[1]), framework="pt") as f:
        return any("lora_B" in key and bool(f.get_tensor(key).abs().sum() > 0) for key in f.keys())


def checked_adapter(directory: pathlib.Path) -> dict:
    hashes = adapter_hashes(directory)
    if not adapter_trained(directory):
        raise SystemExit(f"{directory}: no nonzero lora_B: this adapter is a no-op")
    return hashes


def git_head() -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def build_forms(a, run_dir: pathlib.Path) -> tuple[dict, dict]:
    """The form each name is evaluated in, and the sha256 of what it loads."""
    import evaluate
    merged, checkpoint = pathlib.Path(a.sft_merged), pathlib.Path(a.grpo_checkpoint)
    rft = pathlib.Path(a.rft_dir)
    hashes = {"sft_merged": model_hashes(merged), "grpo_adapter": checked_adapter(checkpoint)}
    forms = {"base": {"model": evaluate.MODEL, "revision": evaluate.REVISION},
             "sft-v2": {"model": str(merged)},
             "grpo": {"model": str(merged), "adapter": str(checkpoint)},
             "grpo-merged": {"model": str(run_dir / "grpo-merged" / "model")}}
    forms["rft"] = {"model": str(merged), "adapter": str(rft / "adapter")}
    forms["rft-merged"] = {"model": str(rft / "merged")}
    hashes["rft_adapter"] = checked_adapter(rft / "adapter") if a.rft_needed else None
    if a.sft_adapter and (pathlib.Path(a.sft_adapter) / ADAPTER_FILES[1]).is_file():
        forms["sft-adapter"] = {"model": evaluate.MODEL, "revision": evaluate.REVISION, "adapter": a.sft_adapter}
        hashes["sft_adapter"] = checked_adapter(pathlib.Path(a.sft_adapter))
    return forms, hashes


def merge_record(grpo: pathlib.Path, step: int) -> pathlib.Path:
    """The evaluated checkpoint's merge.json, from GPU session 2's own directories (its checkpoint evaluation keeps it in eval-N). The main run starts from a copy of the pilot
    winner's step 10 and does not copy its evaluation, so a selected step 10's record is in the winner's directory. Never relative to the input, which is what is being checked."""
    path = grpo / "main" / f"eval-{step}" / "merge.json"
    if not path.exists() and step == 10:
        winner = float(json.loads((grpo / "pilot_winner.json").read_text())["winner"])
        for d in grpo.glob("pilot-*"):
            if d.name.split("pilot-", 1)[1][0].isdigit() and float(d.name.split("pilot-", 1)[1]) == winner:
                path = d / "eval-10" / "merge.json"
    if not path.exists():
        raise SystemExit(f"no merge record at {path}: the selected checkpoint's adapter cannot be checked against what was evaluated")
    return path


def check_selection(selected: pathlib.Path, checkpoint: pathlib.Path, baselines: pathlib.Path) -> dict:
    """The input must be the checkpoint the pre-declared rule selected. selected.json's step, in the directory's name; the adapter's sha256 equal to what that step's evaluation
    merged (a checkpoint of the same number from another directory is not it); and the selection itself derived again from the rows, by the frozen rule."""
    record = json.loads(selected.read_text())
    step = record.get("selected")
    if step is None or checkpoint.name != f"checkpoint-{step}":
        raise SystemExit(f"{checkpoint} is not the selected checkpoint: {selected} says {step!r}")
    grpo = selected.parent
    merged_from = merge_record(grpo, step)
    evaluated = json.loads(merged_from.read_text())["adapter_files_sha256"]
    here = adapter_hashes(checkpoint)
    if {f: evaluated.get(f) for f in ADAPTER_FILES} != here:
        raise SystemExit(f"{checkpoint} is not the adapter step {step} evaluated: its sha256 differs from {merged_from}")
    rows = session2_tools.read_rows(grpo / "main" / "checkpoints.jsonl")
    derived = session2_tools.select_checkpoint(rows, json.loads(baselines.read_text()))["selected"]
    if derived != step:
        raise SystemExit(f"{selected} says step {step}, and the rule on {grpo / 'main' / 'checkpoints.jsonl'} gives step {derived!r}")
    return {"step": step, "sha256": file_sha256(selected), "merge_record": str(merged_from), "derived_again": derived}


def prepare(a) -> dict:
    run_dir = pathlib.Path(a.out)
    if a.phase10 == a.rehearsal:
        raise SystemExit("say which: --phase10 (the held-out sets) or --rehearsal (dev only)")
    if a.phase10 and a.backend != "vllm":
        raise SystemExit("a scripted backend runs only in a rehearsal")
    decision = read_decision(pathlib.Path(a.decision))
    a.rft_needed = decision["rft_test"]
    selected = None
    if a.backend == "vllm":
        if not (a.grpo_checkpoint and a.sft_merged and a.rft_dir):
            raise SystemExit("--grpo-checkpoint, --sft-merged and --rft-dir are needed for a model run")
        if a.phase10:
            if not a.selected or not a.baselines:
                raise SystemExit("--selected and --baselines are needed: the checkpoint must be the one the rule selected")
            selected = check_selection(pathlib.Path(a.selected), pathlib.Path(a.grpo_checkpoint), pathlib.Path(a.baselines))
        forms, hashes = build_forms(a, run_dir)
    else:
        forms, hashes = {m: {"model": a.backend} for m in ("base", "sft-v2", "grpo", "rft", "grpo-merged", "rft-merged")}, {}
    cells = plan_cells(decision, "sft-adapter" in forms)
    sets = {}
    for the_set in {c["set"] for group in cells.values() for c in group}:
        ids, splits = set_ids(the_set, decision, a.rehearsal)
        sets[the_set] = {"ids": ids, "splits": splits}
    planned = sum(len(sets[c["set"]]["ids"]) * max(c["samples"], 1) for c in cells["final"])
    if not a.rehearsal and planned != forecast_episodes(decision):
        raise SystemExit(f"{planned} episodes planned, {forecast_episodes(decision)} priced by the budget rule")
    run = {"freeze": "phase10/1", "run": run_dir.name, "rehearsal": a.rehearsal, "backend": a.backend, "decision": decision, "decision_sha256": file_sha256(pathlib.Path(a.decision)),
           "selected": selected, "forms": forms, "input_sha256": hashes, "models": sorted({c["model"] for c in cells["final"]}), "sets": sets, "cells": cells,
           "episodes_planned": planned, "seed_scheme": SEED_SCHEME, "code_sha256": code_hashes(), "analysis_sha256": file_sha256(ROOT / ANALYSIS_ONLY), "commit": git_head()}
    return write_run(run_dir, run)


def write_run(run_dir: pathlib.Path, run: dict) -> dict:
    """run.json, written once. The same plan again returns what is there; a different one is refused."""
    path = run_dir / "run.json"
    if path.exists():
        before = json.loads(path.read_text())
        # Not the commit: a commit that changes no hashed file changes nothing here. Not the analysis: results.json records that (the module docstring).
        changed = sorted(k for k in set(before) | set(run) if k not in ("commit", "analysis_sha256") and before.get(k) != run.get(k))
        if changed:
            raise SystemExit(f"{path} already holds this run with a different {changed}. Either restore what changed, or start a new run directory (FINAL_RUN), which repeats every cell")
        return before
    leftovers = sorted(str(p) for p in run_dir.rglob("*") if p.is_file() and p.name != "run.json.tmp") if run_dir.exists() else []
    if leftovers:
        raise SystemExit(f"{run_dir} holds files but no run.json, starting with {leftovers[0]}: nothing says what they were run on. Restore run.json, or start a new run directory (FINAL_RUN)")
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(run, indent=1))
    temporary.replace(path)
    return run


def load_run(run_dir: pathlib.Path) -> dict:
    path = run_dir / "run.json"
    if not path.exists():
        raise SystemExit(f"no {path}: run prepare first")
    return json.loads(path.read_text())


def verify_code(run: dict) -> None:
    """Refuses if anything that decides a result is not what prepare recorded, a file added or removed included."""
    now = code_hashes()
    changed = sorted(f for f in set(now) | set(run["code_sha256"]) if now.get(f) != run["code_sha256"].get(f))
    if changed:
        raise SystemExit(f"refusing to run: the code differs from run.json: {changed}. Either restore it (run.json records commit {run.get('commit')}), or start a new run directory (FINAL_RUN), which repeats every cell")


def adapter_key(form_name: str) -> str:
    """Where run.json keeps a form's adapter sha256: grpo_adapter, rft_adapter, sft_adapter (Phase 10), grpo_20261107_adapter (the follow-up)."""
    return {"sft-adapter": "sft_adapter"}.get(form_name, form_name.replace("-", "_") + "_adapter")


def verify_run(run: dict, form_name: str) -> None:
    """Refuses if the code, this form's adapter, or the merged SFT-v2 under it (the base of the primary comparison) is not what prepare recorded."""
    verify_code(run)
    if run["backend"] != "vllm":
        return
    form, key = run["forms"][form_name], adapter_key(form_name)
    if form.get("adapter") and key not in run["input_sha256"]:
        raise SystemExit(f"refusing to run: run.json holds no sha256 for {form_name}'s adapter")
    if form.get("adapter") and adapter_hashes(pathlib.Path(form["adapter"])) != run["input_sha256"].get(key):
        raise SystemExit(f"refusing to run: {form['adapter']} differs from run.json")
    if form["model"] == run["forms"]["sft-v2"]["model"] and model_hashes(pathlib.Path(form["model"])) != run["input_sha256"]["sft_merged"]:
        raise SystemExit(f"refusing to run: the merged SFT-v2 at {form['model']} differs from run.json")


def all_cells(run: dict) -> dict:
    return {c["name"]: c for group in run["cells"].values() for c in group}


# One cell ------------------------------------------------------------------------------------------------------------------
def set_aside(out: pathlib.Path) -> pathlib.Path | None:
    """An interrupted cell's files, kept (logs never overwrite, set on 2026-11-04). Nobody reads them to choose anything; the cell runs again whole."""
    if not out.exists():
        return None
    n = 1
    while (aside := out.with_name(f"{out.name}.interrupted-{n}")).exists():
        n += 1
    out.rename(aside)
    return aside


def run_cell(run_dir: pathlib.Path, name: str, gpu_memory_utilization: float = 0.85) -> dict:
    run = load_run(run_dir)
    cells = all_cells(run)
    if name not in cells:
        raise SystemExit(f"{name} is not in run.json's plan")
    this = cells[name]
    if run.get("freeze") == "followup/1" and name in {n for c in read_cuts(run_dir) for n in c["cells"]}:
        raise SystemExit(f"{name} was cut by the stop rule (cuts.json): a cut stays cut")          # enforced here, not only in the pod script's loop
    out = run_dir / name
    if (out / ".done").exists():
        print(f"{name}: done already")
        return json.loads((out / "cell.json").read_text())
    verify_run(run, this["model"])
    aside = set_aside(out)
    import evaluate
    import grpo_probe
    form = run["forms"][this["model"]]
    cases = load_for_cell(run, this["set"])
    tok = evaluate.tokenizer(local_only=run["backend"] != "vllm")
    started = time.time()
    backend = make_backend(form, this["samples"], gpu_memory_utilization) if run["backend"] == "vllm" else None
    with grpo_probe.GpuMemory() as memory:
        res = evaluate.evaluate(cases, run["backend"], tok, samples=this["samples"], out=out, backend=backend)
    want = len(cases) * max(this["samples"], 1)
    keys = {(r["case"], r.get("sample")) for r in res["records"]}
    if len(res["records"]) != want or len(keys) != want:
        raise RuntimeError(f"{name}: {len(res['records'])} records, {len(keys)} distinct, for {want} episodes")
    faulted = sum(bool(r.get("counted_as_failure")) for r in res["records"])
    if faulted > SYSTEMATIC_FAULT_LIMIT:
        raise RuntimeError(f"{name}: {faulted} episodes failed in the infrastructure twice, more than {SYSTEMATIC_FAULT_LIMIT}: a systematic fault, not this model's result. "
                           "The cell is not done, its files are kept, and running the part again sets them aside and runs it whole")
    timing = res["summary"]["timing"]
    measured = {"peak_gpu_mib": memory.peak_mib, "peak_gpu_note": "the whole GPU, by nvidia-smi: vLLM reserves its share at the start, so this is that reservation over max_num_seqs, not what an episode uses",
                **(engine_figures(backend.llm) if backend else {}), "episodes": want,
                "episodes_per_hour": timing.get("episodes_per_hour"), "wall_seconds_with_engine_start": round(time.time() - started, 1)}
    measured["peak_gpu_mib_per_concurrent_episode"] = (round(memory.peak_mib / min(want, measured["max_num_seqs"]), 2) if measured.get("max_num_seqs") else None)
    record = {**this, "form": form, "commit": git_head(), "run_json_sha256": file_sha256(run_dir / "run.json"), "interrupted_run_kept_at": str(aside) if aside else None, "measured": measured,
              "infrastructure": res["summary"]["infrastructure"]}
    (out / "cell.json").write_text(json.dumps(record, indent=1))
    (out / ".done").touch()
    return record


# Spending ------------------------------------------------------------------------------------------------------------------
def measured_rate(run_dir: pathlib.Path, run: dict) -> tuple[float, str]:
    """The evaluation rate to forecast with: this run's finished cells, else the rate the cap was priced at. Not the preflight's: it runs a few dozen episodes, and the rate
    depends on the batch (GPU session 1: 400 episodes an hour at batch 16, 704.5 at 64, 371 at 200), so it would say little about a whole test set, and could stop the session on nothing."""
    done = [json.loads((run_dir / c["name"] / "cell.json").read_text())["measured"] for c in all_cells(run).values() if (run_dir / c["name"] / ".done").exists()]
    seconds = sum(m["wall_seconds_with_engine_start"] for m in done)
    if seconds > 0:
        return sum(m["episodes"] for m in done) / seconds * 3600, "this run's cells"
    return budget_forecast.EVAL_EPISODES_PER_HOUR, "the budget rule"


def budget(run_dir: pathlib.Path, part: str, spent: float, price: float, volume: float, cap: float | None, above_ceiling: bool = False) -> dict:
    if cap is not None and cap > budget_forecast.RAISED_CAP_LIMIT and not above_ceiling:
        raise SystemExit(f"--cap {cap} is above the ${budget_forecast.RAISED_CAP_LIMIT} ceiling: only the user's ruling moves that, with --override-ceiling")
    run = load_run(run_dir)
    left = [c for c in run["cells"][part] if not (run_dir / c["name"] / ".done").exists()]
    episodes = sum(len(run["sets"][c["set"]]["ids"]) * max(c["samples"], 1) for c in left)
    rate, source = measured_rate(run_dir, run)
    total = round(spent + episodes / rate * price + volume, 2)
    limit = run["decision"]["approved_cap"] if cap is None else cap
    return {"spent": spent, "episodes_left": episodes, "episodes_per_hour": round(rate, 1), "rate_from": source, "forecast_total": total, "cap": limit, "within_cap": total <= limit}


# The preflight -------------------------------------------------------------------------------------------------------------
def check_params(prompts: list[list[int]], **settings) -> list:
    from vllm import SamplingParams

    from ap_sim.evaluation import rollout
    return [SamplingParams(n=1, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0, stop_token_ids=[rollout.IM_END, rollout.ENDOFTEXT],
                           **{k: (v[i] if isinstance(v, list) else v) for k, v in settings.items()}) for i in range(len(prompts))]


def adapter_applied(backend, prompts: list[list[int]]) -> dict:
    """A runtime adapter that silently did nothing would evaluate SFT-v2 under another's name. The same prompts greedy, with log-probabilities: twice without the adapter (the control:
    what repeating alone moves, with the prefix cache warm) and once with it. The adapter's difference must exceed the control's. Log-probabilities the engine did not give are
    a stop with their own message, never a reading of no difference. Dev prompts only."""
    requests = [{"prompt_token_ids": p} for p in prompts]

    def greedy(engine):
        return engine.generate(requests, sampling_params=check_params(prompts, temperature=0.0, max_tokens=CHECK_MAX_TOKENS, logprobs=1), use_tqdm=False)
    plain = innermost(backend.llm)
    first, control, with_lora = greedy(plain), greedy(plain), greedy(backend.llm.engine)
    rows = [{"without": a.outputs[0].cumulative_logprob, "control": b.outputs[0].cumulative_logprob, "with": c.outputs[0].cumulative_logprob,
             "same_tokens_control": list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids), "same_tokens_with": list(a.outputs[0].token_ids) == list(c.outputs[0].token_ids)}
            for a, b, c in zip(first, control, with_lora, strict=True)]
    if any(v is None for r in rows for v in (r["without"], r["control"], r["with"])):
        return {"prompts": rows, "applied": False, "stop": "the engine gave no cumulative_logprob with logprobs=1: whether the adapter applies could not be measured"}
    control_difference = max(abs(r["without"] - r["control"]) for r in rows)
    adapter_difference = max(abs(r["without"] - r["with"]) for r in rows)
    res = {"prompts": rows, "control_difference": control_difference, "adapter_difference": adapter_difference, "applied": adapter_difference > control_difference}
    return res if res["applied"] else {**res, "stop": f"the adapter moves the log-probabilities by {adapter_difference}, no more than repeating without it does ({control_difference})"}


def seed_check(backend, prompts: list[list[int]]) -> dict:
    """On the real engine, in this form: what pass@k's pairing assumes of its seeds. The same seeds give the same tokens, at the cells' temperature, both passes with the prefix cache warm
    (a first pass is thrown away, since repeating alone can move log-probabilities). Other seeds give other tokens somewhere, at a high temperature: at 0.8 a model imitating templated text
    may give the same 32 tokens under any seed, which says nothing about the engine. Dev prompts only."""
    import evaluate
    requests = [{"prompt_token_ids": p} for p in prompts]

    def sampled(seeds, temperature):
        outs = backend.llm.engine.generate(requests, sampling_params=check_params(prompts, temperature=temperature, max_tokens=SEED_CHECK_MAX_TOKENS, seed=seeds), use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]
    seeds = [sample_seed("seed-check", i, 0) for i in range(len(prompts))]
    sampled(seeds, evaluate.SAMPLE_TEMPERATURE)
    first, again = sampled(seeds, evaluate.SAMPLE_TEMPERATURE), sampled(seeds, evaluate.SAMPLE_TEMPERATURE)
    spread = [sampled([sample_seed(f"seed-check-{draw}", i, 0) for i in range(len(prompts))], SEED_SPREAD_TEMPERATURE) for draw in range(SEED_SPREAD_DRAWS)]
    res = {"same_seed_same_tokens": first == again, "other_seed_other_tokens": any(s != spread[0] for s in spread[1:])}
    return res if all(res.values()) else {**res, "stop": f"sampling does not behave as pass@k assumes: {res}"}


def preflight(run_dir: pathlib.Path, form_name: str, greedy_cases: int = PREFLIGHT_GREEDY_CASES, gpu_memory_utilization: float = 0.85, adapter_only: bool = False) -> dict:
    """On dev, before any held-out case: that a runtime adapter really changes the outputs, and the form's throughput greedy and at k: the measurement the plan asks for at this build,
    descriptive, at the preflight's small batch. Its episodes are dev, and nothing is selected on them. Each sampling its own engine, as in a cell, so the rates are comparable.
    adapter_only (the follow-up: greedy only, and every dollar counted): the adapter check alone, on an adapter form."""
    import evaluate
    run = load_run(run_dir)
    verify_run(run, form_name)
    if run["backend"] != "vllm":
        raise SystemExit("the preflight measures a real engine")
    if adapter_only and not run["forms"][form_name].get("adapter"):
        raise SystemExit(f"{form_name} has no runtime adapter to check")
    form = run["forms"][form_name]
    tok = evaluate.tokenizer(local_only=False)
    dev = evaluate.load_cases("dev", limit=greedy_cases)
    from ap_sim.evaluation import rollout
    out = run_dir / "preflight"
    res: dict = {"form": form, "cases": len(dev), "seeds": SEED_SCHEME}
    backend = make_backend(form, 0, gpu_memory_utilization)
    template = rollout.grpo_template(tok)
    prompts = [rollout.Episode(c, f, tok, template).seq for c, f in list(dev.items())[:CHECK_PROMPTS]]
    if form.get("adapter"):
        res["adapter"] = adapter_applied(backend, prompts)
        if res["adapter"].get("stop"):
            res["stop"] = res["adapter"]["stop"]
    if adapter_only:
        res["adapter_only"] = True
    elif not res.get("stop"):
        res["seeds"] = seed_check(backend, prompts)
        if res["seeds"].get("stop"):
            res["stop"] = res["seeds"]["stop"]
    if not res.get("stop") and not adapter_only:
        greedy = evaluate.evaluate(dev, "vllm", tok, backend=backend, out=out / form_name / "greedy")["summary"]
        res["greedy_episodes_per_hour"] = greedy["timing"]["episodes_per_hour"]
        res["engine"] = engine_figures(backend.llm)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{form_name}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def preflight_sampled(run_dir: pathlib.Path, form_name: str, greedy_cases: int = PREFLIGHT_GREEDY_CASES, gpu_memory_utilization: float = 0.85) -> dict:
    """The same form's rate at k, on as many dev episodes as the greedy measurement had, in its own process (temperature is the engine's backend's)."""
    import evaluate
    run = load_run(run_dir)
    verify_run(run, form_name)
    if run["backend"] != "vllm":
        raise SystemExit("the preflight measures a real engine")
    form, k = run["forms"][form_name], run["decision"]["pass_at"]
    tok = evaluate.tokenizer(local_only=False)
    dev = evaluate.load_cases("dev", limit=max(1, greedy_cases // k))
    out = run_dir / "preflight"
    backend = make_backend(form, k, gpu_memory_utilization)
    summary = evaluate.evaluate(dev, "vllm", tok, samples=k, backend=backend, out=out / form_name / f"samples-{k}")["summary"]
    path = out / f"{form_name}.json"
    res = {**json.loads(path.read_text()), "sampled_at": k, "sampled_episodes_per_hour": summary["timing"]["episodes_per_hour"]}
    path.write_text(json.dumps(res, indent=1, default=str))
    return res


# The report and the Claude row ---------------------------------------------------------------------------------------------
def scenarios_of(records: list[dict], the_set: str, run: dict) -> dict:
    """The scenario of each case in these records, from that set's own loader (session2_tools.load_scenarios knows neither forgetting_l1 nor the reserve)."""
    cases = load_for_cell(run, the_set)
    return {c: cases[c]() for c in {r["case"] for r in records}}


def indicators_or_note(records: list[dict], the_set: str, run: dict) -> dict:
    """The two pre-declared indicators on one cell. Descriptive, and they read the frozen session2_tools.accepted_report, which a deep tool result can take past the JSON decoder's limit:
    that must not lose the report at its very end, after every paid cell. Said in the results, for that cell."""
    try:
        return session2_tools.indicators(records, scenarios_of(records, the_set, run))
    except RecursionError:
        return {"descriptive_only": True, "not_computed": "a message nested deeper than the JSON decoder allows"}


def noise_paths(run_dir: pathlib.Path, run: dict, given: list[str]) -> list[str]:
    """The noise floor's runs: as given, or session 1's beside this run's session. A report never goes out without them, because the plan's report states it. A rehearsal has none."""
    if given or run["rehearsal"]:
        return given
    session1 = run_dir.resolve().parents[1] / "session1"
    found = [session1 / "gate" / "dev-greedy", session1 / "noise-2", session1 / "noise-3"]
    missing = [str(p) for p in found if not (p / "episodes.jsonl").exists()]
    if missing:
        raise SystemExit(f"no noise floor: {missing} are not here, and no --noise was given. The report states it (PLAN Phase 6), so it does not go out without")
    return [str(p) for p in found]


def report(run_dir: pathlib.Path, claude: str | None, noise: list[str]) -> dict:
    run = load_run(run_dir)
    verify_code(run)
    noise = noise_paths(run_dir, run, noise)
    written = run_dir / "results.json"
    if not claude and written.exists() and json.loads(written.read_text()).get("claude_row"):
        raise SystemExit(f"{written} holds the Claude row: pass --claude, or it would be written over without it")
    first = None
    if claude:
        from ap_sim.generator.splits import first_test_cases
        import claude_row
        first = first_test_cases(claude_row.CASES)
    res = final_analysis.assemble(run_dir, run, pathlib.Path(claude) if claude else None, [pathlib.Path(p) for p in noise],
                                  indicators_of=lambda records, the_set: indicators_or_note(records, the_set, run), first_cases=first)
    analysis_now = file_sha256(ROOT / ANALYSIS_ONLY)
    res["code"] = {"commit_at_prepare": run["commit"], "commit_at_report": git_head(), "evaluation_code_as_at_prepare": True, "analysis": {
        "file": ANALYSIS_ONLY, "sha256_at_prepare": run["analysis_sha256"], "sha256_at_report": analysis_now, "changed_since_the_cells": analysis_now != run["analysis_sha256"]}}
    written.write_text(json.dumps(res, indent=1, default=str))
    return res


def claude_row_command(out: str) -> list[str]:
    """The Claude row's command (scripts/phase5/claude_row.py, pinned there, frozen): first 100 test cases, claude-sonnet-5-5, the settings of the strong-model check run before
    training, stopping at the API cap. Run by hand on the local machine that holds the API key. Never on a pod, and never from here."""
    return ["uv", "run", "--frozen", "--with", "anthropic==2.3.0", "python", "scripts/phase5/claude_row.py", "--phase10", "--out", out]


# The follow-up study (pre-registered 2026-11-06, with its stop rule specified 2026-11-08 and adopted 2026-11-09) -------------
# All greedy. Five models on the reserve and the stress set; the two new seeds also on test (200) and the three slices. Phase 10's cells for the other models stand.
NEW_SEEDS = (20261107, 20261108)
FOLLOWUP_GRPO = ("grpo-20261101", *(f"grpo-{s}" for s in NEW_SEEDS))
CUT3_SEED = 20261108                     # "then one seed": set on 2026-11-09, before any evaluation
FOLLOWUP_EPISODES = 1768                 # the follow-up plan: 1,000 on the new sets plus 768 for the new seeds on the old sets
FOLLOWUP_DECISION = {"test_cases": 200}  # the test set at Phase 10's size
FOLLOWUP_CAP, COPY_RESERVE = 155.0, 0.50
# Phase 10's measured wall seconds a cell, engine start included (each cell's cell.json in Phase 10's run 2): runtime LoRA on test, V6, policy_shift and forgetting;
# a 100-case set at the slowest 100-case cell (V6's); the stress set at 1.5 times that, since nothing has measured its episodes.
RECORDED_SECONDS = {"test": 2055.5, "v6": 1211.6, "policy_shift": 454.2, "forgetting_l1": 127.6, "test_reserve": 1211.6, "stress": 1817.4}
# What the evaluation loads, as Phase 10 and the seed runs recorded it: the merged SFT-v2 (run 2's run.json), Phase 10's GRPO and RFT adapters, each new seed's step 10
# (the follow-up's SHA256SUMS, copied off the training pod and verified 119 of 119).
PINNED = {
    "sft_merged": {"chat_template.jinja": "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326", "config.json": "12e9ce86058f0896330401b3bf3568054ea909a43e6eb19143e39c15b3b132c0",
                   "generation_config.json": "6283917e52ca41c3bf6d465df45ed68146ff0e3e7aa61513053d98e84b7c7688", "model.safetensors": "96d281c812256f141e39fa6e76b45afacffe7443a2c9c0cd8dada8126d112599",
                   "tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506", "tokenizer_config.json": "ab6bf168ef9bf5117d62a45011a97b78f0f2e25af6901225508fcc55f6fcdade"},
    "rft_adapter": {"adapter_config.json": "4b9450ecc3df8fbfc136eda6f63041850258a8f210ade6371d8855d293ae0437", "adapter_model.safetensors": "dfa45acea29f75cc33fda41844fec122b15317ecf98aae419788839faa71b382"},
    "grpo_20261101_adapter": {"adapter_config.json": "b16e415c8e25bbbc2b43d032b85803069778462eda072914c0027efe380f3c40", "adapter_model.safetensors": "c69ddbcc10ab940c5c8d2d3ece227d8a44a893031881857980b96e8dc44aa73c"},
    "grpo_20261107_adapter": {"adapter_config.json": "30d26218aa537e6a63f01a570dd305b3ae629167c02482d14c0eaa47b4bad86b", "adapter_model.safetensors": "b3d03395840033209c6c4edce5c787f8e90674d87d6e9b127ea6847d92ab072f"},
    "grpo_20261108_adapter": {"adapter_config.json": "624813ffd12c535d3a4909734f04e571ef94f83dafaeee1e84d2e0a82a1c0585", "adapter_model.safetensors": "c03034c713430ae2f315622ccbd94d3291eccce755a8d8696c05f9d16d363453"},
}


# The stress set as frozen on 2026-11-09 (manifests/stress/FREEZE.json). The loader compares the manifest with whatever FREEZE.json is
# beside it, so the driver pins both, and the stress code against the freeze's versions.
STRESS_FREEZE_SHA256 = "1da042d25c2684a084309c1a60a63ef933e50ce2541adca6a4a89619be5759e9"
STRESS_MANIFEST_SHA256 = "15962810d70324cf536bc7dfebe7b7af3651d1335d7897074d3eb358d6d5476c"


def check_stress_freeze() -> str:
    """Refuses unless FREEZE.json, the manifest and the stress code are the frozen ones. Returns the manifest's sha256, hashed here, not taken from the freeze record."""
    stress = ROOT / "manifests" / "stress"
    if file_sha256(stress / "FREEZE.json") != STRESS_FREEZE_SHA256:
        raise SystemExit("manifests/stress/FREEZE.json is not the freeze of 2026-11-09")
    manifest = file_sha256(stress / "messiness.json")
    if manifest != STRESS_MANIFEST_SHA256:
        raise SystemExit("manifests/stress/messiness.json is not the frozen manifest")
    versions = json.loads((stress / "FREEZE.json").read_text())["versions"]["stress"]
    changed = sorted(p for p, h in versions.items() if file_sha256(ROOT / p) != h)
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "src" / "ap_sim" / "stress").glob("*.py")}
    if changed or on_disk != set(versions):
        raise SystemExit(f"the stress code differs from its freeze: {changed or sorted(on_disk ^ set(versions))}")
    return manifest


def followup_cells() -> dict:
    """The pre-registered order: the cells the judging rules need first (the reserve and the stress set for SFT-v2 and the three seeds, test for the new seeds, 20261108 last), then RFT's
    two, then the new seeds' slices. So a cut, which takes the end of the list first, only ever falls on cells that have not run."""
    primary = [cell(m, s) for m in ("sft-v2", "grpo-20261101") for s in ("test_reserve", "stress")]
    primary += [cell(f"grpo-{seed}", s) for seed in NEW_SEEDS for s in ("test", "test_reserve", "stress")]
    rft = [cell("rft", s) for s in ("test_reserve", "stress")]
    slices = [cell(f"grpo-{seed}", s) for seed in NEW_SEEDS for s in SLICES]
    return {"followup": primary + rft + slices}


# The pre-registered cut order, each a rule over a cell, applied whole and in order.
FOLLOWUP_CUTS = (
    ("1: the new seeds' V6, policy_shift and forgetting cells", lambda c: c["model"] in {f"grpo-{s}" for s in NEW_SEEDS} and c["set"] in SLICES),
    ("2: RFT's two cells", lambda c: c["model"] == "rft"),
    (f"3: seed {CUT3_SEED}'s cells", lambda c: c["model"] == f"grpo-{CUT3_SEED}"),
)


def followup_forms(a) -> tuple[dict, dict]:
    """Each model's form and the sha256 of what it loads, refused unless it is the pinned input. GRPO and RFT as runtime LoRA on the merged SFT-v2, as in Phase 10."""
    merged = pathlib.Path(a.sft_merged)
    adapters = {"rft": pathlib.Path(a.rft_dir) / "adapter", "grpo-20261101": pathlib.Path(a.grpo_20261101)}
    adapters.update({f"grpo-{s}": pathlib.Path(a.seeds_dir) / f"seed-{s}" / "checkpoint-10" for s in NEW_SEEDS})
    forms, hashes = {"sft-v2": {"model": str(merged)}}, {"sft_merged": model_hashes(merged)}
    for name, path in adapters.items():
        forms[name] = {"model": str(merged), "adapter": str(path)}
        hashes[adapter_key(name)] = checked_adapter(path)
    for s in NEW_SEEDS:                         # the step-10 checkpoint is the adapter the run saved at its end, and the run is that seed's
        run_dir = pathlib.Path(a.seeds_dir) / f"seed-{s}"
        if adapter_hashes(run_dir / "adapter") != hashes[f"grpo_{s}_adapter"]:
            raise SystemExit(f"{run_dir}: checkpoint-10 is not the adapter the run saved")
        record = json.loads((run_dir / "seed.json").read_text())
        if record.get("seed") != s or record.get("engine_seed_read") != record.get("engine_seed_expected"):
            raise SystemExit(f"{run_dir}/seed.json is not seed {s}'s run, or its seed did not read back")
    wrong = sorted(k for k, v in PINNED.items() if hashes.get(k) != v)
    if wrong:
        raise SystemExit(f"not the pinned inputs: {wrong}")
    return forms, hashes


def prepare_followup(a) -> dict:
    run_dir = pathlib.Path(a.out)
    if a.backend != "vllm" and not a.rehearsal:
        raise SystemExit("a scripted backend runs only in a rehearsal")
    if a.backend == "vllm":
        if not (a.sft_merged and a.rft_dir and a.grpo_20261101 and a.seeds_dir):
            raise SystemExit("--sft-merged, --rft-dir, --grpo-20261101 and --seeds-dir are needed for a model run")
        forms, hashes = followup_forms(a)
    else:
        forms, hashes = {m: {"model": a.backend} for m in ("sft-v2", "rft", *FOLLOWUP_GRPO)}, {}
    cells = followup_cells()
    sets = {}
    for the_set in sorted({c["set"] for c in cells["followup"]}):
        ids, _ = set_ids(the_set, FOLLOWUP_DECISION, a.rehearsal)
        sets[the_set] = {"ids": ids, "splits": ["dev"] if a.rehearsal else [the_set]}
    planned = sum(len(sets[c["set"]]["ids"]) for c in cells["followup"])
    if not a.rehearsal and planned != FOLLOWUP_EPISODES:
        raise SystemExit(f"{planned} episodes planned, {FOLLOWUP_EPISODES} pre-registered")
    stress = check_stress_freeze()
    run = {"freeze": "followup/1", "run": run_dir.name, "rehearsal": a.rehearsal, "backend": a.backend, "decision": {**FOLLOWUP_DECISION, "approved_cap": FOLLOWUP_CAP},
           "forms": forms, "input_sha256": hashes, "models": sorted({c["model"] for c in cells["followup"]}), "sets": sets, "cells": cells, "episodes_planned": planned,
           "stress_manifest_sha256": stress, "cut_order": [label for label, _ in FOLLOWUP_CUTS], "recorded_seconds": RECORDED_SECONDS, "seed_scheme": SEED_SCHEME,
           "code_sha256": code_hashes(), "analysis_sha256": file_sha256(ROOT / ANALYSIS_ONLY), "commit": git_head()}
    return write_run(run_dir, run)


def read_cuts(run_dir: pathlib.Path) -> list[dict]:
    path = run_dir / "cuts.json"
    return json.loads(path.read_text()) if path.exists() else []


def followup_budget(run_dir: pathlib.Path, spent: float, price: float, cap: float = FOLLOWUP_CAP) -> dict:
    """The stop rule, before every cell: spent now, plus the cells left at Phase 10's recorded times (stress at 1.5 times), scaled up by this run's own cells when they ran
    slower (never down), plus $0.50 to copy the results off, must be at most the cap. If not, the pre-registered cuts are applied, whole and in order, to cells not yet run, until
    it is. A cut is written to cuts.json and stays. It reads the cells' wall time only, never an outcome. Fits is false only when every cut is applied and it still does not."""
    if cap > FOLLOWUP_CAP:
        raise SystemExit(f"--cap {cap} is above the follow-up's ${FOLLOWUP_CAP}")
    run = load_run(run_dir)
    every = run["cells"]["followup"]
    cuts = read_cuts(run_dir)
    cut_names = {n for c in cuts for n in c["cells"]}
    done = [c for c in every if (run_dir / c["name"] / ".done").exists()]
    left = [c for c in every if c not in done and c["name"] not in cut_names]
    wall = {c["name"]: json.loads((run_dir / c["name"] / "cell.json").read_text())["measured"]["wall_seconds_with_engine_start"] for c in done}
    recorded_done = sum(RECORDED_SECONDS[c["set"]] for c in done)
    pooled = max(1.0, sum(wall.values()) / recorded_done) if recorded_done else 1.0
    # A set's own cells can only raise its factor above the pooled one, never lower it: a stress cell inside its 1.5 allowance must not hide slow reserve or test cells
    # (the stricter of the two readings of "raised by this run's own cells if they run slower").
    per_set = {}
    for s in {c["set"] for c in done}:
        mine = [c for c in done if c["set"] == s]
        per_set[s] = max(pooled, sum(wall[c["name"]] for c in mine) / sum(RECORDED_SECONDS[s] for c in mine))

    unmeasured = max([pooled, *per_set.values()])      # a set with no cell yet takes the slowest factor seen so far

    def forecast(cells: list[dict]) -> float:
        return round(spent + sum(RECORDED_SECONDS[c["set"]] * per_set.get(c["set"], unmeasured) for c in cells) / 3600 * price + COPY_RESERVE, 2)
    applied = []
    for label, rule in FOLLOWUP_CUTS:
        if forecast(left) <= cap:
            break
        if label in {c["cut"] for c in cuts}:
            continue
        names = [c["name"] for c in left if rule(c)]
        if names:
            applied.append({"cut": label, "cells": names, "spent": spent, "forecast_before": forecast(left), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            left = [c for c in left if c["name"] not in names]
    if applied:
        temporary = run_dir / "cuts.json.tmp"         # whole or not at all
        temporary.write_text(json.dumps(cuts + applied, indent=1))
        temporary.replace(run_dir / "cuts.json")
    total = forecast(left)
    return {"spent": spent, "cells_left": [c["name"] for c in left], "slower_than_recorded": round(pooled, 3),
            "slower_by_set": {s: round(f, 3) for s, f in sorted(per_set.items())}, "slower_unmeasured": round(unmeasured, 3), "forecast_total": total, "cap": cap,
            "cuts_applied_now": [c["cut"] for c in applied], "cut_cells": sorted(cut_names | {n for c in applied for n in c["cells"]}), "fits": total <= cap}


def followup_cell_names(run_dir: pathlib.Path) -> list[str]:
    """The cells to run, in order: the plan's, less any cut."""
    cut = {n for c in read_cuts(run_dir) for n in c["cells"]}
    return [c["name"] for c in load_run(run_dir)["cells"]["followup"] if c["name"] not in cut]


def main() -> None:
    import pipeline_guard
    pipeline_guard.require()          # refuses to start on a settings mismatch (config.require_pipeline)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--phase10", action="store_true", help="the held-out sets")
    p.add_argument("--rehearsal", action="store_true", help="every set is dev")
    p.add_argument("--backend", choices=("vllm", "reference", "reckless"), default="vllm")
    p.add_argument("--decision", required=True, help="Phase 8's decision.json")
    p.add_argument("--selected", default=None, help="selected.json, from the pre-declared selection")
    p.add_argument("--baselines", default=None, help="session 1's alarm baselines: the selection is derived again from the rows")
    p.add_argument("--grpo-checkpoint", default=None, help="the selected GRPO checkpoint")
    p.add_argument("--sft-merged", default=None)
    p.add_argument("--rft-dir", default=None)
    p.add_argument("--sft-adapter", default=None, help="SFT-v2's unmerged adapter, if saved: the descriptive cross-evaluation")
    p.add_argument("--out", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--dir", required=True)
    c.add_argument("--name", required=True)
    f = sub.add_parser("preflight")
    f.add_argument("--dir", required=True)
    f.add_argument("--form", required=True)
    f.add_argument("--sampled", action="store_true", help="the rate at k, in its own process, after the greedy part")
    f.add_argument("--adapter-only", action="store_true", help="the follow-up: the runtime adapter check alone")
    fp = sub.add_parser("prepare-followup")
    fp.add_argument("--rehearsal", action="store_true", help="every set is dev")
    fp.add_argument("--backend", choices=("vllm", "reference", "reckless"), default="vllm")
    fp.add_argument("--sft-merged", default=None)
    fp.add_argument("--rft-dir", default=None)
    fp.add_argument("--grpo-20261101", default=None, help="Phase 10's GRPO checkpoint (grpo/main/checkpoint-10)")
    fp.add_argument("--seeds-dir", default=None, help="the seed runs' directory (followup/), with seed-20261107 and seed-20261108")
    fp.add_argument("--out", required=True)
    fb = sub.add_parser("followup-budget")
    fb.add_argument("--dir", required=True)
    fb.add_argument("--spent", type=float, required=True, help="spending so far: before the pod, storage to the planned end included, plus this pod's clock")
    fb.add_argument("--price", type=float, required=True)
    fb.add_argument("--cap", type=float, default=FOLLOWUP_CAP)
    fc = sub.add_parser("followup-cells")
    fc.add_argument("--dir", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--dir", required=True)
    b.add_argument("--part", default="final", choices=("final", "cross"))
    b.add_argument("--spent", type=float, required=True)
    b.add_argument("--price", type=float, required=True)
    b.add_argument("--volume", type=float, required=True, help="what the volumes add to the session's spending: stated, never assumed")
    b.add_argument("--cap", type=float, default=None, help="in place of the decision's approved cap, when the user has approved another; at most the ceiling")
    b.add_argument("--override-ceiling", action="store_true", help="a --cap above the ceiling, on the user's ruling")
    v = sub.add_parser("verify-code")
    v.add_argument("--dir", required=True)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--claude", default=None)
    r.add_argument("--noise", nargs="*", default=[], help="session 1's SFT-v2 greedy runs on dev")
    k = sub.add_parser("claude-row")
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    code = 0
    if a.cmd == "prepare":
        res = prepare(a)
        res = {"run": res["run"], "episodes_planned": res["episodes_planned"], "cells": [c["name"] for c in res["cells"]["final"]], "opt_in": [c["name"] for c in res["cells"]["cross"]]}
    elif a.cmd == "cell":
        res = run_cell(pathlib.Path(a.dir), a.name)
    elif a.cmd == "preflight":
        if a.adapter_only and a.sampled:
            raise SystemExit("--adapter-only measures no rate: not with --sampled")
        res = preflight_sampled(pathlib.Path(a.dir), a.form) if a.sampled else preflight(pathlib.Path(a.dir), a.form, adapter_only=a.adapter_only)
        code = 2 if res.get("stop") else 0
    elif a.cmd == "prepare-followup":
        res = prepare_followup(a)
        res = {"run": res["run"], "episodes_planned": res["episodes_planned"], "cells": [c["name"] for c in res["cells"]["followup"]]}
    elif a.cmd == "followup-budget":
        res = followup_budget(pathlib.Path(a.dir), a.spent, a.price, a.cap)
        code = 0 if res["fits"] else 2
    elif a.cmd == "followup-cells":
        print(" ".join(followup_cell_names(pathlib.Path(a.dir))))
        sys.exit(0)
    elif a.cmd == "budget":
        res = budget(pathlib.Path(a.dir), a.part, a.spent, a.price, a.volume, a.cap, a.override_ceiling)
        code = 0 if res["within_cap"] else 2
    elif a.cmd == "verify-code":
        verify_code(load_run(pathlib.Path(a.dir)))
        res = {"verified": a.dir}
    elif a.cmd == "report":
        res = report(pathlib.Path(a.dir), a.claude, a.noise)
    else:
        res = claude_row_command(a.out)
    print(json.dumps(res, indent=1, default=str)[:4000])
    sys.exit(code)


if __name__ == "__main__":
    main()
