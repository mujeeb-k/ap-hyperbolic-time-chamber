"""Budget re-forecast for the remaining v2 phases (read-only arithmetic, nothing provisioned or called).

Two caps, tracked separately (set on 2026-10-30, after the strong-model check on the final v2 interface): GPU spending
against $40, and Claude API spending against $30 in total, including what is already spent.

GPU: sizes are the v2 plan's (2026-10-30), except where marked ASSUMED. Two GRPO step sizes are forecast: 16 cases x 16
rollouts (the plan as sized) and 8 x 8. The step size is fixed after Phase 8 (the GRPO plumbing checks) measures the
update time.

API: each planned use is its episodes times the per-episode cost measured in the strong-model check on the final v2
interface (scripts/strong_model_sweep.py --gate1a, 30 episodes, 2026-10-30).

    python scripts/budget_forecast.py
    python scripts/budget_forecast.py decide --measured measured.json --out decision.json --gpu A100 --price-per-hour 1.79 \\
        --volume-gb 60 --volume-per-gb-month 0.07
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

PRICE = 1.79               # $/h, A100 SXM secure (live quote 2026-10-30)
CAP = 40.00                # the GPU cap, $. GPU only
API_CAP = 30.00            # the Claude API cap, $ (2026-10-30): total v2 API spending, including what is spent

# Actual v2 spending, $, by source. GPU is RunPod. API is Anthropic (Claude), summed from each run's saved per-episode
# cost_usd. The labels name the development stage that spent it. "Phase 0.3" is the strong-model check and sweep on the
# version 1 interface (scripts/strong_model_check.py, scripts/strong_model_sweep.py). "Phase 0.6" is the Qwen base-model
# check (scripts/qwen_check.py). "Gate 1A" is the strong-model check on the final v2 interface.
# These tables stop at 2026-11-04 and are kept as the record the forecasts were made from. They are not the running
# total. Later spending (Phase 10, the final evaluation, then the follow-up's seed training and evaluation, and the CPU
# pods) is not recorded here. The paper's cost table (Supplement Table S16) gives the totals.
SPENT_GPU = {"Phase 0.6 part 2 pod, 54 min (2026-10-30)": 1.61,
             "Session 1 setup and part measure, A100 pod, 57 min 25 s at $1.79/h (2026-11-02)": 1.71,
             "Session 1 follow-up measurement to 21:00Z, A100 pod, 2 h 5 min at $1.79/h (2026-11-02)": 3.73,
             "Session 1 parts train and phase7, 21:00Z to 00:34Z, A100 pod, 3 h 34 min at $1.79/h (2026-11-03)": 6.39,
             "Session 2 preparation and one real step, H100 NVL pod, 2 h 45 min at $3.19/h (2026-11-03)": 8.80,
             "Session 2 retry, RFT and one real step, A100 pod, 1 h 34 min at $1.79/h (2026-11-04)": 2.81}
SPENT_API = {"Phase 0.3 strong-model check, 13 episodes (2026-10-29)": 0.52,
             "Phase 0.3 strong-model sweep, 90 episodes (2026-10-29)": 9.47,
             "Gate 1A strong-model check, 30 episodes (2026-10-30)": 3.89}
GATE1A = "Gate 1A strong-model check, 30 episodes (2026-10-30)"
SPENT = round(sum(SPENT_GPU.values()), 2)          # against the GPU cap
# Machines for checks and tests, not the experiment's GPU work (2026-11-03). Listed apart and not in SPENT.
SPENT_TOOLING = {"Round 14 and the full suite, CPU pod, 3 h 52 min at $0.48/h (2026-11-03)": 1.86,
                 "The clean suite, CPU pod, 3 h 6 min at $0.48/h (2026-11-03)": 1.49,
                 "The clean suite at de0c384, CPU pod, 1 h 10 min at $0.48/h (2026-11-04)": 0.56,
                 "The freeze-point suite at c6bc779, CPU pod, 1 h 38 min at $0.48/h (2026-11-04)": 0.78}
SPENT_API_TOTAL = round(sum(SPENT_API.values()), 2)  # against the API cap

# The per-episode API cost in the strong-model check on the final v2 interface: claude-sonnet-5-5 on surfaces/4 with
# instructions/5, the settings the Phase 10 reference row reuses. $3.8898 over 30 episodes (largest $0.1825), summed
# from the saved per-episode cost_usd.
GATE1A_EPISODES, GATE1A_COST = 30, 3.8898
API_PER_EPISODE = GATE1A_COST / GATE1A_EPISODES
API_PER_EPISODE_LARGEST = 0.1825
# Planned API use (episodes). The Phase 10 reference row runs Claude once on each of the FIRST 100 test cases in frozen
# order. This was fixed on 2026-10-30, when the splits were frozen. It is a 100-case partial evaluation, compared with
# SFT-v2 and GRPO on the same 100 cases. No other API use is planned.
CLAUDE_ROW_CASES = 100
API_PLAN = {"Phase 10 Claude reference row, first 100 test cases": CLAUDE_ROW_CASES}

PLAN = {
    # Measured / recorded
    "episodes_per_hour": 1634,          # vLLM 0.30.0, 60 concurrent greedy episodes (the Qwen check, 2026-10-30)
    "step_gen_floor_min": 2.5,          # a step's generation lasts at least as long as its longest episode
    "sft_min_per_100_demo_epochs": 7.9, # version 1's SFT run (Stage A): 148 demos x 2 epochs in 23.3 min
    "browser_episodes_per_hour": 50,    # v1 browser throughput
    "session_overhead_min": 12,         # the Qwen check: start, environments + model, export/terminate, margin
    # The plan's sizes (2026-10-30)
    "sft_demos": 300, "sft_epochs": 2,
    "dev_cases": 60, "regression_cases": 40,     # from 2026-10-31 the 40 are the L1 forgetting set (same size)
    "rl_cases_profiled": 250, "profiling_samples": 8,
    "pilot_lrs": 2, "pilot_steps": 20, "main_steps": 50,
    "cases_per_step": 16, "rollouts_per_case": 16,
    "alarm_every": 10, "alarm_samples": 8,          # dev alarm: 60 dev x 8 at T 0.8, every 10 GRPO steps
    "score_every": 10,                              # checkpoint scoring: greedy on dev + regression, every 10 steps
    "test_cases": 100,                              # pre-registered floor
    "final_models_greedy": 3,                       # base, SFT-v2, GRPO (RFT excluded; about $2.70 if built)
    "final_models_sampled": 2, "final_samples": 8,  # SFT-v2 and GRPO at T 0.8 (not the base)
    # Decided when the splits were built (2026-10-30): V6 manifest 100, policy shift regenerated at 44. The "L3 30"
    # slice is dropped because no definition of it exists in v1's or v2's records.
    "slice_cases": {"V6": 100, "policy_shift": 44}, "slice_models": 3,
    "browser_cases": 20,
    # Not in the plan as sized (0 here); set by the 2026-10-31 decisions below.
    "forgetting_reference_models": 0,   # SFT-v2 greedy on the forgetting set, the regression rule's reference
    "phase10_regression_models": 0,     # the Phase 10 regression evaluation on the forgetting set, greedy per model
    # ASSUMED
    "gate_episodes": 60,                 # SFT-v2 gate: greedy once on dev
    "seconds_per_sequence_update": 6.0,  # LoRA fwd+bwd with gradient checkpointing, ~14k-token sequences (estimate)
    "step_other_min": 0.5,               # weight sync, grading, logging per step (estimate)
    "sessions": 7,                       # SFT, gate, profiling, pilot, main, final + slices, spare
}

SCENARIOS = {
    "optimistic": {"seconds_per_sequence_update": 3.0, "step_gen_floor_min": 2.0},
    "central": {},
    "pessimistic": {"seconds_per_sequence_update": 15.0, "step_gen_floor_min": 4.0, "episodes_per_hour": 1000},
}


def step_minutes(p: dict) -> float:
    n = p["cases_per_step"] * p["rollouts_per_case"]
    gen = p.get("step_generation_measured_min") or max(n / p["episodes_per_hour"] * 60, p["step_gen_floor_min"])
    return gen + n * p["seconds_per_sequence_update"] / 60 + p["step_other_min"]


def per_step_evals_hours(p: dict) -> float:
    """Dev alarm and checkpoint scoring, amortized per GRPO step."""
    eph = p["episodes_per_hour"]
    alarm = p["dev_cases"] * p["alarm_samples"] / eph / p["alarm_every"]
    score = (p["dev_cases"] + p["regression_cases"]) / eph / p["score_every"]
    return alarm + score


def forecast(p: dict, spent: float = SPENT) -> dict:
    eph = p["episodes_per_hour"]
    grpo_steps = p["pilot_lrs"] * p["pilot_steps"] + p["main_steps"]
    alarms = p["pilot_lrs"] * (p["pilot_steps"] // p["alarm_every"]) + p["main_steps"] // p["alarm_every"]
    scorings = p["pilot_lrs"] * (p["pilot_steps"] // p["score_every"]) + p["main_steps"] // p["score_every"]
    sm = step_minutes(p)
    final_eps = p["test_cases"] * (p["final_models_greedy"] + p["final_models_sampled"] * p["final_samples"])
    hours = {
        "SFT": p["sft_demos"] * p["sft_epochs"] / 100 * p["sft_min_per_100_demo_epochs"] / 60,
        "gate evaluation": p["gate_episodes"] / eph,
        "profiling": p["rl_cases_profiled"] * p["profiling_samples"] / eph,
        "learning-rate pilot": p["pilot_lrs"] * p["pilot_steps"] * sm / 60,
        "main GRPO": p["main_steps"] * sm / 60,
        "dev alarm evaluations": alarms * p["dev_cases"] * p["alarm_samples"] / eph,
        "checkpoint scoring": scorings * (p["dev_cases"] + p["regression_cases"]) / eph,
        "final evaluation (test)": final_eps / eph,
        "generalization slices (vLLM)": sum(p["slice_cases"].values()) * p["slice_models"] / eph,
        "browser slice": p["browser_cases"] * p["slice_models"] / p["browser_episodes_per_hour"],
        "forgetting set: SFT-v2 reference": p["forgetting_reference_models"] * p["regression_cases"] / eph,
        "regression evaluation (Phase 10)": p["phase10_regression_models"] * p["regression_cases"] / eph,
        "session overhead": p["sessions"] * p["session_overhead_min"] / 60,
    }
    total = sum(hours.values())
    grpo_related = hours["learning-rate pilot"] + hours["main GRPO"] + hours["dev alarm evaluations"] + hours["checkpoint scoring"]
    fixed_dollars = (total - grpo_related) * PRICE
    per_step_dollars = (sm / 60 + per_step_evals_hours(p)) * PRICE
    return {"episodes_per_step": p["cases_per_step"] * p["rollouts_per_case"], "minutes_per_step": round(sm, 2),
            "grpo_steps": grpo_steps, "alarm_evaluations": alarms, "checkpoint_scorings": scorings,
            "hours": hours, "dollars": {k: round(v * PRICE, 2) for k, v in hours.items()},
            "total_hours": round(total, 2), "total_dollars": round(total * PRICE, 2),
            "spent": round(spent, 2), "remaining": round(CAP - spent, 2), "headroom": round(CAP - spent - total * PRICE, 2),
            "steps_that_fit": int((CAP - spent - fixed_dollars) / per_step_dollars)}


def api_forecast(test_cases: int, per_episode: float = API_PER_EPISODE) -> dict:
    episodes = {use: min(n, test_cases) for use, n in API_PLAN.items()}
    planned = {use: round(e * per_episode, 2) for use, e in episodes.items()}
    total = sum(e * per_episode for e in episodes.values())
    return {"test_cases": test_cases, "episodes": episodes, "planned": planned, "spent": SPENT_API_TOTAL,
            "forecast_total": round(SPENT_API_TOTAL + total, 2), "remaining_after": round(API_CAP - SPENT_API_TOTAL - total, 2),
            "exceeds_cap": SPENT_API_TOTAL + total > API_CAP}


# Re-forecast from the real lengths of the Phase 4 SFT demonstrations (2026-11-01). The earlier forecasts above scale
# nothing. They use the Qwen check's base-model throughput as measured. A trained agent's episodes look like the 300
# frozen demonstrations (sft-demos/v2-1), which are longer, so every generation figure is scaled from the Qwen check by
# length.
PHASE06 = {"episodes": 60, "wall_seconds": 132, "rounds": 17,       # vLLM 0.30.0, 60 concurrent greedy episodes
           "completion_tokens_mean": 6037, "generated_tokens_mean": 682, "turns_mean": 8.8}   # its 60 episode files
PHASE4 = {"completion_tokens_mean": 10694, "completion_tokens_p90": 14711, "completion_tokens_max": 16137,
          "generated_tokens_mean": 1828, "turns_mean": 15.8, "turns_max": 24,
          "context_tokens_mean": 16560, "context_tokens_max": 22007}   # runs/phase4/sft-demos-v2-1 (sidecar)
V1_SFT_DEMO_TOKENS_MEAN = 11650   # sft-demos/5, the 148 demonstrations behind Stage A's 7.9 min per 100 demo-epochs
UPDATE_BASIS_TOKENS = 14000       # the sequence length the 6 s (3 / 15 s) per-sequence update estimate assumed
KL_FORWARD_SHARE = 1 / 3          # the reference forward pass (beta 0.02) as a share of a forward-backward update
GATE_EPISODES = 60 + 60 * 8       # Phase 6 gate on dev: greedy plus 8 samples per case (2026-11-01)
RFT_EPISODES = 2 * 250            # RFT upper bound: at most 2 profiling successes per case, 250 cases, one epoch
RFT_TEST_MODELS = 1               # RFT's greedy test evaluation in Phase 10


def phase4_reforecast(rev4: dict) -> dict:
    """GPU forecast with the Phase 4 lengths, the KL reference pass and the 8-sample gate, for 200 and 300 test cases.

    Scaling (each stated; the first GPU session re-measures all of it):
      - generation throughput: central by total tokens per episode (tool results are prefilled, turns decoded:
        1,634 x 6,037 / 10,694); pessimistic by decoded tokens (1,634 x 682 / 1,828); optimistic as central;
      - a step's generation floor (its longest episode): the longest demonstration's 24 turns at the Qwen check's 7.76 s
        per round (132 s / 17 rounds); pessimistic x 1.49 (decoded tokens per turn, 115.7 against 77.5); optimistic x 0.8;
      - the per-sequence update: the 3 / 6 / 15 s estimates scaled to the demonstrations' mean context (16,560 against
        14,000), plus the KL reference forward pass (one third more);
      - SFT: Stage A's 7.9 min per 100 demo-epochs scaled by tokens per demonstration (16,560 against 11,650);
      - the gate: 540 dev episodes; RFT included at its upper bound (500 episodes, one epoch, plus one greedy test
        evaluation), since the amended cut order cuts it after the test extension and extra seeds."""
    gen_central = PLAN["episodes_per_hour"] * PHASE06["completion_tokens_mean"] / PHASE4["completion_tokens_mean"]
    gen_pess = PLAN["episodes_per_hour"] * PHASE06["generated_tokens_mean"] / PHASE4["generated_tokens_mean"]
    per_round = PHASE06["wall_seconds"] / PHASE06["rounds"]
    floor = PHASE4["turns_max"] * per_round / 60
    per_turn = (PHASE4["generated_tokens_mean"] / PHASE4["turns_mean"]) / (PHASE06["generated_tokens_mean"] / PHASE06["turns_mean"])
    length = PHASE4["context_tokens_mean"] / UPDATE_BASIS_TOKENS
    update = lambda s: s * length * (1 + KL_FORWARD_SHARE)   # noqa: E731
    sft_rate = PLAN["sft_min_per_100_demo_epochs"] * PHASE4["context_tokens_mean"] / V1_SFT_DEMO_TOKENS_MEAN
    scen = {"optimistic": {"episodes_per_hour": gen_central, "step_gen_floor_min": floor * 0.8,
                           "seconds_per_sequence_update": update(3.0)},
            "central": {"episodes_per_hour": gen_central, "step_gen_floor_min": floor,
                        "seconds_per_sequence_update": update(6.0)},
            "pessimistic": {"episodes_per_hour": gen_pess, "step_gen_floor_min": floor * per_turn,
                            "seconds_per_sequence_update": update(15.0)}}
    base = {**PLAN, **rev4, "gate_episodes": GATE_EPISODES, "sft_min_per_100_demo_epochs": sft_rate}
    out = {"scaling": {"episodes_per_hour": {"central": round(gen_central), "pessimistic": round(gen_pess)},
                       "step_generation_floor_min": {k: round(v["step_gen_floor_min"], 2) for k, v in scen.items()},
                       "seconds_per_sequence_update_with_kl": {k: round(v["seconds_per_sequence_update"], 1)
                                                               for k, v in scen.items()},
                       "sft_min_per_100_demo_epochs": round(sft_rate, 1)}}
    for n in (200, 300):
        rows = {}
        for name, s in scen.items():
            p = {**base, **s, "test_cases": n}
            f = forecast(p)
            eph = p["episodes_per_hour"]
            rft = (RFT_EPISODES / 100 * sft_rate + RFT_TEST_MODELS * n / eph * 60) / 60 * PRICE
            per_step = (f["minutes_per_step"] / 60 + per_step_evals_hours(p)) * PRICE
            total = f["total_dollars"] + rft
            rows[name] = {"total_dollars": round(total, 2), "rft_dollars": round(rft, 2), "spent": f["spent"],
                          "headroom": round(CAP - f["spent"] - total, 2),
                          "headroom_without_rft": round(CAP - f["spent"] - f["total_dollars"], 2),
                          "minutes_per_step": f["minutes_per_step"],
                          "main_steps_that_fit": int(f["steps_that_fit"] - rft / per_step) - rev4["pilot_lrs"] * rev4["pilot_steps"],
                          "dollars": {**f["dollars"], "RFT (upper bound)": round(rft, 2)}}
        out[f"test {n}"] = rows
    c300 = out["test 300"]["central"]
    out["phase66_trigger_on_these_figures"] = {
        "full_main_run_fits_central": c300["main_steps_that_fit"] >= rev4["main_steps"],
        "central_headroom_300": c300["headroom"],
        "fires": c300["main_steps_that_fit"] >= rev4["main_steps"] and c300["headroom"] >= 3.0}
    return out


# The decision rule fixed before GPU session 1 (2026-11-01). The measurement block measures the per-sequence
# update time (with the KL reference pass) and generation throughput; `decide` re-runs the forecast with them.
MIN_MAIN_STEPS = 30
TOTAL_TARGET = 90.00          # the rest of session 1 (2026-11-02): the configuration's whole-experiment forecast
# RAISED_CAP_LIMIT: the GPU ceiling for the whole experiment, for every outcome. Set on 2026-11-04, before any data. It
# was $120 earlier that day, $100 from 2026-11-02 and $50 before that.
RAISED_CAP_LIMIT = 135.00
SAVINGS = (("a: profile 120 RL cases instead of 250", {"rl_cases_profiled": 120}),
           ("b: Phase 6 gate greedy plus 4 samples instead of 8", {"gate_episodes": 60 + 60 * 4}),
           ("c: pass@4 instead of pass@8 in Phase 10", {"final_samples": 4}))

# Prices and storage in the rule (2026-11-02): session 1 at the chosen GPU's actual price, sessions 2 and 3 in two
# scenarios, all on H100 at its price (the rule is applied to this one, the conservative one) and on A100 at $1.79/h.
# This session's network volume counts against the GPU budget. Speed comes from the benchmark, on the session's GPU.
H100_SXM_PRICE = 3.99           # $/h, H100 SXM secure (live quote 2026-11-02)
H100_PCIE_PRICE, H100_NVL_PRICE = 2.89, 3.19   # the same quote, for the options only
STANDARD_VOLUME_PER_GB_MONTH = 0.07   # RunPod's storage docs, first 1 TB
HIGH_PERFORMANCE_VOLUME_PER_GB_MONTH = 0.14   # third-party guides only (2026-11-02); check the console
MEASUREMENT_BLOCK_HOURS = 0.5   # part `measure` (preregistration_proof/PLAN_V2.md, Phase 6: about 30 minutes)
ASSUMED_H100_OVER_A100 = 2.0    # ASSUMED, vendor benchmarks only (2026-11-02): used only to carry a speed measured
                                # on H100 over to A100; a speed measured on A100 is not scaled up for H100, conservatively
SESSION_1_ITEMS = ("SFT", "gate evaluation", "profiling", "forgetting set: SFT-v2 reference")
GPU_DEFAULT = {"session_1_price": PRICE, "later_price": PRICE, "later_time_factor": 1.0, "volume_dollars": 0.0,
               "measurement_hours": 0.0}


def volume_dollars(gb: float, per_gb_month: float, months: float) -> float:
    return gb * per_gb_month * months


def gpu_scenario(later: str, measured_on: str, session_price: float, volume: float,
                 measurement_hours: float = MEASUREMENT_BLOCK_HOURS) -> dict:
    """The forecast's prices when session 1 runs on `measured_on` (A100 or H100) at `session_price`, and sessions 2 and
    3 run on `later` (A100 or H100). `measurement_hours`: 0 once the block's cost is in SPENT_GPU."""
    h100 = session_price if measured_on == "H100" else H100_SXM_PRICE
    factor = ASSUMED_H100_OVER_A100 if (later, measured_on) == ("A100", "H100") else 1.0
    return {"session_1_price": session_price, "later_price": PRICE if later == "A100" else h100,
            "later_time_factor": factor, "volume_dollars": volume, "measurement_hours": measurement_hours}


def measured_plan(measured: dict, rev4: dict) -> dict:
    """The forecast's inputs with the measured figures: episodes per hour (from the benchmark's batch nearest a GRPO
    step and its evaluation batch), a step's generation floor (the longest demonstration-length episode in a 64-episode
    batch), the per-sequence update with the KL reference pass, and the SFT rate (Phase 4 lengths; the benchmark does not
    measure SFT, so unless `sft_speed_ratio` is given a faster GPU's SFT is still forecast at A100's rate)."""
    sft_rate = (PLAN["sft_min_per_100_demo_epochs"] * PHASE4["context_tokens_mean"] / V1_SFT_DEMO_TOKENS_MEAN
                / measured.get("sft_speed_ratio", 1.0))
    return {**PLAN, **rev4, "gate_episodes": GATE_EPISODES, "sft_min_per_100_demo_epochs": sft_rate,
            "episodes_per_hour": measured["episodes_per_hour"],
            "step_gen_floor_min": measured["step_generation_minutes"],
            "seconds_per_sequence_update": measured["seconds_per_sequence_update_with_kl"],
            "step_generation_measured_min": measured.get("step_generation_measured_min"),
            "step_other_min": measured.get("step_other_min", PLAN["step_other_min"])}


def _fit(p: dict, cap: float, rev4: dict, gpu: dict = GPU_DEFAULT) -> dict:
    f = forecast(p)
    h = f["hours"]
    sft_rate = p["sft_min_per_100_demo_epochs"]
    later_rate = gpu["later_price"] * gpu["later_time_factor"]
    overhead = p["session_overhead_min"] / 60      # setup does not speed up with the GPU
    rft_sft = RFT_EPISODES / 100 * sft_rate / 60   # in session 1
    rft_test = RFT_TEST_MODELS * p["test_cases"] / p["episodes_per_hour"]   # in session 3
    later_items = sum(v for k, v in h.items() if k not in SESSION_1_ITEMS and k != "session overhead")
    session_1 = (sum(h[k] for k in SESSION_1_ITEMS) + rft_sft + gpu["measurement_hours"] + overhead) * gpu["session_1_price"]
    later = (later_items + rft_test) * later_rate + (p["sessions"] - 1) * overhead * gpu["later_price"]
    total = session_1 + later + gpu["volume_dollars"]
    per_step = (f["minutes_per_step"] / 60 + per_step_evals_hours(p)) * later_rate
    fixed = total - p["main_steps"] * per_step      # everything but the main steps
    main_fit = int((cap - f["spent"] - fixed) / per_step)
    return {"main_steps_that_fit": main_fit, "minutes_per_step": f["minutes_per_step"],
            "total_dollars_at_40_main_steps": round(total, 2), "per_main_step_dollars": round(per_step, 3),
            "fixed_dollars": round(fixed, 2), "session_1_dollars": round(session_1, 2),
            "session_1_rft_dollars": round(rft_sft * gpu["session_1_price"], 2),
            "volume_dollars": round(gpu["volume_dollars"], 2), "cap": cap}


def decide(measured: dict, test_cases: int = 200, gpu: dict = GPU_DEFAULT) -> dict:
    """The 2026-11-01 rule: proceed as planned if at least 30 main GRPO steps fit under the $40 cap; otherwise apply
    savings a, b, c in order until they do, then (d) reduce the main steps (never below 30); if 30 still do not fit,
    raise the cap to the smallest amount that fits 30, at most RAISED_CAP_LIMIT; otherwise stop before SFT."""
    rev4 = {"cases_per_step": 8, "rollouts_per_case": 8, "pilot_lrs": 2, "pilot_steps": 10, "main_steps": 40,
            "sessions": 3, "browser_cases": 0, "forgetting_reference_models": 1, "phase10_regression_models": 3,
            "test_cases": test_cases}
    p = measured_plan(measured, rev4)
    steps = [{"applied": "as planned", **_fit(p, CAP, rev4, gpu)}]
    if steps[-1]["main_steps_that_fit"] >= MIN_MAIN_STEPS:
        main = min(rev4["main_steps"], steps[-1]["main_steps_that_fit"])
        return {"decision": "proceed as planned", "main_steps": main, "cap": CAP, "savings": [], "steps": steps}
    applied = []
    for name, change in SAVINGS:
        p = {**p, **change}
        applied.append(name)
        steps.append({"applied": name, **_fit(p, CAP, rev4, gpu)})
        if steps[-1]["main_steps_that_fit"] >= MIN_MAIN_STEPS:
            return {"decision": "proceed with savings", "savings": applied,
                    "main_steps": min(rev4["main_steps"], steps[-1]["main_steps_that_fit"]), "cap": CAP, "steps": steps}
    applied.append("d: main GRPO steps reduced, never below 30")
    f30 = _fit(p, CAP, rev4, gpu)
    needed = round(f30["fixed_dollars"] + MIN_MAIN_STEPS * f30["per_main_step_dollars"] + SPENT, 2)
    steps.append({"applied": "cap needed for 30 main steps", "cap_needed": needed})
    if needed <= RAISED_CAP_LIMIT:
        return {"decision": "proceed with savings and a raised cap", "savings": applied, "main_steps": MIN_MAIN_STEPS,
                "cap": needed, "steps": steps}
    return {"decision": "stop and report before training SFT", "savings": applied, "cap_needed": needed,
            "cap_limit": RAISED_CAP_LIMIT, "steps": steps}


def decide_with_scenarios(measured: dict, test_cases: int, measured_on: str, session_price: float, volume: float,
                          measurement_hours: float = MEASUREMENT_BLOCK_HOURS, rule_on: str = "H100") -> dict:
    """The rule on the all-H100 scenario (2026-11-02), which is the decision, with the A100 scenario reported beside
    it. `rule_on="A100"` puts the rule on A100 instead, with H100 beside it. It is used only when that choice has been
    confirmed beforehand."""
    h100, a100 = (gpu_scenario(later, measured_on, session_price, volume, measurement_hours) for later in ("H100", "A100"))
    if rule_on == "A100":
        return {**decide(measured, test_cases, a100), "scenario": "sessions 2 and 3 on A100",
                "h100_sessions_2_3": decide(measured, test_cases, h100)}
    return {**decide(measured, test_cases, h100), "scenario": "sessions 2 and 3 on H100",
            "a100_sessions_2_3": decide(measured, test_cases, a100)}


def choose_by_total(measured: dict, test_cases: int, gpu: dict, target: float = TOTAL_TARGET,
                    ceiling: float = RAISED_CAP_LIMIT) -> dict:
    """The rule for the rest of session 1 (2026-11-02): the full plan if the whole experiment's forecast is at most
    `target`; otherwise the recorded savings in order until it is, then fewer main GRPO steps (never below 30). Above
    `target` at 30 main steps but within `ceiling`: that last configuration. Above `ceiling`: stop before SFT."""
    rev4 = {"cases_per_step": 8, "rollouts_per_case": 8, "pilot_lrs": 2, "pilot_steps": 10, "main_steps": 40,
            "sessions": 3, "browser_cases": 0, "forgetting_reference_models": 1, "phase10_regression_models": 3,
            "test_cases": test_cases}
    p = measured_plan(measured, rev4)

    def total(plan: dict) -> float:
        return round(SPENT + _fit(plan, CAP, rev4, gpu)["total_dollars_at_40_main_steps"], 2)

    def result(decision: str, savings: list, main: int, forecast: float) -> dict:
        return {"decision": decision, "savings": savings, "main_steps": main, "cap": forecast, "forecast_total": forecast,
                "target": target, "ceiling": ceiling, "spent": SPENT, "steps": steps}
    steps = [{"applied": "as designed", "forecast_total": total(p)}]
    if steps[-1]["forecast_total"] <= target:
        return result("proceed as designed", [], rev4["main_steps"], steps[-1]["forecast_total"])
    applied = []
    for name, change in SAVINGS:
        p = {**p, **change}
        applied.append(name)
        steps.append({"applied": name, "forecast_total": total(p)})
        if steps[-1]["forecast_total"] <= target:
            return result("proceed with savings", applied, rev4["main_steps"], steps[-1]["forecast_total"])
    applied.append("d: main GRPO steps reduced, never below 30")
    for main in range(rev4["main_steps"] - 1, MIN_MAIN_STEPS - 1, -1):
        forecast = total({**p, "main_steps": main})
        steps.append({"applied": f"d: {main} main steps", "forecast_total": forecast})
        if forecast <= target:
            return result("proceed with savings", applied, main, forecast)
    if forecast <= ceiling:
        return result("proceed with savings, above the target but within the ceiling", applied, MIN_MAIN_STEPS, forecast)
    return {"decision": "stop and report before training SFT", "savings": applied, "forecast_total": forecast,
            "target": target, "ceiling": ceiling, "spent": SPENT, "steps": steps}


def phase66_trigger(measured: dict, gpu: dict) -> dict:
    """The 6.6 trigger (the test extension to 300 cases, in the pre-registered plan preregistration_proof/PLAN_V2.md,
    section 4), computed for the report. It fires if the full main run fits the $40 GPU cap under the central forecast
    and the 300-case scenario leaves at least $3 of central headroom."""
    rev4 = {"cases_per_step": 8, "rollouts_per_case": 8, "pilot_lrs": 2, "pilot_steps": 10, "main_steps": 40,
            "sessions": 3, "browser_cases": 0, "forgetting_reference_models": 1, "phase10_regression_models": 3}
    fit = _fit(measured_plan(measured, {**rev4, "test_cases": 300}), CAP, rev4, gpu)
    headroom = round(CAP - SPENT - fit["total_dollars_at_40_main_steps"], 2)
    fits = fit["main_steps_that_fit"] >= rev4["main_steps"]
    return {"full_main_run_fits_40": fits, "main_steps_that_fit_at_300": fit["main_steps_that_fit"],
            "headroom_at_300": headroom, "fires": fits and headroom >= 3.0}


# Measured in session 1, for the forecast from here on. SFT: 3,307 s for 600 row-epochs. RFT: 240 rows from the 120 profiled
# cases, one epoch, its rows built from session 1's profiling; one more greedy test evaluation in Phase 10.
SFT_SECONDS_PER_ROW_EPOCH = 3307 / 600
RFT_ROWS = 240
# Evaluation rate for the Phase 8 rule (2026-11-03, before any GRPO output), all evaluations: they run as separate processes with
# the whole GPU. The slowest of session 1's dev and profiling runs (956 greedy, 830 profiling, 720 dev samples; siblings, 447, are a
# small batch and not a case set evaluated from here on).
EVAL_EPISODES_PER_HOUR = 720.0
FULL_MAIN_STEPS, FULL_TEST_CASES, FULL_RFT_SAMPLES = 40, 300, 8    # the full plan, no cuts (2026-11-04)
# The selected checkpoint on dev both ways, bf16-merged and runtime LoRA, greedy and 8 samples each (2026-11-03): descriptive.
GAP_PROBE_EPISODES = 2 * (60 + 60 * 8)
PHASE8_CHECK_HOURS = 1.25       # one real step (about 21.5 min), overfitting one case (about 40 min), tests on the pod (about 10 min)
APPROVED_CAP_MARGIN = 2.00      # the Phase 8 cap is the chosen configuration's forecast plus this for unmodelled costs, at most the ceiling


def _with_step(measured: dict, generation_minutes: float, seconds_per_rollout: float) -> dict:
    """The measured figures with a real step's. Its time is on the wall clock, syncing and logging included, so no estimate for them is added."""
    return {**measured, "step_generation_measured_min": generation_minutes, "seconds_per_sequence_update_with_kl": seconds_per_rollout, "step_other_min": 0.0}


def after_session_1(measured: dict, test_cases: int, price: float, main_steps: int = 40, saving_c: bool = True,
                    episodes_per_hour: float | None = None, move_hours: float = 0.0, volume: float = 0.0, spent: float | None = None,
                    rft: bool = True, rft_trained: bool = False, phase8: bool = True, done: dict | None = None,
                    gap_probe: bool = False, rft_samples: int = 0, startups_left: int | None = None) -> dict:
    """The whole experiment now that session 1 is paid for: `spent` (SPENT unless given) plus what is left. decide and
    choose_by_total forecast session 1 as well, so from here they count it twice. `rft`: its test evaluation in Phase 10, and
    training it in session 2 unless `rft_trained` (2026-11-03). `phase8`: the checks' own time. Once those have run, their
    time is in `spent`, so these say so.
    `rft_samples`: RFT's pass@8 on the test set, beside its greedy evaluation (2026-11-04). `startups_left`: sessions still to start, if a running one's is in `spent`.
    `move_hours`: time on the GPU to set up elsewhere. `done`: GRPO work already in `spent`, as {"pilot_steps", "main_steps",
    "evaluations"}. A main run that does not end on a multiple of 10 has one evaluation more than forecast() counts, because the GPU
    session script (not part of this release) also evaluates its last step. `gap_probe`: the selected checkpoint scored both ways, after the main run."""
    rev4 = {"cases_per_step": 8, "rollouts_per_case": 8, "pilot_lrs": 2, "pilot_steps": 10, "main_steps": main_steps,
            "sessions": 3, "browser_cases": 0, "forgetting_reference_models": 1, "phase10_regression_models": 3,
            "test_cases": test_cases}
    p = measured_plan(measured, rev4)
    if saving_c:
        p = {**p, **dict(SAVINGS)["c: pass@4 instead of pass@8 in Phase 10"]}
    if episodes_per_hour:
        p = {**p, "episodes_per_hour": episodes_per_hour}
    f = forecast(p)
    later = sum(v for k, v in f["hours"].items() if k not in SESSION_1_ITEMS and k != "session overhead")
    hours = later + (p["sessions"] - 1 if startups_left is None else startups_left) * p["session_overhead_min"] / 60 + move_hours
    if rft:
        hours += RFT_TEST_MODELS * test_cases * (1 + rft_samples) / p["episodes_per_hour"]
        if not rft_trained:
            hours += RFT_ROWS * SFT_SECONDS_PER_ROW_EPOCH / 3600
    if phase8:
        hours += PHASE8_CHECK_HOURS
    if gap_probe:
        hours += GAP_PROBE_EPISODES / p["episodes_per_hour"]
    evaluation = (p["dev_cases"] * p["alarm_samples"] + p["dev_cases"] + p["regression_cases"]) / p["episodes_per_hour"]
    if main_steps % p["alarm_every"]:
        hours += evaluation
    if done:
        hours -= (done["pilot_steps"] + done["main_steps"]) * f["minutes_per_step"] / 60 + done["evaluations"] * evaluation
    paid = SPENT if spent is None else spent
    return {"forecast_total": round(paid + hours * price + volume, 2), "hours_left": round(hours, 1), "spent": paid,
            "minutes_per_step": round(f["minutes_per_step"], 2), "main_steps": main_steps, "price_per_hour": price,
            "test_cases": test_cases, "pass_at": 4 if saving_c else 8, "rft_included": rft}


def phase8_decision(measured: dict, generation_minutes: float, seconds_per_rollout: float, spent: float, volume: float, price: float = PRICE) -> dict:
    """The rule applied at Phase 8 (2026-11-04), the same for every outcome (A, B, C or D) of the Phase 8 padding-trim diagnostic: the full plan under RAISED_CAP_LIMIT (40 main steps, pass@8, 300 test cases, RFT's test
    evaluation at greedy and pass@8, the gap probe). If it does not fit, the first that does of: without the 300-case extension; then pass@8 goes, with the three trained models
    together at pass@4 (base stays greedy-only); then main steps, never below 30. RFT's test evaluation stays. If none fits, a stop.
    `measured`: part measure's figures, with the real step's generation and per-rollout update put in; evaluation at EVAL_EPISODES_PER_HOUR. The time of RFT and the checks is in
    `spent` by then. The cap set here (`approved_cap`) is the forecast plus $2 for unmodelled costs, never above the ceiling."""
    step = _with_step(measured, generation_minutes, seconds_per_rollout)

    def total(main: int, test: int, samples: int) -> float:
        return after_session_1(step, test, price, main_steps=main, saving_c=samples == 4, episodes_per_hour=EVAL_EPISODES_PER_HOUR, volume=volume, spent=spent, rft=True, rft_trained=True,
                               phase8=False, gap_probe=True, rft_samples=samples, startups_left=1)["forecast_total"]

    def proceed(main: int, test: int, samples: int, ceiling: float) -> dict:
        at = total(main, test, samples)
        return {"decision": "proceed", "main_steps": main, "pass_at": samples, "test_cases": test, "rft_test": True, "rft_samples": samples,
                "forecast_total": at, "approved_cap": round(min(at + APPROVED_CAP_MARGIN, ceiling), 2), "ceiling": ceiling, "rule": rule}
    rule = {"ceiling": RAISED_CAP_LIMIT, "episodes_per_hour": EVAL_EPISODES_PER_HOUR, "generation_minutes": generation_minutes, "seconds_per_rollout": seconds_per_rollout,
            "spent": spent, "price_per_hour": price, "full_plan_forecast": total(FULL_MAIN_STEPS, FULL_TEST_CASES, FULL_RFT_SAMPLES)}
    order = [(FULL_MAIN_STEPS, FULL_TEST_CASES, FULL_RFT_SAMPLES), (FULL_MAIN_STEPS, 200, FULL_RFT_SAMPLES)] + [(n, 200, 4) for n in range(FULL_MAIN_STEPS, MIN_MAIN_STEPS - 1, -1)]
    for main, test, samples in order:
        if total(main, test, samples) <= RAISED_CAP_LIMIT:
            return proceed(main, test, samples, RAISED_CAP_LIMIT)
    return {"decision": "stop: 30 main steps with the cuts are above the ceiling", "forecast_total": total(MIN_MAIN_STEPS, 200, 4), "ceiling": RAISED_CAP_LIMIT, "rule": rule}


def grpo_progress(grpo: pathlib.Path) -> dict:
    """What the GPU session script (not part of this release) has finished under $PERSIST/grpo, from the checkpoints and the evaluations' .done files. The main run starts
    from a copy of the winner's step 10, so its steps are counted from there."""
    pilot = sum((d / "checkpoint-10" / "trainer_state.json").exists() for d in grpo.glob("pilot-*"))
    main = [int(c.name.split("-")[1]) for c in grpo.glob("main/checkpoint-*") if (c / "trainer_state.json").exists()]
    return {"pilot_steps": 10 * pilot, "main_steps": max([n - 10 for n in main], default=0),
            "evaluations": len(list(grpo.glob("pilot-*/eval-*/.done"))) + len(list(grpo.glob("main/eval-*/.done")))}


def left_check(measured: dict, step: dict, decision: dict, spent: float, volume: float, done: dict, price: float = PRICE) -> dict:
    """Spending so far plus the forecast for what is left, in the configuration Phase 8 decided, against the cap Phase 8 set.
    The GPU session script runs it before each stretch of GRPO. The step's measured time: `spent` holds what the steps really took."""
    m = _with_step(measured, step["generation_minutes"], step["seconds_per_rollout"])
    res = after_session_1(m, decision["test_cases"], price, main_steps=decision["main_steps"], saving_c=decision["pass_at"] == 4,
                          episodes_per_hour=EVAL_EPISODES_PER_HOUR, volume=volume, spent=spent, rft=decision["rft_test"], rft_trained=True,
                          phase8=False, done=done, gap_probe=True, rft_samples=decision.get("rft_samples", 0), startups_left=1)
    cap = min(decision["approved_cap"], RAISED_CAP_LIMIT)
    return {**res, "done": done, "approved_cap": cap, "within_cap": res["forecast_total"] <= cap}


def regenerate_decision(measured_path: str, out_path: str, measured_on: str, session_price: float, volume: float,
                        rule_on: str = "H100") -> dict:
    """decision.json again from a measured.json: the current ceiling and spending, with the measurement block already in
    SPENT_GPU, so it is not counted twice."""
    with open(measured_path) as f:
        measured = json.load(f)
    decision = {f"test {n}": decide_with_scenarios(measured, n, measured_on, session_price, volume, measurement_hours=0.0, rule_on=rule_on)
                for n in (200, 300)}
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=1)
    return decision


def gpu_options() -> dict:
    """The options before any measurement (2026-11-02): the central figures, with an H100's speed assumed. The GPU's
    speed scales generation, the update and SFT alike."""
    central = {"episodes_per_hour": PLAN["episodes_per_hour"] * PHASE06["completion_tokens_mean"] / PHASE4["completion_tokens_mean"],
               "step_generation_minutes": PHASE4["turns_max"] * PHASE06["wall_seconds"] / PHASE06["rounds"] / 60,
               "seconds_per_sequence_update_with_kl": 6.0 * PHASE4["context_tokens_mean"] / UPDATE_BASIS_TOKENS * (1 + KL_FORWARD_SHARE)}
    out = {}
    for name, price, on, ratio, per_gb in (
            ("A100 SXM, Standard volume", PRICE, "A100", 1.0, STANDARD_VOLUME_PER_GB_MONTH),
            ("H100 SXM at 1.5x A100, Standard volume", H100_SXM_PRICE, "H100", 1.5, STANDARD_VOLUME_PER_GB_MONTH),
            ("H100 SXM at 2x A100, Standard volume", H100_SXM_PRICE, "H100", 2.0, STANDARD_VOLUME_PER_GB_MONTH),
            ("H100 SXM at 2.5x A100, Standard volume", H100_SXM_PRICE, "H100", 2.5, STANDARD_VOLUME_PER_GB_MONTH),
            ("H100 SXM at 2x A100, High Performance volume", H100_SXM_PRICE, "H100", 2.0, HIGH_PERFORMANCE_VOLUME_PER_GB_MONTH),
            ("H100 PCIe at 1.5x A100, Standard volume", H100_PCIE_PRICE, "H100", 1.5, STANDARD_VOLUME_PER_GB_MONTH),
            ("H100 NVL at 2x A100, Standard volume", H100_NVL_PRICE, "H100", 2.0, STANDARD_VOLUME_PER_GB_MONTH)):
        m = {"episodes_per_hour": central["episodes_per_hour"] * ratio, "sft_speed_ratio": ratio,
             "step_generation_minutes": central["step_generation_minutes"] / ratio,
             "seconds_per_sequence_update_with_kl": central["seconds_per_sequence_update_with_kl"] / ratio}
        d = decide_with_scenarios(m, 200, on, price, volume_dollars(60, per_gb, 1))
        a = d["a100_sessions_2_3"]
        out[name] = {"session_1_dollars": d["steps"][0]["session_1_dollars"], "volume_dollars": d["steps"][0]["volume_dollars"],
                     "session_1_rft_dollars": d["steps"][0]["session_1_rft_dollars"],
                     "all_h100": {k: d.get(k) for k in ("decision", "main_steps", "cap", "cap_needed", "savings")}
                     | {"as_planned_main_steps_that_fit": d["steps"][0]["main_steps_that_fit"],
                        "as_planned_total": d["steps"][0]["total_dollars_at_40_main_steps"]},
                     "a100": {k: a.get(k) for k in ("decision", "main_steps", "cap", "cap_needed", "savings")}
                     | {"as_planned_main_steps_that_fit": a["steps"][0]["main_steps_that_fit"],
                        "as_planned_total": a["steps"][0]["total_dollars_at_40_main_steps"]}}
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "choose":
        ap = argparse.ArgumentParser(description="decision.json by the whole-experiment forecast (user, 2026-11-02)")
        ap.add_argument("cmd")
        ap.add_argument("--measured", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--volume-gb", type=float, required=True)
        ap.add_argument("--volume-per-gb-month", type=float, required=True)
        ap.add_argument("--volume-months", type=float, default=1.0)
        a = ap.parse_args()
        with open(a.measured) as f:
            measured = json.load(f)
        volume = volume_dollars(a.volume_gb, a.volume_per_gb_month, a.volume_months)
        a100 = gpu_scenario("A100", "A100", PRICE, volume, measurement_hours=0.0)
        h100 = gpu_scenario("H100", "A100", PRICE, volume, measurement_hours=0.0)
        out = {f"test {n}": {**choose_by_total(measured, n, a100), "scenario": "all sessions on A100",
                             "h100_for_information": choose_by_total(measured, n, h100)} for n in (200, 300)}
        out["phase66_trigger"] = phase66_trigger(measured, a100)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out, indent=1))
        return
    if len(sys.argv) > 1 and sys.argv[1] in ("rft-check", "phase8", "left"):
        ap = argparse.ArgumentParser(description="Session 2: is RFT within the budget rule; the rule applied at Phase 8")
        ap.add_argument("cmd")
        ap.add_argument("--measured", required=True, help="part measure's measured.json")
        ap.add_argument("--spent", type=float, required=True, help="all GPU spending so far, this pod included")
        ap.add_argument("--volume", type=float, default=4.20)
        ap.add_argument("--price-per-hour", type=float, required=True, help="this session's actual GPU price (user, 2026-11-03)")
        ap.add_argument("--step-json", help="rft-check: a real step measured on this GPU, when it is not the A100 the figures come from")
        ap.add_argument("--checks", help="phase8 and left: checks.json from scripts/gpu/phase8.py")
        ap.add_argument("--decision", help="left: Phase 8's decision.json")
        ap.add_argument("--grpo", help="left: $PERSIST/grpo")
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        with open(a.measured) as f:
            measured = json.load(f)
        if a.cmd == "rft-check":
            # The recorded cut order: RFT goes after the test extension and extra seeds, and before any GRPO step. So it is
            # trained only if the whole plan with it, as is, keeps the 40 main steps, at the session 1 forecast's padding-trimmed
            # step (11.86 min of generation, 8.563 s per rollout) until Phase 8 measures one.
            # On another GPU, the step this GPU really took: at A100's speed and an H100's price it could not pass, whatever the H100 does.
            m = {**measured, "step_generation_measured_min": 11.86, "seconds_per_sequence_update_with_kl": 8.563}
            if a.step_json:
                with open(a.step_json) as f:
                    here = json.load(f)["measured"]
                m = {**measured, "step_generation_measured_min": here["generation_minutes"], "seconds_per_sequence_update_with_kl": here["seconds_per_rollout"]}
            res = after_session_1(m, 200, a.price_per_hour, saving_c=True, episodes_per_hour=EVAL_EPISODES_PER_HOUR, volume=a.volume, spent=a.spent,
                                  gap_probe=True)
            res["limit"] = RAISED_CAP_LIMIT
            res["within_limit"] = res["forecast_total"] <= RAISED_CAP_LIMIT
            code = 0 if res["within_limit"] else 2
        elif a.cmd == "left":
            with open(a.checks) as f:
                step = json.load(f)["measured"]
            with open(a.decision) as f:
                decided = json.load(f)
            res = left_check(measured, step, decided, a.spent, a.volume, grpo_progress(pathlib.Path(a.grpo)), price=a.price_per_hour)
            code = 0 if res["within_cap"] else 2
        else:
            with open(a.checks) as f:
                checks = json.load(f)
            step = checks["measured"]
            res = phase8_decision(measured, step["generation_minutes"], step["seconds_per_rollout"], a.spent, a.volume, price=a.price_per_hour)
            code = 0 if res["decision"] == "proceed" else 2
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
        sys.exit(code)
    if len(sys.argv) > 1 and sys.argv[1] == "decide":
        ap = argparse.ArgumentParser(description="Regenerate decision.json from a measured.json")
        ap.add_argument("cmd")
        ap.add_argument("--measured", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--gpu", choices=("A100", "H100"), required=True)
        ap.add_argument("--price-per-hour", type=float, required=True)
        ap.add_argument("--volume-gb", type=float, required=True)
        ap.add_argument("--volume-per-gb-month", type=float, required=True)
        ap.add_argument("--volume-months", type=float, default=1.0)
        ap.add_argument("--rule-on", choices=("A100", "H100"), default="H100", help="A100 only on the user's confirmation")
        a = ap.parse_args()
        d = regenerate_decision(a.measured, a.out, a.gpu, a.price_per_hour,
                                volume_dollars(a.volume_gb, a.volume_per_gb_month, a.volume_months), a.rule_on)
        print(json.dumps({k: {"scenario": v["scenario"], "decision": v["decision"], "dollars": v.get("cap") or v.get("cap_needed")}
                          for k, v in d.items()}, indent=1))
        return
    out = {"price_per_hour": PRICE, "cap": CAP, "spent": SPENT,
           "spent_by_source": {"gpu": SPENT_GPU, "api": SPENT_API}, "plan": PLAN, "forecasts": {}}
    decided = {"cases_per_step": 8, "rollouts_per_case": 8, "pilot_lrs": 2, "pilot_steps": 10, "main_steps": 50,
               "sessions": 3}   # decided 2026-10-30 (sessions: Phases 6-7, 8-9, 10)
    for size, over in {"16x16 (plan as sized)": {}, "8x8": {"cases_per_step": 8, "rollouts_per_case": 8},
                       "decided: 8x8, pilot 2x10, main up to 50, 3 sessions": decided}.items():
        out["forecasts"][size] = {name: forecast({**PLAN, **s, **over}) for name, s in SCENARIOS.items()}
    # Decided when the splits were built: test 200 (or 300 if the 6.6 trigger fires), V6 100, policy shift 44, profiling
    # a stratified 250 of the 480 RL cases.
    out["phase2_gpu_margin"] = {
        f"test {n}": {name: {k: forecast({**PLAN, **s, **decided, "test_cases": n})[k]
                             for k in ("total_dollars", "remaining", "headroom", "steps_that_fit")}
                      | {"main_steps_that_fit": forecast({**PLAN, **s, **decided, "test_cases": n})["steps_that_fit"]
                         - decided["pilot_lrs"] * decided["pilot_steps"],
                         "final_evaluation_dollars": forecast({**PLAN, **s, **decided, "test_cases": n})["dollars"][
                             "final evaluation (test)"]}
                      for name, s in SCENARIOS.items()} for n in (200, 300)}
    out["decided_gpu_margin"] = {name: {k: forecast({**PLAN, **s, **decided})[k]
                                        for k in ("spent", "remaining", "headroom", "steps_that_fit")}
                                 for name, s in SCENARIOS.items()}
    # Decisions of 2026-10-31 on the plan's fourth revision: the browser slice deferred and removed (the study has no browser result); the
    # pilot 2 x 10 on RL cases 0-79, the winner continued to 50 updates in total (40 more steps, cases 80-399); the
    # L1 forgetting set replaces the earlier regression split (same 40 cases per checkpoint scoring), plus SFT-v2's own
    # score on it (the regression rule's reference) and its Phase 10 evaluation (base, SFT-v2, GRPO: the same three
    # models as the greedy test evaluation, decided later on 2026-10-31). beta 0.02 and T 0.8 do not enter
    # the arithmetic (beta's extra reference forward pass is measured in Phase 8, and the forecast updated then).
    rev4 = {**decided, "main_steps": 40, "browser_cases": 0, "forgetting_reference_models": 1,
            "phase10_regression_models": 3}
    out["gpu_margin_2026_10_31"] = {
        f"test {n}": {name: (lambda f: {k: f[k] for k in ("total_dollars", "remaining", "headroom", "steps_that_fit",
                                                          "checkpoint_scorings", "alarm_evaluations")}
                             | {"main_steps_that_fit": f["steps_that_fit"] - rev4["pilot_lrs"] * rev4["pilot_steps"],
                                "dollars": f["dollars"]})(forecast({**PLAN, **s, **rev4, "test_cases": n}))
                      for name, s in SCENARIOS.items()} for n in (200, 300)}
    # The 6.6 trigger (2026-10-31): extend to 300 only if, after Phase 6, the full main run fits the central
    # forecast AND the 300-case scenario leaves at least $3 of central headroom. Evaluated here on today's figures; it
    # is decided on the figures after Phase 6. The pessimistic figure is reported alongside.
    c300 = out["gpu_margin_2026_10_31"]["test 300"]
    out["phase66_trigger_today"] = {
        "full_main_run_fits_central": c300["central"]["main_steps_that_fit"] >= rev4["main_steps"],
        "central_headroom_300": c300["central"]["headroom"], "at_least_3": c300["central"]["headroom"] >= 3.0,
        "fires": c300["central"]["main_steps_that_fit"] >= rev4["main_steps"] and c300["central"]["headroom"] >= 3.0,
        "pessimistic_headroom_300": c300["pessimistic"]["headroom"]}
    out["reforecast_2026_11_01"] = phase4_reforecast(rev4)
    out["gpu_options_2026_11_02"] = gpu_options()
    remaining = API_CAP - SPENT_API_TOTAL
    out["api"] = {"cap": API_CAP, "spent": SPENT_API_TOTAL, "remaining_before_planned": round(remaining, 2),
                  "per_episode_gate1a": round(API_PER_EPISODE, 4), "per_episode_largest_gate1a": API_PER_EPISODE_LARGEST,
                  "forecast": api_forecast(CLAUDE_ROW_CASES),
                  "forecast_at_largest_episode_cost": api_forecast(CLAUDE_ROW_CASES, API_PER_EPISODE_LARGEST),
                  "largest_test_set_within_cap": int(remaining / API_PER_EPISODE),
                  "largest_test_set_within_cap_at_largest_cost": int(remaining / API_PER_EPISODE_LARGEST)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
