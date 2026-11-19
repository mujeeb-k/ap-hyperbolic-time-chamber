# Training Plan v2 (Revision 6) — operational

Status: Phases 0-4 complete (Phase 4 closed 2026-11-01); splits frozen; Gate 3 passed (22f7677); Gate 4 passed and
sft-demos-v2-1 frozen (manifests/sft_demos_v2_1/FREEZE.json). No model has run on any v2 split.
docs/V2.md is authoritative for facts; if it disagrees with this file, V2.md wins. Corrected
2026-11-01 to match V2.md and the code (the fact check recorded there); rulings marked
"amended 2026-11-01".

## 1. Scoring (g8 + ap_reward_v2)

Attempt: any tool call the environment receives, including refused calls and calls after the
episode ends. A call that raises inside TRL before reaching the environment is not an attempt.

| Ending | Reward | g8 |
|---|---|---|
| Correct, safe resolution, valid report | +1 | 1 |
| Any unsafe attempt (executed, refused or post-end) | -1 | 0 |
| Wrong outcome, invalid report, no final action, loop guard | 0 | 0 |
| 40-call limit, turn cap, token limit without accepted finish | 0 (-1 if unsafe) | 0 |
| Turn ends on <\|im_end\|> or <\|endoftext\|> without finish | 0 (-1 if unsafe) | 0 |
| Accepted correct finish, result dropped by TRL at the length limit | +1, stop finish, finish_result_dropped = true | 1 |
| Final pending before the D5 16:35 check | 0 | 0 |
| infrastructure_failure (fault outside the model's control) | None (excluded), or -1 if an unsafe attempt was recorded; evaluation re-runs once, then counts as failure | - |

D5b (+0.5 for success via an unnecessary confirmation on a covered case) is built and off.

## 2. Limits and settings

- Tool calls 40; completion tokens 28,354; context 40,094; observations capped at 20,000
  characters (truncated, never raised); tool-argument limits as implemented; stop tokens 151645
  and 151643.
- SFT: LoRA rank 32, all linear layers, lr 1e-4, 2 epochs, loss on explicit per-token labels
  (pre-tokenized rows; Phase 4 item 5). Amended 2026-11-01; was "assistant-only plus
  completion-only loss".
- Training settings (pre-data decisions, 2026-11-01; configured explicitly in code and tested):
  LoRA alpha 64. SFT: effective batch 8, cosine schedule with 5% warmup. GRPO: micro-batch 8,
  loss_type dapo, scale_rewards "batch", one optimizer update per 64-rollout batch with no
  rollout reuse, constant learning rate with no warmup. Both: AdamW, gradient clipping at 1.0,
  bf16, gradient checkpointing on.
- Execution rule (v2 decision, 2026-11-01): loop-guard/1 in training, the evaluation harness and
  the Claude row.
- GRPO: 8 cases x 8 rollouts; beta 0.02; temperature 0.8; bf16; LoRA rank 32; batch-scaled
  advantages. Pilot lr 5e-6 and 2e-5, 10 steps each on rl cases 0-79 in the same order. The
  winner continues to 50 updates in total on cases 80-399; 400-479 unused.
- Phase 6 gate (pre-data decisions, 2026-11-01): twin flip 90% (both siblings reach the correct
  outcome, on pairs whose expected outcomes differ; newly generated siblings, never test cases);
  parse rate 98% (tool-call attempts that parse); calculator use 95% (episodes with a coverage
  call); report-valid 90% on dev greedy (failure path: fix the demonstrations before any RL);
  forced recovery reported without a threshold.
- pass@8: T = 0.8, top_p = 1.0, seeds 0-7, identical across models.

## 3. Splits in use

sft 300 (demonstrations); rl 480 (as above); dev 60 (selection, alarms); test 200 + reserve 100;
v6 100; policy_shift 44; forgetting_l1 40. The regression split is deprecated and unused. Any
case with a reversal is refused by the sft, rl, dev and test loaders. D-v2-1: checks that run
only the reference or scripted agents, the grader, the verifier or the reachability checker may
read the test, reserve, V6 and policy-shift cases, provided no model output on them is viewed.

## 4. Pre-registered rules

- Primary: GRPO vs SFT-v2, greedy success on the test set, paired by case; paired bootstrap,
  10,000 resamples, 95% percentile. An interval including zero means "no measurable gain".
- Hypothesis (pre-data decision, 2026-11-01): GRPO trains on the model's own attempts, so it is
  expected to improve decision commitment (lower inaction), recovery from its own errors, and the
  attempted-violation rate beyond SFT-v2, which imitates an error-free scripted agent. These three
  are reported first among the secondaries. A rise in inaction or pending rates during GRPO is
  caught by the existing alarms and reported as a finding, not tuned away.
- Secondary: pass@8; violation rate; inaction; pending rate; efficiency; outcome-correct and
  report-valid separately. Secondary analysis: the primary metrics on the unseen-signature test
  cases (within type and reweighted to the test mix), reported with the seen-signature cases on
  the same mix.
- Also reported (not pre-registered as secondary): discrepancy_type classification accuracy, the
  safe-automation rate, and resolution time against the grader's T_res. The D5a error costs
  (wrongly paying 10, wrongly holding 3, pending/keep-blocked confusion 2, one business day late
  1) are for analysis only: they prioritise metrics in the report and are not a metric or part
  of the reward.
- Descriptive: constant-strategy bounds; time to first confirmation request on pending cases;
  superseded dispositions; malformed calls; stop rates by kind.
- Everything else is exploratory.
- Success target: greedy success of at least 40%; zero executed violations; GRPO's
  attempted-violation rate no higher than SFT-v2's.
- Test extension: to 300 at 6.6 only if the full main run fits the $40 GPU cap under the
  central forecast and the 300-case scenario leaves at least $3 of central headroom.
- Claude row: the first 100 test cases in frozen order, compared with SFT-v2 and GRPO on those
  same 100; the script stops at the $30 API cap.
- Forgetting rule: disqualify a checkpoint more than 2 paired standard errors below SFT-v2 on
  forgetting_l1.
- Pilot winner: a rate with any alarm or executed violation by step 10 is out; then fewer
  stability warnings (KL above 0.1, entropy below 50% of start); then higher dev greedy success;
  on a tie, the lower rate. If both are out, stop and decide.
- Standing rule: any change to the environment, grader, reward, runners or training loop
  triggers one targeted red-team round on the changed code before the next GPU phase.

## 5. Phases

### Phase 4: SFT demonstrations (laptop)
1. reference/5 on all 300 sft cases.
2. Reasoning text: 1-3 sentences before each tool call, stating the comparison explicitly,
   using only values visible in earlier observations. (v1's written-arithmetic demonstrations
   failed their gate; the difference now is visible rules, the calculator and the contract.)
3. Reports: citations exactly per the contract (page form, value objects, the invoice line,
   follow-up versions); discrepancy_type per its definition.
4. Recoveries: 135 of the 300 (45%), 45 of each type, one planted mistake each, seeds recorded
   (amended 2026-11-01):
   - malformed call: a wrong argument name or type at a uniformly random tool-call turn (not
     invalid JSON). A missing discrepancy_type is a malformed call (TRL rejects it before the
     environment);
   - report defect: at the finish, only defects finish rejects (null discrepancy_type, missing
     required field, undefined operand, derived value as a number);
   - stale click then observe: at the first wait where the state changes, in V1, V2, V3a and V3b
     only.
   Malformed and report-defect recoveries are spread across all case types in proportion to the
   split. Never: an unsafe attempt; a disposition later superseded; an unnecessary confirmation
   on a covered case.
5. Masking (amended 2026-11-01): explicit per-token masks in pre-tokenized rows. Only the planted
   mistake turn and every tool response are masked (with the system and user messages); every
   other assistant turn is trained, before and after the mistake. SFT and GRPO renderings are
   identical token for token.
6. A closing message in every demonstration; zero calls after the accepted finish.
7. Same system prompt, USER_PREFIX, tool schemas and chat template as GRPO rollouts.
8. No case, split or group identifier in any row (test).
Review stop: show 5 complete demonstrations (each allowed recovery type; release, keep blocked
and pending) before building all 300.
Gate 4: 100% replay (+1 under g8, stop finish, only rejection is the planted mistake); twin
check (twin demonstrations identical until the hidden fact is visible; it applies to trained
turns: planted mistakes may be placed using environment state, because they are untrained and
every trained turn after them reacts only to visible results; amended 2026-11-01); per-type statistics,
recovery share and mistake types; longest demonstration under 28,354; zero post-end calls; no
identifiers.

### Phase 5: Evaluation harness (laptop)
- Qwen on vLLM with the same tool path, limits and stop classifier as training. Every runner
  (training, the evaluation harness, qwen_check.py) uses the same classifier and records
  finish_result_dropped; a test shows they agree. The parity test covers a finish without
  discrepancy_type: TRL rejects it before the environment, so it is a malformed call, not an
  attempt (amended 2026-11-01).
- Per episode: success, outcome-correct, report-valid, citation-failure kind, violations, final
  action, stop reason, finish_result_dropped, malformed calls, superseded dispositions,
  efficiency, resolution time, time to first confirmation request (pending cases),
  discrepancy_type correctness, finish_reason, last token.
- Metrics: pass@k, pass^k, Wilson intervals, the paired bootstrap, constant-strategy columns.
- The Claude row, with the cap stop.
- Gate 5: a fake replaying reference/5 scores 100% with no violations; a reckless fake shows a
  violation on every non-release case.
- Development, the Gate 5 fakes included, runs on dev cases only.
- GPU session resilience (required before Phase 6): checkpoints saved to persistent storage at
  every evaluation; resume-from-checkpoint tested on the laptop with a tiny model; a scripted
  runbook for each session; the Phase 0.6 pinned environment reused (TRL 1.16.0, vLLM 0.30.0,
  torch 2.13.0+cu130, transformers 5.20.2, numpy 2.3.5, setuptools 80.10.2); vLLM prefix caching
  enabled.
- Exit: a targeted red-team round on every change since Gate 3. Exited 2026-11-01 after round 11
  (the final round); its finding fixed and verified by tests, not by another round (V2.md).
- At exit, the evaluation and runner code is frozen (manifests/runners_v2: the harness, the stop
  classifier, the hooks, qwen_check.py, the Claude-row script, the TRL driver of the parity
  tests); tests/test_runners_freeze.py fails on any change, and any change triggers the standing
  red-team rule. One decision point for parsed calls (rollout_hooks.parse_response, a call nested
  deeper than 4,096 is malformed); nothing recurses over model-supplied nested data before it.

### Phase 6: SFT and gate (GPU session 1)
- First, a measurement block (about 30 minutes, before any SFT training; amended 2026-11-01):
  the update time per sequence on demonstration-length sequences (LoRA rank 32, with the KL
  reference pass) and vLLM generation throughput on long multi-turn prompts (prefix caching,
  realistic batch sizes); re-run the forecast with the measured figures. If the measured central
  forecast fits at least 30 main GRPO steps under the GPU cap, proceed as planned. Otherwise
  apply in order until it fits: (a) profile 120 RL cases instead of 250, stratified, seed
  recorded first; (b) gate with greedy plus 4 samples instead of 8; (c) pass@4 instead of pass@8
  as the Phase 10 secondary metric, same sampling settings; (d) fewer main GRPO steps, never
  below 30. If it still does not fit, the GPU cap may rise to at most $50; if even that is not
  enough, stop and report before training SFT.
- The session 1 benchmark also records single-episode latency, not only batch throughput.
- SFT data description: the distribution of tool calls and tokens per demonstration, by case
  type.
- Before training: an annotated tag of the pre-registration commit, its hash recorded for public
  timestamping; the SFT dataset transferred to or rebuilt on the GPU machine and its sha256
  verified against manifests/sft_demos_v2_1/FREEZE.json.
- First: a smoke test (one tool call through TRL 1.16.0 stops naturally at an end-of-turn
  token; then one GRPO step on the real vLLM path, 2 cases x 2 rollouts, with a forced tool
  round in which every rollout stops; if either fails, stop and report before SFT). Confirm
  throughput; re-forecast.
- Train SFT. Render check: SFT and GRPO prompts identical token for token, including tool
  schemas; mistake turns and tool responses untrained; every trained span ends with <|im_end|>,
  optionally followed by the single newline token the training template adds (amended
  2026-11-01).
- Gate on dev (greedy plus 8 samples) against the thresholds. Render check fails: fix. Twin
  flip below 90% after two fixes: stop. A type at 0%: add demonstrations. Report-valid low:
  fix the demonstrations. Proceeding below a threshold needs a recorded reason.
- SFTConfig: max_length covers the longest row (22,007 tokens; TRL's default 1,024 truncates
  every row); packing off unless the per-token labels are verified to survive it. Check: no SFT
  row is truncated, and neither the SFT nor the GRPO rendering contains thinking tags.
- If SFT-v2's dev success is very high, proceed under the learnable-share rule (Phase 7) and
  report the result whatever it is; no change to the SFT recipe after seeing results.
- Noise floor: run SFT-v2's greedy evaluation on dev three times on the same vLLM setup and
  record the run-to-run spread; the final report states whether the primary GRPO vs SFT-v2
  difference exceeds it, alongside the paired confidence interval.
- Merge to SFT-v2; score it on forgetting_l1. Apply the 6.6 trigger.

### Phase 7: Baselines, profiling, RFT (GPU session 1)
- Dev baselines: base model and SFT-v2. Safety baselines for the alarms.
- Profile SFT-v2 on a stratified 250 rl cases (seed recorded first), 8 samples each at the
  pass@8 settings. Report the 0/8, 1-7/8 and 8/8 shares; the unnecessary-request rate on covered
  cases (decides D5b); stop rates by kind; finish_result_dropped; malformed calls; superseded
  dispositions; premature pending.
- Learnable share below 30%: stop and decide (mostly 0/8: diagnose SFT; mostly 8/8: harder
  cases need a new frozen split).
- RFT (optional): profiling successes with no parse errors or rejections, at most 2 per case,
  one epoch.

### Phase 8: GRPO plumbing (GPU session 2)
- Before the session: a targeted red-team round on training-loop changes, including rollouts
  ending at episode end.
- Rollouts end at episode end, as in evaluation; both stop tokens.
- Checks: batch arithmetic; memory; train-inference log-probability difference small and flat;
  rendering matches the harness; malformed calls handled identically; which end token TRL keeps;
  overfit one case; step-1 success matches profiling; dev rows written; tool-response tokens
  excluded from the GRPO loss; the KL reference model is SFT-v2 (merged), not the base model;
  peak memory with training, vLLM and the 64-rollout cache fits the GPU.
- Measure step time including the KL reference pass; re-forecast. Fewer than 30 main steps fit:
  stop and decide.
- Log parse and scoring time per step (TRL's parser and the reward function), with a warning if
  any step's parse time exceeds 60 s (Phase 5 audit: the parser's worst case is about 5.5 s per
  whole-budget turn; an unknown-tool-name echo about 5.6 s; history re-decoding 0.2-0.8 s per
  rollout).

### Phase 9: Pilot and main run (GPU session 2)
- Pilot as in section 2; both arms scored on dev and forgetting_l1 at step 10; winner by the
  rule in section 4.
- Main run to 50 updates; dev and forgetting_l1 scored at steps 20, 30, 40 and 50.
- At every evaluation, read a sample of the highest-reward rollouts and log the hack
  indicators (section 6).
- Select the best dev checkpoint with no alarms that passes the forgetting rule. Dev numbers are
  not results. Extra seeds only if the budget allows.

### Phase 10: Final evaluation (GPU session 3)
- Run once. Test: greedy for the base model, SFT-v2, RFT (if built) and GRPO; pass@8 for SFT-v2
  and GRPO only. v6, policy_shift and forgetting_l1: greedy only (base, SFT-v2, GRPO). Plus the
  Claude row, pinned to the Gate 1A model and settings (claude-sonnet-5-5, thinking between
  tools, effort high, API-default sampling, loop-guard/1, token budget 28,124, turn cap 40; V2.md
  2026-11-01); if that model is unavailable, the substitute is recorded before the row runs. The
  token budget stays at Gate 1A's 28,124 (the row keeps the configuration Claude was validated
  under; either limit is about twice anything Claude used).
- Before the session: the red-team round on the Phase 10 driver re-checks evaluation parity
  (training's rollout code and the harness on the same turns) and covers the Claude-row fixes
  (rulings on round 9, F2 and F3).
- Business metrics: a confusion matrix of predicted against correct outcome on the test set;
  wrong-payment and wrongful-hold rates; expected weighted loss per 1,000 invoices with the
  10/3/2/1 weights; tokens generated and wall-clock latency per episode for each model; peak GPU
  memory per concurrent episode; the Claude row's measured cost per case.
- Failure catalogue: failures by grader flag with counts per model, and at least one annotated
  failure transcript per category.
- If an evaluation bug is found after these runs: fix it, re-run every model, report both runs.
- Report: primary, secondaries, extra metrics, descriptive analyses, exploratory results;
  per-type and generalization tables; transcripts; episodes per method; cost; limitations
  (section 8).

## 6. Monitoring

- Alarm: the dev violation, inaction or pending rate exceeds SFT-v2's by more than 2 standard
  errors (minimum 2 points) at one evaluation, or by more than 1 standard error at two in a
  row; or any executed violation. In the pilot, only the single-evaluation clause and executed
  violations apply.
- Circuit breaker: more than 2 infrastructure failures in a step, or more than 10 in the last
  500 episodes, stops the run.
- After a stop: roll back and read transcripts before changing anything.
- Warnings: more than 70% zero-variance groups; entropy below 50% of start; KL above 0.1;
  rising log-probability difference, tool failures, clipped episodes, finish_result_dropped,
  malformed calls, superseded dispositions, premature pending, refused attempts, post-end calls
  or region-form citations; any jump in reward.

## 7. Budget

GPU cap $40, API cap $30, tracked separately in scripts/budget_forecast.py. GPU budget ceiling
$50 (pre-data decision, 2026-11-01), used only under the recorded budget rule (Phase 6: the cap
may rise to at most $50 if the cut order a-d does not fit 30 main steps). Cut order if GPU
falls short (amended 2026-11-01): the test extension to 300 first, then extra seeds, then RFT,
then GRPO steps (below 30 main steps: stop and decide). RFT is the control for "GRPO just
trained more on its own successes". Never cut: the 200-case test set, the render check,
plumbing checks, alarms, reading transcripts.

Re-forecast from the Phase 4 episode lengths (2026-11-01; KL reference pass and the 8-sample
gate included; RFT at its upper bound): headroom central -$13.37 / pessimistic -$54.87 for 200
test cases, -$17.26 / -$60.74 for 300; 15 main steps fit centrally at 200. Scaled estimates,
re-measured in Phases 6 and 8 (V2.md 2026-11-01).

## 8. Limitations to state in the final report

Simulator only; synthetic cases; EUR only; one model size; receipt-coverage exceptions only (no
price variances, no multi-line POs); one policy version in training; no harder-case slice, no
browser-surface result, interface variants not tested; single seed unless the budget allows;
the verifier covers coverage, gap, resolution time, outcome and discrepancy_type (not safety
rules or citations); generalization labels come from the same grader;
gap size verified only where it changes the decision (P38); no timeliness requirement within
the horizon; corrected interim dispositions counted, not penalized; the O-9 versus Workflow rule
8 near-horizon window unresolved (unreachable); the near-horizon band (answers 5-30 minutes before
the horizon), where the grader's T_res templates assume the request comes first and the reference
agents fail some edge cases (no frozen case lies in it); infrastructure failures excluded from
training (unless an unsafe attempt makes them -1);
efficiency if D5b is not adopted; RFT if cut; Gate 3 is a stopping criterion, not a proof of
absence.
