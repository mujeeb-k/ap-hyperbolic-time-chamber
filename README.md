# Right Actions, Incomplete Evidence: Post-Training a Small Language Model in a Stateful Accounts Payable Environment

Code, models, evaluation data, and reproducibility tools for an empirical study of SFT, rejection-sampling fine-tuning (RFT), and Group Relative Policy Optimization (GRPO) on Qwen3-4B-Instruct-2507.

The environment simulates an accounts-payable investigation workflow in which agents inspect invoices, purchase orders, and receipts, interact with stateful business records, wait for delayed confirmations, and determine whether a blocked invoice should be released, remain blocked, or be left pending.

Success requires both a correct business outcome and a valid evidence report. The study examines whether post-training improves these objectives together.

**Paper:** *Right Actions, Incomplete Evidence: Post-Training a Small Language Model in a Stateful Accounts Payable Environment* (arXiv forthcoming)

**Models:** [SFT-v2](https://huggingface.co/mujeebk/ap-env-sft-v2) · [SFT adapter](https://huggingface.co/mujeebk/ap-env-sft-v2-adapter) · [GRPO adapter](https://huggingface.co/mujeebk/ap-env-grpo-seed1-adapter)

## Results

On 200 frozen test cases, the pre-registered GRPO run improved full success from **185/200 (92.5%) to 200/200 (100%)** relative to the SFT baseline. The paired gain was 7.5 percentage points (95% CI +4.0 to +11.5).

Both models took the correct final action on all 200 cases. The entire gain in full success therefore came from evidence reporting, chiefly citations of canceled invoices that the SFT model had observed but omitted.

### Main comparison

| Configuration | Full success (200 cases) |
|---|---:|
| Qwen3-4B base | 0/200 |
| SFT | 185/200 |
| RFT | 194/200 |
| GRPO, run 1 | **200/200** |

Two additional GRPO runs achieved 199/200 and 193/200. Both exceeded the original SFT baseline, but run 3's paired interval included zero, so the pre-specified replication criteria were not met.

Post-hoc controls also achieved high test scores, 196/200 with a prompt reminder and 198/200 with a second SFT run, compared against a separately evaluated SFT baseline of 187/200. These findings do not establish that the improvement is unique to reinforcement learning.

### Reference models

| Model | Test (first 100 cases) | Stress (100 cases) |
|---|---:|---:|
| Claude Sonnet 5.5 | 99/100 | 100/100 |
| Inkling 2 Small (post hoc) | 95/100 | 90/100 |

Both models were evaluated without task-specific fine-tuning on the same cases, using the same task instructions and tool interface. These are descriptive reference results, not a controlled comparison of model capabilities. Inkling 2 Small's aggregate results are reported in Supplement Table S22.

## Environment

Each episode begins with a blocked supplier invoice. The agent investigates the discrepancy through a tool interface, observes changing records and delayed confirmations, and determines the permitted business outcome.

The environment includes:

- Invoices, purchase orders, receipts, cancellations, and reversals
- Stateful tool interactions and versioned records
- Hidden delivery facts and delayed warehouse responses
- Business-hour constraints and role-based permissions
- Evidence citations and a deterministic, rule-based grader

An episode succeeds only if the agent reaches the correct business outcome, avoids prohibited or premature actions, and submits an accepted evidence report.

The same grader supplies training rewards, selects successful RFT attempts, and evaluates the models. It is grader g8 (`src/ap_sim/grading/g8.py`), which builds on the earlier versions in `src/ap_sim/grader/`.

## Additional findings

**Correct final actions were procedurally attainable.** A scripted procedural heuristic selected the correct final action on all 444 evaluated cases across four evaluation sets, without generating a scored evidence report.

**Correct decisions can coexist with incomplete calculations.** On a held-out receipt-reversal family (V6 in the code and the CSV), final-action accuracy was 99–100%, while full success was 38–44%.

**The evidence grader has limitations.** A post-hoc audit found that among 3,099 successful reports with a derived accounting figure, at least nine contained an incorrect accounting quantity. The grader enforced specified arithmetic and citation rules but did not independently verify every derived accounting figure.

The findings are specific to this synthetic AP workflow, its training configurations, and its evaluation protocol.

## Paper

*Right Actions, Incomplete Evidence: Post-Training a Small Language Model in a Stateful Accounts Payable Environment.* arXiv (forthcoming).

The paper documents the training configurations, pre-registration, deviations, evaluation protocol, limitations, and additional analyses. The technical supplement contains detailed experimental records and diagnostics. As the paper states, the software was written and reviewed largely with AI coding agents.

## Models

Three of the evaluated models are public on Hugging Face under Apache-2.0.

| Repository | What it is |
|---|---|
| [mujeebk/ap-env-sft-v2](https://huggingface.co/mujeebk/ap-env-sft-v2) | SFT-v2 as merged bf16 weights, the form the paper evaluates. Its base model is `Qwen/Qwen3-4B-Instruct-2507` |
| [mujeebk/ap-env-sft-v2-adapter](https://huggingface.co/mujeebk/ap-env-sft-v2-adapter) | The SFT-v2 LoRA adapter, which loads onto the base model |
| [mujeebk/ap-env-grpo-seed1-adapter](https://huggingface.co/mujeebk/ap-env-grpo-seed1-adapter) | The pre-registered GRPO run's LoRA adapter at the evaluated step 10. It loads onto the merged SFT-v2 weights |

Each file was hashed against its experimental record before upload and again after a fresh download. Loading the models from these repositories has not been independently tested. The RFT, GRPO run 2 and 3, and second SFT run adapters are not released.

## Held-out sets are now public

This release includes the frozen manifests of the test set, the 100-case reserve (`manifests/v2/test_reserve.json`) and the 100-case stress set (`manifests/stress/messiness.json`). The reserve and the stress set are now public and no longer held out. Treat both as public test sets from now on.

## Reproduce

The repository includes the environment, grader, case generator, frozen evaluation manifests, per-case outcome records, and scripts for recomputing the reported tables and paired tests.

The counts, paired comparisons, and confidence intervals of the paper's results tables can be recomputed from the released outcome records without a GPU. The exceptions are listed under [What is not released](#what-is-not-released).

Full evaluation transcripts and training rollouts are not included, apart from three example episodes. Consequently, the text-level failure analyses and grader false-acceptance audit cannot be independently reconstructed from this release alone.

These commands were tested on a fresh clone with Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU, model or network access is needed after the install.

```bash
uv sync --frozen --group dev
```

```bash
.venv/bin/python scripts/recompute_tables.py
```

This recomputes Table 1, Tables S9–S15 and S17, Figure 2's paired differences, and every paired interval and McNemar p-value in the text. It checks each value against the paper and exits non-zero on any difference. It also lists the cells it skips and why.

```bash
.venv/bin/python -m pytest tests/test_v2_freeze.py tests/test_v2_splits.py tests/test_stress_set.py -q
```

This rebuilds the frozen sets from the generator and checks every manifest against its freeze record.

```bash
.venv/bin/python scripts/recompute_tables.py --rerun-heuristic
```

This also reruns the scripted heuristic (the agent in `scripts/posthoc/review_item4.py`) on every case of test, V6, the policy shift and stress, through the environment and the grader, and checks each case's right final action against the CSV. `review_item4.py` itself also checks an anchor against the stored transcripts, so it does not run from this release.

```bash
.venv/bin/python preregistration_proof/verify_prereg.py
```

```bash
shasum -a 256 -c SHA256SUMS
```

These check the pre-registered plan against its timestamped record (see `preregistration_proof/README.md`) and confirm every file's checksum.

### Freeze records

The `FREEZE.json` files in `manifests/` record the sha256 of each file as it was frozen during the study. The manifests, the outcome data and the pre-registration are released byte for byte. For publication, the comments, docstrings and documentation prose of 88 files were edited, 32 of them pinned by a freeze record, with no change to code, cases or policy values. Those files no longer match their recorded hashes. The manifests, freeze records and pre-registered plan are released exactly as frozen, so their metadata still names development documents and checks that are not part of this release. `PUBLIC_EDITS.json` lists each one with the original's sha256, the released sha256 and, for Python files, a hash of the original's syntax tree without docstrings.

```bash
.venv/bin/python scripts/verify_release.py
```

This checks every unedited file named in a freeze record byte for byte, and checks that every edited Python file has its original's syntax tree.

### What needs a GPU or unreleased material

Everything above runs on a CPU from a fresh clone. These parts of the study do not:

- **Training (SFT, RFT, GRPO).** The training code is in `scripts/gpu/training.py` and `src/ap_sim/training/`. The SFT demonstrations, the RFT data and the GRPO rollouts are not released, so training cannot be rerun from this release. Training ran on Linux with an A100-80GB GPU, `uv sync --frozen --group train` and vLLM 0.30.0.
- **Evaluating models again.** This needs the same GPU stack and the evaluated weights. SFT-v2 and GRPO run 1 are on Hugging Face (see [Models](#models)). The other configurations' weights are not released. The harness entry point, `scripts/gpu/final_evaluation.py`, also hashes development scripts that are not part of this release, so it does not run end to end from this release alone.
- **Analyses that need full transcripts:** the text-level failure analyses and the grader false-acceptance audit.

## What is released

| Path | What it is | Licence |
|---|---|---|
| `src/ap_sim/` | The environment (simulator, interfaces, tools), the grader (`grading/g8.py`, `grader/`), the case generator (`generator/`, `stress/`, `forgetting.py`) and the evaluation records code | Apache-2.0 |
| `docs/POLICY.md` | The simulated AP policy the environment enforces, as a specification. Its prose was edited for publication (see [Freeze records](#freeze-records)) | CC BY 4.0 |
| `manifests/` | The frozen case manifests (`v2/`: sft, rl, dev, regression, test, test_reserve, v6, policy_shift; `stress/`; `forgetting_l1/`), the grader's freeze record (`g8/FREEZE.json`), and version 1's frozen cases (`dev_cases.json`, `frozen/`, `frozen_policy_shift/`), which the stress set's no-overlap test checks against | CC BY 4.0 |
| `scripts/gpu/final_evaluation.py`, `scripts/phase5/` | The evaluation harness: the vLLM runner for the 4B models, the scripted reference backend, and the Claude Sonnet 5.5 runner. Some training-side modules come with them because the harness imports them | Apache-2.0 |
| `scripts/verify_release.py`, `PUBLIC_EDITS.json` | The check of this copy against the freeze records, and the record of the files edited for publication | Apache-2.0 |
| Other files in `scripts/` | Training, case-building and pre-training check scripts that the harness imports. Several call files that are not part of this release | Apache-2.0 |
| `scripts/recompute_tables.py` | Recomputes the paper's results tables, paired intervals and McNemar tests from `data/outcomes.csv` and checks each against the printed value | Apache-2.0 |
| `scripts/build_outcomes.py`, `scripts/posthoc/` | How `data/outcomes.csv` was made. These need the stored transcripts, which are not released (see below) | Apache-2.0 |
| `data/outcomes.csv` | One row per evaluated episode (described below) | CC BY 4.0 |
| `data/episodes/` | Three stored episodes, exactly as recorded: test case `test-V5-12` for SFT-v2 and for GRPO seed 1, and V6 case `v6-V4-10` for SFT-v2 | CC BY 4.0 |
| `docs/preregistration/`, `preregistration_proof/` | The pre-registration record, its OpenTimestamps receipt (`.ots`), the pre-registered plan and a script that checks the plan against the record | Apache-2.0 (code), CC BY 4.0 (documents) |
| `tests/` | Tests that rebuild the frozen sets and check them against their freeze records | Apache-2.0 |

## What is not released

- **Episode transcripts.** Only the three example episodes above are released. `data/outcomes.csv` holds each episode's outcome, not its transcript.
- **Model weights in this repository.** No adapters, merged weights or checkpoints are in this repository. Three models are on Hugging Face (see [Models](#models)), and the others are not released.
- **Inkling 2 Small's per-case outcomes.** It is not in `data/outcomes.csv`. The paper reports only its aggregate counts (Supplement Table S22).
- **Costs, the learning-rate pilot, and Inkling.** These can't be recomputed from this release. The costs (Table S16) come from billing and timing records. The learning-rate pilot (Table S5) ran on dev during training, and its episodes are not released. Inkling 2 Small (Table S22) is reported in aggregate only.
- **Columns the CSV does not carry.** The recompute script lists these cells as skipped:
  - Table S9's report-valid column and the base model's outcome-correct counts, which need each record's `finish_accepted` and `outcome_correct`;
  - Table S13's (a)/(b)/(o) counts, which count derived entries, not cases;
  - Table S15's decision-error columns and weighted error, which need final actions, stop reasons and resolution times.
- **The training data and training runs.** The SFT demonstrations, RFT data and GRPO rollouts are not released. The development history is not released either.
- **The set-aside first evaluation run** (deviation D59, Supplement Table S18) is not released.

## `data/outcomes.csv`

There is one row per evaluated episode. That covers every greedy evaluation run the paper reports: 39 runs and 4,520 episodes, including the base model and the three cross-evaluation runs. It also covers Claude Sonnet 5.5's two runs (200 episodes) and the scripted heuristic (444 rows), for 5,164 rows in all. The key `(model, evaluation, set, case_id)` is unique. `case_id` is the `case_id` field of the set's manifest, for example `test-V5-12`, `v6-V4-10` or `PS117000-0-v1`. Booleans are `true` or `false`. An empty cell means the value was not computed for that row.

The paper uses these names, and `data/outcomes.csv` uses its own (Supplement Table S1). SFT is `SFT-v2`, GRPO runs 1–3 are `GRPO seed 1`–`3`, the base model is `Base`, and the second SFT run is `Second SFT run`. Full success is what the repository calls complete-task success.

| Column | Meaning |
|---|---|
| `model` | The configuration, under the CSV's names (Supplement Table S1 gives the paper's): `Base`, `SFT-v2`, `RFT`, `GRPO seed 1`–`3`, `Second SFT run`, `SFT-v2 + prompt reminder`, the cross-evaluations `GRPO seed 1, merged in bf16`, `RFT, merged in bf16` and `SFT-v2 adapter on the base model`, `Claude Sonnet 5.5`, and `Scripted heuristic` |
| `evaluation` | The session that produced the run: `primary`, `follow-up`, `controls`, `cross-evaluation`, `reference` (Claude Sonnet 5.5, run separately), or `post hoc` (the heuristic). SFT-v2 has a `primary` and a `controls` test run, and the paper uses each where its analysis says |
| `set` | The manifest set: `test` (200), `test_reserve` (the reserve, 100), `v6` (100), `policy_shift` (44), `forgetting_l1` (40), `stress` (100). Claude's `test` rows are the first 100 test cases |
| `case_id` | The case id, as in the manifest |
| `full_success` | The grader's verdict, full success (the repository's complete-task success). Empty for the heuristic, whose report is not scored |
| `right_final_action` | The business outcome the episode left equals the expected one (the record's `final_action == expected`) |
| `grader_flags` | The grader's critical flags exactly as stored, joined by `;` and sorted alphabetically. Empty for a success. A failure can have no critical flag, for example a loop or an unaccepted finish |
| `paper_labels` | Every label Table S12 assigns to the episode, joined by `;` and sorted. Computed for every failure on every set by `labels()` in `scripts/posthoc/manuscript_counts.py`, on the episode replayed through the frozen environment and grader. Empty for a success |
| `known_false_pass` | A success the post-hoc false-pass audit found with an accepted derived figure that is not the correct accounting quantity (Table S17). Empty where the audit did not run: the reserve, Claude Sonnet 5.5 and the heuristic |
| `variant_success` | Success under the lenient re-score of the grader-sensitivity analysis (Table S13). Computed only for V6 (base, SFT-v2, GRPO seeds 1–3) and stress (the seven 4B configurations). Empty elsewhere |
| `attempted_unsafe` | The number of prohibited actions the agent attempted in the episode, including refused attempts (the record's `violations.attempted`) |
| `sampled_successes_of_8` | Successes in 8 sampled runs of the case, from the primary evaluation's 8-sample test runs of SFT-v2, RFT and GRPO seed 1. Empty elsewhere |

Claude Sonnet 5.5 ran at its API's default sampling, which cannot be set, so its rows are not greedy. Its one failure's labels were computed by regrading its stored episode log, which reproduced the stored verdict and flags.

### Stored flags and paper labels

The stored flags are kept under their own names. Six paper labels come from the episode, and `no_flag` applies when none of the six does.

| Stored flag | Paper label it yields | Role in the primary category |
|---|---|---|
| (none: the final action differs from the expected one) | `wrong_decision` | 1st |
| `required_evidence_missing` | `missing_evidence`, and also `cancelled_invoice_omission` when an unsatisfied required item is about a canceled invoice | `cancelled_invoice_omission` 2nd, `missing_evidence` 3rd |
| `citation_failure` | `derived_figure` when a derived entry's stated result does not equal its expression, and `other_citation_problem` for any other citation problem. An episode can have both | `derived_figure` 4th, `other_citation_problem` counts as other (5th) |
| `discrepancy_type_wrong` | `discrepancy_type_wrong` | other (5th) |
| `disposition_contradicts_outcome` | No label of its own. Every such episode here also has the wrong final action, so it has `wrong_decision` | none |
| `pending_before_final_check`, `pending_when_resolvable`, `release_while_blocking_comparison_fails`, `release_without_current_eligibility` | No label of their own. They lead to `no_flag` if no other label applies | `no_flag` counts as other (5th) |
| (no critical flag at all) | `no_flag` | other (5th) |

**Primary category (Table S11).** Each failure gets the first of these that it has, in this order: `wrong_decision`, `cancelled_invoice_omission`, `missing_evidence`, `derived_figure`, then `other` for everything else. Table S12 counts every label directly. The stored records also carry non-critical diagnostics, such as citation-failure kinds, violation details, efficiency and timing. These are not in the CSV.

## Licences

- **Code:** Apache-2.0 (`LICENSE`).
- **Models:** Apache-2.0, for the two adapters and the merged SFT-v2 weights on Hugging Face. The base model, Qwen3-4B-Instruct-2507, has its own Apache-2.0 licence.
- **Data and documents:** CC BY 4.0 (`LICENSE-DATA`), for `manifests/`, `data/`, `docs/` and `preregistration_proof/PLAN_V2.md`.

See `NOTICE` for third-party terms that apply to model outputs.

## Acknowledgment

Supported in part by the Tinker research grant from [Thinking Machines Lab](https://thinkingmachines.ai/).
