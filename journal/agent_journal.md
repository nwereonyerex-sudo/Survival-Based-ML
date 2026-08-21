# Survival-Based ML for 10-Year Heart Attack/Stroke Risk — Project Journal

Consolidated implementation journal, covering project setup through Stage 8 (explainability).
Each entry follows CLAUDE.md §10.5: what was built, deviations from spec and why, open
questions, and anything unexpected encountered — plus the sanity check that closed the stage.

**Repository:** https://github.com/nwereonyerex-sudo/Survival-Based-ML
**Generated:** 2026-08-20

---

## Table of contents

1. [Project setup](#1-project-setup)
2. [Stage 1 — Data ingestion](#2-stage-1--data-ingestion)
3. [Stage 2 — Preprocessing](#3-stage-2--preprocessing-imputation-encoding-sex-specific-split)
4. [Stage 3 — Feature selection](#4-stage-3--feature-selection)
5. [Stage 4 — Baselines (Kaplan-Meier, CoxPH, QRISK3-style score)](#5-stage-4--baselines-kaplan-meier-coxph-qrisk3-style-score)
6. [Stage 5 — Survival ML models (RSF, GBSA)](#6-stage-5--survival-ml-models-rsf-gbsa)
7. [Stage 6 — Deep survival models (DeepSurv, DeepHit)](#7-stage-6--deep-survival-models-deepsurv-deephit)
8. [Patch — survival curves for Stage 7's integrated Brier score](#8-patch-stages-4-6--survival-curves-for-stage-7s-integrated-brier-score)
9. [Stage 7 — Evaluation](#9-stage-7--evaluation-c-index-unos-c-integrated-brier-score-calibration)
10. [Stage 8 — Explainability (SHAP, LIME, stability testing)](#10-stage-8--explainability-shap-lime-stability-testing)
11. [Results dashboard (GitHub Pages)](#11-results-dashboard-github-pages)

---

## 1. Project setup

**Date:** 2026-08-20 · **Branch:** (initial `main`)

- Adopted the approved spec as `CLAUDE.md` at the repo root (copied from
  `~/Downloads/CLAUDE.md — Coding Session Specification.md`).
- Created the §11 repo layout (`data/`, `src/`, `results/tables/`, `results/figures/`,
  `journal/`).
- Copied in `data/cvd_synthetic_dataset_v0.2.csv` and
  `data/cvd_synthetic_dataset_v0.2_metadata.xlsx` (both from `~/Downloads`) — both files
  referenced in §3 are present, no open item here.
- No modeling code written yet. Stage 1 (data ingestion) was the first session branch.

---

## 2. Stage 1 — Data ingestion

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage1-data-ingestion`

**What was built:** `src/01_data_ingestion.py` — loads
`data/cvd_synthetic_dataset_v0.2.csv`, verifies row count (100,000), column count (16), and
exact column names/order against §3 (stops with a clear error rather than proceeding if any
mismatch — none occurred), then profiles each missingness/censoring mechanism named in §4
individually rather than a generic null-count dump:
- `smoker` / `family_history_of_cardiovascular_disease`: documented the MCAR 1→0 flip
  (p=0.30) as unobservable directly in the loaded data (it's a silent value flip, not a
  deletion to NaN) — confirmed 0 NaNs in both columns, as expected.
- `systolic_blood_pressure` / `body_mass_index`: observed MCAR drop rates 9.87% / 29.89%
  against expected ~10% / ~30% — within tolerance.
- `forced_expiratory_volume_1`: observed MAR drop rates, conditional on COPD status, of
  5.05% (COPD-positive) / 75.02% (COPD-negative) against expected ~5% / ~75% — within
  tolerance.
- Censoring: profiled event rate (6.61%), administrative censoring at year 10 (93.09%), and
  informative dropout before year 10 (0.30%).
- Documented the noise/irrelevant-predictor point from §4 as a Stage 2 (§5 LASSO) concern,
  not resolved at ingestion time.

**Deviation from spec:** none. Environment note: this machine only has Python 3.14.6
available (no pyenv/conda); pandas 3.0.5 and numpy 2.5.2 installed cleanly into `.venv` with
no compatibility issues, so no fallback to an older Python was needed for this stage. Both
versions pinned into `requirements.txt`.

**Unexpected finding:** informative dropout (censored before year 10) is very rare in this
draw of the dataset — only 0.30% of the cohort — so administrative end-of-study censoring at
year 10 (93.09%) dominates almost entirely. This doesn't block anything, but it's worth
carrying into the §8 evaluation discussion: the IPCW-weighted estimators are used because the
mechanism is informative in principle, even though its practical impact on this dataset draw
looks small given how few patients are censored early.

**Open questions:** none blocking. Row/column verification and all four named missingness
rates passed within tolerance on the first run — no anomalies to flag per §0.7.

**Test/sanity check (§10.4):** ran `python src/01_data_ingestion.py` end-to-end — exits 0,
schema check passes, all four missingness-rate checks land within the ±2pp tolerance band
(no `<-- ANOMALY` flags printed).

---

## 3. Stage 2 — Preprocessing (imputation, encoding, sex-specific split)

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage2-preprocessing`

**What was built:** `src/02_preprocessing.py` — loads and re-verifies the raw CSV (§3),
imputes missing values per sex cohort, documents the encoding step, splits each sex cohort
70/15/15 stratified on the event indicator (§6, `random_state=42`), and writes six CSVs to
`data/processed/` (`{male,female}_{train,val,test}.csv`).

**Deviations from spec, both raised to and approved by the user before writing code (§14):**
1. **`smoker` / `family_history_of_cardiovascular_disease` are left untouched, not given an
   "unknown" category as §4 literally states.** Stage 1 established these columns have zero
   NaNs — the 1→0 flip (p=0.30, Burns, Richardson and Driessens, 2024) is a silent bit-flip
   with no marker distinguishing a corrupted 0 from a true 0, so there is nothing to impute or
   flag. Per the user's explicit instruction, both columns are documented as carrying known
   non-differential measurement error at p=0.30, not treated as missing data.
2. **Imputation order:** computed per-sex, on the whole cohort, *before* the 70/15/15 split —
   matching §11's literal script description ("imputation, encoding, ... split"), at the cost
   of a minor, acknowledged leak of validation/test rows into the training-fold median. User
   chose this over a stricter split-first/impute-from-training-only alternative, given the
   leak is small for a low-capacity statistic like a median.
3. **New `data/processed/` folder**, not listed in §11's repo layout, added to persist the
   split CSVs for Stages 3–8 to read directly (user's choice over having each stage
   regenerate the split deterministically via `random_state=42`).

Imputation medians (male vs female — differ enough to justify sex-specific imputation per
§0.4's independent-pipelines principle, rather than pooling): BMI 27.10 vs 27.10, SBP 130.00
vs 129.00, FEV1 (COPD-negative) 95.01 vs 95.06, FEV1 (COPD-positive) 80.01 vs 80.06.

**Unexpected finding:** none beyond Stage 1's. Split event rates land within 0.02pp of the
full-cohort rate for both sexes (well inside the 1pp sanity-check tolerance) — stratification
worked as expected.

**Open questions:** none blocking.

**Test/sanity check (§10.4):** ran `python src/02_preprocessing.py` end-to-end — exits 0. For
each sex: split sizes reconstruct the cohort exactly, no `patient_id` appears in more than one
partition, no residual NaNs in the imputed columns after the split, and event rate stays
within 1pp of the full cohort in every partition.

---

## 4. Stage 3 — Feature selection

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage3-feature-selection`

**What was built:** `src/03_feature_selection.py` — per sex, on the training fold only:
Pearson/point-biserial multicollinearity screening (§5.1), L1-penalised Cox LASSO via
`sksurv.linear_model.CoxnetSurvivalAnalysis` (Tibshirani, 1996; Pölsterl, 2020) with the
penalty strength tuned on the validation fold by C-index (§5.2), and an explicit QRISK3
cross-reference (§5.3, predictor set from §7). Writes
`results/tables/stage3_feature_selection_{male,female}.csv` with full provenance per
predictor (dropped-by-multicollinearity / LASSO coefficient / selected / in-QRISK3-basis /
disagreement / final set).

**Deviations from spec, raised to and resolved by the user before/during coding (§14):**
1. **"14 predictors" → 12.** §4/§5.1 say feature selection runs across "14 predictors," but
   only 12 dataset columns are usable as survival-model predictors — the figure only reaches
   14 by also counting `time_to_event_or_censoring` and `heart_attack_or_stroke_occurred`
   (the outcome itself). Treated as a spec wording slip; ran on the 12 actual clinical
   columns, per user instruction.
2. **FEV1/COPD multicollinearity tie-break.** The one pair that actually exceeded |r| > 0.80
   (forced_expiratory_volume_1 vs chronic_obstructive_pulmonary_disorder) has neither member
   in the QRISK3 basis, so the standard tie-break rule (§5.1) doesn't apply. User specified a
   fallback rule: recompute the correlation on complete cases only (excluding FEV1's imputed
   rows) — if it stays ≥0.80, keep the more information-rich continuous FEV1 and drop COPD;
   if it drops meaningfully below 0.80, treat the full-sample correlation as inflated by
   Stage 2's imputation and drop FEV1 instead, keeping COPD. Implemented in
   `resolve_tie_by_complete_case_correlation()`.

**Unexpected finding:** the complete-case correlation *did* drop meaningfully below the
threshold — full-sample r = −0.809 (male) / −0.808 (female) fell to −0.777 / −0.787 once each
sex's imputed FEV1 rows (24,022 male / 24,430 female — the majority of the column) were
excluded. So per the user's rule, FEV1 was dropped and COPD kept for both sexes. This is a
genuine, non-trivial result of the imputation strategy: it confirms the raw FEV1↔COPD
relationship is somewhat weaker than the full sample suggested, and it changed the LASSO
outcome materially — with FEV1 removed, COPD's coefficient goes from being shrunk to exactly
0 (when competing with correlated FEV1 in the earlier, unscreened run) to a small but nonzero
signal (−0.055) for the female cohort. This is itself a working demonstration of why the
multicollinearity screening step precedes LASSO in the first place, rather than only a
process footnote.

**Final feature sets:** male keeps 9/11 candidates (drops body_mass_index, FEV1 screened out
pre-LASSO); female keeps 10/11 (drops body_mass_index). LASSO/QRISK3 disagreements: BMI
dropped by LASSO despite QRISK3 basis (both sexes); COPD kept by LASSO despite not being a
QRISK3 predictor (female only). Both logged as disagreements per §5.3, not reconciled.

**Open questions:** none blocking.

**Test/sanity check (§10.4):** ran `python src/03_feature_selection.py` end-to-end — exits 0.
For each sex: no multicollinearity-dropped predictor leaks into the final selected set,
LASSO selects a non-empty feature set, and validation C-index at the chosen alpha is 0.807
(male) / 0.824 (female).

---

## 5. Stage 4 — Baselines (Kaplan-Meier, CoxPH, QRISK3-style score)

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage4-baselines`

**What was built:** `src/04_baselines.py` — per sex: a descriptive Kaplan-Meier curve over
the full cohort (train+val+test recombined, saved to `results/figures/km_curve_{sex}.png`),
a CoxPH model (`sksurv.linear_model.CoxPHSurvivalAnalysis`, Cox 1972) fitted on the training
fold using Stage 3's selected feature set, and the §7 simplified QRISK3-style score (fixed
formula, not fitted, uses its own predictor set regardless of Stage 3's LASSO selection).
Predictions for all three splits, both models, written to
`results/tables/stage4_baseline_predictions_{sex}.csv`. Per §11's architecture, formal
scoring (C-index, Uno's C, Brier score, calibration, bootstrap 95% CIs — §8/§12) was deferred
to Stage 7 (`07_evaluation.py`), applied uniformly across all six models rather than
duplicated here.

**Major deviation from spec, investigated, raised to and resolved by the user before coding
(§14):** §7 asks to reconstruct "the exact simplified variant Burns, Richardson and Driessens
(2024) used to simulate the outcome." Checked directly: their paper states only a qualitative
description of their QRISK3 modification (diabetes≈type 2, smoker≈light smoker,
cholesterol/HDL=3, SBP SD=10mmHg, baseline survival 0.977 male/0.989 female at 10y) — no
coefficients, equations, or code are published anywhere, and the Zenodo deposit contains only
the CSV + metadata, nothing else. Their exact coefficients are permanently unrecoverable from
public information. Per the user's instruction, this rebuild uses the real QRISK3-2017
coefficients instead (Hippisley-Cox, Coupland and Brindle, 2017), sourced from ClinRisk
Ltd.'s own LGPL-licensed reference implementation
(github.com/sisuhealthgroup/qrisk3/blob/master/src/lib/original/qrisk3.c, mirroring
qrisk.org/svn.clinrisk.co.uk, released explicitly "to enable others to implement the
algorithm faithfully"), restricted to this dataset's available predictors per §7's removal
list. This Python port was validated against ClinRisk's own published test suite (8 age×sex×
cholesterol-ratio cases, white ethnicity, no comorbidities) before restriction — all 8 matched
to within 0.05 percentage points. Corroborating evidence this is a reasonable proxy for
Burns et al.'s actual approach: their reported baseline survival values (0.977/0.989) match
the real QRISK3 survivor constants (0.977268.../0.988876...) to 3 decimal places, indicating
they anchored their simulation on the same baseline hazard function.

**Second finding, investigated and resolved with the user (§0.7, not smoothed over):** QRISK3
is only officially validated for ages 25–84 (confirmed via NICE guidance search) — 11.1% of
this dataset (11,110 patients) is under 25. For the ordinary case this doesn't matter much
(scores stay low/plausible), but the fractional-polynomial age/BMI terms are not calibrated
for that range, and for the rare combination of very young age with this dataset's
implausible synthetic BMI floor (values down to 6, no lower bound — traced one case,
age=18/BMI=10.2, to a linear-predictor contribution of +15.5 from the BMI terms alone), the
formula's survival exponentiation saturates at a score of exactly 100.0. Affects 13 patients
total (11 male, 2 female — 0.013% of the dataset). Per the user's decision, left unclipped
and unexcluded: the formula is applied identically to every patient, this behaviour is
documented in the script and here, and no arbitrary clip threshold was invented.

**Unexpected finding:** none beyond the two above.

**Open questions:** none blocking.

**Test/sanity check (§10.4):** ran `python src/04_baselines.py` end-to-end — exits 0. CoxPH
scores are non-null for every patient in every split; QRISK3-style scores fall within
[0, 100] for every patient (the 13-patient ceiling case is a valid boundary value, not an
out-of-range error); KM curves are monotonically non-increasing step functions ending at
92.05% (male) / 94.74% (female) 10-year survival, consistent with each sex's ~7.9%/~5.3%
mean QRISK3-style score and ~7.9%/~5.3% observed event rate from Stage 2.

**Patch (found and fixed while starting Stage 5, same day):** `stage4_baseline_predictions_*.csv`
only carried a CoxPH *risk score*, not a predicted 10-year survival probability — sufficient
for the C-index but not for §8's integrated Brier score / calibration plot, which need
1 − S(10|X) per patient. Added `survival_at_horizon()` to `src/04_baselines.py` (evaluates
`CoxPHSurvivalAnalysis.predict_survival_function()` at the 10-year horizon) and a new
`coxph_survival_at_10y` column. Re-ran Stage 4 end-to-end — exits 0, new column falls within
[0, 1] for every patient, mean 0.9209 (male) matches the KM 10-year survival of 0.9205, and
correlates 0.87 with the existing risk score (expected — related but non-identical
quantities). `qrisk3_style_score` already encodes a 10-year probability (÷100), so it needed
no change. Stage 5's RSF/GBSA outputs were built with both quantities from the start.

---

## 6. Stage 5 — Survival ML models (RSF, GBSA)

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage5-ml-models`

**What was built:** `src/05_survival_ml_models.py` — per sex, on Stage 3's selected feature
set (§5.4): Random Survival Forest and Gradient-Boosting Survival Analysis
(`sksurv.ensemble`), each tuned by a small grid fit on the training fold and scored by
Harrell's C-index on the validation fold (same validation-not-training/test discipline as
Stage 3's LASSO alpha and Stage 4's principle). Both risk score and predicted 10-year
survival probability are written per model to
`results/tables/stage5_ml_model_predictions_{sex}.csv`, matching the format Stage 4's patch
established. `random_state=42` throughout (§0.3).

**Runtime investigation — three background attempts before a working run, worth recording in
full since it changed the tuning grid (§14):**
1. First attempt: `RandomSurvivalForest` was fit without `n_jobs`, so all trees were built on
   a single core. 100+ estimators at unlimited depth on ~35-45k rows/sex didn't complete even
   one grid combination in 10+ minutes — killed.
2. Second attempt: added `n_jobs=-1` to RSF (fixed — full RSF grid, 4 combos × 2 sexes, then
   completed in well under 5 minutes). GBSA has no such parallelism (boosting is inherently
   sequential) and turned out to be the real bottleneck: the run appeared to stall, completing
   only 1 of 8 GBSA combinations for the male cohort after ~2 hours of wall-clock time —
   killed and investigated rather than just waiting longer or assuming a bug.
   - Isolated timing test confirmed this was not a bug or hang: `GradientBoostingSurvivalAnalysis`
     with `n_estimators=20` took ~97 seconds on the male training fold (35,352 rows) — about
     4.9s/boosting-stage. A second test at `subsample=0.5` showed `max_depth=2` and `max_depth=3`
     cost almost the same (~69s vs ~68s for 20 estimators), confirming the dominant cost is
     scikit-survival's Cox partial-likelihood loss evaluating the full risk set at every
     boosting stage, not tree-building — so cost scales with `n_estimators`, barely with depth.
     The original 8-combo grid with `n_estimators` up to 300 would have realistically taken
     3+ hours total.
3. Third attempt (successful, ~47 minutes total, user-approved after seeing the diagnosis):
   trimmed `GBSA_GRID` to 3 combos, capped `n_estimators` at 100, added `subsample=0.5`
   (stochastic gradient boosting — legitimately part of Friedman (2001), the same paper GBSA
   is cited to, and a genuine regularisation technique, not just a speed hack). Logged here as
   a practicality-driven deviation from an exhaustive grid, not a methodology change — the
   validation-set tuning principle itself is unchanged, just applied to a smaller candidate
   set.

**Results:**

| Sex | Model | Best hyperparameters | Val C-index | Test C-index (informal) |
|---|---|---|---|---|
| male | RSF | n_estimators=300, max_depth=8 | 0.8052 | 0.8073 |
| male | GBSA | n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.5 | 0.8053 | 0.8086 |
| female | RSF | n_estimators=300, max_depth=8 | 0.8209 | 0.8127 |
| female | GBSA | n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.5 | 0.8186 | 0.8141 |

Test-set C-index here is an informal sanity check only — the formal §8 evaluation (with
bootstrap 95% CIs and comparison against the sex-specific CoxPH/QRISK3 baseline) happens
uniformly for all six models in Stage 7. Both models land close to Stage 4's CoxPH baseline
(and to each other) for both sexes — a plausible, non-suspicious result, not a sign either
model is badly mis-tuned. Unlimited-depth RSF (`max_depth=None`) is consistently and
meaningfully worse than depth-capped RSF for both sexes (0.777-0.798 vs 0.805-0.821 val
C-index) — expected overfitting from unconstrained trees on a modest per-leaf sample size.

**Unexpected finding:** none beyond the runtime investigation above.

**Open questions:** none blocking.

**Test/sanity check (§10.4):** ran `python src/05_survival_ml_models.py` end-to-end — exits
0. No NaNs in either risk-score column; both survival-at-10y columns fall within [0, 1] for
every patient in every split; row counts match each sex's full cohort size (50,503 male /
49,497 female).

**Post-merge verification note:** re-ran Stage 5 fresh from `main` to confirm reproducibility.
All printed metrics (grid C-index at every combo, best hyperparameters, test C-index) matched
the committed run exactly. The per-patient CSVs showed a harmless diff confined to the RSF
columns only, at the ~15th decimal digit (e.g. `0.45010716018211305` vs `...316`) — expected
floating-point non-associativity from `RandomSurvivalForest`'s `n_jobs=-1` parallel tree
aggregation (summation order across threads isn't guaranteed identical run-to-run), not a
reproducibility failure. GBSA (single-threaded) matched byte-for-byte. Discarded the
regenerated CSVs rather than committing meaningless-diff noise.

---

## 7. Stage 6 — Deep survival models (DeepSurv, DeepHit)

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage6-deep-survival-models`

**Environment note:** the Python-3.14 compatibility risk flagged at project setup did not
materialise — `torch==2.13.0` (with a native `cp314` wheel) and `pycox==0.3.0` both installed
cleanly with no fallback to an older Python needed.

**What was built:** `src/06_deep_survival_models.py` — per sex, on Stage 3's selected
feature set (§5.4): DeepSurv (`pycox.models.CoxPH`, a neural net replacing the Cox linear
predictor — Katzman et al., 2018) and DeepHit single-risk (`pycox.models.DeepHitSingle` —
Lee et al., 2018, single-risk since this dataset has one composite outcome, per §7). Features
standardised (`StandardScaler`, fit on train only) since neural nets need scaled input,
unlike the tree/linear models in Stages 4-5. Each model tuned over a small 2-candidate grid
(hidden-layer size × learning rate) with early stopping (patience=10, max 100 epochs),
scored by C-index on the validation fold — same validation-decides principle as every prior
stage. DeepHit's discrete time grid uses `num_durations=10`, matching the dataset's existing
integer-year granularity exactly (no arbitrary binning choice). Predictions (risk score +
survival-at-10y per model) written to
`results/tables/stage6_deep_survival_predictions_{sex}.csv`.

**Runtime:** trivial compared to Stage 5's GBSA saga — smoke-tested first (learned from that
experience), confirmed a single DeepSurv fit took ~1.7 seconds with early stopping (16
epochs), then ran the full grid × both models × both sexes in 36.7 seconds total wall-clock.

**Results:**

| Sex | Model | Best hyperparameters | Val C-index | Test C-index (informal) |
|---|---|---|---|---|
| male | DeepSurv | hidden=[64,64], lr=0.001 | 0.8066 | 0.8089 |
| male | DeepHit | hidden=[64,64], lr=0.001 | 0.8065 | 0.8085 |
| female | DeepSurv | hidden=[32,32], lr=0.01 | 0.8234 | 0.8151 |
| female | DeepHit | hidden=[32,32], lr=0.01 | 0.8234 | 0.8113 |

Test-set C-index is an informal sanity check only, as in Stage 5 — formal §8 evaluation is
Stage 7's job. Results land in the same ~0.81 (male) / ~0.81-0.82 (female) band as CoxPH, RSF,
and GBSA — consistent across all five fitted models so far, no outliers.

**Unexpected finding:** none.

**Open questions:** none blocking.

**Test/sanity check (§10.4):** ran `python src/06_deep_survival_models.py` end-to-end — exits
0. No NaNs in either risk-score column; both survival-at-10y columns fall within [0, 1] for
every patient in every split; row counts match each sex's full cohort size; mean
`*_survival_at_10y` (0.9196 male DeepHit, 0.9424 female DeepHit) is consistent with the KM
and CoxPH 10-year survival estimates from Stage 4.

---

## 8. Patch (Stages 4-6) — survival curves for Stage 7's integrated Brier score

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage7-evaluation`

**What was found, before writing any Stage 7 code:** §8's integrated Brier score (Graf et
al., 1999) needs each model's predicted survival probability at *multiple* time points (a
curve over the evaluation range), not the single year-10 value Stages 4-6 previously saved —
sufficient for calibration-at-10y but not for the integrated metric. Raised to the user before
coding, per §14 (same pattern as the CoxPH gap found before Stage 5).

**What was changed:** added an `EVAL_TIME_GRID = [1..10]` (integer years, matching the
dataset's own time resolution and the grid DeepHit already used internally) to
`src/04_baselines.py`, `src/05_survival_ml_models.py`, and `src/06_deep_survival_models.py`.
Each now writes `{model}_survival_at_{t}y` for every t in 1-10 (previously only `..._at_10y`)
for CoxPH, RSF, GBSA, DeepSurv, and DeepHit. `survival_at_horizon()` (Stage 5) was superseded
by the new `survival_curve_at_grid()` and removed rather than left as dead code.

**QRISK3-style score is deliberately excluded from this patch — not an oversight.** It is a
fixed, single-horizon clinical formula (§7), not a fitted survival model, so it has no
natural multi-year curve. Stage 7 computes the standard integrated Brier score for the
five curve-producing models, and a single-timepoint (year-10) Brier score for QRISK3-style
specifically, clearly labelled as such in the results table — a direct consequence of §7's
own description of QRISK3 as fixed-horizon, not a new modelling assumption invented here.

**Also fixed (unrelated, user-reported):** the Kaplan-Meier plot title in
`src/04_baselines.py` included a `CLAUDE.md §7` citation, which rendered on the chart itself
like a stray watermark rather than a normal figure title — citations belong in code
comments/docstrings, never on a rendered figure that could end up in the thesis. Title is now
just `"Kaplan-Meier — {sex} cohort (descriptive only)"`. Checked the rest of `src/` for
similar spec-reference text baked into `set_title`/`suptitle`/`ax.text` calls — none found
elsewhere. Standing rule going forward: no spec/AI-assistant references ever get baked into a
rendered chart, for any stage.

**Verification:** re-ran all three patched stages end-to-end (Stage 5 in the background,
~45 minutes given its tuning grid). All exit 0. Every new `*_survival_at_{t}y` column is
monotonically non-increasing across t=1→10 for every patient in every split, in every model
(0 violations checked across ~50k male / ~49k female rows × 2 models × 2 patched stages, plus
Stage 4's CoxPH). Every previously-reported metric (grid C-index, best hyperparameters, test
C-index) matched the pre-patch values exactly for Stages 4 and 6; Stage 5 matched exactly
too, aside from the already-documented harmless RSF floating-point noise from `n_jobs=-1`
parallelism.

**Open questions:** none blocking.

---

## 9. Stage 7 — Evaluation (C-index, Uno's C, integrated Brier score, calibration)

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage7-evaluation`

**What was built:** `src/07_evaluation.py` — on the test set only (§6), per sex, for all six
models: Harrell's C-index (`concordance_index_censored`), Uno et al.'s (2011) IPCW
C-statistic (`concordance_index_ipcw`), and the Brier score (`integrated_brier_score` for the
five curve-producing models; single-timepoint `brier_score` for QRISK3-style — see below),
each with a bootstrap 95% CI from 1,000 resamples (§8). Every model's metrics also get a
paired-bootstrap effect size + CI against both the CoxPH and QRISK3-style baselines (§8's
last bullet — pairing uses the *same* resample indices for both models in each comparison, so
the difference's CI is valid, not just two independent CIs subtracted). A 2×3 decile-grouped
calibration grid (observed KM-estimated risk vs mean predicted risk) is plotted per sex.
Final six-model × two-sex table written to `results/tables/stage7_evaluation_summary.csv`.

**Two technical/methodological points worked out while building this stage, both logged
before writing the relevant code (§14):**
1. **IPCW evaluation horizon: 9.99y, not exactly 10y.** `sksurv`'s IPCW-weighted estimators
   (Uno's C, Brier score) require evaluation times strictly less than the test set's maximum
   follow-up — undefined exactly at that boundary, and ~93% of patients are administratively
   censored at exactly year 10 (Stage 1's censoring profile), so time=10.0 sits right on the
   boundary. Evaluated at `TAU=9.99` instead — a standard technical workaround in this
   literature (the last survival-curve step doesn't change between 9.99 and 10.0, so this is
   exact, not an approximation, for every curve-producing model).
2. **QRISK3-style gets a single-timepoint Brier score, not integrated** — it has no
   multi-year curve (§7: fixed formula, not a fitted model; see Section 8 above). Labelled
   `brier_score_type = "single_timepoint_9.99y"` in the output table so this isn't silently
   presented as equivalent to the other five models' `"integrated"` score.

**Runtime:** learned from Stage 5 — timed a small-scale (50-resample, 1-model) version first
(~42ms/resample), projected the full 1,000×6×2 run at ~500 seconds before committing to it.
Actual: 7:42 wall-clock, in line with the projection. No GBSA-style surprise this time.

**Also fixed (unrelated, user-reported):** the calibration grid originally gave each of the 6
subplots its own axis scale, making cross-model visual comparison harder than it needed to
be. Changed to a single shared scale (max across all 6 models' curves) across the whole grid.

**Notable finding, worth carrying into the thesis discussion (§0.7 — not smoothed over):**
QRISK3-style has the highest point-estimate discrimination of all six models in both sexes
(C-index 0.820 male / 0.823 female vs CoxPH's 0.810 / 0.819), and is clearly and
substantially the worst calibrated in both sexes — its Brier score 95% CI doesn't overlap
the other five models' in either sex (male: QRISK3 [0.0548, 0.0627] vs CoxPH
[0.0360, 0.0422]; female: QRISK3 [0.0375, 0.0441] vs CoxPH [0.0238, 0.0291]). The
calibration gap is a genuine, statistically distinguishable finding in both sexes. The
discrimination advantage, however, is statistically distinguishable from CoxPH only for
males (ΔC-index +0.0098, 95% CI [0.0030, 0.0168]) — for females the same comparison's CI
touches zero (Δ+0.0043, 95% CI [−0.0008, 0.0122]), so it's directional there, not
significant. This is a real example of the "improvement or honest non-improvement...
stated with effect size and CI" §12 asks for.

**Correction (2026-08-21, found while building the results dashboard — see
`src/09_build_dashboard.py`):** the paragraph above originally claimed, for the four
data-fitted ML/DL models (RSF, GBSA, DeepSurv, DeepHit) vs CoxPH, that "all differences...
are small and every CI comfortably contains zero — no fitted model shows a statistically
distinguishable improvement... in either metric, for either sex." **The "no significant
improvement" part is correct; the implication that nothing was significant is not.** Caught
by cross-checking the dashboard's computed significance markers against the raw CSV rather
than trusting the earlier prose summary — exactly the kind of check §0.7 exists for. The
accurate picture: none of the four ever shows a statistically significant *improvement* over
CoxPH on any metric, in either sex, but several show a statistically significant (small)
*regression*: for males, GBSA (ΔBrier +0.00072, 95% CI [0.00033, 0.00111]) and DeepHit
(ΔBrier +0.00088, 95% CI [0.00037, 0.00138]) calibrate significantly worse than CoxPH; for
females, RSF (ΔC-index −0.0065, 95% CI [−0.0111, −0.0012]), GBSA (ΔC-index −0.0051, 95% CI
[−0.0091, −0.0014]), and DeepHit (ΔC-index −0.0079, 95% CI [−0.0131, −0.0024]) all
discriminate significantly worse than CoxPH (the same three also show significantly worse
Uno's C). DeepSurv is the only one of the four that never differs significantly from CoxPH
on any metric, in either sex — the closest match to the linear baseline. This is a more
interesting and more honest finding than "no significant differences": CoxPH's simple
linear model is a genuinely hard baseline to beat on this dataset, and several of the more
complex models measurably underperform it on at least one axis rather than merely failing
to improve on it.

**Open questions:** none blocking.

**Test/sanity check (§10.4):** ran `python src/07_evaluation.py` end-to-end — exits 0. All
three metrics fall within their plausible [0, 1] range for every model/sex; every point
estimate falls within its own bootstrap 95% CI (verified programmatically, not just by eye);
CoxPH's and QRISK3-style's self-comparison deltas are correctly absent (not zero — genuinely
skipped) from the output table.

---

## 10. Stage 8 — Explainability (SHAP, LIME, stability testing)

**Date:** 2026-08-20 · **Branch:** `session/2026-08-20-stage8-explainability`

**What was built:** `src/08_explainability.py` — per sex, on all four ML/DL models (RSF,
GBSA, DeepSurv, DeepHit), refit using the exact best hyperparameters already found in Stages
5-6 (no re-tuning): SHAP global summary (mean |SHAP value| per feature, compared against
CoxPH coefficients in one grouped bar chart), SHAP dependence plots for the continuous
predictors, LIME local explanations on a matched 200-patient sample of the test set (same
patients used for both SHAP and LIME, for direct per-patient comparability), explicit
SHAP/LIME top-3-feature agreement per patient, and §9.4's stability test — SHAP importance
recomputed on 2 independent bootstrap resamples of the training fold (seeds 43/44) per model,
with any predictor whose rank shifts by more than 2 positions across the three fits (main +
2 resamples) flagged as unstable rather than reported as "robust."

**Two spec-vs-reality gaps, investigated and resolved before coding, per §14:**
1. **SHAP explainer choice.** `shap.TreeExplainer` does not support `sksurv` models —
   confirmed directly (`InvalidModelError: Model type not yet supported by TreeExplainer:
   <class 'sksurv.ensemble.forest.RandomSurvivalForest'>`), not assumed. Used
   `shap.Explainer` with each model's `.predict()` as a black-box function instead, uniformly
   across all four model types. With only 9-10 features, SHAP automatically selects its
   exact-computation explainer (not the slow Kernel approximation) — confirmed via timing
   test: ~0.09-0.15s/sample regardless of model type. GBSA's already-known ~6-minute fit time
   (Stage 5) was the only real cost; SHAP/LIME explanation itself is fast for every model
   (LIME: ~0.02s/sample).
2. **BMI dependence plot.** §9 names "age, systolic BP, BMI" for dependence plots, but BMI
   was dropped by Stage 3's LASSO for *both* sexes — confirmed neither sex's final model
   feature set includes it (only `age` and `systolic_blood_pressure` remain as continuous
   predictors for both). Dependence plots cover those two; BMI's absence is a direct,
   already-logged consequence of Stage 3's own feature selection, not a new deviation.

**Runtime and an unplanned interruption:** GBSA's known ~6-minute-per-fit cost (3 fits per
sex: main + 2 stability resamples) dominates, as expected from Stage 5. First attempt was
killed mid-run by something external (not a code bug — the male cohort had already completed
and written all its output correctly by that point; `ps`/output logs show no error, just an
abrupt termination partway through the female cohort's GBSA fit, most likely an environment
interruption rather than anything in the script, since several other Stage 5/6/7 background
runs of similar or longer duration completed without issue earlier in this project). Verified
the male cohort's partial output was valid and complete before relaunching the full script
end-to-end rather than trying to stitch together a resumed run. Second attempt completed
cleanly in 44:46 — male cohort's results reproduced *exactly* against the killed run's
partial output, confirming determinism was unaffected by the interruption.

**Notable findings, worth carrying into the thesis discussion (§0.7 — not smoothed over):**
1. **Tree ensembles concentrate importance on age; CoxPH and the deep models spread it
   across comorbidities — consistent in both sexes.** RSF and GBSA's SHAP importance is
   dominated overwhelmingly by `age` (every other predictor is comparatively minor), while
   CoxPH's coefficients and DeepSurv/DeepHit's SHAP values distribute meaningfully across
   `diabetes`, `atrial_fibrillation`, `family_history_of_cardiovascular_disease`, and
   `hypertension_treated` as well. This is a genuine cross-model disagreement about *which*
   predictors matter, not just their relative ranking — worth flagging as a real modelling
   difference, not an artefact of one run.
2. **DeepSurv and DeepHit disagree with RSF/GBSA on the *functional form* of the systolic
   blood pressure relationship, not just its magnitude.** RSF/GBSA's SBP dependence plots are
   roughly monotonic increasing (higher SBP → higher SHAP contribution, the clinically
   expected direction). DeepSurv and DeepHit instead show a clear U-shape — elevated SHAP
   contribution at *both* low (~80-100 mmHg) and high (160mmHg+) SBP, dipping through the
   normal range. Plausibly connects to finding 3 below (these are also the least stable
   models) rather than being an equally-trustworthy alternative reading of the data — flagged
   here rather than picking one interpretation to report.
3. **DeepSurv and DeepHit's SHAP rankings are substantially less stable, and agree with LIME
   substantially less, than RSF/GBSA's — despite similar C-index in Stage 7.** Unstable
   predictor counts (rank shifts >2 positions across the 3 stability fits): RSF 1/9 (male) and
   3/10 (female); GBSA 2/9 and 3/10; DeepSurv 7/9 and 6/10; DeepHit 6/9 and 5/10 — the
   majority of predictors for the two deep models. Mean SHAP/LIME top-3 agreement follows the
   same pattern: RSF/GBSA 1.38-2.23 out of 3; DeepSurv/DeepHit 0.53-1.68 out of 3. This is a
   real explainability-vs-performance tradeoff: the deep models match the tree/linear models
   on discrimination (§8) but their *explanations* are considerably less trustworthy — exactly
   the kind of finding §9.4 was designed to surface, not something to average away.

**Open questions:** none blocking.

**Test/sanity check (§10.4):** ran `python src/08_explainability.py` end-to-end — exits 0.
SHAP/LIME agreement counts and stability rank-ranges are non-negative for every model/sex
(programmatic assertion, not just visual check); row counts in both output tables match
9 or 10 predictors × 4 models exactly; dependence-plot age relationships are visually
monotonic increasing for all four models, both sexes — consistent with every prior stage's
finding that age is a strong, uncontroversial risk driver.

---

## 11. Results dashboard (GitHub Pages)

**Date:** 2026-08-21 · **Branch:** `session/2026-08-21-results-dashboard`

**What was built:** `src/09_build_dashboard.py` (an added stage beyond §11's original 8 —
see CLAUDE.md §15.5) — reads Stage 3/7/8's actual output tables (never hand-transcribed) and
renders a single self-contained static page to `docs/index.html` plus `docs/figures/`, for
GitHub Pages (static files only, `main` branch `/docs` folder — a one-time manual toggle in
repo Settings is the only step this session can't do itself, no repo admin access). Sections:
the six-model × two-sex results table with bootstrap CIs, per-sex selected predictors, all
eight Stage 4/7/8 figures, a SHAP-stability/SHAP-LIME-agreement summary table, and a written
findings section. Design follows the project's dataviz skill (system-sans type, tabular-nums
in the results table, `prefers-color-scheme` light/dark, no external requests/CDNs).

**A real error was found and corrected while building this, not just a UI bug (§0.7):**
cross-checking the dashboard's computed "95% CI excludes zero" significance markers against
the raw `stage7_evaluation_summary.csv` surfaced that the Stage 7 journal entry's claim —
"all differences [vs CoxPH] are small and every CI comfortably contains zero... no fitted
model shows a statistically distinguishable improvement... in either metric, for either
sex" — was inaccurate. The "no improvement" part was right; the implication that nothing was
significant was not. Corrected in place in Stage 7's journal entry (§10 above) with the full
per-model, per-sex breakdown: no fitted ML/DL model ever improves on CoxPH significantly, but
several regress significantly (male: GBSA and DeepHit calibrate worse; female: RSF, GBSA, and
DeepHit all discriminate worse). DeepSurv is the only one of the four that never differs
significantly from CoxPH on anything, in either sex. This is now also written up in the
dashboard's findings section. Left as a visible correction rather than quietly editing the
original claim away, since the process of catching it is itself relevant to how the pipeline
was audited.

**Two build issues caught and fixed before shipping, neither a data problem:**
1. An unused `--accent-2` CSS custom property from an earlier design pass (had considered,
   then dropped, colour-coding the two sex sections) was left declared but unreferenced —
   removed rather than left as dead code.
2. Figures were initially marked `loading="lazy"`; because the page is short (8 images,
   ~1MB total — no real lazy-load benefit) this caused `document.body.scrollHeight` to
   under-report before the images entered the viewport, which looked like a rendering bug
   (large blank gap) when visually checking the page. Removed lazy loading — simpler, and
   avoids the same false alarm for anyone else scrolling quickly.

**Verification:** ran the build script, then visually checked the rendered page end-to-end
in a local browser (Chrome, via a temporary `python -m http.server` — the browser extension
can't load `file://` URLs directly) — header, ethics notice, results table with correct
significance markers, feature lists, all eight figures, stability table, findings, and
footer all render correctly in dark mode (system default on this machine). Light mode was
not separately screenshotted — its CSS values are the skill's own pre-validated palette
tokens, not custom colours, so it wasn't re-validated with the palette script; worth a quick
visual pass before treating it as fully confirmed, if that matters for submission.

**Open questions:** whether to also spot-check light mode visually, and whether the user
wants the GitHub Pages toggle flipped now or held until they're ready to share the link.

**Test/sanity check (§10.4):** ran `python src/09_build_dashboard.py` end-to-end — exits 0,
`docs/index.html` + 8 figures written; HTML tag balance checked programmatically (div/
section/table open vs close counts match); all six-model × two-sex table cells populated
with no missing values.

---

*End of journal. Stage 8 was the last modelling pipeline stage; the results dashboard (§11
above) is built and pending the GitHub Pages toggle.*
