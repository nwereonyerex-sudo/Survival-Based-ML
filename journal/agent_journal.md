# Agent journal

Stage-by-stage implementation notes, per CLAUDE.md §10.5. Each entry: what was built, any
deviation from spec and why, open questions, and anything unexpected noticed in the data or
results.

## 2026-08-20 — Project setup

- Adopted the approved spec as `CLAUDE.md` at the repo root (copied from
  `~/Downloads/CLAUDE.md — Coding Session Specification.md`).
- Created the §11 repo layout (`data/`, `src/`, `results/tables/`, `results/figures/`,
  `journal/`).
- Copied in `data/cvd_synthetic_dataset_v0.2.csv` and
  `data/cvd_synthetic_dataset_v0.2_metadata.xlsx` (both from `~/Downloads`) — both files
  referenced in §3 are present, no open item here.
- No modeling code written yet. Stage 1 (data ingestion) will be the first session branch.

## 2026-08-20 — Stage 1: data ingestion

**Branch:** `session/2026-08-20-stage1-data-ingestion`

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

## 2026-08-20 — Stage 2: preprocessing (imputation, encoding, sex-specific split)

**Branch:** `session/2026-08-20-stage2-preprocessing`

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
