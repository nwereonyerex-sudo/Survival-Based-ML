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
