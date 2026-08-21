# CLAUDE.md — Survival-Based ML for 10-Year Heart Attack/Stroke Risk

This file is the binding specification for this repository. Read it in full before writing or
modifying any code. It is derived directly from the Planner agent's approved specification
(`planner_specification.md`) and Chapter 3 — Methodology of the dissertation, and it supersedes
any general default you would otherwise apply. Do not deviate from anything fixed here without
first raising the conflict explicitly to the user (the student) — do not silently "improve" or
substitute a different design choice.

## 0. Non-negotiable constraints (read first)

1. **Data:** The only permitted data source is `data/cvd_synthetic_dataset_v0.2.csv` (Burns,
   Richardson and Driessens, 2024) — a 100,000-row, CC0-licensed, fully synthetic dataset. Never
   introduce any real, identifiable, or non-synthetic patient data at any stage. If you are ever
   tempted to "supplement" or "validate against" a real-world dataset, stop and ask the user
   first — this would violate the project's ethics approval scope.
2. **Referencing:** Every methodological claim in code comments, docstrings, or generated
   markdown must use Harvard-style citations to sources already verified in §8 below. Never
   invent, guess, or hallucinate a citation, DOI, author name, or year. If you need a citation you
   don't have, say so explicitly and ask rather than fabricating one.
3. **Reproducibility:** `random_state = 42` everywhere a seed is settable (train/val/test split,
   LASSO CV folds, RSF/GBSA bootstrap/subsampling, DeepSurv/DeepHit training). Pin every library
   version used into `requirements.txt` as you install it — do not leave this until the end.
4. **Sex-stratified pipelines:** Every baseline and model in this project is trained, tuned, and
   evaluated as **two fully independent pipelines** — one for the male cohort, one for the female
   cohort. Never train a single pooled model with sex as a covariate. This roughly doubles the
   work for every stage; do not "simplify" by pooling.
5. **Calibration is mandatory, not optional:** Never report a discrimination metric (C-index)
   without also producing the corresponding calibration plot and Brier score for that model.
6. **No results before the pipeline is built:** Metrics, baselines, and evaluation criteria are
   already fixed below, before any model has been trained. Do not select or change a metric later
   because it happens to favour a particular model.
7. **Report anomalies honestly — do not smooth them over.** If a model beats a sex-specific
   baseline only marginally, if SHAP/LIME disagree, if a feature behaves unexpectedly, if
   calibration is poor despite good discrimination, or if performance is unstable across
   resampling folds — record this explicitly in the stage's implementation note (§10) rather than
   omitting it or averaging it away. These honest failure findings are worth more to the thesis
   than a clean-looking result.

## 1. Objective

Develop and benchmark survival-based machine learning models predicting 10-year risk of heart
attack or stroke on the Burns et al. (2024) synthetic primary-care dataset, compared against a
Cox proportional hazards (CoxPH) baseline and a rebuilt simplified QRISK3-style clinical score, on
both discrimination and calibration, with SHAP and LIME explainability as a core, non-optional
component (Burns, Richardson and Driessens, 2024).

## 2. Environment

- **Local VS Code**, not Google Colab. You (Claude Code) are the AI coding assistant authorised
  under the MSc AI course's Governance of AI Coding Assistants policy.
- Python, with one script or notebook per pipeline stage (see §9 layout).
- Pin exact versions of: `scikit-survival`, `pycox`, `torch`, `scikit-learn`, `shap`, `lime`,
  `pandas`, `numpy`, `matplotlib`, plus the Python interpreter version, into `requirements.txt`.
- `scikit-survival` (Pölsterl, 2020) implements CoxPH, RSF, and GBSA. `pycox` (Kvamme, Borgan and
  Scheel, 2019) implements DeepSurv and DeepHit on a PyTorch backend.

## 3. Dataset

- **File:** `data/cvd_synthetic_dataset_v0.2.csv` (5.2 MB, 100,000 rows). Metadata/data dictionary
  at `data/cvd_synthetic_dataset_v0.2_metadata.xlsx`.
- **Source:** Burns, D., Richardson, K. and Driessens, C. (2024) 'A synthetic dataset for the
  exploration of survival and classification models: prediction of heart attack or stroke within
  a 10-year follow-up period', *NIHR Open Research*, 4:67. doi: 10.3310/nihropenres.13651.1.
  Hosted on Zenodo, doi: 10.5281/zenodo.12567416.
- **Columns (16):** `patient_id` (identifier, not a feature), `gender`, `age`,
  `body_mass_index`, `smoker`, `systolic_blood_pressure`, `hypertension_treated`,
  `family_history_of_cardiovascular_disease`, `atrial_fibrillation`, `chronic_kidney_disease`,
  `rheumatoid_arthritis`, `diabetes`, `chronic_obstructive_pulmonary_disorder`,
  `forced_expiratory_volume_1`, `time_to_event_or_censoring` (years, right-censored at 10),
  `heart_attack_or_stroke_occurred` (binary event indicator).
- **Verify on load:** row count = 100,000, column count = 16, column names match exactly the list
  above. If they don't, stop and report the discrepancy — do not silently proceed.

## 4. Data realism to handle explicitly (do not ignore)

- **Informative censoring:** dropout probability scales with time-to-event (Burns, Richardson and
  Driessens, 2024). Note this explicitly wherever censoring is discussed; use IPCW-weighted
  estimators (Uno's C-statistic, integrated Brier score) precisely because of this.
- **MCAR missingness:** `smoker` and `family_history_of_cardiovascular_disease` flipped 1→0 with
  p = 0.30; `systolic_blood_pressure` dropped with p = 0.1; `body_mass_index` dropped with
  p = 0.3.
  - Imputation: median for the two continuous variables; an explicit "unknown" category (not an
    assumed negative) for the two flipped binary variables.
  - Do **not** use MICE or another multiple-imputation scheme — median/flag imputation is the
    confirmed choice for this project, since the MCAR mechanism gives no reason to expect a more
    elaborate scheme would materially change the imputed values, and it keeps the pipeline
    auditable.
- **MAR missingness:** `forced_expiratory_volume_1` dropped with p = 0.05 if COPD-positive, p =
  0.75 if COPD-negative.
  - Impute conditionally within COPD strata (median FEV1 for COPD-positive patients; median FEV1
    for COPD-negative patients) — never a single unconditional median.
- **Noise/irrelevant variables:** the dataset's release paper states it deliberately includes
  variables "that did not significantly contribute to the analyses" and incorporates
  "interactions, variable irrelevance, and noise" (Burns, Richardson and Driessens, 2024, p. 3).
  Do not assume every one of the 14 predictors is informative — this is what the feature-selection
  stage (§5) exists to test.

## 5. Feature selection (run separately per sex, training fold only)

Run independently within each sex-specific training fold — never let validation/test data
influence which features are selected.

1. **Multicollinearity screening:** Pearson (continuous–continuous) and point-biserial
   (continuous–binary) correlations across the 14 predictors. Flag any pair with |r| > 0.80 as
   redundant; drop the member with weaker QRISK3 clinical basis. Run this before LASSO.
2. **LASSO-based selection:** L1-penalised Cox partial-likelihood LASSO (Tibshirani, 1996) fitted
   per sex-specific training fold. Tune the penalty strength on the validation set, not the
   training set. Precedent: Choi et al. (2025) used LASSO ahead of XGBoost for CVD risk with no
   material performance loss.
3. **QRISK3 cross-reference:** cross-check the surviving feature set against the QRISK3 predictor
   set (Hippisley-Cox, Coupland and Brindle, 2017). Predictors that pass both checks are
   high-confidence; any that disagree (LASSO keeps but QRISK3 doesn't, or vice versa) must be
   flagged explicitly as a disagreement in the results output — not silently reconciled.
4. **Consistent feature set:** the final surviving predictor set is used identically across
   CoxPH, RSF, GBSA, DeepSurv, and DeepHit for that sex, so performance differences are
   attributable to the model, not to differing inputs. The QRISK3-style score is the sole
   exception — its predictor set is fixed by the original clinical formula, not by this
   pipeline's feature selection.

## 6. Train/validation/test split

- 70/15/15 split, matching Liu et al. (2026)'s benchmarking design.
- Performed **independently within each sex cohort** (never split pooled and infer sex subsets
  afterwards).
- Stratified on the event indicator so event rate is preserved across partitions.
- `random_state = 42` for the split itself.
- Training set → model fitting + feature selection. Validation set → hyperparameter tuning + LASSO
  penalty tuning. Test set → held out exclusively for final Chapter 5 evaluation; nothing from it
  may leak into feature or model selection.

## 7. Models (each trained as two independent pipelines: male, female)

| Model | Library | Key property |
|---|---|---|
| Kaplan–Meier | lifelines or scikit-survival | Descriptive only — not scored against acceptance criteria |
| CoxPH | scikit-survival | Linear, proportional-hazards baseline (Cox, 1972) |
| Simplified QRISK3-style score | custom rebuild (see below) | Fixed clinical formula, not fitted |
| Random survival forest (RSF) | scikit-survival | Ishwaran et al. (2008); no PH assumption, captures non-linearity/interactions |
| Gradient-boosting survival analysis (GBSA) | scikit-survival | Friedman (2001) boosting on Cox partial-likelihood loss |
| DeepSurv | pycox | Katzman et al. (2018); neural network risk function, retains PH assumption |
| DeepHit | pycox | Lee et al. (2018); discards PH assumption, joint event-time PMF + ranking loss; single-risk formulation only (dataset has one composite outcome) |

**QRISK3-style score rebuild** — do not import a generic QRISK3 package. Reconstruct the exact
simplified variant Burns, Richardson and Driessens (2024) used to simulate the outcome:
- Remove predictors absent from this dataset (angina, migraines, SLE, severe mental illness,
  atypical antipsychotics, oral corticosteroids, erectile dysfunction, ethnicity, area
  deprivation).
- Diabetes ≈ type 2 diabetes; smoker ≈ light-smoker status; total cholesterol/HDL ratio fixed at
  3; systolic BP standard deviation fixed at 10 mmHg.
- Evaluate using each patient's *measured* (not latent/true) systolic BP — this reproduces
  realistic clinical noise deliberately, do not "fix" it by using the true value.

## 8. Evaluation (fixed in advance; identical metric set for every model, every sex)

- **Discrimination:** Harrell's C-index (Harrell et al., 1982) as primary; Uno et al.'s (2011)
  IPCW C-statistic as a censoring-robust cross-check. Both with 95% CI via 1,000 bootstrap
  resamples of the test set.
- **Calibration:** integrated Brier score at the 10-year horizon (Graf et al., 1999), IPCW-weighted.
  Plus a calibration plot (observed vs. predicted 10-year event probability, decile-grouped) for
  every model — no exceptions.
- **Reporting:** separately for male and female cohorts. Final artefact is a six-model × two-sex
  table (CoxPH, QRISK3-style, RSF, GBSA, DeepSurv, DeepHit), directly comparable in format to Liu
  et al. (2025, 2026); Kaplan–Meier curves appear alongside as descriptive reference, not as a
  seventh scored model.
- Every reported number needs a comparison against the sex-specific CoxPH/QRISK3 baseline with
  effect size and CI — never a bare metric.

## 9. Explainability: SHAP and LIME (core requirement, not optional)

- **SHAP** (Lundberg and Lee, 2017) on all four ML/DL models (RSF, GBSA, DeepSurv, DeepHit), per
  sex: (i) global summary plot ranked by mean |SHAP value|, compared against CoxPH coefficients;
  (ii) dependence plots for top continuous predictors (age, systolic BP, BMI).
- **LIME** (Ribeiro, Singh and Guestrin, 2016) on the same four models, matched sample of
  individual test patients per sex, for local explanation comparison against SHAP.
- **Cross-method agreement:** report per-patient agreement between top SHAP and top LIME features
  explicitly — disagreement is a genuine finding to report, not an error to discard.
- **Stability testing (mandatory before any SHAP/LIME output is used as a thesis claim):**
  recompute the SHAP summary ranking across at least 2 independently resampled training folds.
  Only describe a feature's importance as "robust" if its relative rank is consistent across
  folds. Any feature whose importance changes materially between folds must be reported as an
  instability finding, not dropped or averaged away silently.

## 10. Multi-agent workflow (follow the `multi-agent-thesis-workflow` skill)

Work through Planner → Developer → Review → Test at minimum for every stage, invoking
Adversarial-Review, Risk-Assessor, Compliance, and Data as their triggers require (see the skill
for exact scope of each role), with Reflection logging every handoff. Practically, for each stage
below:

1. Before writing code, restate the stage's spec and acceptance criteria back (Planner check).
2. Write the code (Developer).
3. Self-review against the spec: does the code actually do what §3–§9 above require, not a
   simplified or generic version of it? (Review check.)
4. Write and run a small test/sanity-check for the stage's core claim (e.g. survival curves
   monotonically non-increasing; Brier score matches manual calc on a toy sample) (Test check).
5. **Write a short implementation note** for the stage (append to
   `journal/agent_journal.md`) covering: what was built, any deviation from this spec and why,
   open questions, and — critically — **anything noticed in the data or results that was not
   expected going in** (e.g. an unexpectedly high missingness rate after filtering, a feature
   dropped by LASSO that has strong QRISK3 clinical basis, a proportional-hazards violation, an
   unstable SHAP ranking, a sex-based performance gap). This is what makes the audit trail useful
   for Chapter 5's failure-analysis and Chapter 6's discussion — do not skip it even when a stage
   "goes fine."
6. Only mark a stage complete once its implementation note is written.

## 11. Repository layout to create

```
data/
  cvd_synthetic_dataset_v0.2.csv
  cvd_synthetic_dataset_v0.2_metadata.xlsx
src/
  01_data_ingestion.py          # Stage 1 — load, profile missingness/censoring/noise
  02_preprocessing.py           # Stage 2 — imputation, encoding, sex-specific 70/15/15 split
  03_feature_selection.py       # Stage 2/§5 — multicollinearity, LASSO, QRISK3 cross-ref, per sex
  04_baselines.py               # Stage 3 — KM, CoxPH, QRISK3-style score, per sex
  05_survival_ml_models.py      # Stage 4 — RSF, GBSA, per sex
  06_deep_survival_models.py    # Stage 4 — DeepSurv, DeepHit, per sex
  07_evaluation.py              # Stage 5 — C-index, Uno's C, Brier score, calibration, per sex
  08_explainability.py          # Stage 6 — SHAP + LIME + stability testing, per sex
results/
  tables/                       # six-model x two-sex comparison tables (csv/markdown)
  figures/                      # calibration plots, SHAP summary/dependence, KM curves
journal/
  agent_journal.md              # stage-by-stage implementation notes (see §10.5)
requirements.txt
CLAUDE.md                       # this file
```

## 12. Acceptance criteria checklist (do not consider a stage "done" until true)

- [ ] Data pipeline reproducibly loads all 100,000 records; every missingness mechanism (§4) is
      handled by name, not generically.
- [ ] CoxPH and QRISK3-style score reported per sex with bootstrap 95% CI.
- [ ] RSF/GBSA/DeepSurv/DeepHit report C-index and Brier score per sex, using Liu et al. (2026)'s
      metric definitions; improvement or honest non-improvement vs. sex-specific CoxPH stated with
      effect size and CI.
- [ ] Calibration plot exists for every model, every sex — no exceptions.
- [ ] SHAP summary + dependence plots for every ML/DL model, stability-tested across ≥2 resampled
      folds before being used as a claim.
- [ ] LIME local explanations for the same models/patients, with SHAP/LIME agreement reported.
- [ ] Explicit ethics statement: only the CC0 Burns synthetic dataset was used; no real patient
      data at any stage.
- [ ] `requirements.txt` pinned; `random_state = 42` used everywhere a seed applies.
- [ ] Every methodological claim in code comments/docstrings cites a source from §13 below.

## 13. Verified reference list (Harvard) — do not cite anything not on this list without asking

- Burns, D., Richardson, K. and Driessens, C. (2024) 'A synthetic dataset for the exploration of
  survival and classification models: prediction of heart attack or stroke within a 10-year
  follow-up period', *NIHR Open Research*, 4:67. doi: 10.3310/nihropenres.13651.1.
- Choi, J., Jeon, J., An, H. and Kim, H.C. (2025) 'Explainable SHAP-XGBoost models for identifying
  important social factors associated with the atherosclerotic cardiovascular disease risk score
  using the LASSO feature selection technique', *Epidemiology and Health*, e2025052. doi:
  10.4178/epih.e2025052.
- Cox, D.R. (1972) 'Regression models and life-tables', *Journal of the Royal Statistical Society:
  Series B (Methodological)*, 34(2), pp. 187–220.
- Friedman, J.H. (2001) 'Greedy function approximation: a gradient boosting machine', *Annals of
  Statistics*, 29(5), pp. 1189–1232.
- Graf, E., Schmoor, C., Sauerbrei, W. and Schumacher, M. (1999) 'Assessment and comparison of
  prognostic classification schemes for survival data', *Statistics in Medicine*, 18(17–18), pp.
  2529–2545.
- Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L. and Rosati, R.A. (1982) 'Evaluating the yield
  of medical tests', *JAMA*, 247(18), pp. 2543–2546.
- Hippisley-Cox, J., Coupland, C. and Brindle, P. (2017) 'Development and validation of QRISK3
  risk prediction algorithms to estimate future risk of cardiovascular disease: prospective cohort
  study', *BMJ*, 357, j2099. doi: 10.1136/bmj.j2099.
- Ishwaran, H., Kogalur, U.B., Blackstone, E.H. and Lauer, M.S. (2008) 'Random survival forests',
  *Annals of Applied Statistics*, 2(3), pp. 841–860.
- Katzman, J.L., Shaham, U., Cloninger, A., Bates, J., Jiang, T. and Kluger, Y. (2018) 'DeepSurv: personalized
  treatment recommender system using a Cox proportional hazards deep neural network', *BMC Medical
  Research Methodology*, 18, 24. doi: 10.1186/s12874-018-0482-1.
- Kvamme, H., Borgan, Ø. and Scheel, I. (2019) 'Time-to-event prediction with neural networks and
  Cox regression', *Journal of Machine Learning Research*, 20(129), pp. 1–30.
- Lee, C., Zame, W.R., Yoon, J. and van der Schaar, M. (2018) 'DeepHit: a deep learning approach to
  survival analysis with competing risks', *Proceedings of the AAAI Conference on Artificial
  Intelligence*, 32(1).
- Liu, T., Krentz, A., Lu, L. and Curcin, V. (2025) 'Machine learning based prediction models for
  cardiovascular disease risk using electronic health records data: systematic review and
  meta-analysis', *European Heart Journal - Digital Health*, 6(1), pp. 7–22. doi:
  10.1093/ehjdh/ztae080.
- Liu, T., Krentz, A., Lu, L., Wang, Y. and Curcin, V. (2026) 'Benchmarking survival machine
  learning models for 10-year cardiovascular disease risk prediction using large-scale electronic
  health records', *Digital Health*, 12. doi: 10.1177/20552076251408534.
- Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions',
  *Advances in Neural Information Processing Systems*, 30.
- Pölsterl, S. (2020) 'scikit-survival: a library for time-to-event analysis built on top of
  scikit-learn', *Journal of Machine Learning Research*, 21(212), pp. 1–6.
- Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) '"Why should I trust you?": explaining the
  predictions of any classifier', *Proceedings of the 22nd ACM SIGKDD International Conference on
  Knowledge Discovery and Data Mining*, pp. 1135–1144. doi: 10.1145/2939672.2939778.
- Tibshirani, R. (1996) 'Regression shrinkage and selection via the lasso', *Journal of the Royal
  Statistical Society: Series B (Methodological)*, 58(1), pp. 267–288.
- Tjoa, E. and Guan, C. (2021) 'A survey on explainable artificial intelligence (XAI): toward
  medical XAI', *IEEE Transactions on Neural Networks and Learning Systems*, 32(11), pp.
  4793–4813. doi: 10.1109/TNNLS.2020.3027314.
- Uno, H., Cai, T., Pencina, M.J., D'Agostino, R.B. and Wei, L.J. (2011) 'On the C-statistics for
  evaluating overall adequacy of risk prediction procedures with censored survival data',
  *Statistics in Medicine*, 30(10), pp. 1105–1117.

## 14. What to do if something doesn't match this spec

Stop and flag it rather than improvising — e.g. if a library version conflicts, if the dataset
schema doesn't match §3, if a metric can't be computed as specified, or if a design decision here
seems to contradict something else in the repository. Record the conflict in
`journal/agent_journal.md` and surface it to the user directly; do not silently pick a
workaround.

## 15. Session workflow & Git/GitHub process (added 2026-08-20, per user instruction)

This section governs *how* work happens across sessions, on top of the methodology fixed above.
It applies to every future session in this repository, not just the one that added it.

1. **Ask before acting.** Ask the user for permission before any consequential action —
   installing/upgrading a package, running a long or expensive training job, `git commit`,
   `git push`, creating or merging a branch, or deleting/overwriting anything. Routine read-only
   exploration and drafting code in the working tree doesn't need a fresh ask each time, but
   nothing consequential happens silently.
2. **One branch per work session.** Before writing code in a session, cut a new branch from
   `main` named `session/YYYY-MM-DD-<short-topic>` (e.g. `session/2026-08-21-stage1-data-ingestion`).
   All of that session's work happens on that branch.
3. **Merge gate.** Before merging a session branch into `main`:
   - The stage's script(s) run end-to-end with no errors.
   - The stage's §10 self-review/test checks pass, and the relevant §12 acceptance-criteria
     boxes for that stage are satisfied.
   - The §10.5 implementation note for the stage is written in `journal/agent_journal.md`.
   - Only then, explicitly ask the user for permission to merge. Never merge unverified or
     partially-working code, even temporarily.
4. **GitHub is the remote for this repo.** Push session branches and merge to `main` there once
   approved. Do not force-push `main`.
5. **Deployment (results dashboard) — done 2026-08-21, target: GitHub Pages.** Built by
   `src/09_build_dashboard.py` (an added stage beyond §11's original 8 — no spec section
   covers it, so it's documented here and in the journal rather than folded into §11's list).
   Reads the actual Stage 3/7/8 output tables and renders a static, self-contained page to
   `docs/index.html` (+ `docs/figures/`) — GitHub Pages serves static files only. To go live:
   in the GitHub repo, Settings → Pages → Source: Deploy from branch → `main`, folder `/docs`
   (one-time manual step, not automatable without repo admin access this session doesn't
   have). Re-run the script and re-commit `docs/` whenever `results/` changes.

## 16. Resolved spec clarifications (running log, binding — supersedes the sections referenced)

Each of these was investigated, raised to the user, and resolved with an explicit decision
before code was written, per §14. Full reasoning and evidence for each is in
`journal/agent_journal.md` under the stage named; this section exists so a fresh session
reads the *resolved* interpretation directly here instead of re-deriving or re-litigating it.
Do not revisit these without a new, explicit reason to.

1. **"14 predictors" (§4, §5.1) → 12, resolved at Stage 3.** The dataset only has 12 columns
   usable as survival-model predictors — the "14" figure only works by also counting
   `time_to_event_or_censoring` and `heart_attack_or_stroke_occurred` (the outcome itself) as
   predictors. Treated as a spec wording slip. All feature-selection, baseline, and modelling
   stages use the 12 actual clinical/physiological columns.
2. **`smoker` / `family_history_of_cardiovascular_disease` "unknown category" (§4), resolved
   at Stage 2.** These columns have zero NaNs — the 1→0 flip (p=0.30) is a silent bit-flip
   with no missing-value marker, so there is nothing to impute or flag. Left untouched;
   documented as known non-differential measurement error at p=0.30, not treated as missing
   data.
3. **Imputation order relative to the 70/15/15 split (§4 vs §11), resolved at Stage 2.**
   Imputation is computed per sex on the whole cohort *before* the split (matching §11's
   literal script description), not from the training fold only — an acknowledged, minor
   leak accepted for a low-capacity statistic like a median.
4. **FEV1/COPD multicollinearity tie-break (§5.1), resolved at Stage 3.** Neither variable has
   a QRISK3-basis tie-break. Rule applied: recompute the correlation on complete cases only
   (excluding the imputed majority of FEV1); if it stays ≥0.80, drop the binary COPD flag and
   keep the richer continuous FEV1; if it drops meaningfully below 0.80, drop FEV1 and keep
   COPD instead. In this dataset the complete-case correlation dropped below threshold for
   both sexes (−0.777 male / −0.787 female vs. −0.809 / −0.808 full-sample), so **FEV1 was
   dropped and COPD kept** in both cohorts.
5. **QRISK3-style score coefficients (§7), resolved at Stage 4 — the most significant
   deviation so far.** §7 asks to reconstruct "the exact simplified variant Burns, Richardson
   and Driessens (2024) used to simulate the outcome." Checked directly (paper text +
   Zenodo deposit): their exact coefficients were never published anywhere and are
   permanently unrecoverable. **The QRISK3-style score in this project instead uses the real,
   original QRISK3-2017 coefficients** (Hippisley-Cox, Coupland and Brindle, 2017), sourced
   from ClinRisk Ltd.'s own LGPL-licensed reference implementation
   (github.com/sisuhealthgroup/qrisk3, mirroring qrisk.org/svn.clinrisk.co.uk — released
   explicitly "to enable others to implement the algorithm faithfully"), restricted to this
   dataset's available predictors per §7's removal list, and validated against ClinRisk's own
   published test cases before restriction (8/8 matched within 0.05pp). Corroborating
   evidence this is a reasonable proxy: Burns et al.'s reported 10-year baseline survival
   (0.977 male / 0.989 female) matches the real QRISK3 survivor constants
   (0.977268.../0.988876...) to 3 decimal places.
6. **QRISK3-style score at out-of-validated-range inputs, resolved at Stage 4.** QRISK3 is
   only officially validated for ages 25–84; 11.1% of this dataset is under 25. Combined with
   this dataset's unbounded synthetic BMI (values down to 6), the fractional-polynomial
   age/BMI terms saturate the score at exactly 100.0 for a small number of patients (13 total,
   0.013% of the dataset). Left unclipped and unexcluded deliberately — applied identically
   to every patient, no arbitrary clip threshold invented, so the QRISK3-style baseline covers
   the same population as the other five models.
