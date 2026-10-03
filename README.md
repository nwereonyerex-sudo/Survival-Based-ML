# Survival-Based Machine Learning for 10-Year Heart Attack and Stroke Risk

MSc dissertation project. It benchmarks survival-analysis machine learning models for predicting
a patient's **10-year risk of heart attack or stroke**. The models are compared against a classical
Cox proportional hazards model and a rebuilt QRISK3-style clinical risk score. Both discrimination
and calibration are evaluated, and each model's predictions are explained with SHAP and LIME.

> **Ethics statement:** this project uses only the fully synthetic, CC0-licensed dataset released
> by Burns, Richardson and Driessens (2024). No real, identifiable or non-synthetic patient data was
> used at any stage.

---

## The problem

Cardiovascular disease is a leading cause of death, and in UK primary care the decision to offer
preventive treatment (statins, for example) depends on a patient's estimated 10-year risk. That
estimate usually comes from QRISK3 (Hippisley-Cox, Coupland and Brindle, 2017), a fixed regression
formula.

Survival machine learning models such as random survival forests, gradient boosting and deep
neural survival networks claim to capture non-linear effects and interactions that a fixed
formula cannot. Whether they actually beat a well-specified classical baseline on realistic
primary-care data is still an open question (Liu et al., 2025, 2026). The answer also has to cover
more than ranking patients (discrimination). A risk tool also has to get absolute risk right
(calibration) and give explanations clinicians can trust.

This project asks:

1. **Do survival ML and deep learning models predict 10-year heart attack or stroke risk better
   than a Cox model and a QRISK3-style score**, on both discrimination and calibration?
2. **Are their explanations trustworthy?** In particular, are SHAP and LIME explanations stable
   across resampled training data, and do the two methods agree with each other?

The data is deliberately messy and realistic. It has informative censoring, values missing
completely at random (MCAR) and missing at random (MAR), noise variables that carry no signal, and
measurement error. Handling these explicitly is part of the problem.

## Dataset

[`data/cvd_synthetic_dataset_v0.2.csv`](data/cvd_synthetic_dataset_v0.2.csv): 100,000 synthetic
primary-care patients with 12 clinical predictors (age, sex, BMI, smoking, systolic blood
pressure, treated hypertension, family history of CVD, atrial fibrillation, chronic kidney
disease, rheumatoid arthritis, diabetes, COPD, FEV1), a time-to-event or censoring column (in
years, censored at 10) and a heart attack or stroke event indicator.

Burns, D., Richardson, K. and Driessens, C. (2024) 'A synthetic dataset for the exploration of
survival and classification models: prediction of heart attack or stroke within a 10-year
follow-up period', *NIHR Open Research*, 4:67. doi:
[10.3310/nihropenres.13651.1](https://doi.org/10.3310/nihropenres.13651.1). Zenodo: doi:
[10.5281/zenodo.12567416](https://doi.org/10.5281/zenodo.12567416).

## Approach

Each model is trained, tuned and evaluated as **two independent pipelines, one for men and one for
women**, mirroring QRISK3's sex-specific equations. No pooled model is used.

| Stage | Script | What it does |
|---|---|---|
| 1 | `src/01_data_ingestion.py` | Load and validate the schema; profile missingness, censoring and noise |
| 2 | `src/02_preprocessing.py` | Median imputation (FEV1 imputed within COPD strata); stratified 70/15/15 train/validation/test split per sex |
| 3 | `src/03_feature_selection.py` | Multicollinearity screen, then Cox LASSO (Tibshirani, 1996), then a cross-check against the QRISK3 predictor set |
| 4 | `src/04_baselines.py` | Kaplan–Meier, CoxPH (Cox, 1972), QRISK3-style score |
| 5 | `src/05_survival_ml_models.py` | Random survival forest (Ishwaran et al., 2008), gradient-boosted survival analysis (Friedman, 2001) |
| 6 | `src/06_deep_survival_models.py` | DeepSurv (Katzman et al., 2018), DeepHit (Lee et al., 2018) |
| 7 | `src/07_evaluation.py` | Harrell's C-index, Uno's IPCW C-statistic, integrated Brier score, calibration plots; 1,000-resample bootstrap 95% CIs and paired effect sizes against both baselines |
| 8 | `src/08_explainability.py` | SHAP (Lundberg and Lee, 2017) and LIME (Ribeiro, Singh and Guestrin, 2016), with SHAP/LIME agreement and stability testing across resampled training folds |
| 9 | `src/09_build_dashboard.py` | Builds the static results dashboard in `docs/` |

`src/10_full_pipeline.ipynb` runs the whole pipeline in one notebook with its figures inline.

Libraries: scikit-survival (Pölsterl, 2020) for CoxPH, RSF and GBSA; pycox (Kvamme, Borgan and
Scheel, 2019) on PyTorch for DeepSurv and DeepHit. Every settable seed is `random_state = 42`.

## Key findings

Test-set results, with bootstrap 95% CIs:

| Model | C-index (male) | Brier (male) | C-index (female) | Brier (female) |
|---|---|---|---|---|
| CoxPH | 0.810 [0.794, 0.824] | 0.039 | 0.819 [0.801, 0.838] | 0.026 |
| QRISK3-style | 0.820 [0.806, 0.834] | 0.059* | 0.823 [0.806, 0.841] | 0.041* |
| RSF | 0.807 [0.792, 0.822] | 0.039 | 0.813 [0.794, 0.831] | 0.027 |
| GBSA | 0.809 [0.794, 0.823] | 0.040 | 0.814 [0.795, 0.833] | 0.027 |
| DeepSurv | 0.809 [0.793, 0.824] | 0.039 | 0.815 [0.797, 0.833] | 0.027 |
| DeepHit | 0.808 [0.793, 0.823] | 0.040 | 0.811 [0.794, 0.829] | 0.027 |

\* QRISK3-style has no survival curve, so it gets a single-timepoint Brier score at 9.99 years.
The other models get an integrated Brier score. Lower Brier is better.

- **No ML or deep learning model significantly outperformed the Cox baseline** on any metric, in
  either sex. Several were significantly *worse* by a small margin. For women, RSF, GBSA and
  DeepHit discriminated worse. For men, GBSA and DeepHit calibrated worse. DeepSurv was the only
  model that never differed significantly from CoxPH.
- **The QRISK3-style score ranked patients best but was the worst calibrated** in both sexes.
  Its Brier score CI does not overlap any other model's.
- **Explanations from the deep models were much less stable.** Most predictors' SHAP ranks
  shifted across resampled training folds for DeepSurv and DeepHit, compared with only 1–3
  predictors for RSF and GBSA. The deep models also agreed with LIME less often. Their accuracy
  matched the other models, but their explanations were less trustworthy.
- The tree ensembles attributed almost all of their predictions to age. CoxPH and the deep models
  spread importance across comorbidities.

The full analysis, including every deviation from the original plan and why it was made, is in
[`journal/agent_journal.md`](journal/agent_journal.md). Figures are in
[`results/figures/`](results/figures/) and tables are in [`results/tables/`](results/tables/).

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # exact pinned versions; developed on Python 3.14

python src/01_data_ingestion.py
python src/02_preprocessing.py
# ...run each stage in order up to 09_build_dashboard.py
```

Stages 5 and 8 are the slowest: GBSA takes about 6 minutes per fit, and the Stage 8 explainability
run takes about 45 minutes in total.

## Repository layout

```
data/       synthetic dataset, metadata, processed train/val/test splits
src/        one script per pipeline stage, plus the full-pipeline notebook
results/    tables/ (CSV outputs) and figures/ (KM, calibration, SHAP plots)
docs/       static results dashboard (GitHub Pages)
journal/    stage-by-stage implementation notes and findings
```

## Key references

- Burns, D., Richardson, K. and Driessens, C. (2024). *NIHR Open Research*, 4:67.
- Hippisley-Cox, J., Coupland, C. and Brindle, P. (2017). QRISK3. *BMJ*, 357, j2099.
- Liu, T., Krentz, A., Lu, L., Wang, Y. and Curcin, V. (2026). Benchmarking survival machine
  learning models for 10-year CVD risk prediction. *Digital Health*, 12.
- Pölsterl, S. (2020). scikit-survival. *JMLR*, 21(212).
- Kvamme, H., Borgan, Ø. and Scheel, I. (2019). Time-to-event prediction with neural networks and
  Cox regression. *JMLR*, 20(129).
- Lundberg, S.M. and Lee, S.-I. (2017). SHAP. *NeurIPS*, 30.
- Ribeiro, M.T., Singh, S. and Guestrin, C. (2016). LIME. *KDD '16*, pp. 1135–1144.

The full reference list is kept with the dissertation.
