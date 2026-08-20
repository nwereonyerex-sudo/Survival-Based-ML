"""Stage 4 — baselines (CLAUDE.md §7): Kaplan-Meier (descriptive), CoxPH, and the simplified
QRISK3-style score rebuild. Per sex. Fits/builds the baseline models and persists per-patient
risk predictions to `results/tables/` for Stage 5/§11 (07_evaluation.py) to score uniformly
alongside RSF/GBSA/DeepSurv/DeepHit — C-index, Uno's C, Brier score, calibration, and
bootstrap 95% CIs (§8, §12) are computed there, not here, so every model is scored the same
way. KM curves are saved to `results/figures/`.

**QRISK3 coefficient provenance — flagged to and resolved with the user on 2026-08-20 (§14):**
CLAUDE.md §7 asks for "the exact simplified variant Burns, Richardson and Driessens (2024)
used to simulate the outcome." Their paper (checked directly, plus the Zenodo deposit — no
supplementary code or coefficients exist anywhere) only describes their approach
qualitatively; the actual per-predictor coefficients they used are not published and are
unrecoverable. Per the user's instruction, this rebuild instead uses the real, original
QRISK3-2017 coefficients (Hippisley-Cox, Coupland and Brindle, 2017), sourced from ClinRisk
Ltd.'s own reference implementation (released under LGPL specifically "to enable others to
implement the algorithm faithfully," https://qrisk.org, mirrored at
https://github.com/sisuhealthgroup/qrisk3/blob/master/src/lib/original/qrisk3.c) and
restricted to this dataset's available predictors per §7's rules. This Python port was
validated against ClinRisk's own published test cases (age x sex x cholesterol/HDL grid,
white ethnicity, no comorbidities) before being restricted — all 8 cases matched to within
0.05 percentage points. Corroborating evidence this is a reasonable proxy for what Burns et
al. actually did: their paper's stated 10-year baseline survival values (0.977 male / 0.989
female) match the real QRISK3 baseline survivor constants (0.977268... / 0.988876...) to 3
decimal places — they evidently anchored their simulation on the same baseline hazard.
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"

EVENT_COL = "heart_attack_or_stroke_occurred"
TIME_COL = "time_to_event_or_censoring"
SPLITS = ["train", "val", "test"]
HORIZON_YEARS = 10.0


# --- QRISK3-style score (§7) --------------------------------------------------------------
# Coefficients are the real QRISK3-2017 values (Hippisley-Cox, Coupland and Brindle, 2017),
# restricted to this dataset's predictors. Ethnicity and Townsend deprivation terms (absent
# from this dataset, §7) are omitted entirely, not defaulted to a "neutral" value, since a
# defaulted raw value would still contribute a non-zero centred term. rati (cholesterol/HDL)
# and sbps5 (SBP SD) are retained but fixed at 3 and 10 respectively, per §7.
QRISK3_SURVIVOR_10Y = {"female": 0.988876402378082, "male": 0.977268040180206}
QRISK3_ISMOKE_LIGHT = {"female": 0.13386833786546262, "male": 0.19128222863388983}
FIXED_CHOLESTEROL_HDL_RATIO = 3.0
FIXED_SBP_SD = 10.0

# Known, accepted behaviour (user sign-off 2026-08-20, §0.7 — not silently smoothed over):
# QRISK3 is only officially validated for ages 25-84. 11.1% of this dataset (11,110 patients)
# is under 25, and the fractional-polynomial age/BMI terms below are not calibrated for that
# range. For the rare combination of very young age with the dataset's implausible synthetic
# BMI floor (values down to 6, no lower bound), those terms compound into a very large linear
# predictor, and the survival exponentiation saturates the score at exactly 100.0 — 13
# patients total (0.013% of the dataset) are affected. Left unclipped and unexcluded
# deliberately: no clip threshold is specified anywhere in the source material, and excluding
# these patients would make the QRISK3-style baseline cover a different population than the
# other five models. See journal/agent_journal.md for the full account.


def qrisk3_style_score(row: pd.Series, sex_label: str) -> float:
    age_scaled = row["age"] / 10.0
    bmi_scaled = row["body_mass_index"] / 10.0
    sbp = row["systolic_blood_pressure"]
    smoke_light = 1 if row["smoker"] == 1 else 0  # §7: smoker ≈ light-smoker status
    diabetes_type2 = 1 if row["diabetes"] == 1 else 0  # §7: diabetes ≈ type 2

    if sex_label == "female":
        age_1, age_2 = age_scaled ** -2, age_scaled
        age_1 -= 0.053274843841791
        age_2 -= 4.332503318786621
        bmi_1, bmi_2 = bmi_scaled ** -2, (bmi_scaled ** -2) * math.log(bmi_scaled)
        bmi_1 -= 0.154946178197861
        bmi_2 -= 0.144462317228317
        rati = FIXED_CHOLESTEROL_HDL_RATIO - 3.476326465606690
        sbp_c = sbp - 123.130012512207030
        sbps5 = FIXED_SBP_SD - 9.002537727355957

        a = QRISK3_ISMOKE_LIGHT["female"] * smoke_light
        a += age_1 * -8.1388109247726188 + age_2 * 0.79733376689699098
        a += bmi_1 * 0.29236092275460052 + bmi_2 * -4.1513300213837665
        a += rati * 0.15338035820802554 + sbp_c * 0.013131488407103424
        a += sbps5 * 0.0078894541014586095
        a += row["atrial_fibrillation"] * 1.5923354969269663
        a += row["rheumatoid_arthritis"] * 0.21364803435181942
        a += row["chronic_kidney_disease"] * 0.65194569493845833
        a += row["hypertension_treated"] * 0.50931593683423004
        a += diabetes_type2 * 1.0688773244615468
        a += row["family_history_of_cardiovascular_disease"] * 0.45445319020896213
        a += age_1 * smoke_light * -4.7057161785851891
        a += age_1 * row["atrial_fibrillation"] * 19.938034889546561
        a += age_1 * row["chronic_kidney_disease"] * -3.5874047731694114
        a += age_1 * row["hypertension_treated"] * 11.872809733921812
        a += age_1 * diabetes_type2 * 6.8652342000009599
        a += age_1 * bmi_1 * 23.802623412141742 + age_1 * bmi_2 * -71.184947692087007
        a += age_1 * row["family_history_of_cardiovascular_disease"] * 0.99467807940435127
        a += age_1 * sbp_c * 0.034131842338615485
        a += age_2 * smoke_light * -0.075589244643193026
        a += age_2 * row["atrial_fibrillation"] * -0.076182651011162505
        a += age_2 * row["chronic_kidney_disease"] * -0.22688873086442507
        a += age_2 * row["hypertension_treated"] * 0.00096857823588174436
        a += age_2 * diabetes_type2 * -0.097112252590695489
        a += age_2 * bmi_1 * 0.52369958933664429 + age_2 * bmi_2 * 0.045744190122323759
        a += age_2 * row["family_history_of_cardiovascular_disease"] * -0.076885051698423038
        a += age_2 * sbp_c * -0.0015082501423272358
        survivor = QRISK3_SURVIVOR_10Y["female"]
    else:
        age_1, age_2 = age_scaled ** -1, age_scaled ** 3
        age_1 -= 0.234766781330109
        age_2 -= 77.284080505371094
        bmi_1, bmi_2 = bmi_scaled ** -2, (bmi_scaled ** -2) * math.log(bmi_scaled)
        bmi_1 -= 0.149176135659218
        bmi_2 -= 0.141913309693336
        rati = FIXED_CHOLESTEROL_HDL_RATIO - 4.300998687744141
        sbp_c = sbp - 128.571578979492190
        sbps5 = FIXED_SBP_SD - 8.756621360778809

        a = QRISK3_ISMOKE_LIGHT["male"] * smoke_light
        a += age_1 * -17.839781666005575 + age_2 * 0.0022964880605765492
        a += bmi_1 * 2.4562776660536358 + bmi_2 * -8.3011122314711354
        a += rati * 0.17340196856327111 + sbp_c * 0.012910126542553305
        a += sbps5 * 0.010251914291290456
        a += row["atrial_fibrillation"] * 0.88209236928054657
        a += row["rheumatoid_arthritis"] * 0.20970658013956567
        a += row["chronic_kidney_disease"] * 0.71853261288274384
        a += row["hypertension_treated"] * 0.51659871082695474
        a += diabetes_type2 * 0.85942071430932221
        a += row["family_history_of_cardiovascular_disease"] * 0.54055469009390156
        a += age_1 * smoke_light * -0.21011133933516346
        a += age_1 * row["atrial_fibrillation"] * 3.4896675530623207
        a += age_1 * row["chronic_kidney_disease"] * -0.50656716327223694
        a += age_1 * row["hypertension_treated"] * 6.5114581098532671
        a += age_1 * diabetes_type2 * 3.6461817406221311
        a += age_1 * bmi_1 * 31.004952956033886 + age_1 * bmi_2 * -111.29157184391643
        a += age_1 * row["family_history_of_cardiovascular_disease"] * 2.7808628508531887
        a += age_1 * sbp_c * 0.018858524469865853
        a += age_2 * smoke_light * -0.00049854870275326121
        a += age_2 * row["atrial_fibrillation"] * -0.00034995608340636049
        a += age_2 * row["chronic_kidney_disease"] * -0.0018325930166498813
        a += age_2 * row["hypertension_treated"] * 0.00063838053104165013
        a += age_2 * diabetes_type2 * -0.00024695695588868315
        a += age_2 * bmi_1 * 0.0050380102356322029 + age_2 * bmi_2 * -0.013074483002524319
        a += age_2 * row["family_history_of_cardiovascular_disease"] * -0.00024791809907396037
        a += age_2 * sbp_c * -0.0000127187419158845700
        survivor = QRISK3_SURVIVOR_10Y["male"]

    return 100.0 * (1 - survivor ** math.exp(a))


# --- Loading ---------------------------------------------------------------------------------

def load_split(sex_label: str, part: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{sex_label}_{part}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run src/02_preprocessing.py first.")
    return pd.read_csv(path)


def load_selected_features(sex_label: str) -> list:
    path = RESULTS_TABLES / f"stage3_feature_selection_{sex_label}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run src/03_feature_selection.py first.")
    summary = pd.read_csv(path)
    return sorted(summary.loc[summary["final_selected_feature_set"], "predictor"].tolist())


# --- Kaplan-Meier (descriptive only, §7) ------------------------------------------------------

def plot_kaplan_meier(sex_label: str, full_cohort: pd.DataFrame) -> None:
    time, survival_prob, conf_int = kaplan_meier_estimator(
        full_cohort[EVENT_COL].astype(bool), full_cohort[TIME_COL], conf_type="log-log"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(time, survival_prob, where="post", label=f"{sex_label} (n={len(full_cohort)})")
    ax.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability (event-free)")
    ax.set_title(f"Kaplan-Meier — {sex_label} cohort (descriptive only, CLAUDE.md §7)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    out_path = RESULTS_FIGURES / f"km_curve_{sex_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [{sex_label}] KM curve saved to {out_path} "
          f"(survival at 10y = {survival_prob[time <= 10][-1]:.4f})")


# --- CoxPH (§7) --------------------------------------------------------------------------------

def fit_coxph(train_df: pd.DataFrame, features: list, sex_label: str):
    X_train = train_df[features].astype(float)
    y_train = Surv.from_dataframe(EVENT_COL, TIME_COL, train_df)
    model = CoxPHSurvivalAnalysis()
    model.fit(X_train, y_train)
    print(f"  [{sex_label}] CoxPH fitted on {len(features)} Stage-3-selected features: "
          f"{features}")
    return model


def survival_at_horizon(model, X: pd.DataFrame, horizon: float = HORIZON_YEARS) -> np.ndarray:
    """§8: Brier score and calibration need a predicted P(event by horizon) = 1 - S(horizon|X)
    per patient, not just a relative risk score — the risk score alone is only sufficient for
    the C-index. Evaluates each patient's individual survival step function at the horizon."""
    step_functions = model.predict_survival_function(X)
    return np.array([fn(horizon) for fn in step_functions])


# --- Orchestration -----------------------------------------------------------------------------

def run_sex_pipeline(sex_label: str) -> None:
    print(f"\n--- {sex_label.capitalize()} cohort ---")
    splits = {part: load_split(sex_label, part) for part in SPLITS}
    features = load_selected_features(sex_label)

    full_cohort = pd.concat(splits.values(), ignore_index=True)
    plot_kaplan_meier(sex_label, full_cohort)

    coxph_model = fit_coxph(splits["train"], features, sex_label)

    rows = []
    for part in SPLITS:
        df = splits[part]
        X_part = df[features].astype(float)
        coxph_scores = coxph_model.predict(X_part)
        coxph_survival_10y = survival_at_horizon(coxph_model, X_part)
        qrisk3_scores = df.apply(lambda row: qrisk3_style_score(row, sex_label), axis=1)
        rows.append(pd.DataFrame({
            "patient_id": df["patient_id"],
            "split": part,
            TIME_COL: df[TIME_COL],
            EVENT_COL: df[EVENT_COL],
            "coxph_risk_score": coxph_scores,
            "coxph_survival_at_10y": coxph_survival_10y,
            "qrisk3_style_score": qrisk3_scores,
        }))
    predictions = pd.concat(rows, ignore_index=True)

    assert predictions["coxph_risk_score"].notna().all(), f"[{sex_label}] NaN CoxPH scores"
    assert predictions["coxph_survival_at_10y"].between(0, 1).all(), (
        f"[{sex_label}] coxph_survival_at_10y outside [0, 1]"
    )
    assert predictions["qrisk3_style_score"].between(0, 100).all(), (
        f"[{sex_label}] QRISK3-style score outside [0, 100]"
    )
    print(f"  [{sex_label}] QRISK3-style score range: "
          f"[{predictions['qrisk3_style_score'].min():.2f}, "
          f"{predictions['qrisk3_style_score'].max():.2f}], "
          f"mean = {predictions['qrisk3_style_score'].mean():.2f}")

    out_path = RESULTS_TABLES / f"stage4_baseline_predictions_{sex_label}.csv"
    predictions.to_csv(out_path, index=False)
    print(f"  [{sex_label}] wrote {out_path}")


def main() -> None:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    for sex_label in ["male", "female"]:
        try:
            run_sex_pipeline(sex_label)
        except FileNotFoundError as e:
            print(f"[STOP] {e}", file=sys.stderr)
            sys.exit(1)
    print("\n[OK] Stage 4 complete: KM curves, CoxPH, and QRISK3-style scores built for "
          "both sexes.")


if __name__ == "__main__":
    main()
