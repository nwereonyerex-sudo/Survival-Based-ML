"""Stage 3 — feature selection (CLAUDE.md §5): multicollinearity screening, per-sex LASSO,
QRISK3 cross-reference. Run independently per sex, training-fold only for fitting, validation
fold only for tuning — never test data.

**Predictor-count discrepancy, flagged and resolved with the user on 2026-08-20 (§14):**
CLAUDE.md repeatedly says "14 predictors" (§4, §5.1), but the dataset only has 12 columns
usable as survival-model predictors — the "14" figure only works by also counting
`time_to_event_or_censoring` and `heart_attack_or_stroke_occurred` (the survival outcome
itself) as if they were predictors. Treated here as a spec wording slip; feature selection
runs on the 12 actual clinical/physiological columns.

LASSO: L1-penalised Cox partial-likelihood LASSO (Tibshirani, 1996), fitted via
`sksurv.linear_model.CoxnetSurvivalAnalysis` (Pölsterl, 2020) with `l1_ratio=1.0` for pure L1.
QRISK3 cross-reference set is the simplified rebuild's predictor list from CLAUDE.md §7,
itself derived from Hippisley-Cox, Coupland and Brindle (2017).
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "cvd_synthetic_dataset_v0.2.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

SEX_CODE = {"male": "M", "female": "F"}

EVENT_COL = "heart_attack_or_stroke_occurred"
TIME_COL = "time_to_event_or_censoring"

# The 12 predictors actually usable for survival modelling — patient_id (identifier), gender
# (used only to stratify cohorts, §0.4), and the two outcome columns are excluded. See module
# docstring re: the "14 predictors" spec discrepancy.
CONTINUOUS = ["age", "body_mass_index", "systolic_blood_pressure", "forced_expiratory_volume_1"]
BINARY = [
    "smoker", "hypertension_treated", "family_history_of_cardiovascular_disease",
    "atrial_fibrillation", "chronic_kidney_disease", "rheumatoid_arthritis", "diabetes",
    "chronic_obstructive_pulmonary_disorder",
]
PREDICTORS = CONTINUOUS + BINARY

# Predictors present in this project's simplified QRISK3-style rebuild (CLAUDE.md §7), used
# as the multicollinearity tie-break and the §5.3 cross-reference set. COPD and FEV1 are the
# two columns in this dataset that are *not* part of QRISK3's original predictor set
# (Hippisley-Cox, Coupland and Brindle, 2017) — they are the deliberate noise/extra variables
# named in §4.
QRISK3_BASIS = {
    "age", "smoker", "systolic_blood_pressure", "diabetes",
    "family_history_of_cardiovascular_disease", "chronic_kidney_disease",
    "rheumatoid_arthritis", "hypertension_treated", "atrial_fibrillation", "body_mass_index",
}

CORRELATION_THRESHOLD = 0.80
RANDOM_STATE = 42


def load_split(sex_label: str, part: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{sex_label}_{part}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run src/02_preprocessing.py first.")
    return pd.read_csv(path)


def load_raw_missingness(sex_label: str) -> pd.DataFrame:
    """Raw (pre-Stage-2-imputation) missingness mask per patient, keyed by patient_id. Used
    only by the complete-case tie-break below — needed to tell whether an over-threshold
    correlation involving a continuous, imputed predictor reflects real clinical redundancy
    or is partly inflated by Stage 2's conditional-median imputation."""
    raw = pd.read_csv(RAW_DATA_PATH)
    raw = raw[raw["gender"] == SEX_CODE[sex_label]]
    return raw.set_index("patient_id")[CONTINUOUS].isnull()


def resolve_tie_by_complete_case_correlation(a: str, b: str, full_r: float,
                                              train_df: pd.DataFrame, raw_missing: pd.DataFrame,
                                              sex_label: str) -> str:
    """User-specified tie-break (2026-08-20) for pairs where neither member has a QRISK3
    basis to break the tie: recompute the correlation restricted to training rows where the
    continuous member's raw value was NOT imputed (complete cases only). If that complete-case
    r is still >= the threshold, the redundancy is judged real, and the coarser binary member
    is dropped in favour of the more information-rich continuous one. If it drops meaningfully
    below the threshold, the full-sample correlation is judged to be substantially an artefact
    of the conditional-median imputation, and the continuous member is dropped instead."""
    assert (a in CONTINUOUS) != (b in CONTINUOUS), (
        f"complete-case tie-break only implemented for one-continuous/one-binary pairs, "
        f"got {a} vs {b}"
    )
    continuous_var = a if a in CONTINUOUS else b
    binary_var = b if a in CONTINUOUS else a

    missing_lookup = raw_missing[continuous_var].reindex(train_df["patient_id"]).to_numpy()
    complete_case = train_df.loc[~missing_lookup]
    n_excluded = int(missing_lookup.sum())
    cc_r, _ = pointbiserialr(complete_case[binary_var], complete_case[continuous_var])

    print(
        f"  [{sex_label}] AMBIGUOUS redundancy: {a} vs {b} (full-sample r={full_r:.3f}) — "
        f"neither has a QRISK3 tie-break. Complete-case r (excluding {n_excluded} rows where "
        f"{continuous_var} was imputed) = {cc_r:.3f}."
    )
    if abs(cc_r) >= CORRELATION_THRESHOLD:
        drop, keep = binary_var, continuous_var
        reason = (f"complete-case r={cc_r:.3f} still >= {CORRELATION_THRESHOLD} — redundancy "
                  f"confirmed real; keeping the more information-rich continuous variable")
    else:
        drop, keep = continuous_var, binary_var
        reason = (f"complete-case r={cc_r:.3f} drops below {CORRELATION_THRESHOLD} — "
                  f"full-sample correlation was inflated by imputation; keeping the complete "
                  f"binary variable instead")
    print(f"  [{sex_label}] tie-break: dropping {drop}, keeping {keep} ({reason}).")
    return drop


def screen_multicollinearity(train_df: pd.DataFrame, raw_missing: pd.DataFrame, sex_label: str):
    """§5.1: Pearson (continuous-continuous) and point-biserial (continuous-binary)
    correlation screening on the training fold only. Flags |r| > 0.80 as redundant and drops
    the member with weaker QRISK3 clinical basis (§7). Ties (both or neither in the QRISK3
    basis) fall back to the complete-case correlation tie-break above, per the user's
    2026-08-20 instruction — not silently reconciled by any default rule."""
    rows = []
    for i, a in enumerate(CONTINUOUS):
        for b in CONTINUOUS[i + 1:]:
            r, _ = pearsonr(train_df[a], train_df[b])
            rows.append((a, b, "pearson", r))
    for a in CONTINUOUS:
        for b in BINARY:
            r, _ = pointbiserialr(train_df[b], train_df[a])
            rows.append((a, b, "point-biserial", r))
    corr_df = pd.DataFrame(rows, columns=["var_a", "var_b", "method", "r"])

    dropped = []
    flagged = corr_df[corr_df["r"].abs() > CORRELATION_THRESHOLD]
    for _, row in flagged.iterrows():
        a, b, r = row["var_a"], row["var_b"], row["r"]
        a_in, b_in = a in QRISK3_BASIS, b in QRISK3_BASIS
        if a_in and not b_in:
            drop, keep = b, a
            print(f"  [{sex_label}] redundant pair {a} vs {b} (r={r:.3f}) — dropping {drop} "
                  f"(weaker QRISK3 basis), keeping {keep}.")
        elif b_in and not a_in:
            drop, keep = a, b
            print(f"  [{sex_label}] redundant pair {a} vs {b} (r={r:.3f}) — dropping {drop} "
                  f"(weaker QRISK3 basis), keeping {keep}.")
        else:
            drop = resolve_tie_by_complete_case_correlation(
                a, b, r, train_df, raw_missing, sex_label
            )
        dropped.append(drop)

    if not flagged.shape[0]:
        print(f"  [{sex_label}] no predictor pairs exceed |r| > {CORRELATION_THRESHOLD} — "
              f"no multicollinearity redundancy found.")

    return sorted(set(dropped)), corr_df


def select_lasso_features(train_df: pd.DataFrame, val_df: pd.DataFrame, candidates: list,
                           sex_label: str):
    """§5.2: L1-penalised Cox partial-likelihood LASSO (Tibshirani, 1996), fitted on the
    training fold, penalty strength tuned on the validation fold by C-index — never the
    training set, never the test set."""
    X_train = train_df[candidates].astype(float)
    y_train = Surv.from_dataframe(EVENT_COL, TIME_COL, train_df)
    X_val = val_df[candidates].astype(float)
    y_val = Surv.from_dataframe(EVENT_COL, TIME_COL, val_df)

    model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, normalize=True)
    model.fit(X_train, y_train)

    best_alpha, best_cindex = None, -1.0
    for alpha in model.alphas_:
        risk_scores = model.predict(X_val, alpha=alpha)
        cindex = concordance_index_censored(
            y_val[EVENT_COL], y_val[TIME_COL], risk_scores
        )[0]
        if cindex > best_cindex:
            best_alpha, best_cindex = alpha, cindex

    coefs = pd.Series(
        model.coef_[:, list(model.alphas_).index(best_alpha)], index=candidates
    )
    selected = sorted(coefs[coefs != 0].index.tolist())

    print(f"  [{sex_label}] LASSO: best alpha = {best_alpha:.5f} (val C-index = "
          f"{best_cindex:.4f}), {len(selected)}/{len(candidates)} predictors survive.")
    return selected, coefs, best_alpha, best_cindex


def cross_reference_qrisk3(selected: list, candidates: list, sex_label: str):
    """§5.3: flag disagreements between the LASSO-surviving set and the QRISK3 predictor set
    explicitly — not silently reconciled."""
    disagreements = []
    for p in candidates:
        lasso_keeps = p in selected
        qrisk3_keeps = p in QRISK3_BASIS
        if lasso_keeps != qrisk3_keeps:
            disagreements.append(p)
            print(f"  [{sex_label}] DISAGREEMENT: {p} — LASSO "
                  f"{'keeps' if lasso_keeps else 'drops'}, QRISK3 basis "
                  f"{'includes' if qrisk3_keeps else 'excludes'} it.")
    if not disagreements:
        print(f"  [{sex_label}] no LASSO/QRISK3 disagreements.")
    return disagreements


def run_sex_pipeline(sex_label: str) -> pd.DataFrame:
    print(f"\n--- {sex_label.capitalize()} cohort ---")
    train_df = load_split(sex_label, "train")
    val_df = load_split(sex_label, "val")
    raw_missing = load_raw_missingness(sex_label)

    dropped_by_corr, corr_df = screen_multicollinearity(train_df, raw_missing, sex_label)
    candidates = [p for p in PREDICTORS if p not in dropped_by_corr]

    selected, coefs, best_alpha, best_cindex = select_lasso_features(
        train_df, val_df, candidates, sex_label
    )
    disagreements = cross_reference_qrisk3(selected, candidates, sex_label)

    summary = pd.DataFrame({"predictor": PREDICTORS})
    summary["dropped_by_multicollinearity"] = summary["predictor"].isin(dropped_by_corr)
    summary["lasso_coefficient"] = summary["predictor"].map(coefs)
    summary["selected_by_lasso"] = summary["predictor"].isin(selected)
    summary["in_qrisk3_basis"] = summary["predictor"].isin(QRISK3_BASIS)
    summary["lasso_qrisk3_disagreement"] = summary["predictor"].isin(disagreements)
    summary["final_selected_feature_set"] = summary["selected_by_lasso"]

    assert not summary.loc[summary["dropped_by_multicollinearity"], "final_selected_feature_set"].any(), (
        f"[{sex_label}] a multicollinearity-dropped predictor leaked into the final set"
    )
    assert summary["final_selected_feature_set"].sum() > 0, (
        f"[{sex_label}] LASSO selected zero features — degenerate result"
    )

    out_path = RESULTS_DIR / f"stage3_feature_selection_{sex_label}.csv"
    summary.to_csv(out_path, index=False)
    print(f"  [{sex_label}] wrote {out_path}")
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for sex_label in ["male", "female"]:
        try:
            run_sex_pipeline(sex_label)
        except FileNotFoundError as e:
            print(f"[STOP] {e}", file=sys.stderr)
            sys.exit(1)
    print("\n[OK] Stage 3 complete: multicollinearity screened, LASSO-selected, "
          "QRISK3-cross-referenced for both sexes.")


if __name__ == "__main__":
    main()
