"""Stage 5 — survival ML models (CLAUDE.md §7): Random Survival Forest (RSF, Ishwaran et al.,
2008) and Gradient-Boosting Survival Analysis (GBSA, Friedman, 2001), per sex, on Stage 3's
selected feature set (§5.4 — same predictor set as CoxPH, so differences are attributable to
the model, not the inputs). `random_state=42` everywhere a seed is settable (§0.3).

Neither model has an explicitly specified tuning procedure in CLAUDE.md (unlike LASSO's §5.2,
which is explicit about validation-set tuning). Extending that same principle here: a small
grid is fit on the training fold and scored by Harrell's C-index on the validation fold — the
same validation-not-training, validation-not-test discipline used throughout §5/§6 — never an
exhaustive search, to keep runtime reasonable on a single machine; the exact grids are logged
in journal/agent_journal.md.

Formal scoring (C-index, Uno's C, Brier score, calibration, bootstrap 95% CIs — §8/§12) is
deferred to Stage 7 (07_evaluation.py), applied uniformly across all six models, per §11's
architecture (see Stage 4's docstring). Predictions written to
`results/tables/stage5_ml_model_predictions_{sex}.csv`, with both a risk score (C-index) and
a predicted survival probability at the 10-year horizon (Brier score/calibration) per model —
matching the format Stage 4's patch established.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"

EVENT_COL = "heart_attack_or_stroke_occurred"
TIME_COL = "time_to_event_or_censoring"
SPLITS = ["train", "val", "test"]
# Integer years 1-10 — matches the dataset's own time resolution. Needed for Stage 7's
# integrated Brier score, which requires survival probabilities across a range, not one point
# (patched in when building Stage 7, same as Stage 4's equivalent patch; see journal).
EVAL_TIME_GRID = list(range(1, 11))
RANDOM_STATE = 42

RSF_GRID = [
    {"n_estimators": n, "max_depth": d}
    for n in (100, 300)
    for d in (None, 8)
]
# Practicality-driven deviation from an exhaustive grid, agreed with the user 2026-08-20: an
# 8-combo grid with n_estimators up to 300 measured at 3+ hours total on this machine, because
# scikit-survival's Cox partial-likelihood loss evaluates the full risk set at every boosting
# stage — cost scales with n_estimators, not meaningfully with max_depth. Capped at
# n_estimators=100 and reduced to 3 combos; subsample=0.5 (stochastic gradient boosting,
# legitimately part of Friedman 2001 — the same paper GBSA is cited to, §7) both speeds up
# each fit and is a real regularisation technique, not just a speed hack.
GBSA_GRID = [
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
]
GBSA_SUBSAMPLE = 0.5


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


def survival_curve_at_grid(model, X: pd.DataFrame, times=EVAL_TIME_GRID) -> dict:
    """Same rationale as Stage 4's helper of the same name: Stage 7's integrated Brier score
    needs survival probability at multiple time points, not just one."""
    step_functions = model.predict_survival_function(X)
    return {t: np.array([fn(t) for fn in step_functions]) for t in times}


def tune_and_fit(model_cls, grid: list, X_train, y_train, X_val, y_val, sex_label: str,
                  model_name: str, extra_kwargs: dict = None):
    """Fit each grid candidate on the training fold, score by Harrell's C-index on the
    validation fold, return the best-scoring fitted model. Mirrors Stage 3's LASSO alpha
    tuning: validation set decides, training set never leaks into that decision."""
    extra_kwargs = extra_kwargs or {}
    best_model, best_params, best_cindex = None, None, -1.0
    for params in grid:
        model = model_cls(random_state=RANDOM_STATE, **params, **extra_kwargs)
        model.fit(X_train, y_train)
        risk_scores = model.predict(X_val)
        cindex = concordance_index_censored(y_val[EVENT_COL], y_val[TIME_COL], risk_scores)[0]
        print(f"  [{sex_label}] {model_name} {params} -> val C-index = {cindex:.4f}")
        if cindex > best_cindex:
            best_model, best_params, best_cindex = model, params, cindex
    print(f"  [{sex_label}] {model_name} best: {best_params} (val C-index = {best_cindex:.4f})")
    return best_model, best_params, best_cindex


def run_sex_pipeline(sex_label: str) -> None:
    print(f"\n--- {sex_label.capitalize()} cohort ---")
    splits = {part: load_split(sex_label, part) for part in SPLITS}
    features = load_selected_features(sex_label)
    print(f"  [{sex_label}] using {len(features)} Stage-3-selected features: {features}")

    X = {part: splits[part][features].astype(float) for part in SPLITS}
    y = {part: Surv.from_dataframe(EVENT_COL, TIME_COL, splits[part]) for part in SPLITS}

    rsf_model, rsf_params, rsf_val_cindex = tune_and_fit(
        RandomSurvivalForest, RSF_GRID, X["train"], y["train"], X["val"], y["val"],
        sex_label, "RSF", extra_kwargs={"n_jobs": -1}
    )
    gbsa_model, gbsa_params, gbsa_val_cindex = tune_and_fit(
        GradientBoostingSurvivalAnalysis, GBSA_GRID, X["train"], y["train"], X["val"], y["val"],
        sex_label, "GBSA", extra_kwargs={"subsample": GBSA_SUBSAMPLE}
    )

    rows = []
    for part in SPLITS:
        df = splits[part]
        rsf_scores = rsf_model.predict(X[part])
        rsf_curve = survival_curve_at_grid(rsf_model, X[part])
        gbsa_scores = gbsa_model.predict(X[part])
        gbsa_curve = survival_curve_at_grid(gbsa_model, X[part])
        row_data = {
            "patient_id": df["patient_id"],
            "split": part,
            TIME_COL: df[TIME_COL],
            EVENT_COL: df[EVENT_COL],
            "rsf_risk_score": rsf_scores,
            "gbsa_risk_score": gbsa_scores,
        }
        for t in EVAL_TIME_GRID:
            row_data[f"rsf_survival_at_{t}y"] = rsf_curve[t]
            row_data[f"gbsa_survival_at_{t}y"] = gbsa_curve[t]
        rows.append(pd.DataFrame(row_data))
    predictions = pd.concat(rows, ignore_index=True)

    for col in ["rsf_risk_score", "gbsa_risk_score"]:
        assert predictions[col].notna().all(), f"[{sex_label}] NaN values in {col}"
    for t in EVAL_TIME_GRID:
        for prefix in ["rsf", "gbsa"]:
            col = f"{prefix}_survival_at_{t}y"
            assert predictions[col].between(0, 1).all(), f"[{sex_label}] {col} outside [0, 1]"

    test_mask = predictions["split"] == "test"
    test_rsf_cindex = concordance_index_censored(
        predictions.loc[test_mask, EVENT_COL].astype(bool),
        predictions.loc[test_mask, TIME_COL],
        predictions.loc[test_mask, "rsf_risk_score"],
    )[0]
    test_gbsa_cindex = concordance_index_censored(
        predictions.loc[test_mask, EVENT_COL].astype(bool),
        predictions.loc[test_mask, TIME_COL],
        predictions.loc[test_mask, "gbsa_risk_score"],
    )[0]
    print(f"  [{sex_label}] test-set C-index (informal check, not the formal §8 evaluation): "
          f"RSF = {test_rsf_cindex:.4f}, GBSA = {test_gbsa_cindex:.4f}")

    out_path = RESULTS_TABLES / f"stage5_ml_model_predictions_{sex_label}.csv"
    predictions.to_csv(out_path, index=False)
    print(f"  [{sex_label}] wrote {out_path}")


def main() -> None:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    for sex_label in ["male", "female"]:
        try:
            run_sex_pipeline(sex_label)
        except FileNotFoundError as e:
            print(f"[STOP] {e}", file=sys.stderr)
            sys.exit(1)
    print("\n[OK] Stage 5 complete: RSF and GBSA tuned, fitted, and scored for both sexes.")


if __name__ == "__main__":
    main()
