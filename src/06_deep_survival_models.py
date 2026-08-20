"""Stage 6 — deep survival models (CLAUDE.md §7): DeepSurv (Katzman et al., 2018) and
DeepHit, single-risk formulation (Lee et al., 2018), per sex, on Stage 3's selected feature
set (§5.4), via `pycox`/`torch`. `random_state=42` applied to both torch and numpy (§0.3).

pycox implements DeepSurv as `pycox.models.CoxPH` — a neural net replacing the linear
predictor in a Cox partial-likelihood model, which is exactly Katzman et al.'s architecture
(a PH-assumption-retaining model, per §7). DeepHit is `pycox.models.DeepHitSingle` — the
single-risk variant is used because this dataset has one composite outcome (§7).

Neural nets need standardised input features (fit on the training fold only, applied to
val/test — no leakage) — `StandardScaler`, used only here since it's a DL-specific
requirement, not part of Stage 2's shared preprocessing.

Per the runtime lesson from Stage 5 (an exhaustive grid made GBSA impractically slow): a
small 2-candidate grid per model, trained with early stopping (which bounds each fit's cost
regardless of the max epoch count), scored by C-index on the validation fold — same
validation-set-decides principle as every other stage, deliberately not exhaustive.

DeepHit's discrete time grid uses `num_durations=10` (one bin per observed year) — a natural,
non-arbitrary choice since `time_to_event_or_censoring` is already integer-valued years 1-10.

Formal scoring (§8) is deferred to Stage 7, same architecture as Stages 4-5. Predictions
written to `results/tables/stage6_deep_survival_predictions_{sex}.csv` with both a risk score
and predicted survival probability at the 10-year horizon per model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH, DeepHitSingle
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"

EVENT_COL = "heart_attack_or_stroke_occurred"
TIME_COL = "time_to_event_or_censoring"
SPLITS = ["train", "val", "test"]
HORIZON_YEARS = 10.0
# Integer years 1-10 — matches the dataset's own time resolution and DeepHit's own
# num_durations grid. Needed for Stage 7's integrated Brier score, which requires survival
# probabilities across a range, not one point (patched in when building Stage 7; see journal).
EVAL_TIME_GRID = list(range(1, 11))
RANDOM_STATE = 42
NUM_DURATIONS = 10

DEEPSURV_GRID = [
    {"hidden": [32, 32], "lr": 0.01},
    {"hidden": [64, 64], "lr": 0.001},
]
DEEPHIT_GRID = [
    {"hidden": [32, 32], "lr": 0.01},
    {"hidden": [64, 64], "lr": 0.001},
]
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10


def set_seed() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


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


def prepare_arrays(splits: dict, features: list):
    scaler = StandardScaler().fit(splits["train"][features].values)
    X = {
        part: scaler.transform(splits[part][features].values).astype("float32")
        for part in SPLITS
    }
    durations = {
        part: splits[part][TIME_COL].values.astype("float32") for part in SPLITS
    }
    events = {
        part: splits[part][EVENT_COL].values.astype("float32") for part in SPLITS
    }
    return X, durations, events


def build_net(in_features: int, hidden: list, out_features: int):
    return tt.practical.MLPVanilla(
        in_features, hidden, out_features, batch_norm=True, dropout=0.1,
        output_bias=(out_features == 1),
    )


def tune_and_fit_deepsurv(X, durations, events, sex_label: str):
    best_model, best_params, best_cindex = None, None, -1.0
    for params in DEEPSURV_GRID:
        set_seed()
        net = build_net(X["train"].shape[1], params["hidden"], 1)
        model = CoxPH(net, tt.optim.Adam(lr=params["lr"]))
        callbacks = [tt.callbacks.EarlyStopping(patience=EARLY_STOP_PATIENCE)]
        model.fit(
            X["train"], (durations["train"], events["train"]),
            batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, callbacks=callbacks, verbose=False,
            val_data=(X["val"], (durations["val"], events["val"])),
        )
        risk_scores = model.predict(X["val"]).flatten()
        cindex = concordance_index_censored(
            events["val"].astype(bool), durations["val"], risk_scores
        )[0]
        print(f"  [{sex_label}] DeepSurv {params} -> val C-index = {cindex:.4f}")
        if cindex > best_cindex:
            best_model, best_params, best_cindex = model, params, cindex
    print(f"  [{sex_label}] DeepSurv best: {best_params} (val C-index = {best_cindex:.4f})")
    best_model.compute_baseline_hazards()
    return best_model, best_params, best_cindex


def tune_and_fit_deephit(X, durations, events, sex_label: str):
    labtrans = DeepHitSingle.label_transform(NUM_DURATIONS)
    y_train = labtrans.fit_transform(durations["train"], events["train"])
    y_val = labtrans.transform(durations["val"], events["val"])

    best_model, best_params, best_cindex = None, None, -1.0
    for params in DEEPHIT_GRID:
        set_seed()
        net = build_net(X["train"].shape[1], params["hidden"], labtrans.out_features)
        model = DeepHitSingle(
            net, tt.optim.Adam(lr=params["lr"]), duration_index=labtrans.cuts,
            alpha=0.2, sigma=0.1,
        )
        callbacks = [tt.callbacks.EarlyStopping(patience=EARLY_STOP_PATIENCE)]
        model.fit(
            X["train"], y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
            callbacks=callbacks, verbose=False, val_data=(X["val"], y_val),
        )
        surv_val = model.predict_surv_df(X["val"])
        risk_val = 1.0 - surv_val.loc[surv_val.index <= HORIZON_YEARS].iloc[-1].values
        cindex = concordance_index_censored(
            events["val"].astype(bool), durations["val"], risk_val
        )[0]
        print(f"  [{sex_label}] DeepHit {params} -> val C-index = {cindex:.4f}")
        if cindex > best_cindex:
            best_model, best_params, best_cindex = model, params, cindex
    print(f"  [{sex_label}] DeepHit best: {best_params} (val C-index = {best_cindex:.4f})")
    return best_model, best_params, best_cindex


def survival_at_horizon_from_df(surv_df: pd.DataFrame, horizon: float = HORIZON_YEARS) -> np.ndarray:
    """Both models return a step-function survival DataFrame indexed by time; take the last
    row at or before the horizon (both models' time grids include exactly year 10, since
    training durations are capped there)."""
    at_or_before = surv_df.loc[surv_df.index <= horizon]
    return at_or_before.iloc[-1].values


def survival_curve_at_grid_from_df(surv_df: pd.DataFrame, times=EVAL_TIME_GRID) -> dict:
    """Same rationale as Stages 4-5's `survival_curve_at_grid`: Stage 7's integrated Brier
    score needs survival probability at multiple time points, not just one."""
    return {t: survival_at_horizon_from_df(surv_df, horizon=t) for t in times}


def run_sex_pipeline(sex_label: str) -> None:
    print(f"\n--- {sex_label.capitalize()} cohort ---")
    splits = {part: load_split(sex_label, part) for part in SPLITS}
    features = load_selected_features(sex_label)
    print(f"  [{sex_label}] using {len(features)} Stage-3-selected features: {features}")

    X, durations, events = prepare_arrays(splits, features)

    deepsurv_model, deepsurv_params, _ = tune_and_fit_deepsurv(X, durations, events, sex_label)
    deephit_model, deephit_params, _ = tune_and_fit_deephit(X, durations, events, sex_label)

    rows = []
    for part in SPLITS:
        df = splits[part]
        deepsurv_scores = deepsurv_model.predict(X[part]).flatten()
        deepsurv_surv_df = deepsurv_model.predict_surv_df(X[part])
        deepsurv_curve = survival_curve_at_grid_from_df(deepsurv_surv_df)

        deephit_surv_df = deephit_model.predict_surv_df(X[part])
        deephit_curve = survival_curve_at_grid_from_df(deephit_surv_df)
        deephit_scores = 1.0 - deephit_curve[HORIZON_YEARS]

        row_data = {
            "patient_id": df["patient_id"],
            "split": part,
            TIME_COL: df[TIME_COL],
            EVENT_COL: df[EVENT_COL],
            "deepsurv_risk_score": deepsurv_scores,
            "deephit_risk_score": deephit_scores,
        }
        for t in EVAL_TIME_GRID:
            row_data[f"deepsurv_survival_at_{t}y"] = deepsurv_curve[t]
            row_data[f"deephit_survival_at_{t}y"] = deephit_curve[t]
        rows.append(pd.DataFrame(row_data))
    predictions = pd.concat(rows, ignore_index=True)

    for col in ["deepsurv_risk_score", "deephit_risk_score"]:
        assert predictions[col].notna().all(), f"[{sex_label}] NaN values in {col}"
    for t in EVAL_TIME_GRID:
        for prefix in ["deepsurv", "deephit"]:
            col = f"{prefix}_survival_at_{t}y"
            assert predictions[col].between(0, 1).all(), f"[{sex_label}] {col} outside [0, 1]"

    test_mask = predictions["split"] == "test"
    test_deepsurv_cindex = concordance_index_censored(
        predictions.loc[test_mask, EVENT_COL].astype(bool),
        predictions.loc[test_mask, TIME_COL],
        predictions.loc[test_mask, "deepsurv_risk_score"],
    )[0]
    test_deephit_cindex = concordance_index_censored(
        predictions.loc[test_mask, EVENT_COL].astype(bool),
        predictions.loc[test_mask, TIME_COL],
        predictions.loc[test_mask, "deephit_risk_score"],
    )[0]
    print(f"  [{sex_label}] test-set C-index (informal check, not the formal §8 evaluation): "
          f"DeepSurv = {test_deepsurv_cindex:.4f}, DeepHit = {test_deephit_cindex:.4f}")

    out_path = RESULTS_TABLES / f"stage6_deep_survival_predictions_{sex_label}.csv"
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
    print("\n[OK] Stage 6 complete: DeepSurv and DeepHit tuned, fitted, and scored for "
          "both sexes.")


if __name__ == "__main__":
    main()
