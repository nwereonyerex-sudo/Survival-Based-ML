"""Stage 7 — evaluation (CLAUDE.md §8/§12): C-index (Harrell et al., 1982), Uno's IPCW
C-statistic (Uno et al., 2011), integrated Brier score (Graf et al., 1999) with calibration
plots, all with bootstrap 95% CIs — applied uniformly to all six models, per sex, on the test
set only (§6: held out exclusively for final evaluation).

Every reported number is compared against the sex-specific CoxPH *and* QRISK3-style baselines
with an effect size (paired bootstrap difference) and CI, per §8's last bullet.

**IPCW time-grid technical constraint, discovered while building this stage:**
`sksurv`'s IPCW-weighted estimators (Uno's C, Brier score) require evaluation times strictly
less than the test set's maximum follow-up time — the censoring-distribution estimator is
undefined exactly at that boundary, since ~93% of patients are administratively censored at
exactly year 10 (Stage 1's censoring profile). Evaluated up to 9.99 years instead of exactly
10 (`TAU`) — a standard, well-understood technical workaround in this literature, not a scope
reduction: every model is still assessed at essentially the 10-year horizon the spec asks
for.

**QRISK3-style score gets a single-timepoint Brier score, not the integrated version** — it
has no multi-year curve (§7: fixed clinical formula, not a fitted survival model; see the
Stages 4-6 patch note in the journal, 2026-08-20). Labelled explicitly as such in the output
table, not silently presented as equivalent to the other five models' integrated score.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw, \
    integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"

EVENT_COL = "heart_attack_or_stroke_occurred"
TIME_COL = "time_to_event_or_censoring"
TAU = 9.99  # see module docstring — just inside the [min, max) domain sksurv requires
IBS_TIMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, TAU]
N_BOOTSTRAP = 1000
RANDOM_STATE = 42
N_DECILES = 10

# model -> (risk score column, dict of {t: survival column} or None if no curve, source table)
MODELS = {
    "coxph": {"risk_col": "coxph_risk_score", "has_curve": True, "table": "stage4"},
    "qrisk3_style": {"risk_col": "qrisk3_style_score", "has_curve": False, "table": "stage4"},
    "rsf": {"risk_col": "rsf_risk_score", "has_curve": True, "table": "stage5"},
    "gbsa": {"risk_col": "gbsa_risk_score", "has_curve": True, "table": "stage5"},
    "deepsurv": {"risk_col": "deepsurv_risk_score", "has_curve": True, "table": "stage6"},
    "deephit": {"risk_col": "deephit_risk_score", "has_curve": True, "table": "stage6"},
}
BASELINE_MODELS = ["coxph", "qrisk3_style"]


def load_test_predictions(sex_label: str) -> pd.DataFrame:
    tables = {}
    for name in ["stage4_baseline", "stage5_ml_model", "stage6_deep_survival"]:
        path = RESULTS_TABLES / f"{name}_predictions_{sex_label}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run the corresponding stage first.")
        df = pd.read_csv(path)
        tables[name.split("_")[0]] = df[df["split"] == "test"].reset_index(drop=True)

    merged = tables["stage4"]
    for key in ["stage5", "stage6"]:
        merged = merged.merge(
            tables[key].drop(columns=[TIME_COL, EVENT_COL, "split"]),
            on="patient_id", how="inner", validate="one_to_one",
        )
    assert len(merged) == len(tables["stage4"]), "test-set row count changed after merge"
    return merged


def curve_matrix(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Each model's survival probability at every point in IBS_TIMES (9.99 uses the year-10
    column — the last observed step in a right-continuous step function doesn't change
    between t=9.99 and t=10, so this is exact, not an approximation)."""
    cols = [f"{model_name}_survival_at_{t}y" for t in [1, 2, 3, 4, 5, 6, 7, 8, 9]] + \
           [f"{model_name}_survival_at_10y"]
    return df[cols].values


def qrisk3_survival_10y(df: pd.DataFrame) -> np.ndarray:
    return 1.0 - df["qrisk3_style_score"].values / 100.0


def compute_metrics(model_name: str, df: pd.DataFrame, y_train, idx: np.ndarray) -> dict:
    """§8: C-index, Uno's C, and Brier score (integrated for curve models, single-timepoint
    at TAU for QRISK3-style) for one bootstrap resample (or the full set if idx covers it)."""
    spec = MODELS[model_name]
    sub = df.iloc[idx]
    y_sub = Surv.from_dataframe(EVENT_COL, TIME_COL, sub)
    risk = sub[spec["risk_col"]].values

    cindex = concordance_index_censored(y_sub[EVENT_COL], y_sub[TIME_COL], risk)[0]
    uno = concordance_index_ipcw(y_train, y_sub, risk, tau=TAU)[0]

    if spec["has_curve"]:
        curve = curve_matrix(sub, model_name)
        brier = integrated_brier_score(y_train, y_sub, curve, IBS_TIMES)
    else:
        surv_10y = qrisk3_survival_10y(sub).reshape(-1, 1)
        brier = brier_score(y_train, y_sub, surv_10y, [TAU])[1][0]

    return {"cindex": cindex, "uno_cindex": uno, "brier": brier}


def bootstrap_all_models(df: pd.DataFrame, y_train, sex_label: str) -> pd.DataFrame:
    n = len(df)
    rng = np.random.default_rng(RANDOM_STATE)
    full_idx = np.arange(n)

    point = {m: compute_metrics(m, df, y_train, full_idx) for m in MODELS}
    print(f"  [{sex_label}] point estimates:")
    for m, vals in point.items():
        print(f"    {m}: C-index={vals['cindex']:.4f}, Uno's C={vals['uno_cindex']:.4f}, "
              f"Brier={vals['brier']:.4f}")

    boot = {m: {"cindex": [], "uno_cindex": [], "brier": []} for m in MODELS}
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        for m in MODELS:
            result = compute_metrics(m, df, y_train, idx)
            for k, v in result.items():
                boot[m][k].append(v)
        if (b + 1) % 250 == 0:
            print(f"  [{sex_label}] bootstrap {b + 1}/{N_BOOTSTRAP}")

    rows = []
    for m in MODELS:
        row = {"sex": sex_label, "model": m,
               "brier_score_type": "integrated" if MODELS[m]["has_curve"]
                                    else "single_timepoint_9.99y"}
        for metric in ["cindex", "uno_cindex", "brier"]:
            vals = np.array(boot[m][metric])
            row[metric] = point[m][metric]
            row[f"{metric}_ci_low"] = np.percentile(vals, 2.5)
            row[f"{metric}_ci_high"] = np.percentile(vals, 97.5)
        for baseline in BASELINE_MODELS:
            if m == baseline:
                continue
            for metric in ["cindex", "uno_cindex", "brier"]:
                diffs = np.array(boot[m][metric]) - np.array(boot[baseline][metric])
                row[f"delta_{metric}_vs_{baseline}"] = point[m][metric] - point[baseline][metric]
                row[f"delta_{metric}_vs_{baseline}_ci_low"] = np.percentile(diffs, 2.5)
                row[f"delta_{metric}_vs_{baseline}_ci_high"] = np.percentile(diffs, 97.5)
        rows.append(row)
    return pd.DataFrame(rows)


def decile_calibration(df: pd.DataFrame, model_name: str):
    """§8: observed vs predicted 10-year event probability, decile-grouped. Observed
    probability uses a KM estimate within each decile (accounts for censoring properly),
    not a naive raw event proportion."""
    spec = MODELS[model_name]
    predicted_risk = (1.0 - qrisk3_survival_10y(df)) if not spec["has_curve"] \
        else (1.0 - df[f"{model_name}_survival_at_10y"].values)

    deciles = pd.qcut(predicted_risk, N_DECILES, labels=False, duplicates="drop")
    mean_predicted, observed = [], []
    for d in sorted(pd.unique(deciles)):
        mask = deciles == d
        mean_predicted.append(predicted_risk[mask].mean())
        time_d, surv_d = kaplan_meier_estimator(
            df.loc[mask, EVENT_COL].astype(bool), df.loc[mask, TIME_COL]
        )
        at_or_before = surv_d[time_d <= TAU]
        surv_at_tau = at_or_before[-1] if len(at_or_before) else 1.0
        observed.append(1.0 - surv_at_tau)
    return np.array(mean_predicted), np.array(observed)


def plot_calibration_grid(df: pd.DataFrame, sex_label: str) -> None:
    curves = {m: decile_calibration(df, m) for m in MODELS}
    shared_lim = max(max(pred.max(), obs.max()) for pred, obs in curves.values()) * 1.1

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, model_name in zip(axes.flat, MODELS):
        pred, obs = curves[model_name]
        ax.plot([0, shared_lim], [0, shared_lim], linestyle="--", color="gray", linewidth=1)
        ax.plot(pred, obs, marker="o")
        ax.set_title(model_name)
        ax.set_xlabel("Mean predicted 10y risk")
        ax.set_ylabel("Observed 10y risk (KM)")
        ax.set_xlim(0, shared_lim)
        ax.set_ylim(0, shared_lim)
    fig.suptitle(f"Calibration — {sex_label} cohort, decile-grouped")
    fig.tight_layout()
    out_path = RESULTS_FIGURES / f"calibration_{sex_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [{sex_label}] calibration grid saved to {out_path}")


def run_sex_pipeline(sex_label: str) -> pd.DataFrame:
    print(f"\n--- {sex_label.capitalize()} cohort ---")
    train_df = pd.read_csv(PROCESSED_DIR / f"{sex_label}_train.csv")
    y_train = Surv.from_dataframe(EVENT_COL, TIME_COL, train_df)
    test_df = load_test_predictions(sex_label)
    print(f"  [{sex_label}] test set: {len(test_df)} patients")

    summary = bootstrap_all_models(test_df, y_train, sex_label)
    plot_calibration_grid(test_df, sex_label)
    return summary


def main() -> None:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for sex_label in ["male", "female"]:
        try:
            all_summaries.append(run_sex_pipeline(sex_label))
        except FileNotFoundError as e:
            print(f"[STOP] {e}", file=sys.stderr)
            sys.exit(1)

    final = pd.concat(all_summaries, ignore_index=True)
    for metric in ["cindex", "uno_cindex", "brier"]:
        assert final[metric].between(-0.01, 1.01).all(), f"{metric} outside plausible [0,1] range"
        assert (final[f"{metric}_ci_low"] <= final[metric] + 1e-9).all(), (
            f"{metric} point estimate falls below its own CI lower bound"
        )
        assert (final[f"{metric}_ci_high"] >= final[metric] - 1e-9).all(), (
            f"{metric} point estimate falls above its own CI upper bound"
        )

    out_path = RESULTS_TABLES / "stage7_evaluation_summary.csv"
    final.to_csv(out_path, index=False)
    print(f"\n[OK] Stage 7 complete: six-model x two-sex evaluation table written to {out_path}")


if __name__ == "__main__":
    main()
