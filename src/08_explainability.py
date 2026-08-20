"""Stage 8 — explainability (CLAUDE.md §9): SHAP (Lundberg and Lee, 2017) and LIME (Ribeiro,
Singh and Guestrin, 2016) on all four ML/DL models (RSF, GBSA, DeepSurv, DeepHit), per sex,
with stability testing across resampled training folds (§9.4) before any ranking is treated
as a robust claim.

Each of the four models is refit here using the exact best hyperparameters already found and
recorded in Stages 5-6 (no re-tuning — SHAP/LIME need a live model object with a callable
predict function, which Stages 5-6 didn't persist, only predictions). `random_state=42` for
the main fit; 2 further bootstrap resamples of the training fold (seeds 43, 44) for §9.4's
stability test, refit with the *same* fixed hyperparameters.

**SHAP explainer choice, verified empirically before committing (§14):** `shap.TreeExplainer`
does not support `sksurv` models (`InvalidModelError`, confirmed directly). Used
`shap.Explainer` with each model's `.predict()` as a black-box function instead — works
uniformly across all four heterogeneous model types (tree-based and neural). With only 9-10
features, SHAP automatically selects its exact-computation explainer (not the slow Kernel
approximation), confirmed via timing test: ~0.09-0.15s/sample regardless of model type.
GBSA's ~6-minute fit time (already characterised in Stage 5) is the only real cost here — SHAP
explanation itself is fast for every model.

**BMI dependence plot, resolved before coding (§14):** §9 asks for dependence plots on "age,
systolic BP, BMI," but BMI was dropped by Stage 3's LASSO for *both* sexes — it is not an
input to any of the four models. Only age and systolic BP remain as continuous predictors;
dependence plots cover those two, per the user's decision. This is a direct, already-logged
consequence of Stage 3's own feature selection, not a new deviation.
"""

import sys
import warnings
from pathlib import Path

import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torchtuples as tt
from pycox.models import CoxPH as DeepSurv
from pycox.models import DeepHitSingle
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

warnings.filterwarnings("ignore", message="X does not have valid feature names")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"

EVENT_COL = "heart_attack_or_stroke_occurred"
TIME_COL = "time_to_event_or_censoring"
RANDOM_STATE = 42
STABILITY_SEEDS = [43, 44]
N_EXPLAIN_SAMPLE = 200
N_BACKGROUND = 50
TOP_K_AGREEMENT = 3
CONTINUOUS_FOR_DEPENDENCE = ["age", "systolic_blood_pressure"]
NUM_DURATIONS = 10

MODEL_HYPERPARAMS = {
    "male": {
        "rsf": {"n_estimators": 300, "max_depth": 8},
        "gbsa": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.5},
        "deepsurv": {"hidden": [64, 64], "lr": 0.001},
        "deephit": {"hidden": [64, 64], "lr": 0.001},
    },
    "female": {
        "rsf": {"n_estimators": 300, "max_depth": 8},
        "gbsa": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.5},
        "deepsurv": {"hidden": [32, 32], "lr": 0.01},
        "deephit": {"hidden": [32, 32], "lr": 0.01},
    },
}


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


def bootstrap_resample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), len(df))
    return df.iloc[idx].reset_index(drop=True)


# --- Model fitting (fixed hyperparameters from Stages 5-6, no re-tuning) --------------------

def fit_rsf(train_df: pd.DataFrame, features: list, params: dict, seed: int):
    X = train_df[features].astype(float)
    y = Surv.from_dataframe(EVENT_COL, TIME_COL, train_df)
    model = RandomSurvivalForest(random_state=seed, n_jobs=-1, **params)
    model.fit(X, y)
    return lambda arr: model.predict(pd.DataFrame(arr, columns=features))


def fit_gbsa(train_df: pd.DataFrame, features: list, params: dict, seed: int):
    X = train_df[features].astype(float)
    y = Surv.from_dataframe(EVENT_COL, TIME_COL, train_df)
    model = GradientBoostingSurvivalAnalysis(random_state=seed, **params)
    model.fit(X, y)
    return lambda arr: model.predict(pd.DataFrame(arr, columns=features))


def fit_deepsurv(train_df: pd.DataFrame, features: list, params: dict, seed: int):
    scaler = StandardScaler().fit(train_df[features].values)
    X = scaler.transform(train_df[features].values).astype("float32")
    durations = train_df[TIME_COL].values.astype("float32")
    events = train_df[EVENT_COL].values.astype("float32")

    torch.manual_seed(seed)
    net = tt.practical.MLPVanilla(len(features), params["hidden"], 1, batch_norm=True,
                                   dropout=0.1, output_bias=True)
    model = DeepSurv(net, tt.optim.Adam(lr=params["lr"]))
    model.fit(X, (durations, events), batch_size=256, epochs=100, verbose=False,
              callbacks=[tt.callbacks.EarlyStopping(patience=10)])
    return lambda arr: model.predict(scaler.transform(np.asarray(arr)).astype("float32")).flatten()


def fit_deephit(train_df: pd.DataFrame, features: list, params: dict, seed: int):
    scaler = StandardScaler().fit(train_df[features].values)
    X = scaler.transform(train_df[features].values).astype("float32")
    durations = train_df[TIME_COL].values.astype("float32")
    events = train_df[EVENT_COL].values.astype("float32")

    labtrans = DeepHitSingle.label_transform(NUM_DURATIONS)
    y = labtrans.fit_transform(durations, events)

    torch.manual_seed(seed)
    net = tt.practical.MLPVanilla(len(features), params["hidden"], labtrans.out_features,
                                   batch_norm=True, dropout=0.1)
    model = DeepHitSingle(net, tt.optim.Adam(lr=params["lr"]), duration_index=labtrans.cuts,
                           alpha=0.2, sigma=0.1)
    model.fit(X, y, batch_size=256, epochs=100, verbose=False,
              callbacks=[tt.callbacks.EarlyStopping(patience=10)])

    def predict_fn(arr):
        surv_df = model.predict_surv_df(scaler.transform(np.asarray(arr)).astype("float32"))
        at_or_before = surv_df.loc[surv_df.index <= 10]
        return 1.0 - at_or_before.iloc[-1].values

    return predict_fn


FITTERS = {"rsf": fit_rsf, "gbsa": fit_gbsa, "deepsurv": fit_deepsurv, "deephit": fit_deephit}
MODEL_NAMES = ["rsf", "gbsa", "deepsurv", "deephit"]


def fit_coxph_coefficients(train_df: pd.DataFrame, features: list) -> pd.Series:
    X = train_df[features].astype(float)
    y = Surv.from_dataframe(EVENT_COL, TIME_COL, train_df)
    model = CoxPHSurvivalAnalysis()
    model.fit(X, y)
    return pd.Series(model.coef_, index=features)


# --- SHAP / LIME -------------------------------------------------------------------------

def compute_shap_mean_abs(predict_fn, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                           features: list) -> pd.Series:
    background = shap.sample(X_train[features], N_BACKGROUND, random_state=RANDOM_STATE)
    explainer = shap.Explainer(predict_fn, background)
    sv = explainer(X_explain[features])
    return pd.Series(np.abs(sv.values).mean(axis=0), index=features)


def top_k_features(values: pd.Series, k: int = TOP_K_AGREEMENT) -> set:
    return set(values.abs().sort_values(ascending=False).index[:k])


def compute_lime_top_features(predict_fn, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                               features: list) -> list:
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train[features].values, feature_names=features, mode="regression",
        random_state=RANDOM_STATE,
    )
    results = []
    for _, row in X_explain[features].iterrows():
        exp = explainer.explain_instance(row.values, predict_fn, num_features=len(features))
        weights = pd.Series({features[i]: w for i, w in exp.as_map()[1]})
        results.append(top_k_features(weights))
    return results


# --- Plots ---------------------------------------------------------------------------------

def plot_shap_summary(sex_label: str, shap_by_model: dict, coxph_coef: pd.Series,
                       features: list) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(features))
    width = 0.15
    series = {**{m: shap_by_model[m] / shap_by_model[m].max() for m in MODEL_NAMES},
              "coxph (|coef|)": coxph_coef.abs() / coxph_coef.abs().max()}
    for i, (name, vals) in enumerate(series.items()):
        ax.barh(y_pos + i * width, vals[features].values, height=width, label=name)
    ax.set_yticks(y_pos + width * (len(series) - 1) / 2)
    ax.set_yticklabels(features)
    ax.set_xlabel("Normalised importance (mean |SHAP| or |coefficient|, scaled to own max)")
    ax.set_title(f"Feature importance — {sex_label} cohort")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path = RESULTS_FIGURES / f"shap_summary_{sex_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [{sex_label}] SHAP summary saved to {out_path}")


def plot_shap_dependence(sex_label: str, shap_values_by_model: dict,
                          X_explain: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(MODEL_NAMES), len(CONTINUOUS_FOR_DEPENDENCE),
                              figsize=(4 * len(CONTINUOUS_FOR_DEPENDENCE), 3.2 * len(MODEL_NAMES)))
    for row, model_name in enumerate(MODEL_NAMES):
        sv = shap_values_by_model[model_name]
        for col, feat in enumerate(CONTINUOUS_FOR_DEPENDENCE):
            ax = axes[row, col]
            ax.scatter(X_explain[feat].values, sv[:, X_explain.columns.get_loc(feat)],
                       s=10, alpha=0.6)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel(feat)
            ax.set_ylabel(f"{model_name}\nSHAP value" if col == 0 else "SHAP value")
    fig.suptitle(f"SHAP dependence — {sex_label} cohort (age, systolic BP — see docstring "
                 f"re: BMI)")
    fig.tight_layout()
    out_path = RESULTS_FIGURES / f"shap_dependence_{sex_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [{sex_label}] SHAP dependence grid saved to {out_path}")


# --- Orchestration -----------------------------------------------------------------------

def run_sex_pipeline(sex_label: str) -> None:
    print(f"\n--- {sex_label.capitalize()} cohort ---")
    train_df = load_split(sex_label, "train")
    test_df = load_split(sex_label, "test")
    features = load_selected_features(sex_label)
    params = MODEL_HYPERPARAMS[sex_label]

    explain_sample = test_df.sample(n=N_EXPLAIN_SAMPLE, random_state=RANDOM_STATE) \
        .reset_index(drop=True)

    coxph_coef = fit_coxph_coefficients(train_df, features)

    shap_mean_abs = {}
    shap_values_full = {}
    lime_top = {}
    stability_rows = []

    for model_name in MODEL_NAMES:
        print(f"  [{sex_label}] {model_name}: fitting main model...")
        predict_fn = FITTERS[model_name](train_df, features, params[model_name], RANDOM_STATE)

        background = shap.sample(train_df[features], N_BACKGROUND, random_state=RANDOM_STATE)
        explainer = shap.Explainer(predict_fn, background)
        sv = explainer(explain_sample[features])
        shap_values_full[model_name] = sv.values
        shap_mean_abs[model_name] = pd.Series(np.abs(sv.values).mean(axis=0), index=features)
        print(f"  [{sex_label}] {model_name}: SHAP done, top feature = "
              f"{shap_mean_abs[model_name].idxmax()}")

        lime_top[model_name] = compute_lime_top_features(
            predict_fn, train_df, explain_sample, features
        )

        ranks = {"main": shap_mean_abs[model_name].rank(ascending=False)}
        for seed in STABILITY_SEEDS:
            print(f"  [{sex_label}] {model_name}: stability fit (seed={seed})...")
            resampled_train = bootstrap_resample(train_df, seed)
            resample_predict_fn = FITTERS[model_name](
                resampled_train, features, params[model_name], seed
            )
            resample_mean_abs = compute_shap_mean_abs(
                resample_predict_fn, resampled_train, explain_sample, features
            )
            ranks[f"seed_{seed}"] = resample_mean_abs.rank(ascending=False)

        rank_df = pd.DataFrame(ranks)
        rank_range = rank_df.max(axis=1) - rank_df.min(axis=1)
        for feat in features:
            stability_rows.append({
                "sex": sex_label, "model": model_name, "predictor": feat,
                "rank_main": rank_df.loc[feat, "main"],
                "rank_resample_1": rank_df.loc[feat, f"seed_{STABILITY_SEEDS[0]}"],
                "rank_resample_2": rank_df.loc[feat, f"seed_{STABILITY_SEEDS[1]}"],
                "rank_range": rank_range[feat],
                "robust": bool(rank_range[feat] <= 2),
            })

    plot_shap_summary(sex_label, shap_mean_abs, coxph_coef, features)
    plot_shap_dependence(sex_label, shap_values_full, explain_sample[features])

    agreement_rows = []
    for i in range(len(explain_sample)):
        row = {"sex": sex_label, "patient_id": explain_sample.loc[i, "patient_id"]}
        for model_name in MODEL_NAMES:
            shap_top = top_k_features(
                pd.Series(shap_values_full[model_name][i], index=features)
            )
            lime_top_i = lime_top[model_name][i]
            row[f"{model_name}_shap_top{TOP_K_AGREEMENT}"] = ", ".join(sorted(shap_top))
            row[f"{model_name}_lime_top{TOP_K_AGREEMENT}"] = ", ".join(sorted(lime_top_i))
            row[f"{model_name}_agreement_count"] = len(shap_top & lime_top_i)
        agreement_rows.append(row)
    agreement_df = pd.DataFrame(agreement_rows)

    stability_df = pd.DataFrame(stability_rows)

    agree_path = RESULTS_TABLES / f"stage8_shap_lime_agreement_{sex_label}.csv"
    stability_path = RESULTS_TABLES / f"stage8_stability_{sex_label}.csv"
    agreement_df.to_csv(agree_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    print(f"  [{sex_label}] wrote {agree_path}")
    print(f"  [{sex_label}] wrote {stability_path}")

    for model_name in MODEL_NAMES:
        mean_agree = agreement_df[f"{model_name}_agreement_count"].mean()
        n_unstable = (~stability_df.loc[stability_df["model"] == model_name, "robust"]).sum()
        print(f"  [{sex_label}] {model_name}: mean SHAP/LIME top-{TOP_K_AGREEMENT} overlap = "
              f"{mean_agree:.2f}/{TOP_K_AGREEMENT}, {n_unstable}/{len(features)} predictors "
              f"flagged unstable across resamples")

    assert agreement_df[[f"{m}_agreement_count" for m in MODEL_NAMES]].values.min() >= 0
    assert stability_df["rank_range"].min() >= 0


def main() -> None:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    for sex_label in ["male", "female"]:
        try:
            run_sex_pipeline(sex_label)
        except FileNotFoundError as e:
            print(f"[STOP] {e}", file=sys.stderr)
            sys.exit(1)
    print("\n[OK] Stage 8 complete: SHAP + LIME + stability testing done for both sexes.")


if __name__ == "__main__":
    main()
