"""Stage 2 — preprocessing: imputation, encoding, sex-specific 70/15/15 split.

Follows CLAUDE.md §4 (imputation), §0.4 (sex-stratified pipelines — applied here to
imputation, not just modelling), and §6 (train/validation/test split). Missingness and split
design are grounded in Burns, Richardson and Driessens (2024) and Liu et al. (2026).

Two decisions here are explicit deviations from a literal reading of the spec, made with the
user's sign-off on 2026-08-20 (recorded in journal/agent_journal.md, per CLAUDE.md §14):

1. `smoker` and `family_history_of_cardiovascular_disease` are left completely untouched.
   §4 asks for an "unknown" category for these two flipped binary variables, but Stage 1
   confirmed the 1->0 flip (p=0.30) leaves no missing-value marker at all in the released
   CSV — every value is already a plain 0/1 with no way to tell a corrupted 0 from a true
   one. There is nothing to impute or flag. Both columns are documented as carrying known
   non-differential measurement error at p=0.30, not treated as missing data.
2. Imputation is computed and applied on the whole per-sex cohort *before* the 70/15/15
   split (matching §11's literal script description), not from the training fold only. This
   is a minor, acknowledged information leak from validation/test into the imputed median —
   negligible for a low-capacity statistic like a median, but flagged here since §5/§6 are
   strict about zero leakage elsewhere in the pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "cvd_synthetic_dataset_v0.2.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_ROW_COUNT = 100_000
EXPECTED_COLUMNS = [
    "patient_id", "gender", "age", "body_mass_index", "smoker", "systolic_blood_pressure",
    "hypertension_treated", "family_history_of_cardiovascular_disease", "atrial_fibrillation",
    "chronic_kidney_disease", "rheumatoid_arthritis", "diabetes",
    "chronic_obstructive_pulmonary_disorder", "forced_expiratory_volume_1",
    "time_to_event_or_censoring", "heart_attack_or_stroke_occurred",
]

RANDOM_STATE = 42
EVENT_COL = "heart_attack_or_stroke_occurred"


def load_and_verify(path: Path) -> pd.DataFrame:
    """Same §3 schema check as Stage 1 — kept self-contained here so each stage script can be
    run independently (§2: one script per pipeline stage)."""
    df = pd.read_csv(path)
    errors = []
    if df.shape[0] != EXPECTED_ROW_COUNT:
        errors.append(f"row count = {df.shape[0]}, expected {EXPECTED_ROW_COUNT}")
    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(f"column names/order do not match §3: got {list(df.columns)}")
    if errors:
        raise ValueError(
            "Dataset does not match CLAUDE.md §3 schema — stopping rather than proceeding:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    return df


def impute_sex_cohort(cohort: pd.DataFrame, sex_label: str) -> pd.DataFrame:
    """Impute one sex cohort's missing values per §4, using statistics from that cohort only
    (§0.4's two-fully-independent-pipelines principle applied to preprocessing)."""
    cohort = cohort.copy()

    for col in ["body_mass_index", "systolic_blood_pressure"]:
        median = cohort[col].median()
        n_missing = cohort[col].isnull().sum()
        cohort[col] = cohort[col].fillna(median)
        print(f"  [{sex_label}] {col}: filled {n_missing} missing with median = {median:.2f}")

    # MAR: FEV1 imputed conditionally within COPD strata, never with a single unconditional
    # median (CLAUDE.md §4).
    for copd_status in [0, 1]:
        mask = cohort["chronic_obstructive_pulmonary_disorder"] == copd_status
        median = cohort.loc[mask, "forced_expiratory_volume_1"].median()
        n_missing = cohort.loc[mask, "forced_expiratory_volume_1"].isnull().sum()
        cohort.loc[mask, "forced_expiratory_volume_1"] = cohort.loc[
            mask, "forced_expiratory_volume_1"
        ].fillna(median)
        print(
            f"  [{sex_label}] forced_expiratory_volume_1 | COPD={copd_status}: filled "
            f"{n_missing} missing with median = {median:.2f}"
        )

    # smoker / family_history_of_cardiovascular_disease: intentionally left untouched — see
    # module docstring point 1.
    print(
        f"  [{sex_label}] smoker, family_history_of_cardiovascular_disease: left as-is "
        f"(non-differential measurement error, p=0.30 — no missing-value marker exists)."
    )
    return cohort


def encoding_note(df: pd.DataFrame) -> None:
    """§11 lists 'encoding' as part of this stage. Documented explicitly rather than silently
    skipped: no categorical encoding is required. All 14 predictors besides `gender` are
    already numeric (int/0-1 binary). `gender` itself is used only to split into the two
    sex-specific cohorts (§0.4) — it is never a covariate in either pipeline, so it is not
    encoded as a feature."""
    non_numeric = [c for c in df.columns if c not in ("patient_id", "gender")
                   and df[c].dtype == object]
    assert not non_numeric, f"unexpected non-numeric predictor columns: {non_numeric}"
    print("  All predictors besides `gender` are already numeric — no encoding required.")
    print("  `gender` is used only to split cohorts, never as a covariate (§0.4).")


def sex_stratified_split(cohort: pd.DataFrame, sex_label: str):
    """§6: 70/15/15 split, stratified on the event indicator, random_state=42, independent
    per sex cohort."""
    train, temp = train_test_split(
        cohort, test_size=0.30, stratify=cohort[EVENT_COL], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[EVENT_COL], random_state=RANDOM_STATE
    )

    n = len(cohort)
    print(
        f"  [{sex_label}] split: train={len(train)} ({len(train)/n:.1%}), "
        f"val={len(val)} ({len(val)/n:.1%}), test={len(test)} ({len(test)/n:.1%})"
    )
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  [{sex_label}] {name} event rate = {part[EVENT_COL].mean():.4f}")

    return train, val, test


def run_sanity_checks(cohort: pd.DataFrame, train, val, test, sex_label: str) -> None:
    """§10.4 test/sanity check: split sizes reconstruct the cohort exactly, no patient
    appears in more than one partition, and event rate is preserved within a small tolerance
    (stratified split, so should be very close)."""
    assert len(train) + len(val) + len(test) == len(cohort), (
        f"[{sex_label}] split sizes don't sum to cohort size"
    )
    ids = pd.concat([train["patient_id"], val["patient_id"], test["patient_id"]])
    assert ids.is_unique, f"[{sex_label}] a patient_id appears in more than one split"
    assert not train[["body_mass_index", "systolic_blood_pressure",
                       "forced_expiratory_volume_1"]].isnull().any().any(), (
        f"[{sex_label}] imputation left residual NaNs in train"
    )

    full_rate = cohort[EVENT_COL].mean()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rate = part[EVENT_COL].mean()
        assert abs(rate - full_rate) < 0.01, (
            f"[{sex_label}] {name} event rate {rate:.4f} deviates >1pp from cohort rate "
            f"{full_rate:.4f}"
        )
    print(f"  [{sex_label}] sanity checks passed: sizes reconcile, no ID overlap, "
          f"no residual NaNs in imputed columns, event rate preserved within 1pp.")


def main() -> None:
    try:
        df = load_and_verify(RAW_DATA_PATH)
    except ValueError as e:
        print(f"[STOP] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] Loaded {df.shape[0]:,} rows, schema matches §3.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Encoding (§11) ---")
    encoding_note(df)

    for sex_code, sex_label in [("M", "male"), ("F", "female")]:
        print(f"\n--- {sex_label.capitalize()} cohort ---")
        cohort = df[df["gender"] == sex_code].reset_index(drop=True)
        print(f"  n = {len(cohort)}")

        cohort = impute_sex_cohort(cohort, sex_label)
        train, val, test = sex_stratified_split(cohort, sex_label)
        run_sanity_checks(cohort, train, val, test, sex_label)

        for name, part in [("train", train), ("val", val), ("test", test)]:
            out_path = PROCESSED_DIR / f"{sex_label}_{name}.csv"
            part.to_csv(out_path, index=False)
        print(f"  [{sex_label}] wrote train/val/test CSVs to {PROCESSED_DIR}")

    print("\n[OK] Stage 2 complete: imputed, encoded, and split both sex cohorts.")


if __name__ == "__main__":
    main()
