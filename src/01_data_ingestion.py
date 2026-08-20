"""Stage 1 — data ingestion.

Loads the CVD synthetic dataset, verifies it matches the schema fixed in CLAUDE.md §3, and
profiles the missingness and censoring mechanisms named in CLAUDE.md §4. All mechanisms
(informative censoring, MCAR flips/drops, MAR FEV1 drops, deliberate noise/irrelevant
predictors) are documented in Burns, Richardson and Driessens (2024).
"""

import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cvd_synthetic_dataset_v0.2.csv"

EXPECTED_ROW_COUNT = 100_000
EXPECTED_COLUMNS = [
    "patient_id",
    "gender",
    "age",
    "body_mass_index",
    "smoker",
    "systolic_blood_pressure",
    "hypertension_treated",
    "family_history_of_cardiovascular_disease",
    "atrial_fibrillation",
    "chronic_kidney_disease",
    "rheumatoid_arthritis",
    "diabetes",
    "chronic_obstructive_pulmonary_disorder",
    "forced_expiratory_volume_1",
    "time_to_event_or_censoring",
    "heart_attack_or_stroke_occurred",
]

# Expected MCAR/MAR drop probabilities per CLAUDE.md §4, used only as an honesty check against
# the actually observed rates in this specific draw of the synthetic dataset.
EXPECTED_DROP_RATES = {
    "systolic_blood_pressure": 0.10,
    "body_mass_index": 0.30,
    "forced_expiratory_volume_1_copd_pos": 0.05,
    "forced_expiratory_volume_1_copd_neg": 0.75,
}
RATE_TOLERANCE = 0.02


def load_and_verify(path: Path) -> pd.DataFrame:
    """Load the dataset and verify it matches CLAUDE.md §3. Stops rather than proceeding
    silently on any schema mismatch, per CLAUDE.md §3 and §14."""
    df = pd.read_csv(path)

    errors = []
    if df.shape[0] != EXPECTED_ROW_COUNT:
        errors.append(f"row count = {df.shape[0]}, expected {EXPECTED_ROW_COUNT}")
    if df.shape[1] != len(EXPECTED_COLUMNS):
        errors.append(f"column count = {df.shape[1]}, expected {len(EXPECTED_COLUMNS)}")
    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(f"column names/order do not match §3: got {list(df.columns)}")

    if errors:
        raise ValueError(
            "Dataset does not match CLAUDE.md §3 schema — stopping rather than proceeding:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    print(f"[OK] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns, schema matches §3.")
    return df


def profile_missingness(df: pd.DataFrame) -> None:
    """Profile each MCAR/MAR mechanism named in CLAUDE.md §4, individually and by name —
    not a generic df.isnull().sum() dump."""
    print("\n--- Missingness profile (CLAUDE.md §4) ---")

    # MCAR: smoker and family_history are silently flipped 1->0 (p=0.30), not set to NaN, so
    # this mechanism leaves no missing-value trace in the loaded data and cannot be measured
    # empirically here. Documented as a known, non-recoverable limitation for those two columns.
    for col in ["smoker", "family_history_of_cardiovascular_disease"]:
        n_null = df[col].isnull().sum()
        print(
            f"  {col}: MCAR 1->0 flip (p=0.30, unobservable in the data itself). "
            f"NaN count = {n_null} (expected 0, since flips are silent, not deletions)."
        )

    # MCAR: systolic_blood_pressure and body_mass_index are dropped to NaN outright.
    for col, key in [
        ("systolic_blood_pressure", "systolic_blood_pressure"),
        ("body_mass_index", "body_mass_index"),
    ]:
        observed = df[col].isnull().mean()
        expected = EXPECTED_DROP_RATES[key]
        flag = "" if abs(observed - expected) <= RATE_TOLERANCE else "  <-- ANOMALY, see §0.7"
        print(f"  {col}: MCAR drop, observed rate = {observed:.4f} (expected ~{expected}){flag}")

    # MAR: forced_expiratory_volume_1 dropped conditionally on COPD status.
    fev1_by_copd = df.groupby("chronic_obstructive_pulmonary_disorder")[
        "forced_expiratory_volume_1"
    ].apply(lambda s: s.isnull().mean())
    for copd_status, key in [(1, "forced_expiratory_volume_1_copd_pos"),
                              (0, "forced_expiratory_volume_1_copd_neg")]:
        observed = fev1_by_copd.get(copd_status, float("nan"))
        expected = EXPECTED_DROP_RATES[key]
        flag = "" if abs(observed - expected) <= RATE_TOLERANCE else "  <-- ANOMALY, see §0.7"
        print(
            f"  forced_expiratory_volume_1 | COPD={copd_status}: observed rate = "
            f"{observed:.4f} (expected ~{expected}){flag}"
        )


def profile_censoring(df: pd.DataFrame) -> None:
    """Profile the informative right-censoring mechanism named in CLAUDE.md §4: dropout
    probability scales with time-to-event (Burns, Richardson and Driessens, 2024). This is
    exactly why downstream evaluation must use IPCW-weighted estimators (Uno's C-statistic,
    integrated Brier score, per §8) rather than treating censoring as uninformative."""
    print("\n--- Censoring profile (CLAUDE.md §4) ---")

    event_rate = df["heart_attack_or_stroke_occurred"].mean()
    admin_censored = ((df["heart_attack_or_stroke_occurred"] == 0)
                       & (df["time_to_event_or_censoring"] == 10)).mean()
    early_censored = ((df["heart_attack_or_stroke_occurred"] == 0)
                       & (df["time_to_event_or_censoring"] < 10)).mean()

    print(f"  Event rate (heart attack/stroke within 10y): {event_rate:.4f}")
    print(f"  Administratively censored at year 10 (study end): {admin_censored:.4f}")
    print(f"  Censored before year 10 (informative dropout): {early_censored:.4f}")
    print(
        "  NOTE: dropout-before-year-10 is the informative-censoring component — its rate "
        "scaling with time-to-event is a property of the data-generating process, not a "
        "processing artefact. Must be handled with IPCW-weighted C-index/Brier score at §8."
    )


def profile_noise_note() -> None:
    """CLAUDE.md §4: the release paper states the dataset deliberately includes variables that
    did not significantly contribute to the analyses, incorporating irrelevance and noise
    (Burns, Richardson and Driessens, 2024, p. 3). Which predictors are informative is not
    determined here — that is the explicit purpose of the Stage 2 feature-selection pipeline
    (§5), not an ingestion-time judgement call."""
    print("\n--- Noise/irrelevance note (CLAUDE.md §4) ---")
    print(
        "  Not all 14 predictors are assumed informative by design (Burns, Richardson and "
        "Driessens, 2024). Feature relevance is determined in Stage 2 (§5: multicollinearity "
        "screening + per-sex LASSO), not assumed here."
    )


def main() -> None:
    try:
        df = load_and_verify(DATA_PATH)
    except ValueError as e:
        print(f"[STOP] {e}", file=sys.stderr)
        sys.exit(1)

    profile_missingness(df)
    profile_censoring(df)
    profile_noise_note()

    print("\n[OK] Stage 1 complete: dataset verified, missingness and censoring profiled.")


if __name__ == "__main__":
    main()
