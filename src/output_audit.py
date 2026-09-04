"""Strict completeness checks for a finished full-suite output directory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "protocol", "ra_mode", "window_setting", "covariance_estimator", "bank_seed",
    "aggregation", "subject", "session_train", "session_test", "method", "accuracy",
    "balanced_accuracy",
}
BANK_METHODS = {"B5", "P1", "P2"}
NONBANK_METHODS = {"B1", "B2", "B3", "B3_majority", "B4", "ordered_control"}


def audit_outputs(output: str | Path) -> dict:
    output = Path(output)
    frame = pd.read_csv(output / "classification_results.csv")
    run_paths = sorted((output / "runs").glob("*.csv"))
    partial_paths = sorted((output / "runs").glob("*.partial"))
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    identity = [
        "protocol", "ra_mode", "window_setting", "subject", "session_train",
        "session_test", "fold_id", "method", "bank_seed",
    ]
    duplicates = int(frame.duplicated(identity).sum())
    bank_seeds = sorted(int(value) for value in frame.loc[frame.method.isin(BANK_METHODS), "bank_seed"].dropna().unique())
    bank_missing = int(frame.loc[frame.method.isin(BANK_METHODS), "bank_seed"].isna().sum())
    nonbank_has_seed = int(frame.loc[frame.method.isin(NONBANK_METHODS), "bank_seed"].notna().sum())
    counts = frame.groupby("window_setting").size().to_dict()
    within = frame[frame.protocol == "within_session"]
    within_folds = within.groupby(["subject", "session_train"])["fold_id"].nunique()
    checks = {
        "classification_rows_5166": len(frame) == 5166,
        "run_csv_files_252": len(run_paths) == 252,
        "no_partial_run_files": len(partial_paths) == 0,
        "mandatory_columns_present": not missing_columns,
        "classification_identity_unique": duplicates == 0,
        "bank_seeds_exact_0_to_4": bank_seeds == [0, 1, 2, 3, 4],
        "bank_methods_always_seeded": bank_missing == 0,
        "nonbank_methods_never_seeded": nonbank_has_seed == 0,
        "w1_rows_2520": int(counts.get("W1", 0)) == 2520,
        "w2_rows_2646": int(counts.get("W2", 0)) == 2646,
        "within_session_has_five_folds_each": len(within_folds) == 18 and bool((within_folds == 5).all()),
        "nine_subjects": sorted(frame.subject.unique().tolist()) == list(range(1, 10)),
        "subject_ra_only_loso": bool((frame.loc[frame.ra_mode == "subject_ra", "protocol"] == "loso").all()),
        "no_ensemble_rows": not bool(frame.aggregation.astype(str).str.contains("ensemble", case=False).any()),
        "bank_summary_written": (output / "bank_summary.csv").exists(),
        "data_metadata_written": (output / "data_metadata.json").exists(),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": {
            "classification_rows": int(len(frame)),
            "run_csv_files": len(run_paths),
            "partial_run_files": len(partial_paths),
            "missing_columns": missing_columns,
            "duplicate_identity_rows": duplicates,
            "bank_seeds": bank_seeds,
            "bank_method_missing_seed_rows": bank_missing,
            "nonbank_method_seeded_rows": nonbank_has_seed,
        },
    }
