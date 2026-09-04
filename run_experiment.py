"""Run the complete no-neural-network BNCI2014-001 prerequisite suite."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import moabb
import numpy as np
import pandas as pd
import pyriemann
import sklearn

from src.data import ESTIMATOR_NAME, WINDOWS, build_covariance_log_cache, load_cache, validate_cache
from src.diagnostics import gate0_diagnostics
from src.evaluation import (
    Prediction,
    b1_tangent_lda,
    b2_airm_mdm,
    b3_window_tangent_lda,
    b4_mean_log_features,
    feature_methods,
    lda_prediction,
    metrics,
    ordered_tangent_lda,
    p1_pooled_class_measure,
)
from src.gates import evaluate_all_gates
from src.projections import BANK_SEEDS, N_DIRECTIONS, load_projection_cache, precompute_projection_cache
from src.protocols import assert_trial_disjoint, balanced_source_indices, make_splits
from src.ra import build_subject_ra_cache, load_subject_ra_arrays
from src.report import write_initial_report, write_report


ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = [
    "protocol", "ra_mode", "window_setting", "covariance_estimator", "bank_seed",
    "aggregation", "subject", "session_train", "session_test", "method", "accuracy",
    "balanced_accuracy",
]


def _row(split, setting: str, method: str, aggregation: str, prediction: Prediction, y_test: np.ndarray, *, bank_seed=None, n_train: int) -> dict:
    return {
        "protocol": split.protocol,
        "ra_mode": split.ra_mode,
        "window_setting": setting,
        "covariance_estimator": ESTIMATOR_NAME,
        "bank_seed": bank_seed,
        "aggregation": aggregation,
        "subject": int(split.subject),
        "session_train": split.session_train,
        "session_test": split.session_test,
        "method": method,
        **metrics(y_test, prediction),
        "fold_id": int(split.fold_id),
        "n_train_trials": int(n_train),
        "n_test_trials": int(len(y_test)),
        "transductive_calibration": split.ra_mode == "subject_ra",
    }


def _run_one(split, setting, arrays, labels, subjects, projection_root) -> list[dict]:
    full_cov, window_cov, window_log = arrays
    base_train = np.asarray(split.train_indices, dtype=np.int64)
    test = np.asarray(split.test_indices, dtype=np.int64)
    train = balanced_source_indices(base_train, labels, subjects) if split.protocol == "loso" else base_train
    p1_train = balanced_source_indices(base_train, labels, subjects)
    y_train = np.asarray(labels[train])
    y_test = np.asarray(labels[test])
    rows: list[dict] = []

    b1 = b1_tangent_lda(np.asarray(full_cov[train]), y_train, np.asarray(full_cov[test]))
    rows.append(_row(split, setting, "B1", "trial_tangent", b1, y_test, n_train=len(train)))
    b2 = b2_airm_mdm(np.asarray(full_cov[train]), y_train, np.asarray(full_cov[test]))
    rows.append(_row(split, setting, "B2", "airm_nearest_class_mean", b2, y_test, n_train=len(train)))

    b3_mean, b3_vote = b3_window_tangent_lda(
        np.asarray(window_cov[train]), y_train, np.asarray(window_cov[test])
    )
    rows.append(_row(split, setting, "B3", "mean_decision_score", b3_mean, y_test, n_train=len(train)))
    rows.append(_row(split, setting, "B3_majority", "majority_vote_secondary", b3_vote, y_test, n_train=len(train)))

    b4_train = b4_mean_log_features(np.asarray(window_log[train]))
    b4_test = b4_mean_log_features(np.asarray(window_log[test]))
    b4 = lda_prediction(b4_train, y_train, b4_test)
    rows.append(_row(split, setting, "B4", "mean_window_log_svec", b4, y_test, n_train=len(train)))

    if setting == "W2":
        ordered = ordered_tangent_lda(np.asarray(window_cov[train]), y_train, np.asarray(window_cov[test]))
        rows.append(_row(split, setting, "ordered_control", "ordered_nonoverlap", ordered, y_test, n_train=len(train)))

    for seed in BANK_SEEDS:
        projections = load_projection_cache(projection_root, split.ra_mode, setting, seed)
        train_projection = np.asarray(projections[train])
        test_projection = np.asarray(projections[test])
        b5_train, p2_train = feature_methods(train_projection)
        b5_test, p2_test = feature_methods(test_projection)
        b5 = lda_prediction(b5_train, y_train, b5_test)
        p2 = lda_prediction(p2_train, y_train, p2_test)
        rows.append(_row(split, setting, "B5", "mean_std_projection_moments", b5, y_test, bank_seed=seed, n_train=len(train)))
        rows.append(_row(split, setting, "P2", "exact_sorted_order_statistics", p2, y_test, bank_seed=seed, n_train=len(train)))

        p1 = p1_pooled_class_measure(
            np.asarray(projections[p1_train]), np.asarray(labels[p1_train]), test_projection
        )
        rows.append(_row(split, setting, "P1", "pooled_class_measure", p1, y_test, bank_seed=seed, n_train=len(p1_train)))
        # A paired copy of the same B2 prediction is used only by the P1 diagnostic
        # gate. It is not an ensemble or an independently fit classifier.
        rows.append(_row(split, setting, "B2_bank", "airm_nearest_class_mean_paired_copy", b2, y_test, bank_seed=seed, n_train=len(train)))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "bnci2014_001")
    parser.add_argument("--max-splits", type=int, default=None, help="Development/debug only; a partial run always fails completeness gates.")
    parser.add_argument("--protocol", choices=["all", "within_session", "cross_session", "loso"], default="all")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs_root = args.output / "runs"
    runs_root.mkdir(exist_ok=True)
    report_path = args.output / "REPORT.md"
    write_initial_report(report_path)
    write_initial_report(ROOT / "REPORT.md")

    cache_metadata = build_covariance_log_cache(args.cache_root)
    data = load_cache(args.cache_root)
    cache_validation = validate_cache(data)
    (args.output / "cache_validation.json").write_text(json.dumps(cache_validation, indent=2) + "\n")
    if cache_validation["status"] != "VALID":
        raise FloatingPointError("Covariance/log cache is numerically invalid; classification cannot proceed.")

    gate0_rows: list[dict] = []
    for setting in WINDOWS:
        gate0_rows.append({**gate0_diagnostics(data.window_log[setting], setting=setting), "subject": "all"})
        for subject_value in np.unique(data.subject):
            indices = np.flatnonzero(data.subject == subject_value)
            gate0_rows.append({**gate0_diagnostics(data.window_log[setting][indices], setting=setting), "subject": int(subject_value)})
    pd.DataFrame(gate0_rows).to_csv(args.output / "gate0_diagnostics.csv", index=False)

    ra_metadata = build_subject_ra_cache(data)
    ra_full_cov, _, ra_window_cov, ra_window_log = load_subject_ra_arrays(data)
    arrays_by_mode = {
        "no_ra": (data.full_cov, data.window_cov, data.window_log),
        "subject_ra": (ra_full_cov, ra_window_cov, ra_window_log),
    }
    projection_root = data.root / "projection_features"
    projection_metadata = precompute_projection_cache(
        {(mode, setting): arrays[2][setting] for mode, arrays in arrays_by_mode.items() for setting in WINDOWS},
        projection_root,
    )
    (args.output / "bank_audit.json").write_text(json.dumps(projection_metadata, indent=2, sort_keys=True) + "\n")

    splits = make_splits(data.y, data.subject, data.session, data.trial_id)
    for split in splits:
        assert_trial_disjoint(split, data.trial_id)
    if args.protocol != "all":
        wanted = {args.protocol}
        if args.protocol == "cross_session":
            wanted = {"s1_to_s2", "s2_to_s1"}
        splits = [split for split in splits if split.protocol in wanted]
    if args.max_splits is not None:
        splits = splits[: args.max_splits]

    for split_index, split in enumerate(splits, start=1):
        for setting in WINDOWS:
            run_path = runs_root / f"{split.protocol}_{split.ra_mode}_sub{split.subject}_fold{split.fold_id}_{setting}.csv"
            if run_path.exists():
                continue
            print(f"[{split_index}/{len(splits)}] {split.protocol} {split.ra_mode} subject={split.subject} fold={split.fold_id} {setting}", flush=True)
            full_cov, window_covs, window_logs = arrays_by_mode[split.ra_mode]
            rows = _run_one(
                split,
                setting,
                (full_cov, window_covs[setting], window_logs[setting]),
                data.y,
                data.subject,
                projection_root,
            )
            pd.DataFrame(rows).to_csv(run_path, index=False)

    run_paths = sorted(runs_root.glob("*.csv"))
    if not run_paths:
        raise RuntimeError("No classification run files were produced.")
    frame = pd.concat([pd.read_csv(path) for path in run_paths], ignore_index=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AssertionError(f"Missing classification columns: {missing}")
    frame = frame.sort_values(["protocol", "ra_mode", "subject", "fold_id", "window_setting", "method", "bank_seed"], na_position="first")
    frame.to_csv(args.output / "classification_results.csv", index=False)
    gates = evaluate_all_gates(frame)
    (args.output / "GATES.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")

    inference_total = float(frame["inference_time_seconds"].sum())
    inference_trials = int(frame["n_test_trials"].sum())
    timing = {
        "covariance_log_cache_time_seconds": float(cache_metadata["covariance_log_cache_time_seconds"]),
        "subject_ra_covariance_log_time_seconds": float(ra_metadata["ra_covariance_log_time_seconds"]),
        "projection_feature_precomputation_time_seconds": float(projection_metadata["projection_feature_precomputation_time_seconds"]),
        "classifier_or_prototype_fitting_time_seconds": float(frame["classifier_or_prototype_fitting_time_seconds"].sum()),
        "inference_time_seconds": inference_total,
        "amortized_inference_seconds_per_trial": inference_total / max(1, inference_trials),
        "measured_inference_trial_evaluations": inference_trials,
    }
    pd.DataFrame([timing]).to_csv(args.output / "timing_summary.csv", index=False)
    provenance = {
        "upstream_repository": "isaacrulz93/T3DA",
        "upstream_commit": "7d0b8b1757c8284a6e9a641487d3cf6106073dc5",
        "upstream_blob": "010e445af8fcdf65f8719e465ff2ed2e3672e3ea",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "pyriemann": pyriemann.__version__,
        "moabb": moabb.__version__,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    write_report(
        report_path,
        gates=gates,
        frame=frame,
        cache_metadata=cache_metadata,
        gate0=gate0_rows,
        timing=timing,
        provenance=provenance,
    )
    write_report(
        ROOT / "REPORT.md",
        gates=gates,
        frame=frame,
        cache_metadata=cache_metadata,
        gate0=gate0_rows,
        timing=timing,
        provenance=provenance,
    )
    print(f"REPORT.md: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
