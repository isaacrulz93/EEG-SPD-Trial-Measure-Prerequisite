"""W1-only stage gates and conservative final interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED_SUBJECTS = 9
EXPECTED_BANKS = 5


def _stage_rows(frame: pd.DataFrame, protocol: str, ra_mode: str) -> pd.DataFrame:
    rows = frame[(frame["window_setting"] == "W1") & (frame["ra_mode"] == ra_mode)]
    if protocol == "cross_session":
        return rows[rows["protocol"].isin(["s1_to_s2", "s2_to_s1"])]
    return rows[rows["protocol"] == protocol]


def _bank_comparison(rows: pd.DataFrame, candidate: str, control: str) -> dict:
    selected = rows[rows["method"].isin([candidate, control]) & rows["bank_seed"].notna()].copy()
    if selected.empty:
        return {"status": "NOT_RUN", "pass": False}
    selected["bank_seed"] = selected["bank_seed"].astype(int)
    aggregated = (
        selected.groupby(["subject", "bank_seed", "method"], as_index=False)["balanced_accuracy"].mean()
        .pivot(index=["subject", "bank_seed"], columns="method", values="balanced_accuracy")
        .dropna()
        .reset_index()
    )
    aggregated["delta"] = aggregated[candidate] - aggregated[control]
    bank_delta = aggregated.groupby("bank_seed")["delta"].mean().sort_index()
    subject_delta = aggregated.groupby("subject")["delta"].median().sort_index()
    median_bank_mean_delta = float(bank_delta.median())
    wins = int((subject_delta > 0.0).sum())
    positive_banks = int((bank_delta > 0.0).sum())
    n_subjects = int(subject_delta.size)
    n_banks = int(bank_delta.size)
    complete = n_subjects == EXPECTED_SUBJECTS and n_banks == EXPECTED_BANKS
    passed = complete and median_bank_mean_delta >= 0.010 and wins >= 6 and positive_banks >= 4
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": bool(passed),
        "complete_nine_subject_five_bank_suite": complete,
        "median_across_banks_mean_subject_delta": median_bank_mean_delta,
        "subject_wins": wins,
        "required_subject_wins": 6,
        "positive_mean_delta_banks": positive_banks,
        "required_positive_banks": 4,
        "n_subjects": n_subjects,
        "n_banks": n_banks,
        "bank_mean_deltas": {str(int(key)): float(value) for key, value in bank_delta.items()},
        "subject_median_deltas": {str(int(key)): float(value) for key, value in subject_delta.items()},
        "delta_threshold": 0.010,
    }


def _bank_vs_independent_comparison(rows: pd.DataFrame, candidate: str, control: str) -> dict:
    candidate_rows = rows[(rows["method"] == candidate) & rows["bank_seed"].notna()].copy()
    control_rows = rows[rows["method"] == control].copy()
    if candidate_rows.empty or control_rows.empty:
        return {"status": "NOT_RUN", "pass": False}
    candidate_rows["bank_seed"] = candidate_rows["bank_seed"].astype(int)
    candidate_agg = candidate_rows.groupby(["subject", "bank_seed"])["balanced_accuracy"].mean().reset_index()
    control_agg = control_rows.groupby("subject")["balanced_accuracy"].mean().rename("control").reset_index()
    merged = candidate_agg.merge(control_agg, on="subject", how="inner")
    merged["delta"] = merged["balanced_accuracy"] - merged["control"]
    bank_delta = merged.groupby("bank_seed")["delta"].mean().sort_index()
    subject_delta = merged.groupby("subject")["delta"].median().sort_index()
    median_bank_mean_delta = float(bank_delta.median())
    wins = int((subject_delta > 0.0).sum())
    positive_banks = int((bank_delta > 0.0).sum())
    complete = len(subject_delta) == EXPECTED_SUBJECTS and len(bank_delta) == EXPECTED_BANKS
    passed = complete and median_bank_mean_delta >= 0.010 and wins >= 6 and positive_banks >= 4
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": bool(passed),
        "complete_nine_subject_five_bank_suite": bool(complete),
        "median_across_banks_mean_subject_delta": median_bank_mean_delta,
        "subject_wins": wins,
        "required_subject_wins": 6,
        "positive_mean_delta_banks": positive_banks,
        "required_positive_banks": 4,
        "n_subjects": int(len(subject_delta)),
        "n_banks": int(len(bank_delta)),
        "bank_mean_deltas": {str(int(key)): float(value) for key, value in bank_delta.items()},
        "subject_median_deltas": {str(int(key)): float(value) for key, value in subject_delta.items()},
        "delta_threshold": 0.010,
    }


def _candidate_vs_bank_independent(rows: pd.DataFrame, candidate: str, control: str) -> dict:
    candidate_rows = rows[(rows["method"] == candidate) & rows["bank_seed"].notna()].copy()
    control_rows = rows[rows["method"] == control].copy()
    if candidate_rows.empty or control_rows.empty:
        return {"median_across_banks_mean_subject_delta": float("nan")}
    candidate_rows["bank_seed"] = candidate_rows["bank_seed"].astype(int)
    candidate_agg = candidate_rows.groupby(["subject", "bank_seed"])["balanced_accuracy"].mean().reset_index()
    control_agg = control_rows.groupby("subject")["balanced_accuracy"].mean().rename("control").reset_index()
    merged = candidate_agg.merge(control_agg, on="subject", how="inner")
    merged["delta"] = merged["balanced_accuracy"] - merged["control"]
    bank_delta = merged.groupby("bank_seed")["delta"].mean().sort_index()
    return {
        "median_across_banks_mean_subject_delta": float(bank_delta.median()),
        "bank_mean_deltas": {str(int(key)): float(value) for key, value in bank_delta.items()},
    }


def evaluate_stage(frame: pd.DataFrame, *, name: str, protocol: str, ra_mode: str) -> dict:
    rows = _stage_rows(frame, protocol, ra_mode)
    p2 = _bank_comparison(rows, "P2", "B5")
    p1 = _bank_vs_independent_comparison(rows, "P1", "B2")
    extra = None
    threshold = None
    if name == "Gate 1B":
        extra = _candidate_vs_bank_independent(rows, "P2", "B1")
        threshold = -0.010
    elif name == "Gate 2 subject_ra":
        extra = _candidate_vs_bank_independent(rows, "P2", "B1")
        threshold = 0.000
    extra_pass = True if threshold is None else bool(
        np.isfinite(extra["median_across_banks_mean_subject_delta"])
        and extra["median_across_banks_mean_subject_delta"] >= threshold
    )
    p2_pass = bool(p2.get("pass", False) and extra_pass)
    return {
        "name": name,
        "protocol": protocol,
        "ra_mode": ra_mode,
        "primary_window_setting": "W1",
        "status": "PASS" if p2_pass else "FAIL",
        "p2_pass": p2_pass,
        "p2_vs_b5": p2,
        "p2_vs_b1": extra,
        "p2_vs_b1_threshold": threshold,
        "p2_vs_b1_requirement_pass": extra_pass,
        "p1_diagnostic_vs_b2": p1,
        "p1_failure_stops_p2": False,
    }


def evaluate_all_gates(frame: pd.DataFrame) -> dict:
    stages = {
        "gate_1a": evaluate_stage(frame, name="Gate 1A", protocol="within_session", ra_mode="no_ra"),
        "gate_1b": evaluate_stage(frame, name="Gate 1B", protocol="cross_session", ra_mode="no_ra"),
        "gate_2_no_ra": evaluate_stage(frame, name="Gate 2 no_ra", protocol="loso", ra_mode="no_ra"),
        "gate_2_subject_ra": evaluate_stage(frame, name="Gate 2 subject_ra", protocol="loso", ra_mode="subject_ra"),
    }
    p2_pass = all(stage["p2_pass"] for stage in stages.values())
    p1_pass = all(stage["p1_diagnostic_vs_b2"].get("pass", False) for stage in stages.values())
    if not p2_pass:
        final_case = "close"
    elif p1_pass:
        final_case = "SPDSW-readout"
    else:
        final_case = "quantile-only"
    return {
        **stages,
        "overall_p2_pass_requires_all_four_stages": p2_pass,
        "overall_p1_diagnostic_pass_requires_all_four_stages": p1_pass,
        "final_case": final_case,
        "w2_can_change_gate": False,
        "approximately_equal_tolerance": 0.010,
    }
