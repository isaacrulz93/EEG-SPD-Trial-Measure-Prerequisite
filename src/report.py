"""Generate the decision-first REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _status(gates: dict, key: str) -> str:
    return str(gates.get(key, {}).get("status", "NOT RUN"))


def write_initial_report(path: str | Path) -> None:
    Path(path).write_text(
        "1. Gate 1A status: NOT RUN\n"
        "2. Gate 1B status: NOT RUN\n"
        "3. Gate 2 no-RA status: NOT RUN\n"
        "4. Gate 2 RA status: NOT RUN\n"
        "5. final case: NOT RUN\n\n"
        "The report is replaced only after the full W1/W2 diagnostic suite completes.\n"
    )


def _method_summary(frame: pd.DataFrame, setting: str = "W1") -> str:
    selected = frame[frame["window_setting"] == setting]
    summary = selected.groupby(["protocol", "ra_mode", "method"])["balanced_accuracy"].mean().reset_index()
    return summary.to_markdown(index=False, floatfmt=".4f")


def _interpretation(frame: pd.DataFrame, gates: dict) -> list[str]:
    lines: list[str] = []
    p2_b5 = []
    for key in ("gate_1a", "gate_1b", "gate_2_no_ra", "gate_2_subject_ra"):
        value = gates[key]["p2_vs_b5"].get("median_across_banks_mean_subject_delta")
        if value is not None and np.isfinite(value):
            p2_b5.append(float(value))
    if p2_b5 and max(abs(value) for value in p2_b5) <= 0.010:
        lines.append("- P2 is approximately equal to B5 at the declared 0.010 BA tolerance; moment pooling explains the apparent gain and higher quantiles are unnecessary.")

    w2 = frame[frame["window_setting"] == "W2"]
    ordered = w2[w2["method"] == "ordered_control"].groupby(["protocol", "ra_mode", "subject"])["balanced_accuracy"].mean()
    p2 = w2[w2["method"] == "P2"].groupby(["protocol", "ra_mode", "subject"])["balanced_accuracy"].median()
    paired = pd.concat({"ordered": ordered, "p2": p2}, axis=1).dropna()
    if len(paired) and float((paired["ordered"] - paired["p2"]).mean()) >= 0.010:
        lines.append("- The ordered non-overlapping-window control clearly exceeds P2; temporal order is more useful than an exchangeable trial measure.")
    if not lines:
        lines.append("- Neither the declared moment-equivalence nor ordered-control interpretation trigger fired.")
    return lines


def write_report(
    path: str | Path,
    *,
    gates: dict,
    frame: pd.DataFrame,
    cache_metadata: dict,
    gate0: list[dict],
    timing: dict,
    provenance: dict,
) -> None:
    ra_label = _status(gates, "gate_2_subject_ra") + " (TRANSDUCTIVE CALIBRATION)"
    lines = [
        f"1. Gate 1A status: {_status(gates, 'gate_1a')}",
        f"2. Gate 1B status: {_status(gates, 'gate_1b')}",
        f"3. Gate 2 no-RA status: {_status(gates, 'gate_2_no_ra')}",
        f"4. Gate 2 RA status: {ra_label}",
        f"5. final case: {gates['final_case']}",
        "",
        "# Trial-measure prerequisite report",
        "",
        "This is a no-neural-network prerequisite check. It contains no SPDHSW layer, TTA transport, source-memory alignment, or reference search. Gate 0 and P1 are descriptive/non-stopping; all cheap diagnostics were run after the covariance/log cache was valid.",
        "",
        "## Data and preprocessing",
        "",
        f"- Raw epoch tensor shape: `{tuple(cache_metadata['raw_epoch_shape'])}`",
        f"- Sampling frequency: `{cache_metadata['sampling_frequency_hz']} Hz`",
        f"- Actual sampled duration: `{cache_metadata['actual_epoch_duration_seconds']:.6f} s` (endpoint span `{cache_metadata['sample_span_seconds']:.6f} s`)",
        f"- W1 exact windows/trial: `{cache_metadata['windows']['W1']['windows_per_trial']}`; total `{cache_metadata['windows']['W1']['total_windows']}`",
        f"- W2 exact windows/trial: `{cache_metadata['windows']['W2']['windows_per_trial']}`; total `{cache_metadata['windows']['W2']['total_windows']}`",
        f"- Covariance estimator: `{cache_metadata['covariance_estimator']}` = OAS, trace division, `1e-6 I`, symmetrization.",
        "- Filtering was performed once by the frozen MOABB `MotorImagery` paradigm on each full epoch before windowing.",
        "",
        "## Gate 0 — descriptive only",
        "",
        pd.DataFrame([row for row in gate0 if row.get("subject") == "all"])[[
            "setting", "within_trial_dispersion", "between_trial_dispersion", "ratio_within",
            "isotropic_fraction", "traceless_shape_fraction", "pca_effective_rank", "pc1_fraction"
        ]].to_markdown(index=False, floatfmt=".6g"),
        "",
        "No dispersion, effective-rank, or PC1 threshold was used to stop the experiment.",
        "",
        "## W1 primary classification summary",
        "",
        _method_summary(frame, "W1"),
        "",
        "Every projection-bank row is a separate fixed-bank result. Medians across seeds 0–4 are robustness summaries, not prediction ensembles.",
        "",
        "## W2 sensitivity summary (non-gating)",
        "",
        _method_summary(frame, "W2"),
        "",
        "W2 never rescues or overturns a W1 decision.",
        "",
        "## Gate details",
        "",
        "```json",
        json.dumps(gates, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation controls",
        "",
        *_interpretation(frame, gates),
        "",
        "## Timing",
        "",
        f"- Covariance/log-cache time: `{timing['covariance_log_cache_time_seconds']:.6f} s`",
        f"- Subject-RA covariance/log time: `{timing['subject_ra_covariance_log_time_seconds']:.6f} s`",
        f"- Projection-feature precomputation time: `{timing['projection_feature_precomputation_time_seconds']:.6f} s`",
        f"- Classifier/prototype fitting time: `{timing['classifier_or_prototype_fitting_time_seconds']:.6f} s`",
        f"- Total measured inference time: `{timing['inference_time_seconds']:.6f} s`",
        f"- Amortized measured inference time: `{timing['amortized_inference_seconds_per_trial']:.9f} s/trial`",
        "",
        "Projection precomputation is reported separately from fitting and inference. Per-row fitting and amortized per-trial inference measurements are in `classification_results.csv`.",
        "",
        "## Provenance and claim discipline",
        "",
        f"- Frozen MOABB factory source: `{provenance['upstream_repository']}` at `{provenance['upstream_commit']}`; file blob `{provenance['upstream_blob']}`.",
        "- `subject_ra` uses each target subject's own unlabeled trials and is explicitly transductive calibration, not fully inductive.",
        "- P1 compares to a pooled class measure. It is not called a Wasserstein barycenter or prototype.",
        "- No downstream neural-layer claim follows from this prerequisite experiment.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")

