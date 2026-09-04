"""Descriptive Gate 0 diagnostics; no heuristic here can stop classification."""

from __future__ import annotations

import numpy as np

from .geometry import svec


def gate0_diagnostics(window_logs: np.ndarray, *, setting: str) -> dict:
    """Compute dispersion, isotropic residual energy, and residual PCA summaries."""

    logs = np.asarray(window_logs, dtype=np.float64)
    if logs.ndim != 4:
        raise ValueError("Expected window logs with shape (trial,window,d,d).")
    vectors = svec(logs)
    trial_means = vectors.mean(axis=1)
    residuals = vectors - trial_means[:, None, :]
    global_mean = trial_means.mean(axis=0)
    within = float(np.mean(np.sum(residuals**2, axis=-1)))
    between = float(np.mean(np.sum((trial_means - global_mean) ** 2, axis=-1)))
    total = within + between
    ratio_within = within / total if total > 0.0 else float("nan")

    d = int(logs.shape[-1])
    h = np.zeros(vectors.shape[-1], dtype=np.float64)
    h[:d] = 1.0 / np.sqrt(d)  # svec stores diagonals first only if handled explicitly below.
    # np.triu_indices interleaves diagonal/off-diagonal entries; build H safely.
    i, j = np.triu_indices(d)
    h = np.where(i == j, 1.0 / np.sqrt(d), 0.0)
    residual_2d = residuals.reshape(-1, residuals.shape[-1])
    denominator = float(np.sum(residual_2d**2))
    numerator = float(np.sum((residual_2d @ h) ** 2))
    isotropic = numerator / denominator if denominator > 0.0 else float("nan")

    centered = residual_2d - residual_2d.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    eigenvalues = singular_values**2
    eigen_sum = float(eigenvalues.sum())
    pc1 = float(eigenvalues[0] / eigen_sum) if eigen_sum > 0.0 else float("nan")
    effective_rank = float(eigen_sum**2 / np.sum(eigenvalues**2)) if eigen_sum > 0.0 else float("nan")
    valid = bool(np.all(np.isfinite(logs)))
    return {
        "setting": setting,
        "status": "VALID" if valid else "INVALID",
        "within_trial_dispersion": within,
        "between_trial_dispersion": between,
        "within_to_between_ratio": within / between if between > 0.0 else float("inf"),
        "ratio_within": ratio_within,
        "isotropic_fraction": isotropic,
        "traceless_shape_fraction": 1.0 - isotropic,
        "pca_effective_rank": effective_rank,
        "pc1_fraction": pc1,
        "descriptive_only": True,
        "stop_rule": "numerical_invalidity_only",
    }

