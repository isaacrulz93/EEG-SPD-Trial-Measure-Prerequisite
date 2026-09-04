"""Disk-backed per-subject AIRM recentering for the LOSO calibration branch."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .data import CacheData, WINDOWS
from .geometry import airm_mean, congruence, matrix_invsqrt, matrix_log


def build_subject_ra_cache(data: CacheData, batch_size: int = 128) -> dict:
    """Use each subject's full unlabeled trial set to fit its own RA transform."""

    root = data.root / "subject_ra"
    marker = root / "COMPLETE.json"
    if marker.exists():
        return json.loads(marker.read_text())
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Incomplete RA cache exists at {root}; move it aside before retrying.")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n, d, _ = data.full_cov.shape
    full_cov = np.lib.format.open_memmap(root / "full_cov.npy", mode="w+", dtype=np.float64, shape=data.full_cov.shape)
    full_log = np.lib.format.open_memmap(root / "full_log.npy", mode="w+", dtype=np.float64, shape=data.full_log.shape)
    window_cov = {
        setting: np.lib.format.open_memmap(root / f"{setting}_cov.npy", mode="w+", dtype=np.float64, shape=value.shape)
        for setting, value in data.window_cov.items()
    }
    window_log = {
        setting: np.lib.format.open_memmap(root / f"{setting}_log.npy", mode="w+", dtype=np.float64, shape=value.shape)
        for setting, value in data.window_log.items()
    }
    transform_audit: dict[str, dict] = {}
    for subject_value in np.unique(data.subject):
        indices = np.flatnonzero(data.subject == subject_value)
        reference = airm_mean(np.asarray(data.full_cov[indices]))
        transform = matrix_invsqrt(reference)
        transform_audit[str(int(subject_value))] = {
            "n_unlabeled_trials": int(len(indices)),
            "reference_min_eigenvalue": float(np.linalg.eigvalsh(reference).min()),
        }
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            transformed = congruence(np.asarray(data.full_cov[batch_indices]), transform)
            full_cov[batch_indices] = transformed
            full_log[batch_indices] = matrix_log(transformed)
            for setting in WINDOWS:
                transformed = congruence(np.asarray(data.window_cov[setting][batch_indices]), transform)
                window_cov[setting][batch_indices] = transformed
                window_log[setting][batch_indices] = matrix_log(transformed)
    full_cov.flush()
    full_log.flush()
    for value in (*window_cov.values(), *window_log.values()):
        value.flush()
    elapsed = time.perf_counter() - started
    metadata = {
        "mode": "subject_ra",
        "label": "TRANSDUCTIVE CALIBRATION",
        "uses_target_subject_own_unlabeled_trials": True,
        "fully_inductive": False,
        "reference_source": "all full-trial covariances within each subject; labels unused",
        "subjects": transform_audit,
        "ra_covariance_log_time_seconds": elapsed,
    }
    marker.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_subject_ra_arrays(data: CacheData, mmap_mode: str = "r") -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = data.root / "subject_ra"
    if not (root / "COMPLETE.json").exists():
        raise FileNotFoundError("Subject-RA cache has not been generated.")
    return (
        np.load(root / "full_cov.npy", mmap_mode=mmap_mode),
        np.load(root / "full_log.npy", mmap_mode=mmap_mode),
        {setting: np.load(root / f"{setting}_cov.npy", mmap_mode=mmap_mode) for setting in WINDOWS},
        {setting: np.load(root / f"{setting}_log.npy", mmap_mode=mmap_mode) for setting in WINDOWS},
    )

