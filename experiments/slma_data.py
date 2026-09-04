"""MOABB covariance loading and split builders for the SLMA TTA pilot.

Vendored byte-for-byte in substance from ``isaacrulz93/T3DA`` commit
``7d0b8b1757c8284a6e9a641487d3cf6106073dc5``.  The prerequisite experiment
imports ``_import_moabb_dataset`` and ``_metadata_arrays`` from this module so
there is exactly one authoritative MOABB dataset/paradigm construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_CONFIGS = {
    "BNCI2014_001": {"n_classes": 4, "fmin": 8.0, "fmax": 32.0},
    "BNCI2015_001": {"n_classes": 2, "fmin": 8.0, "fmax": 32.0},
}


@dataclass(frozen=True)
class CovDataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    session_raw: np.ndarray
    n_classes: int
    input_dim: int
    dataset_name: str


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    scenario: str
    source_subjects: list[int]
    target_subject: int
    source_sessions: list[int]
    target_sessions: list[int]
    source_indices: np.ndarray
    target_indices: np.ndarray


def covariance_matrices(
    x: np.ndarray,
    *,
    eps: float = 1e-6,
    trace_norm: bool = True,
) -> np.ndarray:
    """Convert epoched EEG (trials, channels, time) into SPD covariances."""

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected EEG epochs with shape (trials, channels, time), got {x.shape}.")

    n_trials, n_channels, _ = x.shape
    covs = np.empty((n_trials, n_channels, n_channels), dtype=np.float64)
    eye = np.eye(n_channels, dtype=np.float64)
    for idx in range(n_trials):
        trial = x[idx] - x[idx].mean(axis=1, keepdims=True)
        denom = max(1, trial.shape[1] - 1)
        cov = (trial @ trial.T) / denom
        cov = 0.5 * (cov + cov.T)
        if trace_norm:
            trace = float(np.trace(cov))
            if trace > 0.0:
                cov = cov / trace
        covs[idx] = cov + eps * eye
    return covs


def _import_moabb_dataset(dataset_name: str):
    import importlib

    from moabb.paradigms import MotorImagery

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset {dataset_name}. Use one of {sorted(DATASET_CONFIGS)}.")
    dataset_cls = getattr(importlib.import_module("moabb.datasets"), dataset_name)
    cfg = DATASET_CONFIGS[dataset_name]
    dataset = dataset_cls()
    paradigm = MotorImagery(n_classes=cfg["n_classes"], fmin=cfg["fmin"], fmax=cfg["fmax"])
    return dataset, paradigm


def _factorize_sessions_per_subject(subject: np.ndarray, session_raw: np.ndarray) -> np.ndarray:
    session = np.zeros(len(subject), dtype=np.int64)
    for sub in np.unique(subject):
        mapping: dict[str, int] = {}
        indices = np.flatnonzero(subject == sub)
        for idx in indices:
            raw = str(session_raw[idx])
            if raw not in mapping:
                mapping[raw] = len(mapping)
            session[idx] = mapping[raw]
    return session


def _metadata_arrays(metadata, n_samples: int, subjects: Iterable[int] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = set(getattr(metadata, "columns", []))

    if "subject" in columns:
        subject = np.asarray(metadata["subject"], dtype=np.int64)
    else:
        subject_list = list(subjects or [])
        if len(subject_list) == 1:
            subject = np.full(n_samples, int(subject_list[0]), dtype=np.int64)
        else:
            print("[SLMA] Warning: MOABB metadata has no subject column; assigning subject=0.")
            subject = np.zeros(n_samples, dtype=np.int64)

    if "session" in columns:
        session_raw = np.asarray(metadata["session"].astype(str))
        session = _factorize_sessions_per_subject(subject, session_raw)
    else:
        session_raw = np.asarray([""] * n_samples, dtype=str)
        session = np.zeros(n_samples, dtype=np.int64)

    return subject, session, session_raw


def _cache_path(dataset_name: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{dataset_name}_f8_32_cov.npz"


def load_cov_dataset(
    dataset_name: str,
    *,
    cache_dir: str | Path = "data_cache/slma_covariances",
    subjects: Iterable[int] | None = None,
    eps: float = 1e-6,
    trace_norm: bool = True,
) -> CovDataset:
    """Load or create cached MOABB covariance matrices for the SLMA pilot."""

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset {dataset_name}. Use one of {sorted(DATASET_CONFIGS)}.")

    cache_path = _cache_path(dataset_name, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        X = data["X"].astype(np.float64, copy=False)
        y = data["y"].astype(np.int64, copy=False)
        subject = data["subject"].astype(np.int64, copy=False)
        session = data["session"].astype(np.int64, copy=False)
        session_raw = data["session_raw"].astype(str, copy=False)
    else:
        dataset, paradigm = _import_moabb_dataset(dataset_name)
        all_subjects = getattr(dataset, "subject_list", None)
        eeg, raw_labels, metadata = paradigm.get_data(dataset=dataset, subjects=all_subjects)
        y_values, y = np.unique(np.asarray(raw_labels), return_inverse=True)
        X = covariance_matrices(eeg, eps=eps, trace_norm=trace_norm)
        subject, session, session_raw = _metadata_arrays(metadata, len(y), all_subjects)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y.astype(np.int64),
            y_raw=np.asarray(y_values).astype(str),
            subject=subject.astype(np.int64),
            session=session.astype(np.int64),
            session_raw=session_raw.astype(str),
        )

    if subjects is not None:
        wanted = np.asarray([int(s) for s in subjects], dtype=np.int64)
        keep = np.isin(subject, wanted)
        X = X[keep]
        y = y[keep]
        subject = subject[keep]
        session = session[keep]
        session_raw = session_raw[keep]

    if X.size == 0:
        raise ValueError(f"No samples loaded for {dataset_name} with subjects={subjects}.")

    return CovDataset(
        X=X,
        y=y,
        subject=subject,
        session=session,
        session_raw=session_raw,
        n_classes=int(DATASET_CONFIGS[dataset_name]["n_classes"]),
        input_dim=int(X.shape[-1]),
        dataset_name=dataset_name,
    )


def _limit_folds(folds: list[FoldSpec], max_folds: int | None) -> list[FoldSpec]:
    if max_folds is None or max_folds <= 0:
        return folds
    return folds[: int(max_folds)]


def make_folds(
    dataset: CovDataset,
    scenario: str,
    subjects: Iterable[int] | None = None,
    max_folds: int | None = None,
) -> list[FoldSpec]:
    """Build transductive source/target folds."""

    scenario = str(scenario)
    if scenario not in {"cross_session", "cross_subject", "loso"}:
        raise ValueError("scenario must be one of cross_session, cross_subject, or loso.")

    available_subjects = sorted(int(s) for s in np.unique(dataset.subject))
    if subjects is not None:
        wanted = {int(s) for s in subjects}
        available_subjects = [s for s in available_subjects if s in wanted]

    folds: list[FoldSpec] = []
    if scenario == "cross_session":
        for subject in available_subjects:
            subject_mask = dataset.subject == subject
            sessions = sorted(int(s) for s in np.unique(dataset.session[subject_mask]))
            if len(sessions) < 2:
                continue
            source_session, target_session = sessions[0], sessions[1]
            source_idx = np.flatnonzero(subject_mask & (dataset.session == source_session))
            target_idx = np.flatnonzero(subject_mask & (dataset.session == target_session))
            if len(source_idx) == 0 or len(target_idx) == 0:
                continue
            folds.append(FoldSpec(len(folds), scenario, [subject], subject, [source_session], [target_session], source_idx, target_idx))
        return _limit_folds(folds, max_folds)

    if scenario == "cross_subject":
        for source_subject in available_subjects:
            for target_subject in available_subjects:
                if source_subject == target_subject:
                    continue
                source_idx = np.flatnonzero(dataset.subject == source_subject)
                target_idx = np.flatnonzero(dataset.subject == target_subject)
                if len(source_idx) == 0 or len(target_idx) == 0:
                    continue
                folds.append(FoldSpec(len(folds), scenario, [source_subject], target_subject,
                    sorted(int(s) for s in np.unique(dataset.session[source_idx])),
                    sorted(int(s) for s in np.unique(dataset.session[target_idx])), source_idx, target_idx))
                if max_folds is not None and max_folds > 0 and len(folds) >= max_folds:
                    return folds
        return folds

    for target_subject in available_subjects:
        source_subjects = [s for s in available_subjects if s != target_subject]
        source_idx = np.flatnonzero(np.isin(dataset.subject, source_subjects))
        target_idx = np.flatnonzero(dataset.subject == target_subject)
        if len(source_idx) == 0 or len(target_idx) == 0:
            continue
        folds.append(FoldSpec(len(folds), scenario, source_subjects, target_subject,
            sorted(int(s) for s in np.unique(dataset.session[source_idx])),
            sorted(int(s) for s in np.unique(dataset.session[target_idx])), source_idx, target_idx))
    return _limit_folds(folds, max_folds)

