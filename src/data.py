"""Single-path MOABB loading and disk-backed covariance/log cache creation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.slma_data import DATASET_CONFIGS, _import_moabb_dataset, _metadata_arrays

from .covariance import oas_trace_covariance, window_starts
from .geometry import matrix_log


WINDOWS = {
    "W1": {"window_seconds": 1.0, "hop_seconds": 0.5, "primary": True},
    "W2": {"window_seconds": 1.0, "hop_seconds": 1.0, "primary": False},
}
ESTIMATOR_NAME = "oas_trace_eps1e-6"


@dataclass(frozen=True)
class CacheData:
    root: Path
    metadata: dict
    full_cov: np.ndarray
    full_log: np.ndarray
    window_cov: dict[str, np.ndarray]
    window_log: dict[str, np.ndarray]
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    session_raw: np.ndarray
    trial_id: np.ndarray


def print_cache_shape_metadata(metadata: dict) -> None:
    print(f"raw epoch tensor shape: {tuple(metadata['raw_epoch_shape'])}")
    print(f"sampling frequency: {metadata['sampling_frequency_hz']:.6f} Hz")
    print(
        f"actual epoch duration: {metadata['actual_epoch_duration_seconds']:.6f} s "
        f"(sample span {metadata['sample_span_seconds']:.6f} s)"
    )
    for setting in WINDOWS:
        window = metadata["windows"][setting]
        print(
            f"{setting} exact windows per trial: {window['windows_per_trial']}; "
            f"total windows: {window['total_windows']}"
        )


def infer_sfreq(dataset, n_times: int) -> float:
    """Infer the preserved MOABB rate from its epoch interval and endpoint grid."""

    interval = np.asarray(getattr(dataset, "interval", []), dtype=np.float64)
    if interval.shape != (2,) or interval[1] <= interval[0]:
        raise ValueError("MOABB dataset does not expose a valid two-point epoch interval.")
    rate = (int(n_times) - 1) / float(interval[1] - interval[0])
    rounded = float(round(rate))
    if not np.isclose(rate, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(f"Could not infer an integer sampling rate: {rate} Hz.")
    return rounded


def _write_array(path: Path, value: np.ndarray) -> None:
    target = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype, shape=value.shape)
    target[...] = value
    target.flush()
    del target


def _batch_slices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield slice(start, min(length, start + batch_size))


def build_covariance_log_cache(
    cache_root: str | Path,
    *,
    dataset_name: str = "BNCI2014_001",
    eps: float = 1e-6,
    batch_size: int = 128,
) -> dict:
    """Generate the cache and discard raw epochs only after all covariances exist."""

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset {dataset_name}.")
    root = Path(cache_root) / dataset_name / ESTIMATOR_NAME
    marker = root / "COMPLETE.json"
    if marker.exists():
        metadata = json.loads(marker.read_text())
        print_cache_shape_metadata(metadata)
        print(f"covariance/log-cache time: {metadata['covariance_log_cache_time_seconds']:.3f} s (cached)")
        return metadata
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Incomplete cache exists at {root}; move it aside before retrying.")
    root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    dataset, paradigm = _import_moabb_dataset(dataset_name)
    all_subjects = getattr(dataset, "subject_list", None)
    # This is deliberately the exact call path from experiments/slma_data.py.
    epochs, raw_labels, frame = paradigm.get_data(dataset=dataset, subjects=all_subjects)
    epochs = np.asarray(epochs)
    if epochs.ndim != 3:
        raise ValueError(f"Expected raw epoch tensor (trial,channel,time), got {epochs.shape}.")
    n_trials, d, n_times = map(int, epochs.shape)
    sfreq = infer_sfreq(dataset, n_times)
    actual_duration = n_times / sfreq
    sample_span = (n_times - 1) / sfreq
    label_values, y = np.unique(np.asarray(raw_labels), return_inverse=True)
    subject, session, session_raw = _metadata_arrays(frame, n_trials, all_subjects)
    trial_id = np.arange(n_trials, dtype=np.int64)

    full_cov = np.lib.format.open_memmap(root / "full_cov.npy", mode="w+", dtype=np.float64, shape=(n_trials, d, d))
    full_log = np.lib.format.open_memmap(root / "full_log.npy", mode="w+", dtype=np.float64, shape=(n_trials, d, d))

    window_arrays: dict[str, tuple[np.memmap, np.memmap, np.ndarray, int]] = {}
    window_meta: dict[str, dict] = {}
    for setting, spec in WINDOWS.items():
        starts = window_starts(n_times, sfreq, spec["window_seconds"], spec["hop_seconds"])
        window_samples = int(round(spec["window_seconds"] * sfreq))
        count = int(len(starts))
        cov = np.lib.format.open_memmap(root / f"{setting}_cov.npy", mode="w+", dtype=np.float64, shape=(n_trials, count, d, d))
        log = np.lib.format.open_memmap(root / f"{setting}_log.npy", mode="w+", dtype=np.float64, shape=(n_trials, count, d, d))
        window_arrays[setting] = (cov, log, starts, window_samples)
        window_meta[setting] = {
            **spec,
            "window_samples": window_samples,
            "hop_samples": int(round(spec["hop_seconds"] * sfreq)),
            "starts": starts.tolist(),
            "windows_per_trial": count,
            "total_windows": n_trials * count,
        }

    for batch in _batch_slices(n_trials, batch_size):
        covariance = oas_trace_covariance(epochs[batch], eps=eps)
        full_cov[batch] = covariance
        full_log[batch] = matrix_log(covariance)
        for cov, log, starts, window_samples in window_arrays.values():
            for column, start in enumerate(starts):
                covariance = oas_trace_covariance(epochs[batch, :, start : start + window_samples], eps=eps)
                cov[batch, column] = covariance
                log[batch, column] = matrix_log(covariance)

    full_cov.flush()
    full_log.flush()
    for cov, log, _, _ in window_arrays.values():
        cov.flush()
        log.flush()
    _write_array(root / "y.npy", y.astype(np.int64))
    _write_array(root / "subject.npy", subject.astype(np.int64))
    _write_array(root / "session.npy", session.astype(np.int64))
    _write_array(root / "session_raw.npy", session_raw.astype("U64"))
    _write_array(root / "trial_id.npy", trial_id)

    elapsed = time.perf_counter() - started
    cache_metadata = {
        "dataset": dataset_name,
        "raw_epoch_shape": list(epochs.shape),
        "sampling_frequency_hz": sfreq,
        "actual_epoch_duration_seconds": actual_duration,
        "sample_span_seconds": sample_span,
        "dataset_interval": np.asarray(dataset.interval, dtype=float).tolist(),
        "n_classes": int(len(label_values)),
        "class_names": label_values.astype(str).tolist(),
        "subjects": sorted(int(value) for value in np.unique(subject)),
        "sessions_per_subject": {str(int(value)): int(len(np.unique(session[subject == value]))) for value in np.unique(subject)},
        "covariance_estimator": ESTIMATOR_NAME,
        "estimator_steps": ["OAS", "divide_by_trace", "add_1e-6_I", "symmetrize"],
        "epsilon": float(eps),
        "windows": window_meta,
        "covariance_log_cache_time_seconds": elapsed,
        "moabb_factory": "experiments.slma_data._import_moabb_dataset",
        "raw_epochs_discarded_after_cache": True,
    }
    (root / "metadata.json").write_text(json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n")
    marker.write_text(json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n")
    print_cache_shape_metadata(cache_metadata)
    print(f"covariance/log-cache time: {elapsed:.3f} s")
    del epochs
    return cache_metadata


def load_cache(cache_root: str | Path, dataset_name: str = "BNCI2014_001", mmap_mode: str = "r") -> CacheData:
    root = Path(cache_root) / dataset_name / ESTIMATOR_NAME
    marker = root / "COMPLETE.json"
    if not marker.exists():
        raise FileNotFoundError(f"Complete cache not found: {marker}")
    metadata = json.loads(marker.read_text())
    return CacheData(
        root=root,
        metadata=metadata,
        full_cov=np.load(root / "full_cov.npy", mmap_mode=mmap_mode),
        full_log=np.load(root / "full_log.npy", mmap_mode=mmap_mode),
        window_cov={setting: np.load(root / f"{setting}_cov.npy", mmap_mode=mmap_mode) for setting in WINDOWS},
        window_log={setting: np.load(root / f"{setting}_log.npy", mmap_mode=mmap_mode) for setting in WINDOWS},
        y=np.load(root / "y.npy", mmap_mode=mmap_mode),
        subject=np.load(root / "subject.npy", mmap_mode=mmap_mode),
        session=np.load(root / "session.npy", mmap_mode=mmap_mode),
        session_raw=np.load(root / "session_raw.npy", mmap_mode=mmap_mode),
        trial_id=np.load(root / "trial_id.npy", mmap_mode=mmap_mode),
    )


def validate_cache(data: CacheData) -> dict:
    arrays = [data.full_cov, data.full_log, *data.window_cov.values(), *data.window_log.values()]
    finite = all(bool(np.all(np.isfinite(array))) for array in arrays)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(data.full_cov)))
    for value in data.window_cov.values():
        min_eigenvalue = min(min_eigenvalue, float(np.min(np.linalg.eigvalsh(value))))
    valid = finite and min_eigenvalue > 0.0
    return {"status": "VALID" if valid else "INVALID", "finite": finite, "minimum_eigenvalue": min_eigenvalue}
