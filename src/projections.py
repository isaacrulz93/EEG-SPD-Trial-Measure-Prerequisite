"""Fixed Frobenius-uniform symmetric banks and trial-measure features."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .geometry import svec, unsvec


BANK_SEEDS = (0, 1, 2, 3, 4)
N_DIRECTIONS = 200


def sample_direction_bank(d: int, seed: int, n_directions: int = N_DIRECTIONS) -> np.ndarray:
    """Sample Frobenius-uniform directions through isotropic svec coordinates."""

    dimension = d * (d + 1) // 2
    rng = np.random.default_rng(int(seed))
    vectors = rng.standard_normal((int(n_directions), dimension))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return unsvec(vectors, d)


def direction_vectors(d: int, seed: int, n_directions: int = N_DIRECTIONS) -> np.ndarray:
    return svec(sample_direction_bank(d, seed, n_directions))


def bank_sha256(bank: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bank).view(np.uint8)).hexdigest()


def project_window_logs(window_logs: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Project to shape (trial, direction, window)."""

    log_vectors = svec(window_logs)
    bank_vectors = svec(bank)
    return np.einsum("nwm,lm->nlw", log_vectors, bank_vectors, optimize=True)


def moment_features(projections: np.ndarray) -> np.ndarray:
    projections = np.asarray(projections)
    return np.concatenate((projections.mean(axis=-1), projections.std(axis=-1, ddof=0)), axis=1)


def sorted_projection_features(projections: np.ndarray) -> np.ndarray:
    """Exact L x W order statistics, flattened without reference subtraction."""

    projections = np.asarray(projections)
    ordered = np.sort(projections, axis=-1)
    return ordered.reshape(len(ordered), -1)


def precompute_projection_cache(
    window_logs: dict[tuple[str, str], np.ndarray],
    output_root: str | Path,
    *,
    bank_seeds: tuple[int, ...] = BANK_SEEDS,
    n_directions: int = N_DIRECTIONS,
    batch_size: int = 256,
) -> dict:
    """Precompute fixed banks once and reuse them for all folds/inference calls."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "COMPLETE.json"
    if marker.exists():
        return json.loads(marker.read_text())
    started = time.perf_counter()
    first = next(iter(window_logs.values()))
    d = int(first.shape[-1])
    n_trials = int(first.shape[0])
    audit: dict[str, dict] = {}
    for seed in bank_seeds:
        bank = sample_direction_bank(d, seed, n_directions)
        bank_path = output_root / f"bank_seed{seed}.npy"
        np.save(bank_path, bank)
        audit[str(seed)] = {
            "sha256": bank_sha256(bank),
            "shape": list(bank.shape),
            "frobenius_norm_min": float(np.min(np.linalg.norm(bank, axis=(-2, -1)))),
            "frobenius_norm_max": float(np.max(np.linalg.norm(bank, axis=(-2, -1)))),
        }
        for (ra_mode, setting), logs in window_logs.items():
            n_windows = int(logs.shape[1])
            path = output_root / f"{ra_mode}_{setting}_seed{seed}.npy"
            target = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(n_trials, n_directions, n_windows)
            )
            for start in range(0, n_trials, batch_size):
                stop = min(n_trials, start + batch_size)
                target[start:stop] = project_window_logs(logs[start:stop], bank).astype(np.float32)
            target.flush()
            del target
    elapsed = time.perf_counter() - started
    metadata = {
        "n_directions": int(n_directions),
        "bank_seeds": list(map(int, bank_seeds)),
        "fixed_across_all_folds_and_inference": True,
        "bank_audit": audit,
        "projection_feature_precomputation_time_seconds": elapsed,
        "stored_projection_dtype": "float32",
    }
    marker.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_projection_cache(root: str | Path, ra_mode: str, setting: str, seed: int) -> np.ndarray:
    return np.load(Path(root) / f"{ra_mode}_{setting}_seed{seed}.npy", mmap_mode="r")

