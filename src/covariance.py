"""Primary OAS covariance estimator and full-epoch windowing."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .geometry import symmetrize


def window_starts(n_times: int, sfreq: float, window_seconds: float, hop_seconds: float) -> np.ndarray:
    window_samples = int(round(float(window_seconds) * float(sfreq)))
    hop_samples = int(round(float(hop_seconds) * float(sfreq)))
    if window_samples <= 1 or hop_samples <= 0:
        raise ValueError("Window and hop must map to positive sample counts.")
    if n_times < window_samples:
        raise ValueError(f"Epoch has {n_times} samples, fewer than window length {window_samples}.")
    return np.arange(0, n_times - window_samples + 1, hop_samples, dtype=np.int64)


def oas_trace_covariance(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """OAS, trace division, ``eps I``, then symmetrization.

    The vectorized formula matches ``sklearn.covariance.OAS`` with
    ``assume_centered=False``; its empirical covariance uses denominator ``n``.
    """

    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("Expected (..., channels, time).")
    n_samples = x.shape[-1]
    n_features = x.shape[-2]
    centered = x - x.mean(axis=-1, keepdims=True)
    empirical = np.einsum("...it,...jt->...ij", centered, centered, optimize=True) / float(n_samples)
    alpha = np.mean(empirical**2, axis=(-2, -1))
    mu = np.trace(empirical, axis1=-2, axis2=-1) / float(n_features)
    denominator = (n_samples + 1.0) * (alpha - mu**2 / float(n_features))
    numerator = alpha + mu**2
    shrinkage = np.ones_like(mu)
    np.divide(numerator, denominator, out=shrinkage, where=denominator != 0.0)
    shrinkage = np.minimum(shrinkage, 1.0)
    covariance = (1.0 - shrinkage[..., None, None]) * empirical
    diagonal = np.arange(n_features)
    covariance[..., diagonal, diagonal] += shrinkage[..., None] * mu[..., None]

    trace = np.trace(covariance, axis1=-2, axis2=-1)
    if np.any(~np.isfinite(trace)) or np.any(trace <= 0.0):
        raise FloatingPointError("OAS produced a non-positive or non-finite trace.")
    covariance = covariance / trace[..., None, None]
    covariance[..., diagonal, diagonal] += float(eps)
    return symmetrize(covariance)


def covariance_chunks(
    epochs: np.ndarray,
    selections: Iterable[slice] | None = None,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute primary covariances, optionally from time selections."""

    epochs = np.asarray(epochs)
    if selections is None:
        return oas_trace_covariance(epochs, eps=eps)
    pieces = [oas_trace_covariance(epochs[..., selection], eps=eps) for selection in selections]
    return np.stack(pieces, axis=1)


def trace_normalized_scm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Non-gating sensitivity estimator retained only for explicit comparisons."""

    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=-1, keepdims=True)
    covariance = np.einsum("...it,...jt->...ij", centered, centered, optimize=True) / max(1, x.shape[-1] - 1)
    covariance /= np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
    diagonal = np.arange(x.shape[-2])
    covariance[..., diagonal, diagonal] += float(eps)
    return symmetrize(covariance)

