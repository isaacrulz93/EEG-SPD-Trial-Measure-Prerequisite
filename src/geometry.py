"""Small, explicit SPD geometry routines used by the diagnostic suite."""

from __future__ import annotations

import numpy as np
from pyriemann.utils.mean import mean_riemann


def symmetrize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (x + np.swapaxes(x, -1, -2))


def _spectral_function(x: np.ndarray, fn) -> np.ndarray:
    values, vectors = np.linalg.eigh(symmetrize(x))
    transformed = fn(values)
    return symmetrize(np.einsum("...ik,...k,...jk->...ij", vectors, transformed, vectors, optimize=True))


def matrix_log(x: np.ndarray) -> np.ndarray:
    return _spectral_function(x, lambda values: np.log(np.maximum(values, np.finfo(np.float64).tiny)))


def matrix_invsqrt(x: np.ndarray) -> np.ndarray:
    return _spectral_function(x, lambda values: 1.0 / np.sqrt(np.maximum(values, np.finfo(np.float64).tiny)))


def matrix_sqrt(x: np.ndarray) -> np.ndarray:
    return _spectral_function(x, lambda values: np.sqrt(np.maximum(values, 0.0)))


def svec(x: np.ndarray) -> np.ndarray:
    """Frobenius-isometric vectorization of symmetric matrices."""

    x = symmetrize(x)
    d = x.shape[-1]
    i, j = np.triu_indices(d)
    result = x[..., i, j].copy()
    result[..., i != j] *= np.sqrt(2.0)
    return result


def unsvec(x: np.ndarray, d: int) -> np.ndarray:
    """Inverse of :func:`svec`."""

    x = np.asarray(x, dtype=np.float64)
    expected = d * (d + 1) // 2
    if x.shape[-1] != expected:
        raise ValueError(f"Expected final dimension {expected}, got {x.shape[-1]}.")
    result = np.zeros((*x.shape[:-1], d, d), dtype=np.float64)
    i, j = np.triu_indices(d)
    values = x.copy()
    values[..., i != j] /= np.sqrt(2.0)
    result[..., i, j] = values
    result[..., j, i] = values
    return result


def airm_mean(covariances: np.ndarray) -> np.ndarray:
    covariances = np.asarray(covariances, dtype=np.float64)
    if covariances.ndim != 3 or len(covariances) == 0:
        raise ValueError("AIRM mean needs a non-empty (n,d,d) covariance array.")
    return symmetrize(mean_riemann(covariances, tol=1e-8, maxiter=100))


def congruence(covariances: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return symmetrize(np.einsum("ij,...jk,lk->...il", transform, covariances, transform, optimize=True))


def subject_airm_recenter(covariances: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Recenter each subject using all of that subject's unlabeled trials.

    No label argument is accepted.  Applying this to a held-out target subject is
    therefore explicitly transductive calibration.
    """

    covariances = np.asarray(covariances, dtype=np.float64)
    subject = np.asarray(subject)
    result = np.empty_like(covariances)
    transforms: dict[int, np.ndarray] = {}
    for value in np.unique(subject):
        indices = np.flatnonzero(subject == value)
        transform = matrix_invsqrt(airm_mean(covariances[indices]))
        transforms[int(value)] = transform
        result[indices] = congruence(covariances[indices], transform)
    return result, transforms


def tangent_features(covariances: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """AIRM tangent features, fitting the reference only from supplied matrices."""

    covariances = np.asarray(covariances, dtype=np.float64)
    reference = airm_mean(covariances) if reference is None else np.asarray(reference, dtype=np.float64)
    whiten = matrix_invsqrt(reference)
    tangent = matrix_log(congruence(covariances, whiten))
    return svec(tangent), reference


def airm_squared_distances(covariances: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return (n_samples, n_centers) squared AIRM distances."""

    covariances = np.asarray(covariances, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    result = np.empty((len(covariances), len(centers)), dtype=np.float64)
    for column, center in enumerate(centers):
        whitened = congruence(covariances, matrix_invsqrt(center))
        result[:, column] = np.sum(matrix_log(whitened) ** 2, axis=(-2, -1))
    return result

