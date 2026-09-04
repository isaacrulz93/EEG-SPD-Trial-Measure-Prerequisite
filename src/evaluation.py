"""Classical baselines, moment control, pooled-measure P1, and quantile P2."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .geometry import airm_mean, airm_squared_distances, svec, tangent_features
from .projections import moment_features, sorted_projection_features


@dataclass(frozen=True)
class Prediction:
    y_pred: np.ndarray
    fit_seconds: float
    inference_seconds: float


def shrinkage_lda():
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def lda_prediction(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> Prediction:
    started = time.perf_counter()
    model = shrinkage_lda().fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    y_pred = model.predict(x_test)
    inference_seconds = time.perf_counter() - started
    return Prediction(y_pred, fit_seconds, inference_seconds)


def b1_tangent_lda(train_cov: np.ndarray, y_train: np.ndarray, test_cov: np.ndarray) -> Prediction:
    started = time.perf_counter()
    x_train, reference = tangent_features(train_cov)
    model = shrinkage_lda().fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    x_test, _ = tangent_features(test_cov, reference)
    y_pred = model.predict(x_test)
    return Prediction(y_pred, fit_seconds, time.perf_counter() - started)


def b2_airm_mdm(train_cov: np.ndarray, y_train: np.ndarray, test_cov: np.ndarray) -> Prediction:
    started = time.perf_counter()
    classes = np.unique(y_train)
    centers = np.stack([airm_mean(train_cov[y_train == value]) for value in classes])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    distances = airm_squared_distances(test_cov, centers)
    y_pred = classes[np.argmin(distances, axis=1)]
    return Prediction(y_pred, fit_seconds, time.perf_counter() - started)


def b3_window_tangent_lda(
    train_cov: np.ndarray,
    y_train: np.ndarray,
    test_cov: np.ndarray,
) -> tuple[Prediction, Prediction]:
    n_train, n_windows = train_cov.shape[:2]
    d = train_cov.shape[-1]
    started = time.perf_counter()
    x_train, reference = tangent_features(train_cov.reshape(-1, d, d))
    window_y = np.repeat(y_train, n_windows)
    model = shrinkage_lda().fit(x_train, window_y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    x_test, _ = tangent_features(test_cov.reshape(-1, d, d), reference)
    classes = model.classes_
    decision = model.decision_function(x_test)
    window_predictions = model.predict(x_test).reshape(len(test_cov), n_windows)
    if decision.ndim == 1:
        score = decision.reshape(len(test_cov), n_windows).mean(axis=1)
        mean_pred = np.where(score > 0.0, classes[1], classes[0])
    else:
        score = decision.reshape(len(test_cov), n_windows, len(classes)).mean(axis=1)
        mean_pred = classes[np.argmax(score, axis=1)]
    majority_pred = np.asarray([
        classes[np.argmax([(row == value).sum() for value in classes])] for row in window_predictions
    ])
    inference_seconds = time.perf_counter() - started
    return (
        Prediction(mean_pred, fit_seconds, inference_seconds),
        Prediction(majority_pred, 0.0, 0.0),
    )


def b4_mean_log_features(window_logs: np.ndarray) -> np.ndarray:
    return svec(window_logs).mean(axis=1)


def ordered_tangent_lda(
    train_cov: np.ndarray,
    y_train: np.ndarray,
    test_cov: np.ndarray,
) -> Prediction:
    """Non-gating ordered, non-overlapping-window interpretation control."""

    n_train, n_windows, d, _ = train_cov.shape
    started = time.perf_counter()
    train_tangent, reference = tangent_features(train_cov.reshape(-1, d, d))
    train_tangent = train_tangent.reshape(n_train, n_windows, -1).reshape(n_train, -1)
    model = shrinkage_lda().fit(train_tangent, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    test_tangent, _ = tangent_features(test_cov.reshape(-1, d, d), reference)
    test_tangent = test_tangent.reshape(len(test_cov), n_windows, -1).reshape(len(test_cov), -1)
    pred = model.predict(test_tangent)
    return Prediction(pred, fit_seconds, time.perf_counter() - started)


def p1_pooled_class_measure(
    train_projections: np.ndarray,
    y_train: np.ndarray,
    test_projections: np.ndarray,
) -> Prediction:
    """Nearest direct fixed-bank SPDSW pooled class measure diagnostic."""

    started = time.perf_counter()
    classes = np.unique(y_train)
    n_windows = int(train_projections.shape[-1])
    pooled: list[np.ndarray] = []
    for value in classes:
        values = np.transpose(train_projections[y_train == value], (1, 0, 2)).reshape(train_projections.shape[1], -1)
        pooled.append(np.sort(values, axis=1))
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    ordered_test = np.sort(test_projections, axis=-1)
    distances = np.empty((len(test_projections), len(classes)), dtype=np.float64)
    for column, source in enumerate(pooled):
        if source.shape[1] % n_windows != 0:
            raise ValueError("Pooled support must be divisible by the exact test window count.")
        repeated = source.reshape(source.shape[0], n_windows, -1)
        distances[:, column] = np.mean((ordered_test[..., None] - repeated[None, ...]) ** 2, axis=(1, 2, 3))
    pred = classes[np.argmin(distances, axis=1)]
    return Prediction(pred, fit_seconds, time.perf_counter() - started)


def metrics(y_true: np.ndarray, prediction: Prediction) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, prediction.y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction.y_pred)),
        "classifier_or_pooled_measure_fitting_time_seconds": float(prediction.fit_seconds),
        "inference_time_seconds": float(prediction.inference_seconds),
        "amortized_inference_seconds_per_trial": float(prediction.inference_seconds / max(1, len(y_true))),
    }


def feature_methods(projections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return moment_features(projections), sorted_projection_features(projections)
