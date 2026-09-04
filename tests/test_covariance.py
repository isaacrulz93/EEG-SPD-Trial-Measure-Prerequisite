import numpy as np
from sklearn.covariance import OAS

from src.covariance import oas_trace_covariance, window_starts


def test_oas_trace_eps_matches_sklearn():
    rng = np.random.default_rng(12)
    epoch = rng.normal(size=(5, 7, 101))
    actual = oas_trace_covariance(epoch)
    expected = []
    for trial in epoch:
        covariance = OAS().fit(trial.T).covariance_
        covariance = covariance / np.trace(covariance)
        covariance = covariance + 1e-6 * np.eye(7)
        expected.append(0.5 * (covariance + covariance.T))
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(np.trace(actual, axis1=-2, axis2=-1), 1.0 + 7e-6)


def test_exact_window_counts_use_full_epoch_samples():
    np.testing.assert_array_equal(window_starts(1001, 250.0, 1.0, 0.5), [0, 125, 250, 375, 500, 625, 750])
    np.testing.assert_array_equal(window_starts(1001, 250.0, 1.0, 1.0), [0, 250, 500, 750])

