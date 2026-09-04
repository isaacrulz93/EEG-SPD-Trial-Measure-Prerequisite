import numpy as np

from src.projections import (
    bank_sha256,
    moment_features,
    sample_direction_bank,
    sorted_projection_features,
)


def test_bank_is_fixed_deterministic_and_frobenius_uniform():
    first = sample_direction_bank(6, 2)
    second = sample_direction_bank(6, 2)
    other = sample_direction_bank(6, 3)
    assert bank_sha256(first) == bank_sha256(second)
    assert bank_sha256(first) != bank_sha256(other)
    np.testing.assert_allclose(np.linalg.norm(first, axis=(1, 2)), 1.0, atol=2e-15)


def test_p2_is_exact_order_statistics_without_interpolation_or_translation():
    values = np.array([[[3.0, -1.0, 2.0], [8.0, 7.0, 9.0]]])
    np.testing.assert_array_equal(sorted_projection_features(values), [[-1.0, 2.0, 3.0, 7.0, 8.0, 9.0]])


def test_b5_uses_same_projection_values_mean_and_population_std():
    values = np.array([[[1.0, 3.0], [2.0, 6.0]]])
    np.testing.assert_allclose(moment_features(values), [[2.0, 4.0, 1.0, 2.0]])

