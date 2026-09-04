import inspect

import numpy as np

from src.geometry import svec, subject_airm_recenter, unsvec


def test_svec_is_frobenius_isometry():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(9, 5, 5))
    matrix = 0.5 * (matrix + matrix.transpose(0, 2, 1))
    vector = svec(matrix)
    np.testing.assert_allclose(np.sum(matrix**2, axis=(1, 2)), np.sum(vector**2, axis=1))
    np.testing.assert_allclose(unsvec(vector, 5), matrix)


def test_subject_ra_has_no_label_input():
    assert "y" not in inspect.signature(subject_airm_recenter).parameters
    assert "label" not in inspect.signature(subject_airm_recenter).parameters

