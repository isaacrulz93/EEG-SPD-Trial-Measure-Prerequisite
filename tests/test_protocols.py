import numpy as np

from src.protocols import assert_trial_disjoint, balanced_source_indices, make_splits


def _fixture():
    rows = []
    for subject in range(1, 10):
        for session in range(2):
            for cls in range(4):
                for repeat in range(10):
                    rows.append((subject, session, cls))
    rows = np.asarray(rows)
    trial = np.arange(len(rows))
    return rows[:, 2], rows[:, 0], rows[:, 1], trial


def test_all_splits_are_at_trial_level_and_both_session_directions_exist():
    y, subject, session, trial = _fixture()
    splits = make_splits(y, subject, session, trial)
    assert sum(value.protocol == "within_session" for value in splits) == 9 * 2 * 5
    assert sum(value.protocol == "s1_to_s2" for value in splits) == 9
    assert sum(value.protocol == "s2_to_s1" for value in splits) == 9
    assert sum(value.protocol == "loso" and value.ra_mode == "no_ra" for value in splits) == 9
    assert sum(value.protocol == "loso" and value.ra_mode == "subject_ra" for value in splits) == 9
    for split in splits:
        assert_trial_disjoint(split, trial)


def test_source_balancing_is_deterministic_across_subject_and_class():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    subject = np.array([1] * 5 + [2] * 5)
    indices = np.arange(10)
    selected = balanced_source_indices(indices, y, subject)
    np.testing.assert_array_equal(selected, balanced_source_indices(indices, y, subject))
    counts = {(s, c): int(np.sum((subject[selected] == s) & (y[selected] == c))) for s in [1, 2] for c in [0, 1]}
    assert set(counts.values()) == {2}

