"""Trial-level protocol construction with grouped within-session folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class Split:
    protocol: str
    ra_mode: str
    subject: int
    session_train: str
    session_test: str
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def make_splits(
    y: np.ndarray,
    subject: np.ndarray,
    session: np.ndarray,
    trial_id: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 20260904,
) -> list[Split]:
    y = np.asarray(y)
    subject = np.asarray(subject)
    session = np.asarray(session)
    trial_id = np.asarray(trial_id)
    splits: list[Split] = []
    subject_values = sorted(int(value) for value in np.unique(subject))

    for sub in subject_values:
        subject_mask = subject == sub
        sessions = sorted(int(value) for value in np.unique(session[subject_mask]))
        for ses in sessions:
            indices = np.flatnonzero(subject_mask & (session == ses))
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for fold_id, (train_local, test_local) in enumerate(
                splitter.split(indices, y[indices], groups=trial_id[indices])
            ):
                splits.append(
                    Split("within_session", "no_ra", sub, str(ses + 1), str(ses + 1), fold_id,
                          indices[train_local], indices[test_local])
                )
        if len(sessions) >= 2:
            first, second = sessions[:2]
            first_indices = np.flatnonzero(subject_mask & (session == first))
            second_indices = np.flatnonzero(subject_mask & (session == second))
            splits.append(Split("s1_to_s2", "no_ra", sub, str(first + 1), str(second + 1), 0, first_indices, second_indices))
            splits.append(Split("s2_to_s1", "no_ra", sub, str(second + 1), str(first + 1), 0, second_indices, first_indices))

    for target in subject_values:
        train = np.flatnonzero(subject != target)
        test = np.flatnonzero(subject == target)
        splits.append(Split("loso", "no_ra", target, "all", "all", 0, train, test))
        splits.append(Split("loso", "subject_ra", target, "all", "all", 0, train, test))
    return splits


def assert_trial_disjoint(split: Split, trial_id: np.ndarray) -> None:
    train_trials = set(np.asarray(trial_id)[split.train_indices].tolist())
    test_trials = set(np.asarray(trial_id)[split.test_indices].tolist())
    overlap = train_trials.intersection(test_trials)
    if overlap:
        raise AssertionError(f"Trial leakage in {split.protocol}: {sorted(overlap)[:5]}")


def balanced_source_indices(indices: np.ndarray, y: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """Deterministically equalize every available subject x class cell."""

    indices = np.sort(np.asarray(indices, dtype=np.int64))
    y = np.asarray(y)
    subject = np.asarray(subject)
    subjects = np.unique(subject[indices])
    classes = np.unique(y[indices])
    cells: list[np.ndarray] = []
    for sub in subjects:
        for cls in classes:
            cell = indices[(subject[indices] == sub) & (y[indices] == cls)]
            if len(cell) == 0:
                raise ValueError(f"Missing source cell subject={sub}, class={cls}.")
            cells.append(cell)
    keep_per_cell = min(map(len, cells))
    return np.sort(np.concatenate([cell[:keep_per_cell] for cell in cells]))

