from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValidationFold:
    fold_id: int
    train_indices: tuple[int, ...]
    valid_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    valid_group: int
    test_group: int
    purge_window: int
    embargo_window: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "valid_group": self.valid_group,
            "test_group": self.test_group,
            "purge_window": self.purge_window,
            "embargo_window": self.embargo_window,
            "train_size": len(self.train_indices),
            "valid_size": len(self.valid_indices),
            "test_size": len(self.test_indices),
        }


class CPCVValidator:
    """
    Lightweight CPCV-style validator for time-series alpha search.

    The implementation builds contiguous groups, enumerates ordered
    validation/test group pairs, and purges/embargoes train samples around the
    validation and test windows.
    """

    def __init__(self, purge_window: int = 0, embargo_window: int = 0, min_train_size: int = 3):
        self.purge_window = max(int(purge_window), 0)
        self.embargo_window = max(int(embargo_window), 0)
        self.min_train_size = max(int(min_train_size), 1)

    def generate_purged_splits(self, total_time_steps: int, n_splits: int) -> list[ValidationFold]:
        total_time_steps = max(int(total_time_steps), 0)
        n_splits = max(int(n_splits), 2)
        if total_time_steps <= 0:
            return []

        groups = [group.astype(int) for group in np.array_split(np.arange(total_time_steps), n_splits) if group.size]
        if len(groups) < 3:
            return []

        folds: list[ValidationFold] = []
        fold_id = 0
        for valid_group, valid_indices in enumerate(groups[:-1]):
            for test_group in range(valid_group + 1, len(groups)):
                test_indices = groups[test_group]
                train_indices = self._build_train_indices(total_time_steps, valid_indices, test_indices)
                if len(train_indices) < self.min_train_size:
                    continue
                folds.append(
                    ValidationFold(
                        fold_id=fold_id,
                        train_indices=train_indices,
                        valid_indices=tuple(int(idx) for idx in valid_indices.tolist()),
                        test_indices=tuple(int(idx) for idx in test_indices.tolist()),
                        valid_group=valid_group,
                        test_group=test_group,
                        purge_window=self.purge_window,
                        embargo_window=self.embargo_window,
                    )
                )
                fold_id += 1
        return folds

    def generate_holdout_split(
        self,
        total_time_steps: int,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
    ) -> ValidationFold | None:
        total_time_steps = max(int(total_time_steps), 0)
        if total_time_steps < 3:
            return None

        train_end = max(int(total_time_steps * train_ratio), 1)
        valid_end = max(int(total_time_steps * (train_ratio + valid_ratio)), train_end + 1)
        valid_end = min(valid_end, total_time_steps - 1)

        train_indices = tuple(range(0, train_end))
        valid_indices = tuple(range(train_end, valid_end))
        test_indices = tuple(range(valid_end, total_time_steps))
        if not train_indices or not valid_indices or not test_indices:
            return None

        return ValidationFold(
            fold_id=0,
            train_indices=train_indices,
            valid_indices=valid_indices,
            test_indices=test_indices,
            valid_group=0,
            test_group=1,
            purge_window=0,
            embargo_window=0,
        )

    def _build_train_indices(
        self,
        total_time_steps: int,
        valid_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> tuple[int, ...]:
        excluded: set[int] = set()
        for window in (valid_indices, test_indices):
            if window.size == 0:
                continue
            start = int(window[0])
            end = int(window[-1])
            purge_start = max(0, start - self.purge_window)
            purge_end = min(total_time_steps - 1, end + self.purge_window)
            excluded.update(range(purge_start, purge_end + 1))
            embargo_end = min(total_time_steps - 1, end + self.embargo_window)
            excluded.update(range(end + 1, embargo_end + 1))
        return tuple(idx for idx in range(total_time_steps) if idx not in excluded)
