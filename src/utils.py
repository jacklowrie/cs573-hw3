"""Utility and helper functions."""

from functools import lru_cache
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt


@lru_cache(maxsize=None)
def load_data(
    path: str,
) -> Tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
]:
    """Load an .npz file and return a fixed-order tuple of arrays.

    Returns (train_X, val_X, train_y, val_y, train_y2, val_y2).
    Cached by path so repeated notebook runs reuse memory.
    """
    with np.load(path, allow_pickle=True) as npz:
        # return a fixed-order tuple so callers can unpack directly
        train_X = npz["train_dat_x"]
        val_X = npz["val_dat_x"]
        train_y = npz["train_dat_y"]
        val_y = npz["val_dat_y"]
        train_y2 = npz["train_dat_y2"]
        val_y2 = npz["val_dat_y2"]
        return train_X, val_X, train_y, val_y, train_y2, val_y2


__all__ = ["load_data"]
