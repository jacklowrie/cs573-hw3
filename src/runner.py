"""Runner functions for running bagging and boosting experiments.

This module exposes `run_all` which loads the dataset and runs both
bagging and boosting evaluations producing a consolidated result dict.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from src.bagging import evaluate_bagging
from src.boosting import evaluate_boosting
from src.utils import load_data


def run_all(
    path: str,
    n_list: Optional[Sequence[int]] = None,
    use_tqdm: bool = True,
) -> Dict[str, List[Tuple[int, float, float]]]:
    """Load data and run bagging and boosting evaluations.

    Returns a dictionary with keys 'D1', 'D2', 'D1_boost', and 'D2_boost'.
    """
    train_X, val_X, train_y, val_y, train_y2, val_y2 = load_data(path)

    if n_list is None:
        n_list = [2, 10, 50, 75, 100]

    res1 = evaluate_bagging(
        train_X,
        train_y,
        val_X,
        val_y,
        n_list,
        use_tqdm=use_tqdm,
    )
    res2 = evaluate_bagging(
        train_X,
        train_y2,
        val_X,
        val_y2,
        n_list,
        use_tqdm=use_tqdm,
    )

    bres1 = evaluate_boosting(
        train_X, train_y, val_X, val_y, n_list, use_tqdm=use_tqdm
    )
    bres2 = evaluate_boosting(
        train_X, train_y2, val_X, val_y2, n_list, use_tqdm=use_tqdm
    )

    return {"D1": res1, "D2": res2, "D1_boost": bres1, "D2_boost": bres2}


__all__ = ["run_all"]
