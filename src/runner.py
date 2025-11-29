"""runner.py: Runs bagging and boosting evaluations."""

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

    Returns a dictionary with keys 'bag_ds1', 'bag_ds2', 'boost_ds1', and
    'boost_ds2'.
    """
    train_X, val_X, train_y, val_y, train_y2, val_y2 = load_data(path)

    if n_list is None:
        n_list = [2, 10, 50, 75, 100]

    bag_ds1 = evaluate_bagging(
        train_X,
        train_y,
        val_X,
        val_y,
        n_list,
        use_tqdm=use_tqdm,
    )

    bag_ds2 = evaluate_bagging(
        train_X,
        train_y2,
        val_X,
        val_y2,
        n_list,
        use_tqdm=use_tqdm,
    )

    boost_ds1 = evaluate_boosting(
        train_X, train_y, val_X, val_y, n_list, use_tqdm=use_tqdm
    )

    boost_ds2 = evaluate_boosting(
        train_X, train_y2, val_X, val_y2, n_list, use_tqdm=use_tqdm
    )

    return {
        "bag_ds1": bag_ds1,
        "bag_ds2": bag_ds2,
        "boost_ds1": boost_ds1,
        "boost_ds2": boost_ds2,
    }


__all__ = ["run_all"]
