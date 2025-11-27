"""Bagging evaluation implementation.

Provides functions for problem 1a.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy.typing as npt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, zero_one_loss
from sklearn.svm import SVC
from tqdm import tqdm

from src.utils import load_data


def evaluate_bagging(
    train_X: npt.NDArray[Any],
    train_y: npt.NDArray[Any],
    test_X: npt.NDArray[Any],
    test_y: npt.NDArray[Any],
    n_list: Sequence[int],
    use_tqdm: bool = True,
) -> List[Tuple[int, float, float]]:
    """Train BaggingClassifier ensembles and return results.

    Args:
        train_X: Training features.
        train_y: Training labels.
        test_X: Testing features.
        test_y: Testing labels.
        n_list: List of n_estimators values to evaluate.
        use_tqdm: Whether to use tqdm progress bars.

    Returns:
        a list of tuples (n_estimators, zero_one_loss, roc_auc).
    """
    results: List[Tuple[int, float, float]] = []
    iterator = tqdm(n_list, desc="bagging n_estimators") if use_tqdm else n_list

    # use the modern scikit-learn kwarg `estimator` unconditionally
    for n in iterator:
        base = SVC(probability=True, random_state=0)
        clf = BaggingClassifier(estimator=base, n_estimators=n, random_state=0)
        clf.fit(train_X, train_y)
        y_pred = clf.predict(test_X)
        try:
            y_score = clf.predict_proba(test_X)[:, 1]
        except Exception:
            y_score = clf.decision_function(test_X)
        zol = zero_one_loss(test_y, y_pred)
        auc = roc_auc_score(test_y, y_score)
        results.append((n, float(zol), float(auc)))
    return results


def run_all(
    path: str,
    n_list: Optional[Sequence[int]] = None,
    use_tqdm: bool = True,
) -> Dict[str, List[Tuple[int, float, float]]]:
    """Load data, run bagging evaluations for D1 and D2, and return results.

    Returns a dictionary with keys 'D1' and 'D2' mapping to lists of
    tuples (n_estimators, zero_one_loss, roc_auc).
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

    return {"D1": res1, "D2": res2}


__all__ = [
    'evaluate_bagging',
    'run_all',
]
