"""Bagging evaluation implementation.

Provides functions for problem 1a.
"""

from typing import Any, List, Sequence, Tuple

import numpy.typing as npt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, zero_one_loss
from sklearn.svm import SVC
from tqdm.autonotebook import tqdm


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


__all__ = [
    'evaluate_bagging',
]
