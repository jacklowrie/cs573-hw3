"""boosting.py: Provides functions for problem 1b."""

from typing import Any, List, Sequence, Tuple

import numpy.typing as npt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from tqdm.autonotebook import tqdm


def evaluate_boosting(
    train_X: npt.NDArray[Any],
    train_y: npt.NDArray[Any],
    test_X: npt.NDArray[Any],
    test_y: npt.NDArray[Any],
    n_list: Sequence[int],
    use_tqdm: bool = True,
) -> List[Tuple[int, float, float]]:
    """Train AdaBoostClassifier ensembles and return results.

    Uses DecisionTreeClassifier(max_depth=1) as the base estimator.

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
    iterator = (
        tqdm(n_list, desc="boosting n_estimators") if use_tqdm else n_list
    )

    for n in iterator:
        base = DecisionTreeClassifier(max_depth=1, random_state=0)
        clf = AdaBoostClassifier(estimator=base, n_estimators=n, random_state=0)
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


__all__ = ["evaluate_boosting"]
