"""Command-line entrypoint."""

from typing import Dict, List, Tuple

from src.runner import run_all


def main() -> None:
    """Run the bagging evaluations and print summarized results.

    Loads the dataset, runs `run_all` and prints zero-one loss and ROC AUC
    for the two datasets D1 and D2.
    """
    data_path = "data/datahw3.npz"
    print("Running evaluations using helper module...\n")
    results: Dict[str, List[Tuple[int, float, float]]] = run_all(data_path)

    print("\nBagging results (SVC base estimator)")
    print("\nEvaluating Dataset D1")
    for n, zol, auc in results["bag_ds1"]:
        print(f"n_estimators={n}: zero-one loss={zol:.4f}, ROC AUC={auc:.4f}")

    print("\nEvaluating Dataset D2")
    for n, zol, auc in results["bag_ds2"]:
        print(f"n_estimators={n}: zero-one loss={zol:.4f}, ROC AUC={auc:.4f}")

    print("\nBoosting results (AdaBoost with DecisionTree max_depth=1)")
    print("\nDataset D1 (train_dat_y)")
    for n, zol, auc in results["boost_ds1"]:
        print(f"n_estimators={n}: zero-one loss={zol:.4f}, ROC AUC={auc:.4f}")

    print("\nDataset D2 (train_dat_y2)")
    for n, zol, auc in results["boost_ds2"]:
        print(f"n_estimators={n}: zero-one loss={zol:.4f}, ROC AUC={auc:.4f}")


if __name__ == "__main__":
    main()
