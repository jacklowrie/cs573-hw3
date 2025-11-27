"""Command-line entrypoint for the bagging evaluation script."""

from typing import Dict, List, Tuple

from src.bagging import run_all


def main() -> None:
    """Run the bagging evaluations and print summarized results.

    Loads the dataset, runs `run_all` and prints zero-one loss and ROC AUC
    for the two datasets D1 and D2.
    """
    data_path = "data/datahw3.npz"
    print("Running evaluations using helper module...\n")
    results: Dict[str, List[Tuple[int, float, float]]] = run_all(data_path)

    print("\nEvaluating Dataset D1")
    for n, zol, auc in results["D1"]:
        print(f"n_estimators={n}: zero-one loss={zol:.4f}, ROC AUC={auc:.4f}")

    print("\nEvaluating Dataset D2")
    for n, zol, auc in results["D2"]:
        print(f"n_estimators={n}: zero-one loss={zol:.4f}, ROC AUC={auc:.4f}")


if __name__ == "__main__":
    main()
