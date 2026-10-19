"""Employee Attrition Predictor Using Data Mining - local desktop application.

Run:  python main.py            (desktop GUI)
      python main.py --demo     (overview + train + compare, no questions asked)
      python main.py --regenerate   (re-create the synthetic dataset)
      python main.py --cli          (use the original terminal menu)
"""

import argparse
import sys

from src.data_loader import (DATA_PATH, OUTPUT_DIR, describe_dataset, generate_synthetic_dataset,
                             load_dataset, save_dataset)
from src.evaluation import (compare_models, print_comparison, save_confusion_matrix_plots,
                            save_metrics_csv)
from src.prediction import (WARNING_TEXT, format_prediction, predict_employee,
                            prompt_employee_details, save_prediction_csv)
from src.preprocessing import (clean_data, data_quality_report, explain_preprocessing,
                               split_features_target)
from src.train_models import split_data, train_models, tree_feature_importance


class Session:
    """Keeps the trained models in memory while the program runs."""

    def __init__(self):
        self.models = None
        self.results = None

    def train_and_compare(self):
        df = load_dataset()
        report = data_quality_report(df)
        print(f"\nBefore cleaning: {report['rows']} rows, {report['total_missing']} missing values, "
              f"{report['duplicate_rows']} duplicate rows.")
        df = clean_data(df)
        print(f"After cleaning : {len(df)} rows (duplicates and invalid targets removed).")

        X, y = split_features_target(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        print(f"Train/test split (80/20, stratified, seed 42): {len(X_train)} train rows, "
              f"{len(X_test)} test rows.")
        print(f"Attrition 'Yes' share - train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")

        self.models = train_models(X_train, y_train)
        self.results = compare_models(self.models, X_test, y_test, X_train, y_train)
        print_comparison(self.results)

        print("\nMost important attributes for the Decision Tree:")
        print(tree_feature_importance(self.models["Decision Tree"]).round(3).to_string())

        print(f"\nSaved metrics : {save_metrics_csv(self.results)}")
        for path in save_confusion_matrix_plots(self.results):
            print(f"Saved plot    : {path}")


def show_overview():
    df = load_dataset()
    print()
    describe_dataset(df)
    report = data_quality_report(df)
    print(f"\nData quality: {report['total_missing']} missing values, "
          f"{report['duplicate_rows']} duplicate rows.")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    explain_preprocessing()


def predict_flow(session):
    print("\n" + WARNING_TEXT)
    if session.models is None:
        print("\nNo trained models yet - training now...")
        session.train_and_compare()

    names = list(session.models)
    best = session.results.loc[session.results["f1_score"].astype(float).idxmax(), "model"]
    print("\nWhich model should make the prediction?")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}" + ("  (highest F1 on test set)" if name == best else ""))
    choice = input("Choose 1 or 2 [default = highest F1]: ").strip()
    model_name = names[int(choice) - 1] if choice in ("1", "2") else best

    info = prompt_employee_details()
    result = predict_employee(session.models[model_name], info)
    print(f"\nModel used: {model_name}")
    print(format_prediction(result))
    print("\n" + WARNING_TEXT)
    print(f"Saved to {save_prediction_csv(result, model_name)}")


def regenerate_dataset():
    df = generate_synthetic_dataset()
    path = save_dataset(df)
    print(f"Synthetic dataset created: {path} ({len(df)} rows). This data is SYNTHETIC, not real.")


def menu():
    print("=" * 62)
    print(" Employee Attrition Predictor Using Data Mining (DWM mini-project)")
    print("=" * 62)
    print(WARNING_TEXT)
    session = Session()
    while True:
        print("\nMenu")
        print("  1. Dataset overview and preprocessing explanation")
        print("  2. Train and compare models (Decision Tree vs Naive Bayes)")
        print("  3. Predict attrition for one employee")
        print("  4. Regenerate synthetic dataset")
        print("  0. Exit")
        choice = input("Choose an option: ").strip()
        try:
            if choice == "1":
                show_overview()
            elif choice == "2":
                session.train_and_compare()
            elif choice == "3":
                predict_flow(session)
            elif choice == "4":
                regenerate_dataset()
                session = Session()  # old models were trained on the old data
            elif choice == "0":
                print("Goodbye!")
                return
            else:
                print("Please enter 0, 1, 2, 3 or 4.")
        except (FileNotFoundError, ValueError) as exc:
            print(f"\nError: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Employee Attrition Predictor (educational).")
    parser.add_argument("--demo", action="store_true", help="run overview + training without the menu")
    parser.add_argument("--regenerate", action="store_true", help="re-create the synthetic dataset and exit")
    parser.add_argument("--cli", action="store_true", help="use the original command-line menu")
    args = parser.parse_args()

    try:
        if args.regenerate:
            regenerate_dataset()
        elif args.demo:
            if not DATA_PATH.exists():
                regenerate_dataset()
            show_overview()
            Session().train_and_compare()
        else:
            if not DATA_PATH.exists():
                print("Dataset not found - creating it first.")
                regenerate_dataset()
            if args.cli:
                menu()
            else:
                from gui import run_app
                run_app()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")


if __name__ == "__main__":
    main()
