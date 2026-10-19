"""Basic automated tests.

Run with either:
    python -m pytest -v
    python -m unittest -v tests.test_project
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Make "import src..." work no matter how the tests are started.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (FEATURE_COLUMNS, REQUIRED_COLUMNS, generate_synthetic_dataset,  # noqa: E402
                             load_dataset, save_dataset)
from src.evaluation import (compare_models, evaluate_model, save_confusion_matrix_plots,  # noqa: E402
                            save_metrics_csv)
from src.prediction import predict_employee, save_prediction_csv, validate_employee_input  # noqa: E402
from src.preprocessing import clean_data, data_quality_report, split_features_target  # noqa: E402
from src.train_models import split_data, train_models  # noqa: E402

VALID_EMPLOYEE = {
    "Age": 30, "Department": "Sales", "JobRole": "Sales Executive", "MonthlyIncome": 5000,
    "JobSatisfaction": 2, "OverTime": "Yes", "YearsAtCompany": 3, "DistanceFromHome": 10,
}


def prepared_data():
    df = clean_data(generate_synthetic_dataset())
    X, y = split_features_target(df)
    return split_data(X, y)


class TestDataset(unittest.TestCase):
    def test_generator_is_reproducible(self):
        a = generate_synthetic_dataset(200, seed=1)
        b = generate_synthetic_dataset(200, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_generated_data_has_both_classes_and_missing_values(self):
        df = generate_synthetic_dataset()
        self.assertEqual(set(df["Attrition"]), {"Yes", "No"})
        self.assertGreater(df.isnull().sum().sum(), 0)
        self.assertGreater(df.duplicated().sum(), 0)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_dataset(generate_synthetic_dataset(100), Path(tmp) / "sub" / "d.csv")
            loaded = load_dataset(path)
            self.assertEqual(set(loaded.columns), set(REQUIRED_COLUMNS))
            self.assertEqual(len(loaded), 105)  # 100 rows + 5 duplicates

    def test_load_missing_file_gives_helpful_error(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(PROJECT_ROOT / "data" / "does_not_exist.csv")

    def test_load_missing_column_gives_helpful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            generate_synthetic_dataset(100).drop(columns=["Age"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Age"):
                load_dataset(path)

    def test_shipped_csv_loads(self):
        df = load_dataset()  # data/employee_attrition.csv
        self.assertGreaterEqual(len(df), 500)


class TestPreprocessing(unittest.TestCase):
    def test_quality_report_counts_missing_and_duplicates(self):
        df = generate_synthetic_dataset()
        report = data_quality_report(df)
        self.assertEqual(report["duplicate_rows"], 5)
        self.assertEqual(report["total_missing"], int(df.isnull().sum().sum()))

    def test_clean_data_removes_duplicates_and_bad_targets(self):
        df = generate_synthetic_dataset(100)
        df.loc[0, "Attrition"] = "Maybe"
        cleaned = clean_data(df)
        self.assertEqual(cleaned.duplicated().sum(), 0)
        self.assertTrue(cleaned["Attrition"].isin(["Yes", "No"]).all())

    def test_clean_data_requires_columns(self):
        with self.assertRaises(ValueError):
            clean_data(pd.DataFrame({"Age": [1, 2]}))

    def test_missing_values_are_handled_by_the_pipeline(self):
        X_train, X_test, y_train, _ = prepared_data()
        self.assertGreater(X_train.isnull().sum().sum(), 0)  # still missing before the pipeline
        models = train_models(X_train, y_train)
        for model in models.values():
            preds = model.predict(X_test)  # would raise an error if NaN were not imputed
            self.assertEqual(len(preds), len(X_test))

    def test_split_is_stratified_and_reproducible(self):
        df = clean_data(generate_synthetic_dataset())
        X, y = split_features_target(df)
        a = split_data(X, y)
        b = split_data(X, y)
        pd.testing.assert_frame_equal(a[0], b[0])
        self.assertAlmostEqual(a[2].mean(), a[3].mean(), delta=0.02)

    def test_small_dataset_gives_error(self):
        df = clean_data(generate_synthetic_dataset(100)).head(10)
        X, y = split_features_target(df)
        with self.assertRaisesRegex(ValueError, "too small"):
            split_data(X, y)


class TestModelsAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = prepared_data()
        cls.models = train_models(cls.X_train, cls.y_train)

    def test_both_models_trained(self):
        self.assertEqual(set(self.models), {"Decision Tree", "Naive Bayes"})

    def test_metrics_are_valid(self):
        for model in self.models.values():
            m = evaluate_model(model, self.X_test, self.y_test)
            for key in ["accuracy", "precision", "recall", "f1_score"]:
                self.assertTrue(0.0 <= m[key] <= 1.0)
            total = m["true_negative"] + m["false_positive"] + m["false_negative"] + m["true_positive"]
            self.assertEqual(total, len(self.y_test))

    def test_accuracy_matches_confusion_matrix(self):
        m = evaluate_model(self.models["Decision Tree"], self.X_test, self.y_test)
        expected = (m["true_negative"] + m["true_positive"]) / len(self.y_test)
        self.assertAlmostEqual(m["accuracy"], expected)

    def test_export_metrics_and_plots(self):
        results = compare_models(self.models, self.X_test, self.y_test)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new_output_folder"  # does not exist yet
            csv_path = save_metrics_csv(results, out)
            self.assertTrue(csv_path.exists())
            self.assertEqual(len(pd.read_csv(csv_path)), 2)
            plots = save_confusion_matrix_plots(results, out)
            self.assertEqual(len(plots), 2)
            self.assertTrue(all(p.exists() for p in plots))


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X_train, _, y_train, _ = prepared_data()
        cls.models = train_models(X_train, y_train)

    def test_prediction_output(self):
        for model in self.models.values():
            result = predict_employee(model, VALID_EMPLOYEE)
            self.assertIn(result["Prediction"], ["Yes", "No"])
            self.assertTrue(0.0 <= result["ProbabilityYes"] <= 1.0)

    def test_text_input_is_cleaned(self):
        info = dict(VALID_EMPLOYEE, Department=" sales ", OverTime="yes", Age="30")
        self.assertEqual(validate_employee_input(info)["Department"], "Sales")

    def test_invalid_inputs_raise_value_error(self):
        bad_cases = [
            {"Age": "abc"}, {"Age": 5}, {"Department": "Cooking"},
            {"JobRole": "Recruiter"},            # role does not belong to Sales
            {"JobSatisfaction": 9}, {"OverTime": "Maybe"}, {"MonthlyIncome": -50},
            {"YearsAtCompany": 29},              # impossible for age 30
        ]
        for change in bad_cases:
            with self.assertRaises(ValueError, msg=str(change)):
                validate_employee_input({**VALID_EMPLOYEE, **change})

    def test_missing_field_raises(self):
        info = dict(VALID_EMPLOYEE)
        del info["Age"]
        with self.assertRaises(ValueError):
            validate_employee_input(info)

    def test_prediction_csv_is_appended(self):
        result = predict_employee(self.models["Naive Bayes"], VALID_EMPLOYEE)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs"
            save_prediction_csv(result, "Naive Bayes", out)
            path = save_prediction_csv(result, "Naive Bayes", out)
            self.assertEqual(len(pd.read_csv(path)), 2)


if __name__ == "__main__":
    unittest.main()
