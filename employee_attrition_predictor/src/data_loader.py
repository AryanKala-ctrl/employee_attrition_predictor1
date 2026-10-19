"""Dataset management: generate, save, load and describe the employee dataset.

The dataset used by this project is SYNTHETIC (computer-generated). It is NOT
real employee data. It exists only so the data-mining pipeline can be demonstrated.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# All paths are relative to the project folder, so the project works on any laptop.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "employee_attrition.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET = "Attrition"
NUMERIC_COLUMNS = ["Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "DistanceFromHome"]
CATEGORICAL_COLUMNS = ["Department", "JobRole", "OverTime"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]

# Which job roles exist in which department (also used to validate user input).
DEPARTMENT_ROLES = {
    "Sales": ["Sales Executive", "Sales Representative", "Sales Manager"],
    "Research": ["Research Scientist", "Lab Technician", "Software Engineer", "Research Manager"],
    "HR": ["HR Executive", "Recruiter"],
    "Finance": ["Accountant", "Financial Analyst"],
}

# Typical starting monthly income (in rupees) for each role - used only by the generator.
_ROLE_BASE_INCOME = {
    "Sales Executive": 5500, "Sales Representative": 3500, "Sales Manager": 9500,
    "Research Scientist": 5000, "Lab Technician": 3300, "Software Engineer": 7000,
    "Research Manager": 10500, "HR Executive": 4500, "Recruiter": 3600,
    "Accountant": 5200, "Financial Analyst": 6500,
}


def generate_synthetic_dataset(n_rows=1000, seed=42):
    """Create a reproducible synthetic employee dataset.

    Attrition is generated from a *probability* that depends on several
    attributes plus random noise. So the target is NOT a fixed rule and the
    classifiers cannot reach perfect accuracy.

    The data deliberately contains ~3% missing values (MonthlyIncome,
    JobSatisfaction) and a few duplicate rows, so that data cleaning has
    something real to do.
    """
    if n_rows < 50:
        raise ValueError("n_rows must be at least 50 to generate a useful dataset.")

    rng = np.random.default_rng(seed)

    departments = rng.choice(list(DEPARTMENT_ROLES), size=n_rows, p=[0.30, 0.45, 0.10, 0.15])
    roles = np.array([rng.choice(DEPARTMENT_ROLES[d]) for d in departments])

    age = np.clip(rng.normal(37, 9, n_rows), 18, 60).round().astype(int)
    years = np.minimum(rng.gamma(2.0, 2.5, n_rows).round().astype(int), np.maximum(age - 18, 0))
    base = np.array([_ROLE_BASE_INCOME[r] for r in roles], dtype=float)
    income = (base * (1 + 0.03 * years) * rng.lognormal(0, 0.18, n_rows)).round().astype(int)
    satisfaction = rng.choice([1, 2, 3, 4], size=n_rows, p=[0.20, 0.20, 0.30, 0.30])
    overtime = rng.choice(["Yes", "No"], size=n_rows, p=[0.28, 0.72])
    distance = rng.integers(1, 30, n_rows)

    # Attrition probability: logistic function of the attributes plus random noise.
    logit = (
        -1.9
        + 1.0 * (overtime == "Yes")
        - 0.45 * (satisfaction - 2.5)
        - 0.00035 * (income - 5500)
        - 0.04 * (age - 37)
        - 0.06 * years
        + 0.02 * distance
        + rng.normal(0, 0.8, n_rows)
    )
    prob = 1 / (1 + np.exp(-logit))
    attrition = np.where(rng.random(n_rows) < prob, "Yes", "No")

    df = pd.DataFrame({
        "Age": age,
        "Department": departments,
        "JobRole": roles,
        "MonthlyIncome": income.astype(float),
        "JobSatisfaction": satisfaction.astype(float),
        "OverTime": overtime,
        "YearsAtCompany": years,
        "DistanceFromHome": distance,
        "Attrition": attrition,
    })

    # Inject ~3% missing values in two columns.
    for col in ["MonthlyIncome", "JobSatisfaction"]:
        missing_idx = rng.choice(n_rows, size=int(0.03 * n_rows), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Add 5 duplicate rows and shuffle.
    duplicates = df.sample(5, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def save_dataset(df, path=DATA_PATH):
    """Save a dataset to CSV, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path=DATA_PATH):
    """Load the CSV dataset and check that all required columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Create it with:  python main.py --regenerate"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable / corrupted CSV
        raise ValueError(f"Could not read the CSV file {path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"The dataset {path} is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The dataset is missing required column(s): {missing}.\n"
            f"Required columns are: {REQUIRED_COLUMNS}"
        )
    return df


def describe_dataset(df):
    """Print dimensions, column names, data types and missing-value counts."""
    print(f"Dimensions : {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names, data types and missing values:")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
    })
    print(summary.to_string())
    if TARGET in df.columns:
        print("\nTarget distribution (Attrition):")
        print(df[TARGET].value_counts(dropna=False).to_string())
    return summary
