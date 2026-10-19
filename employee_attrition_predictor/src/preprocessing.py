"""Data cleaning and preprocessing.

Key idea to avoid DATA LEAKAGE: the imputer, scaler and encoder are placed inside
a scikit-learn Pipeline. The pipeline learns (fits) these steps ONLY from the
training data and then merely applies them to the test data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import (CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
                             REQUIRED_COLUMNS, TARGET)


def data_quality_report(df):
    """Detect missing values and duplicate rows (does not change the data)."""
    return {
        "rows": len(df),
        "missing_per_column": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_data(df):
    """Basic data cleaning. Returns a NEW cleaned DataFrame.

    Steps:
      1. Check required columns exist.
      2. Remove leading/trailing spaces in text columns.
      3. Make numeric columns numeric (bad values become missing).
      4. Drop duplicate rows.
      5. Drop rows whose target (Attrition) is missing or not Yes/No.

    Missing values in the FEATURES are NOT filled here; that is done later inside
    the pipeline (fitted on training data only, to avoid leakage).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    for col in CATEGORICAL_COLUMNS + [TARGET]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].astype(object).where(df[col].notna(), None)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()

    df = df[df[TARGET].isin(["Yes", "No"])]
    if df.empty:
        raise ValueError("No valid rows left after cleaning (Attrition must be 'Yes' or 'No').")
    return df.reset_index(drop=True)


def split_features_target(df):
    """Separate input features (X) from the target (y). y: Yes -> 1, No -> 0."""
    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def build_preprocessor():
    """Build the preprocessing step.

    Numeric columns     : fill missing values with the median, then standardise.
    Categorical columns : fill missing values with the most frequent value, then
                          one-hot encode (each category becomes a 0/1 column).
    """
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERIC_COLUMNS),
        ("cat", categorical_steps, CATEGORICAL_COLUMNS),
    ])


def explain_preprocessing():
    """Beginner-friendly explanation printed by the program."""
    print(
        "\nPreprocessing steps used in this project\n"
        "----------------------------------------\n"
        "1. Data cleaning      : remove duplicate rows, trim spaces, drop rows with an invalid target.\n"
        "2. Missing values     : numbers -> median, text -> most frequent value.\n"
        "3. Data transformation: numbers are standardised (mean 0, spread 1);\n"
        "                        text categories are one-hot encoded into 0/1 columns.\n"
        "4. Features vs target : the 8 input columns go in X, 'Attrition' (Yes=1/No=0) goes in y.\n"
        "5. No data leakage    : all of the above are learned from the TRAINING data only\n"
        "                        (inside a Pipeline) and then applied to the test data."
    )
