"""Train the two classifiers: Decision Tree and Naive Bayes."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import build_preprocessor

RANDOM_SEED = 42
MIN_ROWS = 30


def split_data(X, y, test_size=0.2, seed=RANDOM_SEED):
    """Stratified train/test split (keeps the Yes/No ratio the same in both parts)."""
    if len(X) < MIN_ROWS:
        raise ValueError(
            f"Dataset too small: {len(X)} rows after cleaning, at least {MIN_ROWS} are needed."
        )
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("The target has only one class; need both 'Yes' and 'No' to train.")
    if counts.min() < 5:
        raise ValueError(
            f"The smaller class has only {counts.min()} rows; need at least 5 to train and test."
        )
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_models(seed=RANDOM_SEED):
    """Return two untrained pipelines (preprocessing + classifier).

    Class imbalance: attrition 'Yes' is the minority class. Both models are trained
    with balanced sample weights, so mistakes on the minority class count more.
    (We also report precision, recall and F1, not only accuracy.)
    """
    tree = Pipeline([
        ("prep", build_preprocessor()),
        ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=seed)),
    ])
    bayes = Pipeline([
        ("prep", build_preprocessor()),
        ("model", GaussianNB()),
    ])
    return {"Decision Tree": tree, "Naive Bayes": bayes}


def train_models(X_train, y_train, seed=RANDOM_SEED):
    """Fit both pipelines on the TRAINING data only and return them in a dict."""
    weights = compute_sample_weight("balanced", y_train)
    models = build_models(seed)
    for name, pipe in models.items():
        pipe.fit(X_train, y_train, model__sample_weight=weights)
    return models


def tree_feature_importance(tree_pipeline, top_n=8):
    """Which (encoded) attributes did the decision tree use most?"""
    names = tree_pipeline.named_steps["prep"].get_feature_names_out()
    importance = tree_pipeline.named_steps["model"].feature_importances_
    result = pd.Series(importance, index=names).sort_values(ascending=False)
    return result.head(top_n)
