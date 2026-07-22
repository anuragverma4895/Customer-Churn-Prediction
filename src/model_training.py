"""
Model Training for Customer Churn Prediction.
Trains Logistic Regression and Random Forest with hyperparameter tuning.
"""

import os
import sys
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_model_path, print_section


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    tune: bool = True,
) -> LogisticRegression:
    """
    Train a Logistic Regression model with optional hyperparameter tuning.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    cv : int
        Number of cross-validation folds.
    tune : bool
        Whether to perform GridSearchCV.

    Returns
    -------
    LogisticRegression
        Trained model (best estimator if tuned).
    """
    print_section("Training Logistic Regression")

    if tune:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        }
        grid = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV F1:  {grid.best_score_:.4f}")
    else:
        model = LogisticRegression(C=1, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print(f"  CV F1 scores: {cv_scores.round(4)}")
    print(f"  CV F1 mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    tune: bool = True,
) -> RandomForestClassifier:
    """
    Train a Random Forest model with optional hyperparameter tuning.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    cv : int
        Number of cross-validation folds.
    tune : bool
        Whether to perform GridSearchCV.

    Returns
    -------
    RandomForestClassifier
        Trained model (best estimator if tuned).
    """
    print_section("Training Random Forest")

    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV F1:  {grid.best_score_:.4f}")
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=2, random_state=42
        )
        model.fit(X_train, y_train)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print(f"  CV F1 scores: {cv_scores.round(4)}")
    print(f"  CV F1 mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model


def save_model(model, name: str) -> str:
    """Save a trained model to the models/ directory."""
    path = get_model_path(f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Model saved: {path}")
    return path


def load_model(name: str):
    """Load a trained model from the models/ directory."""
    path = get_model_path(f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
