"""
Model Evaluation for Customer Churn Prediction.
Computes metrics, confusion matrix, ROC curve data, and feature importances.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.utils import print_section


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a trained model and return all metrics.

    Parameters
    ----------
    model : trained sklearn estimator
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True test labels.
    model_name : str
        Display name of the model.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, confusion_matrix,
                    roc_fpr, roc_tpr, roc_thresholds, classification_report
    """
    print_section(f"Evaluating {model_name}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\n{report}")

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "roc_thresholds": thresholds.tolist(),
        "classification_report": report,
    }


def get_feature_importance(
    model,
    feature_names: list[str],
    model_type: str = "random_forest",
    top_n: int = 15,
) -> list[dict]:
    """
    Extract feature importances from a trained model.

    For Random Forest: uses Gini importance (feature_importances_).
    For Logistic Regression: uses absolute coefficient values.

    Parameters
    ----------
    model : trained sklearn estimator
    feature_names : list[str]
        Names of the features.
    model_type : str
        Either 'random_forest' or 'logistic_regression'.
    top_n : int
        Number of top features to return.

    Returns
    -------
    list[dict]
        Sorted list of {"feature": name, "importance": value}.
    """
    if model_type == "random_forest":
        importances = model.feature_importances_
    elif model_type == "logistic_regression":
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Normalize to [0, 1]
    max_imp = importances.max()
    if max_imp > 0:
        importances_norm = importances / max_imp
    else:
        importances_norm = importances

    # Sort descending
    indices = np.argsort(importances_norm)[::-1][:top_n]

    result = []
    for idx in indices:
        result.append(
            {
                "feature": feature_names[idx],
                "importance": round(float(importances_norm[idx]), 4),
                "raw_importance": round(float(importances[idx]), 6),
            }
        )

    return result


def compare_models(metrics_list: list[dict]) -> dict:
    """
    Compare multiple model metrics and determine the best model.

    Parameters
    ----------
    metrics_list : list[dict]
        List of evaluation results from evaluate_model().

    Returns
    -------
    dict with comparison data and best model recommendation.
    """
    comparison = []
    for m in metrics_list:
        comparison.append(
            {
                "model": m["model_name"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "auc": m["auc"],
            }
        )

    # Best model by F1 score
    best = max(metrics_list, key=lambda x: x["f1"])

    print_section("Model Comparison")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'-'*75}")
    for m in comparison:
        marker = " <-- Best" if m["model"] == best["model_name"] else ""
        print(
            f"  {m['model']:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}{marker}"
        )

    return {
        "comparison": comparison,
        "best_model": best["model_name"],
        "best_f1": best["f1"],
    }
