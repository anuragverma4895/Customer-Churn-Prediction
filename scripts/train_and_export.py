"""
Full Training & Export Pipeline
Run: python scripts/train_and_export.py

Generates data (if needed) → preprocesses → trains models → evaluates → exports JSON.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import joblib
from src.utils import get_data_path, get_model_path, get_output_path, save_json, print_section
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import get_dataset_stats
from src.model_training import (
    train_logistic_regression,
    train_random_forest,
    save_model,
)
from src.model_evaluation import (
    evaluate_model,
    get_feature_importance,
    compare_models,
)


def run_pipeline():
    """Execute the full training and export pipeline."""

    # ─── Step 1: Generate data if it doesn't exist ───
    data_path = get_data_path("telco_churn.csv")
    if not os.path.exists(data_path):
        print_section("Step 1: Generating Dataset")
        from scripts.generate_data import main as generate_main
        generate_main()
    else:
        print_section("Step 1: Dataset Found")
        print(f"  Using existing dataset: {data_path}")

    # ─── Step 2: Preprocess ───
    print_section("Step 2: Preprocessing Data")
    pipeline_data = preprocess_pipeline(data_path)

    X_train = pipeline_data["X_train"]
    X_test = pipeline_data["X_test"]
    y_train = pipeline_data["y_train"]
    y_test = pipeline_data["y_test"]
    feature_names = pipeline_data["feature_names"]
    scaler = pipeline_data["scaler"]
    df_clean = pipeline_data["df_clean"]

    print(f"  Training set:  {X_train.shape}")
    print(f"  Test set:      {X_test.shape}")
    print(f"  Features:      {len(feature_names)}")

    # Save scaler and feature names for prediction
    joblib.dump(scaler, get_model_path("scaler.pkl"))
    joblib.dump(feature_names, get_model_path("feature_names.pkl"))
    print(f"  Scaler saved:  {get_model_path('scaler.pkl')}")

    # ─── Step 3: Train Models ───
    lr_model = train_logistic_regression(X_train, y_train, cv=5, tune=True)
    save_model(lr_model, "logistic_regression")

    rf_model = train_random_forest(X_train, y_train, cv=5, tune=True)
    save_model(rf_model, "random_forest")

    # ─── Step 4: Evaluate Models ───
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # ─── Step 5: Compare Models ───
    comparison = compare_models([lr_metrics, rf_metrics])

    # ─── Step 6: Feature Importances ───
    print_section("Feature Importances")
    lr_importance = get_feature_importance(
        lr_model, feature_names, model_type="logistic_regression", top_n=15
    )
    rf_importance = get_feature_importance(
        rf_model, feature_names, model_type="random_forest", top_n=15
    )

    print("\n  Top 10 Features (Random Forest):")
    for i, feat in enumerate(rf_importance[:10], 1):
        bar = "=" * int(feat["importance"] * 30)
        print(f"    {i:2d}. {feat['feature']:<35} {feat['importance']:.4f}  {bar}")

    # ─── Step 7: Dataset Statistics ───
    print_section("Dataset Statistics")
    dataset_stats = get_dataset_stats(df_clean)
    print(f"  Total customers:  {dataset_stats['total_customers']}")
    print(f"  Churn rate:       {dataset_stats['churn_rate']}%")

    # ─── Step 8: Export All JSON ───
    print_section("Exporting Results")

    # Metrics
    metrics_export = {
        "logistic_regression": {
            "accuracy": lr_metrics["accuracy"],
            "precision": lr_metrics["precision"],
            "recall": lr_metrics["recall"],
            "f1": lr_metrics["f1"],
            "auc": lr_metrics["auc"],
        },
        "random_forest": {
            "accuracy": rf_metrics["accuracy"],
            "precision": rf_metrics["precision"],
            "recall": rf_metrics["recall"],
            "f1": rf_metrics["f1"],
            "auc": rf_metrics["auc"],
        },
        "best_model": comparison["best_model"],
        "best_f1": comparison["best_f1"],
    }
    save_json(metrics_export, get_output_path("metrics.json"))
    print(f"  Saved: metrics.json")

    # Feature importance
    importance_export = {
        "logistic_regression": lr_importance,
        "random_forest": rf_importance,
    }
    save_json(importance_export, get_output_path("feature_importance.json"))
    print(f"  Saved: feature_importance.json")

    # Confusion matrices
    cm_export = {
        "logistic_regression": {
            "matrix": lr_metrics["confusion_matrix"],
            "labels": ["No Churn", "Churn"],
        },
        "random_forest": {
            "matrix": rf_metrics["confusion_matrix"],
            "labels": ["No Churn", "Churn"],
        },
    }
    save_json(cm_export, get_output_path("confusion_matrix.json"))
    print(f"  Saved: confusion_matrix.json")

    # ROC curve data
    roc_export = {
        "logistic_regression": {
            "fpr": lr_metrics["roc_fpr"],
            "tpr": lr_metrics["roc_tpr"],
            "auc": lr_metrics["auc"],
        },
        "random_forest": {
            "fpr": rf_metrics["roc_fpr"],
            "tpr": rf_metrics["roc_tpr"],
            "auc": rf_metrics["auc"],
        },
    }
    save_json(roc_export, get_output_path("roc_data.json"))
    print(f"  Saved: roc_data.json")

    # Dataset stats
    save_json(dataset_stats, get_output_path("dataset_stats.json"))
    print(f"  Saved: dataset_stats.json")

    # ─── Done ───
    print_section("Pipeline Complete")
    print(f"  Best model: {comparison['best_model']} (F1: {comparison['best_f1']:.4f})")
    print(f"  All outputs saved to: {os.path.join(PROJECT_ROOT, 'outputs')}")
    print(f"\n  To start the dashboard:")
    print(f"    python app/server.py")
    print()


if __name__ == "__main__":
    run_pipeline()
