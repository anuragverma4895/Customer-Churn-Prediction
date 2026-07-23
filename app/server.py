"""
Flask API server for the Customer Churn Prediction dashboard.
Serves the dashboard, exported model metrics, and prediction endpoints.
"""

import csv
import os
import sys
from typing import Callable, Optional, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.predict import predict_single
from src.utils import get_data_path, get_output_path, load_json

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
CORS(app)


MODEL_LABEL_TO_KEY = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
}

CUSTOMER_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUMERIC_FEATURES = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}


def _load_output(filename: str) -> Union[dict, list]:
    """Load a JSON file from the outputs/ directory."""
    path = get_output_path(filename)
    if not os.path.exists(path):
        return {"error": f"File not found: {filename}. Run the training pipeline first."}
    return load_json(path)


def _best_model_key() -> str:
    """Return the saved model key selected by exported metrics."""
    metrics = _load_output("metrics.json")
    if isinstance(metrics, dict):
        return MODEL_LABEL_TO_KEY.get(metrics.get("best_model"), "logistic_regression")
    return "logistic_regression"


def _read_raw_customers() -> list[dict]:
    """Read real customer records from the local IBM Telco CSV."""
    path = get_data_path("telco_churn.csv")
    if not os.path.exists(path):
        return []

    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _number_or_zero(value: str) -> float:
    """Convert CSV numeric values, including blank TotalCharges, to floats."""
    cleaned = str(value).strip()
    return float(cleaned) if cleaned else 0.0


def _profile_from_row(label: str, row: dict) -> dict:
    """Convert one CSV row into a dashboard-ready example profile."""
    customer_data = {}
    for field in CUSTOMER_FEATURES:
        value = row.get(field, "")
        customer_data[field] = _number_or_zero(value) if field in NUMERIC_FEATURES else value

    return {
        "label": label,
        "customer_id": row.get("customerID"),
        "actual_churn": row.get("Churn"),
        "customer_data": customer_data,
    }


def _first_matching(rows: list[dict], predicate: Callable[[dict], bool]) -> Optional[dict]:
    """Return the first matching real record; no generated or random examples."""
    for row in rows:
        try:
            if predicate(row):
                return row
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _real_customer_examples() -> list[dict]:
    """Build a small deterministic set of real examples from the CSV."""
    rows = _read_raw_customers()
    if not rows:
        return []

    profile_rules = [
        (
            "Recent churned customer",
            lambda r: r["Churn"] == "Yes" and _number_or_zero(r["tenure"]) <= 3,
        ),
        (
            "Long-term retained customer",
            lambda r: r["Churn"] == "No"
            and _number_or_zero(r["tenure"]) >= 60
            and r["Contract"] == "Two year",
        ),
        (
            "Fiber month-to-month customer",
            lambda r: r["InternetService"] == "Fiber optic" and r["Contract"] == "Month-to-month",
        ),
        (
            "DSL customer with tech support",
            lambda r: r["InternetService"] == "DSL" and r["TechSupport"] == "Yes",
        ),
    ]

    examples = []
    seen_ids = set()
    for label, predicate in profile_rules:
        row = _first_matching(rows, predicate)
        customer_id = row.get("customerID") if row else None
        if row and customer_id not in seen_ids:
            examples.append(_profile_from_row(label, row))
            seen_ids.add(customer_id)

    return examples


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Basic health endpoint for deployment checks."""
    return jsonify({"status": "ok"})


@app.route("/api/metrics")
def api_metrics():
    """Return model evaluation metrics."""
    return jsonify(_load_output("metrics.json"))


@app.route("/api/feature-importance")
def api_feature_importance():
    """Return feature importance rankings."""
    return jsonify(_load_output("feature_importance.json"))


@app.route("/api/confusion-matrix")
def api_confusion_matrix():
    """Return confusion matrix data."""
    return jsonify(_load_output("confusion_matrix.json"))


@app.route("/api/roc-curve")
def api_roc_curve():
    """Return ROC curve coordinates."""
    return jsonify(_load_output("roc_data.json"))


@app.route("/api/dataset-stats")
def api_dataset_stats():
    """Return dataset distribution statistics."""
    return jsonify(_load_output("dataset_stats.json"))


@app.route("/api/customer-examples")
def api_customer_examples():
    """Return deterministic examples from the real IBM Telco CSV."""
    examples = _real_customer_examples()
    if not examples:
        return jsonify({"error": "No local dataset found. Run the training pipeline first."}), 404

    return jsonify(
        {
            "source": "IBM Telco Customer Churn CSV stored at data/telco_churn.csv",
            "examples": examples,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict churn probability for a single customer profile."""
    try:
        customer_data = request.get_json()
        if not customer_data:
            return jsonify({"error": "No JSON data provided"}), 400

        model_name = customer_data.pop("model", _best_model_key())
        result = predict_single(customer_data, model_name=model_name)
        result["model"] = model_name
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Customer Churn Prediction Dashboard")
    print("=" * 60)
    print("  Dashboard:  http://localhost:5000")
    print("  API Base:   http://localhost:5000/api/")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
