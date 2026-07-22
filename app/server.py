"""
Flask API Server for Customer Churn Prediction Dashboard.
Serves prediction endpoints and static dashboard files.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from typing import Union

from src.utils import load_json, get_output_path
from src.predict import predict_single

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
CORS(app)


# ─── Helper: Load pre-computed JSON ───

def _load_output(filename: str) -> Union[dict, list]:
    """Load a JSON file from the outputs/ directory."""
    path = get_output_path(filename)
    if not os.path.exists(path):
        return {"error": f"File not found: {filename}. Run the training pipeline first."}
    return load_json(path)


# ─── Routes ───

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


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


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict churn probability for a single customer.

    Expects JSON body with customer features:
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2
    }
    """
    try:
        customer_data = request.get_json()
        if not customer_data:
            return jsonify({"error": "No JSON data provided"}), 400

        model_name = customer_data.pop("model", "random_forest")
        result = predict_single(customer_data, model_name=model_name)
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ─── Main ───

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Customer Churn Prediction Dashboard")
    print("=" * 60)
    print(f"  Dashboard:  http://localhost:5000")
    print(f"  API Base:   http://localhost:5000/api/")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
