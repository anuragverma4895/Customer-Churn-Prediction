"""
Prediction Interface for Customer Churn Prediction.
Loads trained models and provides single/batch prediction capabilities.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_model_path


def load_prediction_artifacts(model_name: str = "random_forest") -> tuple:
    """
    Load trained model, scaler, and feature names for prediction.

    Parameters
    ----------
    model_name : str
        Name of the model file (without .pkl extension).

    Returns
    -------
    tuple of (model, scaler, feature_names)
    """
    model = joblib.load(get_model_path(f"{model_name}.pkl"))
    scaler = joblib.load(get_model_path("scaler.pkl"))
    feature_names = joblib.load(get_model_path("feature_names.pkl"))
    return model, scaler, feature_names


def prepare_single_input(customer_data: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Prepare a single customer's data for prediction.

    Parameters
    ----------
    customer_data : dict
        Raw customer feature values (pre-encoding values).
    feature_names : list[str]
        Expected feature names after encoding.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame ready for scaling and prediction.
    """
    # Create a DataFrame with all expected feature columns, initialized to 0
    row = pd.DataFrame(0, index=[0], columns=feature_names)

    # Map simple numeric fields
    numeric_fields = {
        "SeniorCitizen": "SeniorCitizen",
        "tenure": "tenure",
        "MonthlyCharges": "MonthlyCharges",
        "TotalCharges": "TotalCharges",
    }
    for raw_key, col_name in numeric_fields.items():
        if raw_key in customer_data and col_name in row.columns:
            row[col_name] = float(customer_data[raw_key])

    # Map binary encoded fields
    binary_map = {
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
    }
    for raw_key, mapping in binary_map.items():
        if raw_key in customer_data and raw_key in row.columns:
            row[raw_key] = mapping.get(customer_data[raw_key], 0)

    # Map one-hot encoded fields
    onehot_fields = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    for field in onehot_fields:
        if field in customer_data:
            col_name = f"{field}_{customer_data[field]}"
            if col_name in row.columns:
                row[col_name] = 1

    return row


def predict_single(
    customer_data: dict,
    model_name: str = "random_forest",
) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer_data : dict
        Raw customer feature values.
    model_name : str
        Which trained model to use.

    Returns
    -------
    dict with keys: churn_probability, churn_prediction, risk_level
    """
    model, scaler, feature_names = load_prediction_artifacts(model_name)

    # Prepare input
    input_df = prepare_single_input(customer_data, feature_names)
    input_scaled = scaler.transform(input_df.values)

    # Predict
    prob = float(model.predict_proba(input_scaled)[0][1])
    pred = int(model.predict(input_scaled)[0])

    # Risk level
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": pred,
        "churn_label": "Yes" if pred == 1 else "No",
        "risk_level": risk,
    }


def predict_batch(
    df: pd.DataFrame,
    model_name: str = "random_forest",
) -> pd.DataFrame:
    """
    Predict churn for a batch of customers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with customer features (raw or encoded).
    model_name : str
        Which trained model to use.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added prediction columns.
    """
    model, scaler, feature_names = load_prediction_artifacts(model_name)

    results = []
    for _, row in df.iterrows():
        result = predict_single(row.to_dict(), model_name)
        results.append(result)

    predictions_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
