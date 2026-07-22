"""
Data Preprocessing Pipeline for Customer Churn Prediction.
Handles loading, cleaning, encoding, scaling, and train/test splitting.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_data_path


# Columns that will be label-encoded (binary Yes/No)
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# Columns that will be one-hot encoded (multi-category)
ONEHOT_COLS = [
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

# Numeric columns to scale
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(filepath: str | None = None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to data/telco_churn.csv.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = get_data_path("telco_churn.csv")
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Convert TotalCharges whitespace to NaN, then to float
    - Fill NaN TotalCharges with median
    - Drop customerID (not a feature)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df = df.copy()

    # Fix TotalCharges: whitespace → NaN → float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

    # Drop customerID — it's an identifier, not a feature
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
        
    df = df.fillna(0)

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features:
    - Binary columns: LabelEncoder (Yes=1, No=0, Male=1, Female=0)
    - Multi-category columns: One-Hot Encoding (drop_first=True)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Encoded DataFrame and dict of LabelEncoders for inverse transforms.
    """
    df = df.copy()
    encoders = {}

    # Label encode binary columns
    for col in BINARY_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # One-hot encode multi-category columns
    existing_onehot = [c for c in ONEHOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_onehot, drop_first=True, dtype=int)

    return df, encoders


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply StandardScaler to numeric features.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple[np.ndarray, np.ndarray, StandardScaler]
        Scaled arrays and fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    filepath: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline: load → clean → engineer → encode → split → scale.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file.
    test_size : float
        Fraction of data for test set.
    random_state : int
        Random seed.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test : arrays
        scaler : StandardScaler
        feature_names : list[str]
        encoders : dict of LabelEncoders
        df_clean : pd.DataFrame (pre-encoded, for stats)
    """
    # Load and clean
    df_raw = load_data(filepath)
    df = clean_data(df_raw)
    df_clean = df.copy()

    # Encode
    df_encoded, encoders = encode_features(df)

    # Separate features and target
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]
    feature_names = list(X.columns)

    # Train/test split (stratified to preserve churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "scaler": scaler,
        "feature_names": feature_names,
        "encoders": encoders,
        "df_clean": df_clean,
    }
