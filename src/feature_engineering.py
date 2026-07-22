"""
Feature Engineering for Customer Churn Prediction.
Creates derived features and provides feature analysis utilities.
"""

import numpy as np
import pandas as pd


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'TenureBucket' column categorizing tenure into groups.

    Buckets: 0-12, 13-24, 25-48, 49-60, 61-72
    """
    df = df.copy()
    bins = [0, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    df["TenureBucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    return df


def add_avg_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'AvgMonthlyCharge' = TotalCharges / tenure.
    For tenure=0, uses MonthlyCharges as fallback.
    """
    df = df.copy()
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )
    df["AvgMonthlyCharge"] = df["AvgMonthlyCharge"].round(2)
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'ServiceCount' — number of active services per customer.
    Counts: PhoneService, InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
    """
    df = df.copy()
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    count = (df["PhoneService"] == "Yes").astype(int)
    count += (df["InternetService"] != "No").astype(int)
    for col in service_cols:
        if col in df.columns:
            count += (df[col] == "Yes").astype(int)
    df["ServiceCount"] = count
    return df


def compute_correlation_with_target(
    df_encoded: pd.DataFrame, target_col: str = "Churn"
) -> pd.Series:
    """
    Compute Pearson correlation of all features with the target variable.

    Parameters
    ----------
    df_encoded : pd.DataFrame
        Encoded (numeric) DataFrame including the target column.
    target_col : str
        Name of the target column.

    Returns
    -------
    pd.Series
        Correlations sorted by absolute value (descending).
    """
    corr = df_encoded.corr()[target_col].drop(target_col)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the dashboard.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned (but not encoded) DataFrame.

    Returns
    -------
    dict with dataset statistics.
    """
    total = len(df)
    churn_yes = int((df["Churn"] == "Yes").sum())
    churn_no = total - churn_yes
    churn_rate = round(churn_yes / total * 100, 1)

    # Demographics
    gender_dist = df["gender"].value_counts().to_dict()
    senior_dist = int(df["SeniorCitizen"].sum())
    partner_dist = df["Partner"].value_counts().to_dict()

    # Contract distribution
    contract_dist = df["Contract"].value_counts().to_dict()

    # Internet service distribution
    internet_dist = df["InternetService"].value_counts().to_dict()

    # Payment method distribution
    payment_dist = df["PaymentMethod"].value_counts().to_dict()

    # Numeric summaries
    tenure_stats = {
        "mean": round(df["tenure"].mean(), 1),
        "median": int(df["tenure"].median()),
        "std": round(df["tenure"].std(), 1),
    }
    monthly_stats = {
        "mean": round(df["MonthlyCharges"].mean(), 2),
        "median": round(df["MonthlyCharges"].median(), 2),
        "std": round(df["MonthlyCharges"].std(), 2),
    }

    return {
        "total_customers": total,
        "churn_yes": churn_yes,
        "churn_no": churn_no,
        "churn_rate": churn_rate,
        "gender_distribution": gender_dist,
        "senior_citizens": senior_dist,
        "partner_distribution": partner_dist,
        "contract_distribution": contract_dist,
        "internet_distribution": internet_dist,
        "payment_distribution": payment_dist,
        "tenure_stats": tenure_stats,
        "monthly_charges_stats": monthly_stats,
    }
