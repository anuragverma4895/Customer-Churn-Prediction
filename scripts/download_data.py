"""
Download the Real IBM Telco Customer Churn Dataset.
Source: IBM Sample Data Sets (Kaggle / IBM GitHub).
Run: python scripts/download_data.py
"""

import os
import sys
import urllib.request
import urllib.error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Official IBM GitHub raw URL for the Telco Customer Churn dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

EXPECTED_ROWS = 7043
EXPECTED_COLS = 21
REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def download_dataset(output_path: str) -> str:
    """
    Download the real Telco Customer Churn CSV from IBM's GitHub.

    Parameters
    ----------
    output_path : str
        Destination file path for the CSV.

    Returns
    -------
    str
        Path to the downloaded file.

    Raises
    ------
    RuntimeError
        If download fails or validation fails.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Downloading from: {DATASET_URL}")
    print(f"  Saving to:        {output_path}")

    try:
        urllib.request.urlretrieve(DATASET_URL, output_path)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download dataset. Check your internet connection.\n"
            f"  URL: {DATASET_URL}\n"
            f"  Error: {e}"
        )

    # Validate the downloaded file
    validate_dataset(output_path)
    return output_path


def validate_dataset(filepath: str) -> None:
    """
    Validate the downloaded CSV matches expected schema.

    Checks:
    - File exists and is non-empty
    - Has exactly 7043 rows and 21 columns
    - Contains all required column names
    """
    import pandas as pd

    if not os.path.exists(filepath):
        raise RuntimeError(f"Downloaded file not found: {filepath}")

    file_size = os.path.getsize(filepath)
    if file_size < 100_000:  # Should be ~955 KB
        raise RuntimeError(
            f"Downloaded file is too small ({file_size} bytes). "
            f"Expected ~955 KB. File may be corrupted."
        )

    df = pd.read_csv(filepath)

    # Check shape
    n_rows, n_cols = df.shape
    if n_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Row count mismatch: got {n_rows}, expected {EXPECTED_ROWS}"
        )
    if n_cols != EXPECTED_COLS:
        raise RuntimeError(
            f"Column count mismatch: got {n_cols}, expected {EXPECTED_COLS}"
        )

    # Check columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Check churn rate is approximately 26.5%
    churn_rate = (df["Churn"] == "Yes").mean()
    if not (0.24 <= churn_rate <= 0.29):
        raise RuntimeError(
            f"Churn rate is {churn_rate:.1%}, expected ~26.5%. "
            f"Data may not be the real IBM Telco dataset."
        )

    print(f"  [OK] Validation passed: {n_rows} rows x {n_cols} columns")
    print(f"  [OK] Churn rate: {churn_rate:.1%} (expected ~26.5%)")


def main():
    """Download and validate the real dataset."""
    output_dir = os.path.join(PROJECT_ROOT, "data")
    output_path = os.path.join(output_dir, "telco_churn.csv")

    print("Downloading real IBM Telco Customer Churn dataset...")
    download_dataset(output_path)
    print("Dataset ready.")


if __name__ == "__main__":
    main()
