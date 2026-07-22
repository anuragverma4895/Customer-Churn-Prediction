"""
Synthetic Telco Customer Churn Dataset Generator
Generates 7,043 rows mirroring the Kaggle Telco Customer Churn dataset schema.
Realistic distributions for demographics, services, and billing.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def generate_customer_id(n: int) -> list[str]:
    """Generate unique customer IDs in format XXXX-XXXXX."""
    rng = np.random.default_rng(42)
    ids = set()
    while len(ids) < n:
        part1 = "".join(rng.choice(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 4))
        part2 = "".join(rng.choice(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 5))
        ids.add(f"{part1}-{part2}")
    return sorted(ids)[:n]


def generate_dataset(n_samples: int = 7043, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Telco Customer Churn dataset.

    Parameters
    ----------
    n_samples : int
        Number of customer records to generate (default 7043).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 21 columns matching the Kaggle Telco schema.
    """
    rng = np.random.default_rng(seed)

    # --- Demographics ---
    gender = rng.choice(["Male", "Female"], n_samples, p=[0.505, 0.495])
    senior_citizen = rng.choice([0, 1], n_samples, p=[0.838, 0.162])
    partner = rng.choice(["Yes", "No"], n_samples, p=[0.484, 0.516])
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.299, 0.701])

    # --- Tenure (months, 1-72) ---
    # Bimodal distribution: many new + many long-term customers
    tenure_mix = rng.choice([0, 1], n_samples, p=[0.4, 0.6])
    tenure = np.where(
        tenure_mix == 0,
        np.clip(rng.exponential(8, n_samples).astype(int), 1, 72),
        np.clip(rng.normal(50, 15, n_samples).astype(int), 1, 72),
    )

    # --- Services ---
    phone_service = rng.choice(["Yes", "No"], n_samples, p=[0.903, 0.097])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n_samples, p=[0.422, 0.578]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.344, 0.440, 0.216]
    )

    # Services that depend on having internet
    def internet_dependent(p_yes: float) -> np.ndarray:
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], n_samples, p=[p_yes, 1 - p_yes]),
        )

    online_security = internet_dependent(0.286)
    online_backup = internet_dependent(0.343)
    device_protection = internet_dependent(0.341)
    tech_support = internet_dependent(0.290)
    streaming_tv = internet_dependent(0.384)
    streaming_movies = internet_dependent(0.388)

    # --- Contract & Billing ---
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.551, 0.209, 0.240],
    )
    paperless_billing = rng.choice(["Yes", "No"], n_samples, p=[0.593, 0.407])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_samples,
        p=[0.336, 0.228, 0.219, 0.217],
    )

    # --- Monthly Charges ---
    # Correlated with internet service type
    base_charge = np.where(
        internet_service == "No",
        rng.normal(22, 3, n_samples),
        np.where(
            internet_service == "DSL",
            rng.normal(55, 15, n_samples),
            rng.normal(80, 18, n_samples),  # Fiber optic
        ),
    )
    monthly_charges = np.clip(np.round(base_charge, 2), 18.25, 118.75)

    # --- Total Charges ---
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n_samples), 2)
    total_charges = np.clip(total_charges, 18.80, 8700.0)

    # --- Churn (target) ---
    # Churn probability influenced by contract, tenure, charges, internet type
    churn_logit = (
        -1.2
        + 1.5 * (contract == "Month-to-month").astype(float)
        - 0.8 * (contract == "Two year").astype(float)
        - 0.02 * tenure
        + 0.008 * monthly_charges
        + 0.6 * (internet_service == "Fiber optic").astype(float)
        - 0.4 * (online_security == "Yes").astype(float)
        - 0.4 * (tech_support == "Yes").astype(float)
        + 0.3 * (paperless_billing == "Yes").astype(float)
        + 0.5 * (payment_method == "Electronic check").astype(float)
        + 0.3 * senior_citizen.astype(float)
        + rng.normal(0, 0.5, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(rng.random(n_samples) < churn_prob, "Yes", "No")

    # --- Assemble DataFrame ---
    df = pd.DataFrame(
        {
            "customerID": generate_customer_id(n_samples),
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    # Introduce a few whitespace-only TotalCharges values (mimics real dataset quirk)
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    blank_idx = rng.choice(df.index, size=11, replace=False)
    df.loc[blank_idx, "TotalCharges"] = " "

    return df


def main():
    """Generate and save the dataset."""
    output_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "telco_churn.csv")

    print("Generating synthetic Telco Customer Churn dataset...")
    df = generate_dataset()
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")


if __name__ == "__main__":
    main()
