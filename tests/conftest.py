"""Shared test fixtures for the churn prediction test suite."""

import pandas as pd
import pytest


@pytest.fixture()
def sample_telco_df() -> pd.DataFrame:
    """Create a minimal Telco Churn DataFrame for testing.

    Returns:
        DataFrame with the expected Telco Churn schema and 20 rows.
    """
    data = {
        "customerID": [f"CUST-{i:04d}" for i in range(20)],
        "gender": ["Male", "Female"] * 10,
        "SeniorCitizen": [0, 1] * 10,
        "Partner": ["Yes", "No"] * 10,
        "Dependents": ["No", "Yes"] * 10,
        "tenure": list(range(1, 21)),
        "PhoneService": ["Yes"] * 20,
        "MultipleLines": ["No", "Yes", "No phone service"] * 6 + ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"] * 6 + ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No", "No internet service"] * 6 + ["Yes", "No"],
        "OnlineBackup": ["Yes", "No", "No internet service"] * 6 + ["Yes", "No"],
        "DeviceProtection": ["Yes", "No", "No internet service"] * 6 + ["Yes", "No"],
        "TechSupport": ["Yes", "No", "No internet service"] * 6 + ["Yes", "No"],
        "StreamingTV": ["Yes", "No", "No internet service"] * 6 + ["Yes", "No"],
        "StreamingMovies": ["Yes", "No", "No internet service"] * 6 + ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"] * 6
        + ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"] * 10,
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
        * 5,
        "MonthlyCharges": [29.85 + i * 5 for i in range(20)],
        "TotalCharges": [str(29.85 + i * 50) for i in range(20)],
        "Churn": ["No"] * 14 + ["Yes"] * 6,
    }
    return pd.DataFrame(data)
