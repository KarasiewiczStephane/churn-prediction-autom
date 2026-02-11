"""Dataset download, validation, and sampling for Telco Customer Churn data."""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "customerID",
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
    "Churn",
]

EXPECTED_ROWS = 7043


class DataDownloader:
    """Handles downloading, validating, and sampling the Telco Churn dataset.

    Args:
        data_dir: Root directory for data storage.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.sample_dir = self.data_dir / "sample"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        """Download Telco Customer Churn dataset from Kaggle.

        Returns:
            Path to the downloaded CSV file.

        Raises:
            FileNotFoundError: If the expected CSV file is not found after download.
            RuntimeError: If Kaggle API authentication fails.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise RuntimeError(
                "kaggle package is required for download. "
                "Install it with: pip install kaggle"
            ) from e

        api = KaggleApi()
        api.authenticate()

        logger.info("Downloading Telco Customer Churn dataset...")
        api.dataset_download_files(
            "blastchar/telco-customer-churn",
            path=str(self.raw_dir),
            unzip=True,
        )

        csv_path = self.raw_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected file not found: {csv_path}")

        logger.info("Dataset downloaded to %s", csv_path)
        return csv_path

    def validate(self, csv_path: Path) -> pd.DataFrame:
        """Validate downloaded dataset structure and content.

        Args:
            csv_path: Path to the CSV file to validate.

        Returns:
            Validated DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        df = pd.read_csv(csv_path)

        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        if len(df) != EXPECTED_ROWS:
            logger.warning("Expected %d rows, got %d", EXPECTED_ROWS, len(df))

        churn_rate = df["Churn"].value_counts(normalize=True)
        logger.info("Churn rate: %s", churn_rate.to_dict())

        return df

    def create_sample(
        self, df: pd.DataFrame, n: int = 500, random_state: int = 42
    ) -> Path:
        """Create a stratified sample for CI tests.

        Args:
            df: Source DataFrame to sample from.
            n: Target number of rows in the sample.
            random_state: Random seed for reproducibility.

        Returns:
            Path to the saved sample CSV file.
        """
        frames = []
        for _, group in df.groupby("Churn"):
            count = max(1, int(len(group) * n / len(df)))
            frames.append(group.sample(n=count, random_state=random_state))
        sample = pd.concat(frames, ignore_index=True)
        sample_path = self.sample_dir / "telco_sample.csv"
        sample.to_csv(sample_path, index=False)
        logger.info("Created sample with %d rows at %s", len(sample), sample_path)
        return sample_path
