"""Data preprocessing pipeline with encoding, scaling, and splitting."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessedData:
    """Container for preprocessed train/validation/test splits.

    Attributes:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_val: Validation target vector.
        y_test: Test target vector.
        feature_names: List of feature column names.
        encoders: Dictionary mapping column names to fitted LabelEncoders.
        scaler: Fitted StandardScaler instance.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    encoders: dict = field(default_factory=dict)
    scaler: StandardScaler = field(default_factory=StandardScaler)


class DataPreprocessor:
    """Handles data cleaning, encoding, scaling, and splitting.

    Args:
        random_state: Random seed for reproducible splits.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.numerical_cols: list[str] = []
        self.categorical_cols: list[str] = []

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "Churn"
    ) -> PreprocessedData:
        """Run the full preprocessing pipeline.

        Args:
            df: Raw input DataFrame.
            target_col: Name of the target column.

        Returns:
            PreprocessedData with train/validation/test splits.
        """
        df = df.copy()

        df = self._handle_missing(df)
        self._identify_columns(df, target_col)

        y = (df[target_col] == "Yes").astype(int)
        X = df.drop(columns=[target_col, "customerID"])

        X = self._encode_categoricals(X)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            stratify=y_temp,
            random_state=self.random_state,
        )

        X_train, X_val, X_test = self._scale_numerical(X_train, X_val, X_test)

        logger.info(
            "Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test)
        )

        return PreprocessedData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=list(X_train.columns),
            encoders=self.label_encoders,
            scaler=self.scaler,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders and scaler.

        Args:
            df: New DataFrame to transform (must have same columns as training data).

        Returns:
            Transformed DataFrame ready for prediction.

        Raises:
            ValueError: If the preprocessor has not been fitted yet.
        """
        if not self.label_encoders:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")

        df = df.copy()
        df = self._handle_missing(df)

        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])

        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        if self.numerical_cols:
            num_cols_present = [c for c in self.numerical_cols if c in df.columns]
            df[num_cols_present] = self.scaler.transform(df[num_cols_present])

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.

        TotalCharges contains whitespace strings representing missing values.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        return df

    def _identify_columns(self, df: pd.DataFrame, target_col: str) -> None:
        """Identify numerical and categorical columns.

        Args:
            df: Input DataFrame.
            target_col: Name of the target column to exclude.
        """
        exclude = {target_col, "customerID"}
        self.numerical_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]
        self.categorical_cols = [
            c for c in df.select_dtypes(include=["object"]).columns if c not in exclude
        ]

    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Label encode categorical columns.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with categorical columns encoded as integers.
        """
        for col in self.categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        return X

    def _scale_numerical(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Standardize numerical features using training set statistics.

        Args:
            X_train: Training feature matrix.
            X_val: Validation feature matrix.
            X_test: Test feature matrix.

        Returns:
            Tuple of scaled (X_train, X_val, X_test) DataFrames.
        """
        if not self.numerical_cols:
            return X_train, X_val, X_test

        num_cols = [c for c in self.numerical_cols if c in X_train.columns]
        self.scaler.fit(X_train[num_cols])

        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        X_train[num_cols] = self.scaler.transform(X_train[num_cols])
        X_val[num_cols] = self.scaler.transform(X_val[num_cols])
        X_test[num_cols] = self.scaler.transform(X_test[num_cols])

        return X_train, X_val, X_test

    def save(
        self, data: PreprocessedData, output_dir: str, version: str = "v1"
    ) -> None:
        """Save preprocessed data as parquet files.

        Args:
            data: PreprocessedData to save.
            output_dir: Base output directory.
            version: Version string for subdirectory naming.
        """
        output_path = Path(output_dir) / version
        output_path.mkdir(parents=True, exist_ok=True)

        data.X_train.to_parquet(output_path / "X_train.parquet")
        data.X_val.to_parquet(output_path / "X_val.parquet")
        data.X_test.to_parquet(output_path / "X_test.parquet")
        pd.DataFrame({"y": data.y_train}).to_parquet(output_path / "y_train.parquet")
        pd.DataFrame({"y": data.y_val}).to_parquet(output_path / "y_val.parquet")
        pd.DataFrame({"y": data.y_test}).to_parquet(output_path / "y_test.parquet")

        logger.info("Saved preprocessed data to %s", output_path)
