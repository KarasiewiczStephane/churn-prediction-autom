"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor, PreprocessedData


class TestDataPreprocessor:
    """Tests for the DataPreprocessor class."""

    def test_handle_missing_converts_total_charges(self, sample_telco_df):
        """_handle_missing() should convert TotalCharges to numeric."""
        df = sample_telco_df.copy()
        df.loc[0, "TotalCharges"] = " "

        preprocessor = DataPreprocessor()
        result = preprocessor._handle_missing(df)

        assert result["TotalCharges"].dtype in [np.float64, np.float32]
        assert not result["TotalCharges"].isna().any()

    def test_encode_categoricals(self, sample_telco_df):
        """_encode_categoricals() should transform all categorical columns."""
        preprocessor = DataPreprocessor()
        preprocessor._identify_columns(sample_telco_df, "Churn")

        X = sample_telco_df.drop(columns=["Churn", "customerID"])
        X_encoded = preprocessor._encode_categoricals(X)

        for col in preprocessor.categorical_cols:
            if col in X_encoded.columns:
                assert X_encoded[col].dtype in [np.int64, np.int32]

        assert len(preprocessor.label_encoders) > 0

    def test_split_ratios(self, sample_telco_df):
        """Train/val/test split should approximate 60/20/20 ratios."""
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(sample_telco_df)

        total = len(data.X_train) + len(data.X_val) + len(data.X_test)
        train_ratio = len(data.X_train) / total
        val_ratio = len(data.X_val) / total
        test_ratio = len(data.X_test) / total

        assert 0.5 <= train_ratio <= 0.7
        assert 0.15 <= val_ratio <= 0.3
        assert 0.15 <= test_ratio <= 0.3

    def test_stratification_preserves_churn_rate(self, sample_telco_df):
        """Stratification should maintain approximate churn rate across splits."""
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(sample_telco_df)

        original_rate = (sample_telco_df["Churn"] == "Yes").mean()
        train_rate = data.y_train.mean()
        val_rate = data.y_val.mean()
        test_rate = data.y_test.mean()

        tolerance = 0.2
        assert abs(train_rate - original_rate) < tolerance
        assert abs(val_rate - original_rate) < tolerance
        assert abs(test_rate - original_rate) < tolerance

    def test_scale_numerical_zero_mean(self, sample_telco_df):
        """Scaled training set numerical features should have mean close to 0."""
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(sample_telco_df)

        for col in preprocessor.numerical_cols:
            if col in data.X_train.columns:
                assert abs(data.X_train[col].mean()) < 0.5

    def test_save_and_load_roundtrip(self, sample_telco_df, tmp_path):
        """Saved parquet files should be loadable and match original data."""
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(sample_telco_df)
        preprocessor.save(data, str(tmp_path / "processed"))

        output_dir = tmp_path / "processed" / "v1"
        X_train_loaded = pd.read_parquet(output_dir / "X_train.parquet")
        y_train_loaded = pd.read_parquet(output_dir / "y_train.parquet")

        assert len(X_train_loaded) == len(data.X_train)
        assert len(y_train_loaded) == len(data.y_train)

    def test_transform_new_data(self, sample_telco_df):
        """transform() should apply fitted encoders to new data."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_telco_df)

        new_data = sample_telco_df.head(5).copy()
        transformed = preprocessor.transform(new_data)

        assert "customerID" not in transformed.columns
        assert "Churn" not in transformed.columns
        assert len(transformed) == 5

    def test_transform_raises_when_not_fitted(self, sample_telco_df):
        """transform() should raise ValueError if not fitted."""
        preprocessor = DataPreprocessor()
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.transform(sample_telco_df)

    def test_no_data_leakage_scaler(self, sample_telco_df):
        """Scaler should be fit only on training data."""
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(sample_telco_df)

        assert preprocessor.scaler.n_features_in_ == len(preprocessor.numerical_cols)
        assert preprocessor.scaler.n_samples_seen_ == len(data.X_train)

    def test_preprocessed_data_fields(self, sample_telco_df):
        """PreprocessedData should have all expected fields populated."""
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(sample_telco_df)

        assert isinstance(data, PreprocessedData)
        assert isinstance(data.X_train, pd.DataFrame)
        assert isinstance(data.y_train, pd.Series)
        assert len(data.feature_names) > 0
        assert isinstance(data.encoders, dict)
        assert isinstance(data.scaler, type(preprocessor.scaler))

    def test_feature_names_match_columns(self, sample_telco_df):
        """feature_names should match the columns of X_train."""
        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(sample_telco_df)

        assert data.feature_names == list(data.X_train.columns)
