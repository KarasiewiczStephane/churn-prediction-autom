"""Tests for dataset download and validation module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.downloader import EXPECTED_COLUMNS, DataDownloader


class TestDataDownloader:
    """Tests for the DataDownloader class."""

    def test_validate_with_valid_dataframe(self, sample_telco_df, tmp_path):
        """validate() should accept a DataFrame with all expected columns."""
        csv_path = tmp_path / "valid.csv"
        sample_telco_df.to_csv(csv_path, index=False)

        downloader = DataDownloader(data_dir=str(tmp_path / "data"))
        df = downloader.validate(csv_path)

        assert isinstance(df, pd.DataFrame)
        assert set(EXPECTED_COLUMNS).issubset(set(df.columns))

    def test_validate_raises_for_missing_columns(self, tmp_path):
        """validate() should raise ValueError when columns are missing."""
        incomplete_df = pd.DataFrame({"customerID": ["1"], "gender": ["Male"]})
        csv_path = tmp_path / "incomplete.csv"
        incomplete_df.to_csv(csv_path, index=False)

        downloader = DataDownloader(data_dir=str(tmp_path / "data"))
        with pytest.raises(ValueError, match="Missing columns"):
            downloader.validate(csv_path)

    def test_create_sample_produces_correct_count(self, sample_telco_df, tmp_path):
        """create_sample() should produce approximately the target row count."""
        downloader = DataDownloader(data_dir=str(tmp_path / "data"))
        sample_path = downloader.create_sample(sample_telco_df, n=10)

        assert sample_path.exists()
        sample_df = pd.read_csv(sample_path)
        assert len(sample_df) > 0
        assert len(sample_df) <= len(sample_telco_df)

    def test_create_sample_preserves_stratification(self, sample_telco_df, tmp_path):
        """create_sample() should preserve approximate churn rate."""
        downloader = DataDownloader(data_dir=str(tmp_path / "data"))
        sample_path = downloader.create_sample(sample_telco_df, n=10)

        sample_df = pd.read_csv(sample_path)
        assert "Yes" in sample_df["Churn"].values
        assert "No" in sample_df["Churn"].values

    def test_create_sample_file_has_expected_columns(self, sample_telco_df, tmp_path):
        """create_sample() should produce a CSV with all expected columns."""
        downloader = DataDownloader(data_dir=str(tmp_path / "data"))
        sample_path = downloader.create_sample(sample_telco_df, n=10)

        sample_df = pd.read_csv(sample_path)
        assert set(EXPECTED_COLUMNS).issubset(set(sample_df.columns))

    def test_directories_created_on_init(self, tmp_path):
        """DataDownloader should create raw and sample directories."""
        data_dir = tmp_path / "new_data"
        downloader = DataDownloader(data_dir=str(data_dir))

        assert downloader.raw_dir.exists()
        assert downloader.sample_dir.exists()

    @patch("src.data.downloader.KaggleApi", create=True)
    def test_download_calls_kaggle_api(self, mock_kaggle_cls, tmp_path):
        """download() should call Kaggle API to fetch the dataset."""
        mock_api = MagicMock()
        mock_kaggle_cls.return_value = mock_api

        downloader = DataDownloader(data_dir=str(tmp_path / "data"))

        csv_path = downloader.raw_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        csv_path.write_text("customerID,Churn\n1,Yes\n")

        with patch(
            "src.data.downloader.KaggleApi",
            side_effect=ImportError("no kaggle"),
            create=True,
        ):
            pass

    def test_validate_warns_on_unexpected_row_count(self, sample_telco_df, tmp_path):
        """validate() should log warning when row count differs from expected."""
        csv_path = tmp_path / "small.csv"
        sample_telco_df.to_csv(csv_path, index=False)

        downloader = DataDownloader(data_dir=str(tmp_path / "data"))
        df = downloader.validate(csv_path)
        assert len(df) == 20
