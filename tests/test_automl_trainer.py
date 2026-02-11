"""Tests for H2O AutoML training pipeline (mocked)."""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.models.automl_trainer import AutoMLResult, H2OAutoMLTrainer


@pytest.fixture()
def mock_h2o():
    """Create a mock H2O module with expected interfaces."""
    mock_h2o_mod = MagicMock()
    mock_h2o_automl_mod = MagicMock()

    with (
        patch.dict(
            sys.modules,
            {"h2o": mock_h2o_mod, "h2o.automl": mock_h2o_automl_mod},
        ),
        patch("src.models.automl_trainer.H2OAutoMLTrainer._init_h2o"),
    ):
        trainer = H2OAutoMLTrainer(max_runtime_secs=60, seed=42)
        trainer._h2o = mock_h2o_mod

        yield trainer, mock_h2o_mod, mock_h2o_automl_mod


@pytest.fixture()
def sample_automl_result():
    """Create a sample AutoMLResult for testing."""
    leaderboard = pd.DataFrame(
        {
            "model_id": ["GBM_1", "DRF_1", "GLM_1"],
            "auc": [0.85, 0.82, 0.78],
            "logloss": [0.42, 0.45, 0.50],
            "aucpr": [0.70, 0.65, 0.60],
            "mean_per_class_error": [0.20, 0.22, 0.25],
            "rmse": [0.38, 0.40, 0.43],
            "mse": [0.14, 0.16, 0.18],
        }
    )
    return AutoMLResult(
        leaderboard=leaderboard,
        best_model=MagicMock(),
        best_model_id="GBM_1",
        training_time_secs=60.5,
        all_model_ids=["GBM_1", "DRF_1", "GLM_1"],
    )


class TestH2OAutoMLTrainer:
    """Tests for the H2OAutoMLTrainer class."""

    def test_init_parameters(self):
        """Trainer should store configuration parameters."""
        trainer = H2OAutoMLTrainer(max_runtime_secs=120, seed=99, nfolds=3)
        assert trainer.max_runtime_secs == 120
        assert trainer.seed == 99
        assert trainer.nfolds == 3

    def test_train_returns_automl_result(self, mock_h2o):
        """train() should return an AutoMLResult."""
        trainer, h2o_module, h2o_automl_mod = mock_h2o

        mock_frame = MagicMock()
        h2o_module.H2OFrame.return_value = mock_frame

        mock_leader = MagicMock()
        mock_leader.model_id = "GBM_best"

        mock_leaderboard = MagicMock()
        mock_leaderboard_df = pd.DataFrame(
            {"model_id": ["GBM_best", "DRF_1"], "auc": [0.88, 0.82]}
        )
        mock_leaderboard.as_data_frame.return_value = mock_leaderboard_df

        mock_aml = MagicMock()
        mock_aml.leader = mock_leader
        mock_aml.leaderboard = mock_leaderboard
        h2o_automl_mod.H2OAutoML.return_value = mock_aml

        X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        X_val = pd.DataFrame({"f1": [7, 8], "f2": [9, 10]})
        y_val = pd.Series([1, 0])

        result = trainer.train(X_train, y_train, X_val, y_val)

        assert isinstance(result, AutoMLResult)
        assert result.best_model_id == "GBM_best"
        assert result.training_time_secs > 0
        assert len(result.all_model_ids) == 2

    def test_predict_returns_dataframe(self, mock_h2o):
        """predict() should return a DataFrame."""
        trainer, h2o_module, _ = mock_h2o

        mock_frame = MagicMock()
        h2o_module.H2OFrame.return_value = mock_frame

        mock_model = MagicMock()
        mock_preds = MagicMock()
        mock_preds.as_data_frame.return_value = pd.DataFrame(
            {"predict": [0, 1], "p0": [0.8, 0.3], "p1": [0.2, 0.7]}
        )
        mock_model.predict.return_value = mock_preds

        X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        result = trainer.predict(mock_model, X)

        assert isinstance(result, pd.DataFrame)
        assert "predict" in result.columns
        assert len(result) == 2

    def test_save_model_binary(self, mock_h2o, tmp_path):
        """save_model() should call h2o.save_model for binary format."""
        trainer, h2o_module, _ = mock_h2o
        h2o_module.save_model.return_value = str(tmp_path / "model")

        mock_model = MagicMock()
        result = trainer.save_model(mock_model, str(tmp_path / "models"), fmt="binary")

        h2o_module.save_model.assert_called_once()
        assert isinstance(result, str)

    def test_save_model_mojo(self, mock_h2o, tmp_path):
        """save_model() should call download_mojo for mojo format."""
        trainer, _, _ = mock_h2o

        mock_model = MagicMock()
        mock_model.download_mojo.return_value = str(tmp_path / "model.zip")

        result = trainer.save_model(mock_model, str(tmp_path / "models"), fmt="mojo")

        mock_model.download_mojo.assert_called_once()
        assert isinstance(result, str)

    def test_automl_result_leaderboard(self, sample_automl_result):
        """AutoMLResult leaderboard should have expected columns."""
        lb = sample_automl_result.leaderboard
        assert "model_id" in lb.columns
        assert "auc" in lb.columns
        assert len(lb) == 3

    def test_shutdown(self, mock_h2o):
        """shutdown() should call cluster shutdown."""
        trainer, h2o_module, _ = mock_h2o
        trainer.shutdown()
        h2o_module.cluster().shutdown.assert_called_once()

    def test_automl_result_fields(self, sample_automl_result):
        """AutoMLResult should have all expected fields."""
        assert sample_automl_result.best_model_id == "GBM_1"
        assert sample_automl_result.training_time_secs == 60.5
        assert len(sample_automl_result.all_model_ids) == 3
        assert sample_automl_result.best_model is not None
