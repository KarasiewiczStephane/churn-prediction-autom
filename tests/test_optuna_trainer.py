"""Tests for Optuna hyperparameter optimization pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.models.optuna_trainer import OptunaResult, OptunaTrainer


@pytest.fixture()
def classification_data():
    """Create synthetic binary classification data for testing."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
            "f4": rng.randn(n),
        }
    )
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    return X, y


class TestOptunaTrainer:
    """Tests for the OptunaTrainer class."""

    def test_optimize_lightgbm(self, classification_data):
        """optimize_lightgbm() should return valid OptunaResult."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=3, cv_folds=3)
        result = trainer.optimize_lightgbm(X, y)

        assert isinstance(result, OptunaResult)
        assert result.model_name == "lightgbm"
        assert result.best_score > 0.5
        assert result.training_time_secs > 0
        assert "n_estimators" in result.best_params
        assert "learning_rate" in result.best_params

    def test_optimize_logistic_regression(self, classification_data):
        """optimize_logistic_regression() should return valid OptunaResult."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=3, cv_folds=3)
        result = trainer.optimize_logistic_regression(X, y)

        assert isinstance(result, OptunaResult)
        assert result.model_name == "logistic_regression"
        assert result.best_score > 0.5
        assert "C" in result.best_params
        assert "penalty" in result.best_params

    def test_lightgbm_model_can_predict(self, classification_data):
        """Best LightGBM model should generate predictions."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=3, cv_folds=3)
        result = trainer.optimize_lightgbm(X, y)

        predictions = result.best_model.predict(X)
        probabilities = result.best_model.predict_proba(X)

        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)

    def test_logistic_regression_model_can_predict(self, classification_data):
        """Best Logistic Regression model should generate predictions."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=3, cv_folds=3)
        result = trainer.optimize_logistic_regression(X, y)

        predictions = result.best_model.predict(X)
        probabilities = result.best_model.predict_proba(X)

        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)

    def test_cv_scores_length(self, classification_data):
        """cv_scores should have length equal to n_folds."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=3, cv_folds=4)
        result = trainer.optimize_lightgbm(X, y)

        assert len(result.cv_scores) == 4

    def test_save_model_creates_files(self, classification_data, tmp_path):
        """save_model() should create joblib files."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=3, cv_folds=3)
        result = trainer.optimize_lightgbm(X, y)

        trainer.save_model(result, str(tmp_path / "models"))

        assert (tmp_path / "models" / "lightgbm_model.joblib").exists()
        assert (tmp_path / "models" / "lightgbm_params.joblib").exists()

    def test_study_has_completed_trials(self, classification_data):
        """Optuna study should have the expected number of completed trials."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=5, cv_folds=3)
        result = trainer.optimize_lightgbm(X, y)

        assert len(result.study.trials) == 5

    def test_training_time_tracked(self, classification_data):
        """Training time should be positive and reasonable."""
        X, y = classification_data
        trainer = OptunaTrainer(n_trials=3, cv_folds=3)
        result = trainer.optimize_lightgbm(X, y)

        assert result.training_time_secs > 0
        assert result.training_time_secs < 300
