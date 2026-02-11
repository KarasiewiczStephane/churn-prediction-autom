"""Tests for model evaluation suite."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluator import EvaluationMetrics, ModelEvaluator


@pytest.fixture()
def evaluator(tmp_path):
    """Create a ModelEvaluator with temp output directory."""
    return ModelEvaluator(output_dir=str(tmp_path / "reports"))


@pytest.fixture()
def evaluation_data():
    """Create synthetic prediction data for testing."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = np.array([0] * 70 + [1] * 30)
    y_prob_good = np.clip(y_true + rng.randn(n) * 0.2, 0.01, 0.99)
    y_pred_good = (y_prob_good >= 0.5).astype(int)
    y_prob_bad = rng.uniform(0.2, 0.8, n)
    y_pred_bad = (y_prob_bad >= 0.5).astype(int)
    return {
        "y_true": y_true,
        "y_prob_good": y_prob_good,
        "y_pred_good": y_pred_good,
        "y_prob_bad": y_prob_bad,
        "y_pred_bad": y_pred_bad,
    }


class TestModelEvaluator:
    """Tests for the ModelEvaluator class."""

    def test_evaluate_returns_metrics(self, evaluator, evaluation_data):
        """evaluate() should return EvaluationMetrics with all fields."""
        metrics = evaluator.evaluate(
            "good_model",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.model_name == "good_model"
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.auc_roc <= 1
        assert metrics.log_loss_value > 0

    def test_evaluate_good_model_better_than_bad(self, evaluator, evaluation_data):
        """Good model should have higher AUC than random model."""
        evaluator.evaluate(
            "good_model",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )
        evaluator.evaluate(
            "bad_model",
            evaluation_data["y_true"],
            evaluation_data["y_pred_bad"],
            evaluation_data["y_prob_bad"],
        )

        assert (
            evaluator.results["good_model"].auc_roc
            > evaluator.results["bad_model"].auc_roc
        )

    def test_plot_roc_curves(self, evaluator, evaluation_data):
        """plot_roc_curves() should create HTML file."""
        evaluator.evaluate(
            "model_a",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )

        fig = evaluator.plot_roc_curves()
        assert fig is not None
        assert (evaluator.output_dir / "roc_curves.html").exists()

    def test_plot_precision_recall_curves(self, evaluator, evaluation_data):
        """plot_precision_recall_curves() should create HTML file."""
        evaluator.evaluate(
            "model_a",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )

        fig = evaluator.plot_precision_recall_curves()
        assert fig is not None
        assert (evaluator.output_dir / "pr_curves.html").exists()

    def test_plot_confusion_matrices(self, evaluator, evaluation_data):
        """plot_confusion_matrices() should create PNG file."""
        evaluator.evaluate(
            "model_a",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )

        evaluator.plot_confusion_matrices()
        assert (evaluator.output_dir / "confusion_matrices.png").exists()

    def test_plot_calibration_curves(self, evaluator, evaluation_data):
        """plot_calibration_curves() should create HTML file."""
        evaluator.evaluate(
            "model_a",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )

        fig = evaluator.plot_calibration_curves()
        assert fig is not None
        assert (evaluator.output_dir / "calibration_curves.html").exists()

    def test_mcnemar_test(self, evaluator, evaluation_data):
        """mcnemar_test() should return valid statistics."""
        evaluator.evaluate(
            "good_model",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )
        evaluator.evaluate(
            "bad_model",
            evaluation_data["y_true"],
            evaluation_data["y_pred_bad"],
            evaluation_data["y_prob_bad"],
        )

        result = evaluator.mcnemar_test("good_model", "bad_model")

        assert "p_value" in result
        assert "statistic" in result
        assert "significant" in result
        assert isinstance(result["p_value"], float)
        assert isinstance(result["significant"], bool)

    def test_generate_comparison_table(self, evaluator, evaluation_data):
        """generate_comparison_table() should have correct columns and rows."""
        evaluator.evaluate(
            "model_a",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )
        evaluator.evaluate(
            "model_b",
            evaluation_data["y_true"],
            evaluation_data["y_pred_bad"],
            evaluation_data["y_prob_bad"],
        )

        table = evaluator.generate_comparison_table()

        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert "Model" in table.columns
        assert "AUC-ROC" in table.columns
        assert "F1" in table.columns
        assert "Log Loss" in table.columns

    def test_evaluate_perfect_predictions(self, evaluator):
        """evaluate() should handle perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = evaluator.evaluate("perfect", y_true, y_pred, y_prob)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_multiple_confusion_matrices(self, evaluator, evaluation_data):
        """plot_confusion_matrices() should handle multiple models."""
        evaluator.evaluate(
            "model_a",
            evaluation_data["y_true"],
            evaluation_data["y_pred_good"],
            evaluation_data["y_prob_good"],
        )
        evaluator.evaluate(
            "model_b",
            evaluation_data["y_true"],
            evaluation_data["y_pred_bad"],
            evaluation_data["y_prob_bad"],
        )

        evaluator.plot_confusion_matrices()
        assert (evaluator.output_dir / "confusion_matrices.png").exists()
