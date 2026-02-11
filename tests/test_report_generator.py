"""Tests for report generation module."""

import numpy as np
import pytest

from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.business.report_generator import ReportGenerator


@pytest.fixture()
def report_setup(tmp_path):
    """Set up evaluator and registry with sample data for report tests."""
    evaluator = ModelEvaluator(output_dir=str(tmp_path / "reports"))
    registry = ModelRegistry(models_dir=str(tmp_path / "models"))

    rng = np.random.RandomState(42)
    y_true = np.array([0] * 70 + [1] * 30)

    y_prob_a = np.clip(y_true + rng.randn(100) * 0.2, 0.01, 0.99)
    y_pred_a = (y_prob_a >= 0.5).astype(int)
    evaluator.evaluate("ModelA", y_true, y_pred_a, y_prob_a)

    y_prob_b = rng.uniform(0.2, 0.8, 100)
    y_pred_b = (y_prob_b >= 0.5).astype(int)
    evaluator.evaluate("ModelB", y_true, y_pred_b, y_prob_b)

    return evaluator, registry, tmp_path


class TestReportGenerator:
    """Tests for the ReportGenerator class."""

    def test_generate_markdown_report_creates_file(self, report_setup):
        """generate_markdown_report() should create a markdown file."""
        evaluator, registry, tmp_path = report_setup
        gen = ReportGenerator(evaluator, registry, output_dir=str(tmp_path / "reports"))
        report = gen.generate_markdown_report()

        assert (tmp_path / "reports" / "model_comparison_report.md").exists()
        assert len(report) > 0

    def test_report_contains_expected_sections(self, report_setup):
        """Report should contain all required sections."""
        evaluator, registry, tmp_path = report_setup
        gen = ReportGenerator(evaluator, registry, output_dir=str(tmp_path / "reports"))
        report = gen.generate_markdown_report()

        assert "# Churn Prediction Model Comparison Report" in report
        assert "## Summary" in report
        assert "## Model Comparison" in report
        assert "## Visualizations" in report
        assert "## Statistical Significance" in report

    def test_report_contains_model_names(self, report_setup):
        """Report should reference evaluated model names."""
        evaluator, registry, tmp_path = report_setup
        gen = ReportGenerator(evaluator, registry, output_dir=str(tmp_path / "reports"))
        report = gen.generate_markdown_report()

        assert "ModelA" in report
        assert "ModelB" in report

    def test_report_contains_mcnemar_results(self, report_setup):
        """Report should include McNemar test results for model pairs."""
        evaluator, registry, tmp_path = report_setup
        gen = ReportGenerator(evaluator, registry, output_dir=str(tmp_path / "reports"))
        report = gen.generate_markdown_report()

        assert "ModelA vs ModelB" in report
        assert "p-value=" in report

    def test_report_with_best_model_set(self, report_setup):
        """Report should show best model when set in registry."""
        evaluator, registry, tmp_path = report_setup
        registry.registry["best_model_id"] = "ModelA"

        gen = ReportGenerator(evaluator, registry, output_dir=str(tmp_path / "reports"))
        report = gen.generate_markdown_report()

        assert "**Best Model:** ModelA" in report

    def test_report_chart_references(self, report_setup):
        """Report should reference chart files."""
        evaluator, registry, tmp_path = report_setup
        gen = ReportGenerator(evaluator, registry, output_dir=str(tmp_path / "reports"))
        report = gen.generate_markdown_report()

        assert "roc_curves.html" in report
        assert "pr_curves.html" in report
        assert "confusion_matrices.png" in report
        assert "calibration_curves.html" in report
