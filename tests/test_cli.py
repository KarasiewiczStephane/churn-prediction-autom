"""Tests for Click CLI commands."""

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture()
def runner():
    """Create a Click test runner."""
    return CliRunner()


class TestCLI:
    """Tests for the CLI command group."""

    def test_cli_help(self, runner):
        """CLI should display help text."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Churn Prediction CLI" in result.output

    def test_cli_loads_config(self, runner):
        """CLI should load config from default path."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_train_help(self, runner):
        """train command should display help text."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "time-budget" in result.output
        assert "feature-method" in result.output

    def test_predict_help(self, runner):
        """predict command should display help text."""
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "input" in result.output.lower()
        assert "output" in result.output.lower()

    def test_evaluate_help(self, runner):
        """evaluate command should display help text."""
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Regenerate" in result.output

    def test_impact_help(self, runner):
        """impact command should display help text."""
        result = runner.invoke(cli, ["impact", "--help"])
        assert result.exit_code == 0
        assert "revenue-col" in result.output

    def test_compare_help(self, runner):
        """compare command should display help text."""
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "comparison" in result.output.lower()

    def test_invalid_config_path(self, runner):
        """CLI should error with invalid config path."""
        result = runner.invoke(
            cli, ["-c", "/nonexistent/config.yaml", "train", "--help"]
        )
        assert result.exit_code != 0

    def test_verbose_flag(self, runner):
        """CLI should accept verbose flag."""
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0

    def test_quiet_flag(self, runner):
        """CLI should accept quiet flag."""
        result = runner.invoke(cli, ["-q", "--help"])
        assert result.exit_code == 0

    def test_train_feature_method_choices(self, runner):
        """train should accept valid feature method choices."""
        result = runner.invoke(
            cli, ["train", "--feature-method", "correlation", "--help"]
        )
        assert result.exit_code == 0

    def test_predict_requires_input(self, runner):
        """predict command should require --input option."""
        result = runner.invoke(cli, ["predict"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()
