"""Tests for configuration management module."""

import pytest
import yaml

from src.utils.config import (
    BusinessConfig,
    Config,
    DataConfig,
    DatabaseConfig,
    FeatureConfig,
    LoggingConfig,
    ModelConfig,
)


@pytest.fixture()
def config_file(tmp_path):
    """Create a temporary config YAML file."""
    config_data = {
        "data": {
            "raw_path": "data/raw",
            "processed_path": "data/processed",
            "sample_path": "data/sample",
            "test_size": 0.2,
            "val_size": 0.2,
        },
        "model": {
            "automl_max_runtime_secs": 300,
            "optuna_trials": 50,
            "cv_folds": 5,
            "random_state": 42,
        },
        "feature": {
            "correlation_threshold": 0.95,
            "mi_top_k": 15,
            "selection_method": "boruta",
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs",
        },
        "database": {
            "path": "results/results.duckdb",
        },
        "business": {
            "high_value_threshold": 100.0,
            "medium_value_threshold": 50.0,
            "intervention_cost": 20.0,
            "intervention_success_rate": 0.3,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture()
def minimal_config_file(tmp_path):
    """Create a minimal config YAML with only partial sections."""
    config_data = {
        "data": {"raw_path": "custom/raw"},
    }
    config_path = tmp_path / "minimal_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestConfig:
    """Tests for the Config class."""

    def test_loads_yaml_correctly(self, config_file):
        """Config should load all sections from YAML file."""
        config = Config(str(config_file))

        assert config.data.raw_path == "data/raw"
        assert config.data.test_size == 0.2
        assert config.model.optuna_trials == 50
        assert config.model.random_state == 42
        assert config.feature.correlation_threshold == 0.95
        assert config.feature.selection_method == "boruta"
        assert config.logging.level == "INFO"
        assert config.database.path == "results/results.duckdb"
        assert config.business.high_value_threshold == 100.0

    def test_defaults_applied_for_missing_sections(self, minimal_config_file):
        """Config should apply defaults when sections are missing."""
        config = Config(str(minimal_config_file))

        assert config.data.raw_path == "custom/raw"
        assert config.data.test_size == 0.2
        assert config.model.optuna_trials == 50
        assert config.feature.mi_top_k == 15

    def test_invalid_config_path_raises_error(self):
        """Config should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config("/nonexistent/config.yaml")

    def test_ensure_directories(self, config_file, tmp_path):
        """Config should create necessary directories."""
        config = Config(str(config_file))
        config.data.raw_path = str(tmp_path / "test_data/raw")
        config.data.processed_path = str(tmp_path / "test_data/processed")
        config.data.sample_path = str(tmp_path / "test_data/sample")
        config.logging.log_dir = str(tmp_path / "test_logs")
        config.database.path = str(tmp_path / "test_results/results.duckdb")

        config._ensure_directories()

        from pathlib import Path

        assert Path(config.data.raw_path).exists()
        assert Path(config.data.processed_path).exists()
        assert Path(config.data.sample_path).exists()
        assert Path(config.logging.log_dir).exists()
        assert Path(config.database.path).parent.exists()


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_defaults(self):
        """DataConfig should have sensible defaults."""
        dc = DataConfig()
        assert dc.raw_path == "data/raw"
        assert dc.test_size == 0.2
        assert dc.val_size == 0.2


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self):
        """ModelConfig should have sensible defaults."""
        mc = ModelConfig()
        assert mc.automl_max_runtime_secs == 300
        assert mc.cv_folds == 5


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_defaults(self):
        """FeatureConfig should have sensible defaults."""
        fc = FeatureConfig()
        assert fc.correlation_threshold == 0.95
        assert fc.mi_top_k == 15


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_defaults(self):
        """LoggingConfig should have sensible defaults."""
        lc = LoggingConfig()
        assert lc.level == "INFO"


class TestDatabaseConfig:
    """Tests for DatabaseConfig dataclass."""

    def test_defaults(self):
        """DatabaseConfig should have sensible defaults."""
        dc = DatabaseConfig()
        assert dc.path == "results/results.duckdb"


class TestBusinessConfig:
    """Tests for BusinessConfig dataclass."""

    def test_defaults(self):
        """BusinessConfig should have sensible defaults."""
        bc = BusinessConfig()
        assert bc.high_value_threshold == 100.0
        assert bc.intervention_success_rate == 0.3
