"""Configuration management with YAML loading and dataclass validation."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Configuration for data paths and split ratios."""

    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    sample_path: str = "data/sample"
    test_size: float = 0.2
    val_size: float = 0.2


@dataclass
class ModelConfig:
    """Configuration for model training parameters."""

    automl_max_runtime_secs: int = 300
    optuna_trials: int = 50
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class FeatureConfig:
    """Configuration for feature selection methods."""

    correlation_threshold: float = 0.95
    mi_top_k: int = 15
    selection_method: str = "boruta"


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    log_dir: str = "logs"


@dataclass
class DatabaseConfig:
    """Configuration for DuckDB database."""

    path: str = "results/results.duckdb"


@dataclass
class BusinessConfig:
    """Configuration for business impact analysis."""

    high_value_threshold: float = 100.0
    medium_value_threshold: float = 50.0
    intervention_cost: float = 20.0
    intervention_success_rate: float = 0.3


@dataclass
class Config:
    """Main configuration container that loads from YAML.

    Args:
        config_path: Path to the YAML configuration file.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = Path(config_path)
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            cfg = yaml.safe_load(f) or {}

        self.data = DataConfig(**cfg.get("data", {}))
        self.model = ModelConfig(**cfg.get("model", {}))
        self.feature = FeatureConfig(**cfg.get("feature", {}))
        self.logging = LoggingConfig(**cfg.get("logging", {}))
        self.database = DatabaseConfig(**cfg.get("database", {}))
        self.business = BusinessConfig(**cfg.get("business", {}))

    def _ensure_directories(self) -> None:
        """Create necessary directories from config paths."""
        for path_str in [
            self.data.raw_path,
            self.data.processed_path,
            self.data.sample_path,
            self.logging.log_dir,
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)

        Path(self.database.path).parent.mkdir(parents=True, exist_ok=True)
