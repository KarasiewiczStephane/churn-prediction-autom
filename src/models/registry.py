"""Model registry for tracking, saving, and loading trained models."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model.

    Attributes:
        model_id: Unique identifier for the model.
        model_type: Type of model ('sklearn', 'lightgbm', 'h2o').
        created_at: ISO timestamp of creation.
        metrics: Dictionary of evaluation metrics.
        hyperparameters: Dictionary of model hyperparameters.
        feature_selection_method: Method used for feature selection.
        training_time_secs: Training duration in seconds.
        file_path: Path where the model file is stored.
    """

    model_id: str
    model_type: str
    created_at: str
    metrics: dict
    hyperparameters: dict
    feature_selection_method: str
    training_time_secs: float
    file_path: str = ""


class ModelRegistry:
    """Manages model persistence, versioning, and retrieval.

    Args:
        models_dir: Directory for storing model files.
        db: Optional ResultsDB instance for database persistence.
    """

    def __init__(self, models_dir: str = "models", db: object | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self.db = db
        self._load_registry()

    def _load_registry(self) -> None:
        """Load existing registry from disk or create a new one."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": [], "best_model_id": None}

    def _save_registry(self) -> None:
        """Persist registry to disk as JSON."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(self, model: Any, metadata: ModelMetadata) -> None:
        """Register and save a model to the registry.

        Args:
            model: The trained model object to save.
            metadata: ModelMetadata describing the model.
        """
        model_path = self.models_dir / metadata.model_id
        model_path.mkdir(parents=True, exist_ok=True)

        joblib_path = model_path / f"{metadata.model_id}.joblib"
        joblib.dump(model, joblib_path)
        metadata.file_path = str(joblib_path)

        with open(model_path / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

        self.registry["models"].append(asdict(metadata))
        self._save_registry()

        if self.db and hasattr(self.db, "insert_model_run"):
            self.db.insert_model_run(
                model_type=metadata.model_type,
                model_name=metadata.model_id,
                metrics=metadata.metrics,
                training_time_secs=metadata.training_time_secs,
                hyperparameters=json.dumps(metadata.hyperparameters),
                feature_selection_method=metadata.feature_selection_method,
            )

        logger.info("Registered model: %s", metadata.model_id)

    def set_best_model(self, model_id: str) -> None:
        """Set the best model in the registry.

        Args:
            model_id: Identifier of the best model.
        """
        self.registry["best_model_id"] = model_id
        self._save_registry()
        logger.info("Set best model: %s", model_id)

    def load_model(self, model_id: str) -> Any:
        """Load a model from the registry by its identifier.

        Args:
            model_id: Identifier of the model to load.

        Returns:
            The deserialized model object.

        Raises:
            FileNotFoundError: If the model metadata file is not found.
        """
        metadata_path = self.models_dir / model_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        return joblib.load(metadata["file_path"])

    def get_best_model(self) -> tuple[Any, dict]:
        """Load the best model and its metadata.

        Returns:
            Tuple of (model object, metadata dictionary).

        Raises:
            ValueError: If no best model has been set.
        """
        best_id = self.registry.get("best_model_id")
        if not best_id:
            raise ValueError("No best model set in registry")

        model = self.load_model(best_id)
        metadata = next(
            (m for m in self.registry["models"] if m["model_id"] == best_id),
            {},
        )
        return model, metadata

    def list_models(self) -> list[dict]:
        """List all registered models.

        Returns:
            List of model metadata dictionaries.
        """
        return self.registry["models"]
