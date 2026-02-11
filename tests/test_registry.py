"""Tests for model registry module."""

from unittest.mock import MagicMock

import pytest
from sklearn.linear_model import LogisticRegression

from src.models.registry import ModelMetadata, ModelRegistry


@pytest.fixture()
def registry(tmp_path):
    """Create a ModelRegistry with temp directory."""
    return ModelRegistry(models_dir=str(tmp_path / "models"))


@pytest.fixture()
def sample_model():
    """Create a simple trained model for testing."""
    import numpy as np

    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    return model


@pytest.fixture()
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        model_id="test_model_v1",
        model_type="sklearn",
        created_at="2024-01-01T00:00:00",
        metrics={"accuracy": 0.85, "f1": 0.80, "auc_roc": 0.88},
        hyperparameters={"C": 1.0, "penalty": "l2"},
        feature_selection_method="boruta",
        training_time_secs=45.0,
    )


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_register_model_saves_joblib(self, registry, sample_model, sample_metadata):
        """register_model() should save model as joblib file."""
        registry.register_model(sample_model, sample_metadata)

        model_dir = registry.models_dir / "test_model_v1"
        assert (model_dir / "test_model_v1.joblib").exists()
        assert (model_dir / "metadata.json").exists()

    def test_register_model_updates_registry(
        self, registry, sample_model, sample_metadata
    ):
        """register_model() should update the registry file."""
        registry.register_model(sample_model, sample_metadata)

        assert len(registry.registry["models"]) == 1
        assert registry.registry["models"][0]["model_id"] == "test_model_v1"

    def test_set_best_model(self, registry, sample_model, sample_metadata):
        """set_best_model() should update the registry."""
        registry.register_model(sample_model, sample_metadata)
        registry.set_best_model("test_model_v1")

        assert registry.registry["best_model_id"] == "test_model_v1"

    def test_load_model(self, registry, sample_model, sample_metadata):
        """load_model() should retrieve the correct model."""
        registry.register_model(sample_model, sample_metadata)
        loaded = registry.load_model("test_model_v1")

        assert hasattr(loaded, "predict")

    def test_get_best_model(self, registry, sample_model, sample_metadata):
        """get_best_model() should return model and metadata."""
        registry.register_model(sample_model, sample_metadata)
        registry.set_best_model("test_model_v1")

        model, metadata = registry.get_best_model()
        assert hasattr(model, "predict")
        assert metadata["model_id"] == "test_model_v1"

    def test_get_best_model_raises_when_not_set(self, registry):
        """get_best_model() should raise ValueError if no best model set."""
        with pytest.raises(ValueError, match="No best model set"):
            registry.get_best_model()

    def test_load_model_raises_for_missing(self, registry):
        """load_model() should raise FileNotFoundError for unknown model."""
        with pytest.raises(FileNotFoundError):
            registry.load_model("nonexistent_model")

    def test_registry_persists_across_instances(
        self, tmp_path, sample_model, sample_metadata
    ):
        """Registry should persist data across instances."""
        models_dir = str(tmp_path / "models")
        reg1 = ModelRegistry(models_dir=models_dir)
        reg1.register_model(sample_model, sample_metadata)
        reg1.set_best_model("test_model_v1")

        reg2 = ModelRegistry(models_dir=models_dir)
        assert reg2.registry["best_model_id"] == "test_model_v1"
        assert len(reg2.registry["models"]) == 1

    def test_list_models(self, registry, sample_model, sample_metadata):
        """list_models() should return all registered models."""
        registry.register_model(sample_model, sample_metadata)
        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "test_model_v1"

    def test_register_model_with_db(self, tmp_path, sample_model, sample_metadata):
        """register_model() should call db.insert_model_run when db is provided."""
        mock_db = MagicMock()
        registry = ModelRegistry(models_dir=str(tmp_path / "models"), db=mock_db)
        registry.register_model(sample_model, sample_metadata)

        mock_db.insert_model_run.assert_called_once()
