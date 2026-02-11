"""H2O AutoML training pipeline with configurable time budget and leaderboard."""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AutoMLResult:
    """Container for H2O AutoML training results.

    Attributes:
        leaderboard: DataFrame with model performance rankings.
        best_model: The top-performing H2O model estimator.
        best_model_id: Identifier of the best model.
        training_time_secs: Total training duration in seconds.
        all_model_ids: List of all model identifiers from the leaderboard.
    """

    leaderboard: pd.DataFrame
    best_model: object
    best_model_id: str
    training_time_secs: float
    all_model_ids: list[str]


class H2OAutoMLTrainer:
    """Manages H2O AutoML training, prediction, and model export.

    Args:
        max_runtime_secs: Maximum time budget for AutoML in seconds.
        seed: Random seed for reproducibility.
        nfolds: Number of cross-validation folds.
    """

    def __init__(
        self,
        max_runtime_secs: int = 300,
        seed: int = 42,
        nfolds: int = 5,
    ) -> None:
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.nfolds = nfolds
        self._h2o = None
        self._h2o_automl = None

    def _init_h2o(self) -> None:
        """Initialize H2O cluster."""
        import h2o

        self._h2o = h2o
        h2o.init(nthreads=-1, max_mem_size="4G")
        logger.info("H2O cluster initialized")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> AutoMLResult:
        """Run H2O AutoML training.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Validation feature matrix.
            y_val: Validation target vector.

        Returns:
            AutoMLResult with leaderboard and best model.
        """
        if self._h2o is None:
            self._init_h2o()

        h2o = self._h2o
        from h2o.automl import H2OAutoML

        train_df = X_train.copy()
        train_df["target"] = y_train.values
        val_df = X_val.copy()
        val_df["target"] = y_val.values

        train_h2o = h2o.H2OFrame(train_df)
        val_h2o = h2o.H2OFrame(val_df)

        train_h2o["target"] = train_h2o["target"].asfactor()
        val_h2o["target"] = val_h2o["target"].asfactor()

        predictors = [c for c in train_h2o.columns if c != "target"]
        response = "target"

        aml = H2OAutoML(
            max_runtime_secs=self.max_runtime_secs,
            seed=self.seed,
            nfolds=self.nfolds,
            balance_classes=True,
            sort_metric="AUC",
            include_algos=[
                "GLM",
                "DRF",
                "GBM",
                "XGBoost",
                "DeepLearning",
                "StackedEnsemble",
            ],
            verbosity="warn",
        )

        start_time = time.time()
        aml.train(
            x=predictors,
            y=response,
            training_frame=train_h2o,
            validation_frame=val_h2o,
        )
        training_time = time.time() - start_time

        leaderboard = aml.leaderboard.as_data_frame()
        model_ids = leaderboard["model_id"].tolist()

        logger.info("AutoML completed in %.2fs", training_time)
        logger.info("Best model: %s", aml.leader.model_id)

        return AutoMLResult(
            leaderboard=leaderboard,
            best_model=aml.leader,
            best_model_id=aml.leader.model_id,
            training_time_secs=training_time,
            all_model_ids=model_ids,
        )

    def predict(self, model: object, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with probabilities.

        Args:
            model: Trained H2O model estimator.
            X: Feature DataFrame for prediction.

        Returns:
            DataFrame with predictions and probability columns.
        """
        if self._h2o is None:
            self._init_h2o()

        h2o = self._h2o
        h2o_frame = h2o.H2OFrame(X)
        predictions = model.predict(h2o_frame).as_data_frame()
        return predictions

    def save_model(self, model: object, path: str, fmt: str = "binary") -> str:
        """Save model in binary or MOJO format.

        Args:
            model: Trained H2O model to save.
            path: Output directory path.
            fmt: Format to save ('binary' or 'mojo').

        Returns:
            Path where the model was saved.
        """
        if self._h2o is None:
            self._init_h2o()

        h2o = self._h2o
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        if fmt == "mojo":
            saved_path = model.download_mojo(
                path=str(output_path), get_genmodel_jar=True
            )
            logger.info("Saved MOJO model to %s", output_path)
        else:
            saved_path = h2o.save_model(model, path=str(output_path), force=True)
            logger.info("Saved binary model to %s", output_path)

        return str(saved_path)

    def shutdown(self) -> None:
        """Shutdown H2O cluster."""
        if self._h2o is not None:
            self._h2o.cluster().shutdown()
            logger.info("H2O cluster shut down")
