"""Optuna hyperparameter optimization for LightGBM and Logistic Regression."""

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptunaResult:
    """Container for Optuna optimization results.

    Attributes:
        model_name: Name of the optimized model.
        best_model: The trained model with best hyperparameters.
        best_params: Dictionary of best hyperparameters.
        best_score: Best cross-validation score achieved.
        training_time_secs: Total optimization duration in seconds.
        study: The Optuna Study object with trial history.
        cv_scores: Cross-validation scores for the best model.
    """

    model_name: str
    best_model: object
    best_params: dict
    best_score: float
    training_time_secs: float
    study: optuna.Study
    cv_scores: np.ndarray


class OptunaTrainer:
    """Manages Optuna-based hyperparameter optimization for multiple models.

    Args:
        n_trials: Number of Optuna trials to run per model.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )

    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> OptunaResult:
        """Optimize LightGBM hyperparameters with Optuna.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Returns:
            OptunaResult with the best LightGBM model and parameters.
        """

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "random_state": self.random_state,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            model = lgb.LGBMClassifier(**params)
            scores = cross_val_score(
                model, X, y, cv=self.cv, scoring="roc_auc", n_jobs=-1
            )
            return scores.mean()

        start_time = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        training_time = time.time() - start_time

        best_params = study.best_params.copy()
        best_params.update(
            {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "random_state": self.random_state,
            }
        )
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X, y)

        cv_scores = cross_val_score(best_model, X, y, cv=self.cv, scoring="roc_auc")

        logger.info("LightGBM optimization completed in %.2fs", training_time)
        logger.info("Best AUC: %.4f", study.best_value)

        return OptunaResult(
            model_name="lightgbm",
            best_model=best_model,
            best_params=best_params,
            best_score=study.best_value,
            training_time_secs=training_time,
            study=study,
            cv_scores=cv_scores,
        )

    def optimize_logistic_regression(
        self, X: pd.DataFrame, y: pd.Series
    ) -> OptunaResult:
        """Optimize Logistic Regression hyperparameters with Optuna.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Returns:
            OptunaResult with the best Logistic Regression model and parameters.
        """

        def objective(trial: optuna.Trial) -> float:
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            params: dict = {
                "penalty": penalty,
                "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "solver": "saga",
                "max_iter": 1000,
                "random_state": self.random_state,
            }

            if penalty == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

            model = LogisticRegression(**params)
            scores = cross_val_score(
                model, X, y, cv=self.cv, scoring="roc_auc", n_jobs=-1
            )
            return scores.mean()

        start_time = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        training_time = time.time() - start_time

        best_params = study.best_params.copy()
        best_params.update(
            {
                "solver": "saga",
                "max_iter": 1000,
                "random_state": self.random_state,
            }
        )
        best_model = LogisticRegression(**best_params)
        best_model.fit(X, y)

        cv_scores = cross_val_score(best_model, X, y, cv=self.cv, scoring="roc_auc")

        logger.info("LogReg optimization completed in %.2fs", training_time)
        logger.info("Best AUC: %.4f", study.best_value)

        return OptunaResult(
            model_name="logistic_regression",
            best_model=best_model,
            best_params=best_params,
            best_score=study.best_value,
            training_time_secs=training_time,
            study=study,
            cv_scores=cv_scores,
        )

    def save_model(self, result: OptunaResult, path: str) -> None:
        """Save model and parameters with joblib.

        Args:
            result: OptunaResult containing the model to save.
            path: Output directory path.
        """
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            result.best_model, output_path / f"{result.model_name}_model.joblib"
        )
        joblib.dump(
            result.best_params, output_path / f"{result.model_name}_params.joblib"
        )
        logger.info("Saved %s model to %s", result.model_name, output_path)
