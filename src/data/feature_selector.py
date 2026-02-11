"""Feature selection using correlation, mutual information, and Boruta methods."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result from a single feature selection method.

    Attributes:
        method: Name of the feature selection method.
        selected_features: List of feature names that were kept.
        dropped_features: List of feature names that were removed.
        scores: Optional dictionary mapping feature names to their scores.
    """

    method: str
    selected_features: list[str]
    dropped_features: list[str]
    scores: dict[str, float] | None = None


class FeatureSelector:
    """Runs multiple feature selection methods and compares results.

    Args:
        db: Optional ResultsDB instance for persisting selection results.
    """

    def __init__(self, db: object | None = None) -> None:
        self.db = db
        self.results: dict[str, FeatureSelectionResult] = {}

    def correlation_filter(
        self, X: pd.DataFrame, threshold: float = 0.95
    ) -> FeatureSelectionResult:
        """Drop features with pairwise correlation above threshold.

        Args:
            X: Feature DataFrame.
            threshold: Correlation threshold above which to drop features.

        Returns:
            FeatureSelectionResult with selected and dropped features.
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        selected = [col for col in X.columns if col not in to_drop]

        result = FeatureSelectionResult(
            method="correlation",
            selected_features=selected,
            dropped_features=to_drop,
        )
        self.results["correlation"] = result
        logger.info(
            "Correlation filter: kept %d, dropped %d", len(selected), len(to_drop)
        )
        return result

    def mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 15,
        random_state: int = 42,
    ) -> FeatureSelectionResult:
        """Select top-k features by mutual information score.

        Args:
            X: Feature DataFrame.
            y: Target series.
            top_k: Number of top features to select.
            random_state: Random seed for reproducibility.

        Returns:
            FeatureSelectionResult with selected and dropped features.
        """
        top_k = min(top_k, len(X.columns))
        mi_scores = mutual_info_classif(X, y, random_state=random_state)
        mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
        mi_df = mi_df.sort_values("mi_score", ascending=False)

        selected = mi_df.head(top_k)["feature"].tolist()
        dropped = mi_df.tail(len(mi_df) - top_k)["feature"].tolist()
        scores = dict(zip(mi_df["feature"], mi_df["mi_score"]))

        result = FeatureSelectionResult(
            method="mutual_information",
            selected_features=selected,
            dropped_features=dropped,
            scores=scores,
        )
        self.results["mutual_information"] = result
        logger.info("MI selection: kept %d, dropped %d", len(selected), len(dropped))
        return result

    def boruta_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_iter: int = 100,
        random_state: int = 42,
    ) -> FeatureSelectionResult:
        """Use Boruta algorithm for all-relevant feature selection.

        Args:
            X: Feature DataFrame.
            y: Target series.
            max_iter: Maximum number of Boruta iterations.
            random_state: Random seed for reproducibility.

        Returns:
            FeatureSelectionResult with selected and dropped features.
        """
        rf = RandomForestClassifier(n_jobs=-1, random_state=random_state, max_depth=5)
        boruta = BorutaPy(
            rf,
            n_estimators="auto",
            verbose=0,
            random_state=random_state,
            max_iter=max_iter,
        )

        boruta.fit(X.values, y.values)

        selected = X.columns[boruta.support_].tolist()
        dropped = X.columns[~boruta.support_].tolist()
        scores = dict(zip(X.columns, [float(r) for r in boruta.ranking_]))

        result = FeatureSelectionResult(
            method="boruta",
            selected_features=selected,
            dropped_features=dropped,
            scores=scores,
        )
        self.results["boruta"] = result
        logger.info(
            "Boruta selection: kept %d, dropped %d", len(selected), len(dropped)
        )
        return result

    def run_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        correlation_threshold: float = 0.95,
        mi_top_k: int = 15,
        boruta_max_iter: int = 100,
    ) -> dict[str, FeatureSelectionResult]:
        """Run all three feature selection methods.

        Args:
            X: Feature DataFrame.
            y: Target series.
            correlation_threshold: Threshold for correlation filter.
            mi_top_k: Number of top features for mutual information.
            boruta_max_iter: Maximum iterations for Boruta.

        Returns:
            Dictionary mapping method names to their results.
        """
        self.correlation_filter(X, correlation_threshold)
        self.mutual_information(X, y, mi_top_k)
        self.boruta_selection(X, y, boruta_max_iter)
        return self.results

    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate a comparison report showing feature selection across methods.

        Returns:
            DataFrame with features as rows and selection status per method.
        """
        all_features: set[str] = set()
        for result in self.results.values():
            all_features.update(result.selected_features)
            all_features.update(result.dropped_features)

        report_data = []
        for feature in sorted(all_features):
            row: dict = {"feature": feature}
            for method, result in self.results.items():
                row[f"{method}_selected"] = feature in result.selected_features
                if result.scores:
                    row[f"{method}_score"] = result.scores.get(feature)
            report_data.append(row)

        return pd.DataFrame(report_data)

    def get_selected_features(self, method: str) -> list[str]:
        """Get the list of selected features for a specific method.

        Args:
            method: Name of the feature selection method.

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If the specified method has not been run.
        """
        if method not in self.results:
            raise ValueError(f"Method '{method}' not run yet")
        return self.results[method].selected_features
