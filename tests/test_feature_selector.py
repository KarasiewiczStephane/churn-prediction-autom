"""Tests for feature selection pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_selector import FeatureSelector, FeatureSelectionResult


@pytest.fixture()
def synthetic_features():
    """Create synthetic feature data with known properties.

    Returns:
        Tuple of (X, y) where X has correlated and informative features.
    """
    rng = np.random.RandomState(42)
    n = 200

    f1 = rng.randn(n)
    f2 = f1 + rng.randn(n) * 0.01  # highly correlated with f1
    f3 = rng.randn(n)  # informative
    f4 = rng.randn(n)  # noise
    f5 = rng.randn(n) * 0.5 + 1  # somewhat informative

    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5})
    y = pd.Series((f1 + f3 > 0).astype(int))

    return X, y


class TestFeatureSelector:
    """Tests for the FeatureSelector class."""

    def test_correlation_filter_drops_correlated(self, synthetic_features):
        """correlation_filter() should drop highly correlated features."""
        X, _ = synthetic_features
        selector = FeatureSelector()
        result = selector.correlation_filter(X, threshold=0.95)

        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "correlation"
        # f1 and f2 are highly correlated, one should be dropped
        assert len(result.dropped_features) >= 1
        assert len(result.selected_features) < len(X.columns)

    def test_correlation_filter_keeps_all_below_threshold(self):
        """correlation_filter() should keep all features below threshold."""
        X = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [1.0, 3.0, 2.0, 4.0],
                "c": [4.0, 1.0, 3.0, 2.0],
            }
        )
        selector = FeatureSelector()
        result = selector.correlation_filter(X, threshold=0.99)

        assert len(result.dropped_features) == 0
        assert len(result.selected_features) == 3

    def test_mutual_information_returns_top_k(self, synthetic_features):
        """mutual_information() should return exactly top_k features."""
        X, y = synthetic_features
        selector = FeatureSelector()
        result = selector.mutual_information(X, y, top_k=3)

        assert len(result.selected_features) == 3
        assert len(result.dropped_features) == 2
        assert result.scores is not None
        assert len(result.scores) == len(X.columns)

    def test_mutual_information_top_k_exceeds_columns(self, synthetic_features):
        """mutual_information() should handle top_k > number of columns."""
        X, y = synthetic_features
        selector = FeatureSelector()
        result = selector.mutual_information(X, y, top_k=100)

        assert len(result.selected_features) == len(X.columns)
        assert len(result.dropped_features) == 0

    def test_boruta_selection_runs(self, synthetic_features):
        """boruta_selection() should run without errors."""
        X, y = synthetic_features
        selector = FeatureSelector()
        result = selector.boruta_selection(X, y, max_iter=10)

        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "boruta"
        assert len(result.selected_features) + len(result.dropped_features) == len(
            X.columns
        )
        assert result.scores is not None

    def test_run_all_executes_all_methods(self, synthetic_features):
        """run_all() should populate results for all three methods."""
        X, y = synthetic_features
        selector = FeatureSelector()
        results = selector.run_all(
            X, y, correlation_threshold=0.95, mi_top_k=3, boruta_max_iter=10
        )

        assert "correlation" in results
        assert "mutual_information" in results
        assert "boruta" in results

    def test_generate_comparison_report(self, synthetic_features):
        """generate_comparison_report() should have all features and methods."""
        X, y = synthetic_features
        selector = FeatureSelector()
        selector.run_all(X, y, mi_top_k=3, boruta_max_iter=10)

        report = selector.generate_comparison_report()

        assert "feature" in report.columns
        assert "correlation_selected" in report.columns
        assert "mutual_information_selected" in report.columns
        assert "boruta_selected" in report.columns
        assert len(report) == len(X.columns)

    def test_get_selected_features(self, synthetic_features):
        """get_selected_features() should return the correct list."""
        X, y = synthetic_features
        selector = FeatureSelector()
        selector.mutual_information(X, y, top_k=3)

        features = selector.get_selected_features("mutual_information")
        assert len(features) == 3
        assert all(isinstance(f, str) for f in features)

    def test_get_selected_features_raises_for_unknown_method(self):
        """get_selected_features() should raise ValueError for unrun method."""
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="not run yet"):
            selector.get_selected_features("unknown_method")
