"""Tests for the churn prediction dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    generate_business_impact,
    generate_churn_distribution,
    generate_feature_importance,
    generate_model_comparison,
    generate_roc_curve,
)


class TestModelComparison:
    def test_returns_dataframe(self) -> None:
        df = generate_model_comparison()
        assert isinstance(df, pd.DataFrame)

    def test_has_four_models(self) -> None:
        df = generate_model_comparison()
        assert len(df) == 4

    def test_has_required_columns(self) -> None:
        df = generate_model_comparison()
        for col in ["model", "auc_roc", "precision", "recall", "f1", "accuracy"]:
            assert col in df.columns

    def test_scores_bounded(self) -> None:
        df = generate_model_comparison()
        for col in ["auc_roc", "precision", "recall", "f1", "accuracy"]:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()

    def test_reproducible(self) -> None:
        df1 = generate_model_comparison(seed=99)
        df2 = generate_model_comparison(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestFeatureImportance:
    def test_returns_dataframe(self) -> None:
        df = generate_feature_importance()
        assert isinstance(df, pd.DataFrame)

    def test_correct_feature_count(self) -> None:
        df = generate_feature_importance()
        assert len(df) == 10

    def test_importance_positive(self) -> None:
        df = generate_feature_importance()
        assert (df["importance"] > 0).all()

    def test_sorted_descending(self) -> None:
        df = generate_feature_importance()
        assert df["importance"].is_monotonic_decreasing


class TestChurnDistribution:
    def test_returns_dataframe(self) -> None:
        df = generate_churn_distribution()
        assert isinstance(df, pd.DataFrame)

    def test_has_segments(self) -> None:
        df = generate_churn_distribution()
        assert len(df) == 6

    def test_churned_plus_retained_equals_total(self) -> None:
        df = generate_churn_distribution()
        assert (df["churned"] + df["retained"] == df["total_customers"]).all()

    def test_churn_rate_bounded(self) -> None:
        df = generate_churn_distribution()
        assert (df["churn_rate"] >= 0).all()
        assert (df["churn_rate"] <= 1).all()


class TestBusinessImpact:
    def test_returns_dict(self) -> None:
        impact = generate_business_impact()
        assert isinstance(impact, dict)

    def test_has_required_keys(self) -> None:
        impact = generate_business_impact()
        for key in [
            "monthly_revenue_at_risk",
            "avg_customer_lifetime_value",
            "retention_cost_per_customer",
            "break_even_retention_rate",
            "predicted_churners",
            "optimal_threshold",
        ]:
            assert key in impact

    def test_positive_values(self) -> None:
        impact = generate_business_impact()
        assert impact["monthly_revenue_at_risk"] > 0
        assert impact["predicted_churners"] > 0


class TestRocCurve:
    def test_returns_dataframe(self) -> None:
        df = generate_roc_curve()
        assert isinstance(df, pd.DataFrame)

    def test_starts_at_origin(self) -> None:
        df = generate_roc_curve()
        assert df["fpr"].iloc[0] == 0.0

    def test_ends_at_one(self) -> None:
        df = generate_roc_curve()
        assert df["fpr"].iloc[-1] == 1.0

    def test_values_bounded(self) -> None:
        df = generate_roc_curve()
        assert (df["fpr"] >= 0).all() and (df["fpr"] <= 1).all()
        assert (df["tpr"] >= 0).all() and (df["tpr"] <= 1).all()
