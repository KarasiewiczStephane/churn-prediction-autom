"""Tests for business impact calculator."""

import pandas as pd
import pytest

from src.business.impact_calculator import (
    BusinessImpactCalculator,
    CustomerValueTiers,
    InterventionCost,
)


@pytest.fixture()
def customer_df():
    """Create a synthetic customer DataFrame with churn probabilities."""
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(10)],
            "MonthlyCharges": [30, 45, 55, 70, 85, 95, 105, 110, 120, 130],
            "churn_prob": [0.1, 0.15, 0.35, 0.45, 0.55, 0.65, 0.75, 0.82, 0.9, 0.95],
        }
    )


@pytest.fixture()
def calculator():
    """Create a BusinessImpactCalculator with default parameters."""
    return BusinessImpactCalculator()


class TestBusinessImpactCalculator:
    """Tests for the BusinessImpactCalculator class."""

    def test_calculate_revenue_at_risk(self, calculator, customer_df):
        """calculate_revenue_at_risk() should compute annualized risk."""
        result = calculator.calculate_revenue_at_risk(customer_df)

        assert "revenue_at_risk" in result.columns
        assert "value_tier" in result.columns
        expected_risk_0 = 0.1 * 30 * 12
        assert abs(result.iloc[0]["revenue_at_risk"] - expected_risk_0) < 0.01

    def test_assign_tier_high(self, calculator):
        """_assign_tier() should classify high-value customers."""
        assert calculator._assign_tier(150.0) == "high"
        assert calculator._assign_tier(100.0) == "high"

    def test_assign_tier_medium(self, calculator):
        """_assign_tier() should classify medium-value customers."""
        assert calculator._assign_tier(75.0) == "medium"
        assert calculator._assign_tier(50.0) == "medium"

    def test_assign_tier_low(self, calculator):
        """_assign_tier() should classify low-value customers."""
        assert calculator._assign_tier(30.0) == "low"
        assert calculator._assign_tier(0.0) == "low"

    def test_bucket_analysis(self, calculator, customer_df):
        """bucket_analysis() should return correct counts per bucket."""
        df = calculator.calculate_revenue_at_risk(customer_df)
        results = calculator.bucket_analysis(df)

        assert len(results) > 0
        total_customers = sum(b.customer_count for b in results)
        assert total_customers == len(customer_df)

    def test_bucket_analysis_probabilities(self, calculator, customer_df):
        """bucket_analysis() should compute valid average probabilities."""
        df = calculator.calculate_revenue_at_risk(customer_df)
        results = calculator.bucket_analysis(df)

        for bucket in results:
            assert 0 <= bucket.avg_probability <= 1
            assert bucket.customer_count > 0
            assert bucket.total_revenue_at_risk >= 0

    def test_cost_benefit_analysis(self, calculator, customer_df):
        """cost_benefit_analysis() should return DataFrame with expected columns."""
        df = calculator.calculate_revenue_at_risk(customer_df)
        cb = calculator.cost_benefit_analysis(df)

        assert isinstance(cb, pd.DataFrame)
        assert "threshold" in cb.columns
        assert "net_benefit" in cb.columns
        assert "roi" in cb.columns
        assert len(cb) == 6

    def test_get_optimal_threshold(self, calculator, customer_df):
        """get_optimal_threshold() should return threshold with max net benefit."""
        df = calculator.calculate_revenue_at_risk(customer_df)
        cb = calculator.cost_benefit_analysis(df)
        threshold = calculator.get_optimal_threshold(cb)

        assert 0 <= threshold <= 1
        max_benefit = cb["net_benefit"].max()
        optimal_row = cb[cb["threshold"] == threshold].iloc[0]
        assert optimal_row["net_benefit"] == max_benefit

    def test_generate_executive_summary(self, calculator, customer_df):
        """generate_executive_summary() should contain all required fields."""
        df = calculator.calculate_revenue_at_risk(customer_df)
        buckets = calculator.bucket_analysis(df)
        cb = calculator.cost_benefit_analysis(df)
        summary = calculator.generate_executive_summary(df, buckets, cb)

        assert "total_revenue_at_risk" in summary
        assert "total_customers" in summary
        assert "high_risk_customers" in summary
        assert "optimal_threshold" in summary
        assert "expected_net_benefit" in summary
        assert "segment_recommendations" in summary
        assert summary["total_customers"] == 10

    def test_export_results(self, calculator, customer_df, tmp_path):
        """export_results() should create CSV and JSON files."""
        df = calculator.calculate_revenue_at_risk(customer_df)
        buckets = calculator.bucket_analysis(df)
        cb = calculator.cost_benefit_analysis(df)
        summary = calculator.generate_executive_summary(df, buckets, cb)

        output_dir = str(tmp_path / "reports")
        calculator.export_results(df, summary, output_dir)

        from pathlib import Path

        assert (Path(output_dir) / "customer_impact_analysis.csv").exists()
        assert (Path(output_dir) / "executive_summary.json").exists()

    def test_custom_tiers_and_intervention(self, customer_df):
        """Calculator should respect custom tier and intervention parameters."""
        tiers = CustomerValueTiers(high_threshold=80, medium_threshold=40)
        intervention = InterventionCost(cost_per_customer=10, success_rate=0.5)
        calc = BusinessImpactCalculator(value_tiers=tiers, intervention=intervention)

        df = calc.calculate_revenue_at_risk(customer_df)
        assert df.iloc[0]["value_tier"] == "low"
        assert df.iloc[-1]["value_tier"] == "high"

    def test_no_high_risk_customers(self, calculator):
        """Should handle case with no high-risk customers."""
        df = pd.DataFrame(
            {
                "MonthlyCharges": [50, 60, 70],
                "churn_prob": [0.1, 0.2, 0.15],
            }
        )
        df = calculator.calculate_revenue_at_risk(df)
        buckets = calculator.bucket_analysis(df)
        cb = calculator.cost_benefit_analysis(df)
        summary = calculator.generate_executive_summary(df, buckets, cb)

        assert summary["high_risk_customers"] == 0
