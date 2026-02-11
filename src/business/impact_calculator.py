"""Business impact analysis with revenue-at-risk and cost-benefit calculations."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CustomerValueTiers:
    """Thresholds for customer value segmentation.

    Attributes:
        high_threshold: Monthly revenue at or above this is high value.
        medium_threshold: Monthly revenue at or above this (below high) is medium.
    """

    high_threshold: float = 100.0
    medium_threshold: float = 50.0


@dataclass
class InterventionCost:
    """Cost parameters for retention interventions.

    Attributes:
        cost_per_customer: Cost to attempt retention per customer.
        success_rate: Probability that intervention prevents churn.
    """

    cost_per_customer: float = 20.0
    success_rate: float = 0.3


@dataclass
class BucketAnalysis:
    """Analysis results for a churn probability bucket.

    Attributes:
        bucket: Label for the probability range (e.g., '60-80%').
        customer_count: Number of customers in this bucket.
        avg_probability: Average churn probability in this bucket.
        total_revenue_at_risk: Total annualized revenue at risk.
        expected_churn_count: Expected number of churning customers.
    """

    bucket: str
    customer_count: int
    avg_probability: float
    total_revenue_at_risk: float
    expected_churn_count: float


class BusinessImpactCalculator:
    """Calculates business impact metrics from churn predictions.

    Args:
        value_tiers: Customer value tier thresholds.
        intervention: Retention intervention cost parameters.
    """

    def __init__(
        self,
        value_tiers: CustomerValueTiers | None = None,
        intervention: InterventionCost | None = None,
    ) -> None:
        self.value_tiers = value_tiers or CustomerValueTiers()
        self.intervention = intervention or InterventionCost()
        self.buckets = [
            (0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
        ]

    def calculate_revenue_at_risk(
        self,
        df: pd.DataFrame,
        prob_col: str = "churn_prob",
        revenue_col: str = "MonthlyCharges",
    ) -> pd.DataFrame:
        """Calculate annualized revenue at risk per customer.

        Args:
            df: DataFrame with prediction probabilities and revenue.
            prob_col: Column name for churn probability.
            revenue_col: Column name for monthly revenue.

        Returns:
            DataFrame with added revenue_at_risk and value_tier columns.
        """
        df = df.copy()
        df["revenue_at_risk"] = df[prob_col] * df[revenue_col] * 12
        df["value_tier"] = df[revenue_col].apply(self._assign_tier)
        return df

    def _assign_tier(self, revenue: float) -> str:
        """Assign a customer to a value tier based on monthly revenue.

        Args:
            revenue: Monthly revenue amount.

        Returns:
            Tier label: 'high', 'medium', or 'low'.
        """
        if revenue >= self.value_tiers.high_threshold:
            return "high"
        elif revenue >= self.value_tiers.medium_threshold:
            return "medium"
        return "low"

    def bucket_analysis(
        self, df: pd.DataFrame, prob_col: str = "churn_prob"
    ) -> list[BucketAnalysis]:
        """Analyze customers grouped by churn probability buckets.

        Args:
            df: DataFrame with churn probabilities and revenue_at_risk.
            prob_col: Column name for churn probability.

        Returns:
            List of BucketAnalysis results, one per non-empty bucket.
        """
        results = []
        for low, high in self.buckets:
            mask = (df[prob_col] >= low) & (df[prob_col] < high)
            bucket_df = df[mask]

            if len(bucket_df) == 0:
                continue

            results.append(
                BucketAnalysis(
                    bucket=f"{int(low * 100)}-{int(high * 100)}%",
                    customer_count=len(bucket_df),
                    avg_probability=float(bucket_df[prob_col].mean()),
                    total_revenue_at_risk=float(bucket_df["revenue_at_risk"].sum()),
                    expected_churn_count=float(bucket_df[prob_col].sum()),
                )
            )
        return results

    def cost_benefit_analysis(
        self,
        df: pd.DataFrame,
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        """Calculate net benefit at different intervention thresholds.

        Args:
            df: DataFrame with churn_prob, MonthlyCharges, and revenue_at_risk.
            thresholds: List of probability thresholds to evaluate.

        Returns:
            DataFrame with cost-benefit analysis at each threshold.
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        results = []
        for threshold in thresholds:
            target_customers = df[df["churn_prob"] >= threshold]
            n_customers = len(target_customers)

            intervention_cost = n_customers * self.intervention.cost_per_customer
            if n_customers > 0:
                avg_prob = float(target_customers["churn_prob"].mean())
                avg_revenue = float(target_customers["MonthlyCharges"].mean())
            else:
                avg_prob = 0.0
                avg_revenue = 0.0

            expected_saves = n_customers * avg_prob * self.intervention.success_rate
            saved_revenue = expected_saves * avg_revenue * 12
            net_benefit = saved_revenue - intervention_cost

            roi = (
                (saved_revenue / intervention_cost - 1) * 100
                if intervention_cost > 0
                else 0
            )

            results.append(
                {
                    "threshold": threshold,
                    "customers_targeted": n_customers,
                    "intervention_cost": intervention_cost,
                    "expected_saves": expected_saves,
                    "saved_revenue": saved_revenue,
                    "net_benefit": net_benefit,
                    "roi": roi,
                }
            )

        return pd.DataFrame(results)

    def get_optimal_threshold(self, cost_benefit_df: pd.DataFrame) -> float:
        """Find the threshold that maximizes net benefit.

        Args:
            cost_benefit_df: DataFrame from cost_benefit_analysis().

        Returns:
            The optimal threshold value.
        """
        optimal_row = cost_benefit_df.loc[cost_benefit_df["net_benefit"].idxmax()]
        return float(optimal_row["threshold"])

    def generate_executive_summary(
        self,
        df: pd.DataFrame,
        bucket_results: list[BucketAnalysis],
        cost_benefit_df: pd.DataFrame,
    ) -> dict:
        """Generate an executive summary of the business impact analysis.

        Args:
            df: DataFrame with revenue_at_risk and churn_prob columns.
            bucket_results: List of BucketAnalysis from bucket_analysis().
            cost_benefit_df: DataFrame from cost_benefit_analysis().

        Returns:
            Dictionary with summary metrics and segment recommendations.
        """
        total_at_risk = float(df["revenue_at_risk"].sum())
        optimal_threshold = self.get_optimal_threshold(cost_benefit_df)
        optimal_row = cost_benefit_df[
            cost_benefit_df["threshold"] == optimal_threshold
        ].iloc[0]

        segment_actions = []
        for bucket in bucket_results:
            if "80-100%" in bucket.bucket:
                action = "Immediate personal outreach required"
            elif "60-80%" in bucket.bucket:
                action = "Targeted retention offer recommended"
            elif "40-60%" in bucket.bucket:
                action = "Proactive engagement campaign"
            else:
                action = "Monitor and maintain relationship"

            segment_actions.append(
                {
                    "segment": bucket.bucket,
                    "customers": bucket.customer_count,
                    "revenue_at_risk": bucket.total_revenue_at_risk,
                    "recommended_action": action,
                }
            )

        return {
            "total_revenue_at_risk": total_at_risk,
            "total_customers": len(df),
            "high_risk_customers": int(len(df[df["churn_prob"] >= 0.6])),
            "optimal_threshold": optimal_threshold,
            "expected_net_benefit": float(optimal_row["net_benefit"]),
            "expected_roi": float(optimal_row["roi"]),
            "segment_recommendations": segment_actions,
        }

    def export_results(
        self,
        df: pd.DataFrame,
        summary: dict,
        output_dir: str = "reports",
    ) -> None:
        """Export analysis results to CSV and JSON files.

        Args:
            df: DataFrame with customer impact analysis.
            summary: Executive summary dictionary.
            output_dir: Directory for output files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path / "customer_impact_analysis.csv", index=False)

        with open(output_path / "executive_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Exported results to %s", output_path)
