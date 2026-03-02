"""Streamlit dashboard for churn prediction visualization.

Displays model performance metrics, feature importance, churn
distribution, and business impact analysis using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

FEATURE_NAMES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract_TwoYear",
    "InternetService_Fiber",
    "PaymentMethod_Electronic",
    "OnlineSecurity_No",
    "TechSupport_No",
    "Contract_Monthly",
    "PaperlessBilling",
]


def generate_model_comparison(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic model comparison metrics."""
    rng = np.random.default_rng(seed)
    models = ["LightGBM (Optuna)", "Logistic Regression", "Random Forest", "XGBoost"]
    rows = []
    base_scores = [0.86, 0.79, 0.83, 0.85]
    for model, base in zip(models, base_scores):
        noise = rng.uniform(-0.02, 0.02)
        auc = round(base + noise, 4)
        rows.append(
            {
                "model": model,
                "auc_roc": auc,
                "precision": round(auc - rng.uniform(0.02, 0.06), 4),
                "recall": round(auc - rng.uniform(0.03, 0.08), 4),
                "f1": round(auc - rng.uniform(0.02, 0.05), 4),
                "accuracy": round(auc - rng.uniform(0.01, 0.03), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_feature_importance(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic feature importance scores."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.02, 0.25, size=len(FEATURE_NAMES))
    scores = np.sort(scores)[::-1]
    return pd.DataFrame({"feature": FEATURE_NAMES, "importance": np.round(scores, 4)})


def generate_churn_distribution(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic churn distribution by segment."""
    rng = np.random.default_rng(seed)
    segments = [
        "Month-to-Month",
        "One Year",
        "Two Year",
        "Fiber Optic",
        "DSL",
        "No Internet",
    ]
    rows = []
    for seg in segments:
        total = int(rng.integers(200, 800))
        churn_rate = rng.uniform(0.05, 0.55)
        churned = int(total * churn_rate)
        rows.append(
            {
                "segment": seg,
                "total_customers": total,
                "churned": churned,
                "retained": total - churned,
                "churn_rate": round(churn_rate, 4),
            }
        )
    return pd.DataFrame(rows)


def generate_business_impact(seed: int = 42) -> dict:
    """Generate synthetic business impact metrics."""
    rng = np.random.default_rng(seed)
    monthly_revenue_at_risk = round(rng.uniform(150000, 350000), 2)
    avg_customer_value = round(rng.uniform(55, 95), 2)
    retention_cost = round(rng.uniform(15, 35), 2)
    return {
        "monthly_revenue_at_risk": monthly_revenue_at_risk,
        "avg_customer_lifetime_value": avg_customer_value * 12,
        "retention_cost_per_customer": retention_cost,
        "break_even_retention_rate": round(retention_cost / avg_customer_value, 4),
        "predicted_churners": int(monthly_revenue_at_risk / avg_customer_value),
        "optimal_threshold": round(rng.uniform(0.35, 0.55), 2),
    }


def generate_roc_curve(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ROC curve points."""
    rng = np.random.default_rng(seed)
    fpr = np.sort(np.concatenate([[0.0], rng.uniform(0, 1, 50), [1.0]]))
    tpr = np.sort(np.concatenate([[0.0], rng.uniform(0, 1, 50), [1.0]]))
    tpr = np.clip(tpr + 0.15, 0.0, 1.0)
    tpr = np.sort(tpr)
    return pd.DataFrame({"fpr": np.round(fpr, 4), "tpr": np.round(tpr, 4)})


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Customer Churn Prediction Dashboard")
    st.caption(
        "AutoML-powered churn analysis with feature selection, "
        "model comparison, and business impact estimation"
    )


def render_summary_metrics(impact: dict, models_df: pd.DataFrame) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    best = models_df.loc[models_df["auc_roc"].idxmax()]
    col1.metric("Best Model", best["model"].split("(")[0].strip())
    col2.metric("AUC-ROC", f"{best['auc_roc']:.4f}")
    col3.metric("Revenue at Risk", f"${impact['monthly_revenue_at_risk']:,.0f}/mo")
    col4.metric("Predicted Churners", f"{impact['predicted_churners']:,}")


def render_model_comparison(models_df: pd.DataFrame) -> None:
    """Render model comparison bar chart."""
    st.subheader("Model Performance Comparison")
    metrics = ["auc_roc", "precision", "recall", "f1", "accuracy"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric.upper().replace("_", " "),
                x=models_df["model"],
                y=models_df[metric],
                text=models_df[metric].apply(lambda x: f"{x:.3f}"),
                textposition="auto",
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0.6, 1.0]},
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(feat_df: pd.DataFrame) -> None:
    """Render feature importance horizontal bar chart."""
    st.subheader("Feature Importance")
    fig = px.bar(
        feat_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_churn_distribution(dist_df: pd.DataFrame) -> None:
    """Render churn distribution by segment."""
    st.subheader("Churn Rate by Segment")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dist_df["segment"],
            y=dist_df["churn_rate"],
            marker_color=[
                "#F44336" if r > 0.3 else "#FF9800" if r > 0.15 else "#4CAF50"
                for r in dist_df["churn_rate"]
            ],
            text=dist_df["churn_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="auto",
        )
    )
    fig.update_layout(
        yaxis_title="Churn Rate",
        yaxis={"tickformat": ".0%"},
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_roc_curve(roc_df: pd.DataFrame) -> None:
    """Render ROC curve."""
    st.subheader("ROC Curve")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_df["fpr"],
            y=roc_df["tpr"],
            mode="lines",
            name="Model",
            line={"color": "#2196F3", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "gray", "dash": "dash"},
        )
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_business_impact(impact: dict) -> None:
    """Render business impact metrics."""
    st.subheader("Business Impact Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Customer Lifetime Value",
            f"${impact['avg_customer_lifetime_value']:,.0f}",
        )
        st.metric(
            "Retention Cost / Customer",
            f"${impact['retention_cost_per_customer']:,.0f}",
        )
    with col2:
        st.metric(
            "Break-Even Retention Rate",
            f"{impact['break_even_retention_rate']:.1%}",
        )
        st.metric("Optimal Threshold", f"{impact['optimal_threshold']:.2f}")


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    models_df = generate_model_comparison()
    feat_df = generate_feature_importance()
    dist_df = generate_churn_distribution()
    impact = generate_business_impact()
    roc_df = generate_roc_curve()

    render_summary_metrics(impact, models_df)
    st.markdown("---")

    render_model_comparison(models_df)

    col_left, col_right = st.columns(2)
    with col_left:
        render_feature_importance(feat_df)
    with col_right:
        render_roc_curve(roc_df)

    st.markdown("---")
    render_churn_distribution(dist_df)
    st.markdown("---")
    render_business_impact(impact)


if __name__ == "__main__":
    main()
