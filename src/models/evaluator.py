"""Model evaluation with metrics, visualizations, and statistical testing."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics.

    Attributes:
        model_name: Identifier for the model.
        accuracy: Classification accuracy.
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        auc_roc: Area under the ROC curve.
        log_loss_value: Logarithmic loss.
    """

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    log_loss_value: float


class ModelEvaluator:
    """Evaluates models and generates comparison plots and reports.

    Args:
        output_dir: Directory for saving evaluation artifacts.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, EvaluationMetrics] = {}
        self.predictions: dict[str, dict] = {}

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> EvaluationMetrics:
        """Calculate all evaluation metrics for a model.

        Args:
            model_name: Identifier for the model.
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities for the positive class.

        Returns:
            EvaluationMetrics with all computed metrics.
        """
        metrics = EvaluationMetrics(
            model_name=model_name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_true, y_prob),
            log_loss_value=log_loss(y_true, y_prob),
        )

        self.results[model_name] = metrics
        self.predictions[model_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        logger.info("%s - AUC: %.4f, F1: %.4f", model_name, metrics.auc_roc, metrics.f1)
        return metrics

    def plot_roc_curves(self) -> go.Figure:
        """Generate interactive ROC curves for all evaluated models.

        Returns:
            Plotly Figure with ROC curves.
        """
        fig = go.Figure()

        for name, preds in self.predictions.items():
            fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_prob"])
            auc = self.results[name].auc_roc
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})")
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="Random"
            )
        )
        fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
        fig.write_html(self.output_dir / "roc_curves.html")
        return fig

    def plot_precision_recall_curves(self) -> go.Figure:
        """Generate interactive precision-recall curves for all models.

        Returns:
            Plotly Figure with PR curves.
        """
        fig = go.Figure()

        for name, preds in self.predictions.items():
            precision, recall, _ = precision_recall_curve(
                preds["y_true"], preds["y_prob"]
            )
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))

        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        fig.write_html(self.output_dir / "pr_curves.html")
        return fig

    def plot_confusion_matrices(self) -> None:
        """Generate confusion matrix plots for all evaluated models."""
        n_models = len(self.predictions)
        if n_models == 0:
            return

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for ax, (name, preds) in zip(axes, self.predictions.items()):
            cm = confusion_matrix(preds["y_true"], preds["y_pred"])
            disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
            disp.plot(ax=ax, cmap="Blues")
            ax.set_title(name)

        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrices.png", dpi=150)
        plt.close()

    def plot_calibration_curves(self) -> go.Figure:
        """Generate calibration curves (reliability diagrams) for all models.

        Returns:
            Plotly Figure with calibration curves.
        """
        fig = go.Figure()

        for name, preds in self.predictions.items():
            prob_true, prob_pred = calibration_curve(
                preds["y_true"], preds["y_prob"], n_bins=10
            )
            fig.add_trace(
                go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=name)
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash"},
                name="Perfectly Calibrated",
            )
        )
        fig.update_layout(
            title="Calibration Curves",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
        )
        fig.write_html(self.output_dir / "calibration_curves.html")
        return fig

    def mcnemar_test(self, model_a: str, model_b: str) -> dict:
        """Perform McNemar's test to compare two models.

        Args:
            model_a: Name of the first model.
            model_b: Name of the second model.

        Returns:
            Dictionary with test statistic, p-value, and significance flag.
        """
        pred_a = self.predictions[model_a]["y_pred"]
        pred_b = self.predictions[model_b]["y_pred"]
        y_true = self.predictions[model_a]["y_true"]

        a_correct_b_wrong = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
        a_wrong_b_correct = int(np.sum((pred_a != y_true) & (pred_b == y_true)))

        n = a_correct_b_wrong + a_wrong_b_correct
        if n == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            statistic = float(
                (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2
                / (a_correct_b_wrong + a_wrong_b_correct)
            )
            p_value = float(
                binomtest(
                    min(a_correct_b_wrong, a_wrong_b_correct),
                    n,
                    0.5,
                ).pvalue
            )

        return {
            "model_a": model_a,
            "model_b": model_b,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all evaluated models.

        Returns:
            DataFrame with one row per model and metric columns.
        """
        rows = []
        for name, metrics in self.results.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": f"{metrics.accuracy:.4f}",
                    "Precision": f"{metrics.precision:.4f}",
                    "Recall": f"{metrics.recall:.4f}",
                    "F1": f"{metrics.f1:.4f}",
                    "AUC-ROC": f"{metrics.auc_roc:.4f}",
                    "Log Loss": f"{metrics.log_loss_value:.4f}",
                }
            )
        return pd.DataFrame(rows)
