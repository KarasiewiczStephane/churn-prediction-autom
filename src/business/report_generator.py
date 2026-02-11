"""Markdown report generation for model comparison results."""

from datetime import datetime
from pathlib import Path

from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates comprehensive Markdown comparison reports.

    Args:
        evaluator: ModelEvaluator with evaluation results.
        registry: ModelRegistry with registered models.
        output_dir: Directory for saving report files.
    """

    def __init__(
        self,
        evaluator: ModelEvaluator,
        registry: ModelRegistry,
        output_dir: str = "reports",
    ) -> None:
        self.evaluator = evaluator
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self) -> str:
        """Generate a comprehensive Markdown comparison report.

        Returns:
            The generated Markdown report as a string.
        """
        comparison_df = self.evaluator.generate_comparison_table()
        best_model_id = self.registry.registry.get("best_model_id", "N/A")

        report = f"""# Churn Prediction Model Comparison Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Best Model:** {best_model_id}
- **Models Evaluated:** {len(comparison_df)}

## Model Comparison

{comparison_df.to_markdown(index=False)}

## Visualizations

### ROC Curves
![ROC Curves](roc_curves.html)

### Precision-Recall Curves
![PR Curves](pr_curves.html)

### Confusion Matrices
![Confusion Matrices](confusion_matrices.png)

### Calibration Curves
![Calibration Curves](calibration_curves.html)

## Statistical Significance

"""
        if len(self.evaluator.results) >= 2:
            model_names = list(self.evaluator.results.keys())
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i + 1 :]:
                    test_result = self.evaluator.mcnemar_test(m1, m2)
                    sig = "Yes" if test_result["significant"] else "No"
                    report += (
                        f"- **{m1} vs {m2}:** "
                        f"p-value={test_result['p_value']:.4f}, "
                        f"Significant: {sig}\n"
                    )

        report_path = self.output_dir / "model_comparison_report.md"
        report_path.write_text(report)
        logger.info("Generated comparison report at %s", report_path)
        return report
