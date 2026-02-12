"""Click-based CLI for the churn prediction pipeline."""

import click
import pandas as pd

from src.business.impact_calculator import BusinessImpactCalculator
from src.business.report_generator import ReportGenerator
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelMetadata, ModelRegistry
from src.utils.config import Config
from src.utils.database import ResultsDB
from src.utils.logger import get_logger


@click.group()
@click.option(
    "--config",
    "-c",
    default="configs/config.yaml",
    help="Path to config file.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output.")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool, quiet: bool) -> None:
    """Churn Prediction CLI - AutoML-powered customer churn analysis."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config(config)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["logger"] = get_logger("cli")


@cli.command()
@click.option(
    "--time-budget",
    "-t",
    default=300,
    help="AutoML time budget in seconds.",
)
@click.option(
    "--feature-method",
    "-f",
    type=click.Choice(["correlation", "mutual_info", "boruta"]),
    default="boruta",
    help="Feature selection method.",
)
@click.option(
    "--optuna-trials",
    "-n",
    default=50,
    help="Number of Optuna optimization trials.",
)
@click.pass_context
def train(
    ctx: click.Context, time_budget: int, feature_method: str, optuna_trials: int
) -> None:
    """Run full training pipeline: data -> features -> models -> evaluation."""
    config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    db = ResultsDB(config.database.path)

    logger.info("Starting training pipeline...")

    from src.data.downloader import DataDownloader
    from src.data.feature_selector import FeatureSelector
    from src.data.preprocessor import DataPreprocessor
    from src.models.optuna_trainer import OptunaTrainer

    downloader = DataDownloader(
        config.data.raw_path.rsplit("/", 1)[0]
        if "/" in config.data.raw_path
        else "data"
    )
    csv_path = downloader.download()
    df = downloader.validate(csv_path)
    click.echo("Step 1/5: Data loaded and validated.")

    preprocessor = DataPreprocessor(random_state=config.model.random_state)
    data = preprocessor.fit_transform(df)
    preprocessor.save(data, config.data.processed_path)
    preprocessor.save_state()
    click.echo("Step 2/5: Data preprocessed.")

    selector = FeatureSelector(db=db)
    selector.run_all(
        data.X_train,
        data.y_train,
        correlation_threshold=config.feature.correlation_threshold,
        mi_top_k=config.feature.mi_top_k,
        boruta_max_iter=50,
    )
    method_map = {"mutual_info": "mutual_information"}
    method_key = method_map.get(feature_method, feature_method)
    selected_features = selector.get_selected_features(method_key)
    X_train = data.X_train[selected_features]
    X_test = data.X_test[selected_features]  # noqa: F841 (X_val used by H2O path)

    import json
    from pathlib import Path

    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/selected_features.json", "w") as f:
        json.dump(selected_features, f)

    click.echo(
        f"Step 3/5: Feature selection complete ({len(selected_features)} features)."
    )

    optuna_trainer = OptunaTrainer(
        n_trials=optuna_trials,
        cv_folds=config.model.cv_folds,
        random_state=config.model.random_state,
    )
    lgb_result = optuna_trainer.optimize_lightgbm(X_train, data.y_train)
    lr_result = optuna_trainer.optimize_logistic_regression(X_train, data.y_train)
    click.echo("Step 4/5: Model training complete.")

    evaluator = ModelEvaluator()
    registry = ModelRegistry(db=db)

    for result in [lgb_result, lr_result]:
        y_pred = result.best_model.predict(X_test)
        y_prob = result.best_model.predict_proba(X_test)[:, 1]
        metrics = evaluator.evaluate(
            result.model_name, data.y_test.values, y_pred, y_prob
        )

        metadata = ModelMetadata(
            model_id=result.model_name,
            model_type="lightgbm" if result.model_name == "lightgbm" else "sklearn",
            created_at=pd.Timestamp.now().isoformat(),
            metrics={
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "auc_roc": metrics.auc_roc,
                "log_loss": metrics.log_loss_value,
            },
            hyperparameters=result.best_params,
            feature_selection_method=feature_method,
            training_time_secs=result.training_time_secs,
        )
        registry.register_model(result.best_model, metadata)

    best_model = max(evaluator.results.values(), key=lambda m: m.auc_roc)
    registry.set_best_model(best_model.model_name)

    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_calibration_curves()

    report_gen = ReportGenerator(evaluator, registry)
    report_gen.generate_markdown_report()

    click.echo("Step 5/5: Evaluation complete.")
    click.echo(click.style("Training pipeline finished!", fg="green"))


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input CSV file with customer data.",
)
@click.option(
    "--output",
    "-o",
    default="predictions.csv",
    help="Output file path for predictions.",
)
@click.pass_context
def predict(ctx: click.Context, input_path: str, output: str) -> None:
    """Predict churn on new customer data using the best model."""
    import json

    from src.data.preprocessor import DataPreprocessor

    registry = ModelRegistry()
    model, metadata = registry.get_best_model()

    preprocessor = DataPreprocessor.load_state()
    with open("models/selected_features.json") as f:
        selected_features = json.load(f)

    df_raw = pd.read_csv(input_path)
    df_transformed = preprocessor.transform(df_raw)
    df_predict = df_transformed[selected_features]

    predictions = model.predict_proba(df_predict)[:, 1]
    df_raw["churn_probability"] = predictions
    df_raw["churn_predicted"] = (predictions >= 0.5).astype(int)
    df_raw.to_csv(output, index=False)

    click.echo(f"Predictions saved to {output}")


@cli.command()
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Regenerate evaluation report from saved models."""
    evaluator = ModelEvaluator()
    registry = ModelRegistry()

    report_gen = ReportGenerator(evaluator, registry)
    report_gen.generate_markdown_report()
    click.echo("Evaluation report generated in reports/")


@cli.command()
@click.option(
    "--revenue-col",
    "-r",
    default="MonthlyCharges",
    help="Column with monthly revenue.",
)
@click.pass_context
def impact(ctx: click.Context, revenue_col: str) -> None:
    """Run business impact analysis on predictions."""
    calculator = BusinessImpactCalculator()

    predictions_path = "predictions.csv"
    df = pd.read_csv(predictions_path)
    if "churn_probability" in df.columns:
        df = df.rename(columns={"churn_probability": "churn_prob"})

    df = calculator.calculate_revenue_at_risk(df, revenue_col=revenue_col)
    buckets = calculator.bucket_analysis(df)
    cb = calculator.cost_benefit_analysis(df)
    summary = calculator.generate_executive_summary(df, buckets, cb)
    calculator.export_results(df, summary)

    click.echo("Business impact analysis complete. See reports/")


@cli.command()
@click.pass_context
def compare(ctx: click.Context) -> None:
    """Generate model comparison report."""
    evaluator = ModelEvaluator()
    registry = ModelRegistry()

    report_gen = ReportGenerator(evaluator, registry)
    report_gen.generate_markdown_report()
    click.echo("Comparison report generated: reports/model_comparison_report.md")


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
