# Customer Churn Prediction with AutoML

![CI](https://github.com/KarasiewiczStephane/churn-prediction-autom/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> End-to-end machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. Features Optuna-tuned LightGBM and Logistic Regression with automated feature selection, model evaluation, and business impact analysis.

## Features

- **Data Pipeline**: Automated download from Kaggle, validation, preprocessing with stratified 60/20/20 splitting
- **Feature Selection**: Three methods compared side-by-side -- Correlation filtering, Mutual Information ranking, and Boruta algorithm
- **Model Training**: Optuna-tuned LightGBM and Logistic Regression with cross-validation
- **Model Evaluation**: ROC curves, precision-recall curves, confusion matrices, calibration plots, and McNemar's statistical test
- **Model Registry**: Versioned model storage with metadata tracking and DuckDB results database
- **Business Impact**: Revenue-at-risk calculation, cost-benefit analysis, optimal intervention threshold, and executive summary
- **CLI Interface**: Full pipeline control via Click-based command line
- **Docker Support**: Multi-stage build with health checks for containerized deployment
- **CI/CD**: GitHub Actions pipeline with linting, testing (80%+ coverage), and Docker build verification

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Data      │───>│   Feature    │───>│    Model     │
│   Pipeline   │    │  Selection   │    │   Training   │
│              │    │              │    │              │
│ - Download   │    │ - Correlation│    │ - LightGBM   │
│ - Validate   │    │ - Mutual Info│    │ - Log. Reg.  │
│ - Preprocess │    │ - Boruta     │    │ - Optuna HPO │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐           v
│   Business   │<───│    Model     │<──────────┘
│   Impact     │    │  Evaluation  │
│              │    │              │
│ - Revenue    │    │ - ROC/PR     │
│ - Cost-Bene. │    │ - McNemar    │
│ - Exec Summ. │    │ - Comparison │
└──────────────┘    └──────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/KarasiewiczStephane/churn-prediction-autom.git
cd churn-prediction-autom

# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Run the full training pipeline
python -m src.cli train --time-budget 300

# Predict on new data (requires training first)
python -m src.cli predict --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output predictions.csv

# Run business impact analysis
python -m src.cli impact

# View results
cat reports/model_comparison_report.md
```

## CLI Commands

```bash
# Full training pipeline (data -> features -> models -> evaluation)
python -m src.cli train --time-budget 300 --feature-method boruta --optuna-trials 50

# Predict churn on new customer data (preprocesses raw CSV automatically)
python -m src.cli predict --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output predictions.csv

# Regenerate evaluation report from saved models
python -m src.cli evaluate

# Run business impact analysis on predictions
python -m src.cli impact --revenue-col MonthlyCharges

# Generate model comparison report
python -m src.cli compare
```

### CLI Options

| Command   | Option              | Default              | Description                              |
|-----------|---------------------|----------------------|------------------------------------------|
| `train`   | `--time-budget`     | 300                  | AutoML time budget in seconds            |
| `train`   | `--feature-method`  | boruta               | Feature selection: correlation, mutual_info, boruta |
| `train`   | `--optuna-trials`   | 50                   | Number of Optuna optimization trials     |
| `predict` | `--input`           | *(required)*         | Input CSV file with customer data        |
| `predict` | `--output`          | predictions.csv      | Output file path for predictions         |
| `impact`  | `--revenue-col`     | MonthlyCharges       | Column with monthly revenue              |
| *(all)*   | `--config`          | configs/config.yaml  | Path to config file                      |
| *(all)*   | `--verbose`         | false                | Enable verbose output                    |
| *(all)*   | `--quiet`           | false                | Suppress non-essential output            |

## Results

| Model               | AUC-ROC | F1 Score | Precision | Recall | Accuracy | Training Time |
|---------------------|---------|----------|-----------|--------|----------|---------------|
| **Optuna LightGBM** | 0.844   | 0.522    | 0.696     | 0.417  | 0.797    | ~545s         |
| Logistic Regression  | 0.835   | 0.570    | 0.614     | 0.532  | 0.787    | ~3s           |

*Results from the Telco Customer Churn dataset (7,043 customers, Boruta feature selection). Metrics computed on a held-out 20% test set with 50 Optuna trials.*

## Configuration

All pipeline parameters are centralized in `configs/config.yaml`:

```yaml
data:
  raw_path: data/raw
  processed_path: data/processed
  sample_path: data/sample
  test_size: 0.2
  val_size: 0.2

model:
  automl_max_runtime_secs: 300
  optuna_trials: 50
  cv_folds: 5
  random_state: 42

feature:
  correlation_threshold: 0.95
  mi_top_k: 15
  selection_method: boruta

logging:
  level: INFO
  log_dir: logs

database:
  path: results/results.duckdb

business:
  high_value_threshold: 100.0
  medium_value_threshold: 50.0
  intervention_cost: 20.0
  intervention_success_rate: 0.3
```

## Docker

```bash
# Build the image
docker build -t churn-predict .

# Run training pipeline
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  -e KAGGLE_USERNAME=your_username \
  -e KAGGLE_KEY=your_key \
  churn-predict train --time-budget 60

# Run predictions
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  churn-predict predict --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Interactive shell
docker run --rm -it --entrypoint /bin/bash churn-predict

# Docker Compose (runs training with 60s budget)
docker compose up
```

> **Note**: If you get a Docker permission error, add your user to the docker group:
> `sudo usermod -aG docker $USER` then log out and back in.

## Development

```bash
# Install dependencies
make install

# Run tests with coverage
make test

# Lint with ruff
make lint

# Format code
make format

# Run pre-commit hooks
make pre-commit

# Clean build artifacts
make clean
```

### Running Tests

```bash
# Full test suite with coverage (142 tests, 85%+ coverage)
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# Run specific test module
pytest tests/test_evaluator.py -v

# Run with coverage threshold check
coverage report --fail-under=80
```

## Project Structure

```
churn-prediction-autom/
├── src/
│   ├── __init__.py
│   ├── cli.py                          # Click CLI entry point
│   ├── data/
│   │   ├── downloader.py               # Kaggle dataset download and validation
│   │   ├── preprocessor.py             # Data preprocessing pipeline
│   │   └── feature_selector.py         # Feature selection (3 methods)
│   ├── models/
│   │   ├── automl_trainer.py           # H2O AutoML training
│   │   ├── optuna_trainer.py           # Optuna hyperparameter optimization
│   │   ├── evaluator.py               # Model evaluation and visualization
│   │   └── registry.py                # Model versioning and registry
│   ├── business/
│   │   ├── impact_calculator.py        # Revenue-at-risk and cost-benefit analysis
│   │   └── report_generator.py         # Markdown report generation
│   └── utils/
│       ├── config.py                   # YAML configuration management
│       ├── database.py                 # DuckDB results storage
│       └── logger.py                   # Structured logging
├── tests/                              # 142 tests, 85%+ coverage
│   ├── conftest.py                     # Shared test fixtures
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_logger.py
│   ├── test_downloader.py
│   ├── test_preprocessor.py
│   ├── test_feature_selector.py
│   ├── test_automl_trainer.py
│   ├── test_optuna_trainer.py
│   ├── test_evaluator.py
│   ├── test_registry.py
│   ├── test_report_generator.py
│   ├── test_impact_calculator.py
│   ├── test_cli.py
│   ├── test_docker.py
│   └── test_ci.py
├── configs/
│   └── config.yaml                     # Pipeline configuration
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI pipeline
├── Dockerfile                          # Multi-stage Docker build
├── docker-compose.yml                  # Docker Compose for development
├── Makefile                            # Build and development targets
├── requirements.txt                    # Python dependencies
├── .pre-commit-config.yaml             # Pre-commit hook configuration
├── .gitignore
├── .env.example                        # Environment variable template
└── README.md
```

## Dataset

This project uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, containing 7,043 customer records with 21 features including demographics, account information, and service usage.

**Target variable**: `Churn` (Yes/No) -- approximately 26.5% churn rate.

**Key features**: tenure, MonthlyCharges, TotalCharges, Contract type, InternetService, PaymentMethod, and more.

## License

MIT
