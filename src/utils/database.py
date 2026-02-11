"""DuckDB-based results storage for model runs and feature selections."""

from pathlib import Path

import duckdb

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResultsDB:
    """Database interface for storing and querying experiment results.

    Args:
        db_path: Path to the DuckDB database file.
    """

    def __init__(self, db_path: str = "results/results.duckdb") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create database tables if they do not exist."""
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS model_runs_id_seq;
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER DEFAULT nextval('model_runs_id_seq') PRIMARY KEY,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_type VARCHAR,
                model_name VARCHAR,
                accuracy DOUBLE,
                precision_score DOUBLE,
                recall DOUBLE,
                f1 DOUBLE,
                auc_roc DOUBLE,
                log_loss DOUBLE,
                training_time_secs DOUBLE,
                hyperparameters JSON,
                feature_selection_method VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS feature_selections_id_seq;
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_selections (
                id INTEGER DEFAULT nextval('feature_selections_id_seq') PRIMARY KEY,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                method VARCHAR,
                features_selected JSON,
                features_dropped JSON,
                n_features_selected INTEGER
            )
        """)
        logger.info("Database schema initialized at %s", self.db_path)

    def insert_model_run(
        self,
        model_type: str,
        model_name: str,
        metrics: dict,
        training_time_secs: float,
        hyperparameters: str,
        feature_selection_method: str,
    ) -> None:
        """Insert a model run record into the database.

        Args:
            model_type: Type of model (e.g., 'h2o', 'sklearn', 'lightgbm').
            model_name: Unique model identifier.
            metrics: Dictionary with keys: accuracy, precision, recall, f1, auc_roc, log_loss.
            training_time_secs: Training duration in seconds.
            hyperparameters: JSON string of hyperparameters.
            feature_selection_method: Method used for feature selection.
        """
        self.conn.execute(
            """
            INSERT INTO model_runs (
                model_type, model_name, accuracy, precision_score,
                recall, f1, auc_roc, log_loss, training_time_secs,
                hyperparameters, feature_selection_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model_type,
                model_name,
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("auc_roc"),
                metrics.get("log_loss"),
                training_time_secs,
                hyperparameters,
                feature_selection_method,
            ],
        )

    def insert_feature_selection(
        self,
        method: str,
        features_selected: str,
        features_dropped: str,
        n_features_selected: int,
    ) -> None:
        """Insert a feature selection record into the database.

        Args:
            method: Feature selection method name.
            features_selected: JSON string of selected feature names.
            features_dropped: JSON string of dropped feature names.
            n_features_selected: Number of features selected.
        """
        self.conn.execute(
            """
            INSERT INTO feature_selections (
                method, features_selected, features_dropped, n_features_selected
            ) VALUES (?, ?, ?, ?)
            """,
            [method, features_selected, features_dropped, n_features_selected],
        )

    def get_model_runs(self) -> list[dict]:
        """Retrieve all model runs from the database.

        Returns:
            List of dictionaries representing model run records.
        """
        result = self.conn.execute(
            "SELECT * FROM model_runs ORDER BY run_timestamp DESC"
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
