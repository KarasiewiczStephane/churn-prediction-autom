"""Tests for DuckDB results storage module."""

import json

import pytest

from src.utils.database import ResultsDB


@pytest.fixture()
def db(tmp_path):
    """Create a temporary database instance."""
    db_path = str(tmp_path / "test_results.duckdb")
    database = ResultsDB(db_path=db_path)
    yield database
    database.close()


class TestResultsDB:
    """Tests for the ResultsDB class."""

    def test_connection_and_schema_creation(self, db):
        """Database should create tables on initialization."""
        tables = db.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "model_runs" in table_names
        assert "feature_selections" in table_names

    def test_insert_and_query_model_run(self, db):
        """Should insert and retrieve model run records."""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
            "auc_roc": 0.88,
            "log_loss": 0.45,
        }
        db.insert_model_run(
            model_type="lightgbm",
            model_name="lgb_v1",
            metrics=metrics,
            training_time_secs=120.5,
            hyperparameters=json.dumps({"n_estimators": 100}),
            feature_selection_method="boruta",
        )

        runs = db.get_model_runs()
        assert len(runs) == 1
        assert runs[0]["model_name"] == "lgb_v1"
        assert runs[0]["accuracy"] == 0.85
        assert runs[0]["feature_selection_method"] == "boruta"

    def test_insert_feature_selection(self, db):
        """Should insert feature selection records."""
        db.insert_feature_selection(
            method="correlation",
            features_selected=json.dumps(["f1", "f2", "f3"]),
            features_dropped=json.dumps(["f4"]),
            n_features_selected=3,
        )

        result = db.conn.execute("SELECT * FROM feature_selections").fetchall()
        assert len(result) == 1
        assert result[0][2] == "correlation"
        assert result[0][5] == 3

    def test_multiple_model_runs(self, db):
        """Should handle multiple model run insertions."""
        for i in range(3):
            db.insert_model_run(
                model_type="sklearn",
                model_name=f"model_{i}",
                metrics={"accuracy": 0.80 + i * 0.05},
                training_time_secs=float(i * 10),
                hyperparameters=json.dumps({}),
                feature_selection_method="mutual_info",
            )

        runs = db.get_model_runs()
        assert len(runs) == 3

    def test_database_file_created(self, tmp_path):
        """Database file should be created at the specified path."""
        db_path = str(tmp_path / "subdir" / "test.duckdb")
        database = ResultsDB(db_path=db_path)
        from pathlib import Path

        assert Path(db_path).parent.exists()
        database.close()
