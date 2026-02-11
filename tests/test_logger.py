"""Tests for structured logging module."""

import logging

from src.utils.logger import get_logger


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_logger_instance(self):
        """Should return a logging.Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_console_handler_attached(self):
        """Logger should have a console handler."""
        logger = get_logger("test_console")
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types

    def test_file_handler_when_log_file_provided(self, tmp_path):
        """Logger should add file handler when log_file is specified."""
        log_file = tmp_path / "test.log"
        logger = get_logger("test_file", log_file=log_file)

        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "FileHandler" in handler_types

    def test_log_file_created(self, tmp_path):
        """Log file should be created when specified."""
        log_file = tmp_path / "logs" / "test.log"
        logger = get_logger("test_file_creation", log_file=log_file)
        logger.info("test message")
        assert log_file.exists()

    def test_no_duplicate_handlers(self):
        """Calling get_logger twice should not duplicate handlers."""
        logger1 = get_logger("test_dedup")
        handler_count = len(logger1.handlers)
        logger2 = get_logger("test_dedup")
        assert len(logger2.handlers) == handler_count
        assert logger1 is logger2

    def test_debug_level_set(self):
        """Logger should be set to DEBUG level."""
        logger = get_logger("test_level")
        assert logger.level == logging.DEBUG

    def test_log_file_in_subdirectory(self, tmp_path):
        """Logger should create parent directories for log file."""
        log_file = tmp_path / "deep" / "nested" / "test.log"
        logger = get_logger("test_nested_dir", log_file=log_file)
        logger.info("test")
        assert log_file.parent.exists()
