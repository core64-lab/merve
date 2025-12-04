"""Unit tests for logging configuration module."""
import pytest
import logging
import json
from unittest.mock import patch, MagicMock

from mlserver.logging_conf import (
    StructuredFormatter,
    configure_logging,
    set_correlation_id,
    get_correlation_id,
    log_request,
    log_response,
    log_prediction,
    correlation_id_var,
    DEFAULT_LOG_FORMAT,
)


class TestStructuredFormatter:
    """Test the StructuredFormatter class."""

    def test_basic_format(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" not in data  # Default: no timestamp

    def test_format_with_timestamp(self):
        """Test formatting with timestamp enabled."""
        formatter = StructuredFormatter(include_timestamp=True)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")

    def test_format_with_correlation_id(self):
        """Test formatting includes correlation ID when set."""
        formatter = StructuredFormatter()

        # Set correlation ID
        correlation_id_var.set("test-correlation-123")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message with correlation",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["correlation_id"] == "test-correlation-123"

        # Cleanup
        correlation_id_var.set(None)

    def test_format_with_extra_fields(self):
        """Test formatting includes extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message with extras",
            args=(),
            exc_info=None,
        )
        # Add extra fields
        record.custom_field = "custom_value"
        record.count = 42

        result = formatter.format(record)
        data = json.loads(result)

        assert data["custom_field"] == "custom_value"
        assert data["count"] == 42

    def test_format_hides_task_name_by_default(self):
        """Test that taskName is hidden by default."""
        formatter = StructuredFormatter(show_tasks=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.taskName = "Task-1"

        result = formatter.format(record)
        data = json.loads(result)

        assert "taskName" not in data

    def test_format_shows_task_name_when_enabled(self):
        """Test that taskName is shown when enabled."""
        formatter = StructuredFormatter(show_tasks=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.taskName = "Task-1"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["taskName"] == "Task-1"


class TestConfigureLogging:
    """Test the configure_logging function."""

    def setup_method(self):
        """Clear handlers before each test."""
        logging.getLogger().handlers.clear()

    def test_configure_default(self):
        """Test default configuration."""
        configure_logging()
        logger = logging.getLogger()

        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, StructuredFormatter)
        assert logger.level == logging.INFO

    def test_configure_with_level(self):
        """Test configuration with custom level."""
        configure_logging(level="DEBUG")
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

        configure_logging(level="WARNING")
        assert logger.level == logging.WARNING

    def test_configure_non_structured(self):
        """Test non-structured logging."""
        configure_logging(structured=False)
        logger = logging.getLogger()

        formatter = logger.handlers[0].formatter
        assert not isinstance(formatter, StructuredFormatter)
        assert formatter._fmt == DEFAULT_LOG_FORMAT

    def test_configure_with_custom_format(self):
        """Test configuration with custom format string."""
        custom_fmt = "%(name)s: %(message)s"
        configure_logging(custom_format=custom_fmt)
        logger = logging.getLogger()

        formatter = logger.handlers[0].formatter
        assert formatter._fmt == custom_fmt

    def test_configure_clears_existing_handlers(self):
        """Test that existing handlers are cleared."""
        logger = logging.getLogger()
        initial_count = len(logger.handlers)
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.StreamHandler())
        assert len(logger.handlers) == initial_count + 2

        configure_logging()
        # After configure_logging, should have exactly 1 handler (others cleared)
        assert len(logger.handlers) == 1


class TestCorrelationId:
    """Test correlation ID functions."""

    def setup_method(self):
        """Reset correlation ID before each test."""
        correlation_id_var.set(None)

    def test_set_correlation_id_generates_uuid(self):
        """Test that set_correlation_id generates UUID when none provided."""
        result = set_correlation_id()

        assert result is not None
        assert len(result) == 36  # UUID format
        assert get_correlation_id() == result

    def test_set_correlation_id_uses_provided_value(self):
        """Test that set_correlation_id uses provided value."""
        result = set_correlation_id("custom-id-123")

        assert result == "custom-id-123"
        assert get_correlation_id() == "custom-id-123"

    def test_get_correlation_id_returns_none_initially(self):
        """Test that get_correlation_id returns None when not set."""
        assert get_correlation_id() is None


class TestStructuredLogFunctions:
    """Test the structured logging helper functions."""

    def setup_method(self):
        """Set up logging for tests."""
        configure_logging(level="DEBUG", structured=True)

    def test_log_request(self, caplog):
        """Test log_request function."""
        with caplog.at_level(logging.INFO, logger="mlserver.request"):
            log_request("POST", "/predict", client_ip="127.0.0.1")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Request started"
        assert record.event == "request_start"
        assert record.method == "POST"
        assert record.path == "/predict"
        assert record.client_ip == "127.0.0.1"

    def test_log_response(self, caplog):
        """Test log_response function."""
        with caplog.at_level(logging.INFO, logger="mlserver.response"):
            log_response(200, 45.5, content_type="application/json")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Request completed"
        assert record.event == "request_complete"
        assert record.status_code == 200
        assert record.duration_ms == 45.5
        assert record.content_type == "application/json"

    def test_log_prediction(self, caplog):
        """Test log_prediction function."""
        with caplog.at_level(logging.INFO, logger="mlserver.prediction"):
            log_prediction("sentiment-classifier", 12.3, 100, batch_id="batch-1")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Prediction completed"
        assert record.event == "prediction_complete"
        assert record.model == "sentiment-classifier"
        assert record.duration_ms == 12.3
        assert record.sample_count == 100
        assert record.batch_id == "batch-1"


class TestStructuredFormatterEdgeCases:
    """Test edge cases in structured formatter."""

    def test_format_with_args(self):
        """Test formatting with message args."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Value is %d",
            args=(42,),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Value is 42"

    def test_format_all_log_levels(self):
        """Test formatting with all log levels."""
        formatter = StructuredFormatter()
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for level, level_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            result = formatter.format(record)
            data = json.loads(result)
            assert data["level"] == level_name
