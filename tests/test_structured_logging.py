"""
Tests for the structured logging infrastructure.

Verifies that:
- Logging is properly configured
- JSON format is correct
- All log functions work without errors
- Timer and context managers work
"""

import json
import pytest
from io import StringIO
from unittest.mock import patch
import logging

from core.structured_logging import (
    setup_logging,
    get_logger,
    JSONFormatter,
    ConsoleFormatter,
    LogContext,
    Timer,
    log_query,
    log_intent,
    log_filters,
    log_search,
    log_llm_call,
    log_products_shown,
    log_response,
    log_error,
    log_guidance,
    timed,
)


class TestJSONFormatter:
    """Tests for JSON log formatting."""

    def test_basic_format(self):
        """Test that basic log record is formatted as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_extra_fields(self):
        """Test that extra fields are included in JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.session_id = "test-session"
        record.query = "USB-C cable"
        record.intent = "new_search"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["session_id"] == "test-session"
        assert data["query"] == "USB-C cable"
        assert data["intent"] == "new_search"


class TestConsoleFormatter:
    """Tests for console log formatting."""

    def test_basic_format(self):
        """Test that console output is human-readable."""
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        assert "INFO" in result
        assert "test.module" in result
        assert "Test message" in result


class TestLogContext:
    """Tests for logging context manager."""

    def test_context_manager(self):
        """Test that LogContext tracks timing."""
        with LogContext(session_id="test-123") as ctx:
            assert ctx.session_id == "test-123"
            # Do some work
            pass

        assert ctx.elapsed_ms() > 0

    def test_auto_generated_session_id(self):
        """Test that session ID is auto-generated if not provided."""
        with LogContext() as ctx:
            assert ctx.session_id is not None
            assert len(ctx.session_id) == 8


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_measures_time(self):
        """Test that Timer measures elapsed time."""
        with Timer() as t:
            # Do some work
            _ = sum(range(1000))

        assert t.elapsed_ms > 0

    def test_timer_attributes(self):
        """Test Timer attributes are set correctly."""
        with Timer() as t:
            pass

        assert t.start_time is not None
        assert t.end_time is not None
        assert t.end_time > t.start_time


class TestLoggingFunctions:
    """Tests for convenience logging functions."""

    def test_log_query(self):
        """Test log_query doesn't raise."""
        # Should not raise any exceptions
        log_query(
            session_id="test",
            query="USB-C cable",
            intent="new_search",
            confidence=0.95,
        )

    def test_log_intent(self):
        """Test log_intent doesn't raise."""
        log_intent(
            session_id="test",
            query="USB-C cable",
            intent="new_search",
            confidence=0.95,
            reasoning="Product search detected",
            classification_time_ms=1.5,
        )

    def test_log_filters(self):
        """Test log_filters doesn't raise."""
        log_filters(
            session_id="test",
            query="6ft USB-C cable",
            filters={
                "connector_from": "USB-C",
                "length": 6.0,
            },
            extraction_time_ms=0.5,
        )

    def test_log_search(self):
        """Test log_search doesn't raise."""
        log_search(
            session_id="test",
            filters={"connector_from": "USB-C"},
            products_found=25,
            tier="tier1",
            search_time_ms=150.0,
            dropped_filters=None,
        )

    def test_log_llm_call(self):
        """Test log_llm_call doesn't raise."""
        log_llm_call(
            session_id="test",
            model="gpt-4o",
            endpoint="chat/completions",
            latency_ms=500.0,
            tokens_used=150,
            success=True,
        )

    def test_log_llm_call_error(self):
        """Test log_llm_call with error doesn't raise."""
        log_llm_call(
            session_id="test",
            model="gpt-4o",
            endpoint="chat/completions",
            latency_ms=1000.0,
            success=False,
            error="Rate limit exceeded",
        )

    def test_log_response(self):
        """Test log_response doesn't raise."""
        log_response(
            session_id="test",
            intent="new_search",
            products_found=5,
            response_time_ms=450.0,
        )

    def test_log_error(self):
        """Test log_error doesn't raise."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            log_error(
                session_id="test",
                error=e,
                context="Testing error logging",
            )

    def test_log_guidance(self):
        """Test log_guidance doesn't raise."""
        log_guidance(
            session_id="test",
            setup_type="multi_monitor",
            phase="initial_questions",
            monitor_count=3,
        )


class TestTimedDecorator:
    """Tests for @timed decorator."""

    def test_timed_decorator(self):
        """Test that @timed decorator works."""
        @timed("test_operation")
        def slow_function():
            return sum(range(1000))

        result = slow_function()
        assert result == sum(range(1000))

    def test_timed_decorator_with_exception(self):
        """Test that @timed decorator handles exceptions."""
        @timed("failing_operation")
        def failing_function():
            raise ValueError("Intentional failure")

        with pytest.raises(ValueError):
            failing_function()


class TestGetLogger:
    """Tests for logger retrieval."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logging.Logger."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_namespace(self):
        """Test that logger name is properly namespaced."""
        logger = get_logger("my_module")
        assert "stbot" in logger.name

    def test_get_logger_caches_loggers(self):
        """Test that same logger is returned for same name."""
        logger1 = get_logger("cached_module")
        logger2 = get_logger("cached_module")
        assert logger1 is logger2
