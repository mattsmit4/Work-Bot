"""
Structured logging infrastructure for ST-Bot.

Provides production-grade JSON logging with:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Rotating file handlers (daily rotation, 30-day retention)
- Separate error log file
- Performance tracking (latency metrics)
- Context tracking (session_id, query, intent)

Usage:
    from core.structured_logging import get_logger, log_query, log_error

    logger = get_logger(__name__)
    logger.info("Starting search", extra={"query": "USB-C cable"})

    # Or use convenience functions:
    log_query(session_id="abc", query="USB-C cable", intent="new_search")
"""

import json
import logging
import sys
import time
import traceback
import uuid
from datetime import datetime
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# =============================================================================
# JSON Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.

    Output format:
    {
        "timestamp": "2024-12-08T10:30:00.123456Z",
        "level": "INFO",
        "logger": "core.intent",
        "message": "Intent classified",
        "event": "intent_classification",
        "session_id": "abc123",
        ...
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        # These are passed via logger.info("msg", extra={...})
        extra_fields = [
            "event", "session_id", "query", "intent", "confidence",
            "filters", "products_found", "response_time_ms", "tier",
            "dropped_filters", "error_type", "stack_trace", "user_id",
            "llm_model", "llm_tokens", "llm_latency_ms", "products_shown",
            "product_skus", "search_latency_ms", "filter_extraction_ms",
            "intent_classification_ms", "total_latency_ms", "category",
            "connector_from", "connector_to", "length", "features",
            "reasoning", "match_quality", "port_count", "setup_type",
            "guidance_phase", "api_endpoint", "status_code", "request_id",
        ]

        for field in extra_fields:
            if hasattr(record, field):
                value = getattr(record, field)
                if value is not None:
                    log_data[field] = value

        # Add exception info if present
        if record.exc_info:
            log_data["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            log_data["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.

    Output format:
    2024-12-08 10:30:00 | INFO | core.intent | Intent classified | session=abc123
    """

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # Color the level
        color = self.COLORS.get(level, "")
        reset = self.COLORS["RESET"]

        # Build base message
        msg = f"{timestamp} | {color}{level:8}{reset} | {record.name} | {record.getMessage()}"

        # Add key context fields
        context_parts = []
        for field in ["session_id", "event", "response_time_ms"]:
            if hasattr(record, field) and getattr(record, field) is not None:
                context_parts.append(f"{field}={getattr(record, field)}")

        if context_parts:
            msg += f" | {', '.join(context_parts)}"

        # Add exception info
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


# =============================================================================
# Logger Setup
# =============================================================================

_loggers: Dict[str, logging.Logger] = {}
_initialized = False


def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """
    Initialize the logging system.

    Creates:
    - logs/stbot.log (all logs, rotating daily, 30-day retention)
    - logs/errors.log (ERROR and above, rotating daily, 30-day retention)
    - Console output (if enabled)

    Args:
        log_dir: Directory for log files
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        enable_console: Whether to output to console
        enable_file: Whether to write to files
    """
    global _initialized
    if _initialized:
        return

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get root logger for our application
    root_logger = logging.getLogger("stbot")
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers filter

    # Remove any existing handlers
    root_logger.handlers = []

    # Console handler (human-readable)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ConsoleFormatter())
        root_logger.addHandler(console_handler)

    # File handler - all logs (JSON, rotating daily)
    if enable_file:
        main_log_file = log_path / "stbot.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(main_log_file),
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(JSONFormatter())
        file_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(file_handler)

        # Error log file (ERROR and above only)
        error_log_file = log_path / "errors.log"
        error_handler = TimedRotatingFileHandler(
            filename=str(error_log_file),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        error_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(error_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Search started", extra={"query": "USB-C cable"})
    """
    # Ensure logging is set up
    if not _initialized:
        setup_logging()

    # Create child logger under stbot namespace
    if name.startswith("stbot."):
        logger_name = name
    else:
        logger_name = f"stbot.{name}"

    if logger_name not in _loggers:
        _loggers[logger_name] = logging.getLogger(logger_name)

    return _loggers[logger_name]


# =============================================================================
# Context Manager for Session Tracking
# =============================================================================

class LogContext:
    """
    Context manager for tracking session-level logging context.

    Usage:
        with LogContext(session_id="abc123") as ctx:
            ctx.log_query("USB-C cable")
            # ... do work ...
            ctx.log_response(products_found=5, response_time_ms=450)
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize log context."""
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.start_time = None
        self.logger = get_logger("context")

    def __enter__(self) -> "LogContext":
        """Enter context."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context, log any errors."""
        if exc_type is not None:
            self.log_error(exc_val, exc_tb)
        return False  # Don't suppress exceptions

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000

    def log_query(self, query: str, **extra) -> None:
        """Log an incoming query."""
        self.logger.info(
            "User query received",
            extra={
                "event": "user_query",
                "session_id": self.session_id,
                "query": query,
                **extra
            }
        )

    def log_response(self, **extra) -> None:
        """Log the response being sent."""
        self.logger.info(
            "Response sent",
            extra={
                "event": "response_sent",
                "session_id": self.session_id,
                "total_latency_ms": round(self.elapsed_ms(), 2),
                **extra
            }
        )

    def log_error(self, error: Exception, tb=None) -> None:
        """Log an error."""
        self.logger.error(
            f"Error: {error}",
            extra={
                "event": "error",
                "session_id": self.session_id,
                "error_type": type(error).__name__,
                "stack_trace": "".join(traceback.format_tb(tb)) if tb else None,
            },
            exc_info=(type(error), error, tb) if tb else None
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def log_query(
    session_id: str,
    query: str,
    intent: Optional[str] = None,
    confidence: Optional[float] = None,
    **extra
) -> None:
    """
    Log a user query with context.

    Args:
        session_id: Session identifier
        query: User's query text
        intent: Classified intent type
        confidence: Intent classification confidence
        **extra: Additional fields to log
    """
    logger = get_logger("query")
    logger.info(
        "User query",
        extra={
            "event": "user_query",
            "session_id": session_id,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            **extra
        }
    )


def log_intent(
    session_id: str,
    query: str,
    intent: str,
    confidence: float,
    reasoning: str,
    classification_time_ms: float,
    **extra
) -> None:
    """
    Log intent classification result.

    Args:
        session_id: Session identifier
        query: Original query
        intent: Classified intent type
        confidence: Classification confidence
        reasoning: Why this intent was chosen
        classification_time_ms: Time taken to classify
        **extra: Additional fields
    """
    logger = get_logger("intent")
    logger.info(
        f"Intent classified: {intent}",
        extra={
            "event": "intent_classification",
            "session_id": session_id,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "intent_classification_ms": round(classification_time_ms, 2),
            **extra
        }
    )


def log_filters(
    session_id: str,
    query: str,
    filters: Dict[str, Any],
    extraction_time_ms: float,
    **extra
) -> None:
    """
    Log filter extraction result.

    Args:
        session_id: Session identifier
        query: Original query
        filters: Extracted filters
        extraction_time_ms: Time taken to extract
        **extra: Additional fields
    """
    logger = get_logger("filters")
    logger.info(
        "Filters extracted",
        extra={
            "event": "filter_extraction",
            "session_id": session_id,
            "query": query,
            "filters": filters,
            "filter_extraction_ms": round(extraction_time_ms, 2),
            **extra
        }
    )


def log_search(
    session_id: str,
    filters: Dict[str, Any],
    products_found: int,
    tier: str,
    search_time_ms: float,
    dropped_filters: Optional[list] = None,
    **extra
) -> None:
    """
    Log search result.

    Args:
        session_id: Session identifier
        filters: Filters used for search
        products_found: Number of products found
        tier: Search tier used (tier1, tier2, etc.)
        search_time_ms: Time taken to search
        dropped_filters: Any filters that were relaxed
        **extra: Additional fields
    """
    logger = get_logger("search")
    logger.info(
        f"Search complete: {products_found} products found (tier: {tier})",
        extra={
            "event": "search_complete",
            "session_id": session_id,
            "filters": filters,
            "products_found": products_found,
            "tier": tier,
            "search_latency_ms": round(search_time_ms, 2),
            "dropped_filters": dropped_filters,
            **extra
        }
    )


def log_llm_call(
    session_id: str,
    model: str,
    endpoint: str,
    latency_ms: float,
    tokens_used: Optional[int] = None,
    success: bool = True,
    error: Optional[str] = None,
    **extra
) -> None:
    """
    Log LLM API call.

    Args:
        session_id: Session identifier
        model: Model used (e.g., "gpt-4o")
        endpoint: API endpoint/function called
        latency_ms: Time taken for API call
        tokens_used: Tokens consumed (if available)
        success: Whether call succeeded
        error: Error message if failed
        **extra: Additional fields
    """
    logger = get_logger("llm")
    level = logging.INFO if success else logging.ERROR

    message = f"LLM call: {endpoint}" if success else f"LLM call failed: {endpoint}"

    logger.log(
        level,
        message,
        extra={
            "event": "llm_api_call",
            "session_id": session_id,
            "llm_model": model,
            "api_endpoint": endpoint,
            "llm_latency_ms": round(latency_ms, 2),
            "llm_tokens": tokens_used,
            "success": success,
            "error": error,
            **extra
        }
    )


def log_products_shown(
    session_id: str,
    products: list,
    query: str,
    **extra
) -> None:
    """
    Log products shown to user.

    Args:
        session_id: Session identifier
        products: List of products shown
        query: Original query
        **extra: Additional fields
    """
    logger = get_logger("products")

    product_skus = [p.product_number if hasattr(p, 'product_number') else str(p) for p in products[:10]]

    logger.info(
        f"Showing {len(products)} products",
        extra={
            "event": "products_shown",
            "session_id": session_id,
            "products_shown": len(products),
            "product_skus": product_skus,
            "query": query,
            **extra
        }
    )


def log_response(
    session_id: str,
    intent: str,
    products_found: int,
    response_time_ms: float,
    **extra
) -> None:
    """
    Log final response sent to user.

    Args:
        session_id: Session identifier
        intent: Intent type
        products_found: Number of products shown
        response_time_ms: Total response time
        **extra: Additional fields
    """
    logger = get_logger("response")
    logger.info(
        f"Response sent: {intent} with {products_found} products",
        extra={
            "event": "response_sent",
            "session_id": session_id,
            "intent": intent,
            "products_found": products_found,
            "response_time_ms": round(response_time_ms, 2),
            **extra
        }
    )


def log_error(
    session_id: str,
    error: Exception,
    context: Optional[str] = None,
    **extra
) -> None:
    """
    Log an error with full context.

    Args:
        session_id: Session identifier
        error: The exception
        context: Additional context about what was happening
        **extra: Additional fields
    """
    logger = get_logger("error")
    logger.error(
        f"Error: {type(error).__name__}: {error}",
        extra={
            "event": "error",
            "session_id": session_id,
            "error_type": type(error).__name__,
            "stack_trace": traceback.format_exc(),
            "context": context,
            **extra
        },
        exc_info=True
    )


def log_guidance(
    session_id: str,
    setup_type: str,
    phase: str,
    **extra
) -> None:
    """
    Log guidance flow progress.

    Args:
        session_id: Session identifier
        setup_type: Type of setup (multi_monitor, dock_selection, etc.)
        phase: Current guidance phase
        **extra: Additional fields
    """
    logger = get_logger("guidance")
    logger.info(
        f"Guidance: {setup_type} - {phase}",
        extra={
            "event": "guidance_progress",
            "session_id": session_id,
            "setup_type": setup_type,
            "guidance_phase": phase,
            **extra
        }
    )


# =============================================================================
# Performance Timing Decorator
# =============================================================================

def timed(event_name: str, logger_name: str = "performance"):
    """
    Decorator to time function execution and log it.

    Usage:
        @timed("intent_classification")
        def classify(query: str) -> Intent:
            ...

    Args:
        event_name: Name of the event for logging
        logger_name: Logger to use
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000

                logger = get_logger(logger_name)
                logger.debug(
                    f"{event_name} completed",
                    extra={
                        "event": f"{event_name}_timing",
                        "elapsed_ms": round(elapsed_ms, 2),
                        "function": func.__name__,
                    }
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger = get_logger(logger_name)
                logger.error(
                    f"{event_name} failed after {elapsed_ms:.2f}ms",
                    extra={
                        "event": f"{event_name}_error",
                        "elapsed_ms": round(elapsed_ms, 2),
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer() as t:
            # ... do work ...
        print(f"Took {t.elapsed_ms}ms")
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


# =============================================================================
# Initialize on import (with defaults)
# =============================================================================

# Don't auto-initialize - let the app control this
# setup_logging()
