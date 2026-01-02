"""
Tests for the API retry infrastructure.

Verifies:
- Error classification
- Exponential backoff calculation
- Retry decision logic
- Retry handler execution
- Decorator functionality
- Graceful degradation
"""

import pytest
import time
from unittest.mock import Mock, patch

from core.api_retry import (
    RetryableErrorType,
    classify_error,
    is_retryable,
    RetryConfig,
    RetryHandler,
    RetryResult,
    with_retry,
    retry_api_call,
    with_graceful_degradation,
    DEFAULT_OPENAI_RETRY,
    DEFAULT_PINECONE_RETRY,
)


class TestErrorClassification:
    """Tests for error classification."""

    def test_rate_limit_error(self):
        """Rate limit errors should be classified correctly."""
        error = Exception("RateLimitError: 429 Too Many Requests")
        assert classify_error(error) == RetryableErrorType.RATE_LIMIT

    def test_timeout_error(self):
        """Timeout errors should be classified correctly."""
        error = TimeoutError("Connection timed out")
        assert classify_error(error) == RetryableErrorType.TIMEOUT

    def test_connection_error(self):
        """Connection errors should be classified correctly."""
        error = ConnectionError("Failed to connect")
        assert classify_error(error) == RetryableErrorType.CONNECTION_ERROR

    def test_auth_error(self):
        """Auth errors should be classified correctly."""
        error = Exception("401 Unauthorized")
        assert classify_error(error) == RetryableErrorType.AUTH_ERROR

    def test_unknown_error(self):
        """Unknown errors should be classified as unknown."""
        error = Exception("Something weird happened")
        assert classify_error(error) == RetryableErrorType.UNKNOWN

    def test_is_retryable_rate_limit(self):
        """Rate limit errors should be retryable."""
        error = Exception("rate limit exceeded")
        assert is_retryable(error) is True

    def test_is_retryable_timeout(self):
        """Timeout errors should be retryable."""
        error = TimeoutError("timeout")
        assert is_retryable(error) is True

    def test_is_retryable_auth_error(self):
        """Auth errors should not be retryable."""
        error = Exception("authentication failed")
        assert is_retryable(error) is False

    def test_is_retryable_unknown(self):
        """Unknown errors should not be retryable by default."""
        error = Exception("mystery error")
        assert is_retryable(error) is False


class TestRetryConfig:
    """Tests for retry configuration."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_custom_config(self):
        """Custom config values should be respected."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
            jitter=False,
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0
        assert config.jitter is False

    def test_openai_default_config(self):
        """OpenAI default config should be configured."""
        assert DEFAULT_OPENAI_RETRY.max_attempts == 3
        assert DEFAULT_OPENAI_RETRY.base_delay == 1.0

    def test_pinecone_default_config(self):
        """Pinecone default config should be configured."""
        assert DEFAULT_PINECONE_RETRY.max_attempts == 3
        assert DEFAULT_PINECONE_RETRY.base_delay == 0.5


class TestRetryHandler:
    """Tests for RetryHandler."""

    def test_calculate_delay_exponential(self):
        """Delay should increase exponentially."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        handler = RetryHandler(config=config)

        assert handler.calculate_delay(0) == 1.0   # 1 * 2^0 = 1
        assert handler.calculate_delay(1) == 2.0   # 1 * 2^1 = 2
        assert handler.calculate_delay(2) == 4.0   # 1 * 2^2 = 4

    def test_calculate_delay_capped(self):
        """Delay should be capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)
        handler = RetryHandler(config=config)

        # 1 * 2^10 = 1024, but should be capped at 5
        assert handler.calculate_delay(10) == 5.0

    def test_calculate_delay_with_jitter(self):
        """Delay with jitter should vary."""
        config = RetryConfig(base_delay=1.0, jitter=True, jitter_range=0.5)
        handler = RetryHandler(config=config)

        # Run multiple times - should get different values
        delays = [handler.calculate_delay(0) for _ in range(10)]

        # Not all delays should be the same
        assert len(set(delays)) > 1

        # All delays should be within reasonable range (0.5 to 1.5 for base 1.0)
        for delay in delays:
            assert 0.1 <= delay <= 2.0

    def test_should_retry_within_attempts(self):
        """Should retry if within attempt limit and error is retryable."""
        handler = RetryHandler(config=RetryConfig(max_attempts=3))

        error = TimeoutError("timeout")
        assert handler.should_retry(error, attempt=0) is True
        assert handler.should_retry(error, attempt=1) is True
        assert handler.should_retry(error, attempt=2) is False  # At limit

    def test_should_not_retry_non_retryable(self):
        """Should not retry non-retryable errors."""
        handler = RetryHandler(config=RetryConfig(max_attempts=3))

        error = Exception("authentication failed")
        assert handler.should_retry(error, attempt=0) is False

    def test_should_retry_specific_exceptions(self):
        """Should retry only specified exception types."""
        config = RetryConfig(max_attempts=3, retry_on=[ValueError])
        handler = RetryHandler(config=config)

        assert handler.should_retry(ValueError("test"), attempt=0) is True
        assert handler.should_retry(TypeError("test"), attempt=0) is False

    def test_execute_success_first_try(self):
        """Successful execution on first try."""
        handler = RetryHandler()
        mock_func = Mock(return_value="success")

        result = handler.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_execute_success_after_retry(self):
        """Successful execution after retries."""
        config = RetryConfig(base_delay=0.01, jitter=False)  # Fast retries for test
        handler = RetryHandler(config=config)

        # Fail twice, then succeed
        mock_func = Mock(side_effect=[
            TimeoutError("timeout"),
            TimeoutError("timeout"),
            "success"
        ])

        result = handler.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_execute_failure_with_fallback(self):
        """Return fallback on failure."""
        config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)
        handler = RetryHandler(config=config)

        mock_func = Mock(side_effect=TimeoutError("timeout"))

        result = handler.execute(mock_func, fallback="fallback_value")

        assert result == "fallback_value"
        assert mock_func.call_count == 2

    def test_execute_failure_returns_result(self):
        """Return RetryResult on failure when no fallback."""
        config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)
        handler = RetryHandler(config=config)

        mock_func = Mock(side_effect=TimeoutError("timeout"))

        result = handler.execute(mock_func)

        assert isinstance(result, RetryResult)
        assert result.success is False
        assert result.attempts == 2
        assert len(result.errors) == 2
        assert result.final_error is not None

    def test_execute_raise_on_failure(self):
        """Raise exception on failure when requested."""
        config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)
        handler = RetryHandler(config=config)

        mock_func = Mock(side_effect=TimeoutError("timeout"))

        with pytest.raises(TimeoutError):
            handler.execute(mock_func, raise_on_failure=True)

    def test_execute_stops_on_non_retryable(self):
        """Stop retrying on non-retryable errors."""
        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)
        handler = RetryHandler(config=config)

        # Auth error is not retryable
        mock_func = Mock(side_effect=Exception("401 Unauthorized"))

        result = handler.execute(mock_func)

        assert isinstance(result, RetryResult)
        assert result.attempts == 1  # Should stop after first attempt


class TestWithRetryDecorator:
    """Tests for @with_retry decorator."""

    def test_decorator_success(self):
        """Decorator should pass through successful calls."""
        @with_retry(max_attempts=3)
        def successful_func():
            return "success"

        assert successful_func() == "success"

    def test_decorator_retry_and_succeed(self):
        """Decorator should retry and eventually succeed."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "success"

        assert eventually_succeeds() == "success"
        assert call_count == 3

    def test_decorator_with_fallback(self):
        """Decorator should return fallback on failure."""
        @with_retry(max_attempts=2, base_delay=0.01, fallback="default", raise_on_failure=False)
        def always_fails():
            raise TimeoutError("timeout")

        assert always_fails() == "default"

    def test_decorator_raises_on_failure(self):
        """Decorator should raise on failure when configured."""
        @with_retry(max_attempts=2, base_delay=0.01, raise_on_failure=True)
        def always_fails():
            raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            always_fails()


class TestRetryApiCall:
    """Tests for retry_api_call convenience function."""

    def test_retry_api_call_success(self):
        """Should return result on success."""
        result = retry_api_call(
            func=lambda: "success",
            operation_name="test_call",
        )
        assert result == "success"

    def test_retry_api_call_with_fallback(self):
        """Should return fallback on failure."""
        result = retry_api_call(
            func=lambda: (_ for _ in ()).throw(TimeoutError("timeout")),
            config=RetryConfig(max_attempts=2, base_delay=0.01, jitter=False),
            fallback="fallback",
            operation_name="test_call",
        )
        assert result == "fallback"


class TestGracefulDegradation:
    """Tests for with_graceful_degradation."""

    def test_primary_success(self):
        """Should return primary result when it succeeds."""
        result = with_graceful_degradation(
            func=lambda: "primary",
            fallback_func=lambda: "fallback",
            operation_name="test_degradation",
        )
        assert result == "primary"

    def test_fallback_on_failure(self):
        """Should call fallback when primary fails."""
        call_count = {"primary": 0, "fallback": 0}

        def failing_primary():
            call_count["primary"] += 1
            raise TimeoutError("timeout")

        def fallback():
            call_count["fallback"] += 1
            return "fallback_result"

        config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)

        result = with_graceful_degradation(
            func=failing_primary,
            fallback_func=fallback,
            config=config,
            operation_name="test_degradation",
        )

        assert result == "fallback_result"
        assert call_count["primary"] == 2  # Retried twice
        assert call_count["fallback"] == 1


class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_retry_result_success(self):
        """Successful result should have correct values."""
        result = RetryResult(
            success=True,
            value="data",
            attempts=1,
            total_delay=0.0,
        )
        assert result.success is True
        assert result.value == "data"
        assert result.final_error is None

    def test_retry_result_failure(self):
        """Failed result should have correct values."""
        error = TimeoutError("timeout")
        result = RetryResult(
            success=False,
            attempts=3,
            total_delay=3.5,
            errors=[error, error, error],
            final_error=error,
        )
        assert result.success is False
        assert result.attempts == 3
        assert result.final_error is error
        assert len(result.errors) == 3
