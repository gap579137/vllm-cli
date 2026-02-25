"""Tests for retry mechanisms and backoff strategies.

Covers RetryConfig, retry_with_backoff decorator, RetryableOperation
context manager, and the exponential_backoff utility.
"""

import time

import pytest

from vllm_cli.errors.base import VLLMCLIError
from vllm_cli.errors.retry import (
    RetryableOperation,
    RetryConfig,
    exponential_backoff,
    retry_with_backoff,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass-like configuration."""

    def test_default_values(self):
        """Default config has sensible retry settings."""
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.backoff_multiplier == 2.0
        assert isinstance(cfg.retriable_exceptions, list)
        assert len(cfg.retriable_exceptions) > 0

    def test_custom_values(self):
        """Custom values override defaults."""
        cfg = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            backoff_multiplier=3.0,
            retriable_exceptions=[ValueError],
        )
        assert cfg.max_attempts == 5
        assert cfg.base_delay == 0.5
        assert cfg.max_delay == 30.0
        assert cfg.backoff_multiplier == 3.0
        assert cfg.retriable_exceptions == [ValueError]

    def test_default_retriable_exceptions_include_common_types(self):
        """Default retriable exceptions include ConnectionError and TimeoutError."""
        cfg = RetryConfig()
        exception_types = cfg.retriable_exceptions
        assert ConnectionError in exception_types
        assert TimeoutError in exception_types


class TestExponentialBackoff:
    """Tests for the exponential_backoff utility function."""

    def test_first_attempt_uses_base_delay(self):
        """Attempt 0 returns the base delay."""
        delay = exponential_backoff(0, base_delay=1.0)
        assert delay == 1.0

    def test_increases_exponentially(self):
        """Each subsequent attempt doubles the delay (with default multiplier)."""
        d0 = exponential_backoff(0, base_delay=1.0, multiplier=2.0)
        d1 = exponential_backoff(1, base_delay=1.0, multiplier=2.0)
        d2 = exponential_backoff(2, base_delay=1.0, multiplier=2.0)
        assert d0 == 1.0
        assert d1 == 2.0
        assert d2 == 4.0

    def test_capped_at_max_delay(self):
        """Delay never exceeds max_delay."""
        delay = exponential_backoff(100, base_delay=1.0, max_delay=10.0)
        assert delay == 10.0

    def test_custom_multiplier(self):
        """Custom multiplier is applied correctly."""
        delay = exponential_backoff(2, base_delay=1.0, multiplier=3.0, max_delay=100.0)
        assert delay == 9.0  # 1.0 * 3^2


class TestRetryWithBackoffDecorator:
    """Tests for the retry_with_backoff decorator."""

    def test_succeeds_on_first_try(self):
        """Function that succeeds immediately is called once."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3, retriable_exceptions=[ValueError]))
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_retriable_exception(self):
        """Function is retried on retriable exceptions."""
        call_count = 0

        @retry_with_backoff(
            RetryConfig(
                max_attempts=3,
                base_delay=0.01,  # Tiny delay for fast tests
                retriable_exceptions=[ConnectionError],
            )
        )
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = fail_then_succeed()
        assert result == "recovered"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        """After exhausting retries, a VLLMCLIError(RETRY_EXHAUSTED) is raised."""
        call_count = 0

        @retry_with_backoff(
            RetryConfig(
                max_attempts=2,
                base_delay=0.01,
                retriable_exceptions=[ConnectionError],
            )
        )
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("permanent")

        with pytest.raises(VLLMCLIError, match="RETRY_EXHAUSTED"):
            always_fail()
        assert call_count == 2

    def test_non_retriable_exception_not_retried(self):
        """Non-retriable exceptions are wrapped but not retried."""
        call_count = 0

        @retry_with_backoff(
            RetryConfig(
                max_attempts=5,
                base_delay=0.01,
                retriable_exceptions=[ConnectionError],
            )
        )
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retriable")

        # The decorator wraps all failures in VLLMCLIError after exhausting attempts
        with pytest.raises(VLLMCLIError):
            raise_value_error()


class TestRetryableOperation:
    """Tests for the RetryableOperation context manager."""

    def test_initialization(self):
        """RetryableOperation initializes with defaults."""
        op = RetryableOperation("test_op")
        assert op.config is not None
        assert op.config.max_attempts == 3

    def test_custom_config(self):
        """RetryableOperation accepts custom RetryConfig."""
        cfg = RetryConfig(max_attempts=5)
        op = RetryableOperation("test_op", config=cfg)
        assert op.config.max_attempts == 5

    def test_should_retry_retriable(self):
        """should_retry returns True for retriable exceptions within limit."""
        op = RetryableOperation("test_op")
        op.attempt = 0
        assert op.should_retry(ConnectionError("fail")) is True

    def test_should_retry_non_retriable(self):
        """should_retry returns False for non-retriable exceptions."""
        cfg = RetryConfig(retriable_exceptions=[ConnectionError])
        op = RetryableOperation("test_op", config=cfg)
        op.attempt = 0
        assert op.should_retry(ValueError("nope")) is False

    def test_should_retry_exhausted(self):
        """should_retry returns False after max attempts."""
        cfg = RetryConfig(max_attempts=2, retriable_exceptions=[ConnectionError])
        op = RetryableOperation("test_op", config=cfg)
        op.attempt = 2  # Already at max
        assert op.should_retry(ConnectionError("fail")) is False

    def test_enter_returns_self(self):
        """Context manager __enter__ returns self."""
        op = RetryableOperation("test_op")
        with op as ctx:
            assert ctx is op
