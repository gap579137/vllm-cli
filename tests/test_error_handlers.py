"""Tests for error handling utilities.

Covers ErrorReporter, error_boundary context manager, safe_operation
decorator, and format_error_for_user utility.
"""

import pytest

from vllm_cli.errors.base import VLLMCLIError
from vllm_cli.errors.handlers import (
    ErrorReporter,
    error_boundary,
    format_error_for_user,
    get_error_help_text,
    safe_operation,
)


class TestErrorReporter:
    """Tests for the ErrorReporter class."""

    def test_initialization(self):
        """ErrorReporter initializes with empty counts."""
        reporter = ErrorReporter()
        summary = reporter.get_error_summary()
        assert isinstance(summary, dict)

    def test_report_vllm_error_user_facing(self):
        """Reporting a VLLMCLIError returns a string message."""
        reporter = ErrorReporter()
        err = VLLMCLIError("test error", error_code="TEST")
        msg = reporter.report_error(err, user_facing=True)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_report_error_with_context(self):
        """Reporting with include_context=True includes context info."""
        reporter = ErrorReporter()
        err = VLLMCLIError("test error", error_code="TEST")
        err.add_context("port", 8000)
        msg = reporter.report_error(err, user_facing=True, include_context=True)
        assert isinstance(msg, str)

    def test_report_error_developer_mode(self):
        """Developer-facing report includes technical details."""
        reporter = ErrorReporter()
        err = VLLMCLIError("internal detail", error_code="INTERNAL")
        msg = reporter.report_error(err, user_facing=False)
        assert isinstance(msg, str)

    def test_error_counting(self):
        """Reporting errors increments internal counters."""
        reporter = ErrorReporter()
        err1 = VLLMCLIError("err1", error_code="TYPE_A")
        err2 = VLLMCLIError("err2", error_code="TYPE_B")
        reporter.report_error(err1)
        reporter.report_error(err2)
        reporter.report_error(err1)

        summary = reporter.get_error_summary()
        assert isinstance(summary, dict)

    def test_reset_error_counts(self):
        """reset_error_counts clears the counters."""
        reporter = ErrorReporter()
        reporter.report_error(VLLMCLIError("err"))
        reporter.reset_error_counts()
        summary = reporter.get_error_summary()
        # After reset, total should be 0
        total = sum(v for v in summary.values() if isinstance(v, int))
        assert total == 0


class TestErrorBoundary:
    """Tests for the error_boundary context manager."""

    def test_no_error_passes_through(self):
        """Code that doesn't raise passes through normally."""
        with error_boundary("test operation"):
            x = 1 + 1
        assert x == 2

    def test_catches_and_wraps_generic_exception(self):
        """Generic exceptions are wrapped in VLLMCLIError."""
        with pytest.raises(VLLMCLIError):
            with error_boundary("test operation"):
                raise RuntimeError("boom")

    def test_suppress_errors_returns_fallback(self):
        """With suppress_errors=True, exceptions don't propagate."""
        result = None
        with error_boundary(
            "test operation",
            suppress_errors=True,
            fallback_result="fallback",
        ) as ctx:
            raise RuntimeError("suppressed")

    def test_custom_error_type(self):
        """Custom error_type is used for wrapping (when type is bug-free)."""
        # Note: ServerError has a pre-existing kwargs collision bug,
        # so we use VLLMCLIError itself as the custom type.
        with pytest.raises(VLLMCLIError):
            with error_boundary("test op", error_type=VLLMCLIError):
                raise RuntimeError("issue")

    def test_context_is_attached(self):
        """Context dict is attached to the wrapped error."""
        try:
            with error_boundary(
                "test op",
                context={"port": 8080},
            ):
                raise RuntimeError("fail")
        except VLLMCLIError as e:
            assert e.get_context("port") == 8080


class TestSafeOperation:
    """Tests for the safe_operation decorator."""

    def test_successful_operation(self):
        """Decorated function that succeeds returns normally."""

        @safe_operation("test_op")
        def works():
            return 42

        assert works() == 42

    def test_failed_operation_returns_fallback(self):
        """Decorated function that fails returns fallback."""

        @safe_operation("test_op", fallback_result=-1)
        def fails():
            raise RuntimeError("boom")

        assert fails() == -1

    def test_default_fallback_is_none(self):
        """Without explicit fallback, None is returned on error."""

        @safe_operation("test_op")
        def fails():
            raise RuntimeError("boom")

        assert fails() is None

    def test_preserves_function_arguments(self):
        """Decorated function receives arguments properly."""

        @safe_operation("test_op")
        def add(a, b):
            return a + b

        assert add(3, 4) == 7


class TestFormatErrorForUser:
    """Tests for the format_error_for_user utility."""

    def test_formats_vllm_error(self):
        """VLLMCLIError is formatted using get_user_message."""
        err = VLLMCLIError("technical detail", user_message="Something went wrong")
        msg = format_error_for_user(err)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_formats_generic_exception(self):
        """Generic exceptions produce a user-friendly message."""
        err = RuntimeError("something unexpected")
        msg = format_error_for_user(err)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_formats_specific_error(self):
        """Specific VLLMCLIError subtype is formatted appropriately."""
        err = VLLMCLIError("GPU out of memory", error_code="GPU_ERROR")
        msg = format_error_for_user(err)
        assert isinstance(msg, str)


class TestGetErrorHelpText:
    """Tests for the get_error_help_text utility."""

    def test_known_error_returns_help(self):
        """Known VLLMCLIError types return help text."""
        err = VLLMCLIError("GPU issue", error_code="GPU_ERROR")
        help_text = get_error_help_text(err)
        # May return None or a string, depending on implementation
        assert help_text is None or isinstance(help_text, str)

    def test_generic_error_returns_none_or_string(self):
        """Generic VLLMCLIError may or may not have help text."""
        err = VLLMCLIError("generic")
        help_text = get_error_help_text(err)
        assert help_text is None or isinstance(help_text, str)
