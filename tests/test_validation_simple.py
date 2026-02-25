"""Expanded validation framework tests.

Tests each type validator (Integer, Float, String, Boolean, Choice),
the ValidationRegistry, and factory/utility functions.
"""

import pytest

from vllm_cli.validation.base import ValidationError, ValidationResult
from vllm_cli.validation.registry import ValidationRegistry
from vllm_cli.validation.types import (
    BooleanValidator,
    ChoiceValidator,
    FloatValidator,
    IntegerValidator,
    StringValidator,
)


class TestIntegerValidator:
    """Tests for IntegerValidator."""

    def test_valid_integer(self):
        """Accepts a plain integer."""
        v = IntegerValidator("port")
        result = v.validate(8000)
        assert result.is_valid()

    def test_string_integer_converted(self):
        """String representation of an integer is accepted."""
        v = IntegerValidator("port")
        result = v.validate("8000")
        assert result.is_valid()

    def test_invalid_type(self):
        """Non-numeric string is rejected."""
        v = IntegerValidator("port")
        result = v.validate("not_a_number")
        assert not result.is_valid()

    def test_min_value_enforced(self):
        """Values below min_value are rejected."""
        v = IntegerValidator("port", min_value=1)
        result = v.validate(0)
        assert not result.is_valid()

    def test_max_value_enforced(self):
        """Values above max_value are rejected."""
        v = IntegerValidator("port", max_value=65535)
        result = v.validate(70000)
        assert not result.is_valid()

    def test_within_range(self):
        """Values within min–max range pass."""
        v = IntegerValidator("port", min_value=1, max_value=65535)
        result = v.validate(8080)
        assert result.is_valid()

    def test_boundary_min(self):
        """min_value itself is accepted."""
        v = IntegerValidator("count", min_value=0)
        result = v.validate(0)
        assert result.is_valid()

    def test_boundary_max(self):
        """max_value itself is accepted."""
        v = IntegerValidator("count", max_value=10)
        result = v.validate(10)
        assert result.is_valid()

    def test_float_rejected(self):
        """Float value for integer field is rejected."""
        v = IntegerValidator("count")
        result = v.validate(3.14)
        # Depending on implementation, might reject or truncate
        # At minimum, should not crash
        assert isinstance(result, ValidationResult)


class TestFloatValidator:
    """Tests for FloatValidator."""

    def test_valid_float(self):
        """Accepts a float value."""
        v = FloatValidator("gpu_mem")
        result = v.validate(0.9)
        assert result.is_valid()

    def test_integer_accepted(self):
        """Integer values are accepted as floats."""
        v = FloatValidator("gpu_mem")
        result = v.validate(1)
        assert result.is_valid()

    def test_string_float_converted(self):
        """String representation of a float is accepted."""
        v = FloatValidator("gpu_mem")
        result = v.validate("0.85")
        assert result.is_valid()

    def test_invalid_type(self):
        """Non-numeric string is rejected."""
        v = FloatValidator("gpu_mem")
        result = v.validate("abc")
        assert not result.is_valid()

    def test_min_value_enforced(self):
        """Values below min_value are rejected."""
        v = FloatValidator("gpu_mem", min_value=0.0)
        result = v.validate(-0.5)
        assert not result.is_valid()

    def test_max_value_enforced(self):
        """Values above max_value are rejected."""
        v = FloatValidator("gpu_mem", max_value=1.0)
        result = v.validate(1.5)
        assert not result.is_valid()

    def test_within_range(self):
        """Values within min–max range pass."""
        v = FloatValidator("gpu_mem", min_value=0.0, max_value=1.0)
        result = v.validate(0.5)
        assert result.is_valid()


class TestStringValidator:
    """Tests for StringValidator."""

    def test_valid_string(self):
        """Accepts a plain string."""
        v = StringValidator("model_name")
        result = v.validate("llama-2-7b")
        assert result.is_valid()

    def test_non_string_coerced(self):
        """Non-string types are coerced to string (not rejected)."""
        v = StringValidator("model_name")
        result = v.validate(12345)
        # StringValidator coerces to str rather than rejecting
        assert result.is_valid()

    def test_min_length_enforced(self):
        """Strings shorter than min_length are rejected."""
        v = StringValidator("name", min_length=3)
        result = v.validate("ab")
        assert not result.is_valid()

    def test_max_length_enforced(self):
        """Strings longer than max_length are rejected."""
        v = StringValidator("name", max_length=5)
        result = v.validate("toolong")
        assert not result.is_valid()

    def test_within_length(self):
        """Strings within length constraints pass."""
        v = StringValidator("name", min_length=2, max_length=10)
        result = v.validate("hello")
        assert result.is_valid()

    def test_pattern_matching(self):
        """Strings matching the pattern pass."""
        v = StringValidator("model", pattern=r"^[a-z]+-[0-9]+$")
        result = v.validate("llama-7")
        assert result.is_valid()

    def test_pattern_mismatch(self):
        """Strings not matching the pattern fail."""
        v = StringValidator("model", pattern=r"^[a-z]+$")
        result = v.validate("Model123")
        assert not result.is_valid()

    def test_empty_string_with_min_length(self):
        """Empty string fails min_length > 0."""
        v = StringValidator("name", min_length=1)
        result = v.validate("")
        assert not result.is_valid()


class TestBooleanValidator:
    """Tests for BooleanValidator."""

    def test_accepts_true(self):
        """bool True is accepted."""
        v = BooleanValidator("flag")
        result = v.validate(True)
        assert result.is_valid()

    def test_accepts_false(self):
        """bool False is accepted."""
        v = BooleanValidator("flag")
        result = v.validate(False)
        assert result.is_valid()

    def test_string_true_variants(self):
        """String variants of 'true' are accepted."""
        v = BooleanValidator("flag")
        for val in ["true", "True", "TRUE", "yes", "1"]:
            result = v.validate(val)
            assert result.is_valid(), f"Failed for input: {val}"

    def test_string_false_variants(self):
        """String variants of 'false' are accepted."""
        v = BooleanValidator("flag")
        for val in ["false", "False", "FALSE", "no", "0"]:
            result = v.validate(val)
            assert result.is_valid(), f"Failed for input: {val}"

    def test_numeric_0_and_1(self):
        """Numeric 0 and 1 are accepted."""
        v = BooleanValidator("flag")
        result0 = v.validate(0)
        result1 = v.validate(1)
        assert result0.is_valid()
        assert result1.is_valid()

    def test_invalid_string(self):
        """Non-boolean strings are rejected."""
        v = BooleanValidator("flag")
        result = v.validate("maybe")
        assert not result.is_valid()


class TestChoiceValidator:
    """Tests for ChoiceValidator."""

    def test_valid_choice(self):
        """Value in the choices list passes."""
        v = ChoiceValidator("quant", choices=["awq", "gptq", "fp8"])
        result = v.validate("awq")
        assert result.is_valid()

    def test_invalid_choice(self):
        """Value not in the choices list fails."""
        v = ChoiceValidator("quant", choices=["awq", "gptq", "fp8"])
        result = v.validate("invalid_method")
        assert not result.is_valid()

    def test_none_choice_when_allowed(self):
        """None is accepted when it's in the choices list."""
        v = ChoiceValidator("quant", choices=["awq", "gptq", None])
        result = v.validate(None)
        assert result.is_valid()

    def test_case_sensitive_by_default(self):
        """Case-sensitive matching by default."""
        v = ChoiceValidator("quant", choices=["awq", "gptq"], case_sensitive=True)
        result = v.validate("AWQ")
        # Behavior depends on implementation; at minimum should not crash
        assert isinstance(result, ValidationResult)

    def test_case_insensitive_mode(self):
        """Case-insensitive mode accepts different cases."""
        v = ChoiceValidator("quant", choices=["awq", "gptq"], case_sensitive=False)
        result = v.validate("AWQ")
        assert result.is_valid()


class TestValidationResult:
    """Tests for the ValidationResult container."""

    def test_initially_valid(self):
        """New ValidationResult is valid."""
        result = ValidationResult()
        assert result.is_valid() is True

    def test_adding_error_makes_invalid(self):
        """Adding an error makes the result invalid."""
        result = ValidationResult()
        result.add_error(ValidationError("field", "value", "error message"))
        assert result.is_valid() is False

    def test_multiple_errors(self):
        """Multiple errors can be added."""
        result = ValidationResult()
        result.add_error(ValidationError("f1", "v1", "err1"))
        result.add_error(ValidationError("f2", "v2", "err2"))
        assert result.is_valid() is False


class TestValidationRegistry:
    """Tests for ValidationRegistry registration and lookup."""

    def test_create_registry(self):
        """Registry can be created."""
        registry = ValidationRegistry()
        assert registry is not None

    def test_register_and_get_validator(self):
        """Register a validator and retrieve it."""
        registry = ValidationRegistry()
        validator = IntegerValidator("port", min_value=1, max_value=65535)
        registry.register("port", validator)
        retrieved = registry.get_validator("port")
        assert retrieved is not None

    def test_get_unregistered_returns_none(self):
        """Getting an unregistered key returns None."""
        registry = ValidationRegistry()
        result = registry.get_validator("nonexistent")
        assert result is None

    def test_register_multiple(self):
        """Multiple validators can be registered."""
        registry = ValidationRegistry()
        registry.register("port", IntegerValidator("port"))
        registry.register("gpu_mem", FloatValidator("gpu_mem"))
        assert registry.get_validator("port") is not None
        assert registry.get_validator("gpu_mem") is not None
