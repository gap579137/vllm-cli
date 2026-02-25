"""Tests for the base error classes in vLLM CLI errors package.

Covers VLLMCLIError serialization, context management, user messages,
and the ValidationError subclass.
"""

import pytest

from vllm_cli.errors.base import ValidationError, VLLMCLIError


class TestVLLMCLIError:
    """Tests for the base VLLMCLIError class."""

    def test_basic_creation(self):
        """Error can be created with just a message."""
        err = VLLMCLIError("something failed")
        assert err.message == "something failed"
        assert err.error_code == "UNKNOWN_ERROR"
        assert err.context == {}
        assert err.user_message is None

    def test_creation_with_all_fields(self):
        """Error stores all constructor arguments."""
        ctx = {"key": "val"}
        err = VLLMCLIError(
            "msg",
            error_code="MY_CODE",
            context=ctx,
            user_message="oops",
        )
        assert err.error_code == "MY_CODE"
        assert err.context == {"key": "val"}
        assert err.user_message == "oops"

    def test_is_exception(self):
        """VLLMCLIError inherits from Exception and is raisable."""
        with pytest.raises(VLLMCLIError):
            raise VLLMCLIError("boom")

    # --- Context management ---

    def test_add_and_get_context(self):
        """add_context / get_context round-trips."""
        err = VLLMCLIError("msg")
        err.add_context("port", 8080)
        assert err.get_context("port") == 8080

    def test_get_context_default(self):
        """get_context returns default when key is missing."""
        err = VLLMCLIError("msg")
        assert err.get_context("missing") is None
        assert err.get_context("missing", "fallback") == "fallback"

    def test_add_context_overwrites(self):
        """Adding the same key twice keeps the latest value."""
        err = VLLMCLIError("msg")
        err.add_context("x", 1)
        err.add_context("x", 2)
        assert err.get_context("x") == 2

    # --- User messages ---

    def test_user_message_explicit(self):
        """get_user_message returns explicit user_message when set."""
        err = VLLMCLIError("tech detail", user_message="friendly msg")
        assert err.get_user_message() == "friendly msg"

    def test_user_message_generated_from_code(self):
        """get_user_message falls back to code-based message."""
        err = VLLMCLIError("detail", error_code="VALIDATION_ERROR")
        msg = err.get_user_message()
        assert "validation" in msg.lower()

    def test_user_message_unknown_code(self):
        """Unknown error codes produce a generic user message."""
        err = VLLMCLIError("detail", error_code="CUSTOM_999")
        msg = err.get_user_message()
        assert "error" in msg.lower()

    def test_generate_user_message_known_codes(self):
        """Each known error code produces a distinct message."""
        known_codes = [
            "UNKNOWN_ERROR",
            "VALIDATION_ERROR",
            "FILE_NOT_FOUND",
            "PERMISSION_DENIED",
            "NETWORK_ERROR",
            "TIMEOUT_ERROR",
        ]
        messages = set()
        for code in known_codes:
            err = VLLMCLIError("msg", error_code=code)
            messages.add(err._generate_user_message())
        # All known codes should produce unique messages
        assert len(messages) == len(known_codes)

    # --- Serialization ---

    def test_to_dict_structure(self):
        """to_dict returns the expected keys."""
        err = VLLMCLIError("msg", error_code="TEST", context={"a": 1})
        d = err.to_dict()
        assert set(d.keys()) == {"error_code", "message", "user_message", "context"}
        assert d["error_code"] == "TEST"
        assert d["message"] == "msg"
        assert d["context"] == {"a": 1}

    def test_to_dict_includes_user_message(self):
        """to_dict picks up the user_message."""
        err = VLLMCLIError("tech", user_message="friendly")
        assert err.to_dict()["user_message"] == "friendly"

    # --- __str__ and __repr__ ---

    def test_str_contains_code_and_message(self):
        """str(err) includes '[CODE] message'."""
        err = VLLMCLIError("oops", error_code="E1")
        s = str(err)
        assert "[E1]" in s
        assert "oops" in s

    def test_str_with_context(self):
        """str(err) includes context key=value pairs."""
        err = VLLMCLIError("oops", error_code="E2")
        err.add_context("port", 8000)
        s = str(err)
        assert "port=8000" in s

    def test_repr(self):
        """repr() includes class name and fields."""
        err = VLLMCLIError("msg", error_code="E3")
        r = repr(err)
        assert "VLLMCLIError" in r
        assert "msg" in r
        assert "E3" in r


class TestValidationError:
    """Tests for the ValidationError subclass."""

    def test_inherits_from_vllm_cli_error(self):
        """ValidationError is a VLLMCLIError."""
        err = ValidationError("bad value")
        assert isinstance(err, VLLMCLIError)

    def test_error_code_is_validation(self):
        """ValidationError always uses VALIDATION_ERROR code."""
        err = ValidationError("bad value")
        assert err.error_code == "VALIDATION_ERROR"

    def test_field_name_in_context(self):
        """field_name is stored in context."""
        err = ValidationError("bad", field_name="port")
        assert err.get_context("field_name") == "port"

    def test_field_value_in_context(self):
        """field_value is stored in context."""
        err = ValidationError("bad", field_value=99999)
        assert err.get_context("field_value") == 99999

    def test_user_message_includes_field_name(self):
        """User message mentions the field when provided."""
        err = ValidationError("out of range", field_name="port")
        msg = err.get_user_message()
        assert "port" in msg

    def test_user_message_without_field_name(self):
        """User message is generic when no field_name provided."""
        err = ValidationError("something wrong")
        msg = err.get_user_message()
        assert "validation" in msg.lower()

    def test_is_catchable_as_exception(self):
        """ValidationError can be caught as VLLMCLIError or Exception."""
        with pytest.raises(VLLMCLIError):
            raise ValidationError("bad")

        with pytest.raises(Exception):
            raise ValidationError("bad")
