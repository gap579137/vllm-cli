"""Tests for the Pydantic-based validation models (Phase 5).

Validates that VLLMConfig and PydanticValidationRegistry behave
identically to the old hand-rolled framework for all existing test
scenarios, plus new Pydantic-specific tests.
"""

import pytest

from vllm_cli.validation.pydantic_models import (
    PydanticValidationRegistry,
    VLLMConfig,
    create_pydantic_validation_registry,
)
from vllm_cli.validation.base import ValidationError, ValidationResult
from vllm_cli.validation.schema import create_vllm_validation_registry


# ---------------------------------------------------------------------------
#  VLLMConfig model tests
# ---------------------------------------------------------------------------

class TestVLLMConfigBasic:
    """Basic VLLMConfig construction and defaults."""

    def test_empty_config(self):
        cfg = VLLMConfig()
        assert cfg.port is None
        assert cfg.host is None

    def test_valid_full_config(self):
        cfg = VLLMConfig(
            host="0.0.0.0",
            port=8000,
            dtype="float16",
            max_model_len=4096,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            quantization="awq",
            enforce_eager=True,
            trust_remote_code=False,
        )
        assert cfg.port == 8000
        assert cfg.dtype == "float16"
        assert cfg.gpu_memory_utilization == 0.9

    def test_extra_fields_allowed(self):
        """Unknown keys pass through without errors."""
        cfg = VLLMConfig(future_vllm_flag="some_value", port=8080)
        assert cfg.port == 8080


class TestVLLMConfigPort:
    """Port validation (1–65535)."""

    def test_valid_port(self):
        cfg = VLLMConfig(port=8000)
        assert cfg.port == 8000

    def test_port_boundary_low(self):
        cfg = VLLMConfig(port=1)
        assert cfg.port == 1

    def test_port_boundary_high(self):
        cfg = VLLMConfig(port=65535)
        assert cfg.port == 65535

    def test_port_zero_rejected(self):
        with pytest.raises(Exception):
            VLLMConfig(port=0)

    def test_port_too_high_rejected(self):
        with pytest.raises(Exception):
            VLLMConfig(port=70000)

    def test_port_string_digit_coerced(self):
        cfg = VLLMConfig(port="8080")
        assert cfg.port == 8080

    def test_port_bool_rejected(self):
        with pytest.raises(Exception):
            VLLMConfig(port=True)


class TestVLLMConfigDtype:
    """Dtype choice validation."""

    @pytest.mark.parametrize("val", ["auto", "float16", "bfloat16", "float32"])
    def test_valid_dtype(self, val):
        cfg = VLLMConfig(dtype=val)
        assert cfg.dtype == val

    def test_invalid_dtype(self):
        with pytest.raises(Exception):
            VLLMConfig(dtype="int8")


class TestVLLMConfigQuantization:
    """Quantization choice validation."""

    @pytest.mark.parametrize("val", [
        "awq", "awq_marlin", "gptq", "gptq_marlin",
        "bitsandbytes", "fp8", "gguf", "compressed-tensors",
    ])
    def test_valid_quant(self, val):
        cfg = VLLMConfig(quantization=val)
        assert cfg.quantization == val

    def test_invalid_quant(self):
        with pytest.raises(Exception):
            VLLMConfig(quantization="invalid_method")


class TestVLLMConfigIntegers:
    """Positive and non-negative integer fields."""

    def test_positive_int_valid(self):
        cfg = VLLMConfig(tensor_parallel_size=4)
        assert cfg.tensor_parallel_size == 4

    def test_positive_int_zero_rejected(self):
        with pytest.raises(Exception):
            VLLMConfig(tensor_parallel_size=0)

    def test_positive_int_negative_rejected(self):
        with pytest.raises(Exception):
            VLLMConfig(max_model_len=-1)

    def test_non_neg_int_zero_accepted(self):
        cfg = VLLMConfig(max_paddings=0)
        assert cfg.max_paddings == 0

    def test_non_neg_int_negative_rejected(self):
        with pytest.raises(Exception):
            VLLMConfig(seed=-1)

    def test_bool_rejected_as_int(self):
        with pytest.raises(Exception):
            VLLMConfig(max_num_seqs=True)


class TestVLLMConfigFloats:
    """Float field validation."""

    def test_gpu_mem_valid(self):
        cfg = VLLMConfig(gpu_memory_utilization=0.85)
        assert cfg.gpu_memory_utilization == 0.85

    def test_gpu_mem_boundary_low(self):
        cfg = VLLMConfig(gpu_memory_utilization=0.0)
        assert cfg.gpu_memory_utilization == 0.0

    def test_gpu_mem_boundary_high(self):
        cfg = VLLMConfig(gpu_memory_utilization=1.0)
        assert cfg.gpu_memory_utilization == 1.0

    def test_gpu_mem_too_high(self):
        with pytest.raises(Exception):
            VLLMConfig(gpu_memory_utilization=1.5)

    def test_gpu_mem_negative(self):
        with pytest.raises(Exception):
            VLLMConfig(gpu_memory_utilization=-0.1)

    def test_gpu_mem_int_accepted(self):
        cfg = VLLMConfig(gpu_memory_utilization=1)
        assert cfg.gpu_memory_utilization == 1.0

    def test_cpu_offload_valid(self):
        cfg = VLLMConfig(cpu_offload_gb=4.0)
        assert cfg.cpu_offload_gb == 4.0

    def test_cpu_offload_zero(self):
        cfg = VLLMConfig(cpu_offload_gb=0.0)
        assert cfg.cpu_offload_gb == 0.0

    def test_cpu_offload_negative(self):
        with pytest.raises(Exception):
            VLLMConfig(cpu_offload_gb=-1.0)


class TestVLLMConfigBooleans:
    """Boolean field validation with flexible coercion."""

    def test_bool_true(self):
        cfg = VLLMConfig(enforce_eager=True)
        assert cfg.enforce_eager is True

    def test_bool_false(self):
        cfg = VLLMConfig(enforce_eager=False)
        assert cfg.enforce_eager is False

    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "yes", "1", "on", "enabled"])
    def test_truthy_strings(self, val):
        cfg = VLLMConfig(trust_remote_code=val)
        assert cfg.trust_remote_code is True

    @pytest.mark.parametrize("val", ["false", "False", "FALSE", "no", "0", "off", "disabled"])
    def test_falsy_strings(self, val):
        cfg = VLLMConfig(trust_remote_code=val)
        assert cfg.trust_remote_code is False

    def test_numeric_0_and_1(self):
        cfg0 = VLLMConfig(enforce_eager=0)
        cfg1 = VLLMConfig(enforce_eager=1)
        assert cfg0.enforce_eager is False
        assert cfg1.enforce_eager is True

    def test_invalid_bool_string(self):
        with pytest.raises(Exception):
            VLLMConfig(enforce_eager="maybe")


class TestVLLMConfigStrings:
    """String field validation."""

    def test_valid_host(self):
        cfg = VLLMConfig(host="0.0.0.0")
        assert cfg.host == "0.0.0.0"

    def test_empty_host_rejected(self):
        with pytest.raises(Exception):
            VLLMConfig(host="")


class TestVLLMConfigBlockSize:
    """Block size range (1–256)."""

    def test_valid(self):
        cfg = VLLMConfig(block_size=16)
        assert cfg.block_size == 16

    def test_boundary_low(self):
        cfg = VLLMConfig(block_size=1)
        assert cfg.block_size == 1

    def test_boundary_high(self):
        cfg = VLLMConfig(block_size=256)
        assert cfg.block_size == 256

    def test_too_high(self):
        with pytest.raises(Exception):
            VLLMConfig(block_size=257)

    def test_zero(self):
        with pytest.raises(Exception):
            VLLMConfig(block_size=0)


class TestVLLMConfigKvCacheDtype:
    """KV cache dtype choices."""

    @pytest.mark.parametrize("val", ["auto", "fp8", "fp16", "bf16"])
    def test_valid(self, val):
        cfg = VLLMConfig(kv_cache_dtype=val)
        assert cfg.kv_cache_dtype == val

    def test_invalid(self):
        with pytest.raises(Exception):
            VLLMConfig(kv_cache_dtype="int8")


class TestVLLMConfigLoadFormat:
    """Load format choices."""

    @pytest.mark.parametrize("val", ["auto", "pt", "safetensors", "npcache", "dummy"])
    def test_valid(self, val):
        cfg = VLLMConfig(load_format=val)
        assert cfg.load_format == val

    def test_invalid(self):
        with pytest.raises(Exception):
            VLLMConfig(load_format="invalid")


# ---------------------------------------------------------------------------
#  PydanticValidationRegistry adapter tests
# ---------------------------------------------------------------------------

class TestPydanticValidationRegistry:
    """Tests for the backward-compatible registry adapter."""

    @pytest.fixture
    def registry(self):
        return create_pydantic_validation_registry()

    def test_factory_returns_pydantic_registry(self):
        reg = create_vllm_validation_registry()
        assert isinstance(reg, PydanticValidationRegistry)

    def test_validate_config_valid(self, registry):
        result = registry.validate_config({"port": 8000, "dtype": "float16"})
        assert result.is_valid()
        assert len(result.errors) == 0

    def test_validate_config_invalid(self, registry):
        result = registry.validate_config({"port": 99999})
        assert not result.is_valid()
        assert len(result.errors) > 0

    def test_validate_config_multiple_errors(self, registry):
        result = registry.validate_config({
            "port": 99999,
            "gpu_memory_utilization": 5.0,
        })
        assert not result.is_valid()
        assert len(result.errors) >= 2

    def test_validate_field_valid(self, registry):
        result = registry.validate_field("port", 8080)
        assert result.is_valid()

    def test_validate_field_invalid(self, registry):
        result = registry.validate_field("port", 0)
        assert not result.is_valid()

    def test_validate_config_empty(self, registry):
        result = registry.validate_config({})
        assert result.is_valid()

    def test_validate_config_with_unknown_keys(self, registry):
        """Unknown keys should not cause validation errors (extra=allow)."""
        result = registry.validate_config({"some_future_flag": True})
        assert result.is_valid()

    def test_has_validator(self, registry):
        assert registry.has_validator("port") is True
        assert registry.has_validator("nonexistent") is False

    def test_get_validator(self, registry):
        assert registry.get_validator("port") is not None
        assert registry.get_validator("nonexistent") is None

    def test_get_registered_fields(self, registry):
        fields = registry.get_registered_fields()
        assert "port" in fields
        assert "gpu_memory_utilization" in fields
        assert "dtype" in fields

    def test_get_validation_summary(self, registry):
        summary = registry.get_validation_summary()
        assert summary["total_fields"] > 0
        assert "fields" in summary

    def test_clear_is_noop(self, registry):
        """clear() should not crash."""
        registry.clear()

    def test_register_is_noop(self, registry):
        """register() should not crash."""
        registry.register("test_field", None)


# ---------------------------------------------------------------------------
#  Backward compatibility: same scenarios as test_validation_simple.py
# ---------------------------------------------------------------------------

class TestBackwardCompatIntegerValidation:
    """Mirrors TestIntegerValidator from test_validation_simple.py via registry."""

    @pytest.fixture
    def registry(self):
        return create_pydantic_validation_registry()

    def test_valid_integer(self, registry):
        result = registry.validate_field("max_model_len", 4096)
        assert result.is_valid()

    def test_string_integer_coerced(self, registry):
        """Pydantic coerces digit-strings to int via field_validator."""
        result = registry.validate_field("max_model_len", "4096")
        assert result.is_valid()

    def test_invalid_type(self, registry):
        result = registry.validate_field("max_model_len", "not_a_number")
        assert not result.is_valid()

    def test_min_value_enforced(self, registry):
        result = registry.validate_field("max_model_len", 0)
        assert not result.is_valid()


class TestBackwardCompatFloatValidation:
    """Mirrors TestFloatValidator from test_validation_simple.py via registry."""

    @pytest.fixture
    def registry(self):
        return create_pydantic_validation_registry()

    def test_valid_float(self, registry):
        result = registry.validate_field("gpu_memory_utilization", 0.9)
        assert result.is_valid()

    def test_integer_accepted(self, registry):
        result = registry.validate_field("gpu_memory_utilization", 1)
        assert result.is_valid()

    def test_max_value_enforced(self, registry):
        result = registry.validate_field("gpu_memory_utilization", 1.5)
        assert not result.is_valid()


class TestBackwardCompatBooleanValidation:
    """Mirrors TestBooleanValidator from test_validation_simple.py via registry."""

    @pytest.fixture
    def registry(self):
        return create_pydantic_validation_registry()

    def test_accepts_true(self, registry):
        result = registry.validate_field("enforce_eager", True)
        assert result.is_valid()

    def test_accepts_false(self, registry):
        result = registry.validate_field("enforce_eager", False)
        assert result.is_valid()

    def test_string_true_variants(self, registry):
        for val in ["true", "True", "TRUE", "yes", "1"]:
            result = registry.validate_field("enforce_eager", val)
            assert result.is_valid(), f"Failed for input: {val}"


class TestBackwardCompatChoiceValidation:
    """Mirrors TestChoiceValidator from test_validation_simple.py via registry."""

    @pytest.fixture
    def registry(self):
        return create_pydantic_validation_registry()

    def test_valid_choice(self, registry):
        result = registry.validate_field("quantization", "awq")
        assert result.is_valid()

    def test_invalid_choice(self, registry):
        result = registry.validate_field("quantization", "invalid_method")
        assert not result.is_valid()
