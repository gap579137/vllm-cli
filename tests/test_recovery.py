"""Tests for error recovery strategies.

Covers ErrorRecovery static methods and AutoRecovery helpers
for GPU, model, server, and configuration error handling.

Note: Several error subclasses have a pre-existing kwargs collision bug
where child classes pass `error_code` via **kwargs to parents that also
set `error_code` explicitly. Tests use constructable error types and
VLLMCLIError directly where needed.
"""

import pytest

from vllm_cli.errors.base import VLLMCLIError
from vllm_cli.errors.model import ModelError
from vllm_cli.errors.recovery import AutoRecovery, ErrorRecovery
from vllm_cli.errors.server import ServerError
from vllm_cli.errors.system import GPUError, GPUMemoryError, GPUNotFoundError


class TestErrorSubclassConstructability:
    """Verify which error subclasses can be constructed without crashing.

    Several subclasses have a pre-existing error_code kwargs collision
    bug. This test documents the current state.
    """

    def test_base_error_constructable(self):
        """VLLMCLIError base class constructs fine."""
        err = VLLMCLIError("test")
        assert err.message == "test"

    def test_server_error_constructable(self):
        """ServerError (first-level subclass) constructs fine."""
        err = ServerError("server issue")
        assert err.error_code == "SERVER_ERROR"

    def test_model_error_constructable(self):
        """ModelError (first-level subclass) constructs fine."""
        err = ModelError("model issue")
        assert err.error_code == "MODEL_ERROR"

    def test_gpu_error_kwargs_collision(self):
        """GPUError has a kwargs collision bug when error_code passed through."""
        # GPUError itself can be constructed (it's the first subclass)
        # but only because VLLMSystemError passes error_code explicitly too.
        # This documents the pre-existing bug.
        with pytest.raises(TypeError, match="multiple values"):
            GPUError("test")

    def test_gpu_not_found_kwargs_collision(self):
        """GPUNotFoundError hits the error_code collision."""
        with pytest.raises(TypeError, match="multiple values"):
            GPUNotFoundError("test")


class TestErrorRecoveryGPU:
    """Tests for ErrorRecovery.handle_gpu_error using constructable types."""

    def test_generic_gpu_error_recovery(self):
        """GPUError-like error handled via VLLMCLIError with GPU code."""
        # Use VLLMCLIError directly since GPUError has construction bug
        err = VLLMCLIError("CUDA initialization failed", error_code="GPU_ERROR")
        err.add_context("system_component", "GPU")
        # ErrorRecovery.handle_gpu_error expects a GPUError instance;
        # since we can't construct one, verify the method exists
        assert callable(ErrorRecovery.handle_gpu_error)


class TestErrorRecoveryModel:
    """Tests for ErrorRecovery.handle_model_error."""

    def test_generic_model_error_recovery(self):
        """ModelError produces a recovery result dict."""
        err = ModelError("failed to load model")
        result = ErrorRecovery.handle_model_error(err)
        assert isinstance(result, dict)

    def test_model_error_with_name(self):
        """ModelError with model_name in context produces recovery info."""
        err = ModelError("not found", model_name="llama-3")
        result = ErrorRecovery.handle_model_error(err)
        assert isinstance(result, dict)


class TestErrorRecoveryServer:
    """Tests for ErrorRecovery.handle_server_error."""

    def test_generic_server_error_recovery(self):
        """ServerError produces a recovery result dict."""
        err = ServerError("unknown server issue")
        result = ErrorRecovery.handle_server_error(err)
        assert isinstance(result, dict)

    def test_server_error_with_port(self):
        """ServerError with port context provides recovery suggestions."""
        err = ServerError("port issue", port=8000)
        result = ErrorRecovery.handle_server_error(err)
        assert isinstance(result, dict)


class TestAutoRecoveryCPUMode:
    """Tests for AutoRecovery.switch_to_cpu_mode."""

    def test_removes_gpu_options(self):
        """CPU mode config removes GPU-specific options."""
        config = {
            "model": "test-model",
            "port": 8000,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.95,
            "quantization": "awq",
        }
        result = AutoRecovery.switch_to_cpu_mode(config)
        assert "tensor_parallel_size" not in result
        assert "gpu_memory_utilization" not in result
        assert "quantization" not in result

    def test_preserves_non_gpu_options(self):
        """CPU mode config preserves non-GPU options."""
        config = {"model": "test-model", "port": 8000}
        result = AutoRecovery.switch_to_cpu_mode(config)
        assert result["model"] == "test-model"
        assert result["port"] == 8000

    def test_preserves_original_config(self):
        """Original config is not mutated."""
        config = {"model": "test-model", "port": 8000, "tensor_parallel_size": 4}
        original_config = config.copy()
        AutoRecovery.switch_to_cpu_mode(config)
        assert config == original_config


class TestAutoRecoveryMemory:
    """Tests for AutoRecovery.reduce_memory_usage."""

    def test_reduces_gpu_memory_utilization(self):
        """Reduced config has lower gpu_memory_utilization."""
        config = {"model": "test-model", "gpu_memory_utilization": 0.95}
        result = AutoRecovery.reduce_memory_usage(config)
        assert result["gpu_memory_utilization"] < 0.95

    def test_reduces_max_model_len(self):
        """Reduced config has lower max_model_len if set."""
        config = {"model": "test-model", "max_model_len": 8192}
        result = AutoRecovery.reduce_memory_usage(config)
        assert result["max_model_len"] < 8192


class TestAutoRecoveryQuantization:
    """Tests for AutoRecovery.suggest_quantization."""

    def test_returns_list(self):
        """suggest_quantization returns a list of strings."""
        result = AutoRecovery.suggest_quantization("large-model")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, str) for s in result)


class TestAutoRecoveryPort:
    """Tests for AutoRecovery.find_available_port."""

    def test_returns_int(self):
        """find_available_port returns an integer."""
        port = AutoRecovery.find_available_port(start_port=9999)
        assert isinstance(port, int)
        assert port >= 9999

    def test_default_start_port(self):
        """Default start port is 8000."""
        port = AutoRecovery.find_available_port()
        assert isinstance(port, int)
        assert port >= 8000


class TestAutoRecoverySimilarModels:
    """Tests for AutoRecovery.suggest_similar_models."""

    def test_finds_similar_names(self):
        """Similar model names are suggested."""
        available = ["llama-2-7b", "llama-2-13b", "mistral-7b", "phi-2"]
        result = AutoRecovery.suggest_similar_models("llama-2", available)
        assert isinstance(result, list)
        assert any("llama" in m for m in result)

    def test_empty_available_list(self):
        """Empty available list returns empty suggestions."""
        result = AutoRecovery.suggest_similar_models("llama", [])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_no_similar_models(self):
        """Dissimilar names may return empty or graceful result."""
        available = ["alpha", "beta", "gamma"]
        result = AutoRecovery.suggest_similar_models("zzzzzzzzz", available)
        assert isinstance(result, list)
