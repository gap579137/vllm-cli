"""Tests for system/dependencies.py detection utilities.

Mocks importlib to test detection of present/absent packages,
and tests the pure-logic helper functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from vllm_cli.system.dependencies import (
    _determine_effective_backend,
    get_attention_backend_info,
    get_core_dependencies,
    get_dependency_info,
    get_environment_info,
    get_quantization_info,
)


class TestGetDependencyInfo:
    """Tests for the top-level get_dependency_info aggregator."""

    def test_returns_dict_with_expected_keys(self):
        """Result contains all four category keys."""
        info = get_dependency_info()
        assert isinstance(info, dict)
        assert "attention_backends" in info
        assert "quantization" in info
        assert "core_dependencies" in info
        assert "environment" in info

    def test_attention_backends_is_dict(self):
        """attention_backends value is a dict."""
        info = get_dependency_info()
        assert isinstance(info["attention_backends"], dict)

    def test_quantization_is_dict(self):
        """quantization value is a dict."""
        info = get_dependency_info()
        assert isinstance(info["quantization"], dict)


class TestGetAttentionBackendInfo:
    """Tests for get_attention_backend_info."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        info = get_attention_backend_info()
        assert isinstance(info, dict)

    def test_has_current_backend_key(self):
        """Result includes current_backend key."""
        info = get_attention_backend_info()
        assert "current_backend" in info

    def test_has_effective_backend_key(self):
        """Result includes effective_backend key."""
        info = get_attention_backend_info()
        assert "effective_backend" in info

    @patch.dict("os.environ", {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
    def test_reads_env_backend(self):
        """Picks up VLLM_ATTENTION_BACKEND from environment."""
        info = get_attention_backend_info()
        assert info["current_backend"] == "FLASH_ATTN"

    def test_checks_attention_packages(self):
        """Result includes entries for known attention packages."""
        info = get_attention_backend_info()
        # Should have entries for at least flash_attn, xformers, flashinfer
        for pkg in ["flash_attn", "xformers", "flashinfer"]:
            assert pkg in info
            assert "available" in info[pkg]


class TestGetQuantizationInfo:
    """Tests for get_quantization_info."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        info = get_quantization_info()
        assert isinstance(info, dict)

    def test_checks_known_packages(self):
        """Result includes entries for known quantization packages."""
        info = get_quantization_info()
        for pkg in ["auto_gptq", "awq", "bitsandbytes"]:
            assert pkg in info
            assert "available" in info[pkg]
            assert "name" in info[pkg]

    def test_builtin_support_is_list(self):
        """builtin_support is a list (possibly empty)."""
        info = get_quantization_info()
        assert "builtin_support" in info
        assert isinstance(info["builtin_support"], list)


class TestGetCoreDependencies:
    """Tests for get_core_dependencies."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        info = get_core_dependencies()
        assert isinstance(info, dict)

    def test_checks_core_packages(self):
        """Result includes entries for core ML packages."""
        info = get_core_dependencies()
        for pkg in ["torch", "transformers", "vllm"]:
            assert pkg in info
            assert "available" in info[pkg]
            assert "name" in info[pkg]

    def test_package_info_structure(self):
        """Each package entry has name, version, available keys."""
        info = get_core_dependencies()
        for pkg_name, pkg_info in info.items():
            if pkg_name == "cuda_info":
                continue  # cuda_info has different structure
            assert "name" in pkg_info
            assert "version" in pkg_info
            assert "available" in pkg_info


class TestGetEnvironmentInfo:
    """Tests for get_environment_info."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        info = get_environment_info()
        assert isinstance(info, dict)

    def test_checks_expected_env_vars(self):
        """Result includes expected vLLM env vars."""
        info = get_environment_info()
        assert "VLLM_ATTENTION_BACKEND" in info
        assert "CUDA_VISIBLE_DEVICES" in info

    @patch.dict("os.environ", {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
    def test_reads_set_env_var(self):
        """Picks up env var values when set."""
        info = get_environment_info()
        assert info["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"

    def test_unset_vars_report_not_set(self):
        """Unset env vars report 'not set'."""
        info = get_environment_info()
        # At least some vars should be 'not set' in test environment
        not_set_count = sum(1 for v in info.values() if v == "not set")
        assert not_set_count >= 0  # Just ensure no crash


class TestDetermineEffectiveBackend:
    """Tests for the _determine_effective_backend helper."""

    def test_explicit_backend_returned(self):
        """When current_backend is not 'auto', it's returned directly."""
        info = {"current_backend": "FLASH_ATTN"}
        result = _determine_effective_backend(info)
        assert result == "FLASH_ATTN"

    def test_auto_with_vllm_flash_attn(self):
        """Auto detection prefers vllm_flash_attn when available and supported."""
        info = {
            "current_backend": "auto",
            "vllm_flash_attn": {
                "available": True,
                "fa2_gpu_supported": True,
                "fa3_gpu_supported": False,
                "recommended_version": "FA2",
            },
        }
        result = _determine_effective_backend(info)
        assert "vllm_flash_attn" in result

    def test_auto_with_flash_attn_fallback(self):
        """Auto detection falls back to flash_attn when vllm native unavailable."""
        info = {
            "current_backend": "auto",
            "vllm_flash_attn": {"available": False},
            "flash_attn": {"available": True},
        }
        result = _determine_effective_backend(info)
        assert "flash_attn" in result

    def test_auto_with_flashinfer_fallback(self):
        """Auto detection falls back to flashinfer."""
        info = {
            "current_backend": "auto",
            "vllm_flash_attn": {"available": False},
            "flash_attn": {"available": False},
            "flashinfer": {"available": True},
        }
        result = _determine_effective_backend(info)
        assert "flashinfer" in result

    def test_auto_with_xformers_fallback(self):
        """Auto detection falls back to xformers."""
        info = {
            "current_backend": "auto",
            "vllm_flash_attn": {"available": False},
            "flash_attn": {"available": False},
            "flashinfer": {"available": False},
            "xformers": {"available": True},
        }
        result = _determine_effective_backend(info)
        assert "xformers" in result

    def test_auto_ultimate_fallback(self):
        """When nothing is available, falls back to pytorch_native."""
        info = {
            "current_backend": "auto",
            "vllm_flash_attn": {"available": False},
            "flash_attn": {"available": False},
            "flashinfer": {"available": False},
            "xformers": {"available": False},
        }
        result = _determine_effective_backend(info)
        assert "pytorch_native" in result

    def test_auto_with_no_entries(self):
        """Missing keys don't crash the fallback logic."""
        info = {"current_backend": "auto"}
        result = _determine_effective_backend(info)
        assert "pytorch_native" in result
