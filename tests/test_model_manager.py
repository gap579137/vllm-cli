"""Expanded tests for model management and discovery.

Covers ModelManager operations with mocked scan_for_models,
search, filtering by publisher/type, cache lifecycle,
and ModelCache edge cases.
"""

import time
from unittest.mock import patch

import pytest

from vllm_cli.models.cache import ModelCache
from vllm_cli.models.manager import ModelManager


# --- Sample model data for mocking ---

SAMPLE_MODELS = [
    {
        "name": "meta-llama/Llama-2-7b-hf",
        "path": "/models/llama-2-7b",
        "size": 13_000_000_000,
        "type": "model",
        "publisher": "meta-llama",
        "display_name": "Llama-2-7b-hf",
    },
    {
        "name": "mistralai/Mistral-7B-v0.1",
        "path": "/models/mistral-7b",
        "size": 14_000_000_000,
        "type": "model",
        "publisher": "mistralai",
        "display_name": "Mistral-7B-v0.1",
    },
    {
        "name": "meta-llama/Llama-2-13b-hf",
        "path": "/models/llama-2-13b",
        "size": 26_000_000_000,
        "type": "model",
        "publisher": "meta-llama",
        "display_name": "Llama-2-13b-hf",
    },
    {
        "name": "my-custom-model",
        "path": "/models/custom",
        "size": 5_000_000_000,
        "type": "custom_model",
        "publisher": "local",
        "display_name": "my-custom-model",
    },
]


def _mock_scan():
    """Return sample models for testing."""
    return SAMPLE_MODELS.copy()


def _mock_build_dict(item):
    """Pass through item as-is for testing."""
    return item


class TestModelManager:
    """Test ModelManager functionality."""

    def test_init(self):
        """Test ModelManager initialization."""
        manager = ModelManager()
        assert manager.cache is not None
        assert isinstance(manager.cache, ModelCache)

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_list_available_models(self, mock_build, mock_scan):
        """list_available_models returns processed models."""
        manager = ModelManager()
        models = manager.list_available_models()
        assert len(models) == 4
        assert mock_scan.called

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_models_sorted_by_name(self, mock_build, mock_scan):
        """Models are returned sorted by name."""
        manager = ModelManager()
        models = manager.list_available_models()
        names = [m["name"] for m in models]
        assert names == sorted(names)

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_list_uses_cache(self, mock_build, mock_scan):
        """Second call uses cache instead of re-scanning."""
        manager = ModelManager()
        manager.list_available_models()
        manager.list_available_models()
        assert mock_scan.call_count == 1  # Only called once

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_refresh_forces_rescan(self, mock_build, mock_scan):
        """refresh=True bypasses cache."""
        manager = ModelManager()
        manager.list_available_models()
        manager.list_available_models(refresh=True)
        assert mock_scan.call_count == 2

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_search_by_name(self, mock_build, mock_scan):
        """search_models filters by model name."""
        manager = ModelManager()
        results = manager.search_models("llama")
        assert len(results) == 2
        assert all("llama" in m["name"].lower() for m in results)

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_search_by_publisher(self, mock_build, mock_scan):
        """search_models matches on publisher field."""
        manager = ModelManager()
        results = manager.search_models("mistralai")
        assert len(results) == 1
        assert results[0]["publisher"] == "mistralai"

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_search_case_insensitive(self, mock_build, mock_scan):
        """Search is case-insensitive."""
        manager = ModelManager()
        results = manager.search_models("LLAMA")
        assert len(results) == 2

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_search_no_match(self, mock_build, mock_scan):
        """Search with no match returns empty list."""
        manager = ModelManager()
        results = manager.search_models("nonexistent-model-xyz")
        assert results == []

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_get_model_count(self, mock_build, mock_scan):
        """get_model_count returns total count."""
        manager = ModelManager()
        assert manager.get_model_count() == 4

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_get_models_by_publisher(self, mock_build, mock_scan):
        """get_models_by_publisher filters correctly."""
        manager = ModelManager()
        results = manager.get_models_by_publisher("meta-llama")
        assert len(results) == 2
        assert all(m["publisher"] == "meta-llama" for m in results)

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_get_models_by_publisher_case_insensitive(self, mock_build, mock_scan):
        """get_models_by_publisher is case-insensitive."""
        manager = ModelManager()
        results = manager.get_models_by_publisher("Meta-Llama")
        assert len(results) == 2

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_get_models_by_type(self, mock_build, mock_scan):
        """get_models_by_type filters by model type."""
        manager = ModelManager()
        customs = manager.get_models_by_type("custom_model")
        assert len(customs) == 1
        assert customs[0]["type"] == "custom_model"

    @patch("vllm_cli.models.manager.scan_for_models", side_effect=_mock_scan)
    @patch("vllm_cli.models.manager.build_model_dict", side_effect=_mock_build_dict)
    def test_get_models_by_type_no_match(self, mock_build, mock_scan):
        """get_models_by_type with unknown type returns empty list."""
        manager = ModelManager()
        results = manager.get_models_by_type("nonexistent_type")
        assert results == []

    @patch("vllm_cli.models.manager.scan_for_models", return_value=[])
    def test_empty_scan_returns_empty_list(self, mock_scan):
        """When no models are found, returns empty list."""
        manager = ModelManager()
        models = manager.list_available_models()
        assert models == []

    def test_get_cache_stats(self):
        """get_cache_stats returns a dict."""
        manager = ModelManager()
        stats = manager.get_cache_stats()
        assert isinstance(stats, dict)


class TestModelCache:
    """Test ModelCache functionality."""

    def test_cache_models_and_get(self):
        """Test caching and getting models."""
        cache = ModelCache()
        test_models = [{"name": "model1"}, {"name": "model2"}]
        cache.cache_models(test_models)
        retrieved = cache.get_cached_models()
        assert retrieved == test_models

    def test_cache_expiry(self):
        """Test cache expiry based on TTL."""
        cache = ModelCache(ttl_seconds=0.1)  # 100ms TTL
        test_models = [{"name": "model1"}]
        cache.cache_models(test_models)
        assert cache.get_cached_models() is not None
        time.sleep(0.2)
        assert cache.get_cached_models() is None

    def test_get_from_empty_cache(self):
        """Empty cache returns None."""
        cache = ModelCache()
        assert cache.get_cached_models() is None

    def test_clear_cache(self):
        """clear_cache removes cached data."""
        cache = ModelCache()
        cache.cache_models([{"name": "model1"}])
        cache.clear_cache()
        assert cache.get_cached_models() is None

    def test_cache_stats(self):
        """get_cache_stats returns stats dict."""
        cache = ModelCache()
        stats = cache.get_cache_stats()
        assert isinstance(stats, dict)

    def test_cache_models_replaces_old(self):
        """Caching new models replaces old ones."""
        cache = ModelCache()
        cache.cache_models([{"name": "old"}])
        cache.cache_models([{"name": "new"}])
        retrieved = cache.get_cached_models()
        assert len(retrieved) == 1
        assert retrieved[0]["name"] == "new"

    def test_custom_ttl(self):
        """Custom TTL is respected."""
        cache = ModelCache(ttl_seconds=10.0)
        cache.cache_models([{"name": "model1"}])
        # Should still be valid after a short time
        time.sleep(0.05)
        assert cache.get_cached_models() is not None
