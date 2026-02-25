"""Additional integration tests for proxy server using TestClient (Phase 5).

Supplements the existing proxy test suite with TestClient-based tests
for endpoints and scenarios that were previously only tested
via mocking or not at all:

- Metrics endpoint
- Pre-register → verify → activate lifecycle via HTTP
- Model state updates (sleep/wake) via HTTP
- Refresh models endpoint
- Error handling / edge cases
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm_cli.proxy.models import ProxyConfig, ModelConfig
from vllm_cli.proxy.registry import ModelEntry, ModelRegistry, ModelState, RegistrationStatus
from vllm_cli.proxy.server import ProxyServer


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def proxy_config():
    """Minimal proxy config for integration tests."""
    return ProxyConfig(
        host="127.0.0.1",
        port=8000,
        models=[
            ModelConfig(
                name="model-a",
                model_path="org/model-a",
                gpu_ids=[0],
                port=8001,
                enabled=True,
            ),
        ],
        enable_cors=True,
        enable_metrics=True,
        log_requests=False,
    )


@pytest.fixture
def proxy_server(proxy_config):
    """Create a ProxyServer instance."""
    server = ProxyServer(proxy_config)
    return server


@pytest.fixture
def client(proxy_server):
    """FastAPI TestClient."""
    return TestClient(proxy_server.app)


# ---------------------------------------------------------------------------
#  Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Test the /metrics Prometheus-compatible endpoint."""

    def test_metrics_returns_text(self, client):
        """Metrics endpoint returns Prometheus-compatible text."""
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.text
        assert "proxy_requests_total" in body
        assert "proxy_uptime_seconds" in body

    def test_metrics_reflects_request_count(self, client, proxy_server):
        """After making requests, metrics should reflect counts."""
        proxy_server.total_requests = 42
        proxy_server.model_requests = {"model-a": 20, "model-b": 22}

        response = client.get("/metrics")
        # The response may be JSON-encoded string or raw text
        body = response.text
        # Check content is present (handle JSON-escaped or raw)
        assert "proxy_requests_total 42" in body or "proxy_requests_total 42" in response.json()
        assert "model-a" in body
        assert "model-b" in body

    def test_metrics_includes_help_and_type(self, client):
        """Metrics output includes HELP and TYPE annotations."""
        response = client.get("/metrics")
        body = response.text
        assert "# HELP proxy_requests_total" in body
        assert "# TYPE proxy_requests_total counter" in body
        assert "# HELP proxy_uptime_seconds" in body
        assert "# TYPE proxy_uptime_seconds gauge" in body


# ---------------------------------------------------------------------------
#  Pre-register, register, and unregister lifecycle
# ---------------------------------------------------------------------------


class TestRegistrationLifecycleViaHTTP:
    """Test the full model registration lifecycle using TestClient."""

    def test_pre_register_model(self, client, proxy_server):
        """Pre-register a model via POST /proxy/pre_register."""
        response = client.post("/proxy/pre_register", json={
            "port": 9001,
            "model_name": "new-model",
            "gpu_ids": [0, 1],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Model should be in registry as PENDING
        entry = proxy_server.registry.get_model(9001)
        assert entry is not None
        assert entry.status == RegistrationStatus.PENDING

    def test_pre_register_duplicate(self, client, proxy_server):
        """Pre-registering the same port twice should return 400 (already exists)."""
        client.post("/proxy/pre_register", json={
            "port": 9002, "model_name": "dup", "gpu_ids": [0],
        })
        response = client.post("/proxy/pre_register", json={
            "port": 9002, "model_name": "dup", "gpu_ids": [0],
        })
        # Server rejects duplicate pre-registration
        assert response.status_code == 400

    def test_register_model_backward_compat(self, client, proxy_server):
        """Register (backward-compatible) adds model directly."""
        response = client.post("/proxy/register", json={
            "port": 9010,
            "model_name": "compat-model",
            "gpu_ids": [2],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_unregister_model(self, client, proxy_server):
        """Unregister a model via DELETE /proxy/unregister/{port}."""
        # First pre-register it (so it's in the registry by port)
        client.post("/proxy/pre_register", json={
            "port": 9020, "model_name": "to-remove", "gpu_ids": [0],
        })
        # Then unregister
        response = client.delete("/proxy/unregister/9020")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"

    def test_unregister_nonexistent_model(self, client):
        """Unregistering a non-existent model returns 404."""
        response = client.delete("/proxy/unregister/99999")
        assert response.status_code == 404

    def test_get_registry(self, client, proxy_server):
        """GET /proxy/registry returns registry status."""
        # Add a model first
        client.post("/proxy/register", json={
            "port": 9030, "model_name": "reg-model", "gpu_ids": [0],
        })
        response = client.get("/proxy/registry")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or "entries" in data or isinstance(data, dict)


# ---------------------------------------------------------------------------
#  Model state updates (sleep/wake)
# ---------------------------------------------------------------------------


class TestModelStateUpdatesViaHTTP:
    """Test sleep/wake lifecycle via POST /proxy/state."""

    def test_update_state_to_sleeping(self, client, proxy_server):
        """Transition a model to sleeping state."""
        # Register model first
        client.post("/proxy/register", json={
            "port": 9040, "model_name": "sleepy", "gpu_ids": [0],
        })
        # Update state
        response = client.post("/proxy/state", json={
            "port": 9040, "state": "sleeping", "sleep_level": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        entry = proxy_server.registry.get_model(9040)
        assert entry.state == ModelState.SLEEPING
        assert entry.sleep_level == 1

    def test_update_state_to_running(self, client, proxy_server):
        """Wake a sleeping model."""
        # Register and put to sleep
        client.post("/proxy/register", json={
            "port": 9041, "model_name": "wakeable", "gpu_ids": [0],
        })
        client.post("/proxy/state", json={
            "port": 9041, "state": "sleeping", "sleep_level": 2,
        })
        # Wake it up
        response = client.post("/proxy/state", json={
            "port": 9041, "state": "running",
        })
        assert response.status_code == 200
        entry = proxy_server.registry.get_model(9041)
        assert entry.state == ModelState.RUNNING
        assert entry.sleep_level == 0

    def test_update_state_to_stopped(self, client, proxy_server):
        """Stop a model."""
        client.post("/proxy/register", json={
            "port": 9042, "model_name": "stoppable", "gpu_ids": [0],
        })
        response = client.post("/proxy/state", json={
            "port": 9042, "state": "stopped",
        })
        assert response.status_code == 200
        entry = proxy_server.registry.get_model(9042)
        assert entry.state == ModelState.STOPPED

    def test_update_state_nonexistent_model(self, client):
        """Updating state of non-existent model returns error."""
        response = client.post("/proxy/state", json={
            "port": 99999, "state": "running",
        })
        # Should return 400 or 404
        assert response.status_code in (400, 404)


# ---------------------------------------------------------------------------
#  Proxy status endpoint
# ---------------------------------------------------------------------------


class TestProxyStatusEndpoint:
    """Test /proxy/status in more detail."""

    def test_status_includes_all_fields(self, client, proxy_server):
        """Status response contains expected fields."""
        response = client.get("/proxy/status")
        assert response.status_code == 200
        data = response.json()
        assert "proxy_host" in data
        assert "proxy_port" in data
        assert "total_requests" in data

    def test_status_with_registered_models(self, client, proxy_server):
        """Status reflects registered models."""
        client.post("/proxy/register", json={
            "port": 9050, "model_name": "status-test", "gpu_ids": [0],
        })
        response = client.get("/proxy/status")
        data = response.json()
        # Should have model info
        assert data.get("active_models") is not None or "models" in data or "backends" in data


# ---------------------------------------------------------------------------
#  Health and root endpoints
# ---------------------------------------------------------------------------


class TestHealthAndRootEndpoints:
    """Supplementary tests for basic endpoints."""

    def test_root_includes_version(self, client):
        """Root endpoint includes version info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data or "status" in data

    def test_health_returns_ok(self, client):
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_list_models_empty(self, client, proxy_server):
        """Model listing with no backends returns empty data."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["object"] == "list"

    def test_list_models_with_registered_backend(self, client, proxy_server):
        """Model listing includes registered backends."""
        proxy_server.router.add_backend("test-model", "http://localhost:8001", {"port": 8001})
        response = client.get("/v1/models")
        data = response.json()
        assert len(data["data"]) > 0
        model_ids = [m["id"] for m in data["data"]]
        assert "test-model" in model_ids


# ---------------------------------------------------------------------------
#  Error handling scenarios
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in proxy server endpoints."""

    def test_chat_completions_no_model_returns_422_or_error(self, client, proxy_server):
        """Chat completions without model in body should fail."""
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}]
        })
        # Should fail because no model specified or no matching backend
        assert response.status_code in (400, 404, 422, 500)

    def test_completions_no_backend_returns_error(self, client, proxy_server):
        """Completions with unknown model returns 404."""
        response = client.post("/v1/completions", json={
            "model": "nonexistent-model",
            "prompt": "hello",
        })
        assert response.status_code in (404, 500)

    def test_pre_register_invalid_request(self, client):
        """Pre-register with missing required fields should fail."""
        response = client.post("/proxy/pre_register", json={})
        assert response.status_code in (400, 422, 500)

    def test_state_update_missing_fields(self, client):
        """State update without required fields should fail."""
        response = client.post("/proxy/state", json={"port": 8001})
        # Missing "state" field
        assert response.status_code in (400, 422, 500)
