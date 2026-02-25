# vLLM CLI — Improvement Checklist

## Phase 1: Quick Fixes (< 30 min each) ✅

- [x] **Fix dead code in `__main__.py`** (lines 42–54)
  - The `try: pass / except ImportError` block never raises — replaced `pass` with `import torch` check so the PyTorch-missing message can display
- [x] **Consolidate duplicate `ConfigManager()` in `server/manager.py`**
  - `start()` instantiated `ConfigManager()` at line 109 and again at line 166 — now uses a single instance
- [x] **Fix README typos**
  - "seperately" → "separately" on lines 81 and 106
- [x] **Fix CHANGELOG comparison links**
  - Added missing URLs for v0.2.3, v0.2.4, and v0.2.5
- [x] **Rename shadowed builtins in `errors/system.py`**
  - `SystemError` → `VLLMSystemError`
  - `PermissionError` → `VLLMPermissionError`
  - `MemoryError` → `VLLMMemoryError`
  - `EnvironmentError` → `VLLMEnvironmentError`
  - Updated `errors/__init__.py` with backward-compatible aliases

## Phase 2: Dependency & CI Cleanup (1–2 hours) ✅

- [x] **Remove `requests` dependency — standardize on `httpx`**
  - Replaced `requests.get` with `httpx.get` in `validation/token.py`, `ui/settings.py`, `server/monitoring.py`, `server/manager.py`
  - Mapped `requests.exceptions.Timeout` → `httpx.TimeoutException`, `requests.exceptions.ConnectionError` → `httpx.ConnectError`
  - Removed `requests` from `pyproject.toml` main dependencies
  - Removed `requests` from `requirements.txt`
  - Kept `requests` in `[test]` optional deps for test compatibility
- [x] **Update `actions/cache@v3` → `v4`** in all CI workflow files
  - `ci.yml` (3 occurrences)
  - `quality.yml` (1 occurrence)
- [x] **Make lint/format checks blocking** in `ci.yml`
  - Changed `continue-on-error: true` → `false` for lint and format jobs
  - Type-check remains informational (non-blocking)

## Phase 3: Code Decomposition (2–4 hours) ✅

- [x] **Decompose `handle_proxy()` in `cli/handlers.py`**
  - Extracted `_handle_proxy_start()` (~150 lines)
  - Extracted `_handle_proxy_stop()` (~5 lines)
  - Extracted `_handle_proxy_status()` (~60 lines)
  - Extracted `_handle_proxy_add()` (~40 lines)
  - Extracted `_handle_proxy_remove()` (~30 lines)
  - Extracted `_handle_proxy_config()` (~45 lines)
  - Extracted `_get_proxy_connection()` shared helper (eliminates 3× duplicate pattern)
  - `handle_proxy()` is now a thin dispatcher using a handler dict
- [x] **Assessed `system/dependencies.py`** (1,292 lines) — **deferred**
  - Already has 16 well-scoped functions; no single function dominates
  - `get_attention_backend_usability()` (338 lines) is inherently complex (probes 7 GPU backends)
  - Splitting further would scatter tightly-coupled GPU detection logic into many tiny helpers
  - Better addressed in Phase 5 if needed
- [x] **Assessed large UI files** — **deferred**
  - `ui/custom_config.py` and `ui/server_control.py` are large but handle distinct UI flows
  - Decomposing interactive TUI modules risks breaking fragile prompt sequences
  - Better addressed alongside Phase 5 architecture improvements

## Phase 4: Test Coverage (4–8 hours) ✅

- [x] **Add tests for `errors/` package** (170 tests total, all passing)
  - `test_error_base.py` — 22 tests: VLLMCLIError serialization, context management, user messages, to_dict, str/repr, ValidationError subclass
  - `test_retry.py` — 16 tests: RetryConfig defaults/custom, exponential_backoff calculations, retry_with_backoff decorator (success/retry/exhaust/wrap), RetryableOperation context manager
  - `test_recovery.py` — 17 tests: ErrorRecovery handlers (GPU/model/server), AutoRecovery (CPU mode, memory reduction, quantization, port finding, similar models), documented pre-existing error_code kwargs collision bug
  - `test_error_handlers.py` — 14 tests: ErrorReporter (report/count/reset), error_boundary (wrap/suppress/context), safe_operation decorator, format_error_for_user, get_error_help_text
- [x] **Expand validation framework tests** (`test_validation_simple.py`)
  - 30 tests: IntegerValidator (8 incl. boundary), FloatValidator (7), StringValidator (8 incl. pattern), BooleanValidator (6 incl. string/numeric variants), ChoiceValidator (5 incl. case sensitivity)
  - ValidationResult container (3 tests)
  - ValidationRegistry register/get_validator/lookup (4 tests)
- [x] **Add tests for `system/dependencies.py`** (`test_dependencies.py`)
  - 21 tests: get_dependency_info aggregator, attention backend detection, quantization library checks, core dependencies structure, environment variable reads, _determine_effective_backend fallback chain (7 tests covering all priority levels)
- [x] **Expand `test_model_manager.py`**
  - 23 tests: ModelManager with mocked scan_for_models (listing, caching, refresh, search by name/publisher, case-insensitive search, filter by publisher/type, model count, empty scan, cache stats)
  - ModelCache (7 tests: store/retrieve, expiry, clear, replace, custom TTL)
  - **Bug found**: Pre-existing error_code kwargs collision in error subclass hierarchy (GPUError, GPUNotFoundError, PortInUseError, etc.) — child `__init__` passes `error_code` via `**kwargs` to parent that also sets it explicitly. Tests document this.

## Phase 5: Architecture Improvements (1–2 days)

- [x] **Leverage Pydantic for config validation**
  - ✅ Created `validation/pydantic_models.py` with `VLLMConfig` Pydantic model
  - ✅ `PydanticValidationRegistry` adapter is a drop-in replacement for old `ValidationRegistry`
  - ✅ `create_vllm_validation_registry()` now returns Pydantic-backed registry
  - ✅ All 35+ fields validated with `field_validator(mode="before")` for correct coercion
  - ✅ Cross-field dependency checks via `model_validator(mode="after")`
  - ✅ 101 tests in `test_pydantic_validation.py`, full backward compat maintained
  - ✅ Old framework files (`base.py`, `types.py`, `factory.py`, `registry.py`) kept for `load_validation_schema_from_file` utility
- [x] **Abstract platform-specific process management**
  - ✅ Created `server/platform.py` with OS-detection (`is_posix()`, `is_windows()`)
  - ✅ Wrapped `os.killpg()` / `os.kill()` / signals in `send_signal_to_process()` / `send_signal_to_process_group()`
  - ✅ High-level helpers: `graceful_stop_process_group()`, `graceful_stop_process()`, `start_subprocess_platform()`
  - ✅ Windows path stubs using `taskkill` / `CREATE_NEW_PROCESS_GROUP`
  - ✅ Refactored `server/manager.py` to use platform helpers (removed ~70 lines of raw signal code)
  - ✅ Refactored `server/monitoring.py` to use `check_process_alive()`
  - ✅ 21 tests in `test_platform.py` including real subprocess integration tests
- [ ] **Add integration tests for proxy server**
  - Use `TestClient` from FastAPI for in-process testing
  - Test model registration/unregistration lifecycle
  - Test request routing to correct backend
  - Test streaming responses
  - Test sleep/wake model lifecycle
