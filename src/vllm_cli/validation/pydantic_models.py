#!/usr/bin/env python3
"""
Pydantic-based configuration validation models for vLLM CLI.

Replaces the custom hand-rolled validation framework (base.py, types.py,
factory.py, registry.py, schema.py) with Pydantic v2 models.  The public
surface is kept backward-compatible so that ConfigManager and tests
continue to work unchanged.

Key design decisions:
    - VLLMConfig is the single Pydantic model for all vLLM parameters.
    - ``model_config = ConfigDict(extra="allow")`` lets unknown keys pass
      through (future vLLM flags the CLI doesn't know about yet).
    - A thin ``PydanticValidationRegistry`` adapter exposes the same
      ``validate_config`` / ``validate_field`` / ``get_validator`` API that
      ConfigManager already calls.
    - The ``CompatibilityValidator`` is kept as-is (it deals with semantic
      cross-field warnings, not type checking).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, ValidationError as PydanticValidationError
from pydantic import field_validator, model_validator

from .base import ValidationError, ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Flexible coercion helpers
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"true", "yes", "1", "on", "enabled"})
_FALSY = frozenset({"false", "no", "0", "off", "disabled"})


def _coerce_bool(v: Any) -> bool:
    """Accept True/False, string synonyms, and 0/1."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        low = v.strip().lower()
        if low in _TRUTHY:
            return True
        if low in _FALSY:
            return False
    raise ValueError(f"Cannot interpret {v!r} as a boolean")


def _coerce_pos_int(v: Any) -> Optional[int]:
    """Coerce to int ≥ 1, allow None."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("Booleans are not valid integers")
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v.strip())
    if not isinstance(v, int):
        raise ValueError(f"Expected an integer, got {type(v).__name__}")
    if v < 1:
        raise ValueError(f"Must be >= 1, got {v}")
    return v


def _coerce_non_neg_int(v: Any) -> Optional[int]:
    """Coerce to int ≥ 0, allow None."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("Booleans are not valid integers")
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v.strip())
    if not isinstance(v, int):
        raise ValueError(f"Expected an integer, got {type(v).__name__}")
    if v < 0:
        raise ValueError(f"Must be >= 0, got {v}")
    return v


# ---------------------------------------------------------------------------
#  Core Pydantic model
# ---------------------------------------------------------------------------


class VLLMConfig(BaseModel):
    """
    Pydantic model that validates all known vLLM CLI configuration fields.

    Extra fields are allowed so that new vLLM flags can be passed through
    without modifying this model.
    """

    model_config = ConfigDict(extra="allow")

    # ── Core server ──────────────────────────────────────────────────────
    host: Optional[str] = None
    port: Optional[int] = None

    # ── Model ────────────────────────────────────────────────────────────
    dtype: Optional[str] = None
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None

    # ── Parallelism ──────────────────────────────────────────────────────
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None

    # ── Memory ───────────────────────────────────────────────────────────
    gpu_memory_utilization: Optional[float] = None
    max_num_batched_tokens: Optional[int] = None
    max_paddings: Optional[int] = None

    # ── Quantization ─────────────────────────────────────────────────────
    quantization: Optional[str] = None

    # ── KV cache ─────────────────────────────────────────────────────────
    kv_cache_dtype: Optional[str] = None
    block_size: Optional[int] = None

    # ── Attention ────────────────────────────────────────────────────────
    enable_prefix_caching: Optional[bool] = None
    enable_chunked_prefill: Optional[bool] = None
    max_num_on_the_fly_seq_groups: Optional[int] = None

    # ── Performance ──────────────────────────────────────────────────────
    enforce_eager: Optional[bool] = None
    max_context_len_to_capture: Optional[int] = None

    # ── Loading ──────────────────────────────────────────────────────────
    load_format: Optional[str] = None
    download_dir: Optional[str] = None

    # ── CPU offloading ───────────────────────────────────────────────────
    cpu_offload_gb: Optional[float] = None

    # ── Trust & safety ───────────────────────────────────────────────────
    trust_remote_code: Optional[bool] = None
    disable_log_stats: Optional[bool] = None
    disable_log_requests: Optional[bool] = None

    # ── Advanced ─────────────────────────────────────────────────────────
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: Optional[int] = None

    # ── Workers / distribution ───────────────────────────────────────────
    worker_use_ray: Optional[bool] = None
    ray_workers_use_nsight: Optional[bool] = None

    # ── Speculative decoding ─────────────────────────────────────────────
    speculative_model: Optional[str] = None
    num_speculative_tokens: Optional[int] = None

    # ── LoRA ─────────────────────────────────────────────────────────────
    enable_lora: Optional[bool] = None
    max_loras: Optional[int] = None
    max_lora_rank: Optional[int] = None
    lora_extra_vocab_size: Optional[int] = None

    # ------------------------------------------------------------------
    #  Field validators (class-level, reusable)
    # ------------------------------------------------------------------

    # --- port ---
    @field_validator("port", mode="before")
    @classmethod
    def _validate_port(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("Booleans are not valid port numbers")
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        if not isinstance(v, int):
            raise ValueError(f"port must be an integer, got {type(v).__name__}")
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    # --- host / download_dir / revision / tokenizer_revision / speculative_model ---
    @field_validator("host", "download_dir", "revision", "tokenizer_revision", "speculative_model")
    @classmethod
    def _validate_nonempty_string(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v)
        if len(s) < 1:
            raise ValueError("Must be at least 1 character long")
        return s

    # --- dtype ---
    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        allowed = {"auto", "float16", "bfloat16", "float32"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of: {', '.join(sorted(allowed))}")
        return v

    # --- quantization ---
    @field_validator("quantization")
    @classmethod
    def _validate_quantization(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        allowed = {
            "awq", "awq_marlin", "gptq", "gptq_marlin",
            "bitsandbytes", "fp8", "gguf", "compressed-tensors",
        }
        if v not in allowed:
            raise ValueError(f"quantization must be one of: {', '.join(sorted(allowed))}")
        return v

    # --- kv_cache_dtype ---
    @field_validator("kv_cache_dtype")
    @classmethod
    def _validate_kv_cache_dtype(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        allowed = {"auto", "fp8", "fp16", "bf16"}
        if v not in allowed:
            raise ValueError(f"kv_cache_dtype must be one of: {', '.join(sorted(allowed))}")
        return v

    # --- load_format ---
    @field_validator("load_format")
    @classmethod
    def _validate_load_format(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        allowed = {"auto", "pt", "safetensors", "npcache", "dummy"}
        if v not in allowed:
            raise ValueError(f"load_format must be one of: {', '.join(sorted(allowed))}")
        return v

    # --- positive integers ---
    @field_validator(
        "max_model_len", "max_num_seqs", "tensor_parallel_size",
        "pipeline_parallel_size", "max_num_batched_tokens",
        "max_num_on_the_fly_seq_groups", "max_context_len_to_capture",
        "max_loras", "max_lora_rank", "num_speculative_tokens",
        mode="before",
    )
    @classmethod
    def _validate_positive_int(cls, v: Any) -> Optional[int]:
        return _coerce_pos_int(v)

    # --- non-negative integers ---
    @field_validator("max_paddings", "lora_extra_vocab_size", "seed", mode="before")
    @classmethod
    def _validate_non_neg_int(cls, v: Any) -> Optional[int]:
        return _coerce_non_neg_int(v)

    # --- block_size (1–256) ---
    @field_validator("block_size", mode="before")
    @classmethod
    def _validate_block_size(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("Booleans are not valid integers")
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        if not isinstance(v, int):
            raise ValueError(f"block_size must be an integer, got {type(v).__name__}")
        if not 1 <= v <= 256:
            raise ValueError(f"block_size must be between 1 and 256, got {v}")
        return v

    # --- gpu_memory_utilization (0.0–1.0) ---
    @field_validator("gpu_memory_utilization", mode="before")
    @classmethod
    def _validate_gpu_mem(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("Booleans are not valid numbers")
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                raise ValueError(f"gpu_memory_utilization must be a number, got {v!r}")
        if not isinstance(v, (int, float)):
            raise ValueError(f"gpu_memory_utilization must be a number, got {type(v).__name__}")
        fv = float(v)
        if not 0.0 <= fv <= 1.0:
            raise ValueError(f"gpu_memory_utilization must be between 0.0 and 1.0, got {fv}")
        return fv

    # --- cpu_offload_gb (>= 0) ---
    @field_validator("cpu_offload_gb", mode="before")
    @classmethod
    def _validate_cpu_offload(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("Booleans are not valid numbers")
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                raise ValueError(f"cpu_offload_gb must be a number, got {v!r}")
        if not isinstance(v, (int, float)):
            raise ValueError(f"cpu_offload_gb must be a number, got {type(v).__name__}")
        fv = float(v)
        if fv < 0.0:
            raise ValueError(f"cpu_offload_gb must be >= 0.0, got {fv}")
        return fv

    # --- boolean fields ---
    @field_validator(
        "enable_prefix_caching", "enable_chunked_prefill", "enforce_eager",
        "trust_remote_code", "disable_log_stats", "disable_log_requests",
        "worker_use_ray", "ray_workers_use_nsight", "enable_lora",
        mode="before",
    )
    @classmethod
    def _validate_bool(cls, v: Any) -> Optional[bool]:
        if v is None:
            return None
        return _coerce_bool(v)

    # ------------------------------------------------------------------
    #  Model-level validators (cross-field dependency checks)
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_dependencies(self) -> "VLLMConfig":
        """Warn about dependency issues (LoRA, speculative, ray)."""
        # LoRA dependencies
        lora_fields = {
            "max_loras": self.max_loras,
            "max_lora_rank": self.max_lora_rank,
            "lora_extra_vocab_size": self.lora_extra_vocab_size,
        }
        for fname, fval in lora_fields.items():
            if fval is not None and not self.enable_lora:
                logger.warning(
                    f"{fname} is set but enable_lora is not True – "
                    "LoRA parameters will be ignored by vLLM"
                )

        # Speculative decoding dependency
        if self.num_speculative_tokens is not None and not self.speculative_model:
            logger.warning(
                "num_speculative_tokens is set but speculative_model is not – "
                "speculative decoding parameters will be ignored"
            )

        # Ray dependency
        if self.ray_workers_use_nsight and not self.worker_use_ray:
            logger.warning(
                "ray_workers_use_nsight is set but worker_use_ray is not True"
            )

        return self


# ---------------------------------------------------------------------------
#  Adapter: PydanticValidationRegistry
#
#  Drop-in replacement for ``ValidationRegistry`` so that
#  ``ConfigManager`` keeps working without code changes.
# ---------------------------------------------------------------------------


class PydanticValidationRegistry:
    """
    Adapter that wraps VLLMConfig to expose the same API as the old
    hand-rolled ``ValidationRegistry``.

    Public methods used by ConfigManager:
        - ``validate_config(config) -> ValidationResult``
        - ``validate_field(field_name, value, context=None) -> ValidationResult``
        - ``get_validator(field_name) -> object | None``
        - ``has_validator(field_name) -> bool``
        - ``get_registered_fields() -> list[str]``
        - ``get_validation_summary() -> dict``
    """

    # All fields known to VLLMConfig (excluding Pydantic internals)
    _known_fields: frozenset[str] = frozenset(VLLMConfig.model_fields.keys())

    # ------------------------------------------------------------------
    #  Core validation
    # ------------------------------------------------------------------

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate an entire configuration dictionary."""
        result = ValidationResult()
        try:
            VLLMConfig(**config)
        except PydanticValidationError as exc:
            for err in exc.errors():
                field = ".".join(str(loc) for loc in err["loc"]) if err["loc"] else "unknown"
                msg = err["msg"]
                result.add_error(ValidationError(field, config.get(field), msg))
        return result

    def validate_field(
        self,
        field_name: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate a single field value."""
        config = dict(context or {})
        config[field_name] = value
        # Run full model but only report errors for the target field
        full_result = self.validate_config(config)
        result = ValidationResult()
        for error in full_result.errors:
            if error.field == field_name:
                result.add_error(error)
        result.warnings = full_result.warnings
        return result

    # ------------------------------------------------------------------
    #  Introspection helpers (kept for backward-compat)
    # ------------------------------------------------------------------

    def get_validator(self, field_name: str) -> Optional[Any]:
        """Return a truthy sentinel when the field is known."""
        if field_name in self._known_fields:
            return True  # non-None → "validator exists"
        return None

    def has_validator(self, field_name: str) -> bool:
        return field_name in self._known_fields

    def get_registered_fields(self) -> List[str]:
        return sorted(self._known_fields)

    def get_validation_summary(self) -> Dict[str, Any]:
        return {
            "total_fields": len(self._known_fields),
            "fields": sorted(self._known_fields),
            "validator_counts": {"PydanticFieldValidator": len(self._known_fields)},
        }

    def clear(self) -> None:
        """No-op for backward compatibility."""

    def register(self, field_name: str, validator: Any) -> None:
        """No-op for backward compatibility (schema is declarative)."""
        logger.debug(
            f"PydanticValidationRegistry.register() called for {field_name} – "
            "ignored (validation is now declarative via Pydantic model)"
        )


# ---------------------------------------------------------------------------
#  Public factory kept for backward-compat with ConfigManager import
# ---------------------------------------------------------------------------


def create_pydantic_validation_registry() -> PydanticValidationRegistry:
    """Create a PydanticValidationRegistry (drop-in for create_vllm_validation_registry)."""
    return PydanticValidationRegistry()
