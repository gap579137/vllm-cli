#!/usr/bin/env python3
"""
Validation schema builder for vLLM CLI configuration.

This module provides a declarative way to define validation rules
for vLLM configuration parameters, building on the validators module
to create a comprehensive validation system.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import CompositeValidator
from .factory import (
    create_boolean_validator,
    create_choice_validator,
    create_float_validator,
    create_integer_validator,
    create_string_validator,
    validate_non_negative_integer,
    validate_port_number,
    validate_positive_integer,
    validate_probability,
)
from .registry import ValidationRegistry

logger = logging.getLogger(__name__)


def create_vllm_validation_registry() -> "PydanticValidationRegistry":
    """
    Create a validation registry for vLLM configuration parameters.

    Uses the Pydantic-based VLLMConfig model for type checking, range
    validation, and dependency checking for all supported parameters.

    Returns:
        PydanticValidationRegistry backed by VLLMConfig
    """
    from .pydantic_models import create_pydantic_validation_registry

    return create_pydantic_validation_registry()


# Keep legacy alias importable for backward-compat
_add_dependency_validators = None  # no longer needed; deps live in VLLMConfig model_validator


def create_compatibility_validator(
    registry: ValidationRegistry,
) -> "CompatibilityValidator":
    """
    Create a compatibility validator for checking parameter combinations.

    Args:
        registry: Base validation registry

    Returns:
        CompatibilityValidator for checking parameter interactions
    """
    return CompatibilityValidator(registry)


class CompatibilityValidator:
    """
    Validator for checking compatibility between configuration parameters.

    This validator checks for known incompatible parameter combinations
    and provides warnings or errors for potentially problematic configurations.
    """

    def __init__(self, base_registry: ValidationRegistry):
        self.base_registry = base_registry
        self.compatibility_rules = self._build_compatibility_rules()

    def _build_compatibility_rules(self) -> List[Dict[str, Any]]:
        """Build list of compatibility rules."""
        return [
            {
                "name": "eager_mode_conflicts",
                "condition": lambda config: config.get("enforce_eager")
                and config.get("enable_prefix_caching"),
                "message": "enforce_eager disables CUDA graphs, which may conflict with prefix caching",
                "severity": "warning",
            },
            {
                "name": "multiple_parallelism",
                "condition": lambda config: (
                    config.get("tensor_parallel_size", 1) > 1
                    and config.get("pipeline_parallel_size", 1) > 1
                ),
                "message": "Using both tensor and pipeline parallelism may not be optimal",
                "severity": "warning",
            },
            {
                "name": "cpu_offload_high_gpu_util",
                "condition": lambda config: (
                    config.get("cpu_offload_gb", 0) > 0
                    and config.get("gpu_memory_utilization", 0.9) > 0.9
                ),
                "message": "CPU offloading with high GPU utilization may cause memory thrashing",
                "severity": "warning",
            },
            {
                "name": "high_parallelism_small_model",
                "condition": lambda config: (
                    config.get("tensor_parallel_size", 1) > 4
                    and "max_model_len" in config
                    and config["max_model_len"] < 8192
                ),
                "message": "High tensor parallelism with small context may be inefficient",
                "severity": "warning",
            },
        ]

    def validate_compatibility(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Check configuration for compatibility issues.

        Args:
            config: Configuration dictionary to check

        Returns:
            List of compatibility issues with severity levels
        """
        issues = []

        for rule in self.compatibility_rules:
            try:
                if rule["condition"](config):
                    issues.append(
                        {
                            "rule": rule["name"],
                            "message": rule["message"],
                            "severity": rule["severity"],
                        }
                    )
            except Exception as e:
                logger.warning(f"Error checking compatibility rule {rule['name']}: {e}")

        return issues


def load_validation_schema_from_file(schema_file: Path) -> ValidationRegistry:
    """
    Load validation schema from a JSON schema file.

    This allows for external configuration of validation rules
    without modifying the code.

    Args:
        schema_file: Path to JSON schema file

    Returns:
        ValidationRegistry built from the schema file
    """
    import json

    registry = ValidationRegistry()

    try:
        with open(schema_file, "r") as f:
            schema_data = json.load(f)

        arguments = schema_data.get("arguments", {})

        for arg_name, arg_info in arguments.items():
            validator = _create_validator_from_schema(arg_name, arg_info)
            if validator:
                registry.register(arg_name, validator)

        logger.info(f"Loaded validation schema from {schema_file}")

    except Exception as e:
        logger.error(f"Failed to load validation schema from {schema_file}: {e}")
        # Fall back to default schema
        registry = create_vllm_validation_registry()

    return registry


def _create_validator_from_schema(
    arg_name: str, arg_info: Dict[str, Any]
) -> Optional[CompositeValidator]:
    """
    Create a validator from schema information.

    Args:
        arg_name: Argument name
        arg_info: Argument information from schema

    Returns:
        CompositeValidator or None if creation fails
    """
    try:
        arg_type = arg_info.get("type")
        validation_info = arg_info.get("validation", {})

        if arg_type == "integer":
            return create_integer_validator(
                arg_name,
                min_value=validation_info.get("min"),
                max_value=validation_info.get("max"),
            )
        elif arg_type == "float":
            return create_float_validator(
                arg_name,
                min_value=validation_info.get("min"),
                max_value=validation_info.get("max"),
            )
        elif arg_type == "string":
            return create_string_validator(
                arg_name,
                min_length=validation_info.get("min_length"),
                max_length=validation_info.get("max_length"),
                pattern=validation_info.get("pattern"),
            )
        elif arg_type == "boolean":
            return create_boolean_validator(arg_name)
        elif arg_type == "choice":
            choices = arg_info.get("choices", [])
            return create_choice_validator(arg_name, choices)
        else:
            logger.warning(f"Unknown argument type for {arg_name}: {arg_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to create validator for {arg_name}: {e}")
        return None
