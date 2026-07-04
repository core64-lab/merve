"""
Multi-Classifier Configuration Support

This module handles loading and managing configurations for repositories
that contain multiple classifiers.
"""

from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from .config import AppConfig, ObservabilityConfig, ServerConfig
from .errors import ConfigurationError


class MultiClassifierConfig(BaseModel):
    """Configuration for repositories with multiple classifiers."""

    # Global configurations (shared by all classifiers)
    server: ServerConfig = ServerConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    # Repository metadata
    repository: dict[str, Any] = Field(default_factory=dict)

    # Multiple classifier configurations
    classifiers: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Default classifier to use when none specified
    default_classifier: Optional[str] = None

    # Deployment configuration
    deployment: Optional[dict[str, Any]] = None


def _normalize_classifiers_section(classifiers: Any) -> Any:
    """Normalize the ``classifiers`` section to dict format.

    Multi-classifier configs are accepted in two YAML shapes:

    Dict format (canonical)::

        classifiers:
          my-classifier:
            predictor: {module: ..., class_name: ...}

    List format::

        classifiers:
          - name: my-classifier
            predictor: {module: ..., class_name: ...}

    List entries are keyed by ``entry["name"]``, falling back to
    ``entry["classifier"]["name"]``. The name key is left in place inside the
    entry so downstream consumers see the entry unchanged.

    Dict input is returned unchanged. Any other type is also returned
    unchanged so pydantic validation can report the type error.

    Raises:
        ConfigurationError: If a list entry has no resolvable name or two
            entries resolve to the same name.
    """
    if not isinstance(classifiers, list):
        return classifiers

    normalized: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(classifiers):
        name = None
        if isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                classifier_meta = entry.get("classifier")
                if isinstance(classifier_meta, dict):
                    name = classifier_meta.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(
                f"Classifier entry #{index + 1} in list-format 'classifiers' "
                f"has no resolvable name",
                suggestion=(
                    "Give each list entry a name:\n"
                    "  classifiers:\n"
                    "    - name: my-classifier\n"
                    "      predictor: {module: my_module, class_name: MyPredictor}\n"
                    "or under the classifier metadata:\n"
                    "    - classifier: {name: my-classifier}\n"
                    "      predictor: {module: my_module, class_name: MyPredictor}\n"
                    "Alternatively use dict format:\n"
                    "  classifiers:\n"
                    "    my-classifier:\n"
                    "      predictor: {module: my_module, class_name: MyPredictor}"
                ),
            )

        if name in normalized:
            raise ConfigurationError(
                f"Duplicate classifier name '{name}' in list-format 'classifiers'",
                suggestion="Give each classifier entry a unique name.",
            )

        normalized[name] = entry

    return normalized


def load_multi_classifier_config(config_file: str) -> MultiClassifierConfig:
    """Load a multi-classifier configuration file.

    Accepts both dict-format and list-format ``classifiers`` sections; list
    format is normalized to a dict keyed by classifier name before validation
    (see :func:`_normalize_classifiers_section`).

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        MultiClassifierConfig object

    Raises:
        ConfigurationError: If a list-format classifier entry has no
            resolvable name.
    """
    with open(config_file) as f:
        raw_config = yaml.safe_load(f)

    if isinstance(raw_config, dict) and "classifiers" in raw_config:
        raw_config["classifiers"] = _normalize_classifiers_section(raw_config["classifiers"])

    return MultiClassifierConfig.model_validate(raw_config)


def extract_single_classifier_config(
    multi_config: MultiClassifierConfig, classifier_name: str
) -> AppConfig:
    """Extract a single classifier configuration from multi-classifier config.

    Args:
        multi_config: The multi-classifier configuration
        classifier_name: Name of the classifier to extract

    Returns:
        AppConfig for the specified classifier

    Raises:
        ValueError: If classifier not found in configuration
        ConfigurationError: If the classifier block lacks a 'predictor' section
    """
    if classifier_name not in multi_config.classifiers:
        available = list(multi_config.classifiers.keys())
        raise ValueError(
            f"Classifier '{classifier_name}' not found. Available classifiers: {available}"
        )

    classifier_config = multi_config.classifiers[classifier_name]

    if "predictor" not in classifier_config:
        raise ConfigurationError(
            f"Classifier '{classifier_name}' is missing the required 'predictor' section",
            suggestion=(
                f"Add a predictor block for '{classifier_name}' in your "
                f"multi-classifier config:\n"
                f"  classifiers:\n"
                f"    {classifier_name}:\n"
                f"      predictor:\n"
                f"        module: my_module\n"
                f"        class_name: MyPredictor"
            ),
        )

    # Build AppConfig combining global and classifier-specific settings
    app_config_dict = {
        # Use global server and observability settings
        "server": multi_config.server.model_dump(),
        "observability": multi_config.observability.model_dump(),
        # Use classifier-specific predictor settings
        "predictor": classifier_config["predictor"],
        # Use classifier metadata (handle both old 'metadata' and new 'classifier' keys)
        "classifier": {
            **classifier_config.get("classifier", classifier_config.get("metadata", {})),
            "repository": multi_config.repository.get("name", "unknown"),
        },
        # Use classifier-specific API settings
        "api": classifier_config.get("api", {}),
        # Model metadata if available (for backward compatibility)
        "model": classifier_config.get("model", {}),
        # Build configuration if available
        "build": classifier_config.get("build"),
    }

    return AppConfig.model_validate(app_config_dict)


def list_available_classifiers(config_file: str) -> list[str]:
    """List all available classifiers in a configuration file.

    Args:
        config_file: Path to the configuration file

    Returns:
        List of classifier names
    """
    try:
        with open(config_file) as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict) or "classifiers" not in raw_config:
            return []

        # Same normalization as load_multi_classifier_config, so all entry
        # points agree on names (handles both dict and list formats)
        classifiers = _normalize_classifiers_section(raw_config["classifiers"])

        if isinstance(classifiers, dict):
            return list(classifiers.keys())

        return []
    except Exception:
        # Fall back to single classifier (no list available)
        return []


def detect_multi_classifier_config(config_file: str) -> bool:
    """Detect if a configuration file is multi-classifier format.

    Args:
        config_file: Path to the configuration file

    Returns:
        True if multi-classifier format, False otherwise
    """
    try:
        with open(config_file) as f:
            raw_config = yaml.safe_load(f)

        # Check for multi-classifier structure (supports both dict and list formats)
        return "classifiers" in raw_config and (
            isinstance(raw_config["classifiers"], dict)
            or isinstance(raw_config["classifiers"], list)
        )
    except Exception:
        return False


def get_default_classifier(config_file: str) -> Optional[str]:
    """Get the default classifier from a multi-classifier config.

    Args:
        config_file: Path to the configuration file

    Returns:
        Default classifier name or None
    """
    try:
        multi_config = load_multi_classifier_config(config_file)

        # Return explicitly set default
        if multi_config.default_classifier:
            return multi_config.default_classifier

        # Return first classifier as fallback
        if multi_config.classifiers:
            return list(multi_config.classifiers.keys())[0]

        return None
    except Exception:
        return None
