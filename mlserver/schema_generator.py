"""JSON Schema generation for MLServer configuration files.

This module generates JSON schemas from Pydantic models to enable
IDE autocompletion and validation for mlserver.yaml files.

Phase 6: IDE Support Implementation
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from .config import AppConfig, ServerConfig, PredictorConfig, ApiConfig, ObservabilityConfig
from .multi_classifier import MultiClassifierConfig


def generate_config_schema() -> Dict[str, Any]:
    """Generate JSON schema for single-classifier AppConfig.

    Returns:
        Dict containing valid JSON Schema for mlserver.yaml
    """
    schema = AppConfig.model_json_schema()

    # Add metadata for IDE integration
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = "MLServer Configuration"
    schema["description"] = (
        "Configuration schema for mlserver.yaml. "
        "Supports single-classifier deployments with predictor, server, and API settings."
    )

    return schema


def generate_multi_classifier_schema() -> Dict[str, Any]:
    """Generate JSON schema for multi-classifier configuration.

    Returns:
        Dict containing valid JSON Schema for multi-classifier mlserver.yaml
    """
    schema = MultiClassifierConfig.model_json_schema()

    # Add metadata
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = "MLServer Multi-Classifier Configuration"
    schema["description"] = (
        "Configuration schema for multi-classifier mlserver.yaml. "
        "Supports deploying multiple classifiers from a single configuration file."
    )

    return schema


def generate_combined_schema() -> Dict[str, Any]:
    """Generate a combined schema supporting both single and multi-classifier configs.

    Returns:
        Dict containing JSON Schema with oneOf for both formats
    """
    single_schema = AppConfig.model_json_schema()
    multi_schema = MultiClassifierConfig.model_json_schema()

    # Merge definitions
    all_defs = {}
    all_defs.update(single_schema.get("$defs", {}))
    all_defs.update(multi_schema.get("$defs", {}))

    combined = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "MLServer Configuration",
        "description": (
            "Configuration schema for mlserver.yaml. "
            "Supports both single-classifier and multi-classifier deployments."
        ),
        "oneOf": [
            {
                "title": "Single Classifier",
                "description": "Configuration for a single classifier deployment",
                "type": "object",
                "properties": single_schema.get("properties", {}),
                "required": single_schema.get("required", []),
            },
            {
                "title": "Multi-Classifier",
                "description": "Configuration for multiple classifiers",
                "type": "object",
                "properties": multi_schema.get("properties", {}),
                "required": multi_schema.get("required", []),
            }
        ],
        "$defs": all_defs
    }

    return combined


def get_schema_for_config_type(
    config_type: Literal["single", "multi", "auto"] = "auto"
) -> Dict[str, Any]:
    """Get JSON schema for the specified configuration type.

    Args:
        config_type: Type of configuration schema to generate
            - "single": Single-classifier config (AppConfig)
            - "multi": Multi-classifier config (MultiClassifierConfig)
            - "auto": Combined schema supporting both formats

    Returns:
        Dict containing the JSON Schema
    """
    if config_type == "single":
        return generate_config_schema()
    elif config_type == "multi":
        return generate_multi_classifier_schema()
    else:  # auto
        return generate_combined_schema()


def save_schema(schema: Dict[str, Any], output_path: str) -> None:
    """Save JSON schema to a file.

    Args:
        schema: The JSON schema dict to save
        output_path: Path where to save the schema file
    """
    path = Path(output_path)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write with pretty-printing for human readability
    with open(path, 'w') as f:
        json.dump(schema, f, indent=2, sort_keys=False)


def get_vscode_yaml_comment(schema_path: str = ".mlserver/schema.json") -> str:
    """Get the YAML comment for VSCode yaml-language-server integration.

    Args:
        schema_path: Relative path to the schema file

    Returns:
        String comment to add at top of mlserver.yaml
    """
    return f"# yaml-language-server: $schema={schema_path}"


def get_vscode_settings_snippet(schema_path: str = ".mlserver/schema.json") -> Dict[str, Any]:
    """Get VSCode settings.json snippet for automatic schema association.

    Args:
        schema_path: Path to the schema file (relative or absolute)

    Returns:
        Dict that can be added to .vscode/settings.json
    """
    return {
        "yaml.schemas": {
            schema_path: ["mlserver.yaml", "mlserver.yml", "**/mlserver.yaml"]
        }
    }


def print_schema_setup_instructions(schema_path: str = ".mlserver/schema.json") -> str:
    """Generate setup instructions for IDE integration.

    Args:
        schema_path: Path where schema was saved

    Returns:
        Human-readable setup instructions
    """
    return f"""
╭─────────────────────────────────────────────────────────────────╮
│  JSON Schema generated successfully!                            │
╰─────────────────────────────────────────────────────────────────╯

Schema saved to: {schema_path}

┌─────────────────────────────────────────────────────────────────┐
│  VSCode Setup (Option 1 - Per-file)                             │
└─────────────────────────────────────────────────────────────────┘
Add this comment to the top of your mlserver.yaml:

  # yaml-language-server: $schema={schema_path}

┌─────────────────────────────────────────────────────────────────┐
│  VSCode Setup (Option 2 - Project-wide)                         │
└─────────────────────────────────────────────────────────────────┘
Add to .vscode/settings.json:

  {{
    "yaml.schemas": {{
      "{schema_path}": ["mlserver.yaml", "**/mlserver.yaml"]
    }}
  }}

┌─────────────────────────────────────────────────────────────────┐
│  Requirements                                                    │
└─────────────────────────────────────────────────────────────────┘
Install the YAML extension for VSCode:
  - Extension ID: redhat.vscode-yaml
  - Or search "YAML" in VSCode extensions

Once configured, you'll get:
  ✓ Autocomplete for all configuration options
  ✓ Inline documentation on hover
  ✓ Validation errors for invalid values
  ✓ Type checking for nested objects
"""
