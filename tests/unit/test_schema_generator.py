"""Unit tests for JSON Schema generation (Phase 6).

Tests for mlserver schema command and schema generation functionality.
These tests are written FIRST as part of test-driven development.
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import will fail until we implement the module
try:
    from mlserver.schema_generator import (
        generate_config_schema,
        generate_multi_classifier_schema,
        save_schema,
        get_schema_for_config_type,
    )
    SCHEMA_MODULE_EXISTS = True
except ImportError:
    SCHEMA_MODULE_EXISTS = False


@pytest.mark.skipif(not SCHEMA_MODULE_EXISTS, reason="Schema module not yet implemented")
class TestSchemaGeneration:
    """Test JSON schema generation from Pydantic models."""

    def test_generate_config_schema_returns_valid_json(self):
        """Test that generate_config_schema returns valid JSON schema."""
        schema = generate_config_schema()

        assert isinstance(schema, dict)
        assert "$schema" in schema or "type" in schema
        assert "properties" in schema or "$defs" in schema

    def test_generate_config_schema_includes_required_sections(self):
        """Test schema includes all required configuration sections."""
        schema = generate_config_schema()

        # Should have definitions or properties for main config sections
        props = schema.get("properties", {})
        defs = schema.get("$defs", schema.get("definitions", {}))

        # Check key sections are defined
        expected_sections = ["server", "predictor", "classifier", "api", "observability"]
        for section in expected_sections:
            assert section in props or any(section in str(defs)), f"Missing section: {section}"

    def test_generate_config_schema_has_descriptions(self):
        """Test that schema fields have descriptions for IDE hints."""
        schema = generate_config_schema()

        # At least some fields should have descriptions
        schema_str = json.dumps(schema)
        assert "description" in schema_str, "Schema should include field descriptions"

    def test_generate_multi_classifier_schema(self):
        """Test schema generation for multi-classifier config."""
        schema = generate_multi_classifier_schema()

        assert isinstance(schema, dict)
        # Multi-classifier should have classifiers section
        props = schema.get("properties", {})
        assert "classifiers" in props or "classifiers" in json.dumps(schema)

    def test_schema_is_valid_json_schema(self):
        """Test that generated schema is valid JSON Schema."""
        schema = generate_config_schema()

        # Basic JSON Schema validation
        assert schema.get("type") == "object" or "$ref" in schema or "anyOf" in schema

        # Should be serializable to JSON
        json_str = json.dumps(schema)
        assert len(json_str) > 100  # Should have substantial content

    def test_get_schema_for_config_type_single(self):
        """Test getting schema for single-classifier config."""
        schema = get_schema_for_config_type("single")

        assert isinstance(schema, dict)
        assert "predictor" in json.dumps(schema)

    def test_get_schema_for_config_type_multi(self):
        """Test getting schema for multi-classifier config."""
        schema = get_schema_for_config_type("multi")

        assert isinstance(schema, dict)
        assert "classifiers" in json.dumps(schema)

    def test_get_schema_for_config_type_auto(self):
        """Test getting combined schema for auto-detection."""
        schema = get_schema_for_config_type("auto")

        assert isinstance(schema, dict)
        # Auto should support both formats
        schema_str = json.dumps(schema)
        # Should have oneOf/anyOf for multiple formats or include both
        assert "predictor" in schema_str or "classifiers" in schema_str


@pytest.mark.skipif(not SCHEMA_MODULE_EXISTS, reason="Schema module not yet implemented")
class TestSchemaSaving:
    """Test saving schema to files."""

    def test_save_schema_to_file(self):
        """Test saving schema to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {"type": "object", "properties": {"test": {"type": "string"}}}
            output_path = Path(tmpdir) / "schema.json"

            save_schema(schema, str(output_path))

            assert output_path.exists()
            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded == schema

    def test_save_schema_creates_directory(self):
        """Test that save_schema creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {"type": "object"}
            output_path = Path(tmpdir) / ".mlserver" / "schema.json"

            save_schema(schema, str(output_path))

            assert output_path.exists()

    def test_save_schema_pretty_prints(self):
        """Test that saved schema is human-readable (indented)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {"type": "object", "properties": {"a": {"type": "string"}}}
            output_path = Path(tmpdir) / "schema.json"

            save_schema(schema, str(output_path))

            with open(output_path) as f:
                content = f.read()
            # Should have newlines (pretty-printed)
            assert "\n" in content
            assert "  " in content  # Indentation


@pytest.mark.skipif(not SCHEMA_MODULE_EXISTS, reason="Schema module not yet implemented")
class TestSchemaContent:
    """Test specific schema content for configuration options."""

    def test_server_config_schema(self):
        """Test server configuration schema has expected fields."""
        schema = generate_config_schema()
        schema_str = json.dumps(schema)

        # Server should have host, port, workers
        assert "host" in schema_str
        assert "port" in schema_str

    def test_predictor_config_schema(self):
        """Test predictor configuration schema has expected fields."""
        schema = generate_config_schema()
        schema_str = json.dumps(schema)

        # Predictor should have module, class_name, init_kwargs
        assert "module" in schema_str
        assert "class_name" in schema_str

    def test_api_config_schema(self):
        """Test API configuration schema has expected fields."""
        schema = generate_config_schema()
        schema_str = json.dumps(schema)

        # API should have adapter, endpoints, etc.
        assert "adapter" in schema_str

    def test_observability_config_schema(self):
        """Test observability configuration schema has expected fields."""
        schema = generate_config_schema()
        schema_str = json.dumps(schema)

        # Observability should have metrics, logging options
        assert "metrics" in schema_str


class TestCLISchemaCommand:
    """Test the CLI schema command."""

    def test_cli_has_schema_command(self):
        """Test that schema command is registered in CLI."""
        from mlserver.cli import app

        command_names = [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]
        assert "schema" in command_names, "CLI should have 'schema' command"

    def test_schema_command_help(self):
        """Test schema command has help text."""
        from typer.testing import CliRunner
        from mlserver.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["schema", "--help"])

        assert result.exit_code == 0
        assert "schema" in result.output.lower() or "json" in result.output.lower()


class TestVSCodeIntegration:
    """Test VSCode yaml-language-server integration."""

    @pytest.mark.skipif(not SCHEMA_MODULE_EXISTS, reason="Schema module not yet implemented")
    def test_schema_includes_yaml_language_server_comment(self):
        """Test that schema output includes VSCode setup hint."""
        schema = generate_config_schema()

        # Schema should have $schema or title for identification
        assert "$schema" in schema or "title" in schema

    def test_schema_url_format(self):
        """Test schema can be referenced by URL or file path."""
        # This is more of a documentation/integration test
        # The schema should work with:
        # # yaml-language-server: $schema=.mlserver/schema.json
        pass  # Documentation-only test
