"""Basic tests for CLI module functionality."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.cli import (
    resolve_relative_paths,
    _is_likely_file_path,
    _resolve_path,
)


class TestPathResolution:
    """Test path resolution utilities."""

    def test_is_likely_file_path(self):
        """Test file path detection."""
        # Should identify as file paths (relative paths with path indicators)
        assert _is_likely_file_path("model_path", "./model.pkl")
        assert _is_likely_file_path("config_file", "../config.yaml")
        assert _is_likely_file_path("artifact_path", "artifacts/model.joblib")

        # Should not identify as file paths (absolute paths or non-path keys)
        assert not _is_likely_file_path("config_file", "/path/to/config.yaml")  # absolute path
        assert not _is_likely_file_path("port", "8000")
        assert not _is_likely_file_path("name", "my_model")
        assert not _is_likely_file_path("debug", "true")

    def test_resolve_path_relative(self, temp_dir):
        """Test resolving relative paths."""
        config_dir = temp_dir
        relative_path = "./model.pkl"

        resolved = _resolve_path(relative_path, config_dir)
        expected = str(Path(config_dir) / "model.pkl")
        assert resolved == expected

    def test_resolve_path_absolute(self):
        """Test absolute paths remain unchanged."""
        config_dir = "/some/config/dir"
        absolute_path = "/absolute/path/model.pkl"

        resolved = _resolve_path(absolute_path, config_dir)
        assert resolved == absolute_path

    def test_resolve_relative_paths_dict(self, temp_dir):
        """Test resolving paths in init_kwargs dictionary."""
        config_dir = temp_dir
        init_kwargs = {
            "model_path": "./model.pkl",
            "preprocessor_path": "../preprocessor.pkl",
            "config_file": "/absolute/config.yaml",
            "port": "8000",  # Should not be changed
            "name": "test_model"  # Should not be changed
        }

        resolved = resolve_relative_paths(init_kwargs, config_dir)

        # File paths should be resolved
        assert resolved["model_path"] == str(Path(config_dir) / "model.pkl")
        assert resolved["preprocessor_path"] == str(Path(config_dir).parent / "preprocessor.pkl")
        assert resolved["config_file"] == "/absolute/config.yaml"  # Absolute unchanged

        # Non-file paths should remain unchanged
        assert resolved["port"] == "8000"
        assert resolved["name"] == "test_model"


class TestTyperCLI:
    """Test the Typer-based CLI (modern interface)."""

    def _get_command_names(self):
        """Helper to get all registered command names."""
        from mlserver.cli import app
        # Typer commands: name can be None, fall back to callback.__name__
        return [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]

    def test_cli_app_exists(self):
        """Test that the Typer app is properly defined."""
        from mlserver.cli import app
        assert app is not None

    def test_cli_has_serve_command(self):
        """Test that serve command is registered."""
        assert "serve" in self._get_command_names()

    def test_cli_has_validate_command(self):
        """Test that validate command is registered."""
        assert "validate" in self._get_command_names()

    def test_cli_has_doctor_command(self):
        """Test that doctor command is registered."""
        assert "doctor" in self._get_command_names()

    def test_cli_has_build_command(self):
        """Test that build command is registered."""
        assert "build" in self._get_command_names()

    def test_cli_has_version_command(self):
        """Test that version command is registered."""
        assert "version" in self._get_command_names()

    def test_cli_has_test_command(self):
        """Test that test command is registered (Phase 4)."""
        assert "test" in self._get_command_names()

    def test_cli_has_tag_command(self):
        """Test that tag command is registered."""
        assert "tag" in self._get_command_names()

    def test_cli_has_push_command(self):
        """Test that push command is registered."""
        assert "push" in self._get_command_names()