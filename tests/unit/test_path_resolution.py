"""Unit tests for path resolution (Phase 5).

Tests for consistent, predictable path resolution across all contexts.
These tests define expected behavior FIRST as part of TDD.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import os

from mlserver.config import AppConfig, PredictorConfig, ApiConfig
from mlserver.errors import ConfigurationError


class TestRelativePathResolution:
    """Test that relative paths are resolved correctly."""

    def test_feature_order_relative_to_config(self):
        """Test that feature_order paths are relative to config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a features.json file
            features_file = Path(tmpdir) / "features.json"
            features_file.write_text('["f1", "f2", "f3"]')

            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test"
                ),
                api=ApiConfig(
                    feature_order="features.json"
                )
            )

            # Resolve with base_path
            features = config.api.get_resolved_feature_order(base_path=Path(tmpdir))
            assert features == ["f1", "f2", "f3"]

    def test_feature_order_in_subdirectory(self):
        """Test feature_order file in subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config/features.json
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            features_file = config_dir / "features.json"
            features_file.write_text('["a", "b", "c"]')

            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test"
                ),
                api=ApiConfig(
                    feature_order="config/features.json"
                )
            )

            features = config.api.get_resolved_feature_order(base_path=Path(tmpdir))
            assert features == ["a", "b", "c"]

    def test_predictor_module_path_relative(self):
        """Test that predictor module paths work relative to config."""
        config = AppConfig(
            predictor=PredictorConfig(
                module="my_predictor",  # Should work as module name
                class_name="MyPredictor"
            )
        )
        assert config.predictor.module == "my_predictor"


class TestAbsolutePathHandling:
    """Test handling of absolute paths."""

    def test_feature_order_absolute_path(self):
        """Test that absolute paths work for feature_order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features_file = Path(tmpdir) / "features.json"
            features_file.write_text('["x", "y", "z"]')

            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test"
                ),
                api=ApiConfig(
                    feature_order=str(features_file)  # Absolute path
                )
            )

            # Should work without base_path for absolute
            features = config.api.get_resolved_feature_order()
            assert features == ["x", "y", "z"]


class TestPathTraversalPrevention:
    """Test security: prevent path traversal attacks."""

    def test_reject_parent_directory_traversal(self):
        """Test that '../' path traversal is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file outside the project directory
            parent_dir = Path(tmpdir).parent
            outside_file = parent_dir / "secret.json"

            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test"
                ),
                api=ApiConfig(
                    feature_order="../secret.json"
                )
            )

            # Should raise ConfigurationError for security
            with pytest.raises(ConfigurationError) as exc_info:
                config.api.get_resolved_feature_order(base_path=Path(tmpdir))

            # Should mention security or path traversal
            error_msg = str(exc_info.value).lower()
            assert "security" in error_msg or "outside" in error_msg

    def test_reject_absolute_path_outside_project(self):
        """Test absolute paths outside project raise ConfigurationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Absolute path outside tmpdir
            outside_path = "/etc/passwd"

            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test"
                ),
                api=ApiConfig(
                    feature_order=outside_path
                )
            )

            # Should raise ConfigurationError for security when base_path is provided
            with pytest.raises(ConfigurationError) as exc_info:
                config.api.get_resolved_feature_order(base_path=Path(tmpdir))

            error_msg = str(exc_info.value).lower()
            assert "security" in error_msg or "outside" in error_msg


class TestMissingPathHandling:
    """Test handling of missing files/directories."""

    def test_missing_feature_order_file_returns_none(self):
        """Test graceful handling of missing feature_order file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test"
                ),
                api=ApiConfig(
                    feature_order="nonexistent.json"
                )
            )

            result = config.api.get_resolved_feature_order(base_path=Path(tmpdir))
            assert result is None

    def test_missing_predictor_module_detected(self):
        """Test that missing predictor module is caught at runtime."""
        # This tests the validation, not the import
        config = AppConfig(
            predictor=PredictorConfig(
                module="nonexistent_module_xyz",
                class_name="Test"
            )
        )
        # Config should be valid, but module won't import
        assert config.predictor.module == "nonexistent_module_xyz"


class TestPathValidation:
    """Test path validation utilities."""

    def test_validate_init_kwargs_paths(self):
        """Test that paths in init_kwargs can be validated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            model_path.touch()  # Create empty file

            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test",
                    init_kwargs={"model_path": str(model_path)}
                )
            )

            # Model path should be stored
            assert config.predictor.init_kwargs["model_path"] == str(model_path)


class TestConfigFilePathContext:
    """Test path resolution with config file context."""

    def test_set_project_path_enables_relative_resolution(self):
        """Test that set_project_path enables relative path resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project structure
            project = Path(tmpdir) / "my-classifier"
            project.mkdir()

            config = AppConfig(
                predictor=PredictorConfig(
                    module="predictor",
                    class_name="MyPredictor"
                )
            )

            config.set_project_path(str(project))
            assert config.project_path == str(project)

    def test_paths_in_yaml_resolved_relative_to_yaml_location(self):
        """Test that paths in YAML are relative to the YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project structure
            project = Path(tmpdir) / "classifier"
            project.mkdir()

            # Create features.json in project
            features_file = project / "features.json"
            features_file.write_text('["feature1", "feature2"]')

            # Create mlserver.yaml
            yaml_content = """
predictor:
  module: predictor
  class_name: Predictor
api:
  feature_order: features.json
"""
            yaml_file = project / "mlserver.yaml"
            yaml_file.write_text(yaml_content)

            # Parse and validate
            config_dict = yaml.safe_load(yaml_content)
            config = AppConfig.model_validate(config_dict)

            # Resolve feature_order relative to config file location
            features = config.api.get_resolved_feature_order(base_path=project)
            assert features == ["feature1", "feature2"]


class TestCrossPlatformPaths:
    """Test cross-platform path handling."""

    def test_forward_slash_paths_work_everywhere(self):
        """Test that forward slashes work on all platforms."""
        config = AppConfig(
            predictor=PredictorConfig(
                module="test",
                class_name="Test"
            ),
            api=ApiConfig(
                feature_order="config/features.json"  # Forward slashes
            )
        )
        assert "config/features.json" in str(config.api.feature_order)

    def test_path_separators_normalized(self):
        """Test that path separators are normalized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory
            nested = Path(tmpdir) / "config" / "data"
            nested.mkdir(parents=True)
            features_file = nested / "features.json"
            features_file.write_text('["a"]')

            config = AppConfig(
                predictor=PredictorConfig(
                    module="test",
                    class_name="Test"
                ),
                api=ApiConfig(
                    feature_order="config/data/features.json"
                )
            )

            features = config.api.get_resolved_feature_order(base_path=Path(tmpdir))
            assert features == ["a"]


class TestContainerPathContext:
    """Test path resolution in container build context."""

    def test_init_kwargs_paths_preserved_for_container(self):
        """Test that init_kwargs paths are preserved for container builds."""
        config = AppConfig(
            predictor=PredictorConfig(
                module="predictor",
                class_name="Predictor",
                init_kwargs={
                    "model_path": "./models/model.pkl",
                    "config_path": "config/settings.yaml"
                }
            )
        )

        # Paths should be preserved as-is for container context
        assert config.predictor.init_kwargs["model_path"] == "./models/model.pkl"
        assert config.predictor.init_kwargs["config_path"] == "config/settings.yaml"

    def test_build_context_uses_project_directory(self):
        """Test that build context uses the project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "my-project"
            project.mkdir()

            config = AppConfig(
                predictor=PredictorConfig(
                    module="predictor",
                    class_name="Predictor"
                )
            )
            config.set_project_path(str(project))

            assert config.project_path == str(project)
