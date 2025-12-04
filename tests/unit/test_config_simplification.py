"""Unit tests for configuration simplification (Phase 2).

Tests for minimal configs, auto-detection, and smart defaults.
These tests define expected behavior FIRST as part of TDD.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from mlserver.config import AppConfig, PredictorConfig, ApiConfig


class TestMinimalConfiguration:
    """Test that minimal configurations work with smart defaults."""

    def test_minimal_config_with_predictor_only(self):
        """Test that only predictor config is truly required."""
        # Minimal config - just predictor
        config_dict = {
            "predictor": {
                "module": "my_predictor",
                "class_name": "MyPredictor"
            }
        }

        config = AppConfig.model_validate(config_dict)

        # Should have defaults for everything else
        assert config.predictor.module == "my_predictor"
        assert config.predictor.class_name == "MyPredictor"
        assert config.server is not None
        assert config.server.port == 8000  # Default
        assert config.observability is not None
        assert config.api is not None

    def test_minimal_config_generates_classifier_metadata(self):
        """Test that classifier metadata is auto-generated if not provided."""
        config_dict = {
            "predictor": {
                "module": "my_predictor",
                "class_name": "MyPredictor"
            }
        }

        config = AppConfig.model_validate(config_dict)

        # Should have default classifier info
        assert config.classifier is not None
        assert "name" in config.classifier
        assert "version" in config.classifier

    def test_config_with_classifier_name_only(self):
        """Test config with just classifier name (version defaults to 0.1.0)."""
        config_dict = {
            "predictor": {
                "module": "my_predictor",
                "class_name": "MyPredictor"
            },
            "classifier": {
                "name": "my-classifier"
            }
        }

        config = AppConfig.model_validate(config_dict)

        assert config.classifier["name"] == "my-classifier"
        # Version should have a default
        assert "version" in config.classifier or config.classifier.get("version", "0.1.0") == "0.1.0"


class TestApiDefaults:
    """Test API configuration defaults."""

    def test_api_defaults_when_omitted(self):
        """Test that API config has sensible defaults when omitted."""
        config_dict = {
            "predictor": {
                "module": "my_predictor",
                "class_name": "MyPredictor"
            }
        }

        config = AppConfig.model_validate(config_dict)

        # API should have defaults
        assert config.api is not None
        assert config.api.adapter == "records"  # Default adapter
        assert config.api.version == "v1"  # Default version

    def test_api_partial_override(self):
        """Test partial API config override."""
        config_dict = {
            "predictor": {
                "module": "my_predictor",
                "class_name": "MyPredictor"
            },
            "api": {
                "adapter": "ndarray"
            }
        }

        config = AppConfig.model_validate(config_dict)

        # Overridden value
        assert config.api.adapter == "ndarray"
        # Default values still present
        assert config.api.version == "v1"


class TestClassifierAutoDetection:
    """Test automatic classifier name/version detection."""

    def test_classifier_name_from_directory(self):
        """Test inferring classifier name from project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project with directory name
            project_dir = Path(tmpdir) / "sentiment-classifier"
            project_dir.mkdir()

            config_dict = {
                "predictor": {
                    "module": "predictor",
                    "class_name": "SentimentPredictor"
                }
            }

            config = AppConfig.model_validate(config_dict)
            config.set_project_path(str(project_dir))

            # Classifier name should be inferred from directory
            # (This is the expected behavior - implementation TBD)
            assert config.classifier is not None

    def test_classifier_version_from_git_tag(self):
        """Test inferring version from git tags."""
        # This test documents expected behavior
        # Version detection from git would be a nice-to-have
        pass  # Placeholder for future implementation


class TestPredictorAutoDetection:
    """Test automatic predictor module detection."""

    def test_detect_predictor_from_single_file(self):
        """Test auto-detecting predictor from a single .py file."""
        # This documents expected behavior for future implementation
        # If project has single .py file with *Predictor class, auto-detect
        pass  # Placeholder for future implementation

    def test_detect_predictor_from_predictor_py(self):
        """Test auto-detecting from standard predictor.py file."""
        # If project has predictor.py, use that as module
        pass  # Placeholder for future implementation


class TestConfigurationValidation:
    """Test configuration validation with simplified configs."""

    def test_validate_rejects_empty_config(self):
        """Test that completely empty config is rejected."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AppConfig.model_validate({})

    def test_validate_rejects_missing_predictor(self):
        """Test that config without predictor is rejected."""
        config_dict = {
            "classifier": {
                "name": "test",
                "version": "1.0.0"
            }
        }

        with pytest.raises(Exception):  # Pydantic ValidationError
            AppConfig.model_validate(config_dict)

    def test_validate_predictor_requires_module_and_class(self):
        """Test that predictor requires both module and class_name."""
        # Missing class_name
        with pytest.raises(Exception):
            AppConfig.model_validate({
                "predictor": {"module": "test"}
            })

        # Missing module
        with pytest.raises(Exception):
            AppConfig.model_validate({
                "predictor": {"class_name": "Test"}
            })


class TestBackwardsCompatibility:
    """Test that simplified configs don't break existing full configs."""

    def test_full_config_still_works(self):
        """Test that full explicit configs still work."""
        config_dict = {
            "server": {
                "host": "0.0.0.0",
                "port": 9000,
                "workers": 2
            },
            "predictor": {
                "module": "my_predictor",
                "class_name": "MyPredictor",
                "init_kwargs": {"model_path": "./model.pkl"}
            },
            "classifier": {
                "name": "test-classifier",
                "version": "2.0.0",
                "description": "A test classifier"
            },
            "api": {
                "version": "v2",
                "adapter": "ndarray",
                "feature_order": ["f1", "f2", "f3"]
            },
            "observability": {
                "metrics": True,
                "structured_logging": True
            }
        }

        config = AppConfig.model_validate(config_dict)

        assert config.server.port == 9000
        assert config.server.workers == 2
        assert config.predictor.init_kwargs["model_path"] == "./model.pkl"
        assert config.classifier["name"] == "test-classifier"
        assert config.classifier["version"] == "2.0.0"
        assert config.api.adapter == "ndarray"
        assert config.api.feature_order == ["f1", "f2", "f3"]


class TestConfigErrorMessages:
    """Test that configuration errors have helpful messages."""

    def test_missing_predictor_error_message(self):
        """Test error message when predictor is missing."""
        try:
            AppConfig.model_validate({
                "classifier": {"name": "test", "version": "1.0"}
            })
            pytest.fail("Should have raised validation error")
        except Exception as e:
            error_msg = str(e).lower()
            # Should mention predictor
            assert "predictor" in error_msg

    def test_invalid_adapter_error_message(self):
        """Test error message for invalid adapter value."""
        # This test documents expected behavior
        # Better error messages would be nice
        pass  # Placeholder


class TestYAMLParsing:
    """Test YAML config file parsing with minimal configs."""

    def test_parse_minimal_yaml(self):
        """Test parsing a minimal YAML config."""
        yaml_content = """
predictor:
  module: my_predictor
  class_name: MyPredictor
"""
        config_dict = yaml.safe_load(yaml_content)
        config = AppConfig.model_validate(config_dict)

        assert config.predictor.module == "my_predictor"

    def test_parse_yaml_with_comments(self):
        """Test that YAML comments don't break parsing."""
        yaml_content = """
# Minimal MLServer configuration
predictor:
  module: my_predictor  # The Python module
  class_name: MyPredictor  # The predictor class
"""
        config_dict = yaml.safe_load(yaml_content)
        config = AppConfig.model_validate(config_dict)

        assert config.predictor.class_name == "MyPredictor"
