"""Unit tests for multi_classifier module."""
import pytest
import tempfile
import yaml
from pathlib import Path

from mlserver.multi_classifier import (
    MultiClassifierConfig,
    load_multi_classifier_config,
    extract_single_classifier_config,
    list_available_classifiers,
    detect_multi_classifier_config,
    get_default_classifier,
    generate_dockerfile_for_classifier,
    build_all_classifiers,
)
from mlserver.config import AppConfig


@pytest.fixture
def multi_classifier_yaml():
    """Return a valid multi-classifier YAML config."""
    return """
server:
  host: "0.0.0.0"
  port: 8000

observability:
  metrics: true
  structured_logging: true

repository:
  name: ml-classifiers
  version: "1.0.0"

classifiers:
  sentiment:
    predictor:
      module: sentiment_predictor
      class_name: SentimentPredictor
    classifier:
      name: sentiment-classifier
      version: "1.0.0"
    api:
      adapter: records

  fraud:
    predictor:
      module: fraud_predictor
      class_name: FraudPredictor
    classifier:
      name: fraud-detector
      version: "2.0.0"
    api:
      adapter: ndarray

default_classifier: sentiment
"""


@pytest.fixture
def multi_config_file(multi_classifier_yaml):
    """Create a temporary multi-classifier config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(multi_classifier_yaml)
        return f.name


class TestMultiClassifierConfig:
    """Test MultiClassifierConfig model."""

    def test_minimal_config(self):
        """Test minimal multi-classifier config."""
        config = MultiClassifierConfig()
        assert config.classifiers == {}
        assert config.default_classifier is None

    def test_full_config(self, multi_classifier_yaml):
        """Test loading full config."""
        config_dict = yaml.safe_load(multi_classifier_yaml)
        config = MultiClassifierConfig.model_validate(config_dict)

        assert config.server.port == 8000
        assert "sentiment" in config.classifiers
        assert "fraud" in config.classifiers
        assert config.default_classifier == "sentiment"


class TestLoadMultiClassifierConfig:
    """Test load_multi_classifier_config function."""

    def test_load_from_file(self, multi_config_file):
        """Test loading config from file."""
        config = load_multi_classifier_config(multi_config_file)

        assert isinstance(config, MultiClassifierConfig)
        assert "sentiment" in config.classifiers
        assert "fraud" in config.classifiers

    def test_load_nonexistent_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_multi_classifier_config("/nonexistent/path.yaml")


class TestExtractSingleClassifierConfig:
    """Test extract_single_classifier_config function."""

    def test_extract_valid_classifier(self, multi_config_file):
        """Test extracting a valid classifier config."""
        multi_config = load_multi_classifier_config(multi_config_file)
        app_config = extract_single_classifier_config(multi_config, "sentiment")

        assert isinstance(app_config, AppConfig)
        assert app_config.predictor.module == "sentiment_predictor"
        assert app_config.predictor.class_name == "SentimentPredictor"
        assert app_config.api.adapter == "records"

    def test_extract_second_classifier(self, multi_config_file):
        """Test extracting a different classifier."""
        multi_config = load_multi_classifier_config(multi_config_file)
        app_config = extract_single_classifier_config(multi_config, "fraud")

        assert app_config.predictor.module == "fraud_predictor"
        assert app_config.api.adapter == "ndarray"

    def test_extract_nonexistent_classifier(self, multi_config_file):
        """Test error when classifier doesn't exist."""
        multi_config = load_multi_classifier_config(multi_config_file)

        with pytest.raises(ValueError) as exc_info:
            extract_single_classifier_config(multi_config, "nonexistent")

        assert "not found" in str(exc_info.value)
        assert "sentiment" in str(exc_info.value)  # Should list available classifiers

    def test_extracted_config_has_repository_info(self, multi_config_file):
        """Test that extracted config includes repository info."""
        multi_config = load_multi_classifier_config(multi_config_file)
        app_config = extract_single_classifier_config(multi_config, "sentiment")

        assert app_config.classifier["repository"] == "ml-classifiers"


class TestListAvailableClassifiers:
    """Test list_available_classifiers function."""

    def test_list_dict_format(self, multi_config_file):
        """Test listing classifiers from dict format."""
        classifiers = list_available_classifiers(multi_config_file)

        assert "sentiment" in classifiers
        assert "fraud" in classifiers
        assert len(classifiers) == 2

    def test_list_from_list_format(self):
        """Test listing classifiers from list format."""
        yaml_content = """
classifiers:
  - name: classifier-a
    predictor:
      module: mod_a
      class_name: ClassA
  - classifier:
      name: classifier-b
    predictor:
      module: mod_b
      class_name: ClassB
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        classifiers = list_available_classifiers(config_file)
        assert "classifier-a" in classifiers
        assert "classifier-b" in classifiers

    def test_list_empty_config(self):
        """Test listing classifiers from config without classifiers."""
        yaml_content = """
server:
  port: 8000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        classifiers = list_available_classifiers(config_file)
        assert classifiers == []

    def test_list_from_invalid_file(self):
        """Test listing from invalid/nonexistent file."""
        classifiers = list_available_classifiers("/nonexistent/path.yaml")
        assert classifiers == []


class TestDetectMultiClassifierConfig:
    """Test detect_multi_classifier_config function."""

    def test_detect_multi_classifier(self, multi_config_file):
        """Test detecting multi-classifier config."""
        assert detect_multi_classifier_config(multi_config_file) is True

    def test_detect_single_classifier(self):
        """Test detecting single classifier config."""
        yaml_content = """
predictor:
  module: my_predictor
  class_name: MyPredictor
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        assert detect_multi_classifier_config(config_file) is False

    def test_detect_invalid_file(self):
        """Test detecting from invalid file."""
        assert detect_multi_classifier_config("/nonexistent/path.yaml") is False


class TestGetDefaultClassifier:
    """Test get_default_classifier function."""

    def test_get_explicit_default(self, multi_config_file):
        """Test getting explicit default classifier."""
        default = get_default_classifier(multi_config_file)
        assert default == "sentiment"

    def test_get_fallback_default(self):
        """Test getting first classifier as fallback."""
        yaml_content = """
classifiers:
  first-classifier:
    predictor:
      module: mod
      class_name: Class
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        default = get_default_classifier(config_file)
        assert default == "first-classifier"

    def test_get_default_empty_config(self):
        """Test getting default from empty config."""
        yaml_content = """
server:
  port: 8000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        default = get_default_classifier(config_file)
        assert default is None

    def test_get_default_invalid_file(self):
        """Test getting default from invalid file."""
        default = get_default_classifier("/nonexistent/path.yaml")
        assert default is None


class TestGenerateDockerfile:
    """Test generate_dockerfile_for_classifier function."""

    def test_generate_dockerfile(self, multi_config_file):
        """Test generating Dockerfile for classifier."""
        multi_config = load_multi_classifier_config(multi_config_file)

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = generate_dockerfile_for_classifier(
                multi_config, "sentiment", tmpdir
            )

            assert Path(dockerfile_path).exists()
            content = Path(dockerfile_path).read_text()
            assert "sentiment" in content
            assert "CMD" in content

    def test_generate_dockerfile_nonexistent_classifier(self, multi_config_file):
        """Test error when classifier doesn't exist."""
        multi_config = load_multi_classifier_config(multi_config_file)

        with pytest.raises(ValueError):
            generate_dockerfile_for_classifier(multi_config, "nonexistent")

    def test_generate_dockerfile_with_custom_template(self):
        """Test generating Dockerfile with custom template."""
        yaml_content = """
classifiers:
  custom:
    predictor:
      module: mod
      class_name: Class
    build:
      dockerfile_template: |
        FROM custom-image:latest
        RUN echo "custom"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        multi_config = load_multi_classifier_config(config_file)

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = generate_dockerfile_for_classifier(
                multi_config, "custom", tmpdir
            )

            content = Path(dockerfile_path).read_text()
            assert "custom-image:latest" in content


class TestBuildAllClassifiers:
    """Test build_all_classifiers function."""

    def test_build_all(self, multi_config_file, capsys):
        """Test building all classifiers."""
        results = build_all_classifiers(multi_config_file)

        assert "sentiment" in results
        assert "fraud" in results
        assert results["sentiment"]["status"] == "pending"
        assert results["fraud"]["status"] == "pending"

        # Check output
        captured = capsys.readouterr()
        assert "sentiment" in captured.out
        assert "fraud" in captured.out

    def test_build_all_with_registry(self, multi_config_file):
        """Test building all with registry option."""
        results = build_all_classifiers(multi_config_file, registry="ghcr.io/org")

        assert results["sentiment"]["registry"] == "ghcr.io/org"
        assert results["fraud"]["registry"] == "ghcr.io/org"


class TestMultiClassifierEdgeCases:
    """Test edge cases in multi-classifier handling."""

    def test_classifier_with_metadata_key(self):
        """Test classifier using 'metadata' instead of 'classifier' key."""
        yaml_content = """
repository:
  name: test-repo

classifiers:
  legacy:
    predictor:
      module: mod
      class_name: Class
    metadata:
      name: legacy-classifier
      version: "1.0.0"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        multi_config = load_multi_classifier_config(config_file)
        app_config = extract_single_classifier_config(multi_config, "legacy")

        # Should use metadata key as fallback
        assert app_config.classifier["name"] == "legacy-classifier"

    def test_classifier_without_api_section(self):
        """Test classifier without explicit API section."""
        yaml_content = """
classifiers:
  minimal:
    predictor:
      module: mod
      class_name: Class
    classifier:
      name: minimal-classifier
      version: "1.0.0"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        multi_config = load_multi_classifier_config(config_file)
        app_config = extract_single_classifier_config(multi_config, "minimal")

        # Should have default API config
        assert app_config.api is not None
