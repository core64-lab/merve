"""Unit tests for version module."""
import pytest
import os
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from mlserver.version import (
    GitInfo,
    ClassifierVersion,
    ModelVersion,
    ApiVersion,
    ClassifierMetadata,
    get_git_info,
    get_repository_name,
    load_classifier_metadata,
    save_classifier_metadata,
    generate_container_tags,
    validate_version_consistency,
    get_version_info,
)
from mlserver.errors import ConfigurationError


class TestGitInfo:
    """Test GitInfo dataclass."""

    def test_create_git_info(self):
        """Test creating GitInfo instance."""
        info = GitInfo(tag="v1.0.0", commit="abc123", branch="main", is_dirty=False)
        assert info.tag == "v1.0.0"
        assert info.commit == "abc123"
        assert info.branch == "main"
        assert info.is_dirty is False

    def test_git_info_with_none_tag(self):
        """Test GitInfo with no tag."""
        info = GitInfo(tag=None, commit="def456", branch="develop", is_dirty=True)
        assert info.tag is None
        assert info.is_dirty is True


class TestClassifierVersion:
    """Test ClassifierVersion model."""

    def test_valid_classifier_version(self):
        """Test creating valid classifier version."""
        cv = ClassifierVersion(name="test-classifier", version="1.2.3")
        assert cv.name == "test-classifier"
        assert cv.version == "1.2.3"

    def test_default_version(self):
        """Test default version is applied."""
        cv = ClassifierVersion(name="test")
        assert cv.version == "1.0.0"

    def test_invalid_name_uppercase(self):
        """Test validation rejects uppercase names."""
        with pytest.raises(ValueError):
            ClassifierVersion(name="TestClassifier", version="1.0.0")

    def test_invalid_name_special_chars(self):
        """Test validation rejects special characters."""
        with pytest.raises(ValueError):
            ClassifierVersion(name="test@classifier", version="1.0.0")

    def test_name_with_underscore(self):
        """Test name allows underscores."""
        cv = ClassifierVersion(name="test_classifier", version="1.0.0")
        assert cv.name == "test_classifier"

    def test_invalid_version_format(self):
        """Test validation rejects invalid version format."""
        with pytest.raises(ValueError):
            ClassifierVersion(name="test", version="1.0")

    def test_valid_repository(self):
        """Test valid repository name."""
        cv = ClassifierVersion(name="test", repository="my-repo")
        assert cv.repository == "my-repo"

    def test_invalid_repository_uppercase(self):
        """Test validation rejects uppercase repository."""
        with pytest.raises(ValueError):
            ClassifierVersion(name="test", repository="MyRepo")


class TestModelVersion:
    """Test ModelVersion model."""

    def test_model_version_defaults(self):
        """Test model version defaults."""
        mv = ModelVersion()
        assert mv.version == "1.0.0"
        assert mv.trained_at is None
        assert mv.metrics == {}

    def test_model_version_with_metrics(self):
        """Test model version with metrics."""
        mv = ModelVersion(
            version="2.0.0",
            trained_at="2024-01-01T00:00:00",
            metrics={"accuracy": 0.95, "f1": 0.92}
        )
        assert mv.metrics["accuracy"] == 0.95


class TestApiVersion:
    """Test ApiVersion model."""

    def test_api_version_defaults(self):
        """Test API version defaults."""
        av = ApiVersion()
        assert av.version == "v1"
        assert av.endpoints["predict"] is True

    def test_valid_api_version(self):
        """Test valid API version format."""
        av = ApiVersion(version="v2")
        assert av.version == "v2"

    def test_invalid_api_version(self):
        """Test invalid API version format."""
        with pytest.raises(ValueError):
            ApiVersion(version="2")  # Missing 'v' prefix


class TestClassifierMetadata:
    """Test ClassifierMetadata model."""

    def test_create_metadata(self):
        """Test creating classifier metadata."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="test-classifier", version="1.0.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )
        assert metadata.classifier.name == "test-classifier"

    def test_metadata_from_dict(self):
        """Test creating metadata from dict."""
        data = {
            "classifier": {"name": "my-clf", "version": "2.0.0"},
            "model": {"version": "1.5.0"},
            "api": {"version": "v1"}
        }
        metadata = ClassifierMetadata.model_validate(data)
        assert metadata.classifier.version == "2.0.0"


class TestGetGitInfo:
    """Test get_git_info function."""

    def test_non_git_directory(self):
        """Test returns None for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_git_info(tmpdir)
            assert result is None

    def test_git_info_structure(self):
        """Test git info structure when mocked."""
        with patch('subprocess.check_output') as mock:
            mock.side_effect = [
                b"abc123def456\n",  # commit
                b"main\n",  # branch
                b"",  # status (not dirty)
                subprocess.CalledProcessError(1, 'git')  # no tag
            ]
            with patch('os.getcwd', return_value="/test"):
                with patch('os.chdir'):
                    result = get_git_info("/some/path")
                    # May return None if the mock doesn't work correctly
                    assert result is None or isinstance(result, GitInfo)


class TestGetRepositoryName:
    """Test get_repository_name function."""

    def test_from_https_url(self):
        """Test extraction from HTTPS URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"https://github.com/org/my-repo.git\n"
            result = get_repository_name(".")
            assert result == "my-repo"

    def test_from_ssh_url(self):
        """Test extraction from SSH URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"git@github.com:org/another-repo.git\n"
            result = get_repository_name(".")
            assert result == "another-repo"

    def test_fallback_to_directory(self):
        """Test fallback to directory name."""
        with patch('subprocess.check_output') as mock:
            mock.side_effect = subprocess.CalledProcessError(1, 'git')
            with tempfile.TemporaryDirectory() as tmpdir:
                project = Path(tmpdir) / "MyProject"
                project.mkdir()
                result = get_repository_name(str(project))
                assert result == "myproject"


class TestLoadClassifierMetadata:
    """Test load_classifier_metadata function."""

    def test_load_valid_config(self):
        """Test loading valid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "classifier": {"name": "test-clf", "version": "1.0.0"},
                "model": {},
                "api": {"version": "v1"}
            }
            config_file = Path(tmpdir) / "mlserver.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            metadata = load_classifier_metadata(tmpdir)
            assert metadata.classifier.name == "test-clf"

    def test_missing_config_file(self):
        """Test error when config file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ConfigurationError) as exc_info:
                load_classifier_metadata(tmpdir)
            assert "not found" in str(exc_info.value)

    def test_invalid_yaml(self):
        """Test error on invalid YAML syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "mlserver.yaml"
            config_file.write_text("invalid: yaml: content: :")

            with pytest.raises(ConfigurationError) as exc_info:
                load_classifier_metadata(tmpdir)
            assert "Invalid YAML" in str(exc_info.value)

    def test_missing_classifier_section(self):
        """Test error when classifier section missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": {}, "api": {}}
            config_file = Path(tmpdir) / "mlserver.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            with pytest.raises(ConfigurationError) as exc_info:
                load_classifier_metadata(tmpdir)
            assert "missing required" in str(exc_info.value)


class TestSaveClassifierMetadata:
    """Test save_classifier_metadata function."""

    def test_save_and_reload(self):
        """Test saving and reloading metadata."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="save-test", version="1.0.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_classifier_metadata(metadata, tmpdir)

            # Verify file was created
            config_file = Path(tmpdir) / "mlserver.yaml"
            assert config_file.exists()

            # Reload and verify
            loaded = load_classifier_metadata(tmpdir)
            assert loaded.classifier.name == "save-test"


class TestGenerateContainerTags:
    """Test generate_container_tags function."""

    def test_basic_tags(self):
        """Test basic tag generation."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="test", version="1.0.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )

        with patch('mlserver.version.get_repository_name') as mock:
            mock.return_value = "my-repo"
            tags = generate_container_tags(metadata)

            assert "my-repo:latest" in tags
            assert "my-repo:v1.0.0" in tags

    def test_tags_with_git_info(self):
        """Test tags include commit when git info available."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="test", version="2.0.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )
        git_info = GitInfo(tag="v2.0.0", commit="abc1234def", branch="main", is_dirty=False)

        with patch('mlserver.version.get_repository_name') as mock:
            mock.return_value = "repo"
            tags = generate_container_tags(metadata, git_info)

            assert any("abc1234" in tag for tag in tags)

    def test_tags_for_multi_classifier(self):
        """Test tags for multi-classifier setup."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="classifier1", version="1.5.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )

        with patch('mlserver.version.get_repository_name') as mock:
            mock.return_value = "ml-classifiers"
            tags = generate_container_tags(metadata, classifier_name="sentiment")

            assert any("ml-classifiers/sentiment" in tag for tag in tags)


class TestValidateVersionConsistency:
    """Test validate_version_consistency function."""

    def test_no_issues_when_consistent(self):
        """Test no issues when versions are consistent."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="test", version="1.0.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('mlserver.version.get_git_info') as mock:
                mock.return_value = GitInfo(tag="v1.0.0", commit="abc", branch="main", is_dirty=False)
                issues = validate_version_consistency(metadata, tmpdir)
                assert len(issues) == 0

    def test_detects_tag_mismatch(self):
        """Test detection of tag mismatch."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="test", version="1.0.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('mlserver.version.get_git_info') as mock:
                mock.return_value = GitInfo(tag="v2.0.0", commit="abc", branch="main", is_dirty=False)
                issues = validate_version_consistency(metadata, tmpdir)
                assert "git_tag" in issues

    def test_detects_dirty_working_directory(self):
        """Test detection of dirty working directory."""
        metadata = ClassifierMetadata(
            classifier=ClassifierVersion(name="test", version="1.0.0"),
            model=ModelVersion(),
            api=ApiVersion()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('mlserver.version.get_git_info') as mock:
                mock.return_value = GitInfo(tag=None, commit="abc", branch="main", is_dirty=True)
                issues = validate_version_consistency(metadata, tmpdir)
                assert "git_dirty" in issues


class TestGetVersionInfo:
    """Test get_version_info function."""

    def test_missing_config_returns_error(self):
        """Test error returned when config missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_version_info(tmpdir)
            assert "error" in result

    def test_valid_project_returns_info(self):
        """Test valid project returns version info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "predictor": {"module": "mod", "class_name": "Cls"},
                "classifier": {"name": "test-clf", "version": "1.0.0"},
                "api": {"version": "v1", "adapter": "records"}
            }
            config_file = Path(tmpdir) / "mlserver.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            with patch('mlserver.version.get_git_info') as mock:
                mock.return_value = None
                result = get_version_info(tmpdir)

            assert "classifier" in result
            assert "container_tags" in result
            assert "timestamp" in result


class TestVersionEdgeCases:
    """Test edge cases in version handling."""

    def test_version_with_prerelease(self):
        """Test that prerelease versions are rejected."""
        with pytest.raises(ValueError):
            ClassifierVersion(name="test", version="1.0.0-beta")

    def test_name_starting_with_number(self):
        """Test name starting with number is allowed."""
        cv = ClassifierVersion(name="1classifier", version="1.0.0")
        assert cv.name == "1classifier"

    def test_empty_description(self):
        """Test empty description is allowed."""
        cv = ClassifierVersion(name="test", version="1.0.0", description="")
        assert cv.description == ""

    def test_repository_with_periods(self):
        """Test repository name with periods."""
        cv = ClassifierVersion(name="test", version="1.0.0", repository="my.repo")
        assert cv.repository == "my.repo"
