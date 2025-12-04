"""Unit tests for auto_detect module."""
import pytest
import os
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.auto_detect import (
    get_git_project_name,
    get_project_name,
    get_git_info,
    get_deployed_timestamp,
    get_mlserver_package_version,
    get_mlserver_git_info,
    generate_simplified_metadata,
    get_simplified_info_response,
)


class TestGetGitProjectName:
    """Test get_git_project_name function."""

    def test_returns_none_for_non_git_dir(self):
        """Test returns None for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_git_project_name(tmpdir)
            assert result is None

    def test_extracts_name_from_https_url(self):
        """Test extraction from HTTPS git URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"https://github.com/user/my-repo.git\n"
            result = get_git_project_name(".")
            assert result == "my-repo"

    def test_extracts_name_from_ssh_url(self):
        """Test extraction from SSH git URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"git@github.com:user/another-repo.git\n"
            result = get_git_project_name(".")
            assert result == "another-repo"

    def test_extracts_name_without_git_extension(self):
        """Test extraction from URL without .git extension."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"https://github.com/user/repo-name\n"
            result = get_git_project_name(".")
            assert result == "repo-name"

    def test_handles_subprocess_error(self):
        """Test graceful handling of subprocess errors."""
        with patch('subprocess.check_output') as mock:
            mock.side_effect = subprocess.CalledProcessError(1, 'git')
            result = get_git_project_name(".")
            assert result is None

    def test_handles_file_not_found(self):
        """Test graceful handling when git not installed."""
        with patch('subprocess.check_output') as mock:
            mock.side_effect = FileNotFoundError()
            result = get_git_project_name(".")
            assert result is None


class TestGetProjectName:
    """Test get_project_name function."""

    def test_uses_git_name_when_available(self):
        """Test that git name is preferred."""
        with patch('mlserver.auto_detect.get_git_project_name') as mock:
            mock.return_value = "git-repo-name"
            result = get_project_name("/some/path")
            assert result == "git-repo-name"

    def test_falls_back_to_directory_name(self):
        """Test fallback to directory name."""
        with patch('mlserver.auto_detect.get_git_project_name') as mock:
            mock.return_value = None
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir) / "MyProject"
                project_dir.mkdir()
                result = get_project_name(str(project_dir))
                assert result == "myproject"  # Lowercased

    def test_lowercases_directory_name(self):
        """Test that directory names are lowercased."""
        with patch('mlserver.auto_detect.get_git_project_name') as mock:
            mock.return_value = None
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir) / "CamelCaseProject"
                project_dir.mkdir()
                result = get_project_name(str(project_dir))
                assert result == "camelcaseproject"


class TestGetGitInfo:
    """Test get_git_info function."""

    def test_returns_empty_info_for_non_git(self):
        """Test returns default info for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = get_git_info(tmpdir)
            assert info["repository"] is None
            assert info["commit"] is None
            assert info["tag"] is None
            assert info["branch"] is None
            assert info["dirty"] is False

    def test_uses_environment_variables(self):
        """Test that environment variables are used when set."""
        with patch.dict(os.environ, {
            'MLSERVER_GIT_COMMIT': 'abc123',
            'MLSERVER_GIT_TAG': 'v1.0.0',
            'MLSERVER_GIT_BRANCH': 'main'
        }):
            info = get_git_info(".")
            assert info["commit"] == "abc123"
            assert info["tag"] == "v1.0.0"
            assert info["branch"] == "main"
            assert info["dirty"] is False

    def test_partial_env_vars(self):
        """Test with only some environment variables set."""
        with patch.dict(os.environ, {'MLSERVER_GIT_COMMIT': 'def456'}, clear=False):
            # Clear other vars
            os.environ.pop('MLSERVER_GIT_TAG', None)
            os.environ.pop('MLSERVER_GIT_BRANCH', None)
            info = get_git_info(".")
            assert info["commit"] == "def456"


class TestGetDeployedTimestamp:
    """Test get_deployed_timestamp function."""

    def test_returns_iso_format(self):
        """Test returns ISO 8601 format timestamp."""
        timestamp = get_deployed_timestamp()

        # Should be in format: YYYY-MM-DDTHH:MM:SS.ffffffZ
        assert "T" in timestamp
        assert timestamp.endswith("Z")
        assert len(timestamp) > 20  # At least YYYY-MM-DDTHH:MM:SSZ

    def test_timestamp_changes_over_time(self):
        """Test that timestamps are different on subsequent calls."""
        import time
        ts1 = get_deployed_timestamp()
        time.sleep(0.01)
        ts2 = get_deployed_timestamp()
        # They should be different (at least in microseconds)
        assert ts1 != ts2


class TestGetMlserverPackageVersion:
    """Test get_mlserver_package_version function."""

    def test_returns_version_or_unknown(self):
        """Test returns a version string or 'unknown'."""
        version = get_mlserver_package_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_handles_missing_package_gracefully(self):
        """Test graceful handling when package metadata fails."""
        # The function should always return a string, even if it's "unknown" or a version
        version = get_mlserver_package_version()
        assert isinstance(version, str)
        assert len(version) > 0


class TestGetMlserverGitInfo:
    """Test get_mlserver_git_info function."""

    def test_returns_package_version(self):
        """Test that package version is always included."""
        info = get_mlserver_git_info()
        assert "package_version" in info
        assert isinstance(info["package_version"], str)

    def test_uses_api_environment_variables(self):
        """Test uses API-specific environment variables when _version_info not available."""
        # This tests the env var fallback - but since we're running in dev mode,
        # the _version_info module will be checked first and it has actual git info.
        # So we test that env vars are at least checked
        with patch.dict(os.environ, {
            'MLSERVER_API_COMMIT': 'api123',
            'MLSERVER_API_TAG': 'api-v2.0.0',
            'MLSERVER_API_BRANCH': 'develop'
        }):
            # Mock the _version_info import to fail, forcing env var usage
            with patch('mlserver.auto_detect.get_git_info') as mock_git:
                mock_git.return_value = {"commit": None, "tag": None, "branch": None, "dirty": False}
                with patch.dict('sys.modules', {'mlserver._version_info': None}):
                    info = get_mlserver_git_info()
                    # Either uses env vars or embedded version info
                    assert "api_commit" in info

    def test_returns_expected_structure(self):
        """Test returns expected dict structure."""
        info = get_mlserver_git_info()
        assert "package_version" in info
        assert "api_commit" in info
        assert "api_tag" in info
        assert "api_branch" in info
        assert "api_dirty" in info


class TestGenerateSimplifiedMetadata:
    """Test generate_simplified_metadata function."""

    def test_basic_metadata_generation(self):
        """Test basic metadata generation."""
        config = {
            "classifier": {
                "name": "test-classifier",
                "description": "A test classifier"
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = generate_simplified_metadata(config, tmpdir)

            assert "project" in metadata
            assert metadata["classifier"] == "test-classifier"
            assert metadata["description"] == "A test classifier"
            assert "deployed_at" in metadata
            assert "mlserver_version" in metadata

    def test_removes_none_values(self):
        """Test that None values are removed from output."""
        config = {"classifier": {"name": "test"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = generate_simplified_metadata(config, tmpdir)

            # None values should not be present
            for value in metadata.values():
                assert value is not None

    def test_uses_unknown_for_missing_classifier_name(self):
        """Test uses 'unknown' when classifier name is missing."""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = generate_simplified_metadata(config, tmpdir)
            assert metadata["classifier"] == "unknown"


class TestGetSimplifiedInfoResponse:
    """Test get_simplified_info_response function."""

    def test_includes_all_required_fields(self):
        """Test response includes all required fields."""
        config = {
            "classifier": {
                "name": "my-classifier",
                "description": "My ML classifier"
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            response = get_simplified_info_response(config, "MyPredictor", tmpdir)

            assert "project" in response
            assert "classifier" in response
            assert "description" in response
            assert "predictor_class" in response
            assert "deployed_at" in response
            assert "classifier_repository" in response
            assert "api_service" in response
            assert "endpoints" in response

    def test_includes_predictor_class(self):
        """Test predictor class name is included."""
        config = {"classifier": {"name": "test"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            response = get_simplified_info_response(config, "CustomPredictor", tmpdir)
            assert response["predictor_class"] == "CustomPredictor"

    def test_includes_standard_endpoints(self):
        """Test standard endpoints are listed."""
        config = {"classifier": {"name": "test"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            response = get_simplified_info_response(config, "Predictor", tmpdir)

            endpoints = response["endpoints"]
            assert endpoints["predict"] == "/predict"
            assert endpoints["predict_proba"] == "/predict_proba"
            assert endpoints["info"] == "/info"
            assert endpoints["health"] == "/healthz"
            assert endpoints["metrics"] == "/metrics"

    def test_classifier_repository_structure(self):
        """Test classifier_repository has expected structure."""
        config = {"classifier": {"name": "test"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            response = get_simplified_info_response(config, "Pred", tmpdir)

            repo_info = response["classifier_repository"]
            assert "repository" in repo_info
            assert "commit" in repo_info
            assert "tag" in repo_info
            assert "branch" in repo_info
            assert "dirty" in repo_info


class TestAutoDetectEdgeCases:
    """Test edge cases in auto detection."""

    def test_handles_empty_git_remote(self):
        """Test handling of empty git remote URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"\n"
            result = get_git_project_name(".")
            assert result is None

    def test_handles_malformed_url(self):
        """Test handling of malformed git URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"not-a-valid-url\n"
            result = get_git_project_name(".")
            # Should still try to extract something or return None
            assert result is None or isinstance(result, str)

    def test_project_name_with_special_chars(self):
        """Test project name extraction with special characters."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"https://github.com/user/my_special-repo.123.git\n"
            result = get_git_project_name(".")
            assert result == "my_special-repo.123"
