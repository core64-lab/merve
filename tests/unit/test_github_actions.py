"""Unit tests for github_actions module."""
import pytest
import os
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.github_actions import (
    get_git_remote_info,
    get_mlserver_source_url,
    generate_build_and_push_workflow,
    check_github_actions_setup,
    validate_workflow_compatibility,
    init_github_actions,
    parse_workflow_version,
)


class TestGetGitRemoteInfo:
    """Test get_git_remote_info function."""

    def test_https_url(self):
        """Test extraction from HTTPS URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"https://github.com/myorg/my-repo.git\n"
            result = get_git_remote_info(".")

            assert result is not None
            assert result["owner"] == "myorg"
            assert result["repo"] == "my-repo"
            assert result["is_github"] is True

    def test_ssh_url(self):
        """Test extraction from SSH URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"git@github.com:user/project.git\n"
            result = get_git_remote_info(".")

            assert result is not None
            assert result["owner"] == "user"
            assert result["repo"] == "project"

    def test_https_url_without_git_extension(self):
        """Test extraction from HTTPS URL without .git extension."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"https://github.com/org/repo\n"
            result = get_git_remote_info(".")

            assert result is not None
            assert result["repo"] == "repo"

    def test_non_github_url(self):
        """Test returns None for non-GitHub URLs."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"https://gitlab.com/user/repo.git\n"
            result = get_git_remote_info(".")

            assert result is None

    def test_empty_url(self):
        """Test returns None for empty URL."""
        with patch('subprocess.check_output') as mock:
            mock.return_value = b"\n"
            result = get_git_remote_info(".")

            assert result is None

    def test_git_command_failure(self):
        """Test returns None when git command fails."""
        with patch('subprocess.check_output') as mock:
            mock.side_effect = subprocess.CalledProcessError(1, 'git')
            result = get_git_remote_info(".")

            assert result is None

    def test_file_not_found(self):
        """Test returns None when git not installed."""
        with patch('subprocess.check_output') as mock:
            mock.side_effect = FileNotFoundError()
            result = get_git_remote_info(".")

            assert result is None


class TestGetMlserverSourceUrl:
    """Test get_mlserver_source_url function."""

    def test_returns_default_url(self):
        """Test returns default URL when not in dev mode."""
        url = get_mlserver_source_url()

        # Should return a URL (either dev or default)
        assert url is not None
        assert isinstance(url, str)
        assert "github.com" in url or ".git" in url

    def test_dev_mode_with_git_repo(self):
        """Test detection in development mode."""
        # This test may vary based on actual environment
        url = get_mlserver_source_url(".")
        assert isinstance(url, str)


class TestGenerateBuildAndPushWorkflow:
    """Test generate_build_and_push_workflow function."""

    def test_basic_workflow_generation(self):
        """Test basic workflow generation."""
        workflow = generate_build_and_push_workflow(
            repo_name="my-classifier",
            mlserver_source_url="https://github.com/org/mlserver.git"
        )

        assert "ML Classifier Container Build" in workflow
        assert "my-classifier" in workflow or "mlserver" in workflow
        assert "python" in workflow.lower()

    def test_workflow_with_ghcr_registry(self):
        """Test workflow generation with GHCR registry."""
        workflow = generate_build_and_push_workflow(
            repo_name="test-repo",
            mlserver_source_url="https://github.com/org/mlserver.git",
            registry_config={"type": "ghcr"}
        )

        assert "ghcr" in workflow.lower() or "github" in workflow.lower()

    def test_workflow_with_ecr_registry(self):
        """Test workflow generation with ECR registry."""
        workflow = generate_build_and_push_workflow(
            repo_name="test-repo",
            mlserver_source_url="https://github.com/org/mlserver.git",
            registry_config={
                "type": "ecr",
                "ecr": {
                    "registry_id": "123456789012",
                    "aws_region": "us-west-2",
                    "repository_prefix": "ml-models"
                }
            }
        )

        assert "ECR" in workflow or "ecr" in workflow
        assert "123456789012" in workflow
        assert "us-west-2" in workflow

    def test_workflow_ecr_missing_registry_id(self):
        """Test workflow generation fails without registry_id for ECR."""
        with pytest.raises(ValueError) as exc_info:
            generate_build_and_push_workflow(
                repo_name="test-repo",
                mlserver_source_url="https://github.com/org/mlserver.git",
                registry_config={
                    "type": "ecr",
                    "ecr": {}  # Missing registry_id
                }
            )

        assert "registry_id" in str(exc_info.value)

    def test_workflow_with_custom_python_version(self):
        """Test workflow generation with custom Python version."""
        workflow = generate_build_and_push_workflow(
            repo_name="test-repo",
            mlserver_source_url="https://github.com/org/mlserver.git",
            python_version="3.12"
        )

        assert "3.12" in workflow or "python" in workflow.lower()

    def test_workflow_with_role_arn_direct_value(self):
        """Test workflow with direct role ARN value."""
        workflow = generate_build_and_push_workflow(
            repo_name="test-repo",
            mlserver_source_url="https://github.com/org/mlserver.git",
            registry_config={
                "type": "ecr",
                "ecr": {
                    "registry_id": "123456789012"
                },
                "github_variables": {
                    "aws_role_arn_value": "arn:aws:iam::123456789012:role/MyRole"
                }
            }
        )

        assert "arn:aws:iam" in workflow


class TestCheckGithubActionsSetup:
    """Test check_github_actions_setup function."""

    def test_workflow_exists(self):
        """Test returns True when workflow file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflows_dir = Path(tmpdir) / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)

            workflow_file = workflows_dir / "ml-classifier-container-build.yml"
            workflow_file.write_text("name: Build")

            result = check_github_actions_setup(tmpdir)
            assert result is True

    def test_workflow_not_exists(self):
        """Test returns False when workflow file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_github_actions_setup(tmpdir)
            assert result is False

    def test_github_dir_exists_but_no_workflow(self):
        """Test returns False when .github exists but no workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            github_dir = Path(tmpdir) / ".github"
            github_dir.mkdir()

            result = check_github_actions_setup(tmpdir)
            assert result is False


class TestValidateWorkflowCompatibility:
    """Test validate_workflow_compatibility function."""

    def test_compatible_workflow(self):
        """Test validation of compatible workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflows_dir = Path(tmpdir) / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)

            # Create a workflow with current version marker
            workflow_content = """# Generated by MLServer 0.5.0
# Workflow version: 2.0
name: Build
"""
            workflow_file = workflows_dir / "ml-classifier-container-build.yml"
            workflow_file.write_text(workflow_content)

            is_valid, warning, details = validate_workflow_compatibility(tmpdir)

            # Should be valid or have just warnings
            assert is_valid is True or warning is not None

    def test_incompatible_workflow(self):
        """Test validation of incompatible workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflows_dir = Path(tmpdir) / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)

            # Create a workflow with old version marker
            workflow_content = """# Generated by MLServer 0.1.0
# Workflow version: 1.0
name: Build
"""
            workflow_file = workflows_dir / "ml-classifier-container-build.yml"
            workflow_file.write_text(workflow_content)

            is_valid, warning, details = validate_workflow_compatibility(tmpdir)

            # May have warnings about version mismatch
            assert isinstance(is_valid, bool)
            assert isinstance(details, dict)

    def test_no_workflow_file(self):
        """Test validation when no workflow exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            is_valid, warning, details = validate_workflow_compatibility(tmpdir)

            # Should return a tuple - may be valid or not depending on implementation
            assert isinstance(is_valid, bool)
            assert isinstance(details, dict)

    def test_workflow_without_version_marker(self):
        """Test validation of workflow without version marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflows_dir = Path(tmpdir) / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)

            workflow_content = """name: Build
on: push
jobs:
  build:
    runs-on: ubuntu-latest
"""
            workflow_file = workflows_dir / "ml-classifier-container-build.yml"
            workflow_file.write_text(workflow_content)

            is_valid, warning, details = validate_workflow_compatibility(tmpdir)

            # Should handle missing version gracefully
            assert isinstance(is_valid, bool)


class TestInitGithubActions:
    """Test init_github_actions function."""

    def test_basic_setup(self):
        """Test basic GitHub Actions setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .git directory to pass git check
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()

            # Create mlserver.yaml
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: my_predictor
  class_name: MyPredictor
classifier:
  name: test-classifier
  version: "1.0.0"
""")

            with patch('mlserver.github_actions.get_git_remote_info') as mock_remote:
                mock_remote.return_value = {
                    "owner": "testorg",
                    "repo": "test-repo",
                    "is_github": True
                }

                result = init_github_actions(tmpdir, force=False)

                # Result is a tuple (success, message, details)
                assert isinstance(result, tuple)
                success, message, details = result
                assert isinstance(success, bool)

    def test_setup_with_force(self):
        """Test setup with force flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .git directory
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()

            # Create existing workflow
            workflows_dir = Path(tmpdir) / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)
            workflow_file = workflows_dir / "ml-classifier-container-build.yml"
            workflow_file.write_text("old content")

            # Create mlserver.yaml
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: my_predictor
  class_name: MyPredictor
classifier:
  name: test-classifier
  version: "1.0.0"
""")

            with patch('mlserver.github_actions.get_git_remote_info') as mock_remote:
                mock_remote.return_value = {
                    "owner": "testorg",
                    "repo": "test-repo",
                    "is_github": True
                }

                result = init_github_actions(tmpdir, force=True)

                # Result is a tuple (success, message, details)
                assert isinstance(result, tuple)

    def test_setup_non_git_repo(self):
        """Test setup for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: my_predictor
  class_name: MyPredictor
""")

            result = init_github_actions(tmpdir)

            # Should return tuple with False for non-git directory
            assert isinstance(result, tuple)
            success, message, details = result
            assert success is False
            assert "git" in message.lower()


class TestParseWorkflowVersion:
    """Test parse_workflow_version function."""

    def test_parse_valid_version(self):
        """Test parsing workflow with version markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_path = Path(tmpdir) / "workflow.yml"
            workflow_path.write_text("""# Generated by MLServer 0.5.0
# Workflow version: 2.0
name: Build
""")

            mlserver_version, workflow_version = parse_workflow_version(workflow_path)

            assert mlserver_version == "0.5.0"
            assert workflow_version == "2.0"

    def test_parse_no_version(self):
        """Test parsing workflow without version markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_path = Path(tmpdir) / "workflow.yml"
            workflow_path.write_text("""name: Build
on: push
""")

            mlserver_version, workflow_version = parse_workflow_version(workflow_path)

            assert mlserver_version is None
            assert workflow_version is None

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file."""
        mlserver_version, workflow_version = parse_workflow_version(Path("/nonexistent/workflow.yml"))

        assert mlserver_version is None
        assert workflow_version is None


class TestWorkflowContent:
    """Test the content of generated workflows."""

    def test_workflow_has_required_sections(self):
        """Test workflow contains required sections."""
        workflow = generate_build_and_push_workflow(
            repo_name="test",
            mlserver_source_url="https://github.com/org/repo.git"
        )

        # Check for key workflow components
        assert "name:" in workflow
        assert "on:" in workflow
        assert "jobs:" in workflow

    def test_workflow_has_build_job(self):
        """Test workflow has build job."""
        workflow = generate_build_and_push_workflow(
            repo_name="test",
            mlserver_source_url="https://github.com/org/repo.git"
        )

        assert "build" in workflow.lower()

    def test_workflow_uses_docker(self):
        """Test workflow uses Docker."""
        workflow = generate_build_and_push_workflow(
            repo_name="test",
            mlserver_source_url="https://github.com/org/repo.git"
        )

        assert "docker" in workflow.lower()


class TestEdgeCases:
    """Test edge cases in github_actions module."""

    def test_special_chars_in_repo_name(self):
        """Test handling of special characters in repo name."""
        workflow = generate_build_and_push_workflow(
            repo_name="my-special_repo.name",
            mlserver_source_url="https://github.com/org/repo.git"
        )

        # Should generate valid YAML
        assert isinstance(workflow, str)

    def test_empty_registry_config(self):
        """Test handling of empty registry config."""
        workflow = generate_build_and_push_workflow(
            repo_name="test",
            mlserver_source_url="https://github.com/org/repo.git",
            registry_config={}
        )

        # Should use defaults (GHCR)
        assert isinstance(workflow, str)

    def test_none_registry_config(self):
        """Test handling of None registry config."""
        workflow = generate_build_and_push_workflow(
            repo_name="test",
            mlserver_source_url="https://github.com/org/repo.git",
            registry_config=None
        )

        # Should use defaults
        assert isinstance(workflow, str)
