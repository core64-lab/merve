"""Basic tests for CLI module functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from mlserver.cli import (
    _is_likely_file_path,
    _resolve_path,
    app,
    resolve_relative_paths,
)

runner = CliRunner()


def _write_multi_config(path: Path):
    """Write a minimal two-classifier config to path/mlserver.yaml."""
    (path / "mlserver.yaml").write_text(
        """
classifiers:
  sentiment:
    predictor: {module: predictor, class_name: TestPredictor}
    classifier: {name: sentiment, version: "1.0.0"}
  intent:
    predictor: {module: predictor, class_name: TestPredictor}
    classifier: {name: intent, version: "2.0.0"}
default_classifier: sentiment
"""
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
            "name": "test_model",  # Should not be changed
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


class TestBuildCommandBuildOnce:
    """CLI build: build-once default vs --per-classifier-image (RFC 0001 D4 / W2.5)."""

    def test_multi_classifier_default_builds_commit_image(self, tmp_path):
        _write_multi_config(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {"success": True, "tags": ["repo:abc1234", "repo:latest"]}
            result = runner.invoke(app, ["build", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.stdout
        kwargs = mock_build.call_args.kwargs
        # Default = single commit image: no baked classifier, no per-classifier flag
        assert kwargs["per_classifier_image"] is False
        assert kwargs["classifier_name"] is None

    def test_classifier_ignored_for_commit_image(self, tmp_path):
        _write_multi_config(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {"success": True, "tags": ["repo:latest"]}
            result = runner.invoke(
                app, ["build", "--path", str(tmp_path), "--classifier", "sentiment"]
            )

        assert result.exit_code == 0, result.stdout
        # --classifier is dropped for the commit image (bundles all classifiers)
        assert mock_build.call_args.kwargs["classifier_name"] is None
        assert "ignored for the commit image" in result.stdout

    def test_per_classifier_image_requires_classifier(self, tmp_path):
        _write_multi_config(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            result = runner.invoke(
                app, ["build", "--path", str(tmp_path), "--per-classifier-image"]
            )

        assert result.exit_code == 1
        assert "requires --classifier" in result.stdout
        assert "sentiment" in result.stdout and "intent" in result.stdout
        mock_build.assert_not_called()

    def test_per_classifier_image_with_classifier(self, tmp_path):
        _write_multi_config(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {"success": True, "tags": ["repo/sentiment:latest"]}
            result = runner.invoke(
                app,
                [
                    "build",
                    "--path",
                    str(tmp_path),
                    "--per-classifier-image",
                    "--classifier",
                    "sentiment",
                ],
            )

        assert result.exit_code == 0, result.stdout
        kwargs = mock_build.call_args.kwargs
        assert kwargs["per_classifier_image"] is True
        assert kwargs["classifier_name"] == "sentiment"


class TestRunCommandDeployMany:
    """CLI run: the commit image is selected at run time via MLSERVER_CLASSIFIER."""

    def test_run_multi_classifier_sets_env(self, tmp_path):
        _write_multi_config(tmp_path)
        captured = {}

        def fake_run(cmd, *a, **k):
            captured["cmd"] = cmd
            return MagicMock(returncode=0, stdout="containerid123", stderr="")

        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = runner.invoke(
                app,
                [
                    "run",
                    "--path",
                    str(tmp_path),
                    "--classifier",
                    "sentiment",
                    "--detach",
                ],
            )

        assert result.exit_code == 0, result.stdout
        cmd = captured["cmd"]
        assert "-e" in cmd
        assert "MLSERVER_CLASSIFIER=sentiment" in cmd
        # Runs the commit image (<repo>:latest), NOT <repo>/<classifier>
        assert cmd[-1].endswith(":latest")
        assert "/sentiment" not in cmd[-1]


class TestPushCommandAlias:
    """CLI push: multi-classifier applies release aliases on the commit image."""

    def test_push_multi_classifier_requires_classifier(self, tmp_path):
        _write_multi_config(tmp_path)
        with patch("mlserver.cli.build.check_docker_availability", return_value=True):
            result = runner.invoke(app, ["push", "--registry", "reg.io", "--path", str(tmp_path)])

        assert result.exit_code == 1
        assert "sentiment" in result.stdout and "intent" in result.stdout

    def test_push_multi_classifier_alias_flow(self, tmp_path):
        _write_multi_config(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.GitVersionManager") as mock_gvm,
            patch("mlserver.container.push_classifier_alias") as mock_alias,
        ):
            mgr = mock_gvm.return_value
            mgr.validate_push_readiness.return_value = {"ready": True, "errors": []}
            mgr.get_current_version.return_value = "1.2.0"
            mock_alias.return_value = {
                "success": True,
                "source_image": "myrepo:abc1234",
                "pushed_tags": ["reg.io/myrepo:sentiment-v1.2.0"],
            }
            result = runner.invoke(
                app,
                [
                    "push",
                    "--registry",
                    "reg.io",
                    "--path",
                    str(tmp_path),
                    "--classifier",
                    "sentiment",
                ],
            )

        assert result.exit_code == 0, result.stdout
        mock_alias.assert_called_once()
        akwargs = mock_alias.call_args.kwargs
        assert akwargs["classifier_name"] == "sentiment"
        assert akwargs["version"] == "1.2.0"
        assert akwargs["registry"] == "reg.io"

    def test_push_multi_classifier_validation_failure(self, tmp_path):
        _write_multi_config(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.GitVersionManager") as mock_gvm,
            patch("mlserver.container.push_classifier_alias") as mock_alias,
        ):
            mgr = mock_gvm.return_value
            mgr.validate_push_readiness.return_value = {
                "ready": False,
                "errors": ["Not on a tagged commit for 'sentiment'."],
            }
            result = runner.invoke(
                app,
                [
                    "push",
                    "--registry",
                    "reg.io",
                    "--path",
                    str(tmp_path),
                    "--classifier",
                    "sentiment",
                ],
            )

        assert result.exit_code == 1
        assert "Not on a tagged commit" in result.stdout
        mock_alias.assert_not_called()
