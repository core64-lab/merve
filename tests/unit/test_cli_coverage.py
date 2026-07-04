"""CliRunner coverage suites for the ``mlserver/cli`` package (RFC 0001 W2.1).

Closes the remaining per-module coverage gaps (versioning, build, project,
serve, testing, _app) with behavior-asserting tests: every test checks the
exit code AND either key output or the arguments of a mocked collaborator.

Everything external is mocked — no docker daemon, no network, no real uvicorn.
Hermetic projects are written into ``tmp_path`` (same pattern as
``test_cli_commands.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml
from typer.testing import CliRunner

from mlserver.cli import app, main
from mlserver.cli._app import _is_likely_file_path, detect_config_file, resolve_relative_paths
from mlserver.doctor import CheckResult, CheckStatus, DiagnosticReport
from mlserver.errors import VersionControlError
from mlserver.validation import ValidationResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Hermetic project helpers (pattern shared with test_cli_commands.py)
# ---------------------------------------------------------------------------


def _write_project(tmp_path, port: int = 9123):
    """A minimal, importable single-classifier project (with init_kwargs)."""
    (tmp_path / "covpred.py").write_text(
        "class P:\n"
        "    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n"
        "    def predict(self, X):\n        return [0] * len(X)\n"
        "    def predict_proba(self, X):\n        return [[0.5, 0.5]] * len(X)\n"
    )
    (tmp_path / "model.pkl").write_text("stub")
    (tmp_path / "mlserver.yaml").write_text(
        yaml.safe_dump(
            {
                "predictor": {
                    "module": "covpred",
                    "class_name": "P",
                    "init_kwargs": {"model_path": "./model.pkl"},
                },
                "classifier": {"name": "solo", "version": "1.0.0"},
                "server": {"host": "127.0.0.1", "port": port},
                "observability": {"metrics": False, "structured_logging": False},
            }
        )
    )
    return tmp_path


def _write_multi_project(tmp_path, default: str | None = "alpha"):
    """A minimal, importable multi-classifier project (alpha, beta)."""
    (tmp_path / "covpred_multi.py").write_text(
        "class A:\n"
        "    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n"
        "    def predict(self, X):\n        return [0] * len(X)\n"
        "class B:\n"
        "    def predict(self, X):\n        return [1] * len(X)\n"
    )
    (tmp_path / "model.pkl").write_text("stub")
    doc = {
        "server": {"host": "127.0.0.1", "port": 9123},
        "classifiers": {
            "alpha": {
                "predictor": {
                    "module": "covpred_multi",
                    "class_name": "A",
                    "init_kwargs": {"model_path": "./model.pkl"},
                },
                "classifier": {"name": "alpha", "version": "1.0.0"},
                "observability": {"metrics": False, "structured_logging": False},
            },
            "beta": {
                "predictor": {"module": "covpred_multi", "class_name": "B"},
                "classifier": {"name": "beta", "version": "2.0.0"},
                "observability": {"metrics": False, "structured_logging": False},
            },
        },
    }
    if default:
        doc["default_classifier"] = default
    (tmp_path / "mlserver.yaml").write_text(yaml.safe_dump(doc))
    return tmp_path


def _report(*checks, recommendations=()):
    rep = DiagnosticReport()
    for check in checks:
        rep.add(check)
    for rec in recommendations:
        rep.add_recommendation(rec)
    return rep


# ---------------------------------------------------------------------------
# _app.py helpers
# ---------------------------------------------------------------------------


class TestDetectConfigFile:
    def test_explicit_existing_path_is_returned_as_is(self, tmp_path):
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("a: 1")
        assert detect_config_file(cfg) == cfg

    def test_base_dir_default_is_absolute(self, tmp_path):
        (tmp_path / "mlserver.yaml").write_text("a: 1")
        found = detect_config_file(None, base_dir=tmp_path)
        assert found.is_absolute()
        assert found == tmp_path.resolve() / "mlserver.yaml"

    def test_cwd_default_without_base_dir(self, tmp_path, monkeypatch):
        (tmp_path / "mlserver.yaml").write_text("a: 1")
        monkeypatch.chdir(tmp_path)
        assert detect_config_file(None) == Path("mlserver.yaml")

    def test_explicit_missing_path_raises_with_filename(self, tmp_path):
        with pytest.raises(typer.BadParameter, match="ghost.yaml"):
            detect_config_file(tmp_path / "ghost.yaml", base_dir=tmp_path)

    def test_nothing_found_raises_no_config(self, tmp_path):
        with pytest.raises(typer.BadParameter, match="No config file found"):
            detect_config_file(None, base_dir=tmp_path)


class TestPathHeuristics:
    def test_windows_drive_letter_is_not_relative(self):
        assert not _is_likely_file_path("model_path", "C:/models/m.pkl")

    def test_bare_relative_with_path_key_is_path(self):
        assert _is_likely_file_path("preprocessor_file", "artifacts/prep.pkl")

    def test_path_like_value_with_non_path_key_ignored(self):
        assert not _is_likely_file_path("threshold", "./model.pkl")

    def test_non_string_values_pass_through_untouched(self, tmp_path):
        kwargs = {"model_path": "./m.pkl", "n_jobs": 4, "labels": ["a"], "extra": None}
        resolved = resolve_relative_paths(kwargs, str(tmp_path))
        assert resolved["n_jobs"] == 4
        assert resolved["labels"] == ["a"]
        assert resolved["extra"] is None
        assert resolved["model_path"] == str(tmp_path / "m.pkl")


class TestEntrypoints:
    def test_main_invokes_typer_app(self):
        with patch("mlserver.cli.app") as mock_app:
            main()
        mock_app.assert_called_once_with()

    def test_python_dash_m_entrypoint(self, monkeypatch):
        import runpy

        monkeypatch.setattr(sys, "argv", ["merve", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("mlserver.cli", run_name="__main__")
        assert exc.value.code == 0

    def test_help_brands_as_merve(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "merve" in result.output

    def test_mlserver_alias_prints_deprecation_on_stderr(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mlserver"])
        result = runner.invoke(app, ["version", "-C", str(tmp_path), "--json"])
        assert result.exit_code == 0
        assert "deprecated" in result.stderr
        assert "merve" in result.stderr
        # stdout stays a single parseable JSON document
        json.loads(result.stdout)

    def test_merve_invocation_has_no_deprecation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["merve"])
        result = runner.invoke(app, ["version", "-C", str(tmp_path), "--json"])
        assert result.exit_code == 0
        assert "deprecated" not in result.stderr


# ---------------------------------------------------------------------------
# versioning.py — version
# ---------------------------------------------------------------------------

_SINGLE_VERSION_INFO = {
    "classifier": {"name": "solo", "version": "1.2.3", "description": "demo clf"},
    "model": {"version": "1.2.3"},
    "api": {"version": "v2"},
    "git": {"commit": "abc1234", "branch": "main", "tag": "solo/v1.2.3", "is_dirty": True},
    "container_tags": ["solo:v1.2.3-abc1234"],
    "validation_issues": {"model": "model version mismatch"},
}


class TestVersionCommandHuman:
    def test_single_classifier_table_with_git_and_issues(self, tmp_path):
        with patch(
            "mlserver.cli.versioning.get_version_info", return_value=dict(_SINGLE_VERSION_INFO)
        ) as mock_info:
            result = runner.invoke(app, ["version", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert mock_info.call_args.args[0] == str(tmp_path)
        assert "solo" in result.output
        assert "1.2.3" in result.output
        assert "Uncommitted changes" in result.output
        assert "Container Tags" in result.output
        assert "model version mismatch" in result.output

    def test_detailed_adds_mlserver_tool_section(self, tmp_path):
        with patch(
            "mlserver.cli.versioning.get_version_info", return_value=dict(_SINGLE_VERSION_INFO)
        ):
            result = runner.invoke(app, ["version", "-C", str(tmp_path), "--detailed"])
        assert result.exit_code == 0, result.output
        assert "MLServer Tool" in result.output
        assert "Install Type" in result.output

    def test_detailed_json_includes_tool_metadata(self, tmp_path):
        with patch(
            "mlserver.cli.versioning.get_version_info", return_value=dict(_SINGLE_VERSION_INFO)
        ):
            result = runner.invoke(app, ["version", "-C", str(tmp_path), "--json", "--detailed"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["classifier"]["name"] == "solo"
        assert "version" in data["mlserver_tool"]
        assert "install_location" in data["mlserver_tool"]

    def test_multi_classifier_summary_table(self, tmp_path):
        info = {
            "multi_classifier": True,
            "classifiers": [
                {"name": "alpha", "version": "1.0.0", "description": "first"},
                {"name": "beta", "version": "2.0.0"},
            ],
            "default_classifier": "alpha",
            "git": {"commit": "abcdef01234", "branch": "main"},
        }
        with patch("mlserver.cli.versioning.get_version_info", return_value=info):
            result = runner.invoke(app, ["version", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Multi-Classifier Project" in result.output
        assert "alpha" in result.output and "beta" in result.output
        assert "abcdef0" in result.output  # short git commit
        assert "--classifier <name>" in result.output

    def test_multi_classifier_summary_json(self, tmp_path):
        info = {
            "multi_classifier": True,
            "classifiers": [{"name": "alpha", "version": "1.0.0", "description": ""}],
            "default_classifier": "alpha",
            "git": None,
        }
        with patch("mlserver.cli.versioning.get_version_info", return_value=info):
            result = runner.invoke(app, ["version", "-C", str(tmp_path), "--json"])
        assert result.exit_code == 0
        assert json.loads(result.stdout)["multi_classifier"] is True

    def test_error_branch_exits_nonzero(self, tmp_path):
        with patch(
            "mlserver.cli.versioning.get_version_info",
            return_value={"error": "kaboom happened"},
        ):
            result = runner.invoke(app, ["version", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert "kaboom happened" in result.output


# ---------------------------------------------------------------------------
# versioning.py — tag
# ---------------------------------------------------------------------------


def _passing_suite(warnings=()):
    suite = MagicMock()
    results = [ValidationResult(passed=True, warnings=list(warnings))]
    if warnings:
        results[0].details = {"solution": "commit your changes"}
    suite.validate.return_value = (True, results)
    return suite


class TestTagCreate:
    def _invoke_tag(
        self,
        tmp_path,
        args,
        tag_info=None,
        suite=None,
        gh_configured=True,
        workflow=(True, [], {}),
        tag_side_effect=None,
    ):
        _write_project(tmp_path)
        mgr = MagicMock()
        if tag_side_effect:
            mgr.tag_version.side_effect = tag_side_effect
        else:
            mgr.tag_version.return_value = tag_info or {
                "tag_name": "solo/v1.1.0",
                "version": "1.1.0",
                "previous_version": "1.0.0",
                "mlserver_commit": "abc1234",
            }
        with (
            patch("mlserver.cli.versioning.GitVersionManager", return_value=mgr),
            patch(
                "mlserver.validation.get_tag_validation_suite",
                return_value=suite or _passing_suite(),
            ),
            patch("mlserver.version.get_git_info", return_value=MagicMock(commit="fff0000")),
            patch(
                "mlserver.github_actions.check_github_actions_setup",
                return_value=gh_configured,
            ),
            patch(
                "mlserver.github_actions.validate_workflow_comprehensive",
                return_value=workflow,
            ),
        ):
            result = runner.invoke(app, args)
        return result, mgr

    def test_tag_patch_happy_path_creates_canonical_tag(self, tmp_path):
        result, mgr = self._invoke_tag(
            tmp_path,
            ["tag", "patch", "--classifier", "solo", "-C", str(tmp_path)],
            suite=_passing_suite(warnings=["uncommitted docs change"]),
        )
        assert result.exit_code == 0, result.output
        mgr.tag_version.assert_called_once_with("patch", "solo", None, False)
        assert "solo/v1.1.0" in result.output  # canonical <classifier>/vX.Y.Z
        assert "patch bump" in result.output
        assert "uncommitted docs change" in result.output  # non-blocking warning shown
        assert "git push --tags" in result.output

    def test_tag_message_and_bump_are_forwarded(self, tmp_path):
        result, mgr = self._invoke_tag(
            tmp_path,
            ["tag", "minor", "-c", "solo", "-m", "custom msg", "-C", str(tmp_path)],
            tag_info={
                "tag_name": "solo/v0.1.0",
                "version": "0.1.0",
                "previous_version": None,
                "mlserver_commit": "abc1234",
            },
            workflow=(False, ["workflow is outdated"], {}),
        )
        assert result.exit_code == 0, result.output
        mgr.tag_version.assert_called_once_with("minor", "solo", "custom msg", False)
        assert "initial release" in result.output
        # Invalid workflow => regenerate warning + instructions
        assert "workflow is outdated" in result.output
        assert "init-github --force" in result.output

    def test_tag_without_github_actions_shows_manual_options(self, tmp_path):
        result, _ = self._invoke_tag(
            tmp_path,
            ["tag", "patch", "-c", "solo", "-C", str(tmp_path)],
            gh_configured=False,
        )
        assert result.exit_code == 0, result.output
        assert "GitHub Actions not configured" in result.output
        assert "Option 2: Build and push manually" in result.output

    def test_tag_validation_failure_blocks_tagging(self, tmp_path):
        _write_project(tmp_path)
        failing = MagicMock()
        failing.validate.return_value = (
            False,
            [
                ValidationResult(
                    passed=False,
                    error_message="Working directory not clean",
                    details={"missing_files": ["a_file.py"], "solution": "run merve init"},
                )
            ],
        )
        mgr = MagicMock()
        with (
            patch("mlserver.cli.versioning.GitVersionManager", return_value=mgr),
            patch("mlserver.validation.get_tag_validation_suite", return_value=failing),
        ):
            result = runner.invoke(app, ["tag", "patch", "-c", "solo", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert "Cannot create tag" in result.output
        assert "Working directory not clean" in result.output
        assert "a_file.py" in result.output
        assert "run merve init" in result.output
        mgr.tag_version.assert_not_called()

    def test_tag_requires_classifier(self, tmp_path):
        _write_project(tmp_path)
        with patch("mlserver.cli.versioning.GitVersionManager"):
            result = runner.invoke(app, ["tag", "patch", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert "Classifier name is required" in result.output

    def test_tag_bump_with_status_is_usage_error(self, tmp_path):
        result = runner.invoke(app, ["tag", "patch", "--status", "-C", str(tmp_path)])
        assert result.exit_code == 2
        assert "cannot be combined" in result.output

    def test_version_control_error_is_clean_exit(self, tmp_path):
        result, _ = self._invoke_tag(
            tmp_path,
            ["tag", "patch", "-c", "solo", "-C", str(tmp_path)],
            tag_side_effect=VersionControlError("dirty working tree"),
        )
        assert result.exit_code == 1
        assert "dirty working tree" in result.output
        assert "Traceback" not in result.output

    def test_unexpected_error_is_clean_exit(self, tmp_path):
        result, _ = self._invoke_tag(
            tmp_path,
            ["tag", "patch", "-c", "solo", "-C", str(tmp_path)],
            tag_side_effect=RuntimeError("weird"),
        )
        assert result.exit_code == 1
        assert "Unexpected error" in result.output
        assert "weird" in result.output


_TAG_STATUS = {
    "ready": {
        "current_version": "1.0.0",
        "latest_tag": "ready-v1.0.0-mlserver-abc1234",
        "commits_since_tag": 0,
        "on_tagged_commit": True,
        "status": "Ready",
        "recommendation": None,
    },
    "stale": {
        "current_version": "1.0.0",
        "latest_tag": "stale-v1.0.0-mlserver-1111111",
        "commits_since_tag": 3,
        "on_tagged_commit": False,
        "status": "3 commits behind",
        "recommendation": "merve tag",
    },
    "canon": {
        "current_version": "0.2.0",
        "latest_tag": "canon/v0.2.0",
        "commits_since_tag": 1,
        "on_tagged_commit": False,
        "status": "1 commits behind",
        "recommendation": "merve tag",
    },
    "untagged": {
        "current_version": None,
        "latest_tag": None,
        "commits_since_tag": None,
        "on_tagged_commit": False,
        "status": "No tags",
        "recommendation": "merve tag",
    },
}


class TestTagStatus:
    def _invoke_status(self, tmp_path, extra=()):
        _write_project(tmp_path)
        mgr = MagicMock()
        mgr.get_all_classifiers_tag_status.return_value = dict(_TAG_STATUS)
        with (
            patch("mlserver.cli.versioning.GitVersionManager", return_value=mgr),
            patch(
                "mlserver.version_control.get_mlserver_commit_hash",
                return_value="abc1234def5678",
            ),
        ):
            return runner.invoke(app, ["tag", "-C", str(tmp_path), *extra])

    def test_status_human_table_covers_all_states(self, tmp_path):
        result = self._invoke_status(tmp_path)
        assert result.exit_code == 0, result.output
        assert "Classifier Version Status" in result.output
        assert "Ready" in result.output
        assert "3 commits behind" in result.output
        assert "No tags" in result.output
        # Legacy tag matching current mlserver commit vs mismatching one
        assert "abc1234" in result.output
        assert "1111111" in result.output
        # Canonical tags don't encode the mlserver commit
        assert "n/a" in result.output
        assert "Current MLServer commit: abc1234" in result.output

    def test_status_json_document(self, tmp_path):
        result = self._invoke_status(tmp_path, extra=["--json"])
        assert result.exit_code == 0, result.output
        doc = json.loads(result.stdout)
        assert doc["mlserver_commit"] == "abc1234def5678"
        assert set(doc["classifiers"]) == set(_TAG_STATUS)
        assert doc["classifiers"]["ready"]["tag_format"] == "legacy"
        assert doc["classifiers"]["ready"]["tag_mlserver_commit"] == "abc1234"
        assert doc["classifiers"]["canon"]["tag_format"] == "canonical"
        assert doc["classifiers"]["untagged"]["tag_format"] is None
        assert doc["classifiers"]["untagged"]["status"] == "No tags"


# ---------------------------------------------------------------------------
# build.py — build
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_docker_unavailable_exits_1(self, tmp_path):
        with patch("mlserver.cli.build.check_docker_availability", return_value=False):
            result = runner.invoke(app, ["build", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "Docker is not available" in result.output

    def test_invalid_classifier_name_format(self, tmp_path):
        with patch("mlserver.cli.build.check_docker_availability", return_value=True):
            result = runner.invoke(
                app, ["build", "--path", str(tmp_path), "--classifier", "Bad Name!"]
            )
        assert result.exit_code == 1
        assert "Invalid classifier name format" in result.output

    def test_full_tag_mismatch_with_force_builds_current_code(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch(
                "mlserver.version_control.get_tag_commits",
                return_value={"classifier_commit": "1111111", "mlserver_commit": None},
            ),
            patch("mlserver.version.get_git_info", return_value=MagicMock(commit="2222222")),
            patch("mlserver.version_control.get_mlserver_commit_hash", return_value="3333333"),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {"success": True, "tags": ["solo:v1.0.0"]}
            result = runner.invoke(
                app,
                ["build", "--path", str(tmp_path), "--classifier", "solo/v1.0.0", "--force"],
            )
        assert result.exit_code == 0, result.output
        assert "doesn't match tag specifications" in result.output
        assert "MISMATCH" in result.output
        # The simple name is extracted from the canonical tag for the build
        assert mock_build.call_args.kwargs["classifier_name"] == "solo"

    def test_full_tag_mismatch_confirm_cancel_aborts(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch(
                "mlserver.version_control.get_tag_commits",
                return_value={"classifier_commit": "1111111", "mlserver_commit": None},
            ),
            patch("mlserver.version.get_git_info", return_value=MagicMock(commit="2222222")),
            patch("mlserver.version_control.get_mlserver_commit_hash", return_value="3333333"),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            result = runner.invoke(
                app,
                ["build", "--path", str(tmp_path), "--classifier", "solo/v1.0.0"],
                input="n\n",
            )
        assert result.exit_code == 0
        assert "Build cancelled" in result.output
        mock_build.assert_not_called()

    def test_full_legacy_tag_matching_commits_proceeds(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch(
                "mlserver.version_control.get_tag_commits",
                return_value={"classifier_commit": "abc1234", "mlserver_commit": "def5678"},
            ),
            patch("mlserver.version.get_git_info", return_value=MagicMock(commit="abc1234")),
            patch("mlserver.version_control.get_mlserver_commit_hash", return_value="def5678"),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {"success": True, "tags": ["solo:v1.0.0"]}
            result = runner.invoke(
                app,
                [
                    "build",
                    "--path",
                    str(tmp_path),
                    "--classifier",
                    "solo-v1.0.0-mlserver-def5678",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "matches tag specification" in result.output
        assert mock_build.call_args.kwargs["classifier_name"] == "solo"

    def test_build_arg_parsing_passes_dict(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {"success": True, "tags": ["t"]}
            result = runner.invoke(
                app,
                [
                    "build",
                    "--path",
                    str(tmp_path),
                    "--build-arg",
                    "A=1",
                    "--build-arg",
                    "B=x=y",
                ],
            )
        assert result.exit_code == 0, result.output
        # KEY=value split once: values may themselves contain '='
        assert mock_build.call_args.kwargs["build_args"] == {"A": "1", "B": "x=y"}

    def test_build_arg_without_equals_is_rejected(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            result = runner.invoke(
                app, ["build", "--path", str(tmp_path), "--build-arg", "NOEQUALS"]
            )
        assert result.exit_code == 1
        assert "expected format KEY=value" in result.output
        mock_build.assert_not_called()

    def test_build_failure_exits_1_with_error(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {"success": False, "error": "no Dockerfile"}
            result = runner.invoke(app, ["build", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "Build failed: no Dockerfile" in result.output

    def test_build_verbose_prints_build_output(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            mock_build.return_value = {
                "success": True,
                "tags": ["solo:latest"],
                "build_output": "STEP 1/9: FROM python",
            }
            result = runner.invoke(app, ["build", "--path", str(tmp_path), "-v"])
        assert result.exit_code == 0, result.output
        assert "STEP 1/9" in result.output

    def test_multi_unknown_classifier_per_classifier_image(self, tmp_path):
        _write_multi_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.build_container") as mock_build,
        ):
            result = runner.invoke(
                app,
                [
                    "build",
                    "--path",
                    str(tmp_path),
                    "--per-classifier-image",
                    "--classifier",
                    "nope",
                ],
            )
        assert result.exit_code == 1
        assert "'nope' not found" in result.output
        assert "alpha" in result.output and "beta" in result.output
        mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# build.py — push
# ---------------------------------------------------------------------------


class TestPushCommand:
    def test_docker_unavailable_exits_1(self, tmp_path):
        with patch("mlserver.cli.build.check_docker_availability", return_value=False):
            result = runner.invoke(app, ["push", "--registry", "r.io", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "Docker is not available" in result.output

    def test_single_classifier_push_happy_path(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.safe_push_container") as mock_push,
        ):
            mock_push.return_value = {
                "success": True,
                "version_used": "1.0.0",
                "version_source": "git tag solo/v1.0.0",
                "pushed_tags": ["r.io/solo:v1.0.0"],
                "validation_warnings": ["registry uses http"],
            }
            result = runner.invoke(
                app,
                ["push", "--registry", "r.io", "--path", str(tmp_path), "-c", "solo"],
            )
        assert result.exit_code == 0, result.output
        kwargs = mock_push.call_args.kwargs
        assert kwargs["registry"] == "r.io"
        assert kwargs["classifier_name"] == "solo"
        assert kwargs["force"] is False
        # RFC 0001 D3: no version_source plumbing on the single-classifier path
        assert "version_source" not in kwargs
        assert "Successfully pushed images" in result.output
        assert "r.io/solo:v1.0.0" in result.output
        assert "registry uses http" in result.output

    def test_single_classifier_push_failure_lists_validation_errors(self, tmp_path):
        # Empty dir (no mlserver.yaml) also exercises the config-detect fallback
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.safe_push_container") as mock_push,
        ):
            mock_push.return_value = {
                "success": False,
                "error": "Push validation failed",
                "validation_errors": ["not on a tagged commit"],
            }
            result = runner.invoke(app, ["push", "--registry", "r.io", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "Push validation failed" in result.output
        assert "not on a tagged commit" in result.output

    def test_single_classifier_partial_failure_exits_1(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.safe_push_container") as mock_push,
        ):
            mock_push.return_value = {
                "success": True,
                "version_used": "1.0.0",
                "version_source": "git tag",
                "pushed_tags": ["r.io/solo:v1.0.0"],
                "failed_tags": ["r.io/solo:latest denied"],
            }
            result = runner.invoke(app, ["push", "--registry", "r.io", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "Failed to push 1 images" in result.output

    def test_multi_unknown_classifier_rejected(self, tmp_path):
        _write_multi_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.container.push_classifier_alias") as mock_alias,
        ):
            result = runner.invoke(
                app,
                ["push", "--registry", "r.io", "--path", str(tmp_path), "-c", "nope"],
            )
        assert result.exit_code == 1
        assert "'nope' not found" in result.output
        mock_alias.assert_not_called()

    def test_multi_without_release_tag_points_at_merve_tag(self, tmp_path):
        _write_multi_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.GitVersionManager") as mock_gvm,
            patch("mlserver.container.push_classifier_alias") as mock_alias,
        ):
            mgr = mock_gvm.return_value
            mgr.validate_push_readiness.return_value = {"ready": True, "errors": []}
            mgr.get_current_version.return_value = None
            result = runner.invoke(
                app,
                ["push", "--registry", "r.io", "--path", str(tmp_path), "-c", "alpha"],
            )
        assert result.exit_code == 1
        assert "No release tag found" in result.output
        assert "merve tag" in result.output
        mock_alias.assert_not_called()

    def test_multi_alias_push_failure_exits_1(self, tmp_path):
        _write_multi_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.GitVersionManager") as mock_gvm,
            patch("mlserver.container.push_classifier_alias") as mock_alias,
        ):
            mgr = mock_gvm.return_value
            mgr.validate_push_readiness.return_value = {"ready": True, "errors": []}
            mgr.get_current_version.return_value = "1.0.0"
            mock_alias.return_value = {
                "success": False,
                "error": "denied",
                "failed_tags": ["r.io/repo:alpha-v1.0.0"],
            }
            result = runner.invoke(
                app,
                ["push", "--registry", "r.io", "--path", str(tmp_path), "-c", "alpha"],
            )
        assert result.exit_code == 1
        assert "Push failed: denied" in result.output
        assert "r.io/repo:alpha-v1.0.0" in result.output


# ---------------------------------------------------------------------------
# build.py — images
# ---------------------------------------------------------------------------

_IMAGES = [
    {"tag": "solo:latest", "image_id": "abc123def456", "created": "2026-01-01", "size": "100MB"},
    {"tag": "solo:v1.0.0", "image_id": "abc123def456", "created": "2026-01-01", "size": "100MB"},
]


class TestImagesCommand:
    def test_images_human_table(self, tmp_path):
        with patch("mlserver.cli.build.list_images", return_value=list(_IMAGES)) as mock_list:
            result = runner.invoke(app, ["images", "--path", str(tmp_path), "-c", "solo"])
        assert result.exit_code == 0, result.output
        assert mock_list.call_args.kwargs["classifier_name"] == "solo"
        assert "Docker Images" in result.output
        assert "solo:latest" in result.output
        assert "100MB" in result.output

    def test_images_empty_prints_notice(self, tmp_path):
        with patch("mlserver.cli.build.list_images", return_value=[]):
            result = runner.invoke(app, ["images", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "No images found" in result.output

    def test_images_json_document(self, tmp_path):
        with patch("mlserver.cli.build.list_images", return_value=list(_IMAGES)):
            result = runner.invoke(app, ["images", "--path", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["count"] == 2
        assert data["images"][0]["tag"] == "solo:latest"
        assert data["classifier"] is None


# ---------------------------------------------------------------------------
# build.py — clean
# ---------------------------------------------------------------------------


class TestCleanCommand:
    def test_docker_unavailable_exits_1(self, tmp_path):
        with patch("mlserver.cli.build.check_docker_availability", return_value=False):
            result = runner.invoke(app, ["clean", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "Docker is not available" in result.output

    def test_no_images_is_a_noop(self, tmp_path):
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.list_images", return_value=[]),
            patch("mlserver.cli.build.remove_images") as mock_remove,
        ):
            result = runner.invoke(app, ["clean", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "No images to remove" in result.output
        mock_remove.assert_not_called()

    def test_confirmation_cancel_removes_nothing(self, tmp_path):
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.list_images", return_value=list(_IMAGES)),
            patch("mlserver.cli.build.remove_images") as mock_remove,
        ):
            result = runner.invoke(app, ["clean", "--path", str(tmp_path)], input="n\n")
        assert result.exit_code == 0
        assert "will remove 2 images" in result.output
        assert "Cancelled" in result.output
        mock_remove.assert_not_called()

    def test_confirmation_yes_removes(self, tmp_path):
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.list_images", return_value=list(_IMAGES)),
            patch("mlserver.cli.build.remove_images") as mock_remove,
        ):
            mock_remove.return_value = {
                "success": True,
                "removed_images": ["solo:latest", "solo:v1.0.0"],
                "errors": [],
            }
            result = runner.invoke(app, ["clean", "--path", str(tmp_path)], input="y\n")
        assert result.exit_code == 0, result.output
        assert "Removed 2 images" in result.output
        assert "solo:latest" in result.output

    def test_force_skips_prompt_and_filters_classifier(self, tmp_path):
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.list_images", return_value=[_IMAGES[0]]) as mock_list,
            patch("mlserver.cli.build.remove_images") as mock_remove,
        ):
            mock_remove.return_value = {
                "success": True,
                "removed_images": ["solo:latest"],
                "errors": [],
            }
            result = runner.invoke(app, ["clean", "--path", str(tmp_path), "--force", "-c", "solo"])
        assert result.exit_code == 0, result.output
        assert "Are you sure?" not in result.output
        # --classifier flows through list AND remove (RFC 0001 W2.x clean filter)
        assert mock_list.call_args.kwargs["classifier_name"] == "solo"
        assert mock_remove.call_args.kwargs["classifier_name"] == "solo"
        assert mock_remove.call_args.kwargs["force"] is True

    def test_clean_failure_exits_1(self, tmp_path):
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.list_images", return_value=[_IMAGES[0]]),
            patch("mlserver.cli.build.remove_images") as mock_remove,
        ):
            mock_remove.return_value = {"success": False, "error": "daemon went away"}
            result = runner.invoke(app, ["clean", "--path", str(tmp_path), "--force"])
        assert result.exit_code == 1
        assert "Clean failed: daemon went away" in result.output

    def test_clean_partial_removal_errors_exit_nonzero(self, tmp_path):
        # Exit-code contract (RFC 0001 D7): partial failure is failure -
        # same rule push follows for partially failed pushes.
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.list_images", return_value=list(_IMAGES)),
            patch("mlserver.cli.build.remove_images") as mock_remove,
        ):
            mock_remove.return_value = {
                "success": True,
                "removed_images": ["solo:latest"],
                "errors": ["Failed to remove solo:v1.0.0: in use"],
            }
            result = runner.invoke(app, ["clean", "--path", str(tmp_path), "--force"])
        assert result.exit_code == 1
        assert "Removal errors" in result.output
        assert "in use" in result.output

    def test_clean_no_removed_images_prints_message(self, tmp_path):
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.cli.build.list_images", return_value=[_IMAGES[0]]),
            patch("mlserver.cli.build.remove_images") as mock_remove,
        ):
            mock_remove.return_value = {"success": True, "message": "No images found to remove"}
            result = runner.invoke(app, ["clean", "--path", str(tmp_path), "--force"])
        assert result.exit_code == 0
        assert "No images found to remove" in result.output


# ---------------------------------------------------------------------------
# build.py — run
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_multi_requires_classifier(self, tmp_path):
        _write_multi_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            result = runner.invoke(app, ["run", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "specify which classifier" in result.output
        assert "alpha" in result.output and "beta" in result.output
        mock_run.assert_not_called()

    def test_multi_unknown_classifier_rejected(self, tmp_path):
        _write_multi_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            result = runner.invoke(app, ["run", "--path", str(tmp_path), "-c", "nope"])
        assert result.exit_code == 1
        assert "'nope' not found" in result.output
        mock_run.assert_not_called()

    def test_detach_happy_path_plumbs_env_volume_name_version(self, tmp_path):
        _write_project(tmp_path, port=9123)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.version.get_repository_name", return_value="myrepo"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="cid1234567890\n", stderr="")
            result = runner.invoke(
                app,
                [
                    "run",
                    "--path",
                    str(tmp_path),
                    "--detach",
                    "--name",
                    "myct",
                    "--version",
                    "1.2.3",
                    "-e",
                    "FOO=bar",
                    "--volume",
                    "/h:/c",
                    "-p",
                    "9000",
                ],
            )
        assert result.exit_code == 0, result.output
        cmd = mock_run.call_args.args[0]
        assert cmd[:3] == ["docker", "run", "-d"]
        # Host port maps onto the config's server.port
        assert ["-p", "9000:9123"] == cmd[cmd.index("-p") : cmd.index("-p") + 2]
        assert ["--name", "myct"] == cmd[cmd.index("--name") : cmd.index("--name") + 2]
        assert ["-e", "FOO=bar"] == cmd[cmd.index("-e") : cmd.index("-e") + 2]
        assert ["-v", "/h:/c"] == cmd[cmd.index("-v") : cmd.index("-v") + 2]
        assert cmd[-1] == "myrepo:1.2.3"
        assert "Container ID: cid123456789" in result.output
        assert "docker stop" in result.output

    def test_detach_multi_sets_mlserver_classifier_and_autoname(self, tmp_path):
        _write_multi_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.version.get_repository_name", return_value="myrepo"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="cid", stderr="")
            result = runner.invoke(app, ["run", "--path", str(tmp_path), "-c", "beta", "--detach"])
        assert result.exit_code == 0, result.output
        cmd = mock_run.call_args.args[0]
        # Deploy-time selection on the commit image (RFC 0001 D4)
        assert "MLSERVER_CLASSIFIER=beta" in cmd
        name = cmd[cmd.index("--name") + 1]
        assert name.startswith("beta-")
        assert cmd[-1] == "myrepo:latest"

    def test_detach_failure_exits_1_with_stderr(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.version.get_repository_name", return_value="myrepo"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="no such image")
            result = runner.invoke(app, ["run", "--path", str(tmp_path), "--detach"])
        assert result.exit_code == 1
        assert "no such image" in result.output

    def test_interactive_without_tty_swaps_it_for_rm(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.version.get_repository_name", return_value="myrepo"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["run", "--path", str(tmp_path)])
        assert result.exit_code == 0, result.output
        cmd = mock_run.call_args.args[0]
        assert "--rm" in cmd and "-it" not in cmd  # CliRunner stdin is not a TTY
        assert "Container stopped" in result.output

    def test_interactive_nonzero_exit_propagates(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.version.get_repository_name", return_value="myrepo"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=3)
            result = runner.invoke(app, ["run", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "error code: 3" in result.output

    def test_interactive_keyboard_interrupt_is_clean(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.version.get_repository_name", return_value="myrepo"),
            patch("subprocess.run", side_effect=KeyboardInterrupt),
        ):
            result = runner.invoke(app, ["run", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "interrupted" in result.output

    def test_interactive_unexpected_error_exits_1(self, tmp_path):
        _write_project(tmp_path)
        with (
            patch("mlserver.cli.build.check_docker_availability", return_value=True),
            patch("mlserver.version.get_repository_name", return_value="myrepo"),
            patch("subprocess.run", side_effect=RuntimeError("docker exploded")),
        ):
            result = runner.invoke(app, ["run", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "docker exploded" in result.output


# ---------------------------------------------------------------------------
# project.py — list-classifiers / status
# ---------------------------------------------------------------------------


class TestListClassifiersCommand:
    def test_multi_human_table_marks_default(self, tmp_path):
        _write_multi_project(tmp_path)
        result = runner.invoke(app, ["list-classifiers", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Available Classifiers" in result.output
        assert "alpha" in result.output and "beta" in result.output
        assert "Default classifier" in result.output

    def test_single_config_human_notice(self, tmp_path):
        _write_project(tmp_path)
        result = runner.invoke(app, ["list-classifiers", "-C", str(tmp_path)])
        assert result.exit_code == 0
        assert "Not a multi-classifier configuration" in result.output

    def test_missing_config_human_error(self, tmp_path):
        result = runner.invoke(
            app, ["list-classifiers", str(tmp_path / "ghost.yaml"), "-C", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Traceback" not in result.output

    def test_missing_config_json_error(self, tmp_path):
        result = runner.invoke(
            app,
            ["list-classifiers", str(tmp_path / "ghost.yaml"), "-C", str(tmp_path), "--json"],
        )
        assert result.exit_code == 1
        assert "error" in json.loads(result.stdout)


class TestStatusCommand:
    def test_status_human_all_green(self, tmp_path, monkeypatch):
        _write_project(tmp_path)
        monkeypatch.setenv("VIRTUAL_ENV", "/opt/venvs/demo-venv")
        with (
            patch("mlserver.cli.project.check_docker_availability", return_value=True),
            patch("mlserver.github_actions.check_github_actions_setup", return_value=True),
        ):
            result = runner.invoke(app, ["status", "--path", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Merve Status" in result.output
        assert "Available" in result.output
        assert "mlserver.yaml" in result.output
        assert "demo-venv" in result.output
        assert "Configured" in result.output

    def test_status_human_everything_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        with (
            patch("mlserver.cli.project.check_docker_availability", return_value=False),
            patch("mlserver.github_actions.check_github_actions_setup", return_value=False),
        ):
            result = runner.invoke(app, ["status", "--path", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Not available" in result.output
        assert "No config found" in result.output
        assert "Not configured" in result.output

    def test_status_json_respects_path(self, tmp_path, monkeypatch):
        _write_project(tmp_path)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        with (
            patch("mlserver.cli.project.check_docker_availability", return_value=False),
            patch("mlserver.github_actions.check_github_actions_setup", return_value=False),
        ):
            result = runner.invoke(app, ["status", "--path", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["config_file"] == "mlserver.yaml"  # found inside --path
        assert data["docker_available"] is False
        assert data["virtual_env"] is None


# ---------------------------------------------------------------------------
# project.py — init / init-github
# ---------------------------------------------------------------------------


class TestInitCommand:
    def test_init_happy_path_reports_files_and_skips(self, tmp_path):
        with patch("mlserver.init_project.init_mlserver_project") as mock_init:
            mock_init.return_value = (
                True,
                "Project initialized\nSkipped existing files: predictor.py",
                {"config": "mlserver.yaml", "predictor": "predictor.py"},
            )
            result = runner.invoke(
                app,
                ["init", "-C", str(tmp_path), "-c", "sentiment", "--no-github", "--force"],
            )
        assert result.exit_code == 0, result.output
        kwargs = mock_init.call_args.kwargs
        assert kwargs["project_path"] == str(tmp_path)
        assert kwargs["classifier_name"] == "sentiment"
        assert kwargs["include_github_actions"] is False
        assert kwargs["force"] is True
        assert "Created files" in result.output
        assert "Skipped existing files" in result.output
        assert "Next steps" in result.output

    def test_init_failure_exits_1(self, tmp_path):
        with patch("mlserver.init_project.init_mlserver_project") as mock_init:
            mock_init.return_value = (False, "Directory not writable", {})
            result = runner.invoke(app, ["init", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert "Directory not writable" in result.output


class TestInitGithubCommand:
    def test_init_github_happy_path_forwards_options(self, tmp_path):
        with patch("mlserver.github_actions.init_github_actions") as mock_gh:
            mock_gh.return_value = (
                True,
                "Workflow created\ndetails",
                {"workflow": ".github/workflows/build.yml"},
            )
            result = runner.invoke(
                app,
                [
                    "init-github",
                    "-C",
                    str(tmp_path),
                    "--python-version",
                    "3.12",
                    "--registry",
                    "ghcr.io/acme",
                    "--force",
                ],
            )
        assert result.exit_code == 0, result.output
        kwargs = mock_gh.call_args.kwargs
        assert kwargs["project_path"] == str(tmp_path)
        assert kwargs["python_version"] == "3.12"
        assert kwargs["registry"] == "ghcr.io/acme"
        assert kwargs["force"] is True
        assert "Workflow created" in result.output
        assert "Next steps" in result.output

    def test_init_github_failure_exits_1(self, tmp_path):
        with patch("mlserver.github_actions.init_github_actions") as mock_gh:
            mock_gh.return_value = (False, "Not a git repository", {})
            result = runner.invoke(app, ["init-github", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert "Not a git repository" in result.output


# ---------------------------------------------------------------------------
# project.py — validate / doctor
# ---------------------------------------------------------------------------


def _mixed_report():
    return _report(
        CheckResult("YAML syntax", CheckStatus.PASSED, message="parsed fine"),
        CheckResult(
            "Model files",
            CheckStatus.WARNING,
            message="model.pkl is large",
            suggestion="consider trimming",
        ),
        CheckResult("Predictor import", CheckStatus.SKIPPED, message="imports disabled"),
    )


class TestValidateCommand:
    def test_validate_human_happy_path(self, tmp_path):
        _write_project(tmp_path)
        result = runner.invoke(app, ["validate", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Validating" in result.output
        assert "Configuration valid" in result.output

    def test_validate_warnings_pass_without_strict(self, tmp_path):
        _write_project(tmp_path)
        with patch("mlserver.doctor.run_validation_checks", return_value=_mixed_report()):
            result = runner.invoke(app, ["validate", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "model.pkl is large" in result.output
        assert "consider trimming" in result.output
        assert "with warnings" in result.output

    def test_validate_strict_fails_on_warnings(self, tmp_path):
        _write_project(tmp_path)
        with patch("mlserver.doctor.run_validation_checks", return_value=_mixed_report()):
            result = runner.invoke(app, ["validate", "-C", str(tmp_path), "--strict"])
        assert result.exit_code == 1
        assert "strict mode" in result.output

    def test_validate_verbose_shows_messages_and_skipped(self, tmp_path):
        _write_project(tmp_path)
        with patch("mlserver.doctor.run_validation_checks", return_value=_mixed_report()):
            result = runner.invoke(app, ["validate", "-C", str(tmp_path), "--verbose"])
        assert result.exit_code == 0, result.output
        assert "parsed fine" in result.output  # passed-check detail (verbose only)
        assert "imports disabled" in result.output  # skipped check (verbose only)

    def test_validate_errors_fail(self, tmp_path):
        _write_project(tmp_path)
        report = _report(
            CheckResult(
                "Predictor import",
                CheckStatus.FAILED,
                message="No module named 'nope'",
                suggestion="fix predictor.module",
            )
        )
        with patch("mlserver.doctor.run_validation_checks", return_value=report):
            result = runner.invoke(app, ["validate", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert "No module named 'nope'" in result.output
        assert "fix predictor.module" in result.output
        assert "has errors" in result.output

    def test_validate_json_invalid_config_exits_1(self, tmp_path):
        _write_project(tmp_path)
        report = _report(CheckResult("Predictor import", CheckStatus.FAILED, message="boom"))
        with patch("mlserver.doctor.run_validation_checks", return_value=report):
            result = runner.invoke(app, ["validate", "-C", str(tmp_path), "--json"])
        assert result.exit_code == 1
        data = json.loads(result.stdout)
        assert data["valid"] is False
        assert data["errors"] == 1
        assert data["checks"][0]["status"] == "failed"

    def test_validate_no_config_human_error(self, tmp_path):
        result = runner.invoke(app, ["validate", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert "No config file found" in result.output


class TestDoctorCommand:
    def _doctor_report(self):
        return _report(
            CheckResult("Python version", CheckStatus.PASSED, message="3.11"),
            CheckResult(
                "Docker",
                CheckStatus.WARNING,
                message="daemon not running",
                suggestion="start Docker Desktop",
            ),
            CheckResult("Git", CheckStatus.PASSED),
            CheckResult("Config file", CheckStatus.SKIPPED, message="no project here"),
            recommendations=["Start docker before building images"],
        )

    def test_doctor_warnings_exit_zero(self, tmp_path):
        with patch("mlserver.doctor.run_all_checks", return_value=self._doctor_report()):
            result = runner.invoke(app, ["doctor", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "System Checks" in result.output
        assert "Project Checks" in result.output
        assert "daemon not running" in result.output
        assert "start Docker Desktop" in result.output
        assert "Recommendations" in result.output
        assert "there are warnings" in result.output

    def test_doctor_verbose_shows_pass_details_and_skipped(self, tmp_path):
        with patch("mlserver.doctor.run_all_checks", return_value=self._doctor_report()):
            result = runner.invoke(app, ["doctor", "-C", str(tmp_path), "--verbose"])
        assert result.exit_code == 0, result.output
        assert "3.11" in result.output  # passed message only shown with -v
        assert "no project here" in result.output  # skipped only shown with -v

    def test_doctor_failures_exit_1(self, tmp_path):
        report = _report(
            CheckResult("Python version", CheckStatus.PASSED),
            CheckResult(
                "Config file",
                CheckStatus.FAILED,
                message="mlserver.yaml missing",
                suggestion="run merve init",
            ),
        )
        with patch("mlserver.doctor.run_all_checks", return_value=report) as mock_checks:
            result = runner.invoke(app, ["doctor", "-C", str(tmp_path)])
        assert result.exit_code == 1
        assert mock_checks.call_args.args[0] == str(tmp_path)
        assert "mlserver.yaml missing" in result.output
        assert "run merve init" in result.output
        assert "Some checks failed" in result.output

    def test_doctor_all_passed(self, tmp_path):
        report = _report(
            CheckResult("Python version", CheckStatus.PASSED),
            CheckResult("Docker", CheckStatus.PASSED),
        )
        with patch("mlserver.doctor.run_all_checks", return_value=report):
            result = runner.invoke(app, ["doctor", "-C", str(tmp_path)])
        assert result.exit_code == 0
        assert "All checks passed" in result.output


# ---------------------------------------------------------------------------
# project.py — schema
# ---------------------------------------------------------------------------


class TestSchemaCommand:
    def test_schema_prints_to_stdout(self):
        result = runner.invoke(app, ["schema"])
        assert result.exit_code == 0, result.output
        assert "properties" in result.output

    def test_schema_writes_output_file(self, tmp_path):
        out = tmp_path / "schema.json"
        result = runner.invoke(app, ["schema", "-o", str(out), "--type", "single"])
        assert result.exit_code == 0, result.output
        assert "Schema saved to" in result.output
        assert "properties" in out.read_text()

    def test_schema_vscode_merges_over_broken_settings(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        (vscode_dir / "settings.json").write_text("{not valid json")
        result = runner.invoke(app, ["schema", "-o", "schema.json", "--vscode"])
        assert result.exit_code == 0, result.output
        assert "VSCode settings updated" in result.output
        settings = json.loads((vscode_dir / "settings.json").read_text())
        assert "schema.json" in settings["yaml.schemas"]

    def test_schema_setup_prints_instructions(self, tmp_path):
        out = tmp_path / "schema.json"
        result = runner.invoke(app, ["schema", "-o", str(out), "--setup"])
        assert result.exit_code == 0, result.output
        assert "yaml-language-server" in result.output

    def test_schema_rejects_bad_type(self):
        result = runner.invoke(app, ["schema", "--type", "bogus"])
        assert result.exit_code == 1
        assert "Invalid config type" in result.output

    def test_schema_generation_error_exits_1(self):
        with patch(
            "mlserver.schema_generator.get_schema_for_config_type",
            side_effect=RuntimeError("cannot build schema"),
        ):
            result = runner.invoke(app, ["schema"])
        assert result.exit_code == 1
        assert "Failed to generate schema" in result.output


# ---------------------------------------------------------------------------
# serve.py
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _no_env_classifier(monkeypatch):
    monkeypatch.delenv("MLSERVER_CLASSIFIER", raising=False)


class TestServeCommand:
    def test_missing_config_exits_1(self, tmp_path, monkeypatch, _no_env_classifier):
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "No config file found" in result.output
        mock_run.assert_not_called()

    def test_explicit_missing_config_arg_exits_1(self, tmp_path, monkeypatch, _no_env_classifier):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["serve", "ghost.yaml"])
        assert result.exit_code == 1
        assert "ghost.yaml" in result.output

    def test_host_port_log_level_overrides(self, tmp_path, monkeypatch, _no_env_classifier):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run") as mock_run:
            result = runner.invoke(
                app,
                ["serve", "--host", "127.0.0.2", "--port", "7777", "--log-level", "WARNING"],
            )
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args.kwargs
        assert kwargs["host"] == "127.0.0.2"
        assert kwargs["port"] == 7777
        assert kwargs["log_level"] == "warning"
        assert kwargs["workers"] == 1

    def test_single_config_resolves_relative_init_kwargs(
        self, tmp_path, monkeypatch, _no_env_classifier
    ):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with (
            patch("mlserver.cli.serve.uvicorn.run"),
            patch("mlserver.cli.serve.create_app") as mock_create,
        ):
            mock_create.return_value = MagicMock()
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0, result.output
        cfg = mock_create.call_args.args[0]
        # ./model.pkl was resolved against the config directory
        assert cfg.predictor.init_kwargs["model_path"] == str(tmp_path / "model.pkl")

    def test_multi_no_default_and_no_flag_exits_1(self, tmp_path, monkeypatch, _no_env_classifier):
        _write_multi_project(tmp_path, default=None)
        monkeypatch.chdir(tmp_path)
        with (
            patch("mlserver.cli.serve.get_default_classifier", return_value=None),
            patch("mlserver.cli.serve.uvicorn.run") as mock_run,
        ):
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "No classifier specified and no default configured" in result.output
        assert "alpha" in result.output and "beta" in result.output
        mock_run.assert_not_called()

    def test_multi_unknown_classifier_flag_errors(self, tmp_path, monkeypatch, _no_env_classifier):
        _write_multi_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve", "--classifier", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.output
        mock_run.assert_not_called()

    def test_multi_extract_error_debug_shows_traceback(
        self, tmp_path, monkeypatch, _no_env_classifier
    ):
        _write_multi_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch(
            "mlserver.cli.serve.extract_single_classifier_config",
            side_effect=ValueError("bad classifier block"),
        ):
            result = runner.invoke(app, ["serve", "-c", "alpha", "--log-level", "DEBUG"])
        assert result.exit_code == 1
        assert "bad classifier block" in result.output
        assert "Full traceback" in result.output

    def test_multi_unexpected_extract_error(self, tmp_path, monkeypatch, _no_env_classifier):
        _write_multi_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch(
            "mlserver.cli.serve.extract_single_classifier_config",
            side_effect=RuntimeError("corrupt state"),
        ):
            result = runner.invoke(app, ["serve", "-c", "alpha"])
        assert result.exit_code == 1
        assert "Unexpected error" in result.output
        assert "corrupt state" in result.output

    def test_multi_resolves_selected_classifier_init_kwargs(
        self, tmp_path, monkeypatch, _no_env_classifier
    ):
        _write_multi_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with (
            patch("mlserver.cli.serve.uvicorn.run"),
            patch("mlserver.cli.serve.create_app") as mock_create,
        ):
            mock_create.return_value = MagicMock()
            result = runner.invoke(app, ["serve", "-c", "alpha"])
        assert result.exit_code == 0, result.output
        cfg = mock_create.call_args.args[0]
        assert cfg.predictor.init_kwargs["model_path"] == str(tmp_path / "model.pkl")

    def test_keyboard_interrupt_is_clean_stop(self, tmp_path, monkeypatch, _no_env_classifier):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run", side_effect=KeyboardInterrupt):
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert "Server stopped by user" in result.output

    def test_uvicorn_crash_exits_1(self, tmp_path, monkeypatch, _no_env_classifier):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run", side_effect=RuntimeError("bind failed")):
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "bind failed" in result.output


# ---------------------------------------------------------------------------
# testing.py — merve test
# ---------------------------------------------------------------------------


def _fake_httpx_client(response=None, post_side_effect=None):
    client = MagicMock()
    entered = client.__enter__.return_value
    if post_side_effect is not None:
        entered.post.side_effect = post_side_effect
    else:
        entered.post.return_value = response
    return client


def _ok_response(payload=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.reason_phrase = "OK"
    resp.json.return_value = payload if payload is not None else {"predictions": [1]}
    return resp


class TestTestCommand:
    def test_bare_dict_is_wrapped_in_records(self):
        client = _fake_httpx_client(_ok_response())
        with patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["test", "--data", '{"f1": 1.5}'])
        assert result.exit_code == 0, result.output
        call = client.__enter__.return_value.post.call_args
        assert call.args[0] == "http://localhost:8000/predict"
        assert call.kwargs["json"] == {"records": [{"f1": 1.5}]}

    def test_bare_list_is_wrapped_in_records(self):
        client = _fake_httpx_client(_ok_response())
        with patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["test", "--data", '[{"f1": 1}, {"f1": 2}]'])
        assert result.exit_code == 0, result.output
        call = client.__enter__.return_value.post.call_args
        assert call.kwargs["json"] == {"records": [{"f1": 1}, {"f1": 2}]}

    def test_shaped_payload_passes_through_untouched(self):
        client = _fake_httpx_client(_ok_response())
        with patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["test", "--data", '{"ndarray": [[1, 2]]}'])
        assert result.exit_code == 0, result.output
        call = client.__enter__.return_value.post.call_args
        assert call.kwargs["json"] == {"ndarray": [[1, 2]]}

    def test_file_payload_and_custom_endpoint(self, tmp_path):
        req = tmp_path / "req.json"
        req.write_text('{"records": [{"f1": 9}]}')
        client = _fake_httpx_client(_ok_response())
        with patch("httpx.Client", return_value=client):
            result = runner.invoke(
                app,
                [
                    "test",
                    "--file",
                    str(req),
                    "--url",
                    "http://h:1/",
                    "--endpoint",
                    "/predict_proba",
                ],
            )
        assert result.exit_code == 0, result.output
        call = client.__enter__.return_value.post.call_args
        assert call.args[0] == "http://h:1/predict_proba"
        assert call.kwargs["json"] == {"records": [{"f1": 9}]}

    def test_invalid_json_data_exits_1(self):
        result = runner.invoke(app, ["test", "--data", "{not json"])
        assert result.exit_code == 1
        assert "Invalid JSON data" in result.output

    def test_missing_file_exits_1(self, tmp_path):
        result = runner.invoke(app, ["test", "--file", str(tmp_path / "ghost.json")])
        assert result.exit_code == 1
        assert "File not found" in result.output

    def test_invalid_json_file_exits_1(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{oops")
        result = runner.invoke(app, ["test", "--file", str(bad)])
        assert result.exit_code == 1
        assert "Invalid JSON in file" in result.output

    def test_neither_data_nor_file_exits_1(self):
        result = runner.invoke(app, ["test"])
        assert result.exit_code == 1
        assert "Either --data or --file is required" in result.output

    def test_non_200_response_exits_1(self):
        resp = MagicMock()
        resp.status_code = 422
        resp.reason_phrase = "Unprocessable Entity"
        resp.json.return_value = {"detail": "missing feature f2"}
        with patch("httpx.Client", return_value=_fake_httpx_client(resp)):
            result = runner.invoke(app, ["test", "--data", '{"f1": 1}'])
        assert result.exit_code == 1
        assert "422" in result.output
        assert "missing feature f2" in result.output

    def test_raw_output_is_compact_json(self):
        with patch("httpx.Client", return_value=_fake_httpx_client(_ok_response())):
            result = runner.invoke(app, ["test", "--data", '{"f1": 1}', "--raw"])
        assert result.exit_code == 0, result.output
        assert '{"predictions": [1]}' in result.output

    def test_non_json_response_prints_text(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.reason_phrase = "OK"
        resp.json.side_effect = json.JSONDecodeError("nope", "doc", 0)
        resp.text = "plain body response"
        with patch("httpx.Client", return_value=_fake_httpx_client(resp)):
            result = runner.invoke(app, ["test", "--data", '{"f1": 1}'])
        assert result.exit_code == 0, result.output
        assert "plain body response" in result.output

    def test_timeout_exits_1(self):
        import httpx

        client = _fake_httpx_client(post_side_effect=httpx.TimeoutException("slow"))
        with patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["test", "--data", '{"f1": 1}'])
        assert result.exit_code == 1
        assert "timed out" in result.output

    def test_unexpected_error_exits_1(self):
        client = _fake_httpx_client(post_side_effect=RuntimeError("kaput"))
        with patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["test", "--data", '{"f1": 1}'])
        assert result.exit_code == 1
        assert "Request failed" in result.output
        assert "kaput" in result.output
