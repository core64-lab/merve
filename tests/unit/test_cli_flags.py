"""CliRunner tests for the W2.2 flag contract and W2.1 command registration.

RFC 0001 D8 (Wave 2) makes each short flag mean exactly one thing — a documented
breaking change:

  * ``-p`` is ``--port`` ONLY (serve, run). It is no longer a ``--path`` alias.
  * ``--path`` uses ``-C`` (git/make ``-C <dir>`` convention) on version, init,
    init-github, and doctor.
  * ``-v`` is ``--verbose`` ONLY (build, validate, doctor); ``run``'s ``--volume``
    is long-only (its old ``-v`` short flag is dropped).
  * ``--classifier`` / ``-c`` is unchanged everywhere.

Removed short flags are NOT silently reinterpreted: Typer/Click rejects them with
"No such option" and exit code 2, which is the intended migration signal.

The registration test guards the W2.1 package split: the command handlers live in
separate modules under ``mlserver/cli/`` and are wired onto the shared ``app`` via
import side effects in ``mlserver/cli/__init__.py``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from mlserver.cli import app

runner = CliRunner()

# The full command surface (canonical CLI names, hyphenated where applicable).
EXPECTED_COMMANDS = {
    "serve",
    "version",
    "build",
    "push",
    "images",
    "tag",
    "clean",
    "run",
    "list-classifiers",
    "status",
    "init",
    "init-github",
    "validate",
    "doctor",
    "test",
    "schema",
}

# Path-taking commands whose long flag is --path with the -C short flag (W2.2).
# doctor's function parameter is named ``project_path``; the rest use ``path``.
PATH_COMMANDS = {
    "version": "path",
    "init": "path",
    "init-github": "path",
    "doctor": "project_path",
}


def _opts(command_name: str, param_name: str) -> set[str]:
    """Return the set of option strings declared for a command's parameter."""
    root = typer.main.get_command(app)
    sub = root.commands[command_name]
    for param in sub.params:
        if param.name == param_name:
            return set(param.opts) | set(param.secondary_opts)
    raise AssertionError(f"{command_name!r} has no parameter {param_name!r}")


class TestCommandRegistration:
    """W2.1: every command survives the split into command modules."""

    def test_all_commands_present(self):
        """`--help` lists all 16 commands (guards the registration side effects)."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for name in sorted(EXPECTED_COMMANDS):
            assert name in result.output, f"{name} missing from --help"

    def test_command_table_is_exactly_the_expected_set(self):
        """No commands lost or duplicated by the package restructure."""
        root = typer.main.get_command(app)
        assert set(root.commands.keys()) == EXPECTED_COMMANDS


class TestFlagContractIntrospection:
    """Deterministic assertion of the whole W2.2 short-flag contract."""

    def test_p_is_port_only(self):
        assert _opts("serve", "port") == {"--port", "-p"}
        assert _opts("run", "port") == {"--port", "-p"}

    def test_path_uses_C_not_p(self):
        for cmd, param in PATH_COMMANDS.items():
            opts = _opts(cmd, param)
            assert "--path" in opts, f"{cmd} lost --path"
            assert "-C" in opts, f"{cmd} missing -C short flag"
            assert "-p" not in opts, f"{cmd} still maps -p to --path"

    def test_v_is_verbose_only(self):
        for cmd in ("build", "validate", "doctor"):
            assert _opts(cmd, "verbose") == {"--verbose", "-v"}

    def test_run_volume_is_long_only(self):
        assert _opts("run", "volume") == {"--volume"}

    def test_classifier_short_flag_unchanged(self):
        for cmd in ("serve", "build", "version", "run", "push", "images"):
            assert _opts(cmd, "classifier") == {"--classifier", "-c"}


class TestPortFlag:
    """`-p` resolves to the integer --port on serve and run (parse-level proof)."""

    @pytest.mark.parametrize("command", ["serve", "run"])
    def test_p_maps_to_integer_port(self, command):
        result = runner.invoke(app, [command, "-p", "notanint"])
        assert result.exit_code == 2
        assert "Invalid value" in result.output
        assert "--port" in result.output  # -p is bound to --port
        assert "No such option" not in result.output


class TestPathFlag:
    """`--path`/`-C` replaces the old `--path -p` on version/init/init-github/doctor."""

    @pytest.mark.parametrize("command", list(PATH_COMMANDS))
    def test_p_rejected_as_path(self, command, tmp_path):
        # -p is no longer a path alias on these commands: hard error, no reinterpret.
        result = runner.invoke(app, [command, "-p", str(tmp_path)])
        assert result.exit_code == 2
        assert "No such option" in result.output

    def test_version_C_is_path(self, tmp_path):
        result = runner.invoke(app, ["version", "-C", str(tmp_path), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "mlserver_tool" in data  # no project at tmp_path -> tool-only version

    def test_init_C_is_path(self, tmp_path):
        with patch("mlserver.init_project.init_mlserver_project") as mock_init:
            mock_init.return_value = (True, "Created", {"config": "mlserver.yaml"})
            result = runner.invoke(app, ["init", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert mock_init.call_args.kwargs["project_path"] == str(tmp_path)

    def test_init_github_C_is_path(self, tmp_path):
        with patch("mlserver.github_actions.init_github_actions") as mock_gh:
            mock_gh.return_value = (True, "Created workflow\n(details)", {"wf": "x.yml"})
            result = runner.invoke(app, ["init-github", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert mock_gh.call_args.kwargs["project_path"] == str(tmp_path)

    def test_doctor_C_is_path(self, tmp_path):
        report = MagicMock(checks=[], recommendations=[], has_errors=False, has_warnings=False)
        with patch("mlserver.doctor.run_all_checks", return_value=report) as mock_checks:
            result = runner.invoke(app, ["doctor", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert mock_checks.call_args.args[0] == str(tmp_path)


class TestVerboseFlag:
    """`-v` means --verbose only (build, validate, doctor)."""

    def test_build_v_is_verbose(self, tmp_path):
        # Docker forced unavailable so build stops early; we only assert -v parsed.
        with patch("mlserver.cli.build.check_docker_availability", return_value=False):
            result = runner.invoke(app, ["build", "-v", "--path", str(tmp_path)])
        assert result.exit_code == 1
        assert "No such option" not in result.output

    def test_validate_v_is_verbose(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # empty dir -> config-not-found, but -v must parse
        result = runner.invoke(app, ["validate", "-v"])
        assert result.exit_code == 1
        assert "No such option" not in result.output

    def test_doctor_v_is_verbose(self, tmp_path):
        report = MagicMock(checks=[], recommendations=[], has_errors=False, has_warnings=False)
        with patch("mlserver.doctor.run_all_checks", return_value=report):
            result = runner.invoke(app, ["doctor", "-v", "-C", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "No such option" not in result.output


class TestRunVolumeFlag:
    """`run` volume is long-only; its old `-v` short flag is gone."""

    def test_volume_long_form_accepted(self, tmp_path):
        with patch("mlserver.cli.build.check_docker_availability", return_value=False):
            result = runner.invoke(app, ["run", "--volume", "/host:/cont", "--path", str(tmp_path)])
        assert result.exit_code == 1  # docker unavailable, but --volume parsed fine
        assert "No such option" not in result.output

    def test_v_no_longer_maps_to_volume(self):
        result = runner.invoke(app, ["run", "-v", "/host:/cont"])
        assert result.exit_code == 2
        assert "No such option" in result.output


class TestClassifierFlag:
    """`--classifier` / `-c` is unchanged by the flag pass."""

    def test_c_short_flag_accepted(self, tmp_path):
        result = runner.invoke(app, ["version", "-c", "mymodel", "-C", str(tmp_path), "--json"])
        assert result.exit_code == 0, result.output

    def test_classifier_long_flag_accepted(self, tmp_path):
        result = runner.invoke(
            app, ["version", "--classifier", "mymodel", "-C", str(tmp_path), "--json"]
        )
        assert result.exit_code == 0, result.output
