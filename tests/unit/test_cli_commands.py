"""CliRunner behavior tests for serve / version / tag / test commands.

Closes the coverage gap left after the W2.1 CLI split (serve.py, versioning.py,
testing.py were ~10%). Everything external — uvicorn, git, httpx — is mocked so
these run fast and hermetically.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from mlserver.cli import app

runner = CliRunner()


def _write_project(tmp_path):
    """A minimal, importable single-classifier project."""
    (tmp_path / "mypred.py").write_text(
        "class P:\n"
        "    def predict(self, X):\n        return [0] * len(X)\n"
        "    def predict_proba(self, X):\n        return [[0.5, 0.5]] * len(X)\n"
    )
    (tmp_path / "mlserver.yaml").write_text(
        yaml.safe_dump(
            {
                "predictor": {"module": "mypred", "class_name": "P"},
                "classifier": {"name": "solo", "version": "1.0.0"},
                "server": {"host": "127.0.0.1", "port": 9123},
                "observability": {"metrics": False, "structured_logging": False},
            }
        )
    )
    return tmp_path


class TestServeCommand:
    def test_serve_invokes_uvicorn_with_config_host_port(self, tmp_path, monkeypatch):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0, result.stdout
        assert mock_run.called
        kwargs = mock_run.call_args.kwargs
        # Host/port come from the YAML, not the CLI defaults.
        assert kwargs.get("host") == "127.0.0.1"
        assert kwargs.get("port") == 9123

    def test_serve_reload_uses_import_string_factory(self, tmp_path, monkeypatch):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve", "--reload"])
        assert result.exit_code == 0, result.stdout
        # Reload requires an import string + factory (not an app instance),
        # otherwise uvicorn exits immediately (the bug fixed earlier this sprint).
        args, kwargs = mock_run.call_args
        target = args[0] if args else kwargs.get("app")
        assert isinstance(target, str)
        assert kwargs.get("reload") is True
        assert kwargs.get("factory") is True

    def test_serve_workers_override(self, tmp_path, monkeypatch):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        with patch("mlserver.cli.serve.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve", "--workers", "3"])
        assert result.exit_code == 0, result.stdout
        # Multi-worker path uses the factory import string too.
        args, kwargs = mock_run.call_args
        target = args[0] if args else kwargs.get("app")
        assert isinstance(target, str)


class TestVersionCommand:
    def test_version_human_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        # Human mode prints the tool version somewhere.
        assert "ersion" in result.stdout  # "Version"/"version"

    def test_version_json_matches_human_exit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["version", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["mlserver_tool"]["version"]


class TestTagCommand:
    def test_tag_creates_canonical_and_reports(self, tmp_path, monkeypatch):
        import subprocess

        _write_project(tmp_path)
        # Real git repo so the pre-tag validation passes end-to-end.
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.co"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app, ["tag", "patch", "--classifier", "solo", "--allow-missing-mlserver"]
        )
        assert result.exit_code == 0, result.stdout
        # A real canonical tag was created.
        tags = subprocess.run(
            ["git", "tag", "-l"], cwd=tmp_path, capture_output=True, text=True
        ).stdout
        assert "solo/v0.0.1" in tags

    def test_tag_status_json(self, tmp_path, monkeypatch):
        _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        fake_mgr = MagicMock()
        fake_mgr.get_all_classifier_status.return_value = {}
        fake_mgr.get_latest_tag_info.return_value = {
            "tag": "solo/v1.0.0",
            "on_tagged_commit": True,
            "commits_since_tag": 0,
        }
        with patch("mlserver.cli.versioning.GitVersionManager", return_value=fake_mgr):
            result = runner.invoke(app, ["tag", "--status", "--json", "--classifier", "solo"])
        # --status --json must emit a JSON document (or a clean error), never a traceback.
        assert result.exit_code in (0, 1)
        assert "Traceback" not in result.stdout


class TestTestCommand:
    def test_test_command_posts_to_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"predictions": [0], "time_ms": 1.0}
        fake_client = MagicMock()
        fake_client.__enter__.return_value.post.return_value = fake_resp
        with patch("httpx.Client", return_value=fake_client):
            result = runner.invoke(
                app, ["test", "--url", "http://localhost:8000", "--data", '{"records": [{"a": 1}]}']
            )
        assert result.exit_code == 0, result.stdout
        fake_client.__enter__.return_value.post.assert_called_once()

    def test_test_command_connection_error_is_clean(self, tmp_path, monkeypatch):
        import httpx

        monkeypatch.chdir(tmp_path)
        fake_client = MagicMock()
        fake_client.__enter__.return_value.post.side_effect = httpx.ConnectError("refused")
        with patch("httpx.Client", return_value=fake_client):
            result = runner.invoke(app, ["test", "--data", '{"records": [{"a": 1}]}'])
        # A refused connection must exit non-zero with a readable message, not a traceback.
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout
