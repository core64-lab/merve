"""CliRunner tests for the machine-readable ``--json`` output of read commands.

Covers RFC 0001 W1.5 (D7): every read command must emit exactly one JSON
document to stdout with stable snake_case keys and the same exit code as its
human-readable mode. These tests guard the CI-consumable contract — the
generated GitHub Actions workflows parse this output.
"""
from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from mlserver.cli import app

runner = CliRunner()


def _write(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def single_config(tmp_path):
    """A minimal single-classifier config on disk."""
    _write(tmp_path / "mlserver.yaml", {
        "predictor": {"module": "m", "class_name": "C"},
        "classifier": {"name": "solo", "version": "1.0.0"},
    })
    return tmp_path


@pytest.fixture
def multi_config(tmp_path):
    """A two-classifier config (dict format) on disk."""
    _write(tmp_path / "mlserver.yaml", {
        "classifiers": {
            "alpha": {"predictor": {"module": "m", "class_name": "A"}},
            "beta": {"predictor": {"module": "m", "class_name": "B"}},
        },
        "default_classifier": "beta",
    })
    return tmp_path


def _json_from(result):
    """Assert the command emitted exactly one parseable JSON document."""
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


class TestStatusJson:
    def test_status_json_is_valid_document(self):
        data = _json_from(runner.invoke(app, ["status", "--json"]))
        for key in (
            "docker_available", "config_file", "python_version",
            "virtual_env", "github_actions_configured",
        ):
            assert key in data
        assert isinstance(data["docker_available"], bool)

    def test_status_json_no_rich_markup(self):
        # stdout must be pure JSON — no banners, tables, or ANSI markup.
        out = runner.invoke(app, ["status", "--json"]).stdout.strip()
        assert out.startswith("{") and out.endswith("}")


class TestListClassifiersJson:
    def test_multi_classifier(self, multi_config, monkeypatch):
        monkeypatch.chdir(multi_config)
        data = _json_from(runner.invoke(app, ["list-classifiers", "--json"]))
        assert data["multi_classifier"] is True
        assert sorted(data["classifiers"]) == ["alpha", "beta"]
        assert data["default_classifier"] == "beta"

    def test_single_classifier_reports_not_multi(self, single_config, monkeypatch):
        monkeypatch.chdir(single_config)
        data = _json_from(runner.invoke(app, ["list-classifiers", "--json"]))
        assert data["multi_classifier"] is False
        assert data["classifiers"] == []


class TestValidateJson:
    def test_valid_config_reports_valid(self, tmp_path, monkeypatch):
        # A predictor that imports cleanly so validation passes end-to-end.
        (tmp_path / "mypred.py").write_text(
            "class P:\n    def predict(self, X):\n        return [0] * len(X)\n"
        )
        _write(tmp_path / "mlserver.yaml", {
            "predictor": {"module": "mypred", "class_name": "P"},
            "classifier": {"name": "v", "version": "1.0.0"},
        })
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["validate", "--json"])
        data = json.loads(result.stdout)
        assert data["valid"] is True
        assert result.exit_code == 0

    def test_missing_config_emits_json_error_and_nonzero_exit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # empty dir, no mlserver.yaml
        result = runner.invoke(app, ["validate", "nonexistent.yaml", "--json"])
        assert result.exit_code == 1
        # Even the error path must stay machine-readable.
        assert json.loads(result.stdout)


class TestImagesJson:
    def test_images_json_when_docker_unavailable(self, monkeypatch):
        # No daemon assumption: force docker-unavailable and assert structured output.
        monkeypatch.setattr("mlserver.cli.check_docker_availability", lambda: False)
        result = runner.invoke(app, ["images", "--json"])
        # Command must not crash and must emit JSON regardless of daemon state.
        assert result.exit_code in (0, 1)
        payload = result.stdout.strip()
        assert payload.startswith("{") or payload.startswith("[")
        json.loads(payload)


class TestVersionJson:
    def test_version_json_document(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["version", "--json"])
        data = json.loads(result.stdout)
        # Tool metadata is nested under "mlserver_tool".
        assert "mlserver_tool" in data
        assert "version" in data["mlserver_tool"]
