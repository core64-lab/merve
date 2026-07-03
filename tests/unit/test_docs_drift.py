"""Unit tests for scripts/check_docs_drift.py (the CI docs-drift gate)."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_docs_drift.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_docs_drift", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drift = _load_module()


def make_clean_tree(root: Path) -> None:
    """Create a minimal fake repo tree containing no banned tokens."""
    (root / "docs").mkdir()
    (root / "examples").mkdir()
    (root / "README.md").write_text("# Demo\n\nUse `mlserver serve` to start.\n")
    (root / "CLAUDE.md").write_text("Project notes for agents.\n")
    (root / "docs" / "guide.md").write_text("Call POST /predict with records.\n")
    (root / "examples" / "demo.py").write_text("print('hello')\n")


def test_clean_tree_exits_zero(tmp_path, capsys):
    make_clean_tree(tmp_path)
    assert drift.main(["--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


BANNED_SAMPLES = [
    ("ml_server", "from ml_server import create_app"),
    ("cli_v2", "the new CLI lives in mlserver.cli_v2"),
    ("ainit", "run mlserver ainit notebook.ipynb to bootstrap"),
    ("/readyz", "probe the /readyz endpoint for readiness"),
    ("/startupz", "probe the /startupz endpoint at boot"),
    ("batch_predict", "POST /batch_predict accepts many records"),
    ("max_concurrent_requests", "set max_concurrent_requests: 4"),
]


@pytest.mark.parametrize("label,line", BANNED_SAMPLES, ids=[s[0] for s in BANNED_SAMPLES])
def test_banned_pattern_detected_with_file_and_line(tmp_path, capsys, label, line):
    make_clean_tree(tmp_path)
    # Two clean lines first, so the planted token sits on line 3.
    (tmp_path / "docs" / "bad.md").write_text(f"# Title\n\n{line}\n")
    assert drift.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{Path('docs') / 'bad.md'}:3: {label}" in out


def test_multiple_hits_all_reported(tmp_path, capsys):
    make_clean_tree(tmp_path)
    (tmp_path / "docs" / "bad.md").write_text("use cli_v2 here\nand probe /readyz there\n")
    assert drift.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{Path('docs') / 'bad.md'}:1: cli_v2" in out
    assert f"{Path('docs') / 'bad.md'}:2: /readyz" in out


def test_readme_at_repo_root_is_scanned(tmp_path, capsys):
    make_clean_tree(tmp_path)
    (tmp_path / "README.md").write_text("POST /batch_predict is great\n")
    assert drift.main(["--root", str(tmp_path)]) == 1
    assert "README.md:1: batch_predict" in capsys.readouterr().out


def test_nested_example_files_are_scanned(tmp_path, capsys):
    make_clean_tree(tmp_path)
    nested = tmp_path / "examples" / "sub"
    nested.mkdir()
    (nested / "config.yaml").write_text("api:\n  max_concurrent_requests: 8\n")
    assert drift.main(["--root", str(tmp_path)]) == 1
    expected = Path("examples") / "sub" / "config.yaml"
    assert f"{expected}:2: max_concurrent_requests" in capsys.readouterr().out


def test_docs_archive_is_ignored(tmp_path):
    make_clean_tree(tmp_path)
    archive = tmp_path / "docs" / "archive"
    archive.mkdir()
    (archive / "old.md").write_text(
        "cli_v2 and ainit and /readyz and /startupz\n"
        "ml_server batch_predict max_concurrent_requests\n"
    )
    assert drift.main(["--root", str(tmp_path)]) == 0


def test_docs_rfcs_are_ignored(tmp_path):
    make_clean_tree(tmp_path)
    rfcs = tmp_path / "docs" / "rfcs"
    rfcs.mkdir()
    (rfcs / "0001-plan.md").write_text("gate bans `batch_predict` and `ainit`\n")
    assert drift.main(["--root", str(tmp_path)]) == 0


def test_word_boundary_near_misses_not_flagged(tmp_path):
    make_clean_tree(tmp_path)
    (tmp_path / "docs" / "near_miss.md").write_text(
        "mlserver serve starts the mlserver process\n"
        "the html_server module is unrelated\n"
        "maintain the maintainer's maintainability\n"
        "mainit is not the removed command\n"
        "batch_predictions and batch_prediction_results are fine\n"
    )
    assert drift.main(["--root", str(tmp_path)]) == 0


def test_non_text_extensions_are_ignored(tmp_path):
    make_clean_tree(tmp_path)
    (tmp_path / "docs" / "notes.rst").write_text("legacy /readyz mention\n")
    (tmp_path / "docs" / "blob.bin").write_text("batch_predict\n")
    assert drift.main(["--root", str(tmp_path)]) == 0
