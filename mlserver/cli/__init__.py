"""Modern CLI for ML Server using Typer.

The CLI is split across command modules (``serve``, ``build``, ``versioning``,
``project``, ``testing``). Importing them here registers their ``@app.command``
handlers on the shared Typer ``app`` via import side effects.

``app`` and ``main`` are the public entry points (``mlserver.cli:main`` is the
console-script target in pyproject.toml). ``detect_config_file`` and the path
helpers are re-exported so ``from mlserver.cli import detect_config_file`` keeps
working for internal callers (e.g. ``mlserver.container``).
"""

from . import build, project, serve, testing, versioning  # noqa: F401  (registration side effects)
from ._app import (
    _is_likely_file_path,
    _resolve_path,
    app,
    console,
    detect_config_file,
    resolve_relative_paths,
)


def main():
    """Main entry point."""
    app()


__all__ = [
    "app",
    "console",
    "detect_config_file",
    "main",
    "resolve_relative_paths",
    "_is_likely_file_path",
    "_resolve_path",
]
