"""Shared Typer app, consoles, enums, and path/config helpers for the CLI.

Command modules under this package import ``app`` (and the shared helpers) from
here and attach their handlers with ``@app.command()``. Keeping the ``Typer``
instance in a dedicated module avoids circular imports between the command
modules and the package ``__init__``.
"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Create the main app
app = typer.Typer(
    name="merve",
    help="🚀 Merve - wrap Python predictors into FastAPI inference APIs",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)

console = Console()
# Deprecation/diagnostic messages go to stderr so --json output on stdout
# stays a single parseable document
err_console = Console(stderr=True)


class LogLevel(str, Enum):
    """Log level options."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def resolve_relative_paths(init_kwargs: dict, config_dir: str) -> dict:
    """Resolve relative file paths in init_kwargs to be relative to config file location."""
    resolved_kwargs = {}

    for key, value in init_kwargs.items():
        if isinstance(value, str) and _is_likely_file_path(key, value):
            resolved_kwargs[key] = _resolve_path(value, config_dir)
        else:
            resolved_kwargs[key] = value

    return resolved_kwargs


def _is_likely_file_path(key: str, value: str) -> bool:
    """Check if a key-value pair likely represents a file path."""
    path_indicators = ["path", "file", "model", "preprocessor", "feature_order"]
    key_suggests_path = any(indicator in key.lower() for indicator in path_indicators)

    is_relative = (
        value.startswith("./")
        or value.startswith("../")
        or (not value.startswith("/") and ":" not in value[:3])
    )

    return key_suggests_path and is_relative


def _resolve_path(path: str, config_dir: str) -> str:
    """Resolve a single path relative to config directory."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(config_dir, path))


def detect_config_file(config_path: Optional[Path], base_dir: Optional[Path] = None) -> Path:
    """Detect which config file to use.

    Args:
        config_path: Explicit config file path provided by the user (used as-is)
        base_dir: Directory to search for the default mlserver.yaml. When
                  provided, the returned path is absolute so callers can use it
                  regardless of their working directory. Defaults to CWD.
    """
    if config_path and config_path.exists():
        return config_path

    # Check default location (inside base_dir when provided)
    if base_dir:
        default_config = Path(base_dir).resolve() / "mlserver.yaml"
    else:
        default_config = Path("mlserver.yaml")
    if default_config.exists():
        return default_config

    # Check if specified path exists
    if config_path:
        raise typer.BadParameter(f"Config file not found: {config_path}")

    raise typer.BadParameter(
        "No config file found. Please specify with --config or create mlserver.yaml"
    )


def removed_flag_callback(pointer: str):
    """Build a parse-time callback for a removed flag (RFC 0001 D8).

    Removed flags are declared as hidden options so that using one fails
    with exit code 2 AND a pointer to the replacement — never a bare
    "No such option" and never a silent reinterpretation.
    (typer.BadParameter is the only exception type Typer converts to a
    usage error with exit code 2 from inside a parameter callback.)
    """

    def _callback(value: Optional[str]) -> Optional[str]:
        if value is not None:
            raise typer.BadParameter(pointer)
        return value

    return _callback


def _display_check_result(check, verbose: bool = False):
    """Helper to display a check result with consistent formatting."""
    from ..doctor import CheckStatus

    if check.status == CheckStatus.PASSED:
        console.print(
            f"  [green]✓[/green] {check.name}"
            + (f": {check.message}" if check.message and verbose else "")
        )
    elif check.status == CheckStatus.WARNING:
        console.print(f"  [yellow]⚠[/yellow] {check.name}: {check.message}")
        if check.suggestion:
            console.print(f"    [dim]→ {check.suggestion}[/dim]")
    elif check.status == CheckStatus.FAILED:
        console.print(f"  [red]✗[/red] {check.name}: {check.message}")
        if check.suggestion:
            console.print(f"    [dim]→ {check.suggestion}[/dim]")
    elif check.status == CheckStatus.SKIPPED and verbose:
        console.print(f"  [dim]○[/dim] {check.name}: {check.message}")


@app.callback()
def main_callback():
    """🚀 Merve - Modern CLI for ML model serving."""
    # Deprecation notice when invoked via the legacy `mlserver` alias (RFC 0001
    # D9). The command was renamed `merve` to avoid colliding with Seldon's
    # `mlserver`; the alias is kept for one transition release.
    invoked_as = os.path.basename(sys.argv[0]) if sys.argv else ""
    if invoked_as == "mlserver":
        err_console.print(
            "[yellow]⚠ The 'mlserver' command is deprecated; use 'merve' instead "
            "(the package was renamed to avoid a collision with Seldon's mlserver). "
            "This alias will be removed in a future release.[/yellow]"
        )
