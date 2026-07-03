"""Modern CLI for ML Server using Typer."""

import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
import uvicorn
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import AppConfig
from .container import build_container, check_docker_availability, list_images, remove_images
from .logging_conf import configure_logging
from .multi_classifier import (
    detect_multi_classifier_config,
    extract_single_classifier_config,
    get_default_classifier,
    list_available_classifiers,
    load_multi_classifier_config,
)
from .server import create_app
from .version import get_version_info
from .version_control import GitVersionManager, VersionControlError, safe_push_container

# Create the main app
app = typer.Typer(
    name="mlserver",
    help="🚀 ML Server - FastAPI-based ML model serving",
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


class BumpType(str, Enum):
    """Version bump type options."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


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
    path_indicators = ['path', 'file', 'model', 'preprocessor', 'feature_order']
    key_suggests_path = any(indicator in key.lower() for indicator in path_indicators)

    is_relative = (
        value.startswith('./') or
        value.startswith('../') or
        (not value.startswith('/') and ':' not in value[:3])
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


@app.command()
def serve(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to config file (defaults to mlserver.yaml)",
        exists=False,  # We'll check existence ourselves
    ),
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier", "-c",
        help="Classifier to serve (for multi-classifier configs)"
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Override host address"
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port", "-p",
        help="Override port number"
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers", "-w",
        help="Number of worker processes"
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development"
    ),
    log_level: Optional[LogLevel] = typer.Option(
        None,
        "--log-level", "-l",
        help="Set log level (defaults to server.log_level from config)"
    ),
):
    """🚀 Launch ML FastAPI server from YAML config."""
    try:
        # Detect config file
        config_file = detect_config_file(config)
        console.print(f"[green]✓[/green] Using configuration: [cyan]{config_file}[/cyan]")

        # Get config directory for path resolution
        config_path = config_file.resolve()
        config_dir = config_path.parent

        # Check if multi-classifier configuration
        if detect_multi_classifier_config(str(config_file)):
            console.print("[yellow]⚡[/yellow] Detected multi-classifier configuration")

            # Load multi-classifier config
            multi_config = load_multi_classifier_config(str(config_file))

            # Determine which classifier to use
            if not classifier:
                classifier = get_default_classifier(str(config_file))
                if not classifier:
                    available = list_available_classifiers(str(config_file))
                    console.print(
                        "[red]✗[/red] No classifier specified and no default configured",
                        style="bold red",
                    )
                    console.print(f"Available classifiers: {', '.join(available)}")
                    console.print("Use [cyan]--classifier <name>[/cyan] to select one")
                    raise typer.Exit(1)
                console.print(
                    f"[yellow]→[/yellow] Using default classifier: [cyan]{classifier}[/cyan]"
                )
            else:
                console.print(f"[green]→[/green] Using classifier: [cyan]{classifier}[/cyan]")

            # Extract single classifier config
            try:
                cfg = extract_single_classifier_config(multi_config, classifier)
            except (ValueError, KeyError) as e:
                console.print(f"[red]✗[/red] Error: {e}", style="bold red")
                if log_level == LogLevel.DEBUG:
                    import traceback
                    console.print("[yellow]Full traceback:[/yellow]")
                    console.print(traceback.format_exc())
                raise typer.Exit(1) from e
            except Exception as e:
                console.print(f"[red]✗[/red] Unexpected error: {e}", style="bold red")
                if log_level == LogLevel.DEBUG:
                    import traceback
                    console.print("[yellow]Full traceback:[/yellow]")
                    console.print(traceback.format_exc())
                raise typer.Exit(1) from e

            # Resolve paths
            if cfg.predictor.init_kwargs:
                cfg.predictor.init_kwargs = resolve_relative_paths(
                    cfg.predictor.init_kwargs, str(config_dir)
                )
        else:
            # Single classifier configuration
            with open(config_file) as f:
                raw = yaml.safe_load(f)

            # Resolve relative paths
            if "predictor" in raw and "init_kwargs" in raw["predictor"]:
                raw["predictor"]["init_kwargs"] = resolve_relative_paths(
                    raw["predictor"]["init_kwargs"], str(config_dir)
                )

            cfg = AppConfig.model_validate(raw)

        # Set project path
        cfg.set_project_path(str(config_dir))

        # Apply CLI overrides
        if host:
            cfg.server.host = host
        if port:
            cfg.server.port = port
        if workers:
            cfg.server.workers = workers

        # Override log level only when explicitly provided on the CLI;
        # otherwise the config's server.log_level applies
        if log_level is not None:
            cfg.server.log_level = log_level.value

        # Configure logging with the new logger settings
        logger_config = cfg.server.logger if cfg.server.logger else None
        if logger_config:
            configure_logging(
                level=cfg.server.log_level,
                structured=logger_config.structured,
                include_timestamp=logger_config.timestamp,
                show_tasks=logger_config.show_tasks,
                custom_format=logger_config.format
            )
        else:
            # Fallback to default behavior
            configure_logging(cfg.server.log_level, cfg.observability.structured_logging)

        # Create the app with config file name
        config_file_name = config_file.name
        fastapi_app = create_app(cfg, config_file_name=config_file_name)

        # Show startup info
        console.print(Panel.fit(
            f"[bold cyan]ML Server Starting[/bold cyan]\n\n"
            f"[yellow]→[/yellow] Host: [cyan]{cfg.server.host}:{cfg.server.port}[/cyan]\n"
            f"[yellow]→[/yellow] Workers: [cyan]{cfg.server.workers}[/cyan]\n"
            f"[yellow]→[/yellow] Model: [cyan]{cfg.predictor.class_name}[/cyan]\n"
            f"[yellow]→[/yellow] API: [cyan]http://{cfg.server.host}:{cfg.server.port}[/cyan]\n"
            f"[yellow]→[/yellow] Docs: [cyan]http://{cfg.server.host}:{cfg.server.port}/docs[/cyan]",
            title="🚀 Server Info",
            border_style="cyan"
        ))

        # Run the server
        # Multi-worker and reload modes require an import string (uvicorn
        # cannot use an app instance for those), so use the factory function
        if cfg.server.workers > 1 or reload:
            # Set environment variables for the factory function
            os.environ['MLSERVER_CONFIG_PATH'] = str(config_file)
            if classifier:
                os.environ['MLSERVER_CLASSIFIER'] = classifier

            uvicorn.run(
                "mlserver.server:app",  # Factory function (reads env vars above)
                host=cfg.server.host,
                port=cfg.server.port,
                log_level=cfg.server.log_level.lower(),
                workers=1 if reload else cfg.server.workers,
                reload=reload,
                factory=True,  # Tell uvicorn this is a factory function
            )
        else:
            # Single worker - use the app instance directly
            uvicorn.run(
                fastapi_app,
                host=cfg.server.host,
                port=cfg.server.port,
                log_level=cfg.server.log_level.lower(),
                workers=1,
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Server stopped by user")
        raise typer.Exit(0) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def version(
    path: str = typer.Option(".", "--path", "-p", help="Path to classifier project"),
    classifier: Optional[str] = typer.Option(
        None, "--classifier", "-c", help="Classifier name (for multi-classifier configs)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed MLServer tool information"
    ),
):
    """📦 Display version information for the classifier project.

    Use --detailed to show MLServer tool version, commit, and installation source.
    For multi-classifier configs, use --classifier to specify which one.
    """
    version_info = get_version_info(path, classifier_name=classifier)

    # If no classifier project found, show only MLServer tool version
    if "error" in version_info:
        error_msg = version_info['error']
        if "mlserver.yaml not found" in error_msg:
            # No classifier project - show MLServer tool version only
            import mlserver as mlserver_module

            from .version_control import get_mlserver_commit_hash

            mlserver_commit = get_mlserver_commit_hash()
            mlserver_version = (
                mlserver_module.__version__ if hasattr(mlserver_module, '__version__')
                else "unknown"
            )
            mlserver_location = Path(mlserver_module.__file__).parent

            # Determine installation type
            install_type = "unknown"
            if (mlserver_location.parent / '.git').exists():
                install_type = "git (editable)"
            elif (mlserver_location.parent.name.endswith('.egg-info')
                  or mlserver_location.parent.name.endswith('.dist-info')):
                install_type = "package"
            elif 'site-packages' in str(mlserver_location):
                install_type = "pip"

            if json_output:
                mlserver_info = {
                    "mlserver_tool": {
                        "version": mlserver_version,
                        "commit": mlserver_commit,
                        "install_location": str(mlserver_location),
                        "install_type": install_type
                    }
                }
                print(json.dumps(mlserver_info, indent=2))
            else:
                console.print("[yellow]ℹ[/yellow] No classifier project found in current directory")
                console.print()

                table = Table(
                    title="📦 MLServer Tool Version", show_header=False, title_style="bold cyan"
                )
                table.add_column("Property", style="yellow")
                table.add_column("Value", style="cyan")

                table.add_row("Version", mlserver_version)
                table.add_row("Commit", mlserver_commit or "n/a")
                table.add_row("Install Type", install_type)
                table.add_row("Location", str(mlserver_location))

                console.print(table)
                console.print()
                console.print(
                    "[dim]To see classifier project version, "
                    "run this command from a directory with mlserver.yaml[/dim]"
                )

            return
        else:
            # Other error - show it
            console.print(f"[red]✗[/red] Error: {version_info['error']}", style="bold red")
            raise typer.Exit(1)

    # Handle multi-classifier summary output
    if version_info.get("multi_classifier"):
        if json_output:
            print(json.dumps(version_info, indent=2))
        else:
            table = Table(
                title="Multi-Classifier Project", show_header=True, title_style="bold cyan"
            )
            table.add_column("Classifier", style="yellow")
            table.add_column("Version", style="cyan")
            table.add_column("Description", style="dim")

            for clf in version_info["classifiers"]:
                table.add_row(clf["name"], clf["version"], clf.get("description", ""))

            console.print(table)

            git = version_info.get("git")
            if git:
                console.print(f"\n[dim]Git: {git['commit'][:7]} ({git['branch']})[/dim]")

            console.print(
                "\n[dim]Use --classifier <name> to see details for a specific classifier[/dim]"
            )
        return

    if json_output:
        # Add mlserver tool info to JSON output if detailed
        if detailed:
            import mlserver as mlserver_module

            from .version_control import get_mlserver_commit_hash
            mlserver_commit = get_mlserver_commit_hash()
            version_info["mlserver_tool"] = {
                "version": (
                    mlserver_module.__version__ if hasattr(mlserver_module, '__version__')
                    else "unknown"
                ),
                "commit": mlserver_commit,
                "install_location": str(Path(mlserver_module.__file__).parent)
            }
        print(json.dumps(version_info, indent=2))
    else:
        # Create a nice table for version info
        table = Table(title="Version Information", show_header=False, title_style="bold cyan")
        table.add_column("Property", style="yellow")
        table.add_column("Value", style="cyan")

        classifier_info = version_info["classifier"]
        model = version_info["model"]
        api = version_info["api"]
        git = version_info.get("git")
        issues = version_info.get("validation_issues", {})

        table.add_row("Classifier", f"{classifier_info['name']} v{classifier_info['version']}")
        table.add_row("Description", classifier_info.get('description', ''))
        table.add_row("Model Version", model['version'])
        table.add_row("API Version", api['version'])

        if git:
            table.add_row("Git Commit", f"{git['commit']} ({git['branch']})")
            if git['tag']:
                table.add_row("Git Tag", git['tag'])
            if git['is_dirty']:
                table.add_row("Status", "⚠️  Uncommitted changes")

        # Add MLServer tool information if --detailed
        if detailed:
            import mlserver as mlserver_module

            from .version_control import get_mlserver_commit_hash

            mlserver_commit = get_mlserver_commit_hash()
            mlserver_version = (
                mlserver_module.__version__ if hasattr(mlserver_module, '__version__')
                else "unknown"
            )
            mlserver_location = Path(mlserver_module.__file__).parent

            # Determine installation type
            install_type = "unknown"
            if (mlserver_location.parent / '.git').exists():
                install_type = "git (editable)"
            elif (mlserver_location.parent.name.endswith('.egg-info')
                  or mlserver_location.parent.name.endswith('.dist-info')):
                install_type = "package"
            elif 'site-packages' in str(mlserver_location):
                install_type = "pip"

            table.add_section()
            table.add_row("[bold]MLServer Tool[/bold]", "")
            table.add_row("  Version", mlserver_version)
            table.add_row("  Commit", mlserver_commit or "n/a")
            table.add_row("  Install Type", install_type)
            table.add_row("  Location", str(mlserver_location))

        console.print(table)

        # Container tags
        if version_info["container_tags"]:
            console.print("\n[bold cyan]Container Tags:[/bold cyan]")
            for tag in version_info["container_tags"]:
                console.print(f"  [yellow]→[/yellow] {tag}")

        # Validation issues
        if issues:
            console.print("\n[bold yellow]⚠️  Validation Issues:[/bold yellow]")
            for issue in issues.values():
                console.print(f"  [red]✗[/red] {issue}")


@app.command()
def build(
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier", "-c",
        help="Classifier to build (can be simple name or full tag: name-vX.Y.Z-mlserver-hash)"
    ),
    path: str = typer.Option(
        ".",
        "--path",
        help="Path to classifier project"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Config file to use (auto-detected if not specified)"
    ),
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        help="Container registry URL"
    ),
    tag_prefix: Optional[str] = typer.Option(
        None,
        "--tag-prefix",
        help="Tag prefix for container names"
    ),
    build_arg: Optional[list[str]] = typer.Option(
        None,
        "--build-arg",
        help="Build arguments (key=value)"
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Do not use cache when building"
    ),
    platform: Optional[str] = typer.Option(
        None,
        "--platform",
        help=(
            "Target platform for the image, e.g. linux/amd64 (single platform only). "
            "Cross-architecture builds require BuildKit with binfmt/QEMU emulation "
            "or docker buildx on the host."
        )
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip validation prompts and continue with build"
    ),
    per_classifier_image: bool = typer.Option(
        False,
        "--per-classifier-image",
        help=(
            "Escape hatch: build one baked image per classifier (pre-W2.5 behavior) "
            "instead of a single build-once commit image. Use ONLY for classifiers "
            "whose conflicting dependencies cannot share one image. Requires --classifier."
        )
    ),
):
    """🏗️  Build Docker container for the classifier project.

    Build-once / deploy-many (RFC 0001 D4): for a multi-classifier repo the
    DEFAULT builds ONE commit image (tagged <repo>:<git-sha> and <repo>:latest)
    that bundles every classifier; the classifier is chosen at deploy/run time
    via MLSERVER_CLASSIFIER. Use --per-classifier-image (with --classifier) to
    fall back to one baked image per classifier for diverging-dependency cases.
    Single-classifier repos always build a single image.

    The --classifier parameter accepts simple names and full version tags:
    - Simple: --classifier sentiment
    - Canonical tag: --classifier sentiment/v1.0.0
    - Legacy tag: --classifier sentiment-v1.0.0-mlserver-b5dff2a

    When using a full tag, the build will validate that your current code matches
    the tag's expected commits and warn if there are mismatches.
    """
    if not check_docker_availability():
        console.print("[red]✗[/red] Docker is not available or not running", style="bold red")
        raise typer.Exit(1)

    console.print("[cyan]🏗️  Building container...[/cyan]")

    # Parse classifier name (handles both simple names and full tags)
    original_input = classifier
    if classifier:
        from .version import get_git_info
        from .version_control import (
            extract_classifier_name,
            get_mlserver_commit_hash,
            get_tag_commits,
            parse_classifier_tag,
        )

        # Extract classifier name from full tag if provided
        classifier_name = extract_classifier_name(classifier)
        if not classifier_name:
            console.print(
                f"[red]✗[/red] Invalid classifier name format: {classifier}", style="bold red"
            )
            raise typer.Exit(1)

        # If full tag was provided (canonical or legacy), validate commits
        parsed = parse_classifier_tag(original_input)
        if parsed:
            console.print(f"[yellow]→[/yellow] Full tag provided: [cyan]{original_input}[/cyan]")
            console.print()

            # Get expected commits from tag (mlserver commit is only encoded
            # in legacy tags; canonical tags carry the classifier commit only)
            tag_commits = get_tag_commits(original_input, path)
            expected_classifier_commit = tag_commits['classifier_commit']
            expected_mlserver_commit = parsed['mlserver_commit']

            # Get current commits
            git_info = get_git_info(path)
            current_classifier_commit = git_info.commit if git_info else None
            current_mlserver_commit = get_mlserver_commit_hash()

            # Normalize commit hashes to 7 characters for comparison
            def normalize_commit(commit):
                return commit[:7] if commit else None

            expected_classifier_commit_short = normalize_commit(expected_classifier_commit)
            current_classifier_commit_short = normalize_commit(current_classifier_commit)
            expected_mlserver_commit_short = normalize_commit(expected_mlserver_commit)
            current_mlserver_commit_short = normalize_commit(current_mlserver_commit)

            # Check for mismatches
            classifier_mismatch = (
                expected_classifier_commit_short
                and current_classifier_commit_short
                and expected_classifier_commit_short != current_classifier_commit_short
            )
            mlserver_mismatch = (
                expected_mlserver_commit_short
                and current_mlserver_commit_short
                and expected_mlserver_commit_short != current_mlserver_commit_short
            )

            if classifier_mismatch or mlserver_mismatch:
                console.print(
                    "[yellow]⚠️  Warning: Current code doesn't match tag specifications[/yellow]"
                )
                console.print()
                console.print("[dim]Tag specifies:[/dim]")
                console.print(
                    f"  Classifier commit: {expected_classifier_commit_short or 'unknown'}"
                )
                console.print(
                    f"  MLServer commit:   "
                    f"{expected_mlserver_commit_short or 'n/a (not encoded in tag)'}"
                )
                console.print()
                console.print("[dim]Current working directory:[/dim]")
                classifier_marker = (
                    '[red]⚠️  MISMATCH[/red]' if classifier_mismatch else '[green]✓[/green]'
                )
                mlserver_marker = (
                    '[red]⚠️  MISMATCH[/red]' if mlserver_mismatch else '[green]✓[/green]'
                )
                console.print(
                    f"  Classifier commit: {current_classifier_commit_short or 'unknown'} "
                    f"{classifier_marker}"
                )
                console.print(
                    f"  MLServer commit:   {current_mlserver_commit_short or 'unknown'} "
                    f"{mlserver_marker}"
                )
                console.print()
                console.print(
                    "[yellow]Building with CURRENT code.[/yellow] To build exact tagged version:"
                )
                console.print(f"  [cyan]git checkout {original_input}[/cyan]")
                console.print()

                if not force:
                    if not typer.confirm("Continue with build?"):
                        console.print("[yellow]Build cancelled[/yellow]")
                        raise typer.Exit(0)
            else:
                console.print("[green]✓[/green] Current code matches tag specification")
                console.print()

        # Use extracted classifier name for the rest of the build
        classifier = classifier_name

    # Detect config file (inside the project directory)
    config_file = detect_config_file(config, base_dir=Path(path))

    # Check if multi-classifier config
    from .multi_classifier import detect_multi_classifier_config, list_available_classifiers

    if detect_multi_classifier_config(str(config_file)):
        # Multi-classifier config
        available = list_available_classifiers(str(config_file))

        if per_classifier_image:
            # Escape hatch (RFC 0001 D4): one baked image per classifier.
            if not classifier:
                console.print(
                    "[red]✗[/red] --per-classifier-image requires --classifier <name>.",
                    style="bold red",
                )
                console.print(f"Available classifiers: {', '.join(available)}")
                console.print("Usage: mlserver build --per-classifier-image --classifier <name>")
                raise typer.Exit(1)
            if classifier not in available:
                console.print(
                    f"[red]✗[/red] Classifier '{classifier}' not found.", style="bold red"
                )
                console.print(f"Available classifiers: {', '.join(available)}")
                raise typer.Exit(1)
            console.print(
                f"[yellow]→[/yellow] Building per-classifier image for: [cyan]{classifier}[/cyan]"
            )
        else:
            # Default (RFC 0001 D4 / W2.5): ONE commit image bundling every
            # classifier; selection happens at deploy/run time.
            console.print(
                "[yellow]→[/yellow] Building single commit image bundling all classifiers: "
                f"[cyan]{', '.join(available)}[/cyan]"
            )
            if classifier:
                console.print(
                    "[dim]  (--classifier is ignored for the commit image; use "
                    "--per-classifier-image to bake a single classifier)[/dim]"
                )
                # The commit image bundles all classifiers - drop any selection.
                classifier = None

    # Parse --build-arg values (must be KEY=value)
    parsed_build_args = None
    if build_arg:
        parsed_build_args = {}
        for arg in build_arg:
            if '=' not in arg:
                console.print(
                    f"[red]✗[/red] Invalid --build-arg '{arg}': expected format KEY=value",
                    style="bold red",
                )
                raise typer.Exit(1)
            key, value = arg.split('=', 1)
            parsed_build_args[key] = value

    result = build_container(
        project_path=str(path),
        config_file=str(config_file),
        classifier_name=classifier,
        tag_prefix=tag_prefix,
        registry=registry,
        build_args=parsed_build_args,
        no_cache=no_cache,
        mlserver_source_path=None,  # Auto-detect
        platform=platform,
        per_classifier_image=per_classifier_image
    )

    if result["success"]:
        console.print("[green]✓[/green] Successfully built container")
        console.print(f"[yellow]Tags:[/yellow] {', '.join(result['tags'])}")
        if verbose and result.get("build_output"):
            console.print("\n[dim]Build output:[/dim]")
            console.print(result["build_output"])
    else:
        console.print(f"[red]✗[/red] Build failed: {result['error']}", style="bold red")
        raise typer.Exit(1)


def _push_classifier_alias_cli(
    path: str,
    registry: str,
    classifier: Optional[str],
    tag_prefix: Optional[str],
    force: bool,
    config_file: Optional[Path],
) -> None:
    """Apply a classifier release as registry tag aliases on the commit image.

    Build-once/deploy-many (RFC 0001 D4 / W2.5): validates that HEAD sits on the
    canonical git tag ``<classifier>/vX.Y.Z``, then re-tags and pushes the
    already-built commit image under classifier-scoped registry tags (no
    rebuild). Raises ``typer.Exit(1)`` on any failure.
    """
    from .container import push_classifier_alias
    from .multi_classifier import list_available_classifiers

    available = list_available_classifiers(str(config_file)) if config_file else []

    if not classifier:
        console.print(
            "[red]✗[/red] Multi-classifier config detected. "
            "Specify which classifier release to push:",
            style="bold red",
        )
        console.print(f"Available classifiers: {', '.join(available)}")
        console.print("Usage: mlserver push --classifier <name> --registry <url>")
        raise typer.Exit(1)

    if available and classifier not in available:
        console.print(f"[red]✗[/red] Classifier '{classifier}' not found.", style="bold red")
        console.print(f"Available classifiers: {', '.join(available)}")
        raise typer.Exit(1)

    # Validate the canonical git tag <classifier>/vX.Y.Z exists at HEAD
    git_mgr = GitVersionManager(str(path))
    validation = git_mgr.validate_push_readiness(classifier, force)
    if not validation["ready"] and not force:
        console.print("[red]✗[/red] Push failed: release validation failed", style="bold red")
        for err in validation.get("errors", []):
            console.print(f"  [red]→[/red] {err}")
        raise typer.Exit(1)

    version = git_mgr.get_current_version(classifier)
    if not version:
        console.print(
            f"[red]✗[/red] No release tag found for classifier '{classifier}'.",
            style="bold red",
        )
        console.print(
            f"Create one first: [cyan]mlserver tag <major|minor|patch> "
            f"--classifier {classifier}[/cyan]"
        )
        raise typer.Exit(1)

    console.print(
        f"[cyan]📤 Applying release alias for {classifier} v{version} "
        f"on the commit image → {registry}...[/cyan]"
    )

    result = push_classifier_alias(
        project_path=str(path),
        registry=registry,
        classifier_name=classifier,
        version=version,
        tag_prefix=tag_prefix,
    )

    if result.get("success"):
        console.print("[green]✓[/green] Successfully pushed release aliases")
        console.print(f"  [yellow]→[/yellow] Source image: {result['source_image']}")
        for tag in result.get("pushed_tags", []):
            console.print(f"  [yellow]→[/yellow] {tag}")
    else:
        error_msg = result.get("error") or "Some aliases failed to push"
        console.print(f"[red]✗[/red] Push failed: {error_msg}", style="bold red")
        for err in result.get("failed_tags", []):
            console.print(f"  [red]✗[/red] {err}")
        raise typer.Exit(1)


@app.command()
def push(
    registry: str = typer.Option(
        ...,
        "--registry", "-r",
        help="Container registry URL"
    ),
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier", "-c",
        help="Classifier to push (required for multi-classifier configs)"
    ),
    path: str = typer.Option(
        ".",
        "--path",
        help="Path to classifier project"
    ),
    tag_prefix: Optional[str] = typer.Option(
        None,
        "--tag-prefix",
        help="Tag prefix for container names"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force push even if not on tagged commit or tag exists"
    ),
    version_source: Optional[str] = typer.Option(
        None,
        "--version-source",
        help=(
            "[DEPRECATED] Version source: 'git-tag', 'config', or 'auto'. "
            "Git tags are the canonical version source (RFC 0001 D3); "
            "this flag will be removed in v0.5.0."
        )
    ),
):
    """📤 Push container to registry (requires tagged commit for specific classifier)."""
    if not check_docker_availability():
        console.print("[red]✗[/red] Docker is not available or not running", style="bold red")
        raise typer.Exit(1)

    # --version-source is deprecated (RFC 0001 D3): warn when passed explicitly
    if version_source is not None:
        err_console.print(
            "[yellow]⚠ DeprecationWarning:[/yellow] --version-source is deprecated; "
            "git tags are the canonical version source — RFC 0001 D3. "
            "The flag will be removed in v0.5.0."
        )
    else:
        version_source = "auto"

    # Validate --version-source (see version_control.get_version_for_push)
    allowed_version_sources = ("git-tag", "config", "auto")
    if version_source not in allowed_version_sources:
        console.print(
            f"[red]✗[/red] Invalid --version-source '{version_source}'. "
            f"Allowed values: {', '.join(allowed_version_sources)}",
            style="bold red"
        )
        raise typer.Exit(1)

    # Choose the push strategy from the config shape. Multi-classifier repos
    # use build-once/deploy-many: the release is applied as registry tag aliases
    # on the already-built commit image (RFC 0001 D4 / W2.5), not rebuilt.
    try:
        push_config_file = detect_config_file(None, base_dir=Path(path))
        is_multi = detect_multi_classifier_config(str(push_config_file))
    except Exception:
        push_config_file = None
        is_multi = False

    if is_multi:
        _push_classifier_alias_cli(
            path=path, registry=registry, classifier=classifier,
            tag_prefix=tag_prefix, force=force, config_file=push_config_file
        )
        return

    console.print(f"[cyan]📤 Validating and pushing to {registry}...[/cyan]")
    if classifier:
        console.print(f"[yellow]→[/yellow] Classifier: {classifier}")

    # Single-classifier repos keep the existing per-image push path.
    result = safe_push_container(
        project_path=str(path),
        registry=registry,
        classifier_name=classifier,
        tag_prefix=tag_prefix,
        force=force,
        version_source=version_source
    )

    if result["success"]:
        console.print("[green]✓[/green] Successfully pushed images")
        console.print(
            f"  [yellow]→[/yellow] Version: {result['version_used']} "
            f"(from {result['version_source']})"
        )
        if result.get("pushed_tags"):
            for tag in result["pushed_tags"]:
                console.print(f"  [yellow]→[/yellow] {tag}")
    else:
        error_msg = result.get("error") or "Some images failed to push"
        console.print(f"[red]✗[/red] Push failed: {error_msg}", style="bold red")
        if result.get("validation_errors"):
            for error in result["validation_errors"]:
                console.print(f"  [red]→[/red] {error}")

    if result.get("validation_warnings"):
        console.print("\n[yellow]⚠️  Warnings:[/yellow]")
        for warning in result["validation_warnings"]:
            console.print(f"  [yellow]→[/yellow] {warning}")

    if result.get("failed_tags"):
        console.print(f"\n[yellow]⚠️  Failed to push {len(result['failed_tags'])} images:[/yellow]")
        for error in result["failed_tags"]:
            console.print(f"  [red]✗[/red] {error}")

    # Any failure - including partially failed pushes - must exit nonzero
    if not result["success"] or result.get("failed_tags"):
        raise typer.Exit(1)


@app.command()
def images(
    path: str = typer.Option(
        ".",
        "--path",
        help="Path to classifier project"
    ),
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier", "-c",
        help="Classifier name (for multi-classifier configs)"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON (machine-readable)"
    ),
):
    """📋 List Docker images for the classifier project.

    For multi-classifier configs, use --classifier to filter images.
    """
    image_list = list_images(str(path), classifier_name=classifier)

    if json_output:
        print(json.dumps({
            "images": image_list,
            "count": len(image_list),
            "classifier": classifier,
        }, indent=2))
        return

    if not image_list:
        console.print("[yellow]No images found for this classifier project[/yellow]")
        return

    table = Table(title="🐳 Docker Images", title_style="bold cyan")
    table.add_column("Tag", style="cyan")
    table.add_column("Image ID", style="yellow")
    table.add_column("Created", style="green")
    table.add_column("Size", style="magenta")

    for image in image_list:
        table.add_row(
            image['tag'],
            image['image_id'],
            image['created'],
            image['size']
        )

    console.print(table)


@app.command()
def tag(
    bump_type: Optional[BumpType] = typer.Argument(
        None,
        help="Version bump type: major, minor, or patch"
    ),
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier", "-c",
        help="Classifier name to tag (required for tagging)"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Config file path (defaults to mlserver.yaml)"
    ),
    message: Optional[str] = typer.Option(
        None,
        "--message", "-m",
        help="Tag message (defaults to 'Release <classifier> vX.Y.Z')"
    ),
    path: str = typer.Option(
        ".",
        "--path",
        help="Path to classifier project"
    ),
    allow_missing_mlserver: bool = typer.Option(
        False,
        "--allow-missing-mlserver",
        help="Allow tagging even if mlserver commit cannot be determined (dev/testing only)"
    ),
    status_only: bool = typer.Option(
        False,
        "--status",
        help="Show tag status for all classifiers (default when no bump type is given)"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output tag status as JSON (status mode only, machine-readable)"
    ),
):
    """🏷️  Manage version tags for classifiers.

    Without arguments (or with --status): Show tag status for all classifiers.
    With arguments: Create a version tag for a specific classifier.
    """
    if bump_type and (status_only or json_output):
        raise typer.BadParameter(
            "--status/--json show tag status and cannot be combined with a bump type"
        )

    try:
        git_mgr = GitVersionManager(str(path))

        # If no bump_type, show status table
        if not bump_type:
            # Detect config file (inside the project directory)
            config_file = detect_config_file(config, base_dir=Path(path))

            # Get status for all classifiers
            classifiers_status = git_mgr.get_all_classifiers_tag_status(str(config_file))

            # Get current mlserver commit for comparison
            from .version_control import get_mlserver_commit_hash, parse_classifier_tag
            current_mlserver_commit = get_mlserver_commit_hash() or "unknown"

            if json_output:
                doc = {"classifiers": {}, "mlserver_commit": current_mlserver_commit}
                for clf_name, status in classifiers_status.items():
                    parsed = (
                        parse_classifier_tag(status["latest_tag"])
                        if status.get("latest_tag") else None
                    )
                    doc["classifiers"][clf_name] = {
                        "current_version": status["current_version"],
                        "latest_tag": status["latest_tag"],
                        "tag_format": parsed["format"] if parsed else None,
                        "tag_mlserver_commit": (
                            parsed["mlserver_commit"] if parsed else None
                        ),
                        "commits_since_tag": status["commits_since_tag"],
                        "on_tagged_commit": status["on_tagged_commit"],
                        "status": status["status"],
                        "recommendation": status["recommendation"],
                    }
                print(json.dumps(doc, indent=2))
                return

            # Create table with MLServer commit column
            table = Table(title="🏷️  Classifier Version Status", title_style="bold cyan")
            table.add_column("Classifier", style="cyan")
            table.add_column("Version", style="yellow")
            table.add_column("MLServer", style="blue")
            table.add_column("Status", style="green")
            table.add_column("Action Required", style="magenta")

            for clf_name, status in classifiers_status.items():
                version = status['current_version'] or "No tags"
                status_text = status['status']
                recommendation = status['recommendation'] or "-"

                # Extract mlserver commit from the latest tag (only legacy
                # tags encode it; canonical tags show n/a)
                mlserver_commit = "-"
                if status.get('latest_tag'):
                    parsed = parse_classifier_tag(status['latest_tag'])
                    if parsed and parsed['mlserver_commit']:
                        tag_mlserver = parsed['mlserver_commit'][:7]
                        current_mlserver_short = current_mlserver_commit[:7]
                        if tag_mlserver == current_mlserver_short:
                            mlserver_commit = f"{tag_mlserver} [green]✓[/green]"
                        else:
                            mlserver_commit = f"{tag_mlserver} [yellow]⚠️[/yellow]"
                    else:
                        mlserver_commit = "[dim]n/a[/dim]"

                # Color coding for status
                if status['on_tagged_commit']:
                    status_style = "green"
                elif status['commits_since_tag'] and status['commits_since_tag'] > 0:
                    status_style = "yellow"
                else:
                    status_style = "red"

                table.add_row(
                    clf_name,
                    version,
                    mlserver_commit,
                    f"[{status_style}]{status_text}[/{status_style}]",
                    recommendation
                )

            console.print(table)
            console.print(f"\n[dim]Current MLServer commit: {current_mlserver_commit[:7]}[/dim]")
            return

        # Tagging mode - classifier is required
        if not classifier:
            console.print("[red]✗[/red] Classifier name is required for tagging", style="bold red")
            console.print("Use [cyan]--classifier <name>[/cyan] to specify the classifier")
            raise typer.Exit(1)

        # Run validation suite for tagging
        from .validation import get_tag_validation_suite

        validation_suite = get_tag_validation_suite()
        all_passed, results = validation_suite.validate(
            project_path=path,
            classifier_name=classifier
        )

        if not all_passed:
            console.print("[red]✗[/red] Cannot create tag: Validation failed", style="bold red")
            console.print()

            # Display all validation failures
            for result in results:
                if not result.passed:
                    console.print(f"[red]✗[/red] {result.error_message}")

                    # Show details if available
                    if result.details:
                        if "missing_files" in result.details:
                            console.print()
                            console.print("[yellow]Missing files:[/yellow]")
                            for missing_file in result.details["missing_files"]:
                                console.print(f"  [red]✗[/red] {missing_file}")

                        if "solution" in result.details:
                            console.print()
                            console.print("[cyan]Solution:[/cyan]")
                            console.print(f"  {result.details['solution']}")

                    console.print()

            raise typer.Exit(1)

        # Show any warnings (non-blocking)
        for result in results:
            if result.warnings:
                for warning in result.warnings:
                    console.print(f"[yellow]⚠[/yellow] {warning}")
                    if result.details and "solution" in result.details:
                        console.print(f"  [dim]{result.details['solution']}[/dim]")

        # Create the tag
        tag_info = git_mgr.tag_version(bump_type.value, classifier, message, allow_missing_mlserver)

        # Display success message with all details
        console.print(f"[green]✓[/green] Created tag: [cyan]{tag_info['tag_name']}[/cyan]")
        console.print()

        # Version info
        if tag_info['previous_version']:
            console.print(
                f"  [yellow]📝 Version:[/yellow] {tag_info['previous_version']} → "
                f"{tag_info['version']} ({bump_type.value} bump)"
            )
        else:
            console.print(f"  [yellow]📝 Version:[/yellow] {tag_info['version']} (initial release)")

        # MLServer info
        console.print(f"  [yellow]🔧 MLServer commit:[/yellow] {tag_info['mlserver_commit']}")

        # Get classifier commit for reference
        from .version import get_git_info
        git_info = get_git_info(path)
        if git_info:
            console.print(f"  [yellow]📦 Classifier commit:[/yellow] {git_info.commit}")

        # Check if GitHub Actions is set up and validate workflow
        from .github_actions import check_github_actions_setup, validate_workflow_comprehensive
        github_actions_configured = check_github_actions_setup(path)

        # Validate workflow if it exists
        workflow_valid = True
        workflow_warnings = []
        if github_actions_configured:
            workflow_valid, workflow_warnings, workflow_details = (
                validate_workflow_comprehensive(path)
            )

            if not workflow_valid or workflow_warnings:
                console.print()
                console.print("[yellow]⚠️  GitHub Actions Workflow Issues Detected:[/yellow]")
                for warning in workflow_warnings:
                    console.print(f"  [yellow]•[/yellow] {warning}")
                console.print()
                console.print(
                    "[bold red]⚠️  IMPORTANT: Regenerate workflow before pushing tags![/bold red]"
                )
                console.print("  Run: [cyan]mlserver init-github --force[/cyan]")
                console.print(
                    "  Then: [cyan]git add .github && "
                    "git commit -m 'Update workflow' && git push[/cyan]"
                )
                console.print()

        console.print("\n[cyan]Next steps:[/cyan]")
        if github_actions_configured and workflow_valid:
            console.print("  1. Push tags to remote: [cyan]git push --tags[/cyan]")
            console.print(
                "  2. GitHub Actions will automatically build and publish your container!"
            )
            console.print()
            console.print("[dim]💡 Or build manually:[/dim]")
            console.print(f"  - Build: [cyan]mlserver build --classifier {classifier}[/cyan]")
            console.print(
                f"  - Push: [cyan]mlserver push --classifier {classifier} --registry <url>[/cyan]"
            )
        elif github_actions_configured and not workflow_valid:
            # Workflow exists but is outdated/invalid
            console.print(
                "  [bold]1. Regenerate workflow:[/bold] [cyan]mlserver init-github --force[/cyan]"
            )
            console.print(
                "  2. Commit and push: [cyan]git add .github && "
                "git commit -m 'Update workflow' && git push[/cyan]"
            )
            console.print("  3. Push tags: [cyan]git push --tags[/cyan]")
            console.print()
            console.print("[dim]💡 Or build manually:[/dim]")
            console.print(f"  - Build: [cyan]mlserver build --classifier {classifier}[/cyan]")
            console.print(
                f"  - Push: [cyan]mlserver push --classifier {classifier} --registry <url>[/cyan]"
            )
        else:
            # No workflow at all
            console.print("  [yellow]⚠[/yellow] [bold]GitHub Actions not configured![/bold]")
            console.print()
            console.print(
                "  [dim]You need to set up CI/CD before pushing tags. Choose one option:[/dim]"
            )
            console.print()
            console.print("  [dim]Option 1: Add CI/CD workflow (recommended)[/dim]")
            console.print("  1. Add workflow: [cyan]mlserver init-github[/cyan]")
            console.print(
                "  2. Commit and push: [cyan]git add .github && "
                "git commit -m 'Add CI/CD' && git push[/cyan]"
            )
            console.print("  3. Push tags: [cyan]git push --tags[/cyan]")
            console.print()
            console.print("  [dim]Option 2: Build and push manually[/dim]")
            console.print("  1. Push tags: [cyan]git push --tags[/cyan]")
            console.print(f"  2. Build: [cyan]mlserver build --classifier {classifier}[/cyan]")
            console.print(
                f"  3. Push: [cyan]mlserver push --classifier {classifier} --registry <url>[/cyan]"
            )

    except VersionControlError as e:
        console.print(f"[red]✗[/red] {e}", style="bold red")
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def clean(
    path: str = typer.Option(
        ".",
        "--path",
        help="Path to classifier project"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force removal without confirmation"
    ),
):
    """🧹 Remove Docker images for the classifier project."""
    if not check_docker_availability():
        console.print("[red]✗[/red] Docker is not available or not running", style="bold red")
        raise typer.Exit(1)

    # Get images to be removed
    image_list = list_images(str(path))
    if not image_list:
        console.print("[yellow]No images to remove[/yellow]")
        return

    # Confirm if not forced
    if not force:
        console.print(f"[yellow]⚠️  This will remove {len(image_list)} images[/yellow]")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    console.print("[cyan]🧹 Cleaning images...[/cyan]")

    result = remove_images(str(path), force=True)

    if result["success"]:
        if result.get("removed_images"):
            console.print(f"[green]✓[/green] Removed {len(result['removed_images'])} images:")
            for image in result["removed_images"]:
                console.print(f"  [yellow]→[/yellow] {image}")
        else:
            console.print(result.get("message", "No images removed"))
    else:
        console.print(f"[red]✗[/red] Clean failed: {result['error']}", style="bold red")
        raise typer.Exit(1)

    if result.get("errors"):
        console.print("\n[yellow]⚠️  Removal errors:[/yellow]")
        for error in result["errors"]:
            console.print(f"  [red]✗[/red] {error}")


@app.command()
def run(
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier", "-c",
        help="Classifier to run (required for multi-classifier configs)"
    ),
    path: str = typer.Option(
        ".",
        "--path",
        help="Path to classifier project"
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port to expose the container on"
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        help="Specific version to run (default: latest)"
    ),
    detach: bool = typer.Option(
        False,
        "--detach", "-d",
        help="Run container in background"
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Container name"
    ),
    env: Optional[list[str]] = typer.Option(
        None,
        "--env", "-e",
        help="Environment variables (KEY=value)"
    ),
    volume: Optional[list[str]] = typer.Option(
        None,
        "--volume", "-v",
        help="Volume mounts (host:container)"
    ),
):
    """🚀 Run Docker container for the classifier."""
    if not check_docker_availability():
        console.print("[red]✗[/red] Docker is not available or not running", style="bold red")
        raise typer.Exit(1)

    # Detect config file (inside the project directory)
    config_file = detect_config_file(None, base_dir=Path(path))

    # Check if multi-classifier config
    from .multi_classifier import detect_multi_classifier_config, list_available_classifiers

    if detect_multi_classifier_config(str(config_file)):
        # Multi-classifier config
        available = list_available_classifiers(str(config_file))

        if not classifier:
            console.print(
                "[red]✗[/red] Multi-classifier config detected. "
                "Please specify which classifier to run:",
                style="bold red",
            )
            console.print(f"Available classifiers: {', '.join(available)}")
            console.print("Usage: mlserver run --classifier <name>")
            raise typer.Exit(1)

        if classifier not in available:
            console.print(f"[red]✗[/red] Classifier '{classifier}' not found.", style="bold red")
            console.print(f"Available classifiers: {', '.join(available)}")
            raise typer.Exit(1)

    # Get repository name
    from .version import get_repository_name
    repository = get_repository_name(path)

    # Build-once / deploy-many (RFC 0001 D4 / W2.5): the commit image bundles
    # every classifier, so run the repository image and select the classifier at
    # run time via MLSERVER_CLASSIFIER (added to the docker run command below).
    image_name = repository

    # Add version tag
    if version:
        full_image = f"{image_name}:{version}"
    else:
        full_image = f"{image_name}:latest"

    console.print(f"[cyan]🚀 Running container {full_image}...[/cyan]")

    # Build docker run command
    docker_cmd = ["docker", "run"]

    if detach:
        docker_cmd.append("-d")
    else:
        docker_cmd.append("-it")

    # Add port mapping (host port -> the container's configured server port)
    container_port = 8000
    try:
        with open(config_file) as f:
            raw_run_config = yaml.safe_load(f) or {}
        container_port = int(raw_run_config.get("server", {}).get("port", container_port))
    except Exception:
        pass
    docker_cmd.extend(["-p", f"{port}:{container_port}"])

    # Add container name if specified
    if name:
        docker_cmd.extend(["--name", name])
    else:
        # Auto-generate name if running specific classifier
        if classifier:
            import time
            timestamp = int(time.time())
            docker_cmd.extend(["--name", f"{classifier}-{timestamp}"])

    # Select the classifier at run time for the build-once commit image
    # (RFC 0001 D4 / W2.5). User -e overrides can still follow.
    if classifier:
        docker_cmd.extend(["-e", f"MLSERVER_CLASSIFIER={classifier}"])

    # Add environment variables
    if env:
        for env_var in env:
            docker_cmd.extend(["-e", env_var])

    # Add volume mounts
    if volume:
        for vol in volume:
            docker_cmd.extend(["-v", vol])

    # Add image name
    docker_cmd.append(full_image)

    # Execute docker run
    import subprocess

    if detach:
        # Run detached with output capture
        result = subprocess.run(docker_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            container_id = result.stdout.strip()
            console.print("[green]✓[/green] Container started in background")
            console.print(f"[yellow]→[/yellow] Container ID: {container_id[:12]}")
            console.print(f"[yellow]→[/yellow] Access at: http://localhost:{port}")
            console.print(f"[yellow]→[/yellow] Stop with: docker stop {container_id[:12]}")
        else:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            console.print(f"[red]✗[/red] Failed to run container: {error_msg}", style="bold red")
            raise typer.Exit(1)
    else:
        # Run interactively without capture (allows TTY)
        try:
            # Check if we have a TTY available
            import sys
            if sys.stdin.isatty():
                # We have TTY, run interactively
                result = subprocess.run(docker_cmd)
            else:
                # No TTY, remove -it flag and run without TTY
                docker_cmd[docker_cmd.index("-it")] = "--rm"
                result = subprocess.run(docker_cmd)

            if result.returncode == 0:
                console.print("[green]✓[/green] Container stopped")
            else:
                console.print(
                    f"[red]✗[/red] Container exited with error code: {result.returncode}",
                    style="bold red",
                )
                raise typer.Exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]→[/yellow] Container interrupted")
            raise typer.Exit(0) from None
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to run container: {e}", style="bold red")
            raise typer.Exit(1) from e



@app.command()
def list_classifiers(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to multi-classifier config file"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON (machine-readable)"
    ),
):
    """📋 List available classifiers in multi-classifier config."""
    try:
        config_file = detect_config_file(config)

        if not detect_multi_classifier_config(str(config_file)):
            if json_output:
                print(json.dumps({
                    "config_file": str(config_file),
                    "multi_classifier": False,
                    "classifiers": [],
                    "default_classifier": None,
                }, indent=2))
            else:
                console.print("[yellow]Not a multi-classifier configuration[/yellow]")
            return

        classifiers = list_available_classifiers(str(config_file))
        default = get_default_classifier(str(config_file))

        if json_output:
            print(json.dumps({
                "config_file": str(config_file),
                "multi_classifier": True,
                "classifiers": classifiers,
                "default_classifier": default,
            }, indent=2))
            return

        table = Table(title="📦 Available Classifiers", title_style="bold cyan")
        table.add_column("Classifier", style="cyan")
        table.add_column("Default", style="yellow")

        for classifier in classifiers:
            is_default = "✓" if classifier == default else ""
            table.add_row(classifier, is_default)

        console.print(table)

        if default:
            console.print(f"\n[yellow]→[/yellow] Default classifier: [cyan]{default}[/cyan]")

    except typer.Exit:
        raise
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def status(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON (machine-readable)"
    ),
):
    """📊 Show ML Server status and system info."""
    from .github_actions import check_github_actions_setup

    # Gather status information
    docker_available = check_docker_availability()
    mlserver_yaml = Path("mlserver.yaml")
    config_file = "mlserver.yaml" if mlserver_yaml.exists() else None
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    venv = os.environ.get("VIRTUAL_ENV")
    github_actions_setup = check_github_actions_setup(".")

    if json_output:
        print(json.dumps({
            "docker_available": docker_available,
            "config_file": config_file,
            "python_version": python_version,
            "virtual_env": Path(venv).name if venv else None,
            "github_actions_configured": github_actions_setup,
        }, indent=2))
        return

    table = Table(title="📊 ML Server Status", title_style="bold cyan")
    table.add_column("Component", style="yellow")
    table.add_column("Status", style="cyan")

    # Check Docker
    docker_status = (
        "[green]✓ Available[/green]" if docker_available else "[red]✗ Not available[/red]"
    )
    table.add_row("Docker", docker_status)

    # Check for config files
    if config_file:
        config_status = f"[green]{config_file}[/green]"
    else:
        config_status = "[yellow]No config found[/yellow]"
    table.add_row("Config Files", config_status)

    # Python version
    table.add_row("Python Version", python_version)

    # Check for virtual env
    venv_status = f"[green]{Path(venv).name}[/green]" if venv else "[yellow]None[/yellow]"
    table.add_row("Virtual Env", venv_status)

    # Check for GitHub Actions setup
    github_status = (
        "[green]✓ Configured[/green]" if github_actions_setup
        else "[yellow]Not configured[/yellow]"
    )
    table.add_row("GitHub Actions", github_status)

    console.print(table)


@app.command()
def init(
    path: str = typer.Option(
        ".",
        "--path", "-p",
        help="Path to initialize project in"
    ),
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier", "-c",
        help="Classifier name (defaults to directory name)"
    ),
    predictor_file: Optional[str] = typer.Option(
        None,
        "--predictor-file",
        help="Name of predictor Python file (without .py extension)"
    ),
    predictor_class: Optional[str] = typer.Option(
        None,
        "--predictor-class",
        help="Name of predictor class"
    ),
    no_github: bool = typer.Option(
        False,
        "--no-github",
        help="Skip GitHub Actions workflow creation"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing files"
    ),
):
    """🎬 Initialize a new MLServer classifier project.

    Creates all necessary files for a classifier project:
    - mlserver.yaml (configuration file)
    - <predictor>.py (skeleton predictor class)
    - .github/workflows/ml-classifier-container-build.yml (CI/CD workflow)
    - .gitignore (Python/ML project gitignore)

    This command will NOT overwrite existing files unless --force is used.

    Example:
        mlserver init --classifier sentiment-analyzer
        mlserver init --classifier my-model --predictor-file custom_predictor
    """
    from .init_project import init_mlserver_project

    console.print("[cyan]🎬 Initializing MLServer project...[/cyan]")
    console.print()

    success, message, files = init_mlserver_project(
        project_path=path,
        classifier_name=classifier,
        predictor_file=predictor_file,
        predictor_class=predictor_class,
        include_github_actions=not no_github,
        force=force
    )

    if success:
        if files:
            console.print("[green]✓[/green] Created files:")
            for file_path in files.values():
                console.print(f"  [yellow]→[/yellow] {file_path}")
            console.print()

        # Show the message (may include skipped files)
        for line in message.split('\n'):
            if line.strip():
                if 'Skipped' in line or 'already exist' in line:
                    console.print(f"[yellow]{line}[/yellow]")
                else:
                    console.print(line)
        console.print()

        # Show next steps
        console.print("[bold cyan]Next steps:[/bold cyan]")
        console.print("  1. Implement your predictor: Edit the generated Python file")
        console.print("  2. Configure settings: Review and update [cyan]mlserver.yaml[/cyan]")
        console.print("  3. Test locally: [cyan]mlserver serve[/cyan]")
        console.print(
            "  4. Commit changes: [cyan]git add . && git commit -m 'Initial setup'[/cyan]"
        )
        console.print(
            "  5. Create version tag: [cyan]mlserver tag patch --classifier <name>[/cyan]"
        )
        console.print("  6. Push to trigger CI/CD: [cyan]git push --tags[/cyan]")
    else:
        console.print(f"[red]✗[/red] {message}", style="bold red")
        raise typer.Exit(1)


@app.command(name="init-github")
def init_github(
    path: str = typer.Option(
        ".",
        "--path", "-p",
        help="Path to classifier project"
    ),
    python_version: str = typer.Option(
        "3.11",
        "--python-version",
        help="Python version for CI/CD workflow"
    ),
    registry: str = typer.Option(
        "ghcr.io",
        "--registry",
        help="Container registry (default: ghcr.io for GitHub Container Registry)"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing workflow files"
    ),
):
    """🔧 Initialize GitHub Actions CI/CD workflow for automated container builds.

    Sets up automated building and publishing of containers to GitHub Container Registry
    when version tags are pushed. The workflow is triggered by hierarchical tags created
    with the 'mlserver tag' command.

    This command:
    - Creates .github/workflows/ml-classifier-container-build.yml
    - Configures automated Docker builds on tag push
    - Sets up container publishing to GitHub Container Registry
    - Auto-detects your GitHub repository information

    Note: This is automatically created by 'mlserver init', so you typically don't need
    to run this separately unless you want to add CI/CD to an existing project.

    After running this command, commit the files and use 'mlserver tag' to create version tags.
    """
    from .github_actions import init_github_actions

    console.print("[cyan]🔧 Initializing GitHub Actions CI/CD...[/cyan]")
    console.print()

    success, message, files = init_github_actions(
        project_path=path,
        python_version=python_version,
        registry=registry,
        force=force
    )

    if success:
        console.print("[green]✓[/green] " + message.split('\n')[0])
        console.print()

        # Show created files
        if files:
            console.print("[bold]Created files:[/bold]")
            for file_path in files.values():
                console.print(f"  [yellow]→[/yellow] {file_path}")
            console.print()

        # Show next steps
        console.print("[bold cyan]Next steps:[/bold cyan]")
        console.print(
            "  1. Review workflow: "
            "[cyan].github/workflows/ml-classifier-container-build.yml[/cyan]"
        )
        console.print(
            "  2. Commit changes: "
            "[cyan]git add .github && git commit -m 'Add CI/CD workflow'[/cyan]"
        )
        console.print("  3. Push to GitHub: [cyan]git push[/cyan]")
        console.print(
            "  4. Create version tag: [cyan]mlserver tag patch --classifier <name>[/cyan]"
        )
        console.print("  5. Push tag: [cyan]git push --tags[/cyan]")
        console.print()
        console.print(
            "[dim]The workflow will automatically build and publish your container to GHCR![/dim]"
        )
    else:
        console.print(f"[red]✗[/red] {message}", style="bold red")
        raise typer.Exit(1)


@app.command()
def validate(
    config: Optional[Path] = typer.Argument(
        None, help="Path to config file (default: auto-detect)"
    ),
    strict: bool = typer.Option(False, "--strict", "-s", help="Fail on warnings"),
    check_imports: bool = typer.Option(
        True, "--check-imports/--no-check-imports", help="Check predictor imports"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON (machine-readable)"
    ),
):
    """Validate configuration without starting the server.

    Checks:
    - YAML syntax validity
    - Required fields present
    - Predictor module importable
    - Model files exist
    - Feature order file exists (if configured)

    Examples:
        mlserver validate
        mlserver validate --strict
        mlserver validate mlserver.yaml --no-check-imports
        mlserver validate --json
    """
    from .doctor import CheckStatus, run_validation_checks

    try:
        config_file = detect_config_file(config)
        project_path = str(config_file.parent)
    except typer.BadParameter as e:
        if json_output:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from e

    if not json_output:
        console.print(f"\n[bold]Validating[/bold] {config_file.name}...\n")

    report = run_validation_checks(
        project_path, check_imports=check_imports, config_file=config_file
    )

    if json_output:
        has_errors = report.has_errors
        has_warnings = report.has_warnings
        valid = not has_errors and not (strict and has_warnings)
        print(json.dumps({
            "config_file": str(config_file),
            "valid": valid,
            "strict": strict,
            "errors": sum(1 for c in report.checks if c.status == CheckStatus.FAILED),
            "warnings": sum(1 for c in report.checks if c.status == CheckStatus.WARNING),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "suggestion": c.suggestion,
                }
                for c in report.checks
            ],
        }, indent=2))
        if not valid:
            raise typer.Exit(1)
        return

    has_errors = False
    has_warnings = False

    for check in report.checks:
        if check.status == CheckStatus.PASSED:
            console.print(f"  [green]✓[/green] {check.name}")
            if verbose and check.message:
                console.print(f"    [dim]{check.message}[/dim]")
        elif check.status == CheckStatus.WARNING:
            has_warnings = True
            console.print(f"  [yellow]⚠[/yellow] {check.name}: {check.message}")
            if check.suggestion:
                console.print(f"    [dim]→ {check.suggestion}[/dim]")
        elif check.status == CheckStatus.FAILED:
            has_errors = True
            console.print(f"  [red]✗[/red] {check.name}: {check.message}")
            if check.suggestion:
                console.print(f"    [dim]→ {check.suggestion}[/dim]")
        elif check.status == CheckStatus.SKIPPED:
            if verbose:
                console.print(f"  [dim]○[/dim] {check.name}: {check.message}")

    console.print()

    if has_errors:
        console.print("[red]Configuration has errors. Fix them before serving.[/red]")
        raise typer.Exit(1)
    elif has_warnings and strict:
        console.print("[yellow]Configuration has warnings (--strict mode).[/yellow]")
        raise typer.Exit(1)
    elif has_warnings:
        console.print("[green]Configuration valid[/green] [dim](with warnings)[/dim]")
    else:
        console.print("[green]✓ Configuration valid! Ready to serve.[/green]")


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostics"),
    project_path: Path = typer.Option(".", "--path", "-p", help="Project path to diagnose"),
):
    """Diagnose common issues and environment problems.

    Checks system requirements, project configuration, dependencies,
    and provides recommendations for fixing issues.

    Examples:
        mlserver doctor
        mlserver doctor --verbose
        mlserver doctor --path ./my-project
    """
    from .doctor import run_all_checks

    console.print("\n[bold]MLServer Doctor[/bold] - Diagnosing your environment...\n")

    report = run_all_checks(str(project_path), verbose=verbose)

    # Group checks by category
    system_checks = []
    project_checks = []

    for check in report.checks:
        if check.name in ("Python version", "Docker", "Git"):
            system_checks.append(check)
        else:
            project_checks.append(check)

    # Display system checks
    console.print("[bold]System Checks:[/bold]")
    for check in system_checks:
        _display_check_result(check, verbose)

    # Display project checks
    console.print("\n[bold]Project Checks:[/bold]")
    for check in project_checks:
        _display_check_result(check, verbose)

    # Display recommendations
    if report.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(report.recommendations[:5], 1):
            console.print(f"  {i}. {rec}")

    console.print()

    if report.has_errors:
        console.print("[red]Some checks failed. Review the issues above.[/red]")
        raise typer.Exit(1)
    elif report.has_warnings:
        console.print("[yellow]All critical checks passed, but there are warnings.[/yellow]")
    else:
        console.print("[green]✓ All checks passed! Environment looks good.[/green]")


def _display_check_result(check, verbose: bool = False):
    """Helper to display a check result with consistent formatting."""
    from .doctor import CheckStatus

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


@app.command()
def test(
    data: Optional[str] = typer.Option(None, "--data", "-d", help="JSON data for prediction"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="JSON file with request data"),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL"),
    endpoint: str = typer.Option("/predict", "--endpoint", "-e", help="Prediction endpoint"),
    pretty: bool = typer.Option(True, "--pretty/--raw", help="Pretty-print response"),
):
    """Test prediction against a running server.

    Send a test request to verify the server is working correctly.

    Examples:
        mlserver test --data '{"feature1": 1.5, "feature2": 2.0}'
        mlserver test --file sample_request.json
        mlserver test --url http://localhost:8080 --endpoint /predict
    """
    import json
    import time

    import httpx

    # Prepare request data
    if data:
        try:
            payload_data = json.loads(data)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗[/red] Invalid JSON data: {e}")
            raise typer.Exit(1) from e
    elif file:
        if not file.exists():
            console.print(f"[red]✗[/red] File not found: {file}")
            raise typer.Exit(1)
        try:
            with open(file) as f:
                payload_data = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗[/red] Invalid JSON in file: {e}")
            raise typer.Exit(1) from e
    else:
        console.print("[red]✗[/red] Either --data or --file is required")
        console.print("\n[dim]Examples:")
        console.print("  mlserver test --data '{\"feature1\": 1.5}'")
        console.print("  mlserver test --file request.json[/dim]")
        raise typer.Exit(1)

    # Wrap in payload format if needed
    if "payload" not in payload_data and "instances" not in payload_data:
        # Auto-wrap as records
        if isinstance(payload_data, dict) and not any(
            k in payload_data for k in ["records", "ndarray"]
        ):
            payload_data = {"payload": {"records": [payload_data]}}
        elif isinstance(payload_data, list):
            payload_data = {"payload": {"records": payload_data}}

    # Build URL
    full_url = f"{url.rstrip('/')}{endpoint}"

    console.print("\n[bold]Testing prediction...[/bold]")
    console.print(f"  Server: [cyan]{url}[/cyan]")
    console.print(f"  Endpoint: [cyan]{endpoint}[/cyan]")
    console.print()

    # Send request
    try:
        start_time = time.perf_counter()
        with httpx.Client(timeout=30.0) as client:
            response = client.post(full_url, json=payload_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        status_color = "green" if response.status_code == 200 else "red"
        console.print(
            f"[bold]Response[/bold] "
            f"([{status_color}]{response.status_code} {response.reason_phrase}[/{status_color}], "
            f"{elapsed_ms:.0f}ms):"
        )

        try:
            response_data = response.json()
            if pretty:
                console.print_json(json.dumps(response_data, indent=2, default=str))
            else:
                console.print(json.dumps(response_data, default=str))
        except json.JSONDecodeError:
            console.print(response.text)

        if response.status_code != 200:
            raise typer.Exit(1)

    except httpx.ConnectError:
        console.print(f"[red]✗[/red] Cannot connect to {url}")
        console.print("  [dim]→ Is the server running? Try: mlserver serve[/dim]")
        raise typer.Exit(1) from None
    except httpx.TimeoutException:
        console.print("[red]✗[/red] Request timed out")
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] Request failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def schema(
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for schema file (default: stdout)"
    ),
    config_type: str = typer.Option(
        "auto", "--type", "-t",
        help="Config type: 'single', 'multi', or 'auto' (supports both)"
    ),
    setup: bool = typer.Option(
        False, "--setup", "-s",
        help="Show IDE setup instructions after generating schema"
    ),
    vscode: bool = typer.Option(
        False, "--vscode",
        help="Generate .vscode/settings.json for automatic schema association"
    ),
):
    """Generate JSON schema for mlserver.yaml configuration.

    Creates a JSON schema that enables IDE autocompletion and validation
    for your mlserver.yaml configuration files.

    Examples:
        # Print schema to stdout
        mlserver schema

        # Save to default location with setup instructions
        mlserver schema -o .mlserver/schema.json --setup

        # Generate for multi-classifier configs only
        mlserver schema --type multi -o schema.json

        # Full VSCode setup
        mlserver schema -o .mlserver/schema.json --vscode --setup
    """
    import json

    from .schema_generator import (
        get_schema_for_config_type,
        get_vscode_settings_snippet,
        print_schema_setup_instructions,
        save_schema,
    )

    # Validate config_type
    if config_type not in ("single", "multi", "auto"):
        console.print(f"[red]✗[/red] Invalid config type: {config_type}")
        console.print("  [dim]→ Use 'single', 'multi', or 'auto'[/dim]")
        raise typer.Exit(1)

    try:
        # Generate schema
        schema = get_schema_for_config_type(config_type)

        if output:
            # Save to file
            save_schema(schema, str(output))
            console.print(f"[green]✓[/green] Schema saved to: {output}")

            # Optionally generate VSCode settings
            if vscode:
                vscode_dir = Path.cwd() / ".vscode"
                vscode_dir.mkdir(exist_ok=True)
                settings_path = vscode_dir / "settings.json"

                # Load existing settings or create new
                existing_settings = {}
                if settings_path.exists():
                    try:
                        with open(settings_path) as f:
                            existing_settings = json.load(f)
                    except json.JSONDecodeError:
                        pass

                # Add yaml.schemas
                vscode_settings = get_vscode_settings_snippet(str(output))
                existing_settings.update(vscode_settings)

                with open(settings_path, 'w') as f:
                    json.dump(existing_settings, f, indent=2)

                console.print(f"[green]✓[/green] VSCode settings updated: {settings_path}")

            # Show setup instructions
            if setup:
                console.print(print_schema_setup_instructions(str(output)))

        else:
            # Print to stdout
            console.print_json(json.dumps(schema, indent=2))

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to generate schema: {e}")
        raise typer.Exit(1) from e


@app.callback()
def main_callback():
    """🚀 ML Server - Modern CLI for ML model serving."""
    pass


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
