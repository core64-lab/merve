"""``mlserver serve`` — launch the FastAPI inference server from a YAML config."""

import os
from pathlib import Path
from typing import Optional

import typer
import uvicorn
import yaml
from rich.panel import Panel

from ..config import AppConfig
from ..logging_conf import configure_logging
from ..multi_classifier import (
    detect_multi_classifier_config,
    extract_single_classifier_config,
    get_default_classifier,
    list_available_classifiers,
    load_multi_classifier_config,
)
from ..server import create_app
from ._app import LogLevel, app, console, detect_config_file, resolve_relative_paths


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
