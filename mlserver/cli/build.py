"""Container lifecycle commands: ``build``, ``push``, ``images``, ``clean``, ``run``."""

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.table import Table

from ..container import build_container, check_docker_availability, list_images, remove_images
from ..multi_classifier import detect_multi_classifier_config, list_available_classifiers
from ..version_control import GitVersionManager, safe_push_container
from ._app import app, console, detect_config_file, err_console


@app.command()
def build(
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier",
        "-c",
        help="Classifier to build (can be simple name or full tag: name-vX.Y.Z-mlserver-hash)",
    ),
    path: str = typer.Option(".", "--path", help="Path to classifier project"),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Config file to use (auto-detected if not specified)"
    ),
    registry: Optional[str] = typer.Option(None, "--registry", help="Container registry URL"),
    tag_prefix: Optional[str] = typer.Option(
        None, "--tag-prefix", help="Tag prefix for container names"
    ),
    build_arg: Optional[list[str]] = typer.Option(
        None, "--build-arg", help="Build arguments (key=value)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not use cache when building"),
    platform: Optional[str] = typer.Option(
        None,
        "--platform",
        help=(
            "Target platform for the image, e.g. linux/amd64 (single platform only). "
            "Cross-architecture builds require BuildKit with binfmt/QEMU emulation "
            "or docker buildx on the host."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    force: bool = typer.Option(
        False, "--force", help="Skip validation prompts and continue with build"
    ),
    per_classifier_image: bool = typer.Option(
        False,
        "--per-classifier-image",
        help=(
            "Escape hatch: build one baked image per classifier (pre-W2.5 behavior) "
            "instead of a single build-once commit image. Use ONLY for classifiers "
            "whose conflicting dependencies cannot share one image. Requires --classifier."
        ),
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
        from ..version import get_git_info
        from ..version_control import (
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
            expected_classifier_commit = tag_commits["classifier_commit"]
            expected_mlserver_commit = parsed["mlserver_commit"]

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
                    "[red]⚠️  MISMATCH[/red]" if classifier_mismatch else "[green]✓[/green]"
                )
                mlserver_marker = (
                    "[red]⚠️  MISMATCH[/red]" if mlserver_mismatch else "[green]✓[/green]"
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
            if "=" not in arg:
                console.print(
                    f"[red]✗[/red] Invalid --build-arg '{arg}': expected format KEY=value",
                    style="bold red",
                )
                raise typer.Exit(1)
            key, value = arg.split("=", 1)
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
        per_classifier_image=per_classifier_image,
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
    from ..container import push_classifier_alias

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
    registry: str = typer.Option(..., "--registry", "-r", help="Container registry URL"),
    classifier: Optional[str] = typer.Option(
        None,
        "--classifier",
        "-c",
        help="Classifier to push (required for multi-classifier configs)",
    ),
    path: str = typer.Option(".", "--path", help="Path to classifier project"),
    tag_prefix: Optional[str] = typer.Option(
        None, "--tag-prefix", help="Tag prefix for container names"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force push even if not on tagged commit or tag exists"
    ),
    version_source: Optional[str] = typer.Option(
        None,
        "--version-source",
        help=(
            "[DEPRECATED] Version source: 'git-tag', 'config', or 'auto'. "
            "Git tags are the canonical version source (RFC 0001 D3); "
            "this flag will be removed in v0.5.0."
        ),
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
            style="bold red",
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
            path=path,
            registry=registry,
            classifier=classifier,
            tag_prefix=tag_prefix,
            force=force,
            config_file=push_config_file,
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
        version_source=version_source,
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
    path: str = typer.Option(".", "--path", help="Path to classifier project"),
    classifier: Optional[str] = typer.Option(
        None, "--classifier", "-c", help="Classifier name (for multi-classifier configs)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON (machine-readable)"),
):
    """📋 List Docker images for the classifier project.

    For multi-classifier configs, use --classifier to filter images.
    """
    image_list = list_images(str(path), classifier_name=classifier)

    if json_output:
        print(
            json.dumps(
                {
                    "images": image_list,
                    "count": len(image_list),
                    "classifier": classifier,
                },
                indent=2,
            )
        )
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
        table.add_row(image["tag"], image["image_id"], image["created"], image["size"])

    console.print(table)


@app.command()
def clean(
    path: str = typer.Option(".", "--path", help="Path to classifier project"),
    force: bool = typer.Option(False, "--force", "-f", help="Force removal without confirmation"),
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
        None, "--classifier", "-c", help="Classifier to run (required for multi-classifier configs)"
    ),
    path: str = typer.Option(".", "--path", help="Path to classifier project"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to expose the container on"),
    version: Optional[str] = typer.Option(
        None, "--version", help="Specific version to run (default: latest)"
    ),
    detach: bool = typer.Option(False, "--detach", "-d", help="Run container in background"),
    name: Optional[str] = typer.Option(None, "--name", help="Container name"),
    env: Optional[list[str]] = typer.Option(
        None, "--env", "-e", help="Environment variables (KEY=value)"
    ),
    volume: Optional[list[str]] = typer.Option(
        None, "--volume", help="Volume mounts (host:container)"
    ),
):
    """🚀 Run Docker container for the classifier."""
    if not check_docker_availability():
        console.print("[red]✗[/red] Docker is not available or not running", style="bold red")
        raise typer.Exit(1)

    # Detect config file (inside the project directory)
    config_file = detect_config_file(None, base_dir=Path(path))

    # Check if multi-classifier config
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
    from ..version import get_repository_name

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
