"""Versioning commands: ``version`` (display) and ``tag`` (create/inspect tags)."""

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from ..version import get_version_info
from ..version_control import GitVersionManager, VersionControlError
from ._app import app, console, detect_config_file, removed_flag_callback


class BumpType(str, Enum):
    """Version bump type options."""

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@app.command()
def version(
    path: str = typer.Option(".", "--path", "-C", help="Path to classifier project"),
    classifier: Optional[str] = typer.Option(
        None, "--classifier", "-c", help="Classifier name (for multi-classifier configs)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed MLServer tool information"
    ),
    legacy_path_short: Optional[str] = typer.Option(
        None,
        "-p",
        hidden=True,
        callback=removed_flag_callback(
            "-p no longer means --path here (it is reserved for --port across "
            "the CLI, RFC 0001 D8): use -C or --path for the project directory. "
            "See docs/migration-0.5.md."
        ),
    ),
):
    """📦 Display version information for the classifier project.

    Use --detailed to show MLServer tool version, commit, and installation source.
    For multi-classifier configs, use --classifier to specify which one.
    """
    version_info = get_version_info(path, classifier_name=classifier)

    # If no classifier project found, show only MLServer tool version
    if "error" in version_info:
        error_msg = version_info["error"]
        if "mlserver.yaml not found" in error_msg:
            # No classifier project - show MLServer tool version only
            import mlserver as mlserver_module

            from ..version_control import get_mlserver_commit_hash

            mlserver_commit = get_mlserver_commit_hash()
            mlserver_version = (
                mlserver_module.__version__
                if hasattr(mlserver_module, "__version__")
                else "unknown"
            )
            mlserver_location = Path(mlserver_module.__file__).parent

            # Determine installation type
            install_type = "unknown"
            if (mlserver_location.parent / ".git").exists():
                install_type = "git (editable)"
            elif mlserver_location.parent.name.endswith(
                ".egg-info"
            ) or mlserver_location.parent.name.endswith(".dist-info"):
                install_type = "package"
            elif "site-packages" in str(mlserver_location):
                install_type = "pip"

            if json_output:
                mlserver_info = {
                    "mlserver_tool": {
                        "version": mlserver_version,
                        "commit": mlserver_commit,
                        "install_location": str(mlserver_location),
                        "install_type": install_type,
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

            from ..version_control import get_mlserver_commit_hash

            mlserver_commit = get_mlserver_commit_hash()
            version_info["mlserver_tool"] = {
                "version": (
                    mlserver_module.__version__
                    if hasattr(mlserver_module, "__version__")
                    else "unknown"
                ),
                "commit": mlserver_commit,
                "install_location": str(Path(mlserver_module.__file__).parent),
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
        table.add_row("Description", classifier_info.get("description", ""))
        table.add_row("Model Version", model["version"])
        table.add_row("API Version", api["version"])

        if git:
            table.add_row("Git Commit", f"{git['commit']} ({git['branch']})")
            if git["tag"]:
                table.add_row("Git Tag", git["tag"])
            if git["is_dirty"]:
                table.add_row("Status", "⚠️  Uncommitted changes")

        # Add MLServer tool information if --detailed
        if detailed:
            import mlserver as mlserver_module

            from ..version_control import get_mlserver_commit_hash

            mlserver_commit = get_mlserver_commit_hash()
            mlserver_version = (
                mlserver_module.__version__
                if hasattr(mlserver_module, "__version__")
                else "unknown"
            )
            mlserver_location = Path(mlserver_module.__file__).parent

            # Determine installation type
            install_type = "unknown"
            if (mlserver_location.parent / ".git").exists():
                install_type = "git (editable)"
            elif mlserver_location.parent.name.endswith(
                ".egg-info"
            ) or mlserver_location.parent.name.endswith(".dist-info"):
                install_type = "package"
            elif "site-packages" in str(mlserver_location):
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
def tag(
    bump_type: Optional[BumpType] = typer.Argument(
        None, help="Version bump type: major, minor, or patch"
    ),
    classifier: Optional[str] = typer.Option(
        None, "--classifier", "-c", help="Classifier name to tag (required for tagging)"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Config file path (defaults to mlserver.yaml)"
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="Tag message (defaults to 'Release <classifier> vX.Y.Z')"
    ),
    path: str = typer.Option(".", "--path", "-C", help="Path to classifier project"),
    allow_missing_mlserver: bool = typer.Option(
        False,
        "--allow-missing-mlserver",
        help="Allow tagging even if mlserver commit cannot be determined (dev/testing only)",
    ),
    status_only: bool = typer.Option(
        False,
        "--status",
        help="Show tag status for all classifiers (default when no bump type is given)",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output tag status as JSON (status mode only, machine-readable)"
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
            from ..version_control import get_mlserver_commit_hash, parse_classifier_tag

            current_mlserver_commit = get_mlserver_commit_hash() or "unknown"

            if json_output:
                doc = {"classifiers": {}, "mlserver_commit": current_mlserver_commit}
                for clf_name, status in classifiers_status.items():
                    parsed = (
                        parse_classifier_tag(status["latest_tag"])
                        if status.get("latest_tag")
                        else None
                    )
                    doc["classifiers"][clf_name] = {
                        "current_version": status["current_version"],
                        "latest_tag": status["latest_tag"],
                        "tag_format": parsed["format"] if parsed else None,
                        "tag_mlserver_commit": (parsed["mlserver_commit"] if parsed else None),
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
                version = status["current_version"] or "No tags"
                status_text = status["status"]
                recommendation = status["recommendation"] or "-"

                # Extract mlserver commit from the latest tag (only legacy
                # tags encode it; canonical tags show n/a)
                mlserver_commit = "-"
                if status.get("latest_tag"):
                    parsed = parse_classifier_tag(status["latest_tag"])
                    if parsed and parsed["mlserver_commit"]:
                        tag_mlserver = parsed["mlserver_commit"][:7]
                        current_mlserver_short = current_mlserver_commit[:7]
                        if tag_mlserver == current_mlserver_short:
                            mlserver_commit = f"{tag_mlserver} [green]✓[/green]"
                        else:
                            mlserver_commit = f"{tag_mlserver} [yellow]⚠️[/yellow]"
                    else:
                        mlserver_commit = "[dim]n/a[/dim]"

                # Color coding for status
                if status["on_tagged_commit"]:
                    status_style = "green"
                elif status["commits_since_tag"] and status["commits_since_tag"] > 0:
                    status_style = "yellow"
                else:
                    status_style = "red"

                table.add_row(
                    clf_name,
                    version,
                    mlserver_commit,
                    f"[{status_style}]{status_text}[/{status_style}]",
                    recommendation,
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
        from ..validation import get_tag_validation_suite

        validation_suite = get_tag_validation_suite()
        all_passed, results = validation_suite.validate(
            project_path=path, classifier_name=classifier
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
        if tag_info["previous_version"]:
            console.print(
                f"  [yellow]📝 Version:[/yellow] {tag_info['previous_version']} → "
                f"{tag_info['version']} ({bump_type.value} bump)"
            )
        else:
            console.print(f"  [yellow]📝 Version:[/yellow] {tag_info['version']} (initial release)")

        # MLServer info
        console.print(f"  [yellow]🔧 MLServer commit:[/yellow] {tag_info['mlserver_commit']}")

        # Get classifier commit for reference
        from ..version import get_git_info

        git_info = get_git_info(path)
        if git_info:
            console.print(f"  [yellow]📦 Classifier commit:[/yellow] {git_info.commit}")

        # Check if GitHub Actions is set up and validate workflow
        from ..github_actions import check_github_actions_setup, validate_workflow_comprehensive

        github_actions_configured = check_github_actions_setup(path)

        # Validate workflow if it exists
        workflow_valid = True
        workflow_warnings = []
        if github_actions_configured:
            workflow_valid, workflow_warnings, workflow_details = validate_workflow_comprehensive(
                path
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
                console.print("  Run: [cyan]merve init-github --force[/cyan]")
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
            console.print(f"  - Build: [cyan]merve build --classifier {classifier}[/cyan]")
            console.print(
                f"  - Push: [cyan]merve push --classifier {classifier} --registry <url>[/cyan]"
            )
        elif github_actions_configured and not workflow_valid:
            # Workflow exists but is outdated/invalid
            console.print(
                "  [bold]1. Regenerate workflow:[/bold] [cyan]merve init-github --force[/cyan]"
            )
            console.print(
                "  2. Commit and push: [cyan]git add .github && "
                "git commit -m 'Update workflow' && git push[/cyan]"
            )
            console.print("  3. Push tags: [cyan]git push --tags[/cyan]")
            console.print()
            console.print("[dim]💡 Or build manually:[/dim]")
            console.print(f"  - Build: [cyan]merve build --classifier {classifier}[/cyan]")
            console.print(
                f"  - Push: [cyan]merve push --classifier {classifier} --registry <url>[/cyan]"
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
            console.print("  1. Add workflow: [cyan]merve init-github[/cyan]")
            console.print(
                "  2. Commit and push: [cyan]git add .github && "
                "git commit -m 'Add CI/CD' && git push[/cyan]"
            )
            console.print("  3. Push tags: [cyan]git push --tags[/cyan]")
            console.print()
            console.print("  [dim]Option 2: Build and push manually[/dim]")
            console.print("  1. Push tags: [cyan]git push --tags[/cyan]")
            console.print(f"  2. Build: [cyan]merve build --classifier {classifier}[/cyan]")
            console.print(
                f"  3. Push: [cyan]merve push --classifier {classifier} --registry <url>[/cyan]"
            )

    except VersionControlError as e:
        console.print(f"[red]✗[/red] {e}", style="bold red")
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}", style="bold red")
        raise typer.Exit(1) from e
