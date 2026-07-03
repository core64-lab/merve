"""Project scaffolding and diagnostics commands.

``init``, ``init-github``, ``validate``, ``doctor``, ``status``,
``list-classifiers``, and ``schema``.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from ..container import check_docker_availability
from ..multi_classifier import (
    detect_multi_classifier_config,
    get_default_classifier,
    list_available_classifiers,
)
from ._app import _display_check_result, app, console, detect_config_file


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
    from ..github_actions import check_github_actions_setup

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
        "--path", "-C",
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
    from ..init_project import init_mlserver_project

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
        "--path", "-C",
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
    from ..github_actions import init_github_actions

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
    from ..doctor import CheckStatus, run_validation_checks

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
    project_path: Path = typer.Option(".", "--path", "-C", help="Project path to diagnose"),
):
    """Diagnose common issues and environment problems.

    Checks system requirements, project configuration, dependencies,
    and provides recommendations for fixing issues.

    Examples:
        mlserver doctor
        mlserver doctor --verbose
        mlserver doctor --path ./my-project
    """
    from ..doctor import run_all_checks

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

    from ..schema_generator import (
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
