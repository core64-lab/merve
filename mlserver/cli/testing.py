"""``merve test`` — send a request against a running server to smoke-test it."""

from pathlib import Path
from typing import Optional

import typer

from ._app import app, console


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
        merve test --data '{"feature1": 1.5, "feature2": 2.0}'
        merve test --file sample_request.json
        merve test --url http://localhost:8080 --endpoint /predict
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
        console.print("  merve test --data '{\"feature1\": 1.5}'")
        console.print("  merve test --file request.json[/dim]")
        raise typer.Exit(1)

    # Wrap bare feature dicts/lists in the canonical top-level records shape
    # (RFC 0001 D10). Already-shaped bodies pass through untouched.
    input_keys = ("payload", "records", "instances", "ndarray", "inputs", "features")
    if isinstance(payload_data, dict) and not any(k in payload_data for k in input_keys):
        payload_data = {"records": [payload_data]}
    elif isinstance(payload_data, list):
        payload_data = {"records": payload_data}

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
        console.print("  [dim]→ Is the server running? Try: merve serve[/dim]")
        raise typer.Exit(1) from None
    except httpx.TimeoutException:
        console.print("[red]✗[/red] Request timed out")
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] Request failed: {e}")
        raise typer.Exit(1) from e
