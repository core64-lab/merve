"""Integration tests for CLI v2 workflow using mock classifier repo."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import requests

# Add parent directory to path to import fixtures
sys.path.insert(0, str(Path(__file__).parent.parent))
from fixtures.mock_classifier_repo import MockClassifierRepo


def _docker_daemon_available() -> bool:
    """Return True if the Docker CLI is installed AND the daemon responds.

    `mlserver build` needs a running daemon, not just the docker binary
    (see mlserver.container.check_docker_availability, which only checks
    the binary).
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class TestCLIv2WorkflowIntegration:
    """Test the complete CLI v2 workflow with mock classifier repo."""

    @pytest.fixture
    def single_classifier_repo(self):
        """Create a single classifier repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = MockClassifierRepo(tmpdir, multi_classifier=False)
            yield repo
            repo.cleanup()

    @pytest.fixture
    def multi_classifier_repo(self):
        """Create a multi-classifier repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = MockClassifierRepo(tmpdir, multi_classifier=True)
            yield repo
            repo.cleanup()

    def test_single_classifier_serve(self, single_classifier_repo):
        """Test serving a single classifier."""
        # Start server in background (waits for /healthz to respond)
        process = single_classifier_repo.serve_in_background(port=8080)

        try:
            # Test health endpoint
            response = requests.get("http://localhost:8080/healthz", timeout=5)
            assert response.status_code == 200
            health = response.json()
            assert health["status"] == "ok"

            # Test prediction
            payload = {
                "payload": {"records": [{"f1": 1.0, "f2": 2.0, "f3": 3.0, "f4": 4.0, "f5": 5.0}]}
            }
            response = requests.post("http://localhost:8080/predict", json=payload, timeout=30)
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result

        finally:
            # Clean up
            single_classifier_repo.stop_server(process)

    def test_multi_classifier_serve(self, multi_classifier_repo):
        """Test serving multiple classifiers."""
        process1 = None
        process2 = None
        try:
            # Start sentiment classifier (waits for readiness)
            process1 = multi_classifier_repo.serve_in_background(
                classifier_name="sentiment", port=8081
            )

            # Start intent classifier (waits for readiness)
            process2 = multi_classifier_repo.serve_in_background(
                classifier_name="intent", port=8082
            )

            # Test sentiment server
            response = requests.get("http://localhost:8081/healthz", timeout=5)
            assert response.status_code == 200

            # Test intent server
            response = requests.get("http://localhost:8082/healthz", timeout=5)
            assert response.status_code == 200

        finally:
            # Clean up
            if process1 is not None:
                multi_classifier_repo.stop_server(process1)
            if process2 is not None:
                multi_classifier_repo.stop_server(process2)

    def test_version_command(self, single_classifier_repo):
        """Test the version command."""
        result = single_classifier_repo.run_cli_command("version")
        assert result.returncode == 0
        assert "test-classifier" in result.stdout
        assert "1.0.0" in result.stdout

    def test_tag_workflow(self, single_classifier_repo):
        """Test the tagging workflow."""
        # Check initial status (no tags)
        result = single_classifier_repo.run_cli_command("tag")
        assert result.returncode == 0
        assert "test-classifier" in result.stdout

        # Tag with minor bump
        result = single_classifier_repo.run_cli_command(
            "tag", "minor", "--classifier", "test-classifier"
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

        # Check status again (should show tagged)
        result = single_classifier_repo.run_cli_command("tag")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
        assert "Ready" in result.stdout

    def test_multi_classifier_tagging(self, multi_classifier_repo):
        """Test tagging multiple classifiers independently."""
        # Tag sentiment with patch
        result = multi_classifier_repo.run_cli_command("tag", "patch", "--classifier", "sentiment")
        assert result.returncode == 0
        assert "0.0.1" in result.stdout

        # Tag intent with minor
        result = multi_classifier_repo.run_cli_command("tag", "minor", "--classifier", "intent")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

        # Check status table
        result = multi_classifier_repo.run_cli_command("tag")
        assert result.returncode == 0
        assert "sentiment" in result.stdout
        assert "0.0.1" in result.stdout
        assert "intent" in result.stdout
        assert "0.1.0" in result.stdout

    def test_build_command(self, single_classifier_repo):
        """Test building Docker container."""
        if not _docker_daemon_available():
            pytest.skip("Docker daemon not available")

        # Make a change and tag
        single_classifier_repo.make_change()
        single_classifier_repo.run_cli_command("tag", "minor", "--classifier", "test-classifier")

        # Build container
        result = single_classifier_repo.run_cli_command("build")

        assert result.returncode == 0
        assert "Successfully built" in result.stdout or "Building" in result.stdout

    def test_info_command(self, single_classifier_repo):
        """Test the info command for getting server information."""
        # Start server (waits for readiness)
        process = single_classifier_repo.serve_in_background(port=8083)

        try:
            # Get info
            response = requests.get("http://localhost:8083/info", timeout=5)
            assert response.status_code == 200
            info = response.json()

            # Simplified /info response contract
            assert info["classifier"] == "test-classifier"
            assert info["predictor_class"] == "TestPredictor"
            assert "project" in info
            assert "classifier_repository" in info
            assert info["endpoints"]["predict"] == "/predict"
            assert info["endpoints"]["health"] == "/healthz"

        finally:
            # Clean up
            single_classifier_repo.stop_server(process)

    def test_list_classifiers(self, multi_classifier_repo):
        """Test listing available classifiers."""
        result = multi_classifier_repo.run_cli_command("list-classifiers")
        assert result.returncode == 0
        assert "sentiment" in result.stdout
        assert "intent" in result.stdout
        assert "Available Classifiers" in result.stdout

    def test_build_once_commit_image_serves_selected_classifier(self, multi_classifier_repo):
        """Build-once / deploy-many (RFC 0001 D4 / W2.5), daemon-gated.

        A multi-classifier repo builds ONE commit image (no baked classifier).
        Running it with `-e MLSERVER_CLASSIFIER=<one>` must serve /healthz for
        the selected classifier. Skips cleanly when no Docker daemon is present.
        """
        if not _docker_daemon_available():
            pytest.skip("Docker daemon not available")

        import time

        from mlserver.version import get_repository_name

        repo_path = multi_classifier_repo.repo_path

        # Build ONCE with no --classifier -> a single commit image bundling all.
        build = multi_classifier_repo.run_cli_command("build")
        assert build.returncode == 0, build.stdout + build.stderr

        repo = get_repository_name(str(repo_path))
        image = f"{repo}:latest"
        container_name = "merve-buildonce-smoke"

        # Clean any stale container from a previous run.
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        # Select a classifier at run time via the environment variable.
        run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "-e",
                "MLSERVER_CLASSIFIER=sentiment",
                "--name",
                container_name,
                image,
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr

        try:
            deadline = time.time() + 60
            healthy = False
            while time.time() < deadline:
                # Use the container's own curl (installed in the runtime stage)
                # so the check does not depend on host port routing.
                health = subprocess.run(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "curl",
                        "-fsS",
                        "http://localhost:8000/healthz",
                    ],
                    capture_output=True,
                    text=True,
                )
                if health.returncode == 0:
                    healthy = True
                    break
                time.sleep(2)

            assert healthy, "commit image did not serve /healthz for MLSERVER_CLASSIFIER=sentiment"
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
