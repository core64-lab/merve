"""Comprehensive tests for container module functionality."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlserver.config import AppConfig, BuildConfig, PredictorConfig, ServerConfig
from mlserver.container import (
    ContainerError,
    _analyze_python_imports,
    _build_mlserver_wheel,
    _find_mlserver_source,
    _looks_like_file_path,
    _resolve_local_import,
    build_container,
    check_docker_availability,
    detect_required_files,
    generate_dockerfile,
    generate_dockerignore,
    list_images,
    push_container,
    remove_images,
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create basic project structure
        (project_path / "predictor.py").write_text("""
import pickle
from typing import List, Dict, Any
import pandas as pd

class TestPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X: List[Dict[str, Any]]) -> List[int]:
        return [1, 0, 1]
""")

        # Create a dummy model file
        (project_path / "model.pkl").write_bytes(b"dummy model data")

        # Create requirements.txt
        (project_path / "requirements.txt").write_text("pandas>=1.0\nscikit-learn>=1.0\n")

        # Create config file
        (project_path / "mlserver.yaml").write_text("""
server:
  host: 0.0.0.0
  port: 8000

predictor:
  module: predictor
  class_name: TestPredictor
  init_kwargs:
    model_path: "./model.pkl"

classifier:
  repository: "test-repo"
  name: "test-classifier"
  version: "1.0.0"
  description: "Test classifier"

model:
  version: "1.0.0"
""")

        yield project_path


@pytest.fixture
def mock_config():
    """Create a mock AppConfig for testing."""
    from mlserver.config import ApiConfig

    return AppConfig(
        server=ServerConfig(host="0.0.0.0", port=8000),
        predictor=PredictorConfig(
            module="test.predictor",
            class_name="TestPredictor",
            init_kwargs={"model_path": "./model.pkl"},
        ),
        classifier={
            "name": "test-classifier",
            "version": "1.0.0",
            "description": "Test classifier",
        },
        api=ApiConfig(version="v1", adapter="auto", thread_safe_predict=False),
        build=BuildConfig(registry="test-registry", tag_prefix="ml-models"),
    )


@pytest.fixture(autouse=True)
def _isolate_mlserver_env(monkeypatch):
    """Keep tests deterministic regardless of ambient MLSERVER_* env vars.

    container.py reads these for source discovery and git metadata embedding;
    a developer machine with them set must not change test outcomes.
    """
    for var in (
        "MLSERVER_SOURCE_PATH",
        "MLSERVER_GIT_URL",
        "MLSERVER_GIT_COMMIT",
        "MLSERVER_GIT_TAG",
        "MLSERVER_GIT_BRANCH",
    ):
        monkeypatch.delenv(var, raising=False)


class TestContainerError:
    """Test ContainerError exception."""

    def test_container_error_creation(self):
        """Test ContainerError can be created and raised."""
        with pytest.raises(ContainerError, match="Test error"):
            raise ContainerError("Test error")


class TestFindMLServerSource:
    """Test _find_mlserver_source function."""

    def test_find_mlserver_source_env_var(self):
        """Test finding mlserver source via environment variable."""
        test_path = "/test/mlserver/path"
        with (
            patch.dict(os.environ, {"MLSERVER_SOURCE_PATH": test_path}),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = _find_mlserver_source()
            assert result == test_path

    def test_find_mlserver_source_env_var_not_exists(self):
        """Test env var path that doesn't exist."""
        test_path = "/nonexistent/path"
        with (
            patch.dict(os.environ, {"MLSERVER_SOURCE_PATH": test_path}),
            patch("pathlib.Path.exists", return_value=False),
        ):
            # Should continue to other strategies
            result = _find_mlserver_source()
            # Result depends on other strategies, but shouldn't be the env var path
            assert result != test_path

    def test_find_mlserver_source_pyproject_detection(self, tmp_path):
        """Test finding mlserver source via pyproject.toml walk-up.

        Builds a fake source tree and repoints the module's __file__ so the
        walk-up starts inside the fake tree instead of the real install.
        """
        source_root = tmp_path / "wrapper_src"
        package_dir = source_root / "mlserver"
        package_dir.mkdir(parents=True)
        (source_root / "pyproject.toml").write_text(
            '[project]\nname = "mlserver-fastapi-wrapper"\nversion = "1.0.0"\n'
        )

        fake_module_file = str(package_dir / "container.py")
        with patch("mlserver.container.__file__", fake_module_file):
            result = _find_mlserver_source()

        assert result == str(source_root)

    def test_find_mlserver_source_pyproject_name_mismatch(self, tmp_path):
        """A pyproject.toml belonging to a different project is not a match."""
        # Nest deep enough that the 5-level walk-up stays inside tmp_path
        source_root = tmp_path / "a" / "b" / "c" / "d" / "other_project"
        package_dir = source_root / "mlserver"
        package_dir.mkdir(parents=True)
        (source_root / "pyproject.toml").write_text(
            '[project]\nname = "some-other-package"\nversion = "1.0.0"\n'
        )

        fake_module_file = str(package_dir / "container.py")
        with patch("mlserver.container.__file__", fake_module_file):
            result = _find_mlserver_source()

        assert result is None

    def test_find_mlserver_source_no_tomllib(self):
        """Test when neither tomllib nor tomli are available."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("pathlib.Path.exists", return_value=True),
            patch("importlib.util.find_spec", return_value=None),
        ):  # No tomllib/tomli
            result = _find_mlserver_source()
            # Should handle missing toml libraries gracefully
            assert isinstance(result, (str, type(None)))


class TestBuildMLServerWheel:
    """Test _build_mlserver_wheel function (rewritten 2026-07-03 for current API).

    Current contract: returns the wheel FILENAME (str) after copying it into
    the project dir, or None on any failure (it never raises ContainerError).
    """

    def test_build_mlserver_wheel_success(self, tmp_path):
        """Successful build copies the wheel into the project and returns its name."""
        source_dir = tmp_path / "source"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        wheel_name = "merve-1.0.0-py3-none-any.whl"
        dist_dir = source_dir / "dist"
        dist_dir.mkdir(parents=True)
        (dist_dir / wheel_name).write_bytes(b"dummy wheel")

        with patch("mlserver.container.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = _build_mlserver_wheel(str(source_dir), str(project_dir))

        assert result == wheel_name
        assert (project_dir / wheel_name).exists()

        # The build must run `<python> -m build --wheel` in the source dir
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == [sys.executable, "-m", "build", "--wheel"]
        assert mock_run.call_args.kwargs["cwd"] == str(source_dir)

    def test_build_mlserver_wheel_picks_newest_wheel(self, tmp_path):
        """When multiple wheels exist, the most recently modified one wins."""
        source_dir = tmp_path / "source"
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        dist_dir = source_dir / "dist"
        dist_dir.mkdir(parents=True)

        old_wheel = dist_dir / "merve-0.9.0-py3-none-any.whl"
        new_wheel = dist_dir / "merve-1.1.0-py3-none-any.whl"
        old_wheel.write_bytes(b"old")
        new_wheel.write_bytes(b"new")
        os.utime(old_wheel, (1000000, 1000000))
        os.utime(new_wheel, (2000000, 2000000))

        with patch("mlserver.container.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = _build_mlserver_wheel(str(source_dir), str(project_dir))

        assert result == new_wheel.name

    def test_build_mlserver_wheel_build_failure_returns_none(self, tmp_path):
        """Build failure returns None (no exception) and copies nothing."""
        source_dir = tmp_path / "source"
        project_dir = tmp_path / "project"
        source_dir.mkdir()
        project_dir.mkdir()

        with patch("mlserver.container.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Build failed")

            result = _build_mlserver_wheel(str(source_dir), str(project_dir))

        assert result is None
        assert list(project_dir.glob("*.whl")) == []

    def test_build_mlserver_wheel_no_dist_files(self, tmp_path):
        """Build succeeding but producing no wheel files returns None."""
        source_dir = tmp_path / "source"
        project_dir = tmp_path / "project"
        source_dir.mkdir()
        project_dir.mkdir()

        with patch("mlserver.container.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # No dist/ directory, no wheel files

            result = _build_mlserver_wheel(str(source_dir), str(project_dir))

        assert result is None


class TestDetectRequiredFiles:
    """Test detect_required_files function."""

    def test_detect_required_files_basic(self, temp_project_dir, mock_config):
        """Test basic file detection."""
        result = detect_required_files(str(temp_project_dir), mock_config)

        assert isinstance(result, dict)
        # Check new return structure
        assert "required_files" in result
        assert "analysis" in result

        # Should detect our created files
        required = result["required_files"]
        assert "model.pkl" in required
        assert "requirements.txt" in required
        assert "mlserver.yaml" in required

        # Check analysis structure
        analysis = result["analysis"]
        assert "artifact_files" in analysis
        assert "config_files" in analysis

    def test_detect_required_files_with_config_refs(self, temp_project_dir):
        """Test file detection with config file references."""
        from mlserver.config import ApiConfig

        config = AppConfig(
            predictor=PredictorConfig(
                module="predictor",
                class_name="TestPredictor",
                init_kwargs={
                    "model_path": "./model.pkl",
                    "preprocessor_path": "./preprocessor.pkl",
                    "feature_order_path": "./features.json",
                },
            ),
            classifier={"name": "test-classifier", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )

        # Create the referenced files
        (temp_project_dir / "preprocessor.pkl").write_bytes(b"dummy preprocessor")
        (temp_project_dir / "features.json").write_text('["feature1", "feature2"]')

        result = detect_required_files(str(temp_project_dir), config)

        required = result["required_files"]
        assert "preprocessor.pkl" in required
        assert "features.json" in required

    def test_detect_required_files_missing_files(self, temp_project_dir, mock_config):
        """Test file detection with missing files."""
        # Reference a file that doesn't exist
        from mlserver.config import ApiConfig

        config = AppConfig(
            predictor=PredictorConfig(
                module="predictor",
                class_name="TestPredictor",
                init_kwargs={"model_path": "./nonexistent.pkl"},
            ),
            classifier={"name": "test-classifier", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )

        result = detect_required_files(str(temp_project_dir), config)

        # Should handle missing files gracefully - function doesn't track missing files anymore
        # Just check it returns valid structure
        assert isinstance(result, dict)
        assert "required_files" in result
        assert "analysis" in result


class TestUtilityFunctions:
    """Test utility functions."""

    def test_looks_like_file_path_positive_cases(self):
        """Test _looks_like_file_path with file-like strings."""
        # Key must suggest path AND value must look like path
        assert _looks_like_file_path("model_path", "./model.pkl")
        assert _looks_like_file_path("config_file", "../config.yaml")
        assert _looks_like_file_path("data_file", "data/train.csv")
        # 'script' doesn't match path_indicators, check with 'path' key
        assert _looks_like_file_path("script_path", "scripts/train.py")

    def test_looks_like_file_path_negative_cases(self):
        """Test _looks_like_file_path with non-file strings."""
        assert not _looks_like_file_path("port", "8000")
        assert not _looks_like_file_path("name", "test_model")
        assert not _looks_like_file_path("debug", "true")
        assert not _looks_like_file_path("url", "http://example.com")
        assert not _looks_like_file_path("threshold", "0.5")

    def test_analyze_python_imports_returns_local_files_only(self, temp_project_dir):
        """_analyze_python_imports returns a set of resolved local file paths.

        Rewritten 2026-07-03: current return type is Set[str] containing only
        imports that resolve to files inside the project; stdlib and
        third-party imports are excluded.
        """
        (temp_project_dir / "helper_module.py").write_text("def helper(): pass")
        pkg_dir = temp_project_dir / "utils_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        python_file = temp_project_dir / "main_entry.py"
        python_file.write_text("""
import os
import sys
from pathlib import Path
import pandas as pd
import sklearn.ensemble
import helper_module
from utils_pkg import something
""")

        result = _analyze_python_imports(str(temp_project_dir), "main_entry.py")

        assert isinstance(result, set)
        # Only the two local imports resolve to project files
        assert result == {"helper_module.py", "utils_pkg/__init__.py"}

    def test_resolve_local_import(self, temp_project_dir):
        """Test _resolve_local_import function."""
        # Create a local module
        local_module = temp_project_dir / "local_helper.py"
        local_module.write_text("def helper_function(): pass")

        result = _resolve_local_import(str(temp_project_dir), "local_helper")
        assert result == "local_helper.py"

        # Test non-existent module
        result = _resolve_local_import(str(temp_project_dir), "nonexistent")
        assert result is None


class TestDockerfileGeneration:
    """Test Dockerfile generation and Docker file writing (rewritten 2026-07-03).

    Covers the current list-based generate_dockerfile API and the
    _write_docker_files 3-tuple contract. The old dict-based required_files
    API no longer exists.
    """

    def test_generate_dockerfile_copy_generation(self, temp_project_dir, mock_config):
        """Root files are batched into one COPY; subdirectories copied as trees."""
        models_dir = temp_project_dir / "models"
        models_dir.mkdir()
        (models_dir / "model.pkl").write_bytes(b"model data")

        required_files = ["predictor.py", "model.pkl", "models/model.pkl", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert isinstance(result, str)
        assert "FROM python:" in result
        # Subdirectory files become a directory COPY
        assert "COPY models/ ./models/" in result
        # Root files are batched into a single COPY, in input order
        assert "COPY predictor.py model.pkl mlserver.yaml ./" in result
        # requirements.txt exists in the fixture project -> dependency install
        assert "COPY requirements.txt ." in result
        assert "RUN pip install --no-cache-dir -r requirements.txt" in result
        assert 'CMD ["merve", "serve", "mlserver.yaml"]' in result

    def test_generate_dockerfile_expose_and_healthcheck_use_config_port(self, temp_project_dir):
        """EXPOSE and HEALTHCHECK follow server.port from config (not hardcoded 8000)."""
        from mlserver.config import ApiConfig

        config = AppConfig(
            server=ServerConfig(host="0.0.0.0", port=9123),
            predictor=PredictorConfig(module="predictor", class_name="TestPredictor"),
            classifier={"name": "test-classifier", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )

        result = generate_dockerfile(str(temp_project_dir), config, ["predictor.py"])

        assert "EXPOSE 9123" in result
        assert "http://localhost:9123/healthz" in result
        assert "EXPOSE 8000" not in result

    def test_generate_dockerfile_classifier_name_param_overrides_config(
        self, temp_project_dir, mock_config
    ):
        """An explicit classifier_name wins over the config classifier metadata."""
        result = generate_dockerfile(
            str(temp_project_dir), mock_config, ["predictor.py"], classifier_name="explicit-clf"
        )

        assert "# Generated Dockerfile for explicit-clf v1.0.0" in result
        assert 'LABEL com.classifier.name="explicit-clf"' in result

    def test_generate_dockerfile_classifier_name_falls_back_to_config(
        self, temp_project_dir, mock_config
    ):
        """Without an explicit classifier_name the config classifier name is used."""
        result = generate_dockerfile(str(temp_project_dir), mock_config, ["predictor.py"])

        assert "# Generated Dockerfile for test-classifier v1.0.0" in result
        assert 'LABEL com.classifier.name="test-classifier"' in result

    def test_generate_dockerfile_wheel_install_section(self, temp_project_dir, mock_config):
        """has_wheel=True copies the wheel in, installs it, then removes it.

        Pins a dev version: the wheel path only exists for non-release builds
        (a release would pin the framework instead), so the test must not
        depend on whatever version happens to be installed in the venv.
        """
        from mlserver.defaults import CONTAINER_TEMP_DIR as temp_dir

        with patch("mlserver.container._get_installed_mlserver_version", return_value="0.5.1.dev1"):
            result = generate_dockerfile(
                str(temp_project_dir), mock_config, ["predictor.py"], has_wheel=True
            )

        assert f"COPY merve*.whl {temp_dir}/" in result
        assert (
            f"RUN pip install --no-cache-dir {temp_dir}/merve*.whl && rm {temp_dir}/merve*.whl"
        ) in result

        # The wheel is installed in the BUILDER stage (into /opt/venv), not
        # in the runtime stage
        builder_stage = result.split("# ============================ Stage 2")[0]
        assert f"COPY merve*.whl {temp_dir}/" in builder_stage

    def test_generate_dockerfile_two_stage_structure(self, temp_project_dir, mock_config):
        """Generated Dockerfile is a two-stage build (RFC 0001, D15).

        Stage 1 (builder): build-essential + venv + dependency install.
        Stage 2 (runtime): curl only, venv copied over, no compilers.
        """
        result = generate_dockerfile(
            str(temp_project_dir), mock_config, ["predictor.py", "mlserver.yaml"]
        )

        # Exactly two FROM directives, first one named "builder"
        from_lines = [line for line in result.splitlines() if line.startswith("FROM ")]
        assert len(from_lines) == 2
        assert from_lines[0].endswith(" AS builder")
        assert " AS " not in from_lines[1]

        # Builder creates the venv; runtime copies it and puts it on PATH
        assert "RUN python -m venv /opt/venv" in result
        assert "COPY --from=builder /opt/venv /opt/venv" in result
        assert result.count('ENV PATH="/opt/venv/bin:$PATH"') == 2

        # Build tooling only in the builder stage
        builder_stage, runtime_stage = result.split("# ============================ Stage 2")
        assert "build-essential" in builder_stage
        assert "build-essential" not in runtime_stage
        assert "gcc" not in runtime_stage
        assert "g++" not in runtime_stage

        # Runtime stage installs curl (healthcheck) and runs the app
        assert "curl" in runtime_stage
        assert "WORKDIR /app" in runtime_stage
        assert "HEALTHCHECK" in runtime_stage
        assert 'CMD ["merve", "serve", "mlserver.yaml"]' in runtime_stage

        # Project files are copied in the runtime stage, not the builder
        assert "COPY predictor.py" in runtime_stage

    def test_generate_dockerfile_requirements_installed_in_builder(
        self, temp_project_dir, mock_config
    ):
        """requirements.txt is installed in the builder stage (into the venv)."""
        result = generate_dockerfile(
            str(temp_project_dir), mock_config, ["predictor.py", "mlserver.yaml"]
        )

        builder_stage, runtime_stage = result.split("# ============================ Stage 2")
        assert "COPY requirements.txt ." in builder_stage
        assert "RUN pip install --no-cache-dir -r requirements.txt" in builder_stage
        assert "RUN pip install --no-cache-dir -r requirements.txt" not in runtime_stage

    def test_generate_dockerfile_git_only_in_builder_stage(self, temp_project_dir, mock_config):
        """needs_git installs git in the builder stage only."""
        result = generate_dockerfile(
            str(temp_project_dir), mock_config, ["predictor.py"], needs_git=True
        )

        builder_stage, runtime_stage = result.split("# ============================ Stage 2")
        assert "git" in builder_stage
        # runtime apt line installs curl only
        runtime_apt = [line for line in runtime_stage.splitlines() if "apt-get install" in line]
        assert len(runtime_apt) == 1
        assert "git" not in "".join(runtime_stage.split("apt-get install")[1].split("rm -rf")[0])

    def test_write_docker_files_uses_config_base_image(self, temp_project_dir):
        """_write_docker_files honors build.base_image from config (custom base image)."""
        from mlserver.config import ApiConfig

        config = AppConfig(
            predictor=PredictorConfig(module="predictor", class_name="TestPredictor"),
            classifier={"name": "test-classifier", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
            build=BuildConfig(base_image="python:3.12-slim"),
        )
        from mlserver.container import _write_docker_files

        analysis = {"auto_excludes": {"__pycache__", ".git"}}
        dockerfile_path, dockerignore_path, temp_config_file = _write_docker_files(
            str(temp_project_dir),
            config,
            ["predictor.py"],
            analysis,
            has_wheel=False,
            needs_git=False,
        )

        assert temp_config_file is None
        assert dockerfile_path.exists()
        assert dockerignore_path.exists()
        assert "FROM python:3.12-slim" in dockerfile_path.read_text()

    def test_write_docker_files_multi_classifier_writes_and_renames_temp_config(
        self, temp_project_dir, mock_config
    ):
        """Multi-classifier builds write a per-classifier config and COPY-rename it.

        _write_docker_files returns a 3-tuple; for multi-classifier configs the
        third element is the temporary single-classifier config, and the
        Dockerfile renames it to mlserver.yaml inside the image.
        """
        from mlserver.container import _write_docker_files

        # Make the on-disk config a multi-classifier one
        (temp_project_dir / "mlserver.yaml").write_text("""
classifiers:
  clf-a:
    predictor:
      module: predictor
      class_name: TestPredictor
  clf-b:
    predictor:
      module: predictor
      class_name: TestPredictor
""")

        required_files = ["predictor.py", "mlserver.yaml"]
        analysis = {"auto_excludes": {"__pycache__"}}

        dockerfile_path, dockerignore_path, temp_config_file = _write_docker_files(
            str(temp_project_dir),
            mock_config,
            required_files,
            analysis,
            has_wheel=False,
            needs_git=False,
            classifier_name="clf-a",
        )

        # Temp single-classifier config written for the build
        assert temp_config_file == temp_project_dir / ".mlserver.clf-a.yaml"
        assert temp_config_file.exists()

        # required_files swapped to the temp config
        assert "mlserver.yaml" not in required_files
        assert ".mlserver.clf-a.yaml" in required_files

        # Dockerfile renames the temp config to mlserver.yaml inside the image
        dockerfile_content = dockerfile_path.read_text()
        assert "COPY .mlserver.clf-a.yaml ./mlserver.yaml" in dockerfile_content


def _mock_build_process(returncode=0, lines=None):
    """Create a mock subprocess.Popen process for docker build."""
    if lines is None:
        lines = ["Step 1/5 : FROM python:3.11-slim\n", "Successfully built abc123\n"]
    process = MagicMock()
    process.stdout.readline.side_effect = lines + [""]
    process.returncode = returncode
    return process


def _docker_popen_router(mock_process):
    """Popen side_effect that mocks docker commands only.

    Patching subprocess.Popen patches the shared subprocess module, which
    would also break the real git subprocess calls used for label/metadata
    generation - so git (and anything else) passes through to the real Popen
    while docker gets the mock.
    """
    real_popen = subprocess.Popen

    def router(cmd, *args, **kwargs):
        if cmd and cmd[0] == "docker":
            if isinstance(mock_process, Exception):
                raise mock_process
            return mock_process
        return real_popen(cmd, *args, **kwargs)

    return router


class TestDockerOperations:
    """Test Docker operations against the current API (rewritten 2026-07-03).

    All docker interactions are mocked (subprocess.Popen for builds,
    subprocess.run for push/rmi/images) - no docker daemon required.
    The old check_docker_availability duplicates were deleted: they are
    covered by TestCheckDockerAvailability below.
    """

    def test_build_container_success(self, temp_project_dir):
        """Full build path: config load, file detection, Docker files, tags, build cmd."""
        repo_name = temp_project_dir.name.lower()
        mock_process = _mock_build_process(returncode=0)

        with (
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container._find_mlserver_source", return_value=None),
            patch(
                "mlserver.container.subprocess.Popen",
                side_effect=_docker_popen_router(mock_process),
            ) as mock_popen,
        ):
            result = build_container(
                project_path=str(temp_project_dir),
                config_file=None,
                tag_prefix="ml-models",
                registry="registry.example.com",
            )

        assert result["success"] is True, result.get("error")

        # Tags: registry/prefix applied to <repo>:latest and <repo>:v<version>
        assert f"registry.example.com/ml-models/{repo_name}:latest" in result["tags"]
        assert f"registry.example.com/ml-models/{repo_name}:v1.0.0" in result["tags"]

        # Docker files were written to the project
        assert Path(result["dockerfile"]).exists()
        assert Path(result["dockerignore"]).exists()

        # File detection ran for real
        assert "predictor.py" in result["required_files"]
        assert "model.pkl" in result["required_files"]

        # Build command: docker build with -t for every tag
        # (the router also records passed-through git Popen calls - filter them)
        docker_calls = [c for c in mock_popen.call_args_list if c[0][0][0] == "docker"]
        assert len(docker_calls) == 1
        build_cmd = docker_calls[0][0][0]
        assert build_cmd[:5] == ["docker", "build", ".", "-f", "Dockerfile"]
        for tag in result["tags"]:
            assert tag in build_cmd
        assert docker_calls[0].kwargs["cwd"] == str(temp_project_dir)
        # BuildKit is no longer force-disabled (RFC 0001, D15): no custom env
        # is injected, the daemon default applies
        assert "env" not in docker_calls[0].kwargs

    def test_build_container_docker_missing_fails_gracefully(self, temp_project_dir):
        """Missing docker binary yields success=False, not an exception.

        build_container no longer pre-checks docker availability; the
        FileNotFoundError from launching docker is caught and reported.
        """
        router = _docker_popen_router(FileNotFoundError("No such file or directory: 'docker'"))
        with (
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container._find_mlserver_source", return_value=None),
            patch("mlserver.container.subprocess.Popen", side_effect=router),
        ):
            result = build_container(project_path=str(temp_project_dir), config_file=None)

        assert result["success"] is False
        assert "docker" in result["error"]

    def test_build_container_build_failure_reports_exit_code(self, temp_project_dir):
        """A non-zero docker build exit code is reported in the error message."""
        mock_process = _mock_build_process(returncode=1, lines=["error: build failed\n"])

        with (
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container._find_mlserver_source", return_value=None),
            patch(
                "mlserver.container.subprocess.Popen",
                side_effect=_docker_popen_router(mock_process),
            ),
        ):
            result = build_container(project_path=str(temp_project_dir), config_file=None)

        assert result["success"] is False
        assert "exit code 1" in result["error"]

    def test_list_images_filters_by_repository(self, temp_project_dir):
        """Only images belonging to the project repository are listed."""
        mock_output = (
            "test-repo:latest|abc123|2024-01-01|100MB\n"
            "test-repo/sentiment:v1.0.0|def456|2024-01-02|90MB\n"
            "other-repo:latest|zzz999|2024-01-03|80MB\n"
        )

        with (
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)

            result = list_images(str(temp_project_dir))

        tags = [img["tag"] for img in result]
        assert tags == ["test-repo:latest", "test-repo/sentiment:v1.0.0"]
        assert result[0]["image_id"] == "abc123"
        assert result[0]["created"] == "2024-01-01"
        assert result[0]["size"] == "100MB"

    def test_list_images_classifier_name_filter(self, temp_project_dir):
        """classifier_name narrows the listing to <repo>/<classifier> images."""
        mock_output = (
            "test-repo:latest|abc123|2024-01-01|100MB\n"
            "test-repo/sentiment:v1.0.0|def456|2024-01-02|90MB\n"
            "test-repo/churn:v2.0.0|ghi789|2024-01-03|95MB\n"
        )

        with (
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)

            result = list_images(str(temp_project_dir), classifier_name="sentiment")

        assert [img["tag"] for img in result] == ["test-repo/sentiment:v1.0.0"]

    def test_remove_images_multi_tag_command_construction(self, temp_project_dir):
        """Each image gets its own `docker rmi <image_id>` call (no -f by default)."""
        images = [
            {"tag": "test-repo:latest", "image_id": "abc123"},
            {"tag": "test-repo:v1.0.0", "image_id": "def456"},
            {"tag": "test-repo:v1.0.0-abcdef0", "image_id": "ghi789"},
        ]

        with (
            patch("mlserver.container.list_images", return_value=images),
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)

            result = remove_images(str(temp_project_dir), force=False)

        assert result["success"] is True
        assert result["removed_images"] == [img["tag"] for img in images]
        assert result["errors"] == []

        assert mock_run.call_count == 3
        for call, image in zip(mock_run.call_args_list, images):
            assert call[0][0] == ["docker", "rmi", image["image_id"]]

    def test_push_container_tag_construction_with_prefix_registry_and_version(
        self, temp_project_dir
    ):
        """Pushed tags are <registry>/<prefix>/<tag>; version param overrides metadata."""
        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.generate_container_tags") as mock_tags,
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_tags.return_value = ["test-repo:latest", "test-repo:v2.0.0"]
            mock_run.return_value = MagicMock(returncode=0)

            result = push_container(
                str(temp_project_dir),
                registry="registry.example.com",
                tag_prefix="ml-models",
                version="2.0.0",
            )

        # The validated version overrides the metadata version before tagging
        assert mock_meta.return_value.classifier.version == "2.0.0"

        assert result["success"] is True
        assert result["pushed_tags"] == [
            "registry.example.com/ml-models/test-repo:latest",
            "registry.example.com/ml-models/test-repo:v2.0.0",
        ]
        assert result["failed_tags"] == []
        assert result["registry"] == "registry.example.com"

        # One docker push per final tag
        pushed_cmds = [call[0][0] for call in mock_run.call_args_list]
        assert pushed_cmds == [
            ["docker", "push", "registry.example.com/ml-models/test-repo:latest"],
            ["docker", "push", "registry.example.com/ml-models/test-repo:v2.0.0"],
        ]

    def test_push_container_passes_classifier_name_to_tag_generation(self, temp_project_dir):
        """classifier_name flows into tag generation for single-classifier configs."""
        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.generate_container_tags") as mock_tags,
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_tags.return_value = ["test-repo/sentiment:latest"]
            mock_run.return_value = MagicMock(returncode=0)

            result = push_container(
                str(temp_project_dir), registry="registry.example.com", classifier_name="sentiment"
            )

        assert result["success"] is True
        assert mock_tags.call_args.kwargs["classifier_name"] == "sentiment"

    def test_push_container_multi_classifier_uses_extracted_config(self, temp_project_dir):
        """Multi-classifier configs extract the named classifier before pushing."""
        multi_config = MagicMock()
        extracted_config = MagicMock()

        with (
            patch("mlserver.multi_classifier.detect_multi_classifier_config", return_value=True),
            patch(
                "mlserver.multi_classifier.load_multi_classifier_config", return_value=multi_config
            ),
            patch(
                "mlserver.multi_classifier.extract_single_classifier_config",
                return_value=extracted_config,
            ) as mock_extract,
            patch("mlserver.container._prepare_container_metadata") as mock_prepare,
            patch("mlserver.container.get_git_info", return_value=None),
            patch(
                "mlserver.container.generate_container_tags",
                return_value=["test-repo/clf-a:latest"],
            ),
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_prepare.return_value = MagicMock()
            mock_run.return_value = MagicMock(returncode=0)

            result = push_container(
                str(temp_project_dir), registry="registry.example.com", classifier_name="clf-a"
            )

        assert result["success"] is True
        mock_extract.assert_called_once_with(multi_config, "clf-a")
        mock_prepare.assert_called_once_with(extracted_config, str(temp_project_dir))

    def test_push_container_partial_failure_is_not_success(self, temp_project_dir):
        """If any tag fails to push, success is False (callers exit nonzero)."""
        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.generate_container_tags") as mock_tags,
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_tags.return_value = ["test-repo:latest", "test-repo:v1.0.0"]
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1, stderr="denied: access forbidden"),
            ]

            result = push_container(str(temp_project_dir), registry="registry.example.com")

        assert result["success"] is False
        assert len(result["pushed_tags"]) == 1
        assert len(result["failed_tags"]) == 1
        assert "denied" in result["failed_tags"][0]


class TestGenerateDockerignoreUpdated:
    """Test generate_dockerignore function with current API."""

    def test_generate_dockerignore_basic(self):
        """Test basic dockerignore generation."""
        auto_excludes = {"__pycache__", "*.pyc", ".git"}

        result = generate_dockerignore(".", auto_excludes)

        assert isinstance(result, str)
        assert "__pycache__/" in result
        assert ".git/" in result
        assert "*.py[cod]" in result

    def test_generate_dockerignore_with_additional_excludes(self):
        """Test dockerignore with additional excludes."""
        auto_excludes = {"__pycache__"}
        additional = ["test_data/", "docs/", "*.log"]

        result = generate_dockerignore(".", auto_excludes, additional)

        assert "test_data/" in result
        assert "docs/" in result
        assert "*.log" in result

    def test_generate_dockerignore_empty_excludes(self):
        """Test dockerignore with empty excludes."""
        result = generate_dockerignore(".", set())

        assert isinstance(result, str)
        assert "Auto-generated" in result

    def test_generate_dockerignore_includes_timestamps(self):
        """Test dockerignore includes timestamp."""
        result = generate_dockerignore(".", {"*.pyc"})

        assert "Generated at:" in result

    def test_generate_dockerignore_ide_files(self):
        """Test dockerignore includes IDE patterns."""
        result = generate_dockerignore(".", set())

        assert ".vscode/" in result
        assert ".idea/" in result

    def test_generate_dockerignore_os_files(self):
        """Test dockerignore includes OS file patterns."""
        result = generate_dockerignore(".", set())

        assert ".DS_Store" in result
        assert "Thumbs.db" in result


class TestGenerateDockerfileUpdated:
    """Test generate_dockerfile function with current API."""

    def test_generate_dockerfile_basic(self, temp_project_dir, mock_config):
        """Test basic Dockerfile generation."""
        required_files = ["predictor.py", "model.pkl", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert isinstance(result, str)
        assert "FROM python:" in result
        assert "WORKDIR /app" in result
        assert "CMD [" in result

    def test_generate_dockerfile_with_requirements(self, temp_project_dir, mock_config):
        """Test Dockerfile with requirements.txt."""
        required_files = ["predictor.py", "requirements.txt", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert "COPY requirements.txt" in result
        assert "pip install" in result

    def test_generate_dockerfile_with_wheel(self, temp_project_dir, mock_config):
        """Test Dockerfile with wheel file (dev build - releases pin instead)."""
        required_files = ["predictor.py", "mlserver.yaml"]

        with patch("mlserver.container._get_installed_mlserver_version", return_value="0.5.1.dev1"):
            result = generate_dockerfile(
                str(temp_project_dir), mock_config, required_files, has_wheel=True
            )

        # The generated Dockerfile installs the framework wheel (renamed merve).
        assert "merve*.whl" in result

    def test_generate_dockerfile_with_git(self, temp_project_dir, mock_config):
        """Test Dockerfile with git requirement."""
        required_files = ["predictor.py", "mlserver.yaml"]

        result = generate_dockerfile(
            str(temp_project_dir), mock_config, required_files, needs_git=True
        )

        assert "git" in result.lower()

    def test_generate_dockerfile_custom_base_image(self, temp_project_dir, mock_config):
        """Test Dockerfile with custom base image."""
        from mlserver.config import ApiConfig

        config = AppConfig(
            predictor=PredictorConfig(module="predictor", class_name="TestPredictor"),
            classifier={"name": "test", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )
        required_files = ["predictor.py"]

        result = generate_dockerfile(
            str(temp_project_dir), config, required_files, base_image="python:3.12-slim"
        )

        assert "FROM python:3.12-slim" in result

    def test_generate_dockerfile_includes_healthcheck(self, temp_project_dir, mock_config):
        """Test Dockerfile includes health check."""
        required_files = ["predictor.py", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert "HEALTHCHECK" in result

    def test_generate_dockerfile_non_root_user(self, temp_project_dir, mock_config):
        """Test Dockerfile creates non-root user."""
        required_files = ["predictor.py", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert "useradd" in result
        assert "USER mlserver" in result

    def test_generate_dockerfile_sets_pythonpath(self, temp_project_dir, mock_config):
        """Test Dockerfile sets PYTHONPATH."""
        required_files = ["predictor.py", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert "PYTHONPATH" in result

    def test_generate_dockerfile_exposes_port(self, temp_project_dir, mock_config):
        """Test Dockerfile exposes port 8000."""
        required_files = ["predictor.py", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert "EXPOSE 8000" in result

    def test_generate_dockerfile_with_subdirectory_files(self, temp_project_dir, mock_config):
        """Test Dockerfile with files in subdirectories."""
        # Create subdirectory structure
        models_dir = temp_project_dir / "models"
        models_dir.mkdir()
        (models_dir / "model.pkl").write_bytes(b"model data")

        required_files = ["predictor.py", "models/model.pkl", "mlserver.yaml"]

        result = generate_dockerfile(str(temp_project_dir), mock_config, required_files)

        assert "COPY models/" in result or "models" in result


class TestLooksLikeFilePath:
    """Test _looks_like_file_path function."""

    def test_model_path_key(self):
        """Test model_path key with path value."""
        assert _looks_like_file_path("model_path", "./model.pkl")
        assert _looks_like_file_path("model_path", "models/classifier.pkl")

    def test_file_key(self):
        """Test file-related keys."""
        assert _looks_like_file_path("config_file", "config.yaml")
        assert _looks_like_file_path("data_file", "data/train.csv")

    def test_path_key(self):
        """Test path-related keys."""
        assert _looks_like_file_path("preprocessor_path", "./preproc.pkl")
        assert _looks_like_file_path("artifact_path", "/artifacts/model.bin")

    def test_directory_keys(self):
        """Test directory-related keys."""
        assert _looks_like_file_path("data_dir", "./data/")
        assert _looks_like_file_path("folder_path", "models/v1/")

    def test_non_path_values(self):
        """Test values that are not file paths."""
        assert not _looks_like_file_path("port", "8000")
        assert not _looks_like_file_path("name", "classifier")
        assert not _looks_like_file_path("debug", "true")

    def test_non_path_keys(self):
        """Test keys that don't suggest paths."""
        assert not _looks_like_file_path("version", "1.0.0")
        assert not _looks_like_file_path("host", "0.0.0.0")
        assert not _looks_like_file_path("workers", "4")

    def test_relative_paths(self):
        """Test relative path detection."""
        assert _looks_like_file_path("model_path", "./model.pkl")
        assert _looks_like_file_path("data_path", "../data/train.csv")

    def test_absolute_paths(self):
        """Test absolute path detection."""
        assert _looks_like_file_path("model_path", "/app/models/model.pkl")


class TestAnalyzePythonImports:
    """Test _analyze_python_imports function."""

    def test_analyze_simple_imports(self, temp_project_dir):
        """Test analyzing simple imports."""
        python_file = temp_project_dir / "test_imports.py"
        python_file.write_text("""
import os
import sys
from pathlib import Path
""")

        result = _analyze_python_imports(str(temp_project_dir), "test_imports.py")

        assert isinstance(result, set)

    def test_analyze_local_imports(self, temp_project_dir):
        """Test analyzing local module imports."""
        # Create a local module
        (temp_project_dir / "local_helper.py").write_text("def helper(): pass")

        python_file = temp_project_dir / "main.py"
        python_file.write_text("""
import local_helper
from local_helper import helper
""")

        result = _analyze_python_imports(str(temp_project_dir), "main.py")

        assert isinstance(result, set)
        # Should find the local module
        assert "local_helper.py" in result

    def test_analyze_missing_file(self, temp_project_dir):
        """Test analyzing non-existent file."""
        result = _analyze_python_imports(str(temp_project_dir), "nonexistent.py")

        assert isinstance(result, set)
        assert len(result) == 0

    def test_analyze_syntax_error_file(self, temp_project_dir):
        """Test analyzing file with syntax errors."""
        python_file = temp_project_dir / "broken.py"
        python_file.write_text("""
import os
def broken(
    # Missing closing paren
""")

        result = _analyze_python_imports(str(temp_project_dir), "broken.py")

        # Should handle errors gracefully
        assert isinstance(result, set)


class TestResolveLocalImport:
    """Test _resolve_local_import function."""

    def test_resolve_existing_module(self, temp_project_dir):
        """Test resolving existing local module."""
        (temp_project_dir / "local_module.py").write_text("# local module")

        result = _resolve_local_import(str(temp_project_dir), "local_module")

        assert result == "local_module.py"

    def test_resolve_existing_package(self, temp_project_dir):
        """Test resolving existing package."""
        pkg_dir = temp_project_dir / "my_package"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("# package init")

        result = _resolve_local_import(str(temp_project_dir), "my_package")

        assert result == "my_package/__init__.py"

    def test_resolve_submodule(self, temp_project_dir):
        """Test resolving submodule."""
        pkg_dir = temp_project_dir / "utils"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "helpers.py").write_text("# helpers")

        result = _resolve_local_import(str(temp_project_dir), "utils.helpers")

        assert result == "utils/helpers.py"

    def test_resolve_nonexistent_module(self, temp_project_dir):
        """Test resolving non-existent module."""
        result = _resolve_local_import(str(temp_project_dir), "nonexistent")

        assert result is None

    def test_resolve_stdlib_module(self, temp_project_dir):
        """Test stdlib modules return None (not local)."""
        result = _resolve_local_import(str(temp_project_dir), "os")

        assert result is None


class TestCheckDockerAvailability:
    """Test check_docker_availability function."""

    def test_docker_available(self):
        """Test when Docker is available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = check_docker_availability()

            assert result is True

    def test_docker_not_available(self):
        """Test when Docker is not available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            result = check_docker_availability()

            assert result is False

    def test_docker_not_installed(self):
        """Test when Docker is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = check_docker_availability()

            assert result is False

    def test_docker_subprocess_error(self):
        """Test when subprocess raises error."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "docker")

            result = check_docker_availability()

            assert result is False


class TestDetectRequiredFilesExtended:
    """Extended tests for detect_required_files function."""

    def test_detect_with_subdirectory_artifacts(self, temp_project_dir):
        """Test detection of artifacts in subdirectories."""
        from mlserver.config import ApiConfig

        # Create subdirectory structure
        models_dir = temp_project_dir / "models"
        models_dir.mkdir()
        (models_dir / "classifier.pkl").write_bytes(b"model")

        config = AppConfig(
            predictor=PredictorConfig(
                module="predictor",
                class_name="TestPredictor",
                init_kwargs={"model_path": "models/classifier.pkl"},
            ),
            classifier={"name": "test", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )

        result = detect_required_files(str(temp_project_dir), config)

        assert "models/classifier.pkl" in result["required_files"]

    def test_detect_with_json_config(self, temp_project_dir):
        """Test detection of JSON config files."""
        from mlserver.config import ApiConfig

        (temp_project_dir / "features.json").write_text('["a", "b"]')

        config = AppConfig(
            predictor=PredictorConfig(
                module="predictor",
                class_name="TestPredictor",
                init_kwargs={"feature_order_path": "./features.json"},
            ),
            classifier={"name": "test", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )

        result = detect_required_files(str(temp_project_dir), config)

        assert "features.json" in result["required_files"]

    def test_detect_with_feature_order_file(self, temp_project_dir):
        """Test detection when feature_order is a file path."""
        from mlserver.config import ApiConfig

        (temp_project_dir / "feature_order.json").write_text('["f1", "f2"]')

        config = AppConfig(
            predictor=PredictorConfig(module="predictor", class_name="TestPredictor"),
            classifier={"name": "test", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto", feature_order="./feature_order.json"),
        )

        result = detect_required_files(str(temp_project_dir), config)

        assert "feature_order.json" in result["required_files"]

    def test_detect_analysis_categories(self, temp_project_dir, mock_config):
        """Test analysis includes correct categories."""
        result = detect_required_files(str(temp_project_dir), mock_config)

        analysis = result["analysis"]
        assert "predictor_files" in analysis
        assert "artifact_files" in analysis
        assert "dependency_files" in analysis
        assert "config_files" in analysis
        assert "auto_excludes" in analysis

    def test_detect_auto_excludes(self, temp_project_dir, mock_config):
        """Test auto_excludes contains expected patterns."""
        result = detect_required_files(str(temp_project_dir), mock_config)

        auto_excludes = result["analysis"]["auto_excludes"]
        assert "__pycache__" in auto_excludes
        assert ".git" in auto_excludes
        assert "Dockerfile" in auto_excludes

    def test_detect_includes_local_import_dependencies(self, temp_project_dir):
        """Local modules imported by the predictor are picked up as dependencies."""
        from mlserver.config import ApiConfig

        (temp_project_dir / "feature_utils.py").write_text("def transform(x): return x")
        (temp_project_dir / "predictor.py").write_text("""
import feature_utils

class TestPredictor:
    def predict(self, X):
        return [feature_utils.transform(x) for x in X]
""")

        config = AppConfig(
            predictor=PredictorConfig(module="predictor", class_name="TestPredictor"),
            classifier={"name": "test", "version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )

        result = detect_required_files(str(temp_project_dir), config)

        assert "feature_utils.py" in result["required_files"]
        assert "feature_utils.py" in result["analysis"]["dependency_files"]


class TestAddFileOrDirectory:
    """Test _add_file_or_directory function."""

    def test_add_file(self, temp_project_dir):
        """Test adding a file."""
        from mlserver.container import _add_file_or_directory

        (temp_project_dir / "test.txt").write_text("test")
        file_set = set()

        _add_file_or_directory("test.txt", str(temp_project_dir), file_set)

        assert "test.txt" in file_set

    def test_add_relative_file(self, temp_project_dir):
        """Test adding file with ./ prefix."""
        from mlserver.container import _add_file_or_directory

        (temp_project_dir / "test.txt").write_text("test")
        file_set = set()

        _add_file_or_directory("./test.txt", str(temp_project_dir), file_set)

        assert "test.txt" in file_set

    def test_add_directory(self, temp_project_dir):
        """Test adding a directory."""
        from mlserver.container import _add_file_or_directory

        dir_path = temp_project_dir / "subdir"
        dir_path.mkdir()
        (dir_path / "file1.py").write_text("# file1")
        (dir_path / "file2.py").write_text("# file2")

        file_set = set()

        _add_file_or_directory("subdir", str(temp_project_dir), file_set)

        assert "subdir/file1.py" in file_set
        assert "subdir/file2.py" in file_set

    def test_skip_parent_paths(self, temp_project_dir):
        """Test skipping paths outside project."""
        from mlserver.container import _add_file_or_directory

        file_set = set()

        _add_file_or_directory("../outside.txt", str(temp_project_dir), file_set)

        assert len(file_set) == 0

    def test_skip_nonexistent_file(self, temp_project_dir):
        """Test skipping non-existent files."""
        from mlserver.container import _add_file_or_directory

        file_set = set()

        _add_file_or_directory("nonexistent.txt", str(temp_project_dir), file_set)

        assert len(file_set) == 0

    def test_skip_empty_path(self, temp_project_dir):
        """Test skipping empty paths."""
        from mlserver.container import _add_file_or_directory

        file_set = set()

        _add_file_or_directory("", str(temp_project_dir), file_set)

        assert len(file_set) == 0


class TestGetMlserverGitUrl:
    """Test _get_mlserver_git_url function."""

    def test_env_var_override(self):
        """Test environment variable override."""
        from mlserver.container import _get_mlserver_git_url

        test_url = "git+https://github.com/test/repo.git@main"
        with patch.dict(os.environ, {"MLSERVER_GIT_URL": test_url}):
            result = _get_mlserver_git_url()

        assert result == test_url

    def test_returns_string_or_none(self):
        """Test function returns string or None."""
        from mlserver.container import _get_mlserver_git_url

        with patch.dict(os.environ, {}, clear=False):
            result = _get_mlserver_git_url()

        assert result is None or isinstance(result, str)

    def test_direct_url_git_install(self, tmp_path):
        """A pip git install is detected via direct_url.json (url form)."""
        from mlserver.container import _get_mlserver_git_url

        (tmp_path / "direct_url.json").write_text(
            json.dumps({"url": "git+https://github.com/test/mlserver.git@abc123"})
        )
        mock_dist = MagicMock()
        mock_dist._path = tmp_path

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = _get_mlserver_git_url()

        assert result == "git+https://github.com/test/mlserver.git@abc123"

    def test_direct_url_vcs_info(self, tmp_path):
        """A pip git install is detected via direct_url.json (vcs_info form)."""
        from mlserver.container import _get_mlserver_git_url

        (tmp_path / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "https://github.com/test/mlserver.git",
                    "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                }
            )
        )
        mock_dist = MagicMock()
        mock_dist._path = tmp_path

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = _get_mlserver_git_url()

        assert result == "git+https://github.com/test/mlserver.git@abc123"


class TestFindGitSourceDirectory:
    """Test _find_git_source_directory function."""

    def test_find_returns_none_or_string(self):
        """Test function returns None or string path."""
        from mlserver.container import _find_git_source_directory

        result = _find_git_source_directory()

        assert result is None or isinstance(result, str)

    def test_find_with_mock_distribution(self):
        """Test with mocked distribution exception."""
        from mlserver.container import _find_git_source_directory

        # Mock the distribution function that's called internally
        with patch("importlib.metadata.distribution") as mock_dist:
            mock_dist.side_effect = Exception("Not found")

            result = _find_git_source_directory()

        assert result is None or isinstance(result, str)


class TestContainerLabels:
    """Test generate_container_labels function."""

    def test_generate_container_labels_basic(self, temp_project_dir, mock_config):
        """Test basic container label generation."""
        from mlserver.container import generate_container_labels

        with (
            patch("mlserver.container.get_mlserver_commit_hash", return_value="abc123"),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
        ):
            labels = generate_container_labels(
                str(temp_project_dir), classifier_name="test-classifier", config=mock_config
            )

        assert isinstance(labels, dict)
        assert "com.mlserver.commit" in labels
        assert labels["com.mlserver.commit"] == "abc123"

    def test_generate_container_labels_with_classifier_name(self, temp_project_dir, mock_config):
        """Test labels include classifier name."""
        from mlserver.container import generate_container_labels

        with (
            patch("mlserver.container.get_mlserver_commit_hash", return_value="abc123"),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.GitVersionManager") as mock_gvm,
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
        ):
            mock_gvm.return_value.get_current_version.return_value = "1.0.0"

            labels = generate_container_labels(
                str(temp_project_dir), classifier_name="my-classifier", config=mock_config
            )

        assert "com.classifier.name" in labels
        assert labels["com.classifier.name"] == "my-classifier"

    def test_generate_container_labels_oci_labels(self, temp_project_dir, mock_config):
        """Test OCI standard labels are included."""
        from mlserver.container import generate_container_labels

        with (
            patch("mlserver.container.get_mlserver_commit_hash", return_value="abc123"),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.GitVersionManager") as mock_gvm,
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
        ):
            mock_gvm.return_value.get_current_version.return_value = "1.0.0"

            labels = generate_container_labels(
                str(temp_project_dir), classifier_name="test", config=mock_config
            )

        # Check for OCI standard labels
        assert "org.opencontainers.image.created" in labels
        assert "org.opencontainers.image.title" in labels

    def test_generate_container_labels_predictor_info(self, temp_project_dir, mock_config):
        """Test labels include predictor information."""
        from mlserver.container import generate_container_labels

        with (
            patch("mlserver.container.get_mlserver_commit_hash", return_value=None),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
        ):
            labels = generate_container_labels(str(temp_project_dir), config=mock_config)

        assert "com.classifier.predictor.class" in labels
        assert labels["com.classifier.predictor.class"] == "TestPredictor"
        assert "com.classifier.predictor.module" in labels


class TestOCIAndMerveLabels:
    """OCI + dev.merve.* label set (RFC 0001, D5 / W1.4)."""

    def _labels(self, project_dir, config, git_url=None, git_info=None):
        """Generate labels with fully mocked git/tool lookups."""
        from mlserver.container import generate_container_labels

        def fake_run(cmd, *args, **kwargs):
            if cmd[:4] == ["git", "config", "--get", "remote.origin.url"]:
                if git_url:
                    return MagicMock(returncode=0, stdout=git_url + "\n")
                return MagicMock(returncode=1, stdout="")
            return MagicMock(returncode=1, stdout="")

        with (
            patch("mlserver.container.get_mlserver_commit_hash", return_value="abc1234"),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container.get_git_info", return_value=git_info),
            patch("mlserver.container.GitVersionManager") as mock_gvm,
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
            patch("mlserver.container.subprocess.run", side_effect=fake_run),
        ):
            mock_gvm.return_value.get_current_version.return_value = "1.2.3"

            return generate_container_labels(
                str(project_dir), classifier_name="my-classifier", config=config
            )

    def test_oci_label_set_complete(self, temp_project_dir, mock_config):
        """source/revision/version/created/title are all emitted."""
        from mlserver.version import GitInfo

        git_info = GitInfo(tag=None, commit="deadbee1", branch="main", is_dirty=False)
        labels = self._labels(
            temp_project_dir,
            mock_config,
            git_url="https://github.com/enmacc/test-repo.git",
            git_info=git_info,
        )

        assert labels["org.opencontainers.image.source"] == ("https://github.com/enmacc/test-repo")
        assert labels["org.opencontainers.image.revision"] == "deadbee1"
        assert labels["org.opencontainers.image.version"] == "1.2.3"
        assert "org.opencontainers.image.created" in labels
        # Title is the plain classifier name (D5)
        assert labels["org.opencontainers.image.title"] == "my-classifier"

    def test_oci_source_normalizes_ssh_remote(self, temp_project_dir, mock_config):
        """git@ SSH remotes are normalized to https for the source label."""
        labels = self._labels(
            temp_project_dir,
            mock_config,
            git_url="git@github.com:enmacc/test-repo.git",
        )

        assert labels["org.opencontainers.image.source"] == ("https://github.com/enmacc/test-repo")
        # the raw remote stays available on the legacy label
        assert labels["com.classifier.git_url"] == "git@github.com:enmacc/test-repo.git"

    def test_dev_merve_labels(self, temp_project_dir, mock_config):
        """dev.merve.{classifier,mlserver_version,mlserver_commit} are emitted."""
        import mlserver as mlserver_module

        labels = self._labels(temp_project_dir, mock_config)

        assert labels["dev.merve.classifier"] == "my-classifier"
        assert labels["dev.merve.mlserver_commit"] == "abc1234"
        assert labels["dev.merve.mlserver_version"] == mlserver_module.__version__

    def test_legacy_labels_kept_this_release(self, temp_project_dir, mock_config):
        """All pre-D5 labels survive for dashboard continuity."""
        labels = self._labels(
            temp_project_dir,
            mock_config,
            git_url="https://github.com/enmacc/test-repo.git",
        )

        for legacy_key in (
            "com.mlserver.commit",
            "com.mlserver.version",
            "com.classifier.name",
            "com.classifier.version",
            "com.classifier.repository",
            "com.classifier.git_url",
            "com.classifier.predictor.class",
            "com.classifier.predictor.module",
        ):
            assert legacy_key in labels, f"legacy label {legacy_key} was dropped"

    def test_version_falls_back_to_config_without_git_tags(self, temp_project_dir, mock_config):
        """Without git tags the (display-only) config version fills the label."""
        from mlserver.container import generate_container_labels

        with (
            patch("mlserver.container.get_mlserver_commit_hash", return_value=None),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.GitVersionManager") as mock_gvm,
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
        ):
            mock_gvm.return_value.get_current_version.return_value = None

            labels = generate_container_labels(
                str(temp_project_dir), classifier_name="my-classifier", config=mock_config
            )

        # mock_config carries classifier version 1.0.0
        assert labels["com.classifier.version"] == "1.0.0"
        assert labels["org.opencontainers.image.version"] == "1.0.0"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("git@github.com:org/repo.git", "https://github.com/org/repo"),
            ("git@github.com:org/repo", "https://github.com/org/repo"),
            ("ssh://git@github.com/org/repo.git", "https://github.com/org/repo"),
            ("https://github.com/org/repo.git", "https://github.com/org/repo"),
            ("https://github.com/org/repo", "https://github.com/org/repo"),
            ("http://gitlab.local/org/repo.git", "https://gitlab.local/org/repo"),
            ("git+https://github.com/org/repo.git", "https://github.com/org/repo"),
        ],
    )
    def test_normalize_git_url_to_https(self, raw, expected):
        """Remote URL shapes normalize to a browsable https URL."""
        from mlserver.container import _normalize_git_url_to_https

        assert _normalize_git_url_to_https(raw) == expected


class TestListImages:
    """Test list_images function."""

    def test_list_images_success(self, temp_project_dir):
        """Test successful image listing."""
        mock_output = (
            "test-repo:latest|abc123|2024-01-01|100MB\ntest-repo:v1.0.0|def456|2024-01-02|100MB"
        )

        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
            patch("subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_meta.return_value.classifier = MagicMock()
            mock_meta.return_value.classifier.repository = "test-repo"

            mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)

            result = list_images(str(temp_project_dir))

        assert isinstance(result, list)
        assert len(result) == 2

    def test_list_images_empty(self, temp_project_dir):
        """Test listing when no images exist."""
        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("mlserver.container.get_repository_name", return_value="test-repo"),
            patch("subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_meta.return_value.classifier = MagicMock()
            mock_meta.return_value.classifier.repository = "test-repo"

            mock_run.return_value = MagicMock(returncode=0, stdout="")

            result = list_images(str(temp_project_dir))

        assert isinstance(result, list)
        assert len(result) == 0

    def test_list_images_docker_error(self, temp_project_dir):
        """Test handling of Docker errors."""
        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_meta.return_value.classifier = MagicMock()
            mock_meta.return_value.classifier.repository = "test-repo"

            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

            result = list_images(str(temp_project_dir))

        assert result == []


class TestRemoveImages:
    """Test remove_images function."""

    def test_remove_images_success(self, temp_project_dir):
        """Test successful image removal."""
        with (
            patch("mlserver.container.list_images") as mock_list,
            patch("subprocess.run") as mock_run,
        ):
            mock_list.return_value = [
                {"tag": "test-repo:latest", "image_id": "abc123"},
                {"tag": "test-repo:v1.0.0", "image_id": "def456"},
            ]
            mock_run.return_value = MagicMock(returncode=0)

            result = remove_images(str(temp_project_dir))

        assert result["success"] is True
        assert len(result["removed_images"]) == 2

    def test_remove_images_no_images(self, temp_project_dir):
        """Test removal when no images exist."""
        with patch("mlserver.container.list_images") as mock_list:
            mock_list.return_value = []

            result = remove_images(str(temp_project_dir))

        assert result["success"] is True
        assert "No images found" in result["message"]

    def test_remove_images_force(self, temp_project_dir):
        """Test forced image removal."""
        with (
            patch("mlserver.container.list_images") as mock_list,
            patch("subprocess.run") as mock_run,
        ):
            mock_list.return_value = [{"tag": "test-repo:latest", "image_id": "abc123"}]
            mock_run.return_value = MagicMock(returncode=0)

            remove_images(str(temp_project_dir), force=True)

        # Check -f flag was used
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-f" in call_args

    def test_remove_images_partial_failure(self, temp_project_dir):
        """Test partial failure during removal."""
        with (
            patch("mlserver.container.list_images") as mock_list,
            patch("subprocess.run") as mock_run,
        ):
            mock_list.return_value = [
                {"tag": "test-repo:latest", "image_id": "abc123"},
                {"tag": "test-repo:v1.0.0", "image_id": "def456"},
            ]

            # First succeeds, second fails
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1, stderr="Image in use"),
            ]

            result = remove_images(str(temp_project_dir))

        assert result["success"] is True  # At least one removed
        assert len(result["removed_images"]) == 1
        assert len(result["errors"]) == 1


class TestPushContainer:
    """Test push_container function."""

    def test_push_container_no_registry(self, temp_project_dir):
        """Test push fails without registry."""
        result = push_container(str(temp_project_dir), registry=None)

        assert result["success"] is False
        assert "Registry URL is required" in result.get("error", "")

    def test_push_container_success(self, temp_project_dir):
        """Test successful container push."""
        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.generate_container_tags") as mock_tags,
            patch("subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_tags.return_value = ["test:latest"]
            mock_run.return_value = MagicMock(returncode=0)

            result = push_container(str(temp_project_dir), registry="registry.example.com")

        assert result["success"] is True
        assert len(result["pushed_tags"]) > 0

    def test_push_container_failure(self, temp_project_dir):
        """Test container push failure."""
        with (
            patch("mlserver.container.load_classifier_metadata") as mock_meta,
            patch("mlserver.container.get_git_info", return_value=None),
            patch("mlserver.container.generate_container_tags") as mock_tags,
            patch("subprocess.run") as mock_run,
        ):
            mock_meta.return_value = MagicMock()
            mock_tags.return_value = ["test:latest"]
            mock_run.return_value = MagicMock(returncode=1, stderr="Push failed")

            result = push_container(str(temp_project_dir), registry="registry.example.com")

        assert len(result["failed_tags"]) > 0


class TestHandleWheelPreparation:
    """Test _handle_wheel_preparation function."""

    def test_existing_wheel(self, temp_project_dir):
        """Test when wheel already exists."""
        from mlserver.container import _handle_wheel_preparation

        # Create a fake wheel file
        (temp_project_dir / "merve-1.0.0-py3-none-any.whl").write_bytes(b"wheel")

        wheel_file, needs_git = _handle_wheel_preparation(str(temp_project_dir))

        assert wheel_file is None  # None because we use existing
        assert needs_git is False

    def test_no_wheel_no_git(self, temp_project_dir):
        """Test when no wheel and no git URL."""
        from mlserver.container import _handle_wheel_preparation

        with patch("mlserver.container._find_mlserver_source", return_value=None):
            wheel_file, needs_git = _handle_wheel_preparation(str(temp_project_dir))

        assert needs_git is False

    def test_with_git_url_no_source(self, temp_project_dir):
        """Test with git URL but no local source."""
        from mlserver.container import _handle_wheel_preparation

        with patch("mlserver.container._find_git_source_directory", return_value=None):
            wheel_file, needs_git = _handle_wheel_preparation(
                str(temp_project_dir), git_url="git+https://github.com/test/repo.git@main"
            )

        assert needs_git is True

    def test_git_url_with_local_source_builds_wheel(self, tmp_path):
        """Git install with a local source builds a wheel; no git needed in image."""
        from mlserver.container import _handle_wheel_preparation

        with (
            patch("mlserver.container._find_git_source_directory", return_value="/fake/git/source"),
            patch(
                "mlserver.container._build_mlserver_wheel",
                return_value="merve-1.0.0-py3-none-any.whl",
            ) as mock_build,
        ):
            wheel_file, needs_git = _handle_wheel_preparation(
                str(tmp_path), git_url="git+https://github.com/test/repo.git@main"
            )

        assert wheel_file == "merve-1.0.0-py3-none-any.whl"
        assert needs_git is False
        mock_build.assert_called_once_with("/fake/git/source", str(tmp_path))

    def test_git_url_wheel_build_failure_falls_back_to_git_install(self, tmp_path):
        """Git install where wheel build fails installs git in the Dockerfile."""
        from mlserver.container import _handle_wheel_preparation

        with (
            patch("mlserver.container._find_git_source_directory", return_value="/fake/git/source"),
            patch("mlserver.container._build_mlserver_wheel", return_value=None),
        ):
            wheel_file, needs_git = _handle_wheel_preparation(
                str(tmp_path), git_url="git+https://github.com/test/repo.git@main"
            )

        assert wheel_file is None
        assert needs_git is True

    def test_explicit_source_path_builds_wheel(self, tmp_path):
        """An explicit mlserver_source_path is used for the wheel build."""
        from mlserver.container import _handle_wheel_preparation

        with patch(
            "mlserver.container._build_mlserver_wheel", return_value="merve-1.0.0-py3-none-any.whl"
        ) as mock_build:
            wheel_file, needs_git = _handle_wheel_preparation(
                str(tmp_path), mlserver_source_path="/explicit/source"
            )

        assert wheel_file == "merve-1.0.0-py3-none-any.whl"
        assert needs_git is False
        mock_build.assert_called_once_with("/explicit/source", str(tmp_path))

    def test_auto_discovered_source_builds_wheel(self, tmp_path):
        """A source found by _find_mlserver_source is used for the wheel build."""
        from mlserver.container import _handle_wheel_preparation

        with (
            patch("mlserver.container._find_mlserver_source", return_value="/discovered/source"),
            patch(
                "mlserver.container._build_mlserver_wheel",
                return_value="merve-1.0.0-py3-none-any.whl",
            ) as mock_build,
        ):
            wheel_file, needs_git = _handle_wheel_preparation(str(tmp_path))

        assert wheel_file == "merve-1.0.0-py3-none-any.whl"
        assert needs_git is False
        mock_build.assert_called_once_with("/discovered/source", str(tmp_path))


class TestGenerateLabelDirectives:
    """Test _generate_label_directives function."""

    def test_generate_label_directives(self, temp_project_dir, mock_config):
        """Test label directive generation."""
        from mlserver.container import _generate_label_directives

        with patch("mlserver.container.generate_container_labels") as mock_labels:
            mock_labels.return_value = {
                "com.mlserver.version": "1.0.0",
                "com.classifier.name": "test",
            }

            result = _generate_label_directives(
                str(temp_project_dir), classifier_name="test", config=mock_config
            )

        assert 'LABEL com.classifier.name="test"' in result
        assert 'LABEL com.mlserver.version="1.0.0"' in result

    def test_generate_label_directives_escapes_quotes(self, temp_project_dir, mock_config):
        """Test label directives escape quotes in values."""
        from mlserver.container import _generate_label_directives

        with patch("mlserver.container.generate_container_labels") as mock_labels:
            mock_labels.return_value = {"description": 'A "quoted" value'}

            result = _generate_label_directives(
                str(temp_project_dir), classifier_name="test", config=mock_config
            )

        assert '\\"quoted\\"' in result


class TestLoadContainerConfig:
    """Test _load_container_config function."""

    def test_load_single_classifier_config(self, temp_project_dir):
        """Test loading single classifier config."""
        import os

        from mlserver.container import _load_container_config

        # Need to change to project dir for detect_config_file to work
        original_dir = os.getcwd()
        try:
            os.chdir(str(temp_project_dir))
            result = _load_container_config(str(temp_project_dir))

            assert isinstance(result, AppConfig)
            assert result.predictor.class_name == "TestPredictor"
        finally:
            os.chdir(original_dir)

    def test_load_config_with_explicit_file(self, temp_project_dir):
        """Test loading with explicit config file."""
        from mlserver.container import _load_container_config

        result = _load_container_config(str(temp_project_dir), config_file="mlserver.yaml")

        assert isinstance(result, AppConfig)

    def test_load_multi_classifier_named(self, temp_project_dir):
        """A named classifier is extracted from a multi-classifier config."""
        from mlserver.container import _load_container_config

        multi_config = MagicMock()
        multi_config.classifiers = {"clf-a": object(), "clf-b": object()}

        with (
            patch("mlserver.multi_classifier.detect_multi_classifier_config", return_value=True),
            patch(
                "mlserver.multi_classifier.load_multi_classifier_config", return_value=multi_config
            ),
            patch(
                "mlserver.multi_classifier.extract_single_classifier_config",
                side_effect=lambda cfg, name: f"config-for-{name}",
            ),
        ):
            result = _load_container_config(str(temp_project_dir), classifier_name="clf-a")

        assert result == "config-for-clf-a"

    def test_load_multi_classifier_default(self, temp_project_dir):
        """Without classifier_name the default classifier is used."""
        from mlserver.container import _load_container_config

        multi_config = MagicMock()
        multi_config.classifiers = {"clf-a": object(), "clf-b": object()}
        multi_config.default_classifier = "clf-b"

        with (
            patch("mlserver.multi_classifier.detect_multi_classifier_config", return_value=True),
            patch(
                "mlserver.multi_classifier.load_multi_classifier_config", return_value=multi_config
            ),
            patch(
                "mlserver.multi_classifier.extract_single_classifier_config",
                side_effect=lambda cfg, name: f"config-for-{name}",
            ),
        ):
            result = _load_container_config(str(temp_project_dir))

        assert result == "config-for-clf-b"

    def test_load_multi_classifier_first_when_no_default(self, temp_project_dir):
        """Without a default, the first classifier in the config is used."""
        from mlserver.container import _load_container_config

        multi_config = MagicMock()
        multi_config.classifiers = {"clf-a": object(), "clf-b": object()}
        multi_config.default_classifier = None

        with (
            patch("mlserver.multi_classifier.detect_multi_classifier_config", return_value=True),
            patch(
                "mlserver.multi_classifier.load_multi_classifier_config", return_value=multi_config
            ),
            patch(
                "mlserver.multi_classifier.extract_single_classifier_config",
                side_effect=lambda cfg, name: f"config-for-{name}",
            ),
        ):
            result = _load_container_config(str(temp_project_dir))

        assert result == "config-for-clf-a"


class TestBuildContainerWheelHandling:
    """Wheel lifecycle inside build_container (added 2026-07-03).

    The build must clean up only wheels it created itself; wheels the user
    placed in the project are preserved.
    """

    def test_build_created_wheel_cleaned_up_after_build(self, temp_project_dir):
        """A wheel built by the build process is deleted after the build."""
        wheel_name = "merve-0.1.0-py3-none-any.whl"

        def fake_build_wheel(source, project):
            (Path(project) / wheel_name).write_bytes(b"wheel")
            return wheel_name

        mock_process = _mock_build_process(returncode=0)

        with (
            # Dev version pin: releases skip the wheel path entirely
            patch(
                "mlserver.container._get_installed_mlserver_version",
                return_value="0.5.1.dev1",
            ),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container._find_mlserver_source", return_value="/fake/source"),
            patch("mlserver.container._build_mlserver_wheel", side_effect=fake_build_wheel),
            patch(
                "mlserver.container.subprocess.Popen",
                side_effect=_docker_popen_router(mock_process),
            ),
        ):
            result = build_container(project_path=str(temp_project_dir))

        assert result["success"] is True, result.get("error")
        # The build-created wheel was cleaned up afterwards
        assert not (temp_project_dir / wheel_name).exists()
        # But the Dockerfile was generated for a wheel-based install
        assert "merve*.whl" in Path(result["dockerfile"]).read_text()

    def test_preexisting_user_wheel_preserved(self, temp_project_dir):
        """A wheel the user placed in the project survives the build."""
        wheel_name = "merve-9.9.9-py3-none-any.whl"
        user_wheel = temp_project_dir / wheel_name
        user_wheel.write_bytes(b"user wheel")

        mock_process = _mock_build_process(returncode=0)

        with (
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch(
                "mlserver.container.subprocess.Popen",
                side_effect=_docker_popen_router(mock_process),
            ),
        ):
            result = build_container(project_path=str(temp_project_dir))

        assert result["success"] is True, result.get("error")
        # User-provided wheel is left untouched
        assert user_wheel.exists()
        assert user_wheel.read_bytes() == b"user wheel"


class TestPrepareDockerBuildCommand:
    """Test _prepare_docker_build_command (added 2026-07-03)."""

    def test_basic_command_with_tags(self):
        from mlserver.container import _prepare_docker_build_command

        cmd = _prepare_docker_build_command(["repo:latest", "repo:v1.0.0"])

        assert cmd == [
            "docker",
            "build",
            ".",
            "-f",
            "Dockerfile",
            "-t",
            "repo:latest",
            "-t",
            "repo:v1.0.0",
        ]

    def test_build_args_and_no_cache(self):
        from mlserver.container import _prepare_docker_build_command

        cmd = _prepare_docker_build_command(
            ["repo:latest"], build_args={"HTTP_PROXY": "http://proxy:3128"}, no_cache=True
        )

        assert cmd == [
            "docker",
            "build",
            ".",
            "-f",
            "Dockerfile",
            "-t",
            "repo:latest",
            "--build-arg",
            "HTTP_PROXY=http://proxy:3128",
            "--no-cache",
        ]

    def test_platform_flag(self):
        """--platform is forwarded to docker build (RFC 0001, D15)."""
        from mlserver.container import _prepare_docker_build_command

        cmd = _prepare_docker_build_command(["repo:latest"], platform="linux/amd64")

        assert cmd == [
            "docker",
            "build",
            ".",
            "-f",
            "Dockerfile",
            "-t",
            "repo:latest",
            "--platform",
            "linux/amd64",
        ]

    def test_no_platform_flag_by_default(self):
        """Without a platform argument no --platform flag is emitted."""
        from mlserver.container import _prepare_docker_build_command

        cmd = _prepare_docker_build_command(["repo:latest"])

        assert "--platform" not in cmd


class TestPrepareContainerMetadata:
    """Test _prepare_container_metadata (added 2026-07-03)."""

    def test_metadata_from_config_dict(self, temp_project_dir):
        """Classifier dict from config is validated into ClassifierMetadata."""
        from mlserver.config import ApiConfig
        from mlserver.container import _prepare_container_metadata

        config = AppConfig(
            predictor=PredictorConfig(module="predictor", class_name="TestPredictor"),
            classifier={
                "name": "test-classifier",
                "version": "1.0.0",
                "repository": "test-repo",
                "description": "Test classifier",
            },
            model={"version": "1.0.0"},
            api=ApiConfig(version="v1", adapter="auto"),
        )

        metadata = _prepare_container_metadata(config, str(temp_project_dir))

        assert metadata.classifier.name == "test-classifier"
        assert metadata.classifier.version == "1.0.0"
        assert metadata.classifier.repository == "test-repo"
        assert metadata.model.version == "1.0.0"

    @pytest.mark.parametrize(
        "env_tag,expected_version",
        [
            # Hierarchical tag created by `mlserver tag`
            ("sentiment-v2.3.4-mlserver-abc1234", "2.3.4"),
            # Non-hierarchical tag with a -vX.Y.Z segment (regex fallback)
            ("release-v9.9.9", "9.9.9"),
            # Plain semver tag
            ("3.2.1", "3.2.1"),
        ],
    )
    def test_metadata_version_from_git_tag(self, temp_project_dir, env_tag, expected_version):
        """When config omits the version, it is extracted from the git tag."""
        from mlserver.config import ApiConfig
        from mlserver.container import _prepare_container_metadata

        config = AppConfig(
            predictor=PredictorConfig(module="predictor", class_name="TestPredictor"),
            classifier={"name": "sentiment", "repository": "test-repo"},
            api=ApiConfig(version="v1", adapter="auto"),
        )
        # auto_detect.get_git_info reads these env vars first (container-embedded metadata)
        with patch.dict(
            os.environ, {"MLSERVER_GIT_TAG": env_tag, "MLSERVER_GIT_COMMIT": "abc1234"}
        ):
            metadata = _prepare_container_metadata(config, str(temp_project_dir))

        assert metadata.classifier.version == expected_version
        assert metadata.model.version == expected_version

    def test_metadata_missing_classifier_raises(self, temp_project_dir):
        """A config without any classifier section raises ConfigurationError.

        AppConfig auto-generates classifier metadata from the predictor class
        name (even via model_construct, through model_post_init), so a bare
        mock is used to reach the guard branch.
        """
        from mlserver.container import _prepare_container_metadata
        from mlserver.errors import ConfigurationError

        config = MagicMock()
        config.classifier = None

        with pytest.raises(ConfigurationError, match="missing required 'classifier' section"):
            _prepare_container_metadata(config, str(temp_project_dir))


class TestGenerateContainerTagsInternal:
    """Test _generate_container_tags (added 2026-07-03)."""

    def test_tags_use_git_version_for_classifier(self, temp_project_dir, mock_config):
        """The classifier's git-tag version overrides the metadata version."""
        from mlserver.container import _generate_container_tags

        repo_name = temp_project_dir.name.lower()
        metadata = MagicMock()
        metadata.classifier.version = "1.0.0"

        with patch("mlserver.version_control.GitVersionManager") as mock_gvm:
            mock_gvm.return_value.get_current_version.return_value = "2.1.0"

            tags = _generate_container_tags(
                metadata,
                mock_config,
                str(temp_project_dir),
                tag_prefix="ml-models",
                registry="registry.example.com",
                classifier_name="clf-a",
            )

        assert metadata.classifier.version == "2.1.0"
        assert f"registry.example.com/ml-models/{repo_name}/clf-a:latest" in tags
        assert f"registry.example.com/ml-models/{repo_name}/clf-a:v2.1.0" in tags

    def test_tags_missing_git_tag_uses_untagged_placeholder(self, temp_project_dir, mock_config):
        """No git tag for the classifier yields explicit 'untagged' tags."""
        from mlserver.container import _generate_container_tags

        repo_name = temp_project_dir.name.lower()
        metadata = MagicMock()
        metadata.classifier.version = "1.0.0"

        with patch("mlserver.version_control.GitVersionManager") as mock_gvm:
            mock_gvm.return_value.get_current_version.return_value = None

            tags = _generate_container_tags(
                metadata, mock_config, str(temp_project_dir), classifier_name="clf-b"
            )

        assert metadata.classifier.version == "missing-git-tag"
        assert f"{repo_name}/clf-b:latest" in tags
        assert f"{repo_name}/clf-b:untagged" in tags


def _write_multi_classifier_config(project_dir):
    """Write a two-classifier config sharing one predictor.py in project_dir."""
    (project_dir / "mlserver.yaml").write_text("""
classifiers:
  sentiment:
    predictor:
      module: predictor
      class_name: TestPredictor
    classifier:
      name: sentiment
      version: "1.0.0"
  intent:
    predictor:
      module: predictor
      class_name: TestPredictor
    classifier:
      name: intent
      version: "2.0.0"
default_classifier: sentiment
""")


class TestBuildCommitImage:
    """Build-once commit image for multi-classifier repos (RFC 0001 D4 / W2.5).

    The DEFAULT `mlserver build` for a multi-classifier repo builds ONE commit
    image per git commit: no baked classifier, plain `merve serve` CMD, full
    multi-classifier config shipped, tagged <repo>:<sha> + <repo>:latest.
    """

    def _git_info(self, commit="abcdef12"):
        from mlserver.version import GitInfo

        return GitInfo(tag=None, commit=commit, branch="main", is_dirty=False)

    def test_multi_classifier_build_produces_commit_image(self, temp_project_dir):
        _write_multi_classifier_config(temp_project_dir)
        repo_name = temp_project_dir.name.lower()
        mock_process = _mock_build_process(returncode=0)

        with (
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container._find_mlserver_source", return_value=None),
            patch("mlserver.container.get_git_info", return_value=self._git_info()),
            patch(
                "mlserver.container.subprocess.Popen",
                side_effect=_docker_popen_router(mock_process),
            ) as mock_popen,
        ):
            result = build_container(project_path=str(temp_project_dir), config_file=None)

        assert result["success"] is True, result.get("error")
        assert result["commit_image"] is True
        assert sorted(result["classifiers"]) == ["intent", "sentiment"]

        # Commit-image tags: <repo>:<short-sha> + <repo>:latest
        assert f"{repo_name}:abcdef1" in result["tags"]
        assert f"{repo_name}:latest" in result["tags"]

        # Dockerfile: NO baked classifier, plain serve CMD, repo-level header
        dockerfile = Path(result["dockerfile"]).read_text()
        assert "ENV MLSERVER_CLASSIFIER" not in dockerfile
        assert 'CMD ["merve", "serve", "mlserver.yaml"]' in dockerfile
        assert "commit image, all classifiers" in dockerfile

        # Exactly one docker build call, carrying both commit-image tags
        docker_calls = [c for c in mock_popen.call_args_list if c[0][0][0] == "docker"]
        assert len(docker_calls) == 1
        build_cmd = docker_calls[0][0][0]
        assert build_cmd[:5] == ["docker", "build", ".", "-f", "Dockerfile"]
        for tag in result["tags"]:
            assert tag in build_cmd

    def test_commit_image_two_stage_structure_preserved(self, temp_project_dir):
        """The commit image keeps the two-stage build (no regression of W1.9)."""
        _write_multi_classifier_config(temp_project_dir)
        mock_process = _mock_build_process(returncode=0)

        with (
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container._find_mlserver_source", return_value=None),
            patch("mlserver.container.get_git_info", return_value=self._git_info()),
            patch(
                "mlserver.container.subprocess.Popen",
                side_effect=_docker_popen_router(mock_process),
            ),
        ):
            result = build_container(project_path=str(temp_project_dir), config_file=None)

        dockerfile = Path(result["dockerfile"]).read_text()
        from_lines = [ln for ln in dockerfile.splitlines() if ln.startswith("FROM ")]
        assert len(from_lines) == 2
        assert from_lines[0].endswith(" AS builder")
        builder, runtime = dockerfile.split("# ============================ Stage 2")
        assert "build-essential" in builder
        assert "build-essential" not in runtime
        assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile

    def test_commit_image_tags_helper(self, temp_project_dir):
        from mlserver.container import _commit_image_tags

        with (
            patch("mlserver.container.get_repository_name", return_value="myrepo"),
            patch("mlserver.container.get_git_info", return_value=self._git_info("deadbeef")),
        ):
            tags = _commit_image_tags(str(temp_project_dir))

        assert tags == ["myrepo:deadbee", "myrepo:latest"]

    def test_commit_image_tags_without_git_only_latest(self, temp_project_dir):
        from mlserver.container import _commit_image_tags

        with (
            patch("mlserver.container.get_repository_name", return_value="myrepo"),
            patch("mlserver.container.get_git_info", return_value=None),
        ):
            tags = _commit_image_tags(str(temp_project_dir))

        assert tags == ["myrepo:latest"]


class TestPerClassifierImage:
    """--per-classifier-image escape hatch keeps the pre-W2.5 behavior (baked ENV)."""

    def _git_info(self):
        from mlserver.version import GitInfo

        return GitInfo(tag=None, commit="abcdef12", branch="main", is_dirty=False)

    def test_per_classifier_image_bakes_env_and_is_not_commit_image(self, temp_project_dir):
        _write_multi_classifier_config(temp_project_dir)
        mock_process = _mock_build_process(returncode=0)

        with (
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
            patch("mlserver.container._find_mlserver_source", return_value=None),
            patch("mlserver.container.get_git_info", return_value=self._git_info()),
            patch(
                "mlserver.container.subprocess.Popen",
                side_effect=_docker_popen_router(mock_process),
            ),
        ):
            result = build_container(
                project_path=str(temp_project_dir),
                config_file=None,
                classifier_name="sentiment",
                per_classifier_image=True,
            )

        assert result["success"] is True, result.get("error")
        # Legacy per-classifier path - not the build-once commit image
        assert result.get("commit_image") is not True

        dockerfile = Path(result["dockerfile"]).read_text()
        assert 'ENV MLSERVER_CLASSIFIER="sentiment"' in dockerfile
        # single-classifier config shipped (renamed to mlserver.yaml in-image)
        assert "COPY .mlserver.sentiment.yaml ./mlserver.yaml" in dockerfile


class TestReleasePinnedInstall:
    """W2.6 (RFC 0001 D16): release -> pinned install; dev -> wheel + WARNING."""

    def test_is_release_version_helper(self):
        from mlserver.container import _is_release_version

        assert _is_release_version("0.5.0") is True
        assert _is_release_version("1.2.3") is True
        assert _is_release_version("0.5.0.dev19") is False
        assert _is_release_version("0.5.0+local.abc") is False
        assert _is_release_version("0.5.0.post1") is False
        # Pre-releases cannot be pinned as reproducible releases either
        assert _is_release_version("0.5.0rc1") is False
        assert _is_release_version("0.5.0a1") is False
        assert _is_release_version("0.5.0b2") is False
        assert _is_release_version("") is False
        assert _is_release_version(None) is False

    def test_release_pins_git_tag_when_installed_from_git(self, temp_project_dir, mock_config):
        # merve is not on PyPI (RFC 0001 §8 A8): a release running from a git
        # install pins the SAME git source at the immutable release tag.
        with (
            patch("mlserver.container._get_installed_mlserver_version", return_value="0.5.0"),
            patch(
                "mlserver.container._get_mlserver_git_url",
                return_value="git+https://github.com/acme/merve.git@main",
            ),
        ):
            result = generate_dockerfile(str(temp_project_dir), mock_config, ["predictor.py"])

        # Branch ref stripped, release tag pinned
        assert (
            'RUN pip install --no-cache-dir "git+https://github.com/acme/merve.git@v0.5.0"'
            in result
        )
        assert "merve==" not in result
        assert "COPY merve*.whl" not in result
        # The builder stage must carry git for the clone (builder deps render
        # as "git \" + newline continuation before build-essential)
        assert "git \\" in result

    def test_release_version_pins_framework(self, temp_project_dir, mock_config):
        # Index installs (no git source detected) pin the index version.
        with (
            patch("mlserver.container._get_installed_mlserver_version", return_value="0.5.0"),
            patch("mlserver.container._get_mlserver_git_url", return_value=None),
        ):
            result = generate_dockerfile(str(temp_project_dir), mock_config, ["predictor.py"])

        assert 'RUN pip install --no-cache-dir "merve==0.5.0"' in result
        # No wheel copy path for a clean release
        assert "COPY merve*.whl" not in result

    def test_dev_version_uses_wheel_path_and_warns(self, temp_project_dir, mock_config, capsys):
        with patch(
            "mlserver.container._get_installed_mlserver_version", return_value="0.5.0.dev19"
        ):
            result = generate_dockerfile(
                str(temp_project_dir), mock_config, ["predictor.py"], has_wheel=True
            )

        # Dev build falls back to the wheel-copy path, not the pinned install
        assert "COPY merve*.whl" in result
        assert 'pip install --no-cache-dir "merve==' not in result

        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "not reproducible" in out.lower()


class TestPushClassifierAlias:
    """W2.5: registry tag aliases on the already-built commit image (no rebuild)."""

    def _git_info(self, commit="abcdef12"):
        from mlserver.version import GitInfo

        return GitInfo(tag=None, commit=commit, branch="main", is_dirty=False)

    def test_alias_tag_and_push_sequence(self, temp_project_dir):
        from mlserver.container import push_classifier_alias

        with (
            patch("mlserver.container.get_repository_name", return_value="myrepo"),
            patch("mlserver.container.get_git_info", return_value=self._git_info()),
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = push_classifier_alias(
                str(temp_project_dir),
                registry="reg.example.com",
                classifier_name="sentiment",
                version="1.2.0",
            )

        assert result["success"] is True
        # Source is the commit image at HEAD's short sha - never rebuilt
        assert result["source_image"] == "myrepo:abcdef1"
        assert result["pushed_tags"] == [
            "reg.example.com/myrepo:sentiment-v1.2.0",
            "reg.example.com/myrepo:sentiment-latest",
        ]

        # Exact command sequence: tag -> push -> tag -> push, and NO docker build
        cmds = [c[0][0] for c in mock_run.call_args_list]
        assert cmds == [
            ["docker", "tag", "myrepo:abcdef1", "reg.example.com/myrepo:sentiment-v1.2.0"],
            ["docker", "push", "reg.example.com/myrepo:sentiment-v1.2.0"],
            ["docker", "tag", "myrepo:abcdef1", "reg.example.com/myrepo:sentiment-latest"],
            ["docker", "push", "reg.example.com/myrepo:sentiment-latest"],
        ]
        assert not any(c[:2] == ["docker", "build"] for c in cmds)

    def test_alias_applies_tag_prefix(self, temp_project_dir):
        from mlserver.container import push_classifier_alias

        with (
            patch("mlserver.container.get_repository_name", return_value="myrepo"),
            patch("mlserver.container.get_git_info", return_value=self._git_info()),
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = push_classifier_alias(
                str(temp_project_dir),
                registry="reg.example.com",
                classifier_name="intent",
                version="3.1.0",
                tag_prefix="team-ml",
            )

        assert result["pushed_tags"] == [
            "reg.example.com/team-ml/myrepo:intent-v3.1.0",
            "reg.example.com/team-ml/myrepo:intent-latest",
        ]

    def test_alias_tag_failure_reported_not_success(self, temp_project_dir):
        from mlserver.container import push_classifier_alias

        with (
            patch("mlserver.container.get_repository_name", return_value="myrepo"),
            patch("mlserver.container.get_git_info", return_value=self._git_info()),
            patch("mlserver.container.subprocess.run") as mock_run,
        ):
            # docker tag fails for the very first alias
            mock_run.return_value = MagicMock(returncode=1, stderr="no such image")

            result = push_classifier_alias(
                str(temp_project_dir),
                registry="reg.example.com",
                classifier_name="sentiment",
                version="1.0.0",
            )

        assert result["success"] is False
        assert result["pushed_tags"] == []
        assert len(result["failed_tags"]) >= 1
        # A failed tag short-circuits before push (tag, tag) - never pushes
        cmds = [c[0][0] for c in mock_run.call_args_list]
        assert all(c[1] == "tag" for c in cmds)

    def test_alias_no_registry_errors(self, temp_project_dir):
        from mlserver.container import push_classifier_alias

        result = push_classifier_alias(
            str(temp_project_dir), registry="", classifier_name="x", version="1.0.0"
        )
        assert result["success"] is False
        assert "Registry URL is required" in result["error"]
