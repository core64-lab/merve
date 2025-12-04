"""
Diagnostic utilities for MLServer environment and configuration.

Provides checks for:
- System requirements (Python, Docker, Git)
- Project configuration
- Dependencies
- Common issues

Used by `mlserver doctor` and `mlserver validate` commands.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import subprocess
import sys
import os
import shutil


class CheckStatus(Enum):
    """Status of a diagnostic check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    status: CheckStatus
    message: Optional[str] = None
    suggestion: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASSED

    @property
    def warning(self) -> bool:
        return self.status == CheckStatus.WARNING

    @property
    def failed(self) -> bool:
        return self.status == CheckStatus.FAILED


@dataclass
class DiagnosticReport:
    """Collection of diagnostic check results."""
    checks: List[CheckResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(c.failed for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.warning for c in self.checks)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def add_recommendation(self, rec: str) -> None:
        self.recommendations.append(rec)


# =============================================================================
# System Checks
# =============================================================================

def check_python_version(verbose: bool = False) -> CheckResult:
    """Check Python version is supported."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 9:
        return CheckResult(
            name="Python version",
            status=CheckStatus.PASSED,
            message=f"Python {version_str} (supported: 3.9+)",
            details={"version": version_str}
        )
    elif version.major == 3 and version.minor == 8:
        return CheckResult(
            name="Python version",
            status=CheckStatus.WARNING,
            message=f"Python {version_str} (3.8 support deprecated, upgrade to 3.9+)",
            suggestion="Upgrade to Python 3.9 or later for best compatibility"
        )
    else:
        return CheckResult(
            name="Python version",
            status=CheckStatus.FAILED,
            message=f"Python {version_str} not supported",
            suggestion="Install Python 3.9 or later"
        )


def check_docker(verbose: bool = False) -> CheckResult:
    """Check if Docker is installed and running."""
    docker_path = shutil.which("docker")

    if not docker_path:
        return CheckResult(
            name="Docker",
            status=CheckStatus.WARNING,
            message="Docker not found in PATH",
            suggestion="Install Docker for container builds: https://docs.docker.com/get-docker/"
        )

    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return CheckResult(
                name="Docker",
                status=CheckStatus.PASSED,
                message=f"Docker {version}",
                details={"version": version, "path": docker_path}
            )
        else:
            return CheckResult(
                name="Docker",
                status=CheckStatus.WARNING,
                message="Docker installed but daemon not running",
                suggestion="Start Docker daemon: 'open -a Docker' (macOS) or 'sudo systemctl start docker' (Linux)"
            )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Docker",
            status=CheckStatus.WARNING,
            message="Docker command timed out",
            suggestion="Docker may be starting up. Try again in a moment."
        )
    except Exception as e:
        return CheckResult(
            name="Docker",
            status=CheckStatus.WARNING,
            message=f"Error checking Docker: {e}",
            suggestion="Ensure Docker is properly installed"
        )


def check_git(verbose: bool = False) -> CheckResult:
    """Check if Git is installed."""
    git_path = shutil.which("git")

    if not git_path:
        return CheckResult(
            name="Git",
            status=CheckStatus.FAILED,
            message="Git not found in PATH",
            suggestion="Install Git: https://git-scm.com/downloads"
        )

    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().replace("git version ", "")
            return CheckResult(
                name="Git",
                status=CheckStatus.PASSED,
                message=f"Git {version}",
                details={"version": version, "path": git_path}
            )
    except Exception as e:
        return CheckResult(
            name="Git",
            status=CheckStatus.FAILED,
            message=f"Error checking Git: {e}"
        )

    return CheckResult(
        name="Git",
        status=CheckStatus.FAILED,
        message="Git check failed"
    )


# =============================================================================
# Project Checks
# =============================================================================

def check_config_file(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if mlserver.yaml exists and is valid YAML."""
    config_file = Path(project_path) / "mlserver.yaml"

    if not config_file.exists():
        return CheckResult(
            name="Configuration file",
            status=CheckStatus.FAILED,
            message="mlserver.yaml not found",
            suggestion="Run 'mlserver init' to create a configuration file"
        )

    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            return CheckResult(
                name="Configuration file",
                status=CheckStatus.FAILED,
                message="mlserver.yaml is empty",
                suggestion="Add configuration to mlserver.yaml or run 'mlserver init'"
            )

        return CheckResult(
            name="Configuration file",
            status=CheckStatus.PASSED,
            message="mlserver.yaml found and valid YAML",
            details={"path": str(config_file), "keys": list(config.keys()) if config else []}
        )
    except yaml.YAMLError as e:
        return CheckResult(
            name="Configuration file",
            status=CheckStatus.FAILED,
            message=f"Invalid YAML syntax: {e}",
            suggestion="Check for syntax errors (indentation, colons, quotes)"
        )
    except Exception as e:
        return CheckResult(
            name="Configuration file",
            status=CheckStatus.FAILED,
            message=f"Error reading config: {e}"
        )


def check_config_schema(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if mlserver.yaml has valid schema."""
    config_file = Path(project_path) / "mlserver.yaml"

    if not config_file.exists():
        return CheckResult(
            name="Configuration schema",
            status=CheckStatus.SKIPPED,
            message="No config file to validate"
        )

    try:
        import yaml
        with open(config_file, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Detect config format and validate accordingly
        is_multi_classifier = "classifiers" in raw_config

        if is_multi_classifier:
            from .multi_classifier import MultiClassifierConfig
            config = MultiClassifierConfig.model_validate(raw_config)
            classifier_count = len(config.classifiers)
            return CheckResult(
                name="Configuration schema",
                status=CheckStatus.PASSED,
                message=f"Multi-classifier config valid ({classifier_count} classifiers)",
                details={"format": "multi-classifier", "classifier_count": classifier_count}
            )
        else:
            from .config import AppConfig
            config = AppConfig.model_validate(raw_config)
            return CheckResult(
                name="Configuration schema",
                status=CheckStatus.PASSED,
                message="Configuration schema valid",
                details={
                    "format": "single-classifier",
                    "predictor_module": config.predictor.module,
                    "predictor_class": config.predictor.class_name
                }
            )
    except Exception as e:
        error_msg = str(e)
        # Extract the most relevant part of Pydantic errors
        if "validation error" in error_msg.lower():
            # Simplify Pydantic error messages
            lines = error_msg.split('\n')
            simplified = lines[0] if lines else error_msg
            return CheckResult(
                name="Configuration schema",
                status=CheckStatus.FAILED,
                message=f"Validation error: {simplified}",
                suggestion="Check mlserver.yaml against the configuration schema"
            )
        return CheckResult(
            name="Configuration schema",
            status=CheckStatus.FAILED,
            message=f"Schema validation failed: {error_msg[:100]}...",
            suggestion="Review your mlserver.yaml configuration"
        )


def check_predictor_import(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if predictor module can be imported."""
    config_file = Path(project_path) / "mlserver.yaml"

    if not config_file.exists():
        return CheckResult(
            name="Predictor import",
            status=CheckStatus.SKIPPED,
            message="No config file to check predictor"
        )

    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Handle both single and multi-classifier formats
        is_multi_classifier = "classifiers" in config

        if is_multi_classifier:
            # Multi-classifier: check the first classifier's predictor
            classifiers = config.get("classifiers", {})
            if not classifiers:
                return CheckResult(
                    name="Predictor import",
                    status=CheckStatus.FAILED,
                    message="No classifiers defined in multi-classifier config",
                    suggestion="Add at least one classifier to classifiers section"
                )
            # Get the first classifier
            first_classifier_name = list(classifiers.keys())[0]
            first_classifier = classifiers[first_classifier_name]
            predictor_config = first_classifier.get("predictor", {})
            module_name = predictor_config.get("module")
            class_name = predictor_config.get("class_name")
        else:
            # Single-classifier format
            predictor_config = config.get("predictor", {})
            module_name = predictor_config.get("module")
            class_name = predictor_config.get("class_name")

        if not module_name or not class_name:
            return CheckResult(
                name="Predictor import",
                status=CheckStatus.FAILED,
                message="predictor.module or predictor.class_name not configured",
                suggestion="Add predictor configuration to mlserver.yaml"
            )

        # Try to resolve and import the module
        from .predictor_loader import resolve_module_path
        import importlib

        resolved_module = resolve_module_path(module_name, project_path)

        # Add project path to sys.path temporarily
        original_path = sys.path.copy()
        if project_path not in sys.path:
            sys.path.insert(0, str(Path(project_path).resolve()))

        try:
            mod = importlib.import_module(resolved_module)
            cls = getattr(mod, class_name)

            # Check for predict method
            if hasattr(cls, 'predict'):
                return CheckResult(
                    name="Predictor import",
                    status=CheckStatus.PASSED,
                    message=f"{class_name} from {module_name} importable",
                    details={"module": resolved_module, "class": class_name, "has_predict": True}
                )
            else:
                return CheckResult(
                    name="Predictor import",
                    status=CheckStatus.WARNING,
                    message=f"{class_name} imported but has no predict() method",
                    suggestion="Ensure your predictor class has a predict(self, X) method"
                )
        finally:
            sys.path = original_path

    except ImportError as e:
        return CheckResult(
            name="Predictor import",
            status=CheckStatus.FAILED,
            message=f"Cannot import predictor module: {e}",
            suggestion="Check module path and ensure all dependencies are installed"
        )
    except AttributeError as e:
        return CheckResult(
            name="Predictor import",
            status=CheckStatus.FAILED,
            message=f"Class not found in module: {e}",
            suggestion="Check that class_name matches the class defined in your predictor file"
        )
    except Exception as e:
        return CheckResult(
            name="Predictor import",
            status=CheckStatus.FAILED,
            message=f"Error checking predictor: {e}"
        )


def check_model_files(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if model files referenced in config exist."""
    config_file = Path(project_path) / "mlserver.yaml"

    if not config_file.exists():
        return CheckResult(
            name="Model files",
            status=CheckStatus.SKIPPED,
            message="No config file to check"
        )

    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Handle both single and multi-classifier formats
        is_multi_classifier = "classifiers" in config

        if is_multi_classifier:
            classifiers = config.get("classifiers", {})
            if classifiers:
                first_classifier_name = list(classifiers.keys())[0]
                first_classifier = classifiers[first_classifier_name]
                init_kwargs = first_classifier.get("predictor", {}).get("init_kwargs", {})
            else:
                init_kwargs = {}
        else:
            init_kwargs = config.get("predictor", {}).get("init_kwargs", {})

        if not init_kwargs:
            return CheckResult(
                name="Model files",
                status=CheckStatus.PASSED,
                message="No model files configured (using defaults)",
                details={"checked_files": []}
            )

        # Check common model file parameters
        file_params = ['model_path', 'preprocessor_path', 'weights_path', 'checkpoint_path']
        checked_files = []
        missing_files = []

        base_path = Path(project_path)
        for param in file_params:
            if param in init_kwargs:
                file_path = base_path / init_kwargs[param]
                checked_files.append(str(file_path))
                if not file_path.exists():
                    missing_files.append(init_kwargs[param])

        if missing_files:
            return CheckResult(
                name="Model files",
                status=CheckStatus.FAILED,
                message=f"Missing model files: {', '.join(missing_files)}",
                suggestion="Ensure model files exist at the specified paths"
            )

        if checked_files:
            return CheckResult(
                name="Model files",
                status=CheckStatus.PASSED,
                message=f"All {len(checked_files)} model files exist",
                details={"checked_files": checked_files}
            )

        return CheckResult(
            name="Model files",
            status=CheckStatus.PASSED,
            message="No model file paths to check"
        )

    except Exception as e:
        return CheckResult(
            name="Model files",
            status=CheckStatus.FAILED,
            message=f"Error checking model files: {e}"
        )


def check_feature_order_file(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if feature_order file exists if configured."""
    config_file = Path(project_path) / "mlserver.yaml"

    if not config_file.exists():
        return CheckResult(
            name="Feature order file",
            status=CheckStatus.SKIPPED,
            message="No config file to check"
        )

    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Handle both single and multi-classifier formats
        is_multi_classifier = "classifiers" in config

        if is_multi_classifier:
            # Multi-classifier: check the first classifier's feature_order
            classifiers = config.get("classifiers", {})
            if classifiers:
                first_classifier_name = list(classifiers.keys())[0]
                first_classifier = classifiers[first_classifier_name]
                feature_order = first_classifier.get("api", {}).get("feature_order")
            else:
                feature_order = None
        else:
            feature_order = config.get("api", {}).get("feature_order")

        if feature_order is None:
            return CheckResult(
                name="Feature order file",
                status=CheckStatus.PASSED,
                message="No feature_order configured (using auto-detection)"
            )

        if isinstance(feature_order, list):
            return CheckResult(
                name="Feature order file",
                status=CheckStatus.PASSED,
                message=f"Feature order inline ({len(feature_order)} features)",
                details={"feature_count": len(feature_order)}
            )

        # It's a file path
        feature_file = Path(project_path) / feature_order
        if feature_file.exists():
            import json
            with open(feature_file, 'r') as f:
                features = json.load(f)
            return CheckResult(
                name="Feature order file",
                status=CheckStatus.PASSED,
                message=f"Feature order file exists ({len(features)} features)",
                details={"path": str(feature_file), "feature_count": len(features)}
            )
        else:
            return CheckResult(
                name="Feature order file",
                status=CheckStatus.FAILED,
                message=f"Feature order file not found: {feature_order}",
                suggestion="Create the feature order JSON file or use inline list in config"
            )

    except Exception as e:
        return CheckResult(
            name="Feature order file",
            status=CheckStatus.FAILED,
            message=f"Error checking feature order: {e}"
        )


def check_git_repository(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if project is a git repository."""
    git_dir = Path(project_path) / ".git"

    if git_dir.exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=project_path,
                timeout=5
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                return CheckResult(
                    name="Git repository",
                    status=CheckStatus.PASSED,
                    message=f"Git repository (commit: {commit})",
                    details={"commit": commit}
                )
        except Exception:
            pass

        return CheckResult(
            name="Git repository",
            status=CheckStatus.PASSED,
            message="Git repository initialized"
        )

    return CheckResult(
        name="Git repository",
        status=CheckStatus.WARNING,
        message="Not a git repository",
        suggestion="Initialize git for version control: git init"
    )


def check_gitignore(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if .gitignore has recommended patterns."""
    gitignore_path = Path(project_path) / ".gitignore"

    recommended_patterns = [
        "__pycache__",
        "*.pyc",
        ".env",
        "*.pkl",
        "*.joblib",
        "*.h5",
        ".venv",
        "venv",
    ]

    if not gitignore_path.exists():
        return CheckResult(
            name="Gitignore",
            status=CheckStatus.WARNING,
            message=".gitignore not found",
            suggestion="Create .gitignore with: __pycache__/, *.pkl, .env, .venv/"
        )

    try:
        with open(gitignore_path, 'r') as f:
            content = f.read()

        missing = []
        for pattern in recommended_patterns[:4]:  # Check most important ones
            if pattern not in content:
                missing.append(pattern)

        if missing:
            return CheckResult(
                name="Gitignore",
                status=CheckStatus.WARNING,
                message=f".gitignore missing recommended patterns",
                suggestion=f"Add to .gitignore: {', '.join(missing)}",
                details={"missing_patterns": missing}
            )

        return CheckResult(
            name="Gitignore",
            status=CheckStatus.PASSED,
            message=".gitignore has recommended patterns"
        )

    except Exception as e:
        return CheckResult(
            name="Gitignore",
            status=CheckStatus.WARNING,
            message=f"Error reading .gitignore: {e}"
        )


def check_dependencies(project_path: str = ".", verbose: bool = False) -> CheckResult:
    """Check if key dependencies are installed."""
    required_deps = {
        "fastapi": "Core web framework",
        "uvicorn": "ASGI server",
        "pydantic": "Data validation",
        "yaml": "YAML parsing",  # PyYAML imports as 'yaml'
    }

    optional_deps = {
        "catboost": "CatBoost models",
        "scikit-learn": "Scikit-learn models",
        "numpy": "Numerical operations",
        "pandas": "Data manipulation",
    }

    missing_required = []
    missing_optional = []

    for dep, desc in required_deps.items():
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing_required.append(dep)

    for dep, desc in optional_deps.items():
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing_optional.append(f"{dep} ({desc})")

    if missing_required:
        return CheckResult(
            name="Dependencies",
            status=CheckStatus.FAILED,
            message=f"Missing required: {', '.join(missing_required)}",
            suggestion=f"Install with: pip install {' '.join(missing_required)}"
        )

    if missing_optional and verbose:
        return CheckResult(
            name="Dependencies",
            status=CheckStatus.WARNING,
            message=f"Missing optional: {', '.join(missing_optional[:3])}",
            details={"missing_optional": missing_optional}
        )

    return CheckResult(
        name="Dependencies",
        status=CheckStatus.PASSED,
        message="All required dependencies installed"
    )


def check_port_available(port: int = 8000, verbose: bool = False) -> CheckResult:
    """Check if the configured port is available."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            if result == 0:
                return CheckResult(
                    name=f"Port {port}",
                    status=CheckStatus.WARNING,
                    message=f"Port {port} is already in use",
                    suggestion=f"Use a different port with: mlserver serve --port {port + 1}"
                )
            else:
                return CheckResult(
                    name=f"Port {port}",
                    status=CheckStatus.PASSED,
                    message=f"Port {port} is available"
                )
    except Exception as e:
        return CheckResult(
            name=f"Port {port}",
            status=CheckStatus.PASSED,
            message=f"Port {port} appears available"
        )


# =============================================================================
# Aggregated Check Functions
# =============================================================================

def run_system_checks(verbose: bool = False) -> DiagnosticReport:
    """Run all system-level checks."""
    report = DiagnosticReport()

    report.add(check_python_version(verbose))
    report.add(check_docker(verbose))
    report.add(check_git(verbose))

    return report


def run_project_checks(project_path: str = ".", verbose: bool = False) -> DiagnosticReport:
    """Run all project-level checks."""
    report = DiagnosticReport()

    report.add(check_config_file(project_path, verbose))
    report.add(check_config_schema(project_path, verbose))
    report.add(check_predictor_import(project_path, verbose))
    report.add(check_model_files(project_path, verbose))
    report.add(check_feature_order_file(project_path, verbose))
    report.add(check_git_repository(project_path, verbose))
    report.add(check_gitignore(project_path, verbose))

    return report


def run_all_checks(project_path: str = ".", verbose: bool = False) -> DiagnosticReport:
    """Run all diagnostic checks."""
    report = DiagnosticReport()

    # System checks
    report.add(check_python_version(verbose))
    report.add(check_docker(verbose))
    report.add(check_git(verbose))

    # Project checks
    report.add(check_config_file(project_path, verbose))
    report.add(check_config_schema(project_path, verbose))
    report.add(check_predictor_import(project_path, verbose))
    report.add(check_model_files(project_path, verbose))
    report.add(check_feature_order_file(project_path, verbose))
    report.add(check_git_repository(project_path, verbose))
    report.add(check_gitignore(project_path, verbose))
    report.add(check_dependencies(project_path, verbose))

    # Generate recommendations
    for check in report.checks:
        if check.suggestion and check.status in (CheckStatus.WARNING, CheckStatus.FAILED):
            report.add_recommendation(check.suggestion)

    return report


def run_validation_checks(project_path: str = ".", check_imports: bool = True) -> DiagnosticReport:
    """Run validation checks (for mlserver validate command)."""
    report = DiagnosticReport()

    # Essential validation checks
    report.add(check_config_file(project_path))
    report.add(check_config_schema(project_path))

    if check_imports:
        report.add(check_predictor_import(project_path))

    report.add(check_model_files(project_path))
    report.add(check_feature_order_file(project_path))

    return report
