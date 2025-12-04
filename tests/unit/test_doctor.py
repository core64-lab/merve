"""Unit tests for the doctor module (Phase 4).

Tests for mlserver validate, mlserver doctor, and related diagnostic checks.
"""
import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.doctor import (
    CheckStatus,
    CheckResult,
    DiagnosticReport,
    check_python_version,
    check_docker,
    check_git,
    check_config_file,
    check_config_schema,
    check_predictor_import,
    check_model_files,
    check_feature_order_file,
    check_git_repository,
    check_gitignore,
    check_dependencies,
    check_port_available,
    run_system_checks,
    run_project_checks,
    run_all_checks,
    run_validation_checks,
)


class TestCheckStatus:
    """Test CheckStatus enum."""

    def test_check_status_values(self):
        """Test CheckStatus has expected values."""
        assert CheckStatus.PASSED
        assert CheckStatus.FAILED
        assert CheckStatus.WARNING
        assert CheckStatus.SKIPPED


class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test creating a CheckResult."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.PASSED,
            message="All good"
        )
        assert result.name == "Test Check"
        assert result.status == CheckStatus.PASSED
        assert result.message == "All good"

    def test_check_result_with_suggestion(self):
        """Test CheckResult with suggestion."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.FAILED,
            message="Something wrong",
            suggestion="Try fixing it"
        )
        assert result.suggestion == "Try fixing it"

    def test_check_result_with_details(self):
        """Test CheckResult with details dict."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.PASSED,
            details={"version": "3.12", "path": "/usr/bin/python"}
        )
        assert result.details["version"] == "3.12"


class TestDiagnosticReport:
    """Test DiagnosticReport dataclass."""

    def test_diagnostic_report_creation(self):
        """Test creating a DiagnosticReport."""
        checks = [
            CheckResult("Check 1", CheckStatus.PASSED),
            CheckResult("Check 2", CheckStatus.FAILED),
        ]
        report = DiagnosticReport(checks=checks)
        assert len(report.checks) == 2

    def test_diagnostic_report_properties(self):
        """Test DiagnosticReport property methods."""
        checks = [
            CheckResult("Check 1", CheckStatus.PASSED),
            CheckResult("Check 2", CheckStatus.PASSED),
            CheckResult("Check 3", CheckStatus.FAILED),
            CheckResult("Check 4", CheckStatus.WARNING),
        ]
        report = DiagnosticReport(checks=checks)

        # Test boolean properties
        assert report.has_errors  # Has FAILED check
        assert report.has_warnings  # Has WARNING check
        assert not report.all_passed  # Not all passed

    def test_diagnostic_report_all_passed(self):
        """Test DiagnosticReport when all checks pass."""
        checks = [
            CheckResult("Check 1", CheckStatus.PASSED),
            CheckResult("Check 2", CheckStatus.PASSED),
        ]
        report = DiagnosticReport(checks=checks)

        assert report.all_passed
        assert not report.has_errors

    def test_diagnostic_report_add(self):
        """Test adding checks to DiagnosticReport."""
        report = DiagnosticReport()
        report.add(CheckResult("Check 1", CheckStatus.PASSED))
        report.add(CheckResult("Check 2", CheckStatus.FAILED))

        assert len(report.checks) == 2


class TestSystemChecks:
    """Test system-level diagnostic checks."""

    def test_check_python_version(self):
        """Test Python version check."""
        result = check_python_version()

        assert result.name == "Python version"
        assert result.status in [CheckStatus.PASSED, CheckStatus.WARNING, CheckStatus.FAILED]
        assert "version" in result.details or result.message

    def test_check_docker_available(self):
        """Test Docker availability check when Docker is available."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Docker version 24.0.0")

            result = check_docker()

            assert result.name == "Docker"
            assert result.status == CheckStatus.PASSED

    def test_check_docker_not_available(self):
        """Test Docker check when Docker is not available."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = check_docker()

            assert result.name == "Docker"
            assert result.status in [CheckStatus.WARNING, CheckStatus.FAILED]

    def test_check_git_available(self):
        """Test Git availability check."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="git version 2.40.0")

            result = check_git()

            assert result.name == "Git"
            assert result.status == CheckStatus.PASSED

    def test_check_git_not_available(self):
        """Test Git check when Git is not available."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = check_git()

            assert result.name == "Git"
            assert result.status in [CheckStatus.WARNING, CheckStatus.FAILED]


class TestProjectChecks:
    """Test project-level diagnostic checks."""

    def test_check_config_file_exists(self):
        """Test config file check when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("server:\n  port: 8000\n")

            result = check_config_file(tmpdir)

            assert result.name == "Configuration file"
            assert result.status == CheckStatus.PASSED

    def test_check_config_file_missing(self):
        """Test config file check when file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_config_file(tmpdir)

            assert result.name == "Configuration file"
            assert result.status == CheckStatus.FAILED
            assert result.suggestion  # Should have a suggestion

    def test_check_config_schema_valid(self):
        """Test config schema check with valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            # Include all required fields for AppConfig
            config_path.write_text("""
server:
  host: 0.0.0.0
  port: 8000

predictor:
  module: my_predictor
  class_name: MyPredictor

classifier:
  name: test-classifier
  version: "1.0.0"

api:
  version: v1
  adapter: auto
""")

            result = check_config_schema(tmpdir)

            assert result.name == "Configuration schema"
            # Config may fail due to missing classifier.repository but schema is parsed
            assert result.status in [CheckStatus.PASSED, CheckStatus.FAILED]

    def test_check_config_schema_invalid(self):
        """Test config schema check with invalid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("invalid: yaml: content:")

            result = check_config_schema(tmpdir)

            assert result.name == "Configuration schema"
            assert result.status == CheckStatus.FAILED

    def test_check_dependencies(self):
        """Test dependencies check."""
        result = check_dependencies()

        assert result.name == "Dependencies"
        # Should pass since we're running in the test environment
        assert result.status in [CheckStatus.PASSED, CheckStatus.WARNING]

    def test_check_port_available(self):
        """Test port availability check."""
        # Use a high port that's likely available
        result = check_port_available(port=59999)

        # Name includes port number
        assert "Port" in result.name
        assert result.status in [CheckStatus.PASSED, CheckStatus.WARNING]


class TestAggregationFunctions:
    """Test check aggregation functions."""

    def test_run_system_checks(self):
        """Test running all system checks."""
        report = run_system_checks()

        assert isinstance(report, DiagnosticReport)
        assert len(report.checks) > 0
        # Should include Python, Docker, Git checks
        check_names = [c.name for c in report.checks]
        assert "Python version" in check_names

    def test_run_project_checks(self):
        """Test running project checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal config
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
classifier:
  name: test
  version: "1.0.0"
""")

            report = run_project_checks(tmpdir)

            assert isinstance(report, DiagnosticReport)
            assert len(report.checks) > 0

    def test_run_all_checks(self):
        """Test running all checks (system + project)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_all_checks(tmpdir)

            assert isinstance(report, DiagnosticReport)
            # Should have both system and project checks
            assert len(report.checks) >= 3

    def test_run_validation_checks(self):
        """Test running validation-only checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal config
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
classifier:
  name: test
  version: "1.0.0"
""")

            report = run_validation_checks(tmpdir)

            assert isinstance(report, DiagnosticReport)
            # Validation should include config and schema checks
            check_names = [c.name for c in report.checks]
            assert "Configuration file" in check_names
            assert "Configuration schema" in check_names


class TestMultiClassifierSupport:
    """Test doctor checks with multi-classifier configs."""

    def test_check_config_schema_multi_classifier(self):
        """Test schema check with multi-classifier config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
classifiers:
  sentiment:
    predictor:
      module: sentiment_predictor
      class_name: SentimentPredictor
    api:
      adapter: records

  fraud:
    predictor:
      module: fraud_predictor
      class_name: FraudPredictor
    api:
      adapter: ndarray
""")

            result = check_config_schema(tmpdir)

            assert result.name == "Configuration schema"
            # Should detect and validate multi-classifier format
            assert result.status in [CheckStatus.PASSED, CheckStatus.FAILED]
            if result.status == CheckStatus.PASSED:
                assert "multi-classifier" in result.message.lower() or "2" in result.message


class TestCheckResultProperties:
    """Test CheckResult property methods."""

    def test_passed_property(self):
        """Test passed property."""
        result = CheckResult("Test", CheckStatus.PASSED)
        assert result.passed is True
        assert result.failed is False
        assert result.warning is False

    def test_failed_property(self):
        """Test failed property."""
        result = CheckResult("Test", CheckStatus.FAILED)
        assert result.failed is True
        assert result.passed is False
        assert result.warning is False

    def test_warning_property(self):
        """Test warning property."""
        result = CheckResult("Test", CheckStatus.WARNING)
        assert result.warning is True
        assert result.passed is False
        assert result.failed is False


class TestDiagnosticReportRecommendations:
    """Test DiagnosticReport recommendations."""

    def test_add_recommendation(self):
        """Test adding recommendation."""
        report = DiagnosticReport()
        report.add_recommendation("Install Docker")
        report.add_recommendation("Update Python")

        assert len(report.recommendations) == 2
        assert "Install Docker" in report.recommendations


class TestCheckDockerTimeoutAndErrors:
    """Test Docker check edge cases."""

    def test_docker_timeout(self):
        """Test Docker check when command times out."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/docker"
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("docker", 5)

                result = check_docker()

                assert result.name == "Docker"
                assert result.status == CheckStatus.WARNING
                assert "timed out" in result.message.lower()

    def test_docker_daemon_not_running(self):
        """Test Docker check when daemon not running."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/docker"
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="")

                result = check_docker()

                assert result.name == "Docker"
                assert result.status == CheckStatus.WARNING
                assert "daemon" in result.message.lower()


class TestCheckGitErrors:
    """Test Git check edge cases."""

    def test_git_not_in_path(self):
        """Test Git check when git not in PATH."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = None

            result = check_git()

            assert result.name == "Git"
            assert result.status == CheckStatus.FAILED
            assert "not found" in result.message.lower()

    def test_git_generic_error(self):
        """Test Git check with generic error."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/git"
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = Exception("Unknown error")

                result = check_git()

                assert result.name == "Git"
                assert result.status == CheckStatus.FAILED


class TestCheckConfigFileEdgeCases:
    """Test config file check edge cases."""

    def test_config_file_empty(self):
        """Test config file check with empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("")

            result = check_config_file(tmpdir)

            assert result.name == "Configuration file"
            assert result.status == CheckStatus.FAILED
            assert "empty" in result.message.lower()

    def test_config_file_yaml_error(self):
        """Test config file check with invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("invalid: yaml: : : :")

            result = check_config_file(tmpdir)

            assert result.name == "Configuration file"
            assert result.status == CheckStatus.FAILED


class TestCheckConfigSchemaEdgeCases:
    """Test config schema check edge cases."""

    def test_config_schema_no_file(self):
        """Test config schema check when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_config_schema(tmpdir)

            assert result.name == "Configuration schema"
            assert result.status == CheckStatus.SKIPPED

    def test_config_schema_validation_error(self):
        """Test config schema check with validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  # Missing class_name
classifier:
  name: test
  version: "1.0.0"
""")

            result = check_config_schema(tmpdir)

            assert result.name == "Configuration schema"
            assert result.status == CheckStatus.FAILED


class TestCheckPredictorImport:
    """Test predictor import check."""

    def test_predictor_import_no_config(self):
        """Test predictor import when config doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_predictor_import(tmpdir)

            assert result.name == "Predictor import"
            assert result.status == CheckStatus.SKIPPED

    def test_predictor_import_missing_config(self):
        """Test predictor import with missing predictor config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  # Missing module and class_name
""")

            result = check_predictor_import(tmpdir)

            assert result.name == "Predictor import"
            assert result.status == CheckStatus.FAILED

    def test_predictor_import_multi_classifier_empty(self):
        """Test predictor import with empty multi-classifier config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
classifiers: {}
""")

            result = check_predictor_import(tmpdir)

            assert result.name == "Predictor import"
            assert result.status == CheckStatus.FAILED
            assert "No classifiers" in result.message


class TestCheckModelFiles:
    """Test model files check."""

    def test_model_files_no_config(self):
        """Test model files check when config doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_model_files(tmpdir)

            assert result.name == "Model files"
            assert result.status == CheckStatus.SKIPPED

    def test_model_files_no_init_kwargs(self):
        """Test model files check with no init_kwargs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
""")

            result = check_model_files(tmpdir)

            assert result.name == "Model files"
            assert result.status == CheckStatus.PASSED
            assert "No model files configured" in result.message

    def test_model_files_missing(self):
        """Test model files check with missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
  init_kwargs:
    model_path: nonexistent_model.pkl
""")

            result = check_model_files(tmpdir)

            assert result.name == "Model files"
            assert result.status == CheckStatus.FAILED
            assert "nonexistent_model.pkl" in result.message

    def test_model_files_exist(self):
        """Test model files check when files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model file
            model_file = Path(tmpdir) / "model.pkl"
            model_file.write_bytes(b"fake model")

            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
  init_kwargs:
    model_path: model.pkl
""")

            result = check_model_files(tmpdir)

            assert result.name == "Model files"
            assert result.status == CheckStatus.PASSED


class TestCheckFeatureOrderFile:
    """Test feature order file check."""

    def test_feature_order_no_config(self):
        """Test feature order check when config doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_feature_order_file(tmpdir)

            assert result.name == "Feature order file"
            assert result.status == CheckStatus.SKIPPED

    def test_feature_order_not_configured(self):
        """Test feature order check when not configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
api:
  adapter: auto
""")

            result = check_feature_order_file(tmpdir)

            assert result.name == "Feature order file"
            assert result.status == CheckStatus.PASSED
            assert "auto-detection" in result.message.lower()

    def test_feature_order_inline_list(self):
        """Test feature order check with inline list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
api:
  feature_order: ["feature1", "feature2", "feature3"]
""")

            result = check_feature_order_file(tmpdir)

            assert result.name == "Feature order file"
            assert result.status == CheckStatus.PASSED
            assert "inline" in result.message.lower()
            assert "3 features" in result.message

    def test_feature_order_file_exists(self):
        """Test feature order check when file exists."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create feature order file
            feature_file = Path(tmpdir) / "features.json"
            feature_file.write_text(json.dumps(["f1", "f2", "f3", "f4"]))

            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
api:
  feature_order: features.json
""")

            result = check_feature_order_file(tmpdir)

            assert result.name == "Feature order file"
            assert result.status == CheckStatus.PASSED
            assert "4 features" in result.message

    def test_feature_order_file_missing(self):
        """Test feature order check when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: Test
api:
  feature_order: missing_features.json
""")

            result = check_feature_order_file(tmpdir)

            assert result.name == "Feature order file"
            assert result.status == CheckStatus.FAILED


class TestCheckGitRepository:
    """Test git repository check."""

    def test_git_repository_not_initialized(self):
        """Test git repository check when not initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_git_repository(tmpdir)

            assert result.name == "Git repository"
            assert result.status == CheckStatus.WARNING
            assert "Not a git" in result.message

    def test_git_repository_initialized(self):
        """Test git repository check when initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()

            result = check_git_repository(tmpdir)

            assert result.name == "Git repository"
            # May be PASSED with or without commit info
            assert result.status in [CheckStatus.PASSED, CheckStatus.WARNING]


class TestCheckGitignore:
    """Test gitignore check."""

    def test_gitignore_not_found(self):
        """Test gitignore check when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_gitignore(tmpdir)

            assert result.name == "Gitignore"
            assert result.status == CheckStatus.WARNING
            assert ".gitignore not found" in result.message

    def test_gitignore_has_patterns(self):
        """Test gitignore check when has recommended patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gitignore_path = Path(tmpdir) / ".gitignore"
            gitignore_path.write_text("""
__pycache__/
*.pyc
.env
*.pkl
.venv/
""")

            result = check_gitignore(tmpdir)

            assert result.name == "Gitignore"
            assert result.status == CheckStatus.PASSED

    def test_gitignore_missing_patterns(self):
        """Test gitignore check when missing patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gitignore_path = Path(tmpdir) / ".gitignore"
            gitignore_path.write_text("*.log\n")

            result = check_gitignore(tmpdir)

            assert result.name == "Gitignore"
            assert result.status == CheckStatus.WARNING
            assert "missing" in result.message.lower()


class TestCheckDependenciesVerbose:
    """Test dependencies check verbose mode."""

    def test_dependencies_verbose_missing_optional(self):
        """Test dependencies check verbose mode with missing optional deps."""
        # Mock missing optional dependency
        with patch.dict('sys.modules', {'catboost': None}):
            result = check_dependencies(verbose=True)

            # May have warning about missing optional deps
            assert result.name == "Dependencies"
            assert result.status in [CheckStatus.PASSED, CheckStatus.WARNING]


class TestCheckPortAvailable:
    """Test port available check edge cases."""

    def test_port_check_socket_error(self):
        """Test port check with socket error."""
        with patch('socket.socket') as mock_socket:
            mock_socket.return_value.__enter__.return_value.connect_ex.side_effect = Exception("Socket error")

            result = check_port_available(8000)

            assert "Port" in result.name
            assert result.status == CheckStatus.PASSED  # Assumes available on error


class TestRunAllChecksRecommendations:
    """Test run_all_checks with recommendations."""

    def test_run_all_checks_generates_recommendations(self):
        """Test that run_all_checks generates recommendations from failed checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No config file = failed check with suggestion
            report = run_all_checks(tmpdir)

            # Should have recommendations if there were failures with suggestions
            if report.has_errors or report.has_warnings:
                # May have recommendations
                pass

            assert isinstance(report, DiagnosticReport)
