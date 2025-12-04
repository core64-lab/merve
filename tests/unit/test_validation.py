"""Unit tests for validation module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.validation import (
    ValidationResult,
    FeatureValidationResult,
    FeatureSchemaValidator,
    Validator,
    ProjectInitializedValidator,
    GitWorkingDirectoryCleanValidator,
    GitRepositoryExistsValidator,
    ConfigurationValidValidator,
    GitHubActionsConfiguredValidator,
    ValidationSuite,
    get_tag_validation_suite,
    get_build_validation_suite,
    get_init_validation_suite,
    get_deploy_validation_suite,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_default_values(self):
        """Test default values for ValidationResult."""
        result = ValidationResult(passed=True)
        assert result.passed is True
        assert result.error_message is None
        assert result.warnings == []
        assert result.details == {}

    def test_with_error_message(self):
        """Test ValidationResult with error message."""
        result = ValidationResult(
            passed=False,
            error_message="Something went wrong"
        )
        assert result.passed is False
        assert result.error_message == "Something went wrong"

    def test_with_warnings(self):
        """Test ValidationResult with warnings."""
        result = ValidationResult(
            passed=True,
            warnings=["Warning 1", "Warning 2"]
        )
        assert len(result.warnings) == 2

    def test_with_details(self):
        """Test ValidationResult with details."""
        result = ValidationResult(
            passed=False,
            details={"key": "value"}
        )
        assert result.details["key"] == "value"


class TestFeatureValidationResult:
    """Test FeatureValidationResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = FeatureValidationResult(valid=True)
        assert result.valid is True
        assert result.missing_features == []
        assert result.extra_features == []
        assert result.type_errors == []
        assert result.record_index is None

    def test_to_error_message_valid(self):
        """Test error message for valid result."""
        result = FeatureValidationResult(valid=True)
        assert result.to_error_message() == "Valid"

    def test_to_error_message_missing_features(self):
        """Test error message with missing features."""
        result = FeatureValidationResult(
            valid=False,
            missing_features=["feature1", "feature2"]
        )
        msg = result.to_error_message()
        assert "Missing features" in msg
        assert "feature1" in msg

    def test_to_error_message_extra_features(self):
        """Test error message with extra features."""
        result = FeatureValidationResult(
            valid=False,
            extra_features=["extra1", "extra2"]
        )
        msg = result.to_error_message()
        assert "Unexpected features" in msg
        assert "extra1" in msg

    def test_to_error_message_type_errors(self):
        """Test error message with type errors."""
        result = FeatureValidationResult(
            valid=False,
            type_errors=[{"feature": "f1", "expected": "int", "actual": "str"}]
        )
        msg = result.to_error_message()
        assert "Type errors" in msg
        assert "f1" in msg

    def test_to_error_message_with_record_index(self):
        """Test error message with record index."""
        result = FeatureValidationResult(
            valid=False,
            missing_features=["f1"],
            record_index=5
        )
        msg = result.to_error_message()
        assert "Record 5:" in msg

    def test_to_error_message_truncates_long_lists(self):
        """Test that long feature lists are truncated."""
        result = FeatureValidationResult(
            valid=False,
            missing_features=[f"feature{i}" for i in range(10)]
        )
        msg = result.to_error_message()
        assert "... and 5 more" in msg


class TestFeatureSchemaValidator:
    """Test FeatureSchemaValidator class."""

    def test_init_basic(self):
        """Test basic initialization."""
        validator = FeatureSchemaValidator(["f1", "f2", "f3"])
        assert validator.feature_order == ["f1", "f2", "f3"]
        assert validator.feature_set == {"f1", "f2", "f3"}
        assert validator.strict is False
        assert validator.allow_extra_features is True

    def test_init_strict_mode(self):
        """Test initialization with strict mode."""
        validator = FeatureSchemaValidator(["f1"], strict=True, allow_extra_features=False)
        assert validator.strict is True
        assert validator.allow_extra_features is False

    def test_validate_record_valid(self):
        """Test validating a valid record."""
        validator = FeatureSchemaValidator(["f1", "f2"])
        result = validator.validate_record({"f1": 1.0, "f2": 2.0})
        assert result.valid is True
        assert len(result.missing_features) == 0

    def test_validate_record_missing_features(self):
        """Test validating record with missing features."""
        validator = FeatureSchemaValidator(["f1", "f2", "f3"])
        result = validator.validate_record({"f1": 1.0})
        assert result.valid is False
        assert "f2" in result.missing_features
        assert "f3" in result.missing_features

    def test_validate_record_extra_features_allowed(self):
        """Test that extra features are allowed by default."""
        validator = FeatureSchemaValidator(["f1"])
        result = validator.validate_record({"f1": 1.0, "f2": 2.0})
        assert result.valid is True  # Extra features don't fail

    def test_validate_record_strict_mode_extra_features(self):
        """Test strict mode fails on extra features."""
        validator = FeatureSchemaValidator(["f1"], strict=True)
        result = validator.validate_record({"f1": 1.0, "f2": 2.0})
        assert result.valid is False  # Extra features cause failure in strict mode

    def test_validate_records_all_valid(self):
        """Test validating multiple valid records."""
        validator = FeatureSchemaValidator(["f1", "f2"])
        records = [
            {"f1": 1.0, "f2": 2.0},
            {"f1": 3.0, "f2": 4.0}
        ]
        all_valid, results = validator.validate_records(records)
        assert all_valid is True
        assert len(results) == 2
        assert results[0].record_index == 0
        assert results[1].record_index == 1

    def test_validate_records_some_invalid(self):
        """Test validating records with some invalid."""
        validator = FeatureSchemaValidator(["f1", "f2"])
        records = [
            {"f1": 1.0, "f2": 2.0},  # Valid
            {"f1": 3.0}  # Missing f2
        ]
        all_valid, results = validator.validate_records(records)
        assert all_valid is False
        assert results[0].valid is True
        assert results[1].valid is False

    def test_get_validation_summary_all_valid(self):
        """Test summary when all records are valid."""
        validator = FeatureSchemaValidator(["f1"])
        results = [
            FeatureValidationResult(valid=True),
            FeatureValidationResult(valid=True)
        ]
        summary = validator.get_validation_summary(results)
        assert "All 2 records valid" in summary

    def test_get_validation_summary_some_invalid(self):
        """Test summary when some records are invalid."""
        validator = FeatureSchemaValidator(["f1", "f2"])
        results = [
            FeatureValidationResult(valid=False, missing_features=["f1"]),
            FeatureValidationResult(valid=True)
        ]
        summary = validator.get_validation_summary(results)
        assert "1/2 records invalid" in summary
        assert "Missing features" in summary

    def test_from_config_no_feature_order(self):
        """Test from_config when no feature_order configured."""
        mock_config = MagicMock()
        mock_config.project_path = None
        mock_config.api.get_resolved_feature_order.return_value = None

        validator = FeatureSchemaValidator.from_config(mock_config)
        assert validator is None

    def test_from_config_with_feature_order(self):
        """Test from_config with feature_order configured."""
        mock_config = MagicMock()
        mock_config.project_path = None
        mock_config.api.get_resolved_feature_order.return_value = ["f1", "f2"]

        validator = FeatureSchemaValidator.from_config(mock_config)
        assert validator is not None
        assert validator.feature_order == ["f1", "f2"]

    def test_from_config_handles_exception(self):
        """Test from_config handles exceptions gracefully."""
        mock_config = MagicMock()
        mock_config.api.get_resolved_feature_order.side_effect = Exception("Error")

        validator = FeatureSchemaValidator.from_config(mock_config)
        assert validator is None


class TestValidatorBase:
    """Test Validator base class."""

    def test_validator_init(self):
        """Test base validator initialization."""
        class TestValidator(Validator):
            def validate(self, **kwargs):
                return ValidationResult(passed=True)

        v = TestValidator("test", "Test validator")
        assert v.name == "test"
        assert v.description == "Test validator"

    def test_validator_not_implemented(self):
        """Test base class raises NotImplementedError."""
        v = Validator("test", "Test")
        with pytest.raises(NotImplementedError):
            v.validate()


class TestGitRepositoryExistsValidator:
    """Test GitRepositoryExistsValidator."""

    def test_git_repo_exists(self):
        """Test validation passes when .git exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()

            validator = GitRepositoryExistsValidator()
            result = validator.validate(project_path=tmpdir)

            assert result.passed is True

    def test_git_repo_not_exists(self):
        """Test validation fails when .git doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = GitRepositoryExistsValidator()
            result = validator.validate(project_path=tmpdir)

            assert result.passed is False
            assert "Not a git repository" in result.error_message
            assert "git init" in result.details["solution"]


class TestGitWorkingDirectoryCleanValidator:
    """Test GitWorkingDirectoryCleanValidator."""

    def test_clean_directory(self):
        """Test validation passes when directory is clean."""
        with patch('mlserver.version_control.GitVersionManager') as mock_class:
            mock_mgr = MagicMock()
            mock_mgr.check_working_directory_clean.return_value = (True, None)
            mock_class.return_value = mock_mgr

            validator = GitWorkingDirectoryCleanValidator()
            result = validator.validate(project_path=".")

            assert result.passed is True

    def test_dirty_directory(self):
        """Test validation fails when directory is dirty."""
        with patch('mlserver.version_control.GitVersionManager') as mock_class:
            mock_mgr = MagicMock()
            mock_mgr.check_working_directory_clean.return_value = (False, "Uncommitted changes")
            mock_class.return_value = mock_mgr

            validator = GitWorkingDirectoryCleanValidator()
            result = validator.validate(project_path=".")

            assert result.passed is False
            assert "Uncommitted changes" in result.error_message

    def test_git_error(self):
        """Test validation handles git errors."""
        with patch('mlserver.version_control.GitVersionManager') as mock_class:
            mock_class.side_effect = Exception("Git not found")

            validator = GitWorkingDirectoryCleanValidator()
            result = validator.validate(project_path=".")

            assert result.passed is False
            assert "Failed to check git status" in result.error_message


class TestConfigurationValidValidator:
    """Test ConfigurationValidValidator."""

    def test_config_not_found(self):
        """Test validation fails when config doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ConfigurationValidValidator()
            result = validator.validate(project_path=tmpdir)

            assert result.passed is False
            assert "mlserver.yaml not found" in result.error_message

    def test_valid_single_classifier_config(self):
        """Test validation passes for valid single-classifier config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("""
predictor:
  module: test
  class_name: TestPredictor
classifier:
  name: test-classifier
  version: "1.0.0"
api:
  adapter: records
""")

            validator = ConfigurationValidValidator()
            result = validator.validate(project_path=tmpdir)

            assert result.passed is True

    def test_invalid_config(self):
        """Test validation fails for invalid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mlserver.yaml"
            config_path.write_text("invalid: yaml: :")

            validator = ConfigurationValidValidator()
            result = validator.validate(project_path=tmpdir)

            assert result.passed is False
            assert "Invalid mlserver.yaml" in result.error_message


class TestProjectInitializedValidator:
    """Test ProjectInitializedValidator."""

    def test_project_initialized(self):
        """Test validation passes when project is initialized."""
        with patch('mlserver.init_project.check_project_files') as mock_check:
            mock_check.return_value = (True, [])

            validator = ProjectInitializedValidator()
            result = validator.validate(project_path=".")

            assert result.passed is True

    def test_project_not_initialized(self):
        """Test validation fails when project not initialized."""
        with patch('mlserver.init_project.check_project_files') as mock_check:
            mock_check.return_value = (False, ["mlserver.yaml", "predictor.py"])

            validator = ProjectInitializedValidator()
            result = validator.validate(project_path=".")

            assert result.passed is False
            assert "not properly initialized" in result.error_message
            assert "mlserver.yaml" in result.details["missing_files"]


class TestGitHubActionsConfiguredValidator:
    """Test GitHubActionsConfiguredValidator."""

    def test_workflow_not_configured(self):
        """Test validation passes but warns when workflow not configured."""
        with patch('mlserver.github_actions.check_github_actions_setup') as mock_check:
            mock_check.return_value = False

            validator = GitHubActionsConfiguredValidator()
            result = validator.validate(project_path=".")

            assert result.passed is True
            assert len(result.warnings) > 0
            assert "not configured" in result.warnings[0]

    def test_workflow_valid(self):
        """Test validation passes when workflow is valid."""
        with patch('mlserver.github_actions.check_github_actions_setup') as mock_setup:
            with patch('mlserver.github_actions.validate_workflow_compatibility') as mock_compat:
                mock_setup.return_value = True
                mock_compat.return_value = (True, None, {})

                validator = GitHubActionsConfiguredValidator()
                result = validator.validate(project_path=".")

                assert result.passed is True

    def test_workflow_incompatible_strict(self):
        """Test strict mode fails on incompatible workflow."""
        with patch('mlserver.github_actions.check_github_actions_setup') as mock_setup:
            with patch('mlserver.github_actions.validate_workflow_compatibility') as mock_compat:
                mock_setup.return_value = True
                mock_compat.return_value = (False, "Version mismatch", {"old": "1.0", "new": "2.0"})

                validator = GitHubActionsConfiguredValidator(strict=True)
                result = validator.validate(project_path=".")

                assert result.passed is False
                assert "incompatible" in result.error_message

    def test_workflow_incompatible_lenient(self):
        """Test lenient mode warns on incompatible workflow."""
        with patch('mlserver.github_actions.check_github_actions_setup') as mock_setup:
            with patch('mlserver.github_actions.validate_workflow_compatibility') as mock_compat:
                mock_setup.return_value = True
                mock_compat.return_value = (False, "Version mismatch", {})

                validator = GitHubActionsConfiguredValidator(strict=False)
                result = validator.validate(project_path=".")

                assert result.passed is True
                assert len(result.warnings) > 0


class TestValidationSuite:
    """Test ValidationSuite class."""

    def test_suite_all_pass(self):
        """Test suite when all validators pass."""
        validator1 = MagicMock()
        validator1.validate.return_value = ValidationResult(passed=True)
        validator2 = MagicMock()
        validator2.validate.return_value = ValidationResult(passed=True)

        suite = ValidationSuite("test", [validator1, validator2])
        all_passed, results = suite.validate()

        assert all_passed is True
        assert len(results) == 2

    def test_suite_some_fail(self):
        """Test suite when some validators fail."""
        validator1 = MagicMock()
        validator1.validate.return_value = ValidationResult(passed=True)
        validator2 = MagicMock()
        validator2.validate.return_value = ValidationResult(passed=False, error_message="Error")

        suite = ValidationSuite("test", [validator1, validator2])
        all_passed, results = suite.validate()

        assert all_passed is False
        assert results[0].passed is True
        assert results[1].passed is False

    def test_add_validator(self):
        """Test adding a validator to suite."""
        suite = ValidationSuite("test", [])
        validator = MagicMock()

        suite.add_validator(validator)

        assert len(suite.validators) == 1

    def test_remove_validator(self):
        """Test removing a validator from suite."""
        validator = MagicMock()
        validator.name = "test_validator"
        suite = ValidationSuite("test", [validator])

        suite.remove_validator("test_validator")

        assert len(suite.validators) == 0


class TestValidationSuiteFactories:
    """Test pre-defined validation suite factory functions."""

    def test_get_tag_validation_suite(self):
        """Test tag validation suite has expected validators."""
        suite = get_tag_validation_suite()
        assert suite.name == "tag_validation"
        assert len(suite.validators) >= 4
        validator_names = [v.name for v in suite.validators]
        assert "git_repository_exists" in validator_names
        assert "git_working_directory_clean" in validator_names

    def test_get_build_validation_suite(self):
        """Test build validation suite has expected validators."""
        suite = get_build_validation_suite()
        assert suite.name == "build_validation"
        assert len(suite.validators) >= 2
        validator_names = [v.name for v in suite.validators]
        assert "project_initialized" in validator_names
        assert "configuration_valid" in validator_names

    def test_get_init_validation_suite(self):
        """Test init validation suite has expected validators."""
        suite = get_init_validation_suite()
        assert suite.name == "init_validation"
        validator_names = [v.name for v in suite.validators]
        assert "git_repository_exists" in validator_names

    def test_get_deploy_validation_suite(self):
        """Test deploy validation suite has expected validators."""
        suite = get_deploy_validation_suite()
        assert suite.name == "deploy_validation"
        assert len(suite.validators) >= 5
        validator_names = [v.name for v in suite.validators]
        assert "git_working_directory_clean" in validator_names
        assert "github_actions_configured" in validator_names
