"""
Modular validation system for MLServer projects.

This module provides pluggable validators that can be used at different
stages of the MLServer workflow (init, tag, build, etc.).

Includes:
- Project structure validators (files, git, config)
- Feature schema validators (input validation from feature_order)
- Validation suites for common workflows
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    details: dict = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}


@dataclass
class FeatureValidationResult:
    """Result of feature validation for a single record or batch."""
    valid: bool
    missing_features: List[str] = field(default_factory=list)
    extra_features: List[str] = field(default_factory=list)
    type_errors: List[Dict[str, Any]] = field(default_factory=list)
    record_index: Optional[int] = None

    def to_error_message(self) -> str:
        """Generate human-readable error message."""
        parts = []

        if self.missing_features:
            features_str = ", ".join(f"'{f}'" for f in self.missing_features[:5])
            if len(self.missing_features) > 5:
                features_str += f" ... and {len(self.missing_features) - 5} more"
            parts.append(f"Missing features: {features_str}")

        if self.extra_features:
            features_str = ", ".join(f"'{f}'" for f in self.extra_features[:5])
            if len(self.extra_features) > 5:
                features_str += f" ... and {len(self.extra_features) - 5} more"
            parts.append(f"Unexpected features: {features_str}")

        if self.type_errors:
            type_errs = [f"{e['feature']}: expected {e['expected']}, got {e['actual']}"
                        for e in self.type_errors[:3]]
            parts.append(f"Type errors: {'; '.join(type_errs)}")

        prefix = f"Record {self.record_index}: " if self.record_index is not None else ""
        return prefix + " | ".join(parts) if parts else "Valid"


class FeatureSchemaValidator:
    """Validate input features match expected schema from feature_order.

    This validator catches feature mismatches at request time rather than
    during model prediction, providing better error messages.

    Usage:
        validator = FeatureSchemaValidator(["feature1", "feature2", "feature3"])
        result = validator.validate_record({"feature1": 1.0, "feature2": 2.0})
        if not result.valid:
            print(result.to_error_message())
    """

    def __init__(
        self,
        feature_order: List[str],
        strict: bool = False,
        allow_extra_features: bool = True
    ):
        """
        Initialize feature schema validator.

        Args:
            feature_order: List of expected feature names in order
            strict: If True, fail on any mismatch. If False, allow extra features.
            allow_extra_features: If True, extra features in input are warnings, not errors.
        """
        self.feature_order = feature_order
        self.feature_set: Set[str] = set(feature_order)
        self.strict = strict
        self.allow_extra_features = allow_extra_features

    def validate_record(self, record: Dict[str, Any]) -> FeatureValidationResult:
        """
        Validate a single record against the expected schema.

        Args:
            record: Dictionary of feature name -> value

        Returns:
            FeatureValidationResult with validation details
        """
        record_features = set(record.keys())

        missing = list(self.feature_set - record_features)
        extra = list(record_features - self.feature_set)

        # Determine validity based on mode
        if self.strict:
            valid = len(missing) == 0 and len(extra) == 0
        else:
            # Only missing features cause failure; extra features are allowed
            valid = len(missing) == 0

        return FeatureValidationResult(
            valid=valid,
            missing_features=sorted(missing),
            extra_features=sorted(extra) if not self.allow_extra_features else []
        )

    def validate_records(self, records: List[Dict[str, Any]]) -> Tuple[bool, List[FeatureValidationResult]]:
        """
        Validate multiple records against the expected schema.

        Args:
            records: List of dictionaries to validate

        Returns:
            Tuple of (all_valid, list of results)
        """
        results = []
        all_valid = True

        for i, record in enumerate(records):
            result = self.validate_record(record)
            result.record_index = i
            results.append(result)
            if not result.valid:
                all_valid = False

        return all_valid, results

    def get_validation_summary(self, results: List[FeatureValidationResult]) -> str:
        """
        Generate a summary of validation results.

        Args:
            results: List of validation results

        Returns:
            Human-readable summary string
        """
        invalid_count = sum(1 for r in results if not r.valid)
        if invalid_count == 0:
            return f"All {len(results)} records valid"

        # Aggregate missing features across all records
        all_missing = set()
        for r in results:
            all_missing.update(r.missing_features)

        summary_parts = [f"{invalid_count}/{len(results)} records invalid"]

        if all_missing:
            missing_str = ", ".join(f"'{f}'" for f in sorted(all_missing)[:5])
            if len(all_missing) > 5:
                missing_str += f" ... and {len(all_missing) - 5} more"
            summary_parts.append(f"Missing features: {missing_str}")

        return " | ".join(summary_parts)

    @classmethod
    def from_config(cls, config: "AppConfig") -> Optional["FeatureSchemaValidator"]:
        """
        Create a validator from AppConfig if feature_order is configured.

        Args:
            config: Application configuration

        Returns:
            FeatureSchemaValidator or None if no feature_order configured
        """
        try:
            base_path = Path(config.project_path) if config.project_path else None
            feature_order = config.api.get_resolved_feature_order(base_path=base_path)

            if feature_order:
                return cls(feature_order)
            return None
        except Exception:
            return None


class Validator:
    """Base class for validators."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def validate(self, **kwargs) -> ValidationResult:
        """
        Run the validation check.

        Args:
            **kwargs: Context-specific parameters for validation

        Returns:
            ValidationResult indicating success/failure
        """
        raise NotImplementedError("Subclasses must implement validate()")


class ProjectInitializedValidator(Validator):
    """Validates that a project has been properly initialized with mlserver init."""

    def __init__(self):
        super().__init__(
            name="project_initialized",
            description="Verify all required project files exist"
        )

    def validate(self, project_path: str = ".", classifier_name: Optional[str] = None, **kwargs) -> ValidationResult:
        """
        Check if all required files exist for a classifier project.

        Args:
            project_path: Path to project
            classifier_name: Name of classifier (for specific checks)

        Returns:
            ValidationResult with missing files if any
        """
        from .init_project import check_project_files

        files_ok, missing_files = check_project_files(project_path, classifier_name)

        if files_ok:
            return ValidationResult(passed=True)

        error_msg = "Project not properly initialized - required files are missing"
        details = {
            "missing_files": missing_files,
            "solution": "Run 'mlserver init' to create all required files"
        }

        return ValidationResult(
            passed=False,
            error_message=error_msg,
            details=details
        )


class GitWorkingDirectoryCleanValidator(Validator):
    """Validates that git working directory has no uncommitted changes to tracked files.

    Note: Untracked files are allowed and will not fail validation.
    """

    def __init__(self):
        super().__init__(
            name="git_working_directory_clean",
            description="Verify no uncommitted changes to tracked files (untracked files are allowed)"
        )

    def validate(self, project_path: str = ".", **kwargs) -> ValidationResult:
        """
        Check if git working directory is clean.

        Args:
            project_path: Path to project

        Returns:
            ValidationResult indicating if working directory is clean
        """
        from .version_control import GitVersionManager

        try:
            git_mgr = GitVersionManager(project_path)
            is_clean, error_msg = git_mgr.check_working_directory_clean()

            if is_clean:
                return ValidationResult(passed=True)

            details = {
                "solution": "Commit your changes first with: git add . && git commit -m 'message'"
            }

            return ValidationResult(
                passed=False,
                error_message=error_msg,
                details=details
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                error_message=f"Failed to check git status: {str(e)}"
            )


class GitRepositoryExistsValidator(Validator):
    """Validates that the project is a git repository."""

    def __init__(self):
        super().__init__(
            name="git_repository_exists",
            description="Verify project is a git repository"
        )

    def validate(self, project_path: str = ".", **kwargs) -> ValidationResult:
        """
        Check if project is a git repository.

        Args:
            project_path: Path to project

        Returns:
            ValidationResult indicating if .git directory exists
        """
        git_dir = Path(project_path) / ".git"

        if git_dir.exists():
            return ValidationResult(passed=True)

        details = {
            "solution": "Initialize git repository with: git init"
        }

        return ValidationResult(
            passed=False,
            error_message="Not a git repository",
            details=details
        )


class ConfigurationValidValidator(Validator):
    """Validates that mlserver.yaml is valid and can be parsed."""

    def __init__(self):
        super().__init__(
            name="configuration_valid",
            description="Verify mlserver.yaml is valid"
        )

    def validate(self, project_path: str = ".", classifier_name: Optional[str] = None, **kwargs) -> ValidationResult:
        """
        Check if mlserver.yaml exists and is valid.

        Supports both single-classifier and multi-classifier configurations.

        Args:
            project_path: Path to project
            classifier_name: Classifier name (for multi-classifier configs)

        Returns:
            ValidationResult indicating if config is valid
        """
        mlserver_yaml = Path(project_path) / "mlserver.yaml"

        if not mlserver_yaml.exists():
            return ValidationResult(
                passed=False,
                error_message="mlserver.yaml not found",
                details={"solution": "Run 'mlserver init' to create configuration"}
            )

        try:
            from .config import AppConfig
            from .multi_classifier import detect_multi_classifier_config, extract_single_classifier_config, load_multi_classifier_config
            import yaml

            with open(mlserver_yaml, 'r') as f:
                raw_config = yaml.safe_load(f)

            # Check if this is a multi-classifier config
            if detect_multi_classifier_config(str(mlserver_yaml)):
                # Multi-classifier format
                multi_config = load_multi_classifier_config(str(mlserver_yaml))

                # If classifier_name provided, validate that specific classifier
                if classifier_name:
                    try:
                        # Extract and validate the specific classifier config
                        single_config = extract_single_classifier_config(multi_config, classifier_name)
                        # Validation passed if we got here
                        return ValidationResult(passed=True)
                    except (ValueError, KeyError) as e:
                        return ValidationResult(
                            passed=False,
                            error_message=f"Classifier '{classifier_name}' not found or invalid in multi-classifier config: {str(e)}",
                            details={"solution": "Check classifier name or run 'mlserver init' to recreate configuration"}
                        )
                else:
                    # No classifier specified - just validate the multi-config structure
                    return ValidationResult(passed=True)
            else:
                # Single-classifier format
                AppConfig.model_validate(raw_config)
                return ValidationResult(passed=True)

        except Exception as e:
            return ValidationResult(
                passed=False,
                error_message=f"Invalid mlserver.yaml: {str(e)}",
                details={"solution": "Check your mlserver.yaml for syntax errors"}
            )


class GitHubActionsConfiguredValidator(Validator):
    """Validates that GitHub Actions workflow is configured and compatible."""

    def __init__(self, check_compatibility: bool = True, strict: bool = False):
        """
        Initialize validator.

        Args:
            check_compatibility: If True, validate workflow version compatibility
            strict: If True, fail on version mismatch. If False, only warn.
        """
        super().__init__(
            name="github_actions_configured",
            description="Verify GitHub Actions workflow exists and is compatible"
        )
        self.check_compatibility = check_compatibility
        self.strict = strict

    def validate(self, project_path: str = ".", **kwargs) -> ValidationResult:
        """
        Check if GitHub Actions workflow file exists and is compatible.

        Args:
            project_path: Path to project

        Returns:
            ValidationResult indicating if workflow exists and is compatible
        """
        from .github_actions import check_github_actions_setup, validate_workflow_compatibility

        if not check_github_actions_setup(project_path):
            # Workflow doesn't exist - this is a warning, not an error
            return ValidationResult(
                passed=True,  # Don't fail validation
                warnings=["GitHub Actions workflow not configured"],
                details={
                    "solution": "Run 'mlserver init-github' to add CI/CD automation"
                }
            )

        # Workflow exists - check compatibility if requested
        if self.check_compatibility:
            is_valid, warning, details = validate_workflow_compatibility(project_path, self.strict)

            if not is_valid:
                if self.strict:
                    # Strict mode - fail validation
                    return ValidationResult(
                        passed=False,
                        error_message="GitHub Actions workflow is incompatible with current MLServer version",
                        details=details
                    )
                else:
                    # Lenient mode - pass but warn
                    return ValidationResult(
                        passed=True,
                        warnings=[warning] if warning else [],
                        details=details
                    )
            elif warning:
                # Valid but with warnings
                return ValidationResult(
                    passed=True,
                    warnings=[warning],
                    details=details
                )

        # All good
        return ValidationResult(passed=True)


class ValidationSuite:
    """A collection of validators that can be run together."""

    def __init__(self, name: str, validators: List[Validator]):
        """
        Initialize validation suite.

        Args:
            name: Name of this validation suite
            validators: List of validators to run
        """
        self.name = name
        self.validators = validators

    def validate(self, **kwargs) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validators in the suite.

        Args:
            **kwargs: Context parameters passed to all validators

        Returns:
            Tuple of (all_passed, list_of_results)
        """
        results = []
        all_passed = True

        for validator in self.validators:
            result = validator.validate(**kwargs)
            results.append(result)

            if not result.passed:
                all_passed = False

        return all_passed, results

    def add_validator(self, validator: Validator):
        """Add a validator to this suite."""
        self.validators.append(validator)

    def remove_validator(self, validator_name: str):
        """Remove a validator by name."""
        self.validators = [v for v in self.validators if v.name != validator_name]


# Pre-defined validation suites for common workflows

def get_tag_validation_suite() -> ValidationSuite:
    """Get validation suite for tagging workflow."""
    return ValidationSuite(
        name="tag_validation",
        validators=[
            GitRepositoryExistsValidator(),
            ProjectInitializedValidator(),
            ConfigurationValidValidator(),
            GitWorkingDirectoryCleanValidator(),
            GitHubActionsConfiguredValidator(check_compatibility=True, strict=False),
        ]
    )


def get_build_validation_suite() -> ValidationSuite:
    """Get validation suite for build workflow."""
    return ValidationSuite(
        name="build_validation",
        validators=[
            ProjectInitializedValidator(),
            ConfigurationValidValidator(),
        ]
    )


def get_init_validation_suite() -> ValidationSuite:
    """Get validation suite for init workflow (pre-init checks)."""
    return ValidationSuite(
        name="init_validation",
        validators=[
            GitRepositoryExistsValidator(),
        ]
    )


def get_deploy_validation_suite() -> ValidationSuite:
    """Get validation suite for deployment workflow."""
    return ValidationSuite(
        name="deploy_validation",
        validators=[
            GitRepositoryExistsValidator(),
            ProjectInitializedValidator(),
            ConfigurationValidValidator(),
            GitWorkingDirectoryCleanValidator(),
            GitHubActionsConfiguredValidator(check_compatibility=True, strict=True),
        ]
    )
