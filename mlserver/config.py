from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from . import defaults
from .errors import ConfigurationError
from .version import ClassifierMetadata, load_classifier_metadata

logger = logging.getLogger(__name__)

# Once-per-process guard for the legacy global_config.yaml warning (RFC 0001 D12)
_GLOBAL_CONFIG_WARNING_EMITTED = False


def _warn_if_legacy_global_config() -> None:
    """Warn (once per process) if a legacy global_config.yaml sits in the CWD.

    GlobalSettings and global_config.yaml were removed in RFC 0001 D12;
    defaults now live in mlserver/defaults.py and are overridden per project
    via mlserver.yaml or environment variables.
    """
    global _GLOBAL_CONFIG_WARNING_EMITTED
    if _GLOBAL_CONFIG_WARNING_EMITTED:
        return
    try:
        present = (Path.cwd() / "global_config.yaml").exists()
    except OSError:
        return
    if present:
        _GLOBAL_CONFIG_WARNING_EMITTED = True
        logger.warning(
            "global_config.yaml is no longer read; use mlserver.yaml or "
            "environment variables instead (RFC 0001 D12)"
        )


# Free-form dict fields whose contents are user-defined and must NOT be
# checked for unknown keys (see _warn_unknown_config_keys)
_FREE_FORM_CONFIG_FIELDS = {
    "classifier",
    "model",
    "init_kwargs",
    "endpoints",
    "resource_limits",
    "health_check",
}


def _extract_nested_model_class(annotation: Any) -> Optional[type]:
    """Extract a BaseModel subclass from a field annotation.

    Handles plain model annotations (ServerConfig) as well as
    Optional[Model] / Union[...] annotations.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _warn_unknown_config_keys(data: Any, model_cls: type, prefix: str = "") -> None:
    """Recursively warn about raw config keys that match no declared model field.

    Pydantic silently drops unknown keys, so typos like 'porrt' would
    otherwise be ignored and defaults used. This only warns - it never rejects.
    Free-form dict fields (classifier, model, init_kwargs, ...) are not
    descended into.
    """
    if not isinstance(data, dict):
        return

    fields = model_cls.model_fields
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in fields:
            logger.warning(f"Unknown config key '{path}' — ignored (check for typos)")
            continue
        if key in _FREE_FORM_CONFIG_FIELDS:
            continue
        nested_cls = _extract_nested_model_class(fields[key].annotation)
        if nested_cls is not None and isinstance(value, dict):
            _warn_unknown_config_keys(value, nested_cls, prefix=f"{path}.")


class CORSConfig(BaseModel):
    # Security: Default to no CORS instead of wildcard. Require explicit
    # configuration for production.
    allow_origins: list[str] = Field(
        default_factory=list, description="Allowed origins (empty = CORS disabled)"
    )
    allow_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST"], description="Allowed HTTP methods"
    )
    allow_headers: list[str] = Field(
        default_factory=lambda: ["Content-Type"], description="Allowed headers"
    )
    allow_credentials: bool = Field(default=False, description="Allow credentials in CORS requests")


class LoggerConfig(BaseModel):
    """Logger configuration for structured logging."""

    timestamp: bool = Field(
        default=False,
        description="Include timestamps in logs (default: false for container environments)",
    )
    structured: bool = Field(default=True, description="Use structured JSON logging")
    show_tasks: bool = Field(
        default=False, description="Show async task names (Task-1, Task-2) in logs"
    )
    format: Optional[str] = Field(
        default=None, description="Custom log format string (overrides other settings)"
    )


class ServerConfig(BaseModel):
    title: str = "ML Server"
    host: str = Field(default_factory=defaults.default_host)
    port: int = Field(default_factory=defaults.default_port)
    log_level: str = Field(default_factory=defaults.default_log_level)
    workers: int = Field(default=defaults.DEFAULT_WORKERS)
    cors: Optional[CORSConfig] = None
    logger: Optional[LoggerConfig] = Field(default=None, description="Logger configuration")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if v < 1 or v > 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("workers must be at least 1")
        return v


class PredictorConfig(BaseModel):
    module: str  # e.g., "examples.predictor_catboost"
    class_name: str  # e.g., "CatBoostPredictor"
    init_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("module", "class_name")
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("field cannot be empty")
        return v


class ObservabilityConfig(BaseModel):
    metrics: bool = Field(default=True, description="Enable Prometheus metrics collection")
    metrics_endpoint: str = Field(default="/metrics", description="Metrics endpoint path")
    structured_logging: bool = Field(default=True, description="Enable structured JSON logging")
    log_payloads: bool = Field(default=False, description="Log request/response payloads")
    correlation_ids: bool = Field(
        default=True, description="Generate correlation IDs for request tracing"
    )


class ApiConfig(BaseModel):
    """Unified API configuration from mlserver.yaml"""

    version: str = Field(default="v1", description="API version for metadata tracking")
    endpoints: dict[str, bool] = Field(
        default_factory=lambda: {
            "predict": True,
            # Note: batch_predict removed - /predict handles both single and batch
            "predict_proba": True,
        },
        description="Enabled endpoints",
    )
    adapter: str = Field(default="records", description="records|ndarray|auto")
    feature_order: Optional[Union[list[str], str]] = None  # Can be list[str] or str (file path)
    _resolved_feature_order: Optional[list[str]] = None  # Cached resolved value
    thread_safe_predict: bool = False
    max_concurrent_predictions: int = Field(
        default=1,
        description=(
            "Maximum concurrent predictions (1 for single model protection in K8s, "
            "0 disables concurrency limiting)"
        ),
        ge=0,
    )
    retry_after_seconds: int = Field(
        default=5,
        description=(
            "Value of the Retry-After header (seconds) sent with 503 responses "
            "when the prediction concurrency limit is reached"
        ),
        ge=0,
    )
    # Response format configuration
    response_format: str = Field(
        default="standard",
        description=(
            "Response format: 'standard' (default), 'custom' (flexible), "
            "or 'passthrough' (unmodified)"
        ),
    )
    response_validation: bool = Field(
        default=True,
        description="Enable response validation (can be disabled for complex custom responses)",
    )
    extract_values: bool = Field(
        default=False, description="For dict responses, extract values into predictions list"
    )
    # Warmup configuration
    warmup_on_start: bool = Field(
        default=True,
        description=(
            "Run a warmup prediction at startup to initialize model internals "
            "and reduce first-request latency"
        ),
    )

    @model_validator(mode="after")
    def _warn_deprecated_response_options(self):
        """Warn (once per config load) about deprecated response options (RFC 0001 D11)."""
        if self.response_format == "custom":
            logger.warning(
                "DeprecationWarning: api.response_format 'custom' is deprecated and "
                "will be removed in 1.0 (RFC 0001 D11); use 'standard' or 'passthrough'"
            )
        if self.extract_values:
            logger.warning(
                "DeprecationWarning: api.extract_values is deprecated and will be "
                "removed in 1.0 (RFC 0001 D11); return the desired structure from "
                "your predictor instead"
            )
        return self

    def get_resolved_feature_order(self, base_path: Optional[Path] = None) -> Optional[list[str]]:
        """Resolve feature_order, loading from file if it's a path.

        Args:
            base_path: Base directory to resolve relative paths from.
                       File paths must resolve within this directory (security).

        Returns:
            List of feature names or None

        Raises:
            ValueError: If file path resolves outside base_path (path traversal attempt)
        """
        # Return cached value if already resolved
        if self._resolved_feature_order is not None:
            return self._resolved_feature_order

        if self.feature_order is None:
            return None

        # If it's already a list, return as-is
        if isinstance(self.feature_order, list):
            self._resolved_feature_order = self.feature_order
            return self.feature_order

        # If it's a string, try to load from file
        if isinstance(self.feature_order, str):
            try:
                # Resolve path relative to base_path if provided
                if base_path:
                    # Resolve to absolute path
                    resolved_base = base_path.resolve()
                    file_path = (base_path / self.feature_order).resolve()

                    # Security: Prevent path traversal attacks
                    # Ensure resolved path is within the base directory
                    try:
                        file_path.relative_to(resolved_base)
                    except ValueError as e:
                        raise ConfigurationError(
                            message=f"Security error: feature_order path '{self.feature_order}' "
                            f"resolves to '{file_path}' which is outside the project directory.",
                            suggestion=(
                                "Use a relative path within your project directory, "
                                "like 'features.json' or 'config/features.json'"
                            ),
                        ) from e
                else:
                    file_path = Path(self.feature_order).resolve()

                # Check if file exists
                if not file_path.exists():
                    logger.warning(
                        f"Feature order file not found: {file_path}. "
                        f"Paths are relative to the config file directory."
                    )
                    return None

                # Load JSON file
                with open(file_path) as f:
                    features = json.load(f)

                # Validate it's a list of strings
                if not isinstance(features, list):
                    raise ValueError(
                        f"Feature order file must contain a JSON array, got {type(features)}"
                    )

                # Validate all items are strings
                non_strings = [i for i, f in enumerate(features) if not isinstance(f, str)]
                if non_strings:
                    raise ValueError(
                        f"Feature order must be a list of strings. "
                        f"Found non-string values at indices: {non_strings[:5]}"
                    )

                logger.info(f"Loaded {len(features)} features from {file_path}")
                self._resolved_feature_order = features
                return features

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse feature order file as JSON: {e}")
                return None
            except (ValueError, ConfigurationError):
                # Re-raise ValueError and ConfigurationError (including our security error)
                raise
            except Exception as e:
                logger.error(f"Failed to load feature order from file: {e}")
                return None

        logger.warning(f"Unexpected feature_order type: {type(self.feature_order)}")
        return None


class BuildConfig(BaseModel):
    """Container build configuration"""

    base_image: str = Field(default=defaults.DEFAULT_BASE_IMAGE, description="Base Docker image")
    registry: Optional[str] = Field(default=None, description="Container registry URL")
    tag_prefix: Optional[str] = Field(default=None, description="Tag prefix for container names")
    include_files: Optional[list[str]] = Field(
        default=None, description="Explicit files to include"
    )
    exclude_patterns: Optional[list[str]] = Field(default=None, description="Patterns to exclude")


class GitHubVariablesConfig(BaseModel):
    """GitHub repository variables configuration for CI/CD workflows."""

    aws_role_arn_var: str = Field(
        default="AWS_RUNNER_ROLE_ARN",
        description=(
            "Name of GitHub repository variable containing AWS IAM role ARN for OIDC authentication"
        ),
    )
    aws_role_arn_value: Optional[str] = Field(
        default=None,
        description=(
            "Direct AWS role ARN value to bake into workflow (alternative to variable, less secure)"
        ),
    )


class ECRConfig(BaseModel):
    """AWS ECR (Elastic Container Registry) configuration."""

    aws_region: str = Field(default="eu-central-1", description="AWS region for ECR")
    registry_id: str = Field(description="AWS account ID (12-digit number)")
    repository_prefix: str = Field(
        default="ml-classifiers", description="Repository prefix for ECR image names"
    )


class RegistryConfig(BaseModel):
    """Container registry configuration for deployments."""

    type: str = Field(
        default="ghcr",
        description=(
            "Registry type: 'ghcr' (GitHub Container Registry) "
            "or 'ecr' (AWS Elastic Container Registry)"
        ),
    )
    url: Optional[str] = Field(
        default=None, description="Container registry URL (for GHCR, default: ghcr.io)"
    )
    namespace: Optional[str] = Field(
        default=None, description="Registry namespace (for GHCR, default: auto-detected from git)"
    )
    ecr: Optional[ECRConfig] = Field(
        default=None, description="ECR-specific configuration (required when type='ecr')"
    )
    github_variables: GitHubVariablesConfig = Field(
        default_factory=GitHubVariablesConfig,
        description="GitHub repository variables configuration",
    )
    push_on_build: bool = Field(
        default=False, description="Automatically push to registry after build"
    )


class DeploymentConfig(BaseModel):
    """Deployment configuration for multi-classifier repositories."""

    strategy: str = Field(
        default="single",
        description="Deployment strategy: 'single' or 'multi' for separate services",
    )
    container_naming: str = Field(
        default="{repository}-{classifier}:{version}", description="Container tag format template"
    )
    git_tag_format: str = Field(
        default="{classifier}-v{version}", description="Git tag format for releases"
    )
    parallel_builds: bool = Field(
        default=True, description="Enable parallel container builds for multiple classifiers"
    )
    registry: RegistryConfig = Field(
        default_factory=RegistryConfig, description="Container registry configuration"
    )
    resource_limits: Optional[dict[str, Any]] = Field(
        default=None, description="Resource limits for Kubernetes deployments"
    )
    health_check: Optional[dict[str, Any]] = Field(
        default=None, description="Health check configuration"
    )


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    predictor: PredictorConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Modern format (mlserver.yaml) - now optional with smart defaults
    classifier: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Classifier metadata (name, version, description). Auto-generated if not provided."
        ),
    )
    model: Optional[dict[str, Any]] = Field(default=None)  # Model metadata dict
    api: Optional[ApiConfig] = Field(
        default=None, description="API configuration. Uses sensible defaults if not provided."
    )
    build: Optional[BuildConfig] = Field(default=None)
    deployment: Optional[DeploymentConfig] = Field(default=None)  # Deployment configuration

    # Internal state
    classifier_metadata_internal: Optional[ClassifierMetadata] = Field(default=None, exclude=True)
    project_path_internal: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _warn_on_unknown_keys(cls, data: Any) -> Any:
        """Warn (but never reject) when the raw config contains unknown keys.

        Catches typos like 'porrt: 9999' that pydantic would otherwise
        silently drop, replacing the intended value with a default.
        """
        _warn_if_legacy_global_config()
        if isinstance(data, dict):
            _warn_unknown_config_keys(data, cls)
        return data

    @field_validator("predictor", mode="before")
    @classmethod
    def _coerce_predictor_string_spec(cls, value: Any) -> Any:
        """Accept the compact string spec ``predictor: "module:ClassName"``.

        RFC 0001 D13: the string form is shorthand for the two-field mapping
        (init_kwargs default to {}). The mapping form keeps working unchanged.
        """
        if isinstance(value, str):
            module, sep, class_name = value.partition(":")
            if not sep or not module.strip() or not class_name.strip():
                raise ValueError(
                    f"Invalid predictor spec '{value}': expected 'module:ClassName' "
                    "(e.g. 'my_predictor:MyPredictor')"
                )
            return {
                "module": module.strip(),
                "class_name": class_name.strip(),
                "init_kwargs": {},
            }
        return value

    def model_post_init(self, __context) -> None:
        """Post-initialization processing for modern config format."""
        # Apply smart defaults for classifier if not provided
        if self.classifier is None:
            # Auto-generate classifier metadata from predictor class name
            class_name = self.predictor.class_name
            # Convert CamelCase to kebab-case for name
            import re

            name = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name).lower()
            # Remove common suffixes like -predictor, -classifier
            for suffix in ["-predictor", "-classifier", "-model"]:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            object.__setattr__(
                self, "classifier", {"name": name or "classifier", "version": "0.1.0"}
            )

        # Apply smart defaults for api if not provided
        if self.api is None:
            object.__setattr__(self, "api", ApiConfig())

        # Convert classifier dict to ClassifierMetadata format
        try:
            from .version import ClassifierMetadata

            self.classifier_metadata_internal = ClassifierMetadata.model_validate(
                {
                    "classifier": self.classifier,
                    "model": self.model or {},
                    "api": self.api.model_dump(),
                }
            )
        except Exception:
            # If validation fails, skip creating classifier metadata
            # This allows tests and minimal configs to work without full metadata
            self.classifier_metadata_internal = None

    def set_project_path(self, project_path: str):
        """Set project path and load classifier metadata if available."""
        self.project_path_internal = project_path
        # Check for mlserver.yaml
        mlserver_file = Path(project_path) / "mlserver.yaml"

        if mlserver_file.exists():
            try:
                self.classifier_metadata_internal = load_classifier_metadata(project_path)
            except Exception:
                # If config file exists but is invalid, silently ignore
                self.classifier_metadata_internal = None

    @property
    def classifier_metadata(self) -> Optional[ClassifierMetadata]:
        """Get classifier metadata if available."""
        return self.classifier_metadata_internal

    @property
    def project_path(self) -> Optional[str]:
        """Get project path."""
        return self.project_path_internal

    def _format_api_title(self, name: str, version: str) -> str:
        """Format API title from name and version."""
        formatted_name = name.replace("-", " ").title()
        return f"{formatted_name} API v{version}"

    def get_api_title(self) -> str:
        """Get API title with version information."""
        name = self.classifier.get("name", "ML Classifier")
        version = self.classifier.get("version", "1.0.0")
        return self._format_api_title(name, version)

    def get_base_path(self) -> str:
        """Get base API path for endpoints."""
        # Unified interface - no version or classifier name in URL
        # Each deployment gets its own unique base URL anyway
        return ""

    def is_endpoint_enabled(self, endpoint: str) -> bool:
        """Check if a specific endpoint is enabled."""
        return self.api.endpoints.get(endpoint, False)
