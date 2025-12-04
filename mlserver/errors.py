"""
Unified error hierarchy for MLServer.

All MLServer errors inherit from MLServerError, which provides:
- Consistent error formatting
- Actionable suggestions for resolution
- Links to relevant documentation

Usage:
    from mlserver.errors import ConfigurationError, PredictorError

    raise ConfigurationError(
        message="Invalid model path",
        suggestion="Check that the model file exists and is readable",
        docs_url="https://docs.example.com/configuration"
    )
"""

from __future__ import annotations
from typing import Optional


class MLServerError(Exception):
    """Base exception for all MLServer errors.

    All errors include:
    - message: What went wrong
    - suggestion: How to fix it (optional)
    - docs_url: Link to relevant documentation (optional)

    Example:
        raise MLServerError(
            "Model failed to load",
            suggestion="Ensure the model file exists and is compatible",
            docs_url="https://docs.example.com/models"
        )
    """

    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        docs_url: Optional[str] = None
    ):
        self.message = message
        self.suggestion = suggestion
        self.docs_url = docs_url
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with suggestion and docs link."""
        parts = [self.message]
        if self.suggestion:
            parts.append(f"\n\nTry this: {self.suggestion}")
        if self.docs_url:
            parts.append(f"\n\nDocumentation: {self.docs_url}")
        return "".join(parts)


class ConfigurationError(MLServerError):
    """Raised when configuration is invalid or missing.

    Examples:
    - Missing required fields in mlserver.yaml
    - Invalid field values (port out of range, etc.)
    - File not found (config file, feature order file)
    - Path traversal attempts
    """
    pass


class PredictorError(MLServerError):
    """Raised when predictor loading or execution fails.

    Examples:
    - Module not found
    - Class not found in module
    - Predictor initialization failed
    - Model warmup failed
    - Prediction method crashed
    """
    pass


class AdapterError(MLServerError):
    """Raised when input/output data conversion fails.

    Examples:
    - Invalid payload format
    - Missing required features
    - Feature type mismatch
    - Too many records (exceeds limits)
    """
    pass


class ContainerError(MLServerError):
    """Raised when container operations fail.

    Examples:
    - Docker not installed or not running
    - Build failed
    - Push to registry failed
    - Invalid registry configuration
    """
    pass


class ValidationError(MLServerError):
    """Raised when validation fails.

    Examples:
    - Invalid mlserver.yaml syntax
    - Missing required files
    - Version format invalid
    - Git repository not initialized
    """
    pass


class VersionControlError(MLServerError):
    """Raised when version control operations fail.

    Examples:
    - Git repository not found
    - Dirty working directory
    - Tag already exists
    - Remote push failed
    """
    pass


# Utility function for creating user-friendly error messages
def format_error_for_cli(error: MLServerError) -> str:
    """Format an MLServerError for CLI display with colors.

    Args:
        error: The MLServerError to format

    Returns:
        Formatted string suitable for Rich console output
    """
    parts = [f"[red]✗ {error.message}[/red]"]

    if error.suggestion:
        parts.append(f"\n[yellow]→ Try this:[/yellow] {error.suggestion}")

    if error.docs_url:
        parts.append(f"\n[blue]📚 Docs:[/blue] {error.docs_url}")

    return "".join(parts)
