"""Unit tests for the errors module."""
import pytest
from mlserver.errors import (
    MLServerError,
    ConfigurationError,
    PredictorError,
    AdapterError,
    ContainerError,
    ValidationError,
    VersionControlError,
    format_error_for_cli,
)


class TestMLServerError:
    """Test the base MLServerError class."""

    def test_error_message_only(self):
        """Test error with just a message."""
        error = MLServerError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.suggestion is None
        assert error.docs_url is None

    def test_error_with_suggestion(self):
        """Test error with message and suggestion."""
        error = MLServerError(
            "Model failed to load",
            suggestion="Check the model file path"
        )
        assert "Model failed to load" in str(error)
        assert "Try this: Check the model file path" in str(error)
        assert error.suggestion == "Check the model file path"

    def test_error_with_docs_url(self):
        """Test error with message and docs URL."""
        error = MLServerError(
            "Invalid configuration",
            docs_url="https://example.com/docs"
        )
        assert "Invalid configuration" in str(error)
        assert "Documentation: https://example.com/docs" in str(error)
        assert error.docs_url == "https://example.com/docs"

    def test_error_with_all_fields(self):
        """Test error with message, suggestion, and docs URL."""
        error = MLServerError(
            "Configuration error",
            suggestion="Fix your config file",
            docs_url="https://example.com/config"
        )
        error_str = str(error)
        assert "Configuration error" in error_str
        assert "Try this: Fix your config file" in error_str
        assert "Documentation: https://example.com/config" in error_str

    def test_error_is_exception(self):
        """Test that MLServerError is a proper Exception."""
        with pytest.raises(MLServerError) as exc_info:
            raise MLServerError("Test error")
        assert str(exc_info.value) == "Test error"


class TestSpecificErrors:
    """Test specific error subclasses."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Missing required field",
            suggestion="Add 'predictor' section to mlserver.yaml"
        )
        assert isinstance(error, MLServerError)
        assert "Missing required field" in str(error)

    def test_predictor_error(self):
        """Test PredictorError."""
        error = PredictorError(
            "Module not found: my_predictor",
            suggestion="Check the module name in config"
        )
        assert isinstance(error, MLServerError)
        assert "Module not found" in str(error)

    def test_adapter_error(self):
        """Test AdapterError."""
        error = AdapterError(
            "Invalid payload format",
            suggestion="Use 'records' or 'ndarray' format"
        )
        assert isinstance(error, MLServerError)
        assert "Invalid payload format" in str(error)

    def test_container_error(self):
        """Test ContainerError."""
        error = ContainerError(
            "Docker not available",
            suggestion="Install and start Docker"
        )
        assert isinstance(error, MLServerError)
        assert "Docker not available" in str(error)

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Invalid YAML syntax",
            suggestion="Check line 15 for syntax errors"
        )
        assert isinstance(error, MLServerError)
        assert "Invalid YAML syntax" in str(error)

    def test_version_control_error(self):
        """Test VersionControlError."""
        error = VersionControlError(
            "Git repository not found",
            suggestion="Run 'git init' to initialize"
        )
        assert isinstance(error, MLServerError)
        assert "Git repository not found" in str(error)


class TestFormatErrorForCli:
    """Test the format_error_for_cli utility function."""

    def test_format_message_only(self):
        """Test formatting with just a message."""
        error = MLServerError("Something went wrong")
        formatted = format_error_for_cli(error)
        assert "[red]✗ Something went wrong[/red]" in formatted

    def test_format_with_suggestion(self):
        """Test formatting with suggestion."""
        error = MLServerError(
            "Model failed",
            suggestion="Check the path"
        )
        formatted = format_error_for_cli(error)
        assert "[red]✗ Model failed[/red]" in formatted
        assert "[yellow]→ Try this:[/yellow] Check the path" in formatted

    def test_format_with_docs_url(self):
        """Test formatting with docs URL."""
        error = MLServerError(
            "Config error",
            docs_url="https://example.com/docs"
        )
        formatted = format_error_for_cli(error)
        assert "[red]✗ Config error[/red]" in formatted
        assert "[blue]📚 Docs:[/blue] https://example.com/docs" in formatted

    def test_format_with_all_fields(self):
        """Test formatting with all fields."""
        error = MLServerError(
            "Full error",
            suggestion="Try fixing it",
            docs_url="https://example.com"
        )
        formatted = format_error_for_cli(error)
        assert "[red]✗ Full error[/red]" in formatted
        assert "[yellow]→ Try this:[/yellow] Try fixing it" in formatted
        assert "[blue]📚 Docs:[/blue] https://example.com" in formatted


class TestErrorInheritance:
    """Test error inheritance chain."""

    def test_all_errors_are_exceptions(self):
        """Test all error types are proper exceptions."""
        error_classes = [
            ConfigurationError,
            PredictorError,
            AdapterError,
            ContainerError,
            ValidationError,
            VersionControlError,
        ]
        for cls in error_classes:
            assert issubclass(cls, Exception)
            assert issubclass(cls, MLServerError)

    def test_errors_can_be_caught_as_mlserver_error(self):
        """Test all error types can be caught as MLServerError."""
        with pytest.raises(MLServerError):
            raise ConfigurationError("test")

        with pytest.raises(MLServerError):
            raise PredictorError("test")

        with pytest.raises(MLServerError):
            raise AdapterError("test")

    def test_errors_preserve_message_attribute(self):
        """Test error classes preserve the message attribute."""
        error = ConfigurationError("Test message", suggestion="Test suggestion")
        assert error.message == "Test message"
        assert error.suggestion == "Test suggestion"
