"""Unit tests for predictor_loader module."""
import pytest
import tempfile
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.predictor_loader import (
    _validate_model_files,
    resolve_module_path,
    load_predictor,
)
from mlserver.errors import PredictorError


class TestValidateModelFiles:
    """Test model file validation."""

    def test_no_file_params(self):
        """Test validation with no file parameters."""
        # Should not raise
        _validate_model_files({})
        _validate_model_files({"some_param": "value"})

    def test_nonexistent_file(self):
        """Test validation with nonexistent file path."""
        # Should not raise - file doesn't exist
        _validate_model_files({"model_path": "/nonexistent/path/model.pkl"})

    def test_small_model_file(self):
        """Test validation with small model file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"small model content")
            f.flush()
            try:
                # Should not raise
                _validate_model_files({"model_path": f.name})
            finally:
                os.unlink(f.name)

    def test_model_file_warning_for_large_file(self):
        """Test warning for large (>100MB) but acceptable files."""
        # We can't easily create a 100MB+ file in tests, so we'll mock
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=150 * 1024 * 1024):  # 150MB
                with patch('mlserver.predictor_loader.logger') as mock_logger:
                    _validate_model_files({"model_path": "/path/to/model.pkl"})
                    mock_logger.warning.assert_called_once()
                    assert "150.0MB" in str(mock_logger.warning.call_args)

    def test_model_file_too_large(self):
        """Test error for file exceeding 2GB limit."""
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=2500 * 1024 * 1024):  # 2.5GB
                with pytest.raises(PredictorError) as exc_info:
                    _validate_model_files({"model_path": "/path/to/huge_model.pkl"})
                assert "too large" in str(exc_info.value)

    def test_multiple_file_params(self):
        """Test validation with multiple file parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            preprocessor_path = Path(tmpdir) / "preprocessor.pkl"
            model_path.write_bytes(b"model")
            preprocessor_path.write_bytes(b"preprocessor")

            # Should not raise
            _validate_model_files({
                "model_path": str(model_path),
                "preprocessor_path": str(preprocessor_path),
            })

    def test_non_string_file_param(self):
        """Test that non-string file params are ignored."""
        # Should not raise - non-string values are skipped
        _validate_model_files({"model_path": 123})
        _validate_model_files({"model_path": None})


class TestResolveModulePath:
    """Test module path resolution."""

    def test_full_module_path_unchanged(self):
        """Test that full module paths are returned unchanged."""
        result = resolve_module_path("examples.predictor_catboost")
        assert result == "examples.predictor_catboost"

    def test_full_path_with_multiple_dots(self):
        """Test module paths with multiple dots."""
        result = resolve_module_path("my.package.subpackage.predictor")
        assert result == "my.package.subpackage.predictor"

    def test_simple_name_without_config_dir(self):
        """Test simple name without config directory."""
        result = resolve_module_path("predictor")
        assert result == "predictor"

    def test_py_extension_removed(self):
        """Test that .py extension is handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a predictor file
            predictor_file = Path(tmpdir) / "my_predictor.py"
            predictor_file.write_text("class MyPredictor: pass")

            result = resolve_module_path("my_predictor.py", tmpdir)
            assert result == "my_predictor"

    def test_local_module_resolution(self):
        """Test resolution of local module in config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a predictor file
            predictor_file = Path(tmpdir) / "test_local_predictor.py"
            predictor_file.write_text("class TestLocalPredictor: pass")

            result = resolve_module_path("test_local_predictor", tmpdir)
            assert result == "test_local_predictor"
            # Config dir should be in sys.path
            assert str(Path(tmpdir).resolve()) in sys.path

            # Cleanup
            sys.path.remove(str(Path(tmpdir).resolve()))
            if "test_local_predictor" in sys.modules:
                del sys.modules["test_local_predictor"]

    def test_module_not_found_fallback(self):
        """Test fallback when module file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_module_path("nonexistent_module", tmpdir)
            assert result == "nonexistent_module"


class TestLoadPredictor:
    """Test the load_predictor function."""

    def test_load_builtin_module_class(self):
        """Test loading a class from a built-in module."""
        # Load collections.OrderedDict as a simple test
        result = load_predictor("collections", "OrderedDict", {})
        assert result is not None

    def test_load_with_init_kwargs(self):
        """Test loading with initialization kwargs."""
        # Load a dict subclass with initial data
        result = load_predictor("collections", "OrderedDict", {"a": 1, "b": 2})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_load_nonexistent_module(self):
        """Test error when module doesn't exist."""
        with pytest.raises(PredictorError) as exc_info:
            load_predictor("nonexistent_module_xyz", "SomeClass", {})
        assert "Failed to import module" in str(exc_info.value)

    def test_load_nonexistent_class(self):
        """Test error when class doesn't exist in module."""
        with pytest.raises(PredictorError) as exc_info:
            load_predictor("collections", "NonexistentClass", {})
        assert "not found in module" in str(exc_info.value)

    def test_load_local_predictor(self):
        """Test loading a predictor from local file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a predictor file
            predictor_file = Path(tmpdir) / "local_predictor.py"
            predictor_file.write_text("""
class LocalPredictor:
    def __init__(self, value=42):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)
""")

            # Load the predictor
            predictor = load_predictor(
                "local_predictor",
                "LocalPredictor",
                {"value": 100},
                config_dir=tmpdir
            )

            assert predictor.value == 100
            assert predictor.predict([1, 2, 3]) == [100, 100, 100]

            # Cleanup
            if str(Path(tmpdir).resolve()) in sys.path:
                sys.path.remove(str(Path(tmpdir).resolve()))
            if "local_predictor" in sys.modules:
                del sys.modules["local_predictor"]

    def test_load_with_model_file_validation(self):
        """Test that model file validation is called."""
        with patch('mlserver.predictor_loader._validate_model_files') as mock_validate:
            try:
                load_predictor("collections", "OrderedDict", {"model_path": "/test"})
            except Exception:
                pass
            mock_validate.assert_called_once_with({"model_path": "/test"})

    def test_error_message_suggestions(self):
        """Test that error messages include helpful suggestions."""
        with pytest.raises(PredictorError) as exc_info:
            load_predictor("simple_module", "Class", {}, config_dir="/some/dir")

        error = exc_info.value
        assert error.suggestion is not None
        # Should mention trying filename
        assert "simple_module.py" in error.suggestion or "Ensure" in error.suggestion


class TestModulePathEdgeCases:
    """Test edge cases in module path resolution."""

    def test_module_with_py_suffix_not_file(self):
        """Test module name ending in .py but not a file path."""
        # e.g., "module.py" where py is a subpackage
        result = resolve_module_path("module.py", None)
        # Without config_dir, should try as-is
        assert "module" in result

    def test_reimport_cached_module(self):
        """Test that cached modules are reimported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor_file = Path(tmpdir) / "cached_predictor.py"
            predictor_file.write_text("class Pred: val = 1")

            # First load
            resolve_module_path("cached_predictor", tmpdir)

            # Modify the file
            predictor_file.write_text("class Pred: val = 2")

            # Should reimport
            result = resolve_module_path("cached_predictor", tmpdir)
            assert result == "cached_predictor"

            # Cleanup
            if str(Path(tmpdir).resolve()) in sys.path:
                sys.path.remove(str(Path(tmpdir).resolve()))
            if "cached_predictor" in sys.modules:
                del sys.modules["cached_predictor"]


class TestPredictorLoaderIntegration:
    """Integration tests for predictor loading."""

    def test_full_workflow(self):
        """Test complete predictor loading workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model file
            model_file = Path(tmpdir) / "model.pkl"
            model_file.write_bytes(b"fake model data")

            # Create predictor
            predictor_file = Path(tmpdir) / "predictor.py"
            predictor_file.write_text("""
class TestPredictor:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, X):
        return [1] * len(X)
""")

            # Load it
            predictor = load_predictor(
                module="predictor",
                class_name="TestPredictor",
                init_kwargs={"model_path": str(model_file)},
                config_dir=tmpdir
            )

            assert predictor.model_path == str(model_file)
            assert predictor.predict([1, 2]) == [1, 1]

            # Cleanup
            if str(Path(tmpdir).resolve()) in sys.path:
                sys.path.remove(str(Path(tmpdir).resolve()))
            if "predictor" in sys.modules:
                del sys.modules["predictor"]
