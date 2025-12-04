"""Unit tests for init_project module."""
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.init_project import (
    sanitize_name,
    generate_mlserver_yaml,
    generate_predictor_skeleton,
    generate_gitignore,
    init_mlserver_project,
    check_project_files,
)


class TestSanitizeName:
    """Test sanitize_name function."""

    def test_simple_name(self):
        """Test simple name sanitization."""
        assert sanitize_name("myproject") == "myproject"

    def test_name_with_hyphens(self):
        """Test name with hyphens converted to underscores."""
        assert sanitize_name("my-project") == "my_project"

    def test_name_with_spaces(self):
        """Test name with spaces converted to underscores."""
        assert sanitize_name("my project") == "my_project"

    def test_name_with_special_chars(self):
        """Test name with special characters removed."""
        assert sanitize_name("my@project!") == "myproject"

    def test_name_starting_with_number(self):
        """Test name starting with number gets underscore prefix."""
        assert sanitize_name("123project") == "_123project"

    def test_name_uppercase(self):
        """Test name is lowercased."""
        assert sanitize_name("MyProject") == "myproject"

    def test_mixed_case_hyphens_special(self):
        """Test complex name with mixed issues."""
        assert sanitize_name("My-Cool_Project!123") == "my_cool_project123"


class TestGenerateMlserverYaml:
    """Test generate_mlserver_yaml function."""

    def test_basic_generation(self):
        """Test basic YAML generation."""
        yaml_content = generate_mlserver_yaml(
            classifier_name="sentiment",
            predictor_module="sentiment_predictor",
            predictor_class="SentimentPredictor"
        )

        # Verify it's valid YAML
        config = yaml.safe_load(yaml_content)

        assert config["classifier"]["name"] == "sentiment"
        assert config["predictor"]["module"] == "sentiment_predictor"
        assert config["predictor"]["class_name"] == "SentimentPredictor"

    def test_with_version(self):
        """Test generation with custom version."""
        yaml_content = generate_mlserver_yaml(
            classifier_name="test",
            predictor_module="test_pred",
            predictor_class="TestPredictor",
            version="2.0.0"
        )

        config = yaml.safe_load(yaml_content)
        assert config["classifier"]["version"] == "2.0.0"
        assert config["model"]["version"] == "2.0.0"

    def test_with_description(self):
        """Test generation with custom description."""
        yaml_content = generate_mlserver_yaml(
            classifier_name="test",
            predictor_module="test_pred",
            predictor_class="TestPredictor",
            description="My custom classifier"
        )

        config = yaml.safe_load(yaml_content)
        assert config["classifier"]["description"] == "My custom classifier"

    def test_default_description(self):
        """Test generation uses default description."""
        yaml_content = generate_mlserver_yaml(
            classifier_name="mymodel",
            predictor_module="mod",
            predictor_class="Cls"
        )

        config = yaml.safe_load(yaml_content)
        assert "mymodel" in config["classifier"]["description"].lower()

    def test_includes_server_config(self):
        """Test generated YAML includes server config."""
        yaml_content = generate_mlserver_yaml(
            classifier_name="test",
            predictor_module="test",
            predictor_class="Test"
        )

        config = yaml.safe_load(yaml_content)
        assert "server" in config
        assert config["server"]["host"] == "0.0.0.0"
        assert config["server"]["port"] == 8000

    def test_includes_observability_config(self):
        """Test generated YAML includes observability config."""
        yaml_content = generate_mlserver_yaml(
            classifier_name="test",
            predictor_module="test",
            predictor_class="Test"
        )

        config = yaml.safe_load(yaml_content)
        assert "observability" in config
        assert config["observability"]["metrics"] is True


class TestGeneratePredictorSkeleton:
    """Test generate_predictor_skeleton function."""

    def test_basic_generation(self):
        """Test basic predictor skeleton generation."""
        code = generate_predictor_skeleton(
            class_name="SentimentPredictor",
            classifier_name="Sentiment Classifier"
        )

        assert "class SentimentPredictor:" in code
        assert "def predict(" in code
        assert "def predict_proba(" in code
        assert "def __init__(" in code

    def test_includes_imports(self):
        """Test skeleton includes necessary imports."""
        code = generate_predictor_skeleton(
            class_name="TestPredictor",
            classifier_name="test"
        )

        assert "import numpy" in code
        assert "from typing import" in code

    def test_includes_records_to_array(self):
        """Test skeleton includes helper method."""
        code = generate_predictor_skeleton(
            class_name="MyPredictor",
            classifier_name="my-classifier"
        )

        assert "_records_to_array" in code

    def test_is_valid_python(self):
        """Test generated code is valid Python."""
        code = generate_predictor_skeleton(
            class_name="ValidPredictor",
            classifier_name="valid"
        )

        # This should not raise a SyntaxError
        compile(code, '<string>', 'exec')


class TestGenerateGitignore:
    """Test generate_gitignore function."""

    def test_basic_generation(self):
        """Test basic gitignore generation."""
        content = generate_gitignore()

        assert "__pycache__/" in content
        assert "*.py[cod]" in content  # Pattern covers *.pyc

    def test_includes_venv(self):
        """Test gitignore includes virtualenv patterns."""
        content = generate_gitignore()

        assert "venv/" in content
        assert ".venv" in content

    def test_includes_ide_patterns(self):
        """Test gitignore includes IDE patterns."""
        content = generate_gitignore()

        assert ".vscode/" in content
        assert ".idea/" in content

    def test_includes_mlserver_specific(self):
        """Test gitignore includes MLServer-specific patterns."""
        content = generate_gitignore()

        assert "coverage.json" in content or "coverage" in content


class TestInitMlserverProject:
    """Test init_mlserver_project function."""

    def test_basic_init(self):
        """Test basic project initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('mlserver.github_actions.init_github_actions') as mock_gh:
                mock_gh.return_value = (True, "Created", {})

                success, message, files = init_mlserver_project(
                    project_path=tmpdir,
                    classifier_name="test-classifier",
                    include_github_actions=True
                )

                assert success is True
                assert "mlserver.yaml" in files
                assert (Path(tmpdir) / "mlserver.yaml").exists()

    def test_init_without_github_actions(self):
        """Test init without GitHub Actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            success, message, files = init_mlserver_project(
                project_path=tmpdir,
                classifier_name="test",
                include_github_actions=False
            )

            assert success is True
            assert (Path(tmpdir) / "mlserver.yaml").exists()
            # Should not create GitHub workflow
            assert not (Path(tmpdir) / ".github" / "workflows").exists()

    def test_init_creates_predictor_file(self):
        """Test init creates predictor Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            success, message, files = init_mlserver_project(
                project_path=tmpdir,
                classifier_name="mymodel",
                include_github_actions=False
            )

            assert success is True
            # Check predictor file was created
            predictor_file = Path(tmpdir) / "mymodel_predictor.py"
            assert predictor_file.exists()

    def test_init_creates_gitignore(self):
        """Test init creates .gitignore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            success, message, files = init_mlserver_project(
                project_path=tmpdir,
                classifier_name="test",
                include_github_actions=False
            )

            assert success is True
            assert (Path(tmpdir) / ".gitignore").exists()

    def test_init_doesnt_overwrite_without_force(self):
        """Test init doesn't overwrite existing files without force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing mlserver.yaml
            existing_yaml = Path(tmpdir) / "mlserver.yaml"
            existing_yaml.write_text("original: content")

            success, message, files = init_mlserver_project(
                project_path=tmpdir,
                classifier_name="test",
                include_github_actions=False,
                force=False
            )

            # Should succeed but not overwrite
            assert success is True
            assert existing_yaml.read_text() == "original: content"

    def test_init_overwrites_with_force(self):
        """Test init overwrites existing files with force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing mlserver.yaml
            existing_yaml = Path(tmpdir) / "mlserver.yaml"
            existing_yaml.write_text("original: content")

            success, message, files = init_mlserver_project(
                project_path=tmpdir,
                classifier_name="test",
                include_github_actions=False,
                force=True
            )

            # Should have overwritten
            assert success is True
            content = existing_yaml.read_text()
            assert content != "original: content"
            assert "classifier:" in content

    def test_init_with_custom_predictor_names(self):
        """Test init with custom predictor file and class names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            success, message, files = init_mlserver_project(
                project_path=tmpdir,
                classifier_name="mymodel",
                predictor_file="custom_predictor",
                predictor_class="CustomClass",
                include_github_actions=False
            )

            assert success is True

            # Check mlserver.yaml has correct predictor config
            with open(Path(tmpdir) / "mlserver.yaml") as f:
                config = yaml.safe_load(f)

            assert config["predictor"]["module"] == "custom_predictor"
            assert config["predictor"]["class_name"] == "CustomClass"

    def test_init_invalid_path(self):
        """Test init with invalid path."""
        success, message, files = init_mlserver_project(
            project_path="/nonexistent/path/that/does/not/exist"
        )

        assert success is False
        assert "does not exist" in message or "not a directory" in message

    def test_init_derives_classifier_name_from_directory(self):
        """Test init derives classifier name from directory name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory with specific name
            project_dir = Path(tmpdir) / "my-cool-classifier"
            project_dir.mkdir()

            success, message, files = init_mlserver_project(
                project_path=str(project_dir),
                include_github_actions=False
            )

            assert success is True

            with open(project_dir / "mlserver.yaml") as f:
                config = yaml.safe_load(f)

            assert config["classifier"]["name"] == "my-cool-classifier"


class TestCheckProjectFiles:
    """Test check_project_files function."""

    def test_all_files_present(self):
        """Test returns True when all files present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required files
            mlserver_yaml = Path(tmpdir) / "mlserver.yaml"
            mlserver_yaml.write_text("""
predictor:
  module: my_predictor
  class_name: MyPredictor
""")
            predictor_py = Path(tmpdir) / "my_predictor.py"
            predictor_py.write_text("class MyPredictor: pass")

            workflows_dir = Path(tmpdir) / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)
            workflow_file = workflows_dir / "ml-classifier-container-build.yml"
            workflow_file.write_text("name: Build")

            all_present, missing = check_project_files(tmpdir)

            assert all_present is True
            assert len(missing) == 0

    def test_missing_mlserver_yaml(self):
        """Test returns missing when mlserver.yaml not present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            all_present, missing = check_project_files(tmpdir)

            assert all_present is False
            assert "mlserver.yaml" in missing

    def test_missing_predictor_file(self):
        """Test returns missing when predictor file not present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlserver_yaml = Path(tmpdir) / "mlserver.yaml"
            mlserver_yaml.write_text("""
predictor:
  module: my_predictor
  class_name: MyPredictor
""")

            all_present, missing = check_project_files(tmpdir)

            assert all_present is False
            assert any("my_predictor.py" in m for m in missing)

    def test_missing_github_workflow(self):
        """Test returns missing when GitHub workflow not present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlserver_yaml = Path(tmpdir) / "mlserver.yaml"
            mlserver_yaml.write_text("""
predictor:
  module: my_predictor
  class_name: MyPredictor
""")
            predictor_py = Path(tmpdir) / "my_predictor.py"
            predictor_py.write_text("class MyPredictor: pass")

            all_present, missing = check_project_files(tmpdir)

            assert all_present is False
            assert any("ml-classifier-container-build.yml" in m for m in missing)

    def test_multi_classifier_config(self):
        """Test handles multi-classifier config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlserver_yaml = Path(tmpdir) / "mlserver.yaml"
            mlserver_yaml.write_text("""
classifiers:
  sentiment:
    predictor:
      module: sentiment_predictor
      class_name: SentimentPredictor
  fraud:
    predictor:
      module: fraud_predictor
      class_name: FraudPredictor
""")
            # Create one predictor file
            sentiment_py = Path(tmpdir) / "sentiment_predictor.py"
            sentiment_py.write_text("class SentimentPredictor: pass")

            # Check specific classifier
            all_present, missing = check_project_files(tmpdir, classifier_name="sentiment")

            # Should find sentiment predictor exists
            assert all("sentiment" not in m.lower() for m in missing if "predictor" in m)

    def test_invalid_yaml(self):
        """Test handles invalid YAML gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlserver_yaml = Path(tmpdir) / "mlserver.yaml"
            mlserver_yaml.write_text("invalid: yaml: :")

            # Should not crash
            all_present, missing = check_project_files(tmpdir)

            # mlserver.yaml exists but might not be parseable
            assert isinstance(all_present, bool)


class TestInitProjectEdgeCases:
    """Test edge cases in init_project."""

    def test_sanitize_empty_name(self):
        """Test sanitizing empty name."""
        result = sanitize_name("")
        assert result == ""

    def test_sanitize_only_special_chars(self):
        """Test sanitizing name with only special chars."""
        result = sanitize_name("@#$%")
        assert result == ""

    def test_init_existing_predictor_not_overwritten(self):
        """Test existing predictor file is not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing predictor
            predictor_file = Path(tmpdir) / "test_predictor.py"
            predictor_file.write_text("# existing code")

            success, message, files = init_mlserver_project(
                project_path=tmpdir,
                classifier_name="test",
                include_github_actions=False
            )

            # Predictor should not be in files created
            assert "predictor" not in files
            # Content should be preserved
            assert predictor_file.read_text() == "# existing code"
