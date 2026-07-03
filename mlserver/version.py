"""
Version management utilities for ML server classifier projects.
"""
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .errors import ConfigurationError


@dataclass
class GitInfo:
    """Git repository information."""
    tag: Optional[str]
    commit: str
    branch: str
    is_dirty: bool


class ClassifierVersion(BaseModel):
    """Classifier version metadata schema."""
    repository: Optional[str] = Field(
        None, description="Repository name (auto-detected if not provided)"
    )
    name: str = Field(..., description="Classifier name (used in URLs)")
    version: Optional[str] = Field(
        "1.0.0", description="Semantic version (auto-detected from git tags if not provided)"
    )
    description: str = Field("", description="Classifier description")

    @field_validator('repository')
    @classmethod
    def validate_repository(cls, v):
        if v is None:
            return v
        # Ensure repository name is valid for Docker tags
        if not re.match(r'^[a-z0-9][a-z0-9-_.]*$', v):
            raise ValueError(
                'Repository must start with alphanumeric and contain only lowercase '
                'letters, numbers, hyphens, underscores, and periods'
            )
        return v

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        # Ensure name is URL-safe (allow underscores for compatibility)
        if not re.match(r'^[a-z0-9][a-z0-9_-]*$', v):
            raise ValueError(
                'Name must start with lowercase letter or number, and contain only '
                'lowercase letters, numbers, underscores, and hyphens'
            )
        return v

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        if v is None:
            return "1.0.0"  # Default version
        # Validate semantic versioning
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError('Version must follow semantic versioning (e.g., 1.2.3)')
        return v


class ModelVersion(BaseModel):
    """Model artifact version metadata."""
    version: Optional[str] = Field(
        "1.0.0", description="Model version (auto-detected from git tags if not provided)"
    )
    trained_at: Optional[str] = Field(None, description="Training timestamp (ISO format)")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Model performance metrics"
    )


class ApiVersion(BaseModel):
    """API versioning configuration."""
    version: str = Field("v1", description="API version (v1, v2, etc.)")
    endpoints: dict[str, bool] = Field(
        default_factory=lambda: {
            "predict": True,
            # Note: batch_predict removed - /predict handles both single and batch
            "predict_proba": True
        },
        description="Enabled endpoints"
    )

    @field_validator('version')
    @classmethod
    def validate_api_version(cls, v):
        if not re.match(r'^v\d+$', v):
            raise ValueError('API version must be in format v1, v2, etc.')
        return v


class ClassifierMetadata(BaseModel):
    """Complete classifier project metadata."""
    classifier: ClassifierVersion
    model: ModelVersion
    api: ApiVersion = Field(default_factory=ApiVersion)
    build: dict[str, Any] = Field(default_factory=dict, description="Build configuration")


def get_git_info(project_path: str) -> Optional[GitInfo]:
    """Get git information for the project."""
    try:
        original_cwd = os.getcwd()
        os.chdir(project_path)

        # Get current commit
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()

        # Get current branch
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()

        # Check if repository is dirty
        status = subprocess.check_output(['git', 'status', '--porcelain'],
                                       stderr=subprocess.DEVNULL).decode().strip()
        is_dirty = len(status) > 0

        # Get latest tag (if any)
        try:
            tag = subprocess.check_output(['git', 'describe', '--tags', '--exact-match', 'HEAD'],
                                        stderr=subprocess.DEVNULL).decode().strip()
        except subprocess.CalledProcessError:
            tag = None

        return GitInfo(tag=tag, commit=commit[:8], branch=branch, is_dirty=is_dirty)

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    finally:
        os.chdir(original_cwd)


def get_repository_name(project_path: str = ".") -> str:
    """Get repository name from Git remote or directory name."""
    try:
        # Try to get remote URL
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=project_path,
            stderr=subprocess.DEVNULL
        ).decode().strip()

        if remote_url:
            # Extract repository name from URL
            # Handle both HTTPS and SSH URLs
            match = re.search(r'[/:]([^/]+?)(?:\.git)?$', remote_url)
            if match:
                return match.group(1).lower()
    except Exception:
        pass

    # Fallback to directory name
    return os.path.basename(os.path.abspath(project_path)).lower()


def load_classifier_metadata(project_path: str) -> ClassifierMetadata:
    """Load classifier metadata from mlserver.yaml file."""
    mlserver_file = Path(project_path) / "mlserver.yaml"

    if not mlserver_file.exists():
        raise ConfigurationError(
            message=f"mlserver.yaml not found in {project_path}",
            suggestion=(
                "Run 'mlserver init' to create a new project, "
                "or check you're in the correct directory"
            )
        )

    try:
        with open(mlserver_file) as f:
            full_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            message=f"Invalid YAML syntax in mlserver.yaml: {e}",
            suggestion="Check your mlserver.yaml for syntax errors (indentation, colons, quotes)"
        ) from e

    # Extract classifier metadata from unified config
    if 'classifier' not in full_config:
        raise ConfigurationError(
            message="mlserver.yaml missing required 'classifier' section",
            suggestion="Add a 'classifier:' section with 'name:' and 'version:' fields"
        )

    metadata_dict = {
        'classifier': full_config.get('classifier', {}),
        'model': full_config.get('model', {}),
        'api': full_config.get('api', {}),
        'build': full_config.get('build', {})
    }
    return ClassifierMetadata.model_validate(metadata_dict)


def save_classifier_metadata(metadata: ClassifierMetadata, project_path: str):
    """Save classifier metadata to mlserver.yaml file."""
    mlserver_file = Path(project_path) / "mlserver.yaml"

    with open(mlserver_file, 'w') as f:
        yaml.dump(
            metadata.model_dump(exclude_unset=True), f,
            default_flow_style=False, sort_keys=False
        )


def generate_container_tags(metadata: ClassifierMetadata, git_info: Optional[GitInfo] = None,
                          predictor_class: Optional[str] = None, project_path: str = ".",
                          classifier_name: Optional[str] = None) -> list[str]:
    """Generate container tags for the classifier using hierarchical naming."""
    # Use the repository/directory name as the base
    repository = get_repository_name(project_path)

    # For multi-classifier setups, use <repo>/<classifier>:version format
    if classifier_name:
        # Version should already be from git tag (e.g., "0.1.0")
        version = metadata.classifier.version

        # Image name includes classifier: repo/classifier
        image_name = f"{repository}/{classifier_name}"

        # Handle missing git tag case
        if version == "missing-git-tag":
            tags = [
                f"{image_name}:latest",  # Latest for this specific classifier
                f"{image_name}:untagged",  # Clear indicator that tagging is needed
            ]

            if git_info and git_info.commit:
                # Add commit hash for identification
                tags.append(f"{image_name}:untagged-{git_info.commit[:7]}")
        else:
            tags = [
                f"{image_name}:latest",  # Latest for this specific classifier
                f"{image_name}:v{version}",  # Version tag
            ]

            if git_info and git_info.commit:
                # Add commit-based tag for exact reproducibility
                tags.append(f"{image_name}:v{version}-{git_info.commit[:7]}")
    else:
        # Single classifier - simpler naming
        version = metadata.classifier.version

        tags = [
            f"{repository}:latest",
            f"{repository}:v{version}",
        ]

        if git_info and git_info.commit:
            tags.append(f"{repository}:v{version}-{git_info.commit[:7]}")

    return tags


def validate_version_consistency(
    metadata: ClassifierMetadata, project_path: str
) -> dict[str, str]:
    """Validate version-related git state for the project.

    Per RFC 0001 (D3), git tags are the canonical version source and the
    config's ``classifier.version`` is display-only, so a difference between
    the two is NO LONGER reported as an issue. When such a difference exists
    it is logged informationally (the tag wins everywhere that matters).

    Only genuine problems are returned as issues:
    - git_dirty: uncommitted changes in the working directory
    """
    # Imported here to avoid any risk of an import cycle with version_control
    from .version_control import parse_classifier_tag

    issues = {}

    git_info = get_git_info(project_path)

    # Informational only (D3): config version differing from the tag is
    # expected during normal workflows and must not fail validation.
    if git_info and git_info.tag:
        parsed = parse_classifier_tag(git_info.tag)
        if (parsed and parsed["classifier"] == metadata.classifier.name
                and parsed["version"] != metadata.classifier.version):
            import logging
            logging.getLogger(__name__).info(
                "Git tag '%s' version differs from config classifier.version '%s'; "
                "the git tag is canonical (RFC 0001 D3) and config version is "
                "display-only",
                git_info.tag, metadata.classifier.version,
            )

    # Check if working directory is dirty
    if git_info and git_info.is_dirty:
        issues['git_dirty'] = "Working directory has uncommitted changes"

    return issues


def get_version_info(
    project_path: str = ".", classifier_name: Optional[str] = None
) -> dict[str, Any]:
    """Get comprehensive version information for a classifier project.

    Args:
        project_path: Path to the project directory
        classifier_name: For multi-classifier configs, specify which classifier to get info for.
                        If None and multi-classifier, returns info about all classifiers.
    """
    try:
        mlserver_config_path = os.path.join(project_path, "mlserver.yaml")

        if not os.path.exists(mlserver_config_path):
            return {"error": f"mlserver.yaml not found in {project_path}"}

        from .config import AppConfig

        with open(mlserver_config_path) as f:
            raw_config = yaml.safe_load(f)

        # Check if this is a multi-classifier config
        is_multi_classifier = "classifiers" in raw_config

        if is_multi_classifier:
            from .multi_classifier import (
                extract_single_classifier_config,
                load_multi_classifier_config,
            )

            multi_config = load_multi_classifier_config(mlserver_config_path)
            available_classifiers = list(multi_config.classifiers.keys())

            if classifier_name:
                # Get info for specific classifier
                if classifier_name not in multi_config.classifiers:
                    return {
                        "error": (
                            f"Classifier '{classifier_name}' not found. "
                            f"Available: {available_classifiers}"
                        )
                    }

                config = extract_single_classifier_config(multi_config, classifier_name)
            else:
                # Return summary of all classifiers
                git_info = get_git_info(project_path)
                classifiers_info = []
                for name in available_classifiers:
                    clf_config = multi_config.classifiers[name]
                    clf_meta = clf_config.get("classifier", clf_config.get("metadata", {}))
                    classifiers_info.append({
                        "name": name,
                        "version": clf_meta.get("version", "unknown"),
                        "description": clf_meta.get("description", "")
                    })

                return {
                    "multi_classifier": True,
                    "classifiers": classifiers_info,
                    "default_classifier": multi_config.default_classifier or (
                        available_classifiers[0] if available_classifiers else None
                    ),
                    "git": git_info.__dict__ if git_info else None,
                    "timestamp": datetime.now().isoformat(),
                    "config_source": "mlserver.yaml"
                }
        else:
            config = AppConfig.model_validate(raw_config)

        if not config.classifier:
            return {"error": "mlserver.yaml missing required 'classifier' section"}

        if isinstance(config.classifier, dict):
            metadata = ClassifierMetadata.model_validate({
                "classifier": config.classifier,
                "model": config.model or {},
                "api": config.api.model_dump() if config.api else {}
            })
        else:
            metadata = config.classifier

        git_info = get_git_info(project_path)
        issues = validate_version_consistency(metadata, project_path)

        return {
            "classifier": metadata.classifier.model_dump(),
            "model": metadata.model.model_dump(),
            "api": metadata.api.model_dump(),
            "git": git_info.__dict__ if git_info else None,
            "container_tags": generate_container_tags(metadata, git_info),
            "validation_issues": issues,
            "timestamp": datetime.now().isoformat(),
            "config_source": "mlserver.yaml"
        }

    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to get version info: {str(e)}"}
