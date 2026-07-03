"""
Auto-detection utilities for metadata and configuration.

This module provides automatic detection of:
- Git repository information (name, commit, tag, branch)
- Deployment timestamps
- MLServer package version
- Project name from git or directory
"""

import importlib.metadata
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def get_git_project_name(project_path: str = ".") -> Optional[str]:
    """
    Auto-detect project name from git repository.

    Returns:
        Repository name from git remote origin, or None if not a git repo
    """
    try:
        # Try to get remote URL
        remote_url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=project_path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        if remote_url:
            # Extract repository name from URL
            # Handle both HTTPS and SSH URLs
            # https://github.com/user/repo.git -> repo
            # git@github.com:user/repo.git -> repo
            match = re.search(r"[/:]([^/]+?)(?:\.git)?$", remote_url)
            if match:
                return match.group(1).lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def get_project_name(project_path: str = ".") -> str:
    """
    Get project name from git or fallback to directory name.

    Returns:
        Project name (guaranteed to return a value)
    """
    # Try git first
    git_name = get_git_project_name(project_path)
    if git_name:
        return git_name

    # Fallback to directory name
    return os.path.basename(os.path.abspath(project_path)).lower()


def get_git_info(project_path: str = ".") -> dict[str, Any]:
    """
    Get comprehensive git information.

    Returns:
        Dict with repository, commit, tag, branch, dirty status
    """
    info = {
        "repository": get_git_project_name(project_path),
        "commit": None,
        "tag": None,
        "branch": None,
        "dirty": False,
    }

    # Check environment variables first (for containerized deployments)
    env_commit = os.environ.get("MLSERVER_GIT_COMMIT")
    env_tag = os.environ.get("MLSERVER_GIT_TAG")
    env_branch = os.environ.get("MLSERVER_GIT_BRANCH")

    if env_commit or env_tag or env_branch:
        # Running in container with embedded metadata
        if env_commit:
            info["commit"] = env_commit
        if env_tag:
            info["tag"] = env_tag
        if env_branch:
            info["branch"] = env_branch
        # In container, we can't determine if dirty
        info["dirty"] = False
        return info

    try:
        # Use cwd= for subprocess calls instead of os.chdir: chdir is
        # process-wide and races with concurrent cwd-relative path resolution
        # in a threaded server.

        # Get current commit (short hash)
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=project_path, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        # Get current branch
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=project_path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        # Check if working directory is dirty
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_path, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["dirty"] = len(status) > 0

        # Get tag at current commit (if any)
        try:
            info["tag"] = (
                subprocess.check_output(
                    ["git", "describe", "--tags", "--exact-match", "HEAD"],
                    cwd=project_path,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            # No tag at current commit
            pass

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repository or git not installed
        pass

    return info


def get_deployed_timestamp() -> str:
    """
    Generate ISO format deployment timestamp.

    Returns:
        Current time in ISO 8601 format with timezone
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def get_mlserver_package_version() -> str:
    """
    Get installed mlserver package version.

    Returns:
        Package version or 'unknown' if not found
    """
    # Try explicit distribution names only (no fuzzy matching, which could
    # pick up unrelated packages like Seldon's 'mlserver'). 'merve' is the
    # current distribution name (RFC 0001 D9); the older names are kept for
    # installs that predate the rename.
    possible_names = [
        "merve",
        "mlserver-fastapi-wrapper",
        "mlserver_fastapi_wrapper",
        "mlserver",
        "ml-server",
        "ml_server",
    ]

    for name in possible_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue

    # Fallback: version file generated by setuptools-scm (development mode)
    try:
        from mlserver._version import __version__

        return __version__
    except ImportError:
        pass

    return "unknown"


def get_mlserver_git_info() -> dict[str, Any]:
    """
    Get git information for the MLServer package itself.

    Returns:
        Dict with mlserver's git commit, tag, and comprehensive version info
    """
    info = {
        "package_version": get_mlserver_package_version(),
        "api_commit": None,
        "api_tag": None,
        "api_branch": None,
        "api_dirty": False,
    }

    # Development installs: prefer live git over the build-time _version_info
    # snapshot, which is generated at package build and can be stale.
    mlserver_dir = Path(__file__).parent.parent
    if (mlserver_dir / ".git").exists():
        git_data = get_git_info(str(mlserver_dir))
        if git_data["commit"]:
            info["api_commit"] = git_data["commit"]
            info["api_tag"] = git_data["tag"]
            info["api_branch"] = git_data["branch"]
            info["api_dirty"] = git_data["dirty"]
            return info

    # Wheel installs: use version info embedded at package build time
    try:
        from mlserver import _version_info

        if hasattr(_version_info, "GIT_COMMIT") and _version_info.GIT_COMMIT:
            info["api_commit"] = _version_info.GIT_COMMIT
        if hasattr(_version_info, "GIT_TAG") and _version_info.GIT_TAG:
            info["api_tag"] = _version_info.GIT_TAG
        if hasattr(_version_info, "GIT_BRANCH") and _version_info.GIT_BRANCH:
            info["api_branch"] = _version_info.GIT_BRANCH
        if hasattr(_version_info, "GIT_DIRTY"):
            info["api_dirty"] = _version_info.GIT_DIRTY

        # If we found embedded info, use it
        if info["api_commit"]:
            return info
    except ImportError:
        pass

    # Check environment variables (for containerized deployments)
    env_api_commit = os.environ.get("MLSERVER_API_COMMIT")
    env_api_tag = os.environ.get("MLSERVER_API_TAG")
    env_api_branch = os.environ.get("MLSERVER_API_BRANCH")

    if env_api_commit or env_api_tag or env_api_branch:
        # Running in container with embedded MLServer metadata
        if env_api_commit:
            info["api_commit"] = env_api_commit
        if env_api_tag:
            info["api_tag"] = env_api_tag
        if env_api_branch:
            info["api_branch"] = env_api_branch
        info["api_dirty"] = False

    return info


def generate_simplified_metadata(config: dict[str, Any], project_path: str = ".") -> dict[str, Any]:
    """
    Generate simplified metadata structure from config and auto-detection.

    Args:
        config: The loaded configuration dictionary
        project_path: Path to the project

    Returns:
        Simplified metadata dict for responses
    """
    git_info = get_git_info(project_path)

    # Extract classifier info
    classifier_config = config.get("classifier", {})

    metadata = {
        "project": git_info.get("repository") or get_project_name(project_path),
        "classifier": classifier_config.get("name", "unknown"),
        "description": classifier_config.get("description", ""),
        "git_commit": git_info.get("commit"),
        "git_tag": git_info.get("tag"),
        "git_branch": git_info.get("branch"),
        "git_dirty": git_info.get("dirty", False),
        "deployed_at": get_deployed_timestamp(),
        "mlserver_version": get_mlserver_package_version(),
    }

    # Remove None values for cleaner output
    return {k: v for k, v in metadata.items() if v is not None}


def get_simplified_info_response(
    config: dict[str, Any], predictor_class_name: str, project_path: str = "."
) -> dict[str, Any]:
    """
    Generate simplified /info endpoint response.

    Args:
        config: The loaded configuration dictionary
        predictor_class_name: Name of the predictor class
        project_path: Path to the project

    Returns:
        Simplified info response dict
    """
    git_info = get_git_info(project_path)
    mlserver_info = get_mlserver_git_info()

    classifier_config = config.get("classifier", {})

    return {
        "project": git_info.get("repository") or get_project_name(project_path),
        "classifier": classifier_config.get("name", "unknown"),
        "description": classifier_config.get("description", ""),
        "predictor_class": predictor_class_name,
        "deployed_at": get_deployed_timestamp(),
        "classifier_repository": {
            "repository": git_info.get("repository"),
            "commit": git_info.get("commit"),
            "tag": git_info.get("tag"),
            "branch": git_info.get("branch"),
            "dirty": git_info.get("dirty"),
        },
        "api_service": mlserver_info,
        "endpoints": {
            "predict": "/predict",
            "predict_proba": "/predict_proba",
            "info": "/info",
            "health": "/healthz",
            "metrics": "/metrics",
        },
    }
