"""Package-wide default values.

Replaces the removed GlobalSettings singleton (RFC 0001, D12): plain constants,
overridable per deployment via mlserver.yaml or the documented environment
variables. No file is read at import time.
"""

import os

# Server defaults (mlserver.yaml `server.*` overrides these per project)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_WORKERS = 1

# Container build defaults (mlserver.yaml `build.*` overrides these)
DEFAULT_BASE_IMAGE = "python:3.11-slim"
CONTAINER_TEMP_DIR = "/tmp"
CONTAINER_TEST_PORT = 8012
CONTAINER_EXCLUDED_PATTERNS = [
    ".vscode", ".idea", "*.log", "*.tmp", "catboost_info", "*.swp",
    ".git", ".gitignore", "__pycache__", "*.pyc", ".pytest_cache",
    ".coverage", "htmlcov", ".DS_Store", "node_modules", ".env",
    "venv", ".venv",
]


def default_host() -> str:
    """Server host default, overridable via MLSERVER_DEFAULT_HOST."""
    return os.getenv("MLSERVER_DEFAULT_HOST", DEFAULT_HOST)


def default_port() -> int:
    """Server port default, overridable via MLSERVER_DEFAULT_PORT."""
    value = os.getenv("MLSERVER_DEFAULT_PORT")
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return DEFAULT_PORT


def default_log_level() -> str:
    """Log level default, overridable via MLSERVER_LOG_LEVEL."""
    return os.getenv("MLSERVER_LOG_LEVEL", DEFAULT_LOG_LEVEL)
