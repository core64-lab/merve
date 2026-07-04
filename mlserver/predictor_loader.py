"""Predictor loading with isolated file-based imports (RFC 0001, D13).

File-based predictor modules (a ``.py`` file next to the config) are imported
with :func:`importlib.util.spec_from_file_location` and registered in
``sys.modules`` under the ``merve._user.<name>`` namespace, and never delete
foreign ``sys.modules`` entries — so a predictor file named ``json.py`` or
``types.py`` loads correctly without shadowing the stdlib. The project
directory is APPENDED to ``sys.path`` (never front-inserted; D13 amendment
after a live-smoke regression) so the predictor's own sibling imports
(``import src.features``) resolve while stdlib/installed packages keep
precedence. Dotted module specs (installed packages) are imported normally
via :func:`importlib.import_module`.
"""

import importlib.util
import logging
import os
import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

from .errors import PredictorError

logger = logging.getLogger(__name__)

# Namespace under which file-based user predictor modules are registered in
# sys.modules. Keeps user module names from colliding with stdlib/installed
# packages (RFC 0001, D13).
USER_MODULE_NAMESPACE = "merve._user"


def _validate_model_files(init_kwargs: dict) -> None:
    """Validate model file sizes before loading to prevent memory issues."""
    file_size_limit_mb = 2000  # 2GB limit

    # Common model file parameters to check
    file_params = ["model_path", "preprocessor_path", "weights_path", "checkpoint_path"]

    for param in file_params:
        if param in init_kwargs:
            file_path = init_kwargs[param]
            if isinstance(file_path, str) and os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                size_mb = size_bytes / (1024 * 1024)

                if size_mb > file_size_limit_mb:
                    raise PredictorError(
                        message=(
                            f"Model file {file_path} is too large: "
                            f"{size_mb:.1f}MB (limit: {file_size_limit_mb}MB)"
                        ),
                        suggestion=(
                            "Consider using a smaller model, compressing the "
                            "model file, or increase system resources"
                        ),
                    )
                elif size_mb > 100:  # Warn for files > 100MB
                    logger.warning(f"Loading large model file: {file_path} ({size_mb:.1f}MB)")


def resolve_module_path(module_spec: str, config_dir: Optional[str] = None) -> str:
    """Resolve a module spec to an importable module name.

    Pure name resolution - performs no imports and never mutates
    ``sys.path``/``sys.modules`` (the actual import happens in
    :func:`load_predictor`).

    Handles three cases:
    1. Full module path (e.g., 'examples.predictor_catboost') - use as-is
    2. Relative path with .py (e.g., 'predictor_catboost.py') - resolve relative to config
    3. Simple name (e.g., 'predictor_catboost') - try to resolve relative to config

    Args:
        module_spec: Module specification (can be full path, filename, or simple name)
        config_dir: Directory containing the configuration file

    Returns:
        Module name to import (file-based names load from the config directory)
    """
    # Case 1: Already a full module path (contains dots but not .py extension)
    if "." in module_spec and not module_spec.endswith(".py"):
        return module_spec

    # Case 2 & 3: Simple name or filename - resolve relative to the config dir.
    # Remove .py extension if present (suffix only - a '.py' mid-name like
    # 'my.pyfile' must not be mangled)
    module_name = module_spec.removesuffix(".py")
    if config_dir and (Path(config_dir) / f"{module_name}.py").exists():
        logger.info(
            f"Resolved module '{module_spec}' to '{module_name}' "
            f"(file-based import from {config_dir})"
        )
        return module_name

    # Fallback: return as-is and let the import fail with a clear error
    logger.warning(f"Could not resolve module '{module_spec}', using as-is")
    return module_spec


def _find_local_module_file(module_spec: str, config_dir: Optional[str]) -> Optional[Path]:
    """Return the .py file for a config-dir-relative module spec, if one exists.

    Dotted specs (e.g. 'mypackage.predictor') always target installed
    packages, never local files.
    """
    if not config_dir:
        return None
    if "." in module_spec and not module_spec.endswith(".py"):
        return None
    module_name = module_spec.removesuffix(".py")
    candidate = Path(config_dir) / f"{module_name}.py"
    if candidate.exists():
        return candidate.resolve()
    return None


def _ensure_user_namespace() -> None:
    """Ensure synthetic parent packages for ``merve._user.*`` exist in sys.modules.

    This keeps ``sys.modules[cls.__module__]`` lookups (pickle, dataclasses,
    some serializers) working for the parent packages of namespaced user
    modules. Existing real packages are never replaced.
    """
    parent_name = None
    for part in USER_MODULE_NAMESPACE.split("."):
        name = f"{parent_name}.{part}" if parent_name else part
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # mark as package (no importable submodule paths)
            sys.modules[name] = package
        parent_name = name


def load_module_from_file(module_file: Path, module_name: str) -> types.ModuleType:
    """Import ``module_file`` as ``merve._user.<module_name>``.

    Uses ``spec_from_file_location`` so nothing is added to ``sys.path`` and
    no existing ``sys.modules`` entry (stdlib or installed) is shadowed or
    deleted. Calling this again for the same name re-executes the file and
    replaces only the namespaced entry.
    """
    qualified_name = f"{USER_MODULE_NAMESPACE}.{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, module_file)
    if spec is None or spec.loader is None:
        raise PredictorError(
            message=f"Could not create an import spec for '{module_file}'",
            suggestion="Check that the predictor file is a readable Python (.py) file",
        )

    module = importlib.util.module_from_spec(spec)
    _ensure_user_namespace()
    # Register before exec_module so lookups of sys.modules[cls.__module__]
    # made during module execution (decorators, dataclasses, ...) resolve.
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Only our namespaced entry is removed - never a user/stdlib name.
        sys.modules.pop(qualified_name, None)
        raise
    setattr(sys.modules[USER_MODULE_NAMESPACE], module_name, module)
    logger.info(f"Loaded predictor module '{module_name}' from {module_file} as '{qualified_name}'")
    return module


def _ensure_config_dir_importable(config_dir: Optional[str]) -> None:
    """Make the project directory importable for the predictor's own imports.

    The predictor entry file is loaded in isolation (``spec_from_file_location``
    under ``merve._user.*``) so its own module name can never shadow stdlib.
    But the predictor still needs to import its sibling packages/modules
    (e.g. ``import src.features``), which live in the project directory. We
    *append* that directory to ``sys.path`` — never insert it at the front — so
    stdlib and installed packages keep precedence and a stray local ``types.py``
    or ``json.py`` cannot shadow them.
    """
    if not config_dir:
        return
    resolved = str(Path(config_dir).resolve())
    if resolved not in sys.path:
        sys.path.append(resolved)
        logger.debug(f"Appended '{resolved}' to sys.path for predictor imports")


def _import_predictor_module(module: str, config_dir: Optional[str]) -> types.ModuleType:
    """Import the predictor module for a spec, file-based or installed."""
    module_file = _find_local_module_file(module, config_dir)
    if module_file is not None:
        # The predictor's transitive imports (sibling packages) resolve against
        # the project directory.
        _ensure_config_dir_importable(config_dir)
        try:
            return load_module_from_file(module_file, module_file.stem)
        except PredictorError:
            raise
        except Exception as e:
            raise PredictorError(
                message=f"Failed to load predictor module from '{module_file}': {e}",
                suggestion=(
                    "The predictor file is executed on load - check it for "
                    "import-time errors and missing dependencies"
                ),
            ) from e

    resolved_module = resolve_module_path(module, config_dir)
    try:
        return import_module(resolved_module)
    except ImportError as e:
        # Provide helpful error message with suggestion
        suggestion_parts = []
        if "." not in module:
            suggestion_parts.append(f"Try using just the filename: '{module}.py'")
            suggestion_parts.append("Or specify the full module path (e.g., 'mypackage.predictor')")
        if config_dir:
            suggestion_parts.append(f"Ensure the predictor file exists in: {config_dir}")

        raise PredictorError(
            message=f"Failed to import module '{resolved_module}' (original: '{module}')",
            suggestion=" | ".join(suggestion_parts) if suggestion_parts else None,
        ) from e


def load_predictor(
    module: str, class_name: str, init_kwargs: dict, config_dir: Optional[str] = None
) -> Any:
    """Load a predictor class with intelligent module resolution.

    Args:
        module: Module specification (can be full path, filename, or simple name)
        class_name: Name of the predictor class
        init_kwargs: Keyword arguments for the predictor constructor
        config_dir: Directory containing the configuration file (for relative imports)
    """
    # Validate model file sizes before loading
    _validate_model_files(init_kwargs or {})

    # Import the module (file-based specs are isolated under merve._user.*)
    mod = _import_predictor_module(module, config_dir)

    # Get the class from the module
    try:
        cls = getattr(mod, class_name)
    except AttributeError as e:
        raise PredictorError(
            message=f"Class '{class_name}' not found in module '{module}'",
            suggestion=(
                "Check that the class name in mlserver.yaml matches the class "
                "defined in your predictor file"
            ),
        ) from e

    # Instantiate the predictor
    return cls(**(init_kwargs or {}))
