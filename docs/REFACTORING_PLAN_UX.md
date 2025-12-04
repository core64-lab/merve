# MLServer Refactoring Plan

**Goal**: Transform merve from a "works if you know what you're doing" tool into an intuitive, self-documenting experience that guides users from installation to production deployment.

**Philosophy**: Convention over configuration. Fail fast with helpful messages. Guide, don't gatekeep.

---

## Executive Summary

### Current State
- 25+ configuration options spread across nested YAML structures
- Error messages expose raw Pydantic/Python exceptions
- No validation without starting the server
- Path resolution is confusing and undocumented
- Silent failures in multiple places
- No IDE support (no JSON schema)
- Flag-heavy CLI with no interactive fallbacks
- Critical bugs: thread safety issues, cache collisions, missing imports

### Target State
- Minimal viable config: 3 lines of YAML
- Every error includes "Try this:" suggestions
- `mlserver validate` catches issues before runtime
- Clear path resolution with helpful warnings
- Fail fast with actionable messages
- Full IDE autocomplete via JSON schema
- Interactive prompts when required info is missing
- Rock-solid core: thread-safe, validated, warm-started

---

## Phase 0: Critical Bug Fixes (IMMEDIATE)

> These are production-blocking issues that must be fixed before any UX work.

### 0.1 Thread Safety for Metrics Singleton

**Problem**: Global `_metrics_collector` can race in multi-worker scenarios.

**File**: `mlserver/metrics.py:104-117`

```python
# Current (UNSAFE)
_metrics_collector: Optional[MetricsCollector] = None

def init_metrics(model_name: str) -> MetricsCollector:
    global _metrics_collector
    _metrics_collector = MetricsCollector(model_name)  # Race condition!
    return _metrics_collector
```

**Fix**:
```python
import threading

_metrics_lock = threading.Lock()
_metrics_collector: Optional[MetricsCollector] = None

def init_metrics(model_name: str) -> MetricsCollector:
    global _metrics_collector
    with _metrics_lock:
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector(model_name)
        return _metrics_collector

def get_metrics() -> Optional[MetricsCollector]:
    with _metrics_lock:
        return _metrics_collector
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P0       | 1h     | SWE   |

---

### 0.2 Fix Directory Traversal in Feature Order

**Problem**: `feature_order` file path not validated, allows `../../etc/passwd`.

**File**: `mlserver/config.py:138-141`

```python
# Current (VULNERABLE)
if base_path:
    file_path = base_path / self.feature_order
else:
    file_path = Path(self.feature_order)
```

**Fix**:
```python
def get_resolved_feature_order(self, base_path: Optional[Path] = None) -> Optional[List[str]]:
    if self._resolved_feature_order is not None:
        return self._resolved_feature_order

    if self.feature_order is None:
        return None

    if isinstance(self.feature_order, list):
        self._resolved_feature_order = self.feature_order
        return self._resolved_feature_order

    # It's a file path - resolve and VALIDATE
    if base_path:
        file_path = (base_path / self.feature_order).resolve()
        # Security: ensure resolved path is within base_path
        try:
            file_path.relative_to(base_path.resolve())
        except ValueError:
            raise ValueError(
                f"feature_order path '{self.feature_order}' resolves outside project directory. "
                f"Path traversal is not allowed."
            )
    else:
        file_path = Path(self.feature_order).resolve()

    # ... rest of loading logic
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P0       | 2h     | SWE   |

---

### 0.3 Add Model Warmup Mechanism

**Problem**: First prediction is slow (cold start), no warmup during server startup.

**File**: `mlserver/server.py` - add to lifespan context manager

```python
# Current: No warmup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... setup ...
    yield
    # ... cleanup ...
```

**Fix**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing setup ...

    # Warmup: Run dummy prediction to initialize model internals
    if config.api.warmup_on_start:
        logger.info("Warming up model...")
        try:
            warmup_start = time.perf_counter()
            # Create minimal warmup data based on feature_order
            warmup_data = _create_warmup_data(config)
            if warmup_data is not None:
                _ = predictor_wrapper.predict(warmup_data)
                warmup_duration = time.perf_counter() - warmup_start
                logger.info(f"Model warmup complete in {warmup_duration:.2f}s")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-fatal): {e}")

    yield
    # ... cleanup ...

def _create_warmup_data(config: AppConfig) -> Optional[np.ndarray]:
    """Create minimal warmup data for model initialization."""
    feature_order = config.api.get_resolved_feature_order()
    if feature_order:
        # Create single row of zeros
        return np.zeros((1, len(feature_order)))
    return None
```

**Config addition**:
```yaml
api:
  warmup_on_start: true  # Default: true
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P0       | 4h     | MLE   |

---

### 0.4 Fix Cache Collision for Heterogeneous Records

**Problem**: Feature order cache uses only first record's keys, causing wrong results if records have different schemas.

**File**: `mlserver/adapters.py:33`

```python
# Current (BUG)
first_record_features = frozenset(records[0].keys())  # Only checks first!
```

**Fix**:
```python
def _get_feature_order_cached(records: List[Dict]) -> List[str]:
    """Get or compute feature order with proper cache key."""
    # Use ALL unique features across all records for cache key
    all_features = set()
    for record in records:
        all_features.update(record.keys())

    cache_key = frozenset(all_features)

    if cache_key in _feature_order_cache:
        # Move to end (MRU)
        _feature_order_cache.move_to_end(cache_key)
        return _feature_order_cache[cache_key]

    # Compute: Use first-seen order, not sorted
    seen = []
    seen_set = set()
    for record in records:
        for key in record.keys():
            if key not in seen_set:
                seen.append(key)
                seen_set.add(key)

    # Cache management
    if len(_feature_order_cache) >= MAX_CACHE_SIZE:
        _feature_order_cache.popitem(last=False)

    _feature_order_cache[cache_key] = seen
    return seen
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P0       | 2h     | MLE   |

---

### 0.5 Add Missing Logger Import in `_to_jsonable`

**Problem**: `logger.warning()` called but logger not defined - runtime crash.

**File**: `mlserver/server.py:645`

```python
# Current (CRASH)
def _to_jsonable(obj, depth=0, max_depth=50):
    # ...
    logger.warning(...)  # NameError: logger not defined
```

**Fix**: Add at top of function or use module logger
```python
import logging

def _to_jsonable(obj, depth=0, max_depth=50):
    logger = logging.getLogger(__name__)
    # ... rest of function
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P0       | 5m     | SWE   |

---

### Phase 0 Summary

| Task | File | Priority | Effort | Owner |
|------|------|----------|--------|-------|
| Thread safety for metrics singleton | `metrics.py` | P0 | 1h | SWE |
| Fix directory traversal in feature_order | `config.py` | P0 | 2h | SWE |
| Add model warmup mechanism | `server.py` | P0 | 4h | MLE |
| Fix cache collision for heterogeneous records | `adapters.py` | P0 | 2h | MLE |
| Add logger import in `_to_jsonable` | `server.py` | P0 | 5m | SWE |

**Total Phase 0 Effort**: ~9 hours

---

## Phase 1: API & Code Quality Improvements

> Foundational improvements that make the codebase more maintainable and the API more intuitive.

### 1.1 Create Unified Error Hierarchy

**Problem**: Inconsistent error handling - sometimes silent, sometimes raw exceptions, sometimes typer.Exit.

**New file**: `mlserver/errors.py`

```python
class MLServerError(Exception):
    """Base exception for all MLServer errors."""
    def __init__(self, message: str, suggestion: str = None, docs_url: str = None):
        self.message = message
        self.suggestion = suggestion
        self.docs_url = docs_url
        super().__init__(message)

class ConfigurationError(MLServerError):
    """Invalid or missing configuration."""
    pass

class PredictorError(MLServerError):
    """Error loading or running predictor."""
    pass

class AdapterError(MLServerError):
    """Error converting input/output data."""
    pass

class ContainerError(MLServerError):
    """Error building or pushing containers."""
    pass

class ValidationError(MLServerError):
    """Validation failed."""
    pass
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P1       | 4h     | SWE   |

---

### 1.2 ~~Simplify Payload Structure~~ → SKIPPED: Flexible I/O Pattern Already Implemented

**Decision**: SKIP this task. The current flexible I/O pattern is intentional and matches real-world usage patterns (see `trade-likelihood` project).

**Rationale**:
The existing payload structure `{"payload": {"records": [...]}}` provides flexibility that is actively used in production:

1. **Flexible Input Formats**: Supports `records`, `instances`, `ndarray`, `features` keys
2. **Response Format Options**: Three modes cover all use cases:
   - `response_format: "standard"` - Traditional `{"predictions": [...], "time_ms": ...}`
   - `response_format: "custom"` - Arbitrary structures via `{"result": {...}, "predictions": [...]}`
   - `response_format: "passthrough"` - Return predictor output unchanged (no wrapper)

3. **Real-World Pattern** (from `trade-likelihood` project):
   - Predictor returns complex structured dicts with explanations, metadata, predictions
   - Using `response_format: "custom"` or `"passthrough"` preserves this structure
   - No need for Google AI Platform compatibility - internal deployment patterns differ

**Current Implementation** (`mlserver/schemas.py`):
```python
class PredictRequest(BaseModel):
    payload: Dict[str, Any] = Field(...)  # Fully flexible input

class CustomPredictResponse(BaseModel):
    result: Any  # Arbitrary JSON-serializable structure
    predictions: Optional[List[Any]]  # Optional extracted predictions
    time_ms: float
    metadata: Optional[ClassifierMetadataResponse]
```

**Configuration** (`mlserver.yaml`):
```yaml
api:
  adapter: "records"  # or "ndarray" or "auto"
  response_format: "custom"  # or "standard" or "passthrough"
  response_validation: true  # disable for complex custom responses
  extract_values: false  # extract dict values to list
```

| Status | Decision |
|--------|----------|
| SKIPPED | Use existing flexible I/O pattern |

---

### 1.3 Add Input Schema Validation from Feature Order

**Problem**: Feature mismatches discovered at prediction time, not config time.

**File**: `mlserver/validation.py`

```python
class FeatureSchemaValidator:
    """Validate input features match expected schema."""

    def __init__(self, feature_order: List[str]):
        self.feature_order = set(feature_order)
        self.feature_list = feature_order

    def validate(self, records: List[Dict]) -> ValidationResult:
        """Validate records have expected features."""
        errors = []

        for i, record in enumerate(records):
            record_features = set(record.keys())

            missing = self.feature_order - record_features
            extra = record_features - self.feature_order

            if missing:
                errors.append(f"Record {i}: missing features {sorted(missing)}")
            if extra:
                # Warning, not error - extra features are ignored
                pass

        if errors:
            return ValidationResult(
                passed=False,
                name="Feature validation",
                message=f"Found {len(errors)} records with missing features",
                details={"errors": errors[:5]}  # First 5
            )

        return ValidationResult(passed=True, name="Feature validation")
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P1       | 4h     | MLE   |

---

### 1.4 Document Response Formats in OpenAPI

**Problem**: Response schema varies by config but OpenAPI doesn't reflect this.

**File**: `mlserver/server.py`

```python
def _create_predict_endpoint(app: FastAPI, config: AppConfig, ...):
    # Dynamic response model based on config
    if config.api.response_format == "standard":
        response_model = PredictResponse
        response_example = {
            "predictions": [0, 1, 0],
            "timing": {"inference_ms": 12.5}
        }
    elif config.api.response_format == "custom":
        response_model = CustomPredictResponse
        response_example = {
            "result": {"predictions": [0, 1, 0]},
            "metadata": {"model": "v1.0.0"}
        }
    else:
        response_model = None  # Passthrough
        response_example = None

    @app.post(
        endpoint_path,
        response_model=response_model,
        responses={
            200: {"description": "Successful prediction", "content": {"application/json": {"example": response_example}}},
            400: {"description": "Invalid input format"},
            503: {"description": "Model not ready or overloaded"}
        },
        summary=f"Predict using {config.classifier.get('name', 'model')}",
        description=f"Make predictions using the {config.classifier.get('name', 'model')} classifier."
    )
    def predict(req: PredictRequest):
        ...
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P1       | 2h     | SWE   |

---

### 1.5 Consolidate CLI to Single Module

**Problem**: `cli.py` and `cli_v2.py` confusion, unclear which is active.

**Action**:
1. Verify `cli.py` is the active one (entry point in pyproject.toml)
2. Remove or rename `cli_v2.py` if obsolete
3. Add deprecation note if keeping for compatibility

**File**: Check and update `pyproject.toml`

```toml
[project.scripts]
mlserver = "mlserver.cli:main"  # Confirm this is correct
```

| Priority | Effort | Owner |
|----------|--------|-------|
| P1       | 4h     | SWE   |

---

### Phase 1 Summary

| Task | File | Priority | Effort | Owner |
|------|------|----------|--------|-------|
| Create unified error hierarchy | `errors.py` (NEW) | P1 | 4h | SWE |
| Simplify payload structure | `schemas.py`, `adapters.py` | P1 | 8h | SWE |
| Add input schema validation | `validation.py` | P1 | 4h | MLE |
| Document response formats in OpenAPI | `server.py` | P1 | 2h | SWE |
| Consolidate CLI to single module | `cli.py` | P1 | 4h | SWE |

**Total Phase 1 Effort**: ~22 hours

---

## Phase 2: Configuration Simplification

### 2.1 Reduce Required Fields to Minimum

**Current minimum config (verbose):**
```yaml
predictor:
  module: "my_predictor"
  class_name: "MyPredictor"
  init_kwargs:
    model_path: "./model.pkl"

classifier:
  name: "my-classifier"

api: {}
```

**Target minimum config:**
```yaml
predictor:
  module: "my_predictor"
  class_name: "MyPredictor"
```

**Changes Required:**

| Change | File | Description |
|--------|------|-------------|
| Make `classifier.name` optional | `config.py` | Default to directory name or module name |
| Make `api` section optional | `config.py` | Already has defaults, just allow omission |
| Auto-detect model files | `predictor_loader.py` | If `init_kwargs` empty, scan for common model files |

**Implementation:**

```python
# config.py - Add smart defaults
class AppConfig(BaseModel):
    @model_validator(mode='after')
    def set_smart_defaults(self):
        # Default classifier name from predictor module
        if not self.classifier or not self.classifier.get('name'):
            self.classifier = self.classifier or {}
            self.classifier['name'] = self.predictor.module.replace('_', '-')
        return self
```

### 2.2 Convention-Based Predictor Discovery

**Goal**: If user has a single Python file with a class containing `predict()`, use it automatically.

**New behavior for `mlserver init`:**
```
$ mlserver init

Scanning directory for predictors...
Found: predictor.py with class TitanicPredictor

Generated mlserver.yaml:
  predictor:
    module: "predictor"
    class_name: "TitanicPredictor"

Continue with this configuration? [Y/n]
```

**Implementation:**

```python
# New file: mlserver/discovery.py

def discover_predictor(project_path: Path) -> Optional[Dict]:
    """Auto-discover predictor class in project directory."""
    candidates = []

    for py_file in project_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        # Parse AST to find classes with predict method
        classes = find_predictor_classes(py_file)
        candidates.extend(classes)

    if len(candidates) == 1:
        return {
            "module": candidates[0].module,
            "class_name": candidates[0].class_name,
            "confidence": "high"
        }
    elif len(candidates) > 1:
        return {
            "candidates": candidates,
            "confidence": "low"
        }
    return None
```

### 2.3 Model File Auto-Detection

**Goal**: Automatically find model files if not specified.

**Supported patterns:**
- `*.pkl`, `*.joblib` - Scikit-learn
- `*.pt`, `*.pth` - PyTorch
- `*.h5`, `*.keras` - Keras/TensorFlow
- `*.onnx` - ONNX
- `model/` directory - Any framework

**Implementation:**

```python
# predictor_loader.py

MODEL_PATTERNS = [
    "*.pkl", "*.joblib",  # sklearn
    "*.pt", "*.pth",       # pytorch
    "*.h5", "*.keras",     # keras
    "*.onnx",              # onnx
    "model/**/*",          # model directory
]

def suggest_model_files(project_path: Path) -> List[Path]:
    """Find potential model files in project."""
    found = []
    for pattern in MODEL_PATTERNS:
        found.extend(project_path.glob(pattern))
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)
```

---

## Phase 3: Error Message Revolution

### 3.1 Create Error Message Framework

**Current error:**
```
Error: 1 validation error for AppConfig
predictor.module
  Field required [type=missing]
```

**Target error:**
```
Configuration Error: Missing required field

  predictor.module is required but not found in mlserver.yaml

  Add this to your configuration:

    predictor:
      module: "your_predictor_module"
      class_name: "YourPredictorClass"

  Tip: Run 'mlserver init' to generate a starter configuration
```

**Implementation:**

```python
# New file: mlserver/errors.py

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class MLServerError:
    """Structured error with context and suggestions."""
    title: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    example: Optional[str] = None
    docs_url: Optional[str] = None

ERROR_TEMPLATES = {
    "missing_predictor_module": MLServerError(
        title="Missing required field",
        message="predictor.module is required but not found in mlserver.yaml",
        suggestion="Add this to your configuration:",
        example="""
predictor:
  module: "your_predictor_module"
  class_name: "YourPredictorClass"
""",
        docs_url="https://github.com/core64-lab/merve#quick-start"
    ),

    "invalid_port": MLServerError(
        title="Invalid port number",
        message="Port must be between 1 and 65535",
        suggestion="Use a valid port number:",
        example="""
server:
  port: 8000  # Common choices: 8000, 8080, 3000
"""
    ),

    "module_not_found": MLServerError(
        title="Cannot import predictor module",
        message="The specified module could not be imported",
        suggestion="Check that:\n  1. The file exists\n  2. It's a valid Python module\n  3. All dependencies are installed",
        example="python -c 'import {module}'"
    ),
}

def format_error(error: MLServerError, **kwargs) -> str:
    """Format error for console output."""
    lines = [
        f"[red]✗[/red] [bold]{error.title}[/bold]",
        "",
        f"  {error.message.format(**kwargs)}",
    ]

    if error.suggestion:
        lines.extend(["", f"  [cyan]{error.suggestion}[/cyan]"])

    if error.example:
        example = error.example.format(**kwargs)
        lines.extend(["", "  [dim]" + example.strip() + "[/dim]"])

    if error.docs_url:
        lines.extend(["", f"  [blue]More info: {error.docs_url}[/blue]"])

    return "\n".join(lines)
```

### 3.2 Wrap All Validation Errors

**File**: `mlserver/cli.py`

```python
def handle_config_error(e: Exception, config_path: Path) -> None:
    """Transform raw exceptions into helpful messages."""
    from .errors import ERROR_TEMPLATES, format_error, MLServerError

    if isinstance(e, ValidationError):
        for error in e.errors():
            loc = ".".join(str(x) for x in error['loc'])
            error_type = error['type']

            # Map to custom error template
            template_key = f"{loc.replace('.', '_')}_{error_type}"
            if template_key in ERROR_TEMPLATES:
                console.print(format_error(ERROR_TEMPLATES[template_key]))
            else:
                # Fallback with generic but helpful message
                console.print(format_error(MLServerError(
                    title=f"Configuration error in '{loc}'",
                    message=error['msg'],
                    suggestion=f"Check the '{loc}' section in {config_path}"
                )))

    elif isinstance(e, ImportError):
        module = str(e).split("'")[1] if "'" in str(e) else "unknown"
        console.print(format_error(
            ERROR_TEMPLATES["module_not_found"],
            module=module
        ))

    else:
        # Unknown error - still helpful
        console.print(format_error(MLServerError(
            title="Unexpected error",
            message=str(e),
            suggestion="Run with --log-level DEBUG for more details"
        )))
```

### 3.3 Add "Did You Mean?" for Typos

```python
# mlserver/errors.py

from difflib import get_close_matches

VALID_TOP_LEVEL_KEYS = ["server", "predictor", "classifier", "api", "observability", "build", "deployment"]
VALID_SERVER_KEYS = ["host", "port", "workers", "log_level", "title", "cors"]
# ... etc for each section

def suggest_corrections(invalid_key: str, context: str) -> List[str]:
    """Suggest valid keys based on context."""
    valid_keys = {
        "root": VALID_TOP_LEVEL_KEYS,
        "server": VALID_SERVER_KEYS,
        # ...
    }.get(context, [])

    return get_close_matches(invalid_key, valid_keys, n=3, cutoff=0.6)
```

---

## Phase 4: New CLI Commands

### 4.1 `mlserver validate` - Config Validation Without Starting Server

**Usage:**
```bash
$ mlserver validate

Validating mlserver.yaml...

  [✓] YAML syntax valid
  [✓] Required fields present
  [✓] Predictor module importable
  [✓] Model file exists: ./model.pkl
  [✓] Feature order file exists: ./features.json
  [⚠] Port 8000 may be in use (optional check)

Configuration valid! Ready to serve.
```

**Implementation:**

```python
# cli.py

@app.command()
def validate(
    config: Optional[Path] = typer.Argument(None),
    strict: bool = typer.Option(False, "--strict", help="Fail on warnings"),
    check_imports: bool = typer.Option(True, "--check-imports/--no-check-imports"),
    check_files: bool = typer.Option(True, "--check-files/--no-check-files"),
):
    """Validate configuration without starting the server."""
    from .validation import ConfigValidator

    config_file = detect_config_file(config)
    validator = ConfigValidator(config_file)

    results = validator.run_all_checks(
        check_imports=check_imports,
        check_files=check_files
    )

    # Display results with rich formatting
    for check in results:
        if check.passed:
            console.print(f"  [green]✓[/green] {check.name}")
        elif check.warning:
            console.print(f"  [yellow]⚠[/yellow] {check.name}: {check.message}")
        else:
            console.print(f"  [red]✗[/red] {check.name}: {check.message}")
            if check.suggestion:
                console.print(f"    [dim]→ {check.suggestion}[/dim]")

    if results.has_errors or (strict and results.has_warnings):
        raise typer.Exit(1)

    console.print("\n[green]Configuration valid![/green]")
```

### 4.2 `mlserver doctor` - Diagnose Common Issues

**Usage:**
```bash
$ mlserver doctor

MLServer Doctor - Diagnosing your environment...

System Checks:
  [✓] Python 3.11.0 (supported: 3.9+)
  [✓] Docker available (version 24.0.6)
  [✓] Git available (version 2.42.0)

Project Checks:
  [✓] mlserver.yaml found
  [✓] Configuration valid
  [⚠] .gitignore missing common patterns (model files, __pycache__)
  [✓] GitHub Actions workflow present and up-to-date

Dependencies:
  [✓] fastapi 0.110.0
  [✓] uvicorn 0.27.0
  [⚠] catboost not installed (optional, needed for CatBoost models)

Recommendations:
  1. Add these to .gitignore: *.pkl, __pycache__/, .env
  2. Run 'pip install catboost' if using CatBoost models
```

**Implementation:**

```python
# cli.py

@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Diagnose common issues and environment problems."""
    from .doctor import (
        check_python_version,
        check_docker,
        check_git,
        check_config,
        check_dependencies,
        check_gitignore,
        check_github_actions,
    )

    checks = [
        ("Python version", check_python_version),
        ("Docker", check_docker),
        ("Git", check_git),
        ("Configuration", check_config),
        ("Dependencies", check_dependencies),
        ("Gitignore", check_gitignore),
        ("GitHub Actions", check_github_actions),
    ]

    recommendations = []

    console.print("[bold]MLServer Doctor[/bold] - Diagnosing your environment...\n")

    for name, check_fn in checks:
        result = check_fn(verbose=verbose)
        # ... display results and collect recommendations
```

### 4.3 `mlserver test` - Quick Prediction Test

**Usage:**
```bash
# Test with inline data
$ mlserver test --data '{"age": 30, "income": 50000}'

Testing prediction...
  Server: http://localhost:8000
  Endpoint: /predict

Response (200 OK, 45ms):
{
  "predictions": [1],
  "timing": {"inference_ms": 12.3}
}

# Test with file
$ mlserver test --file sample_request.json

# Generate sample request
$ mlserver test --generate-sample
Generated: sample_request.json
```

**Implementation:**

```python
@app.command()
def test(
    data: Optional[str] = typer.Option(None, "--data", "-d", help="JSON data to send"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="JSON file with request"),
    endpoint: str = typer.Option("predict", "--endpoint", "-e"),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    generate_sample: bool = typer.Option(False, "--generate-sample"),
):
    """Test predictions against a running server or generate sample requests."""
    import httpx
    import time

    if generate_sample:
        # Generate sample based on feature_order in config
        config = load_config()
        sample = generate_sample_request(config)
        Path("sample_request.json").write_text(json.dumps(sample, indent=2))
        console.print("[green]✓[/green] Generated: sample_request.json")
        return

    # Load request data
    if file:
        request_data = json.loads(file.read_text())
    elif data:
        request_data = {"payload": {"records": [json.loads(data)]}}
    else:
        console.print("[red]✗[/red] Provide --data or --file")
        raise typer.Exit(1)

    # Make request
    url = f"http://{host}:{port}/{endpoint}"
    console.print(f"Testing prediction at [cyan]{url}[/cyan]...")

    start = time.perf_counter()
    try:
        response = httpx.post(url, json=request_data, timeout=30)
        duration = (time.perf_counter() - start) * 1000

        console.print(f"\nResponse ([green]{response.status_code}[/green], {duration:.0f}ms):")
        console.print(json.dumps(response.json(), indent=2))

    except httpx.ConnectError:
        console.print(f"[red]✗[/red] Cannot connect to {url}")
        console.print("  Is the server running? Try: [cyan]mlserver serve[/cyan]")
        raise typer.Exit(1)
```

### 4.4 Interactive `mlserver init`

**Current behavior:** Creates files with defaults, user must edit manually.

**Target behavior:** Guided setup with smart defaults.

```bash
$ mlserver init

Welcome to MLServer! Let's set up your project.

? Classifier name: [my-classifier] sentiment-analyzer
? Found predictor.py with SentimentPredictor. Use this? [Y/n]
? Found model.pkl (2.3 MB). Use as model file? [Y/n]
? Enable Prometheus metrics? [Y/n]
? Set up GitHub Actions for CI/CD? [Y/n]

Creating files...
  [✓] mlserver.yaml
  [✓] .github/workflows/ml-classifier-container-build.yml

Next steps:
  1. Review mlserver.yaml
  2. Test locally: mlserver serve
  3. Validate: mlserver validate
  4. Build container: mlserver build
```

**Implementation:**

```python
@app.command()
def init(
    classifier: Optional[str] = typer.Option(None, "--classifier", "-c"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Initialize a new MLServer project."""
    from .discovery import discover_predictor, suggest_model_files
    from .init_project import create_config, create_github_workflow

    if not interactive:
        # Original non-interactive behavior
        return init_non_interactive(classifier, force)

    console.print("[bold]Welcome to MLServer![/bold] Let's set up your project.\n")

    # Step 1: Classifier name
    default_name = Path.cwd().name.replace("_", "-")
    name = typer.prompt("Classifier name", default=default_name)

    # Step 2: Discover predictor
    discovered = discover_predictor(Path.cwd())
    if discovered and discovered.get("confidence") == "high":
        use_discovered = typer.confirm(
            f"Found {discovered['module']}.{discovered['class_name']}. Use this?",
            default=True
        )
        if use_discovered:
            predictor_module = discovered['module']
            predictor_class = discovered['class_name']
        else:
            predictor_module = typer.prompt("Predictor module")
            predictor_class = typer.prompt("Predictor class")
    else:
        predictor_module = typer.prompt("Predictor module")
        predictor_class = typer.prompt("Predictor class")

    # Step 3: Model files
    model_files = suggest_model_files(Path.cwd())
    init_kwargs = {}
    if model_files:
        console.print(f"\nFound potential model files:")
        for i, f in enumerate(model_files[:5]):
            size = f.stat().st_size / (1024 * 1024)
            console.print(f"  {i+1}. {f.name} ({size:.1f} MB)")

        use_model = typer.prompt(
            "Use which file? (number or 'skip')",
            default="1"
        )
        if use_model.isdigit():
            idx = int(use_model) - 1
            if 0 <= idx < len(model_files):
                init_kwargs["model_path"] = f"./{model_files[idx].name}"

    # Step 4: Features
    enable_metrics = typer.confirm("Enable Prometheus metrics?", default=True)
    setup_github = typer.confirm("Set up GitHub Actions for CI/CD?", default=True)

    # Create files
    config = create_config(
        classifier_name=name,
        predictor_module=predictor_module,
        predictor_class=predictor_class,
        init_kwargs=init_kwargs,
        enable_metrics=enable_metrics,
    )

    # ... write files and show next steps
```

---

## Phase 5: Path Resolution Clarity

### 5.1 Explicit Path Resolution with Warnings

**Problem**: Users don't know paths are relative to config file, not CWD.

**Solution**: Always show resolved paths and warn on potential issues.

```python
# config.py

def resolve_path(path: str, config_dir: Path, context: str) -> Path:
    """Resolve path relative to config directory with clear feedback."""
    import logging
    logger = logging.getLogger(__name__)

    if os.path.isabs(path):
        resolved = Path(path)
    else:
        resolved = (config_dir / path).resolve()

    # Log resolution for debugging
    logger.debug(f"Path resolution ({context}): '{path}' -> '{resolved}'")

    # Warn if file doesn't exist
    if not resolved.exists():
        logger.warning(
            f"Path '{path}' in {context} resolves to '{resolved}' which does not exist. "
            f"Paths are relative to the config file directory: {config_dir}"
        )

    return resolved
```

### 5.2 Path Validation in `mlserver validate`

```python
# validation.py

class PathValidator:
    """Validate all paths in configuration."""

    def validate(self, config: AppConfig, config_dir: Path) -> List[ValidationResult]:
        results = []

        # Check predictor.init_kwargs paths
        for key, value in config.predictor.init_kwargs.items():
            if self._looks_like_path(key, value):
                resolved = config_dir / value
                if resolved.exists():
                    results.append(ValidationResult(
                        passed=True,
                        name=f"Path: {key}",
                        message=f"Exists: {resolved}"
                    ))
                else:
                    results.append(ValidationResult(
                        passed=False,
                        name=f"Path: {key}",
                        message=f"File not found: {resolved}",
                        suggestion=f"Check that '{value}' exists relative to mlserver.yaml"
                    ))

        # Check feature_order path
        if isinstance(config.api.feature_order, str):
            # ... similar validation

        return results
```

---

## Phase 6: IDE Support via JSON Schema

### 6.1 Generate JSON Schema from Pydantic Models

```python
# New file: mlserver/schema.py

from .config import AppConfig
import json

def generate_schema() -> dict:
    """Generate JSON Schema for mlserver.yaml."""
    schema = AppConfig.model_json_schema()

    # Add YAML-specific metadata
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = "MLServer Configuration"
    schema["description"] = "Configuration schema for mlserver.yaml"

    # Add examples to key fields
    schema["properties"]["predictor"]["examples"] = [{
        "module": "my_predictor",
        "class_name": "MyPredictor",
        "init_kwargs": {"model_path": "./model.pkl"}
    }]

    return schema

def save_schema(output_path: str = "mlserver.schema.json"):
    """Save schema to file."""
    schema = generate_schema()
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
```

### 6.2 CLI Command to Generate Schema

```python
@app.command()
def schema(
    output: Path = typer.Option("mlserver.schema.json", "--output", "-o"),
):
    """Generate JSON Schema for IDE autocomplete."""
    from .schema import save_schema

    save_schema(str(output))
    console.print(f"[green]✓[/green] Generated: {output}")
    console.print()
    console.print("To enable in VSCode, add to .vscode/settings.json:")
    console.print('[dim]{"yaml.schemas": {"mlserver.schema.json": "mlserver.yaml"}}[/dim]')
```

### 6.3 Include Schema in Package

```python
# pyproject.toml
[tool.setuptools.package-data]
mlserver = ["schema/*.json"]
```

---

## Phase 7: Progress and Feedback

### 7.1 Progress Indicators for Long Operations

```python
# cli.py

from rich.progress import Progress, SpinnerColumn, TextColumn

@app.command()
def build(...):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building container...", total=None)

        # Stream build output
        for line in build_container_streaming(...):
            if line.startswith("Step"):
                progress.update(task, description=line)
            elif "error" in line.lower():
                console.print(f"[red]{line}[/red]")

        progress.update(task, description="[green]Build complete![/green]")
```

### 7.2 Streaming Docker Output

```python
# container.py

def build_container_streaming(project_path: str, **kwargs) -> Iterator[str]:
    """Build container with streaming output."""
    import subprocess

    process = subprocess.Popen(
        ["docker", "build", ...],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        yield line.strip()

    process.wait()
    if process.returncode != 0:
        raise BuildError(f"Docker build failed with code {process.returncode}")
```

---

## Implementation Roadmap

### Week 1: Critical Bug Fixes (Phase 0)

| # | Task | File | Effort | Owner |
|---|------|------|--------|-------|
| 1 | Add logger import in `_to_jsonable` | `server.py` | 5m | SWE |
| 2 | Thread safety for metrics singleton | `metrics.py` | 1h | SWE |
| 3 | Fix directory traversal in feature_order | `config.py` | 2h | SWE |
| 4 | Fix cache collision for heterogeneous records | `adapters.py` | 2h | MLE |
| 5 | Add model warmup mechanism | `server.py` | 4h | MLE |

**Week 1 Total**: ~9 hours

---

### Week 2: API & Code Quality (Phase 1)

| # | Task | File | Effort | Owner |
|---|------|------|--------|-------|
| 6 | Create unified error hierarchy | `errors.py` (NEW) | 4h | SWE |
| 7 | Simplify payload structure (instances format) | `schemas.py`, `adapters.py` | 8h | SWE |
| 8 | Add input schema validation | `validation.py` | 4h | MLE |
| 9 | Document response formats in OpenAPI | `server.py` | 2h | SWE |
| 10 | Consolidate CLI to single module | `cli.py` | 4h | SWE |

**Week 2 Total**: ~22 hours

---

### Week 3: UX Improvements (Phases 2-4)

| # | Task | Phase | Effort | Owner |
|---|------|-------|--------|-------|
| 11 | Reduce minimum config to 3 lines | Phase 2 | 4h | SWE |
| 12 | Error message framework with suggestions | Phase 3 | 6h | SWE |
| 13 | `mlserver validate` command | Phase 4 | 4h | SWE |
| 14 | `mlserver doctor` command | Phase 4 | 4h | SWE |
| 15 | Interactive `mlserver init` | Phase 4 | 6h | SWE |

**Week 3 Total**: ~24 hours

---

### Week 4: Polish & Developer Experience (Phases 5-7)

| # | Task | Phase | Effort | Owner |
|---|------|-------|--------|-------|
| 16 | Path resolution clarity + warnings | Phase 5 | 4h | SWE |
| 17 | JSON Schema generation + CLI | Phase 6 | 4h | SWE |
| 18 | `mlserver test` command | Phase 4 | 4h | SWE |
| 19 | Progress indicators for build/push | Phase 7 | 4h | SWE |
| 20 | "Did you mean?" for typos | Phase 3 | 2h | SWE |
| 21 | Auto-discovery (predictor, models) | Phase 2 | 4h | MLE |
| 22 | Update all documentation | - | 4h | All |

**Week 4 Total**: ~26 hours

---

### Total Effort Summary

| Phase | Description | Effort |
|-------|-------------|--------|
| **Phase 0** | Critical Bug Fixes | ~9h |
| **Phase 1** | API & Code Quality | ~22h |
| **Phase 2** | Configuration Simplification | ~8h |
| **Phase 3** | Error Message Revolution | ~8h |
| **Phase 4** | New CLI Commands | ~18h |
| **Phase 5** | Path Resolution Clarity | ~4h |
| **Phase 6** | IDE Support (JSON Schema) | ~4h |
| **Phase 7** | Progress and Feedback | ~4h |
| | **TOTAL** | **~77h** |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Lines in minimal config | 12 | 3 |
| Errors with suggestions | ~20% | 100% |
| Commands with --help examples | 4/12 | 12/12 |
| Time to first prediction (new user) | ~30 min | ~5 min |
| Config validation coverage | Runtime only | Pre-runtime |
| IDE autocomplete support | None | Full |
| Critical bugs | 5 | 0 |
| Test coverage | 32% | 80% |

---

## File Changes Summary

### New Files

| File | Description | Phase |
|------|-------------|-------|
| `mlserver/errors.py` | Unified error hierarchy with suggestions | 1, 3 |
| `mlserver/doctor.py` | Environment diagnostic checks | 4 |
| `mlserver/discovery.py` | Predictor & model auto-discovery | 2 |
| `mlserver/schema.py` | JSON Schema generation for IDE support | 6 |

### Modified Files

| File | Changes | Phase |
|------|---------|-------|
| `mlserver/metrics.py` | Add thread safety to singleton | 0 |
| `mlserver/config.py` | Fix path traversal, add smart defaults | 0, 2 |
| `mlserver/adapters.py` | Fix cache collision, simplify payload | 0, 1 |
| `mlserver/server.py` | Fix logger, add warmup, OpenAPI docs | 0, 1 |
| `mlserver/schemas.py` | Add `instances` format support | 1 |
| `mlserver/validation.py` | Add feature schema validator, path checks | 1, 5 |
| `mlserver/cli.py` | Add validate, doctor, test, schema commands; interactive init | 4 |
| `mlserver/container.py` | Streaming build output | 7 |
| `docs/configuration.md` | Path resolution documentation | 5 |
| `pyproject.toml` | Include schema files in package | 6 |

---

## Appendix: Error Message Examples

### Before/After Comparisons

**Module Import Error**

Before:
```
ImportError: No module named 'my_predictr'
```

After:
```
✗ Cannot import predictor module

  Module 'my_predictr' not found.

  Did you mean: my_predictor?

  Troubleshooting:
    1. Check spelling in mlserver.yaml
    2. Ensure file exists: my_predictor.py
    3. Test manually: python -c "import my_predictor"
```

**Invalid Configuration**

Before:
```
1 validation error for AppConfig
server.port
  Input should be a valid integer [type=int_parsing]
```

After:
```
✗ Configuration Error: Invalid port value

  server.port must be an integer, got string "8000"

  Fix in mlserver.yaml:

    server:
      port: 8000  # Remove quotes

  Valid port range: 1-65535
```

**Missing File**

Before:
```
FileNotFoundError: [Errno 2] No such file or directory: './model.pkl'
```

After:
```
✗ Model file not found

  Path: ./model.pkl
  Resolved to: /home/user/project/model.pkl

  Paths in mlserver.yaml are relative to the config file location.

  Found similar files in project:
    - ./models/model.pkl
    - ./artifacts/trained_model.pkl

  Update mlserver.yaml:

    predictor:
      init_kwargs:
        model_path: "./models/model.pkl"
```
