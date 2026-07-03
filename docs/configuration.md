# Configuration Guide

> ✅ **Metadata and Versioning System Streamlined!**
>
> The metadata system has been completely overhauled to reduce confusion and eliminate manual configuration.
> All metadata is now automatically detected from your git repository and environment.
>
> **What's Changed:**
> - **Removed redundant version fields** - No more 5 different version types!
> - **Automatic detection** - Git info, deployment timestamps, and project names are auto-detected
> - **Simplified configuration** - Just focus on your predictor and API settings
> - **Cleaner responses** - Streamlined metadata in all API responses
>
> **Key Improvements:**
> - ✅ **Auto-detected from git**: project name, commit hash, tags, branch
> - ✅ **Auto-generated**: deployment timestamps, MLServer version
> - ✅ **New logger config**: Control timestamps, structured logging, task names
> - ✅ **Enhanced metadata**: Shows predictor module, config file, and more
>
> **Migration Guide:**
> 1. **Remove deprecated fields** from your config:
>    - ❌ `model.version`, `model.trained_at`, `model.metrics`
>    - ❌ `classifier.version`, `classifier.repository`
> 2. **Keep only essential fields**:
>    - ✅ `classifier.name` and `classifier.description`
>    - ✅ Your predictor and API configurations
> 3. **Use git tags** for versioning: `mlserver tag --classifier <name> patch`

## Complete Annotated Example

This comprehensive example shows **ALL** available configuration parameters with detailed annotations:

```yaml
# ============================================================================
# COMPLETE mlserver.yaml - All Available Parameters (with defaults shown)
# ============================================================================

# ----------------------------------------------------------------------------
# SERVER CONFIGURATION
# ----------------------------------------------------------------------------
server:
  title: "ML Server"                    # API title shown in docs (default: "ML Server")
  host: "0.0.0.0"                       # Network interface to bind (default: "0.0.0.0")
  port: 8000                            # Port number 1-65535 (default: 8000)
  log_level: "INFO"                     # Logging level: DEBUG|INFO|WARNING|ERROR (default: "INFO")
  workers: 1                            # Number of worker processes (default: 1, use 1 for K8s)
                                        # NOTE: with workers > 1 each process keeps its own
                                        # Prometheus registry - see docs/observability.md

  # Logger Configuration (controls log output format)
  logger:
    timestamp: false                     # Include timestamps in logs (default: false)
    structured: true                     # Use structured JSON logging (default: true)
    show_tasks: false                    # Show async task names (default: false)
    format: null                         # Custom log format string (optional)

  # CORS Configuration (optional, disabled by default for security)
  cors:
    allow_origins: []                   # List of allowed origins, empty = CORS disabled (default: [])
    allow_methods: ["GET", "POST"]      # Allowed HTTP methods (default: ["GET", "POST"])
    allow_headers: ["Content-Type"]     # Allowed headers (default: ["Content-Type"])
    allow_credentials: false            # Allow credentials in CORS (default: false)

# ----------------------------------------------------------------------------
# PREDICTOR CONFIGURATION (REQUIRED)
# ----------------------------------------------------------------------------
predictor:
  module: "my_predictor"                # Python module containing predictor (required)
  class_name: "MyPredictor"             # Class name within module (required)
  init_kwargs:                          # Arguments passed to predictor __init__ (optional)
    model_path: "./models/model.pkl"
    config_param: "value"
    threshold: 0.5

# ----------------------------------------------------------------------------
# API CONFIGURATION (OPTIONAL - sensible defaults if omitted)
# ----------------------------------------------------------------------------
api:
  version: "v1"                         # API version label, metadata tracking ONLY
                                        # (endpoints are never prefixed with it)
  endpoints:                            # Enable/disable specific endpoints
    predict: true                       # POST /predict endpoint (default: true)
    predict_proba: true                 # POST /predict_proba endpoint (default: true)
    # Note: no separate batch endpoint - /predict handles both single and batch

  # Input/Output Configuration
  adapter: "records"                    # Input format: "records"|"ndarray"|"auto" (default: "records")
  feature_order: null                   # Optional: list of feature names, e.g., ["age", "sex"],
                                        # or a path to a JSON file containing the list

  # Response Format Configuration
  response_format: "standard"           # Response format type (default: "standard")
                                        # Options:
                                        # - "standard": Traditional format with predictions list
                                        # - "passthrough": Return predictor output unmodified
                                        # - "custom": DEPRECATED (removal targeted for 1.0) — logs a
                                        #   warning at load time; return the desired structure from
                                        #   your predictor and use "standard"/"passthrough" instead
  response_validation: true             # Enable response validation (default: true)
                                        # Set false for complex custom responses
  extract_values: false                 # DEPRECATED (removal targeted for 1.0): for dict responses,
                                        # extract values to list. Logs a warning when true; shape the
                                        # output in your predictor instead. (default: false)

  # Concurrency Control
  thread_safe_predict: false            # Use thread lock during prediction (default: false)
  max_concurrent_predictions: 1         # Max concurrent predictions (default: 1)
                                        # 1 = one prediction at a time (K8s pod protection),
                                        # overflow requests get HTTP 503 + Retry-After header.
                                        # 0 = concurrency limiting disabled entirely.
  retry_after_seconds: 5                # Value of the Retry-After header (seconds) on 503
                                        # responses when the limit is reached (default: 5)

  # Startup behavior
  warmup_on_start: true                 # Run a dummy prediction at startup to reduce
                                        # first-request latency (needs feature_order)

# ----------------------------------------------------------------------------
# OBSERVABILITY CONFIGURATION
# ----------------------------------------------------------------------------
observability:
  metrics: true                         # Enable Prometheus metrics (default: true)
  metrics_endpoint: "/metrics"          # Metrics endpoint path (default: "/metrics")
  structured_logging: true              # Enable JSON structured logging (default: true)
  log_payloads: false                   # Log request payload + response at INFO (default: false)
                                        # WARNING: May log sensitive data
  correlation_ids: true                 # Generate correlation IDs for tracing (default: true)

# ----------------------------------------------------------------------------
# CLASSIFIER METADATA (SIMPLIFIED - Most fields now auto-detected!)
# ----------------------------------------------------------------------------
classifier:
  name: "my-classifier"                 # Classifier name (required, URL-safe: lowercase + hyphens)
  description: ""                       # Human-readable description (optional)
  # NOTE: version, repository, and other metadata are now auto-detected from git!

# ----------------------------------------------------------------------------
# MODEL METADATA (DEPRECATED - Do not use!)
# ----------------------------------------------------------------------------
# The 'model' section is still accepted but deprecated. All metadata is now
# auto-detected:
# - Version comes from git tags
# - Deployment timestamp is auto-generated
# - Project/repository name is detected from git
# - MLServer version is detected from package

# ----------------------------------------------------------------------------
# BUILD CONFIGURATION (OPTIONAL)
# ----------------------------------------------------------------------------
build:
  base_image: "python:3.9-slim"        # Docker base image (default from settings)
  registry: "docker.io/myorg"          # Container registry URL (optional)
  tag_prefix: "ml-models"               # Prefix for container tags (optional)
  include_files:                        # Explicit files to include (optional)
    - "models/"
    - "configs/"
    - "*.py"
  exclude_patterns:                     # Patterns to exclude (optional)
    - "*.pyc"
    - "__pycache__"
    - ".git"
    - "tests/"

# ----------------------------------------------------------------------------
# DEPLOYMENT CONFIGURATION (OPTIONAL - used by build/CI tooling)
# ----------------------------------------------------------------------------
deployment:
  strategy: "single"                    # "single" or "multi" for separate services
  container_naming: "{repository}-{classifier}:{version}"  # Container tag template
  git_tag_format: "{classifier}-v{version}"                # Git tag format for releases
  parallel_builds: true                 # Parallel container builds for multiple classifiers
  registry:
    type: "ghcr"                        # "ghcr" (GitHub Container Registry) or "ecr" (AWS)
    url: null                           # Registry URL (GHCR default: ghcr.io)
    namespace: null                     # Registry namespace (GHCR: auto-detected from git)
    push_on_build: false                # Automatically push after build
    ecr:                                # Required when type: "ecr"
      aws_region: "eu-central-1"
      registry_id: "123456789012"       # AWS account ID
      repository_prefix: "ml-classifiers"
    github_variables:
      aws_role_arn_var: "AWS_RUNNER_ROLE_ARN"  # GitHub variable with AWS IAM role ARN
      aws_role_arn_value: null                 # Direct ARN value (less secure alternative)
  resource_limits:                      # Resource limits for Kubernetes deployments
    memory: "2Gi"
    cpu: "1000m"
  health_check: null                    # Health check configuration (optional)
```

## Overview

MLServer uses YAML configuration files to define server settings, predictor loading, observability, and API behavior. Configuration is validated using Pydantic for type safety and clear error messages.

**Unknown keys are not silently ignored**: if the YAML contains a key that is not part of the schema (e.g. a typo like `porrt: 9999`), MLServer logs a warning at load time so misconfigurations don't slip through. `mlserver validate` reports the same warnings.

## Configuration Files

### File Detection Priority

1. **Explicit path**: `mlserver serve /path/to/config.yaml`
2. **Auto-detection**: `mlserver.yaml` in the current directory

If neither exists, the CLI exits with an error.

### Configuration Types

- **Single Classifier**: `mlserver.yaml` with a top-level `predictor` section
- **Multi-Classifier**: `mlserver.yaml` with a top-level `classifiers` section (detected automatically)

## Single Classifier Configuration

### Minimal Configuration

```yaml
# mlserver.yaml - Minimal working example
predictor:
  module: my_predictor  # Just the filename!
  class_name: MyPredictor
```

### Complete Configuration

```yaml
# mlserver.yaml - Full realistic example
server:
  title: "Titanic Survival Prediction API"
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"
  workers: 1
  cors:
    allow_origins: ["https://app.example.com"]
    allow_methods: ["GET", "POST"]
    allow_headers: ["Content-Type"]

predictor:
  module: predictor_catboost  # Simple filename
  class_name: CatBoostPredictor
  init_kwargs:
    model_path: "./models/catboost_model.cbm"
    features_path: "./models/features.json"
    threshold: 0.5

observability:
  metrics: true
  metrics_endpoint: "/metrics"
  structured_logging: true
  correlation_ids: true
  log_payloads: false  # Privacy consideration

api:
  adapter: "records"
  feature_order: ["age", "sex", "fare", "pclass"]
  thread_safe_predict: true
  max_concurrent_predictions: 1
  endpoints:
    predict: true
    predict_proba: true

classifier:
  name: "catboost-survival"
  description: "CatBoost model for Titanic survival"

build:
  base_image: "python:3.9-slim"
  registry: "registry.example.com"
  tag_prefix: "ml-models"
```

## Multi-Classifier Configuration

A config with a top-level `classifiers` section defines multiple classifiers in one repository. Each entry contains a full per-classifier configuration (`predictor`, `classifier`, `api`, ...). Global `server` and `observability` sections are shared by all classifiers. See [Multi-Classifier Support](./multi-classifier.md) for the full workflow.

### Dict Format (canonical)

```yaml
# mlserver.yaml (multi-classifier)
server:
  host: 0.0.0.0
  port: 8000

observability:
  metrics: true
  structured_logging: true

repository:
  name: "multi-model-repo"
  description: "Repository with multiple ML models"

classifiers:
  catboost-model:
    predictor:
      module: predictor_catboost
      class_name: CatBoostPredictor
      init_kwargs:
        model_path: "./models/catboost.cbm"
    classifier:
      name: "catboost-model"
      # version auto-detected from git tags
    api:
      response_format: "standard"

  randomforest-model:
    predictor:
      module: predictor_rf
      class_name: RandomForestPredictor
      init_kwargs:
        model_path: "./models/rf.pkl"
    classifier:
      name: "randomforest-model"
    api:
      response_format: "custom"

default_classifier: "catboost-model"

deployment:
  strategy: "multi"
  resource_limits:
    memory: "2Gi"
    cpu: "1000m"
```

### List Format (also accepted)

The `classifiers` section may alternatively be a list; each entry must carry its name in `classifier.name` (or a `name` key). It is normalized to the dict format at load time:

```yaml
classifiers:
  - classifier:
      name: "catboost-model"
    predictor:
      module: predictor_catboost
      class_name: CatBoostPredictor
  - classifier:
      name: "randomforest-model"
    predictor:
      module: predictor_rf
      class_name: RandomForestPredictor
```

Notes:
- The `server` and `observability` sections are **global** — there are no per-classifier server overrides. Each classifier runs as its own server process/container, selected via `mlserver serve --classifier <name>` (or the `MLSERVER_CLASSIFIER` environment variable in containers).
- `repository` holds free-form repository metadata (e.g. `name`, `description`).
- `default_classifier` names the classifier used when `--classifier` is not passed.

## Configuration Sections

### Server Configuration

```yaml
server:
  # API metadata
  title: "API Title"

  # Network settings
  host: "0.0.0.0"  # Bind address
  port: 8000       # Port number

  # Logging
  log_level: "INFO"  # DEBUG|INFO|WARNING|ERROR|CRITICAL

  # Worker configuration
  workers: 1       # Number of processes (not threads!)

  # Log output format
  logger:
    timestamp: false
    structured: true
    show_tasks: false
    format: null

  # CORS settings (omit entirely to disable CORS)
  cors:
    allow_origins: ["https://app.example.com"]  # Empty list = disabled
    allow_methods: ["GET", "POST"]
    allow_headers: ["Content-Type"]
    allow_credentials: false
```

For development auto-reload, use the CLI flag `mlserver serve --reload` — it is not a config key.

### Predictor Configuration

```yaml
predictor:
  # Module loading (intelligent resolution)
  module: predictor_name  # Simple filename
  # OR
  module: path/to/predictor.py  # Relative path
  # OR
  module: package.module.predictor  # Full module path

  # Class configuration
  class_name: PredictorClass

  # Initialization arguments
  init_kwargs:
    model_path: "./model.pkl"
    config:
      threshold: 0.5
      batch_size: 32
    custom_param: "value"
```

#### Compact string spec

`predictor` also accepts a single `"module:ClassName"` string as shorthand for the mapping above (with empty `init_kwargs`). The two forms are equivalent:

```yaml
# String form (no init_kwargs)
predictor: "predictor_name:PredictorClass"

# Mapping form (equivalent)
predictor:
  module: predictor_name
  class_name: PredictorClass
```

Use the mapping form when you need `init_kwargs`.

#### Predictor contract

A predictor is any Python class exposing `predict(X)` — it never needs to import or subclass anything from this package (the `Predictor` protocol is structural). Optional methods are discovered at runtime:

- `predict(self, X)` — **required**; returns predictions for a 2D feature matrix.
- `predict_proba(self, X)` — optional; when present, powers `/predict_proba` (otherwise that endpoint returns 501).
- `load(self)` — optional; called once at startup after `__init__` and before the first prediction. Put expensive artifact loading here so a failure aborts startup instead of failing the first request.
- `close(self)` — optional; called at shutdown for cleanup.

See [Development Guide → Writing Predictors](./development.md#writing-predictors) for full examples.

#### Module Resolution

The system intelligently resolves module paths:

1. **Simple name** (`predictor_catboost`):
   - Looks for `predictor_catboost.py` in config directory
   - Adds config directory to Python path

2. **Relative path** (`models/predictor.py`):
   - Resolves relative to config file location
   - Strips `.py` extension

3. **Full module** (`mlserver.predictors.catboost`):
   - Uses standard Python import

### Observability Configuration

```yaml
observability:
  # Metrics
  metrics: true
  metrics_endpoint: "/metrics"

  # Logging
  structured_logging: true  # JSON request/response log events

  # Request tracking
  correlation_ids: true    # Attach correlation IDs to logs
  log_payloads: false      # Log request payload + response at INFO (privacy!)
```

The log level lives under `server.log_level`; log formatting (timestamps, JSON, task names) under `server.logger`.

### API Configuration

```yaml
api:
  # Input adapter
  adapter: "records"  # records|ndarray|auto (default: records)

  # Feature handling
  feature_order: ["feat1", "feat2"]  # Explicit ordering (or path to JSON file)

  # Concurrency
  thread_safe_predict: true          # Lock during prediction
  max_concurrent_predictions: 1      # 1 = single prediction at a time (default)
                                     # 0 = disable concurrency limiting
  retry_after_seconds: 5             # Retry-After header value (seconds) on 503 (default: 5)

  # Response format
  response_format: "standard"         # standard|passthrough (custom is deprecated)
  response_validation: true           # Enable/disable validation
  extract_values: false               # DEPRECATED — extract dict values to list

  # Endpoints control
  endpoints:
    predict: true
    predict_proba: true

  # Versioning (metadata only - never used in URL paths)
  version: "v1"

  # Startup
  warmup_on_start: true
```

#### Response Formats

The `response_format` configuration controls how predictions are formatted:

1. **`standard`** (default): Traditional format with predictions list
   ```json
   {
     "predictions": [0, 1, 0],
     "time_ms": 12.5,
     "predictor_class": "MyPredictor",
     "metadata": {...}
   }
   ```

2. **`passthrough`**: Return predictor output unmodified
   - No wrapper or metadata
   - Complete control over response structure
   - Useful for legacy systems or special requirements

3. **`custom`** (**deprecated**): Flexible format that wraps the output in a `result` field
   ```json
   {
     "result": {
       "a": [1, 2, 3],
       "b": {"c": [4, 5]}
     },
     "time_ms": 16.4,
     "metadata": {...}
   }
   ```

> **Deprecated (removal targeted for 1.0):** `response_format: custom` and `extract_values` both log a load-time deprecation warning. Return the exact structure you want from your predictor and use `standard` or `passthrough` instead.

Example configurations:

```yaml
# For legacy compatibility
api:
  response_format: "passthrough"
  response_validation: false  # Skip validation

# Standard ML classifier (default, recommended)
api:
  response_format: "standard"
```

### Classifier Metadata

```yaml
classifier:
  name: "model-name"        # Required for tagging/builds; URL-safe
  description: "Model description"
```

Everything else (version, repository, commit, deployment time) is auto-detected from git and the environment and surfaced in `/info` and response metadata.

> **`classifier.version` is deprecated.** Git tags are the canonical version source. A `classifier.version` in the config is display-only and logs a deprecation warning; set the version by creating a git tag with `mlserver tag --classifier <name> <patch|minor|major>` instead.

### Build Configuration

```yaml
build:
  base_image: "python:3.9-slim"       # Docker base image
  registry: "registry.example.com"    # Registry URL
  tag_prefix: "mlserver"              # Container tag prefix
  include_files:                      # Explicit include list
    - "models/"
    - "*.py"
  exclude_patterns:                   # Exclusions
    - "__pycache__"
    - "tests/"
```

### Deployment Configuration

Used by the build/CI tooling (`mlserver build`, `mlserver init-github`) for multi-classifier repositories:

```yaml
deployment:
  strategy: "single"                  # single|multi
  container_naming: "{repository}-{classifier}:{version}"
  git_tag_format: "{classifier}-v{version}"   # tag template (see tag-format note below)
  parallel_builds: true
  registry:
    type: "ghcr"                      # ghcr|ecr
    url: null
    namespace: null
    push_on_build: false
    ecr:
      aws_region: "eu-central-1"
      registry_id: "123456789012"
      repository_prefix: "ml-classifiers"
    github_variables:
      aws_role_arn_var: "AWS_RUNNER_ROLE_ARN"
      aws_role_arn_value: null
  resource_limits:
    memory: "2Gi"
    cpu: "1000m"
  health_check: null
```

> **Tag format (RFC 0001 D1–D3).** `mlserver tag` creates **canonical** `<classifier>/vX.Y.Z` tags (slash-namespaced) — the MLServer commit is no longer part of the tag name; it lives in the annotated-tag message and the container's OCI labels. Legacy `<classifier>-vX.Y.Z-mlserver-<hash>` tags remain **readable** everywhere (build validation, status, version listing all parse both forms), so existing tags keep working. The `git_tag_format` field above and `classifier.version` are deprecated, as is `push --version-source` — git tags are the canonical version source.

## Environment Variables

MLServer does **not** support per-setting override variables like `MLSERVER_PORT` or `MLSERVER_WORKERS` — use the YAML config or CLI flags. The variables that are actually read are tooling defaults and container plumbing:

```bash
# Tooling defaults (settings)
export MLSERVER_DEFAULT_HOST="0.0.0.0"   # Default bind host when config omits it
export MLSERVER_DEFAULT_PORT="8000"      # Default port when config omits it
export MLSERVER_LOG_LEVEL="DEBUG"        # Default log level

# Server factory / containers (usually set for you)
export MLSERVER_CONFIG_PATH="/app/mlserver.yaml"  # Config path for worker processes
export MLSERVER_CLASSIFIER="sentiment"            # Classifier selection in multi-classifier configs
```

See the [CLI Reference](./cli-reference.md#environment-variables) for the complete list (including the git-metadata variables baked into containers at build time).

## CLI Overrides

Command-line arguments override the config file:

```bash
# Override port
mlserver serve --port 9000

# Override workers
mlserver serve --workers 8

# Select classifier from multi-classifier config
mlserver serve mlserver.yaml --classifier staging

# Multiple overrides
mlserver serve \
  --port 9000 \
  --workers 8 \
  --log-level DEBUG
```

Note: `--log-level` only overrides `server.log_level` when explicitly passed on the command line.

## Configuration Validation

### Schema Validation

All configurations are validated using Pydantic:

```yaml
# Invalid config example
server:
  port: "not_a_number"  # Error: port must be integer
  workers: -1           # Error: workers must be positive
```

Error message:
```
Configuration error:
  - server.port: value is not a valid integer
  - server.workers: ensure this value is greater than 0
```

### Unknown Key Warnings

Keys that are not part of the schema produce warnings (not errors) at load time:

```
Warning: Unknown configuration key 'server.porrt' — did you mean 'port'?
```

This catches typos that Pydantic would otherwise silently drop.

### Required Fields

Minimal required configuration:
```yaml
predictor:
  module: <module_name>
  class_name: <class_name>
```

### Type Checking

Configuration types are enforced:
- Integers: `port`, `workers`, `max_concurrent_predictions`
- Booleans: `metrics`, `structured_logging`, `thread_safe_predict`
- Strings: `module`, `class_name`, `log_level`, `adapter`
- Lists: `allow_origins`, `allow_methods`, `feature_order`
- Dictionaries: `init_kwargs`, `endpoints`

## Configuration Patterns

### Development Configuration

```yaml
# mlserver.dev.yaml
server:
  workers: 1        # Single worker for debugging
  log_level: "DEBUG"

observability:
  log_payloads: true  # See all requests (dev only!)

api:
  thread_safe_predict: false  # Faster for single worker
```

```bash
# Auto-reload comes from the CLI flag
mlserver serve mlserver.dev.yaml --reload
```

### Production Configuration

```yaml
# mlserver.prod.yaml
server:
  workers: 1          # 1 worker per pod; scale with replicas (accurate metrics)
  log_level: "WARNING"

observability:
  metrics: true
  structured_logging: true
  log_payloads: false  # Privacy

api:
  thread_safe_predict: true
  max_concurrent_predictions: 1  # Reject overflow with 503 so the LB retries elsewhere
```

### Testing Configuration

```yaml
# mlserver.test.yaml
server:
  port: 8001  # Dedicated test port
  workers: 1

predictor:
  module: mock_predictor
  class_name: MockPredictor

observability:
  metrics: false  # Disable for tests
```

## Troubleshooting

### Common Issues

**Module not found**:
```yaml
predictor:
  module: predictor  # Make sure predictor.py exists
  # OR use full path
  module: /absolute/path/to/predictor.py
```

**Class not found**:
```yaml
predictor:
  class_name: Predictor  # Check exact class name
  # Case sensitive!
```

**Init kwargs error**:
```yaml
predictor:
  init_kwargs:
    model_path: "./model.pkl"  # Check file exists
    # Paths are relative to config file
```

### Debugging Configuration

Enable debug logging:
```bash
mlserver serve --log-level DEBUG
```

Validate configuration (including unknown-key warnings):
```bash
mlserver validate mlserver.yaml -v
```

Diagnose the environment:
```bash
mlserver doctor -v
```
