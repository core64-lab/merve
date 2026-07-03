# CLI Reference

## Overview

MLServer ships a single command-line interface, `mlserver`, built on Typer with rich, colorful output. It covers the full lifecycle: serving, validation, versioning, container builds, and deployment helpers.

## Installation

The CLI is installed with the package:

```bash
pip install mlserver-fastapi-wrapper

mlserver --help
```

## Commands Overview

```bash
mlserver --help  # Show all commands with rich formatting
```

```
╭─ Commands ───────────────────────────────────────────────────────╮
│ serve              🚀 Launch ML FastAPI server from YAML config  │
│ version            📦 Display version information                │
│ build              🏗️  Build Docker container                     │
│ push               📤 Push container to registry                 │
│ images             📋 List Docker images                         │
│ tag                🏷️  Manage version tags for classifiers        │
│ clean              🧹 Remove Docker images                       │
│ run                🚀 Run Docker container for the classifier    │
│ list-classifiers   📋 List classifiers in multi-classifier config│
│ status             📊 Show ML Server status and system info      │
│ init               🎬 Initialize a new classifier project        │
│ init-github        🔧 Initialize GitHub Actions CI/CD workflow   │
│ validate           Validate configuration without starting       │
│ doctor             Diagnose common issues                        │
│ test               Test prediction against a running server      │
│ schema             Generate JSON schema for mlserver.yaml        │
╰──────────────────────────────────────────────────────────────────╯
```

> Note: There is no shell completion support (`--install-completion` is disabled), and `mlserver` is the only CLI entry point — no separate legacy binary exists.

---

## Machine-readable output (`--json`)

The read/inspection commands support a `--json` flag that prints a single JSON document to **stdout** (rich tables and any deprecation/diagnostic messages go to **stderr**), so the output stays parseable in scripts and CI:

| Command | `--json` scope |
|---------|----------------|
| `version` | Classifier/model/API version info (add `--detailed` for tool info) |
| `images` | Image list with `count` and `classifier` |
| `status` | Docker / config / Python / venv / GitHub-Actions status |
| `list-classifiers` | Classifiers and `default_classifier` |
| `validate` | `valid` flag plus per-check results |
| `tag --status` | Per-classifier tag status (status mode only) |

```bash
mlserver images --json | jq '.images[].tag'
mlserver validate --json | jq '.valid'
mlserver tag --status --json | jq '.classifiers'
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Command failed (validation errors, push failure, missing config, ...) |
| `2` | Usage error (bad arguments / options, from Typer) |

---

## Core Commands

### `serve` - Start ML Server

Launch the ML inference server with FastAPI.

#### Syntax
```bash
mlserver serve [CONFIG] [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `CONFIG` | Path to config file | `mlserver.yaml` |
| `--classifier`, `-c` | Classifier to serve (for multi-classifier configs) | Default from config |
| `--host` | Override host address | From config |
| `--port`, `-p` | Override port number | From config |
| `--workers`, `-w` | Number of worker processes | From config |
| `--reload` | Enable auto-reload for development | `false` |
| `--log-level`, `-l` | Set log level (`DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL`) | `INFO` |

> `--log-level` only overrides the YAML `server.log_level` when it is explicitly passed on the command line.

#### Examples
```bash
# Auto-detect configuration (mlserver.yaml)
mlserver serve

# Specific configuration
mlserver serve mlserver.yaml

# Override settings
mlserver serve --port 9000 --workers 4

# Multi-classifier selection
mlserver serve mlserver.yaml --classifier catboost-survival

# Development mode
mlserver serve --reload --log-level DEBUG
```

---

### `version` - Version Information

Display version information for the classifier project. Use `--detailed` to include MLServer tool version, commit, and installation source.

#### Syntax
```bash
mlserver version [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-p` | Path to classifier project | `.` |
| `--classifier`, `-c` | Classifier name (for multi-classifier configs) | None |
| `--json` | Output as JSON | `false` |
| `--detailed` | Show detailed MLServer tool information | `false` |

#### Examples
```bash
# Basic version (classifier only)
mlserver version

# JSON format
mlserver version --json

# With MLServer tool details
mlserver version --detailed

# Specific classifier in a multi-classifier repo
mlserver version --classifier sentiment
```

#### Use Cases
```bash
# Get full version info for bug reports
mlserver version --detailed --json > version-info.json

# Verify MLServer commit in a build pipeline
mlserver version --detailed | grep -i commit
```

---

### `tag` - Version Tagging & Reproducibility

Manage semantic version tags for each classifier. Git tags are the canonical version source.

#### Tag Format

`mlserver tag` creates **canonical** `<classifier>/vX.Y.Z` tags (slash-namespaced), e.g. `sentiment/v1.0.3`. The MLServer commit is no longer encoded in the tag name — it is recorded in the annotated-tag message and in the container's OCI image labels instead.

The **legacy** `<classifier>-vX.Y.Z-mlserver-<hash>` form (e.g. `sentiment-v1.0.3-mlserver-b5dff2a`) is still **read** everywhere — status, `build`/`push` validation, and version listing all parse both forms — so tags created before this change keep working indefinitely.

> **CI note:** the workflow generated by `mlserver init-github` currently still triggers on the legacy tag pattern (`'*-v*-mlserver-*'`); adjusting the trigger to canonical `'*/v*'` tags is part of the ongoing tag migration.

#### Syntax
```bash
# Show tag status for all classifiers
mlserver tag

# Create a version tag for a specific classifier
mlserver tag --classifier <name> <major|minor|patch>
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `BUMP_TYPE` | Version bump type: `major`, `minor`, or `patch` | None (status mode) |
| `--classifier`, `-c` | Classifier name to tag (required for tagging) | None |
| `--config` | Config file path | `mlserver.yaml` |
| `--message`, `-m` | Tag message | `Release <classifier> vX.Y.Z` |
| `--path` | Path to classifier project | `.` |
| `--allow-missing-mlserver` | Allow tagging even if the MLServer commit cannot be determined (dev/testing only) | `false` |
| `--status` | Show tag status for all classifiers (default when no bump type given) | `false` |
| `--json` | Output tag status as JSON (status mode only) | `false` |

#### Examples

**Create Tags:**
```bash
# Patch release (bug fixes)
mlserver tag --classifier sentiment patch

# Minor release (new features, backward compatible)
mlserver tag --classifier intent minor

# Major release (breaking changes)
mlserver tag --classifier fraud major
```

**Output:**
```
✓ Created tag: sentiment/v1.0.3

  📝 Version: 1.0.2 → 1.0.3 (patch bump)
  🔧 MLServer commit: b5dff2a
  📦 Classifier commit: c5f9997

Next steps:
  1. Push tags to remote: git push --tags
  2. Build container: mlserver build --classifier sentiment
  3. Push to registry: mlserver push --registry <url>
```

**View Status:**
```bash
mlserver tag
```

```
                      🏷️  Classifier Version Status
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Classifier        ┃ Version ┃ MLServer  ┃ Status ┃ Action Required ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ sentiment         │ 1.0.3   │ b5dff2a ✓ │ Ready  │ -               │
│ intent            │ 2.0.1   │ b5dff2a ✓ │ Ready  │ -               │
└───────────────────┴─────────┴───────────┴────────┴─────────────────┘
```

#### Reproducibility Workflow

1. **Tag Creation**: Captures exact state
   ```bash
   mlserver tag --classifier sentiment patch
   # Creates: sentiment/v1.0.3   (MLServer commit recorded in the tag message + OCI labels)
   ```

2. **Build Container**: Uses tag information
   ```bash
   mlserver build --classifier sentiment/v1.0.3
   ```
   - Validates current code matches tag
   - Adds container labels with all commit info
   - Warns if code has changed since tag

3. **Reproduce Exact Build**: Checkout and rebuild
   ```bash
   git checkout sentiment/v1.0.3
   mlserver build --classifier sentiment
   ```

#### Multi-Classifier Tagging

Each classifier is tagged independently:
```bash
mlserver tag --classifier sentiment patch  # v1.0.3
mlserver tag --classifier intent minor     # v2.1.0

# View all tags
mlserver tag
```

---

### `build` - Build Docker Container

Build a Docker container for the classifier project with full reproducibility tracking.

The `--classifier` parameter accepts a simple name or a full version tag (canonical or legacy):
- Simple: `--classifier sentiment`
- Canonical tag: `--classifier sentiment/v1.0.0`
- Legacy tag: `--classifier sentiment-v1.0.0-mlserver-b5dff2a`

When using a full tag, the build validates that your current code matches the tag's expected commits and warns on mismatches.

#### Syntax
```bash
mlserver build [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--classifier`, `-c` | Classifier to build (simple name, or full tag `name/vX.Y.Z` / legacy `name-vX.Y.Z-mlserver-hash`) | Auto-detect |
| `--path` | Path to classifier project | `.` |
| `--config` | Config file to use | Auto-detected |
| `--registry` | Container registry URL | None |
| `--tag-prefix` | Tag prefix for container names | None |
| `--build-arg` | Build arguments (`key=value`, repeatable) | None |
| `--no-cache` | Do not use cache when building | `false` |
| `--verbose`, `-v` | Verbose output | `false` |
| `--force` | Skip validation prompts and continue with build | `false` |

#### Examples

**Simple Build:**
```bash
# Auto-build with version tag
mlserver build

# Build specific classifier
mlserver build --classifier sentiment
```

**Build with a Full Version Tag:**
```bash
# Build exact tagged version (with validation)
mlserver build --classifier sentiment/v1.0.3
```

**Validation Warnings** (a legacy `-mlserver-<hash>` tag additionally validates the MLServer commit, which canonical tags no longer encode):
```bash
# If code doesn't match tag
mlserver build --classifier sentiment-v1.0.2-mlserver-old123

⚠️  Warning: Current code doesn't match tag specifications

Tag specifies:
  Classifier commit: 08472c7
  MLServer commit:   old123

Current working directory:
  Classifier commit: c5f9997 ⚠️  MISMATCH
  MLServer commit:   b5dff2a ⚠️  MISMATCH

Building with CURRENT code. To build exact tagged version:
  git checkout sentiment-v1.0.2-mlserver-old123

# Use --force to skip the prompt
mlserver build --classifier sentiment-v1.0.2-mlserver-old123 --force
```

**Advanced Builds:**
```bash
# Fresh build without Docker cache
mlserver build --no-cache

# Custom registry and build arguments
mlserver build --registry gcr.io/project --build-arg PIP_INDEX_URL=https://pypi.internal
```

#### Container Labels

Built containers include comprehensive labels for full traceability:
```dockerfile
# Classifier information
LABEL com.classifier.name="sentiment"
LABEL com.classifier.version="1.0.3"
LABEL com.classifier.git_tag="sentiment/v1.0.3"
LABEL com.classifier.git_commit="c5f9997"
LABEL com.classifier.git_branch="main"

# MLServer tool information
LABEL com.mlserver.version="0.3.2"
LABEL com.mlserver.commit="b5dff2a"

# OCI standard labels
LABEL org.opencontainers.image.version="1.0.3"
LABEL org.opencontainers.image.created="2025-10-27T10:30:00Z"
LABEL org.opencontainers.image.title="sentiment ML Server"
```

#### Reproducibility

Container labels enable exact reproduction:
```bash
# Inspect container to get tag
docker inspect my-container | grep com.classifier.git_tag

# Checkout exact version
git checkout sentiment/v1.0.3

# Rebuild identically
mlserver build --classifier sentiment
```

---

### `push` - Push Container

Push a built container to a registry. Requires a tagged commit for the specific classifier (or `--force`).

#### Syntax
```bash
mlserver push [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--registry`, `-r` | Container registry URL | **Required** |
| `--classifier`, `-c` | Classifier to push (required for multi-classifier configs) | None |
| `--path` | Path to classifier project | `.` |
| `--tag-prefix` | Tag prefix for container names | None |
| `--force`, `-f` | Force push even if not on tagged commit or tag exists | `false` |
| `--version-source` | **Deprecated** — `git-tag`, `config`, or `auto`. Git tags are canonical; flag will be removed next release | `auto` |

> **`--version-source` is deprecated (RFC 0001 D3).** Git tags are the canonical version source; passing this flag logs a deprecation warning and it will be removed in the next release.

#### Examples
```bash
# Push classifier image
mlserver push --registry gcr.io/project --classifier sentiment

# Force push from an untagged commit
mlserver push --registry gcr.io/project --classifier sentiment --force
```

---

### `images` - List Docker Images

List Docker images for the classifier project. For multi-classifier configs, use `--classifier` to filter.

#### Syntax
```bash
mlserver images [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path` | Path to classifier project | `.` |
| `--classifier`, `-c` | Classifier name (for multi-classifier configs) | None |
| `--json` | Output as JSON (machine-readable) | `false` |

#### Output
```
╭─ Docker Images ──────────────────────────────────────────────────╮
│ Repository   │ Tag     │ Size    │ Created              │       │
├──────────────┼─────────┼─────────┼──────────────────────┼───────┤
│ sentiment    │ v1.0.3  │ 450 MB  │ 2 hours ago         │       │
│ sentiment    │ latest  │ 450 MB  │ 2 hours ago         │       │
╰──────────────────────────────────────────────────────────────────╯
```

---

### `clean` - Remove Docker Images

Remove Docker images for the classifier project.

#### Syntax
```bash
mlserver clean [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path` | Path to classifier project | `.` |
| `--force`, `-f` | Force removal without confirmation | `false` |

#### Examples
```bash
# Interactive clean
mlserver clean

# Remove without confirmation
mlserver clean --force
```

---

### `run` - Run Docker Container

Run a built Docker container for the classifier locally.

#### Syntax
```bash
mlserver run [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--classifier`, `-c` | Classifier to run (required for multi-classifier configs) | None |
| `--path` | Path to classifier project | `.` |
| `--port`, `-p` | Port to expose the container on | `8000` |
| `--version` | Specific version to run | latest |
| `--detach`, `-d` | Run container in background | `false` |
| `--name` | Container name | Auto-generated |
| `--env`, `-e` | Environment variables (`KEY=value`, repeatable) | None |
| `--volume`, `-v` | Volume mounts (`host:container`, repeatable) | None |

#### Examples
```bash
# Run latest build on port 8000
mlserver run --classifier sentiment

# Run a specific version in the background
mlserver run --classifier sentiment --version 1.0.2 --detach

# Custom port, env vars, and mounts
mlserver run -c sentiment -p 9000 -e LOG_LEVEL=DEBUG -v ./models:/app/models
```

---

### `list-classifiers` - List Available Models

Show all classifiers in a multi-classifier configuration.

#### Syntax
```bash
mlserver list-classifiers [CONFIG] [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `CONFIG` | Path to multi-classifier config file | Auto-detect (`mlserver.yaml`) |
| `--json` | Output as JSON (machine-readable) | `false` |

#### Output
```
╭─ Available Classifiers ──────────────────────────────────────────╮
│ Name         │ Version │ Status     │ Description              │
├──────────────┼─────────┼────────────┼──────────────────────────┤
│ catboost     │ 1.0.0   │ Default ✓  │ CatBoost model          │
│ randomforest │ 2.0.0   │ Available  │ RandomForest model      │
╰──────────────────────────────────────────────────────────────────╯
```

---

### `status` - System Status

Show ML Server status and system info (local environment overview, Docker availability, config detection).

#### Syntax
```bash
mlserver status [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--json` | Output as JSON (machine-readable) | `false` |

#### Examples
```bash
mlserver status
mlserver status --json
```

---

## Project Setup Commands

### `init` - Initialize a New Project

Initialize a new MLServer classifier project. Creates all necessary files:
- `mlserver.yaml` (configuration file)
- `<predictor>.py` (skeleton predictor class)
- `.github/workflows/ml-classifier-container-build.yml` (CI/CD workflow)
- `.gitignore` (Python/ML project gitignore)

Existing files are never overwritten unless `--force` is used.

#### Syntax
```bash
mlserver init [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-p` | Path to initialize project in | `.` |
| `--classifier`, `-c` | Classifier name | Directory name |
| `--predictor-file` | Name of predictor Python file (without `.py`) | Derived from classifier |
| `--predictor-class` | Name of predictor class | Derived from classifier |
| `--no-github` | Skip GitHub Actions workflow creation | `false` |
| `--force`, `-f` | Overwrite existing files | `false` |

#### Examples
```bash
mlserver init --classifier sentiment-analyzer
mlserver init --classifier my-model --predictor-file custom_predictor
mlserver init --no-github
```

---

### `init-github` - Initialize CI/CD Workflow

Initialize a GitHub Actions workflow for automated container builds. Sets up automated building and publishing of containers to a container registry when hierarchical version tags (created with `mlserver tag`) are pushed.

This command:
- Creates `.github/workflows/ml-classifier-container-build.yml`
- Configures automated Docker builds on tag push
- Sets up container publishing to GitHub Container Registry
- Auto-detects your GitHub repository information

Note: `mlserver init` already creates this workflow; run `init-github` separately only to add CI/CD to an existing project.

#### Syntax
```bash
mlserver init-github [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-p` | Path to classifier project | `.` |
| `--python-version` | Python version for CI/CD workflow | `3.11` |
| `--registry` | Container registry | `ghcr.io` |
| `--force`, `-f` | Overwrite existing workflow files | `false` |

#### Examples
```bash
mlserver init-github
mlserver init-github --python-version 3.12 --registry ghcr.io
```

---

## Development & Diagnostic Commands

### `validate` - Validate Configuration

Validate configuration without starting the server.

Checks:
- YAML syntax validity
- Required fields present
- Predictor module importable
- Model files exist
- Feature order file exists (if configured)

#### Syntax
```bash
mlserver validate [CONFIG] [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `CONFIG` | Path to config file | Auto-detect (`mlserver.yaml`) |
| `--strict`, `-s` | Fail on warnings | `false` |
| `--check-imports/--no-check-imports` | Check predictor imports | `check-imports` |
| `--verbose`, `-v` | Show detailed output | `false` |
| `--json` | Output as JSON (machine-readable) | `false` |

Passing a file path validates that named file (e.g. `mlserver validate staging.yaml`). Unknown configuration keys are reported as warnings. `--json` emits a `valid` flag plus per-check results and exits non-zero when invalid.

#### Examples
```bash
# Validate auto-detected config
mlserver validate

# Validate a specific config with verbose output
mlserver validate mlserver.yaml -v

# Strict mode - fail on any warning
mlserver validate --strict

# Skip import checking
mlserver validate mlserver.yaml --no-check-imports
```

#### Output
```
Validating configuration...
  ✓ Configuration file: Found mlserver.yaml
  ✓ YAML syntax: Valid
  ✓ Configuration schema: Valid
  ✓ Predictor import: Module 'predictor' loadable

✓ All validation checks passed!
```

---

### `doctor` - Diagnose Environment

Diagnose common issues and environment problems. Checks system requirements, project configuration, dependencies, and provides recommendations for fixing issues.

#### Syntax
```bash
mlserver doctor [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--verbose`, `-v` | Show detailed diagnostics | `false` |
| `--path`, `-p` | Project path to diagnose | `.` |

#### Examples
```bash
mlserver doctor
mlserver doctor -v
mlserver doctor --path ./my-project
```

#### Output
```
Running system checks...
  ✓ Python version: 3.12.3 (>= 3.9 required)
  ✓ Docker: Available (v24.0.0)
  ✓ Git: Available (v2.40.0)

Running project checks...
  ✓ Configuration file: Found
  ✓ Configuration schema: Valid
  ⚠ Predictor import: NumPy version conflict
    → Try: pip install numpy>=1.24

Summary: 5 passed, 0 failed, 1 warning
```

---

### `test` - Test Prediction

Send a test request to a running server to verify it is working correctly.

#### Syntax
```bash
mlserver test [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--data`, `-d` | JSON data for prediction | None |
| `--file`, `-f` | JSON file with request data | None |
| `--url`, `-u` | Server URL | `http://localhost:8000` |
| `--endpoint`, `-e` | Prediction endpoint | `/predict` |
| `--pretty/--raw` | Pretty-print response | `pretty` |

#### Examples
```bash
# Test with inline JSON
mlserver test --data '{"feature1": 1.5, "feature2": 2.0}'

# Test with JSON file
mlserver test --file sample_request.json

# Test another server / endpoint
mlserver test --url http://localhost:8080 --endpoint /predict_proba
```

---

### `schema` - Generate JSON Schema

Generate a JSON schema for `mlserver.yaml` configuration files, enabling IDE autocompletion and validation.

#### Syntax
```bash
mlserver schema [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Output path for schema file | stdout |
| `--type`, `-t` | Config type: `single`, `multi`, or `auto` (supports both) | `auto` |
| `--setup`, `-s` | Show IDE setup instructions after generating schema | `false` |
| `--vscode` | Generate `.vscode/settings.json` for automatic schema association | `false` |

#### Examples
```bash
# Print schema to stdout
mlserver schema

# Save schema with setup instructions
mlserver schema -o .mlserver/schema.json --setup

# Full VSCode setup
mlserver schema -o .mlserver/schema.json --vscode --setup

# Generate for multi-classifier configs only
mlserver schema --type multi -o schema.json
```

#### VSCode Integration

After generating the schema, add to your `mlserver.yaml`:
```yaml
# yaml-language-server: $schema=.mlserver/schema.json

server:
  port: 8000
# ... IDE autocomplete now works!
```

Or add to `.vscode/settings.json`:
```json
{
  "yaml.schemas": {
    ".mlserver/schema.json": ["mlserver.yaml", "**/mlserver.yaml"]
  }
}
```

---

## Environment Variables

These are the environment variables MLServer actually reads:

| Variable | Purpose |
|----------|---------|
| `MLSERVER_DEFAULT_HOST` | Default bind host when the config does not set one |
| `MLSERVER_DEFAULT_PORT` | Default port when the config does not set one |
| `MLSERVER_LOG_LEVEL` | Default log level |
| `MLSERVER_CONFIG_PATH` | Config file path used by the server factory (set by `serve` for worker processes) |
| `MLSERVER_CLASSIFIER` | Classifier to serve from a multi-classifier config (set by `serve --classifier`; baked into containers) |
| `MLSERVER_CONFIG_FILE` | Config file name recorded in response metadata (set in built containers) |
| `MLSERVER_GIT_COMMIT` / `MLSERVER_GIT_TAG` / `MLSERVER_GIT_BRANCH` / `MLSERVER_GIT_URL` | Classifier git metadata overrides (baked into containers at build time) |
| `MLSERVER_API_COMMIT` / `MLSERVER_API_TAG` / `MLSERVER_API_BRANCH` | MLServer tool git metadata overrides (baked into containers at build time) |
| `MLSERVER_SOURCE_PATH` / `MLSERVER_BUILD_TIME` | MLServer source path and build timestamp (baked into containers at build time) |

There are no `MLSERVER_HOST`, `MLSERVER_PORT`, `MLSERVER_WORKERS`, `MLSERVER_METRICS`, or similar per-setting override variables — use the YAML config or CLI flags instead.

> **Removed (RFC 0001 D12):** the old `GlobalSettings` singleton and `global_config.yaml` no longer exist. There are no `MLSERVER_GLOBAL_CONFIG`, `MLSERVER_LOG_FILE`, `MLSERVER_PID_FILE`, `MLSERVER_VENV_PATH`, or `MLSERVER_EXEC_PATH` variables. Package defaults live in `mlserver/defaults.py`; if a stray `global_config.yaml` is found in the working directory the server logs a one-time "no longer read" warning. Configure everything through `mlserver.yaml` plus the environment variables above.

## Configuration File Detection

The CLI resolves the configuration file as follows:

1. Explicitly provided path (e.g. `mlserver serve my-config.yaml`)
2. `mlserver.yaml` in the current directory

If neither is found, the command exits with an error asking you to create `mlserver.yaml` or pass a path.

## Advanced Usage

### Multi-Classifier Deployment

```bash
# List available classifiers
mlserver list-classifiers mlserver.yaml

# Serve a specific classifier
mlserver serve mlserver.yaml --classifier production

# Serve multiple classifiers as separate processes
mlserver serve mlserver.yaml --classifier staging --port 8001 &
mlserver serve mlserver.yaml --classifier production --port 8002 &
```

### Development Workflow

```bash
# Start with auto-reload
mlserver serve --reload --log-level DEBUG

# Validate the configuration
mlserver validate -v

# Diagnose the environment
mlserver doctor

# Send a test request
mlserver test --data '{"feature1": 1.5, "feature2": 2.0}'
```

### Production Deployment

```bash
# Tag a release
mlserver tag --classifier sentiment patch

# Build container from the tag
mlserver build --classifier sentiment

# Push to registry
mlserver push --registry gcr.io/project --classifier sentiment

# Deploy to Kubernetes
kubectl apply -f deployment.yaml
```

## Debugging

### Verbose Output

```bash
# Debug logging
mlserver serve --log-level DEBUG

# Verbose validation / diagnostics
mlserver validate -v
mlserver doctor -v
```

### Configuration Validation

```bash
# Validate auto-detected config
mlserver validate

# Validate a specific file
mlserver validate mlserver.yaml
```

## Common Issues

### Port Already in Use

```bash
# Use different port
mlserver serve --port 8001

# Kill existing process
lsof -ti:8000 | xargs kill -9
```

### Module Not Found

```bash
# Verify predictor imports
mlserver validate --check-imports

# Full environment diagnosis
mlserver doctor -v
```

### Permission Denied

```bash
# Use sudo for system ports
sudo mlserver serve --port 80

# Or use high port
mlserver serve --port 8080
```

## Tips and Tricks

### Quick Commands

```bash
# Serve with defaults
mlserver serve

# Quick status check
mlserver status

# List classifiers in the config
mlserver list-classifiers

# Smoke-test a running server
mlserver test
```

### Aliases

Add to your shell profile:

```bash
alias mls='mlserver serve'
alias mlb='mlserver build'
alias mlp='mlserver push'
alias mlv='mlserver validate'
```

### Docker Integration

```bash
# Build and run locally
mlserver build --classifier sentiment
mlserver run --classifier sentiment --port 8000

# Or run the image directly
docker run -p 8000:8000 sentiment:latest
```
