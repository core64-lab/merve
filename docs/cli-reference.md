# CLI Reference

## Overview

Merve ships a single command-line interface, `merve`, built on Typer with rich, colorful output. It covers the full lifecycle: serving, validation, versioning, container builds, and deployment helpers.

## Installation

The CLI is installed with the package:

```bash
pip install merve

merve --help
```

## Commands Overview

```bash
merve --help  # Show all commands with rich formatting
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
│ status             📊 Show Merve status and system info          │
│ init               🎬 Initialize a new classifier project        │
│ init-github        🔧 Initialize GitHub Actions CI/CD workflow   │
│ validate           Validate configuration without starting       │
│ doctor             Diagnose common issues                        │
│ test               Test prediction against a running server      │
│ schema             Generate JSON schema for mlserver.yaml        │
╰──────────────────────────────────────────────────────────────────╯
```

> Note: There is no shell completion support (`--install-completion` is disabled). `merve` is the primary entry point; the old `mlserver` command still exists as a **deprecated alias** that prints a deprecation notice to stderr and will be removed in a future release (the tool was renamed to avoid a collision with Seldon's `mlserver`). The importable Python module (`mlserver`), the `mlserver.yaml` config file, and the `MLSERVER_*` environment variables are unchanged.

### Removed flags (0.5.0)

Removed flag spellings are rejected with **exit code 2 and a pointer to the replacement** — never a bare "No such option" and never a silent reinterpretation:

| Removed spelling | Where | Use instead |
|------------------|-------|-------------|
| `-p` as `--path` | `version`, `init`, `init-github`, `doctor` | `-C` / `--path` (`-p` means `--port` only) |
| `-v` as `--volume` | `run` | `--volume` (long-only; `-v` means `--verbose` only) |
| `--version-source` | `push` | Nothing — the pushed version always comes from git tags |

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
| `doctor` | Full diagnosis: per-check status/message/suggestion, recommendations, pass/warn/fail/skip summary |
| `tag --status` | Per-classifier tag status (status mode only) |

```bash
merve images --json | jq '.images[].tag'
merve validate --json | jq '.valid'
merve tag --status --json | jq '.classifiers'
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
merve serve [CONFIG] [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `CONFIG` | Path to config file | `mlserver.yaml` |
| `--path`, `-C` | Project directory (`mlserver.yaml` auto-detected inside) | `.` |
| `--classifier`, `-c` | Classifier to serve (for multi-classifier configs) | See selection note |
| `--host` | Override host address | From config |
| `--port`, `-p` | Override port number | From config |
| `--workers`, `-w` | Number of worker processes | From config |
| `--reload` | Enable auto-reload for development | `false` |
| `--log-level`, `-l` | Set log level (`DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL`) | From config |

> `--log-level` only overrides the YAML `server.log_level` when it is explicitly passed on the command line.

> **Classifier selection** (multi-classifier configs): the `--classifier` flag wins, then the `MLSERVER_CLASSIFIER` environment variable (deploy-time selection on commit images), then the config's `default_classifier`. An invalid `MLSERVER_CLASSIFIER` value is a hard startup error that lists the available classifiers (also enforced in the uvicorn app factory used by multi-worker mode).

#### Examples
```bash
# Auto-detect configuration (mlserver.yaml)
merve serve

# Specific configuration
merve serve mlserver.yaml

# Override settings
merve serve --port 9000 --workers 4

# Multi-classifier selection
merve serve mlserver.yaml --classifier catboost-survival

# Development mode
merve serve --reload --log-level DEBUG
```

---

### `version` - Version Information

Display version information for the classifier project. Use `--detailed` to include MLServer tool version, commit, and installation source.

#### Syntax
```bash
merve version [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-C` | Path to classifier project | `.` |
| `--classifier`, `-c` | Classifier name (for multi-classifier configs) | None |
| `--json` | Output as JSON | `false` |
| `--detailed` | Show detailed MLServer tool information | `false` |

#### Examples
```bash
# Basic version (classifier only)
merve version

# JSON format
merve version --json

# With MLServer tool details
merve version --detailed

# Specific classifier in a multi-classifier repo
merve version --classifier sentiment
```

#### Use Cases
```bash
# Get full version info for bug reports
merve version --detailed --json > version-info.json

# Verify MLServer commit in a build pipeline
merve version --detailed | grep -i commit
```

---

### `tag` - Version Tagging & Reproducibility

Manage semantic version tags for each classifier. Git tags are the canonical version source.

#### Tag Format

`merve tag` creates **canonical** `<classifier>/vX.Y.Z` tags (slash-namespaced), e.g. `sentiment/v1.0.3`. The MLServer commit is no longer encoded in the tag name — it is recorded in the annotated-tag message and in the container's OCI image labels instead.

The **legacy** `<classifier>-vX.Y.Z-mlserver-<hash>` form (e.g. `sentiment-v1.0.3-mlserver-b5dff2a`) is still **read** everywhere — status, `build`/`push` validation, and version listing all parse both forms — so tags created before this change keep working indefinitely.

> **CI note:** the workflow generated by `merve init-github` (workflow version 3) triggers on the canonical tag pattern `'*/v*'` and parses `<classifier>/vX.Y.Z` tags. Regenerate older workflows with `merve init-github --force`.

#### Syntax
```bash
# Show tag status for all classifiers
merve tag

# Create a version tag for a specific classifier
merve tag --classifier <name> <major|minor|patch>
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `BUMP_TYPE` | Version bump type: `major`, `minor`, or `patch` | None (status mode) |
| `--classifier`, `-c` | Classifier name to tag (required for tagging) | None |
| `--config` | Config file path | `mlserver.yaml` |
| `--message`, `-m` | Tag message | `Release <classifier> vX.Y.Z` |
| `--path`, `-C` | Path to classifier project | `.` |
| `--allow-missing-mlserver` | Allow tagging even if the MLServer commit cannot be determined (dev/testing only) | `false` |
| `--status` | Show tag status for all classifiers (default when no bump type given) | `false` |
| `--json` | Output tag status as JSON (status mode only) | `false` |

#### Examples

**Create Tags:**
```bash
# Patch release (bug fixes)
merve tag --classifier sentiment patch

# Minor release (new features, backward compatible)
merve tag --classifier intent minor

# Major release (breaking changes)
merve tag --classifier fraud major
```

**Output:**
```
✓ Created tag: sentiment/v1.0.3

  📝 Version: 1.0.2 → 1.0.3 (patch bump)
  🔧 MLServer commit: b5dff2a
  📦 Classifier commit: c5f9997

Next steps:
  1. Push tags to remote: git push --tags
  2. Build container: merve build --classifier sentiment
  3. Push to registry: merve push --registry <url>
```

**View Status:**
```bash
merve tag
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
   merve tag --classifier sentiment patch
   # Creates: sentiment/v1.0.3   (MLServer commit recorded in the tag message + OCI labels)
   ```

2. **Build Container**: Uses tag information
   ```bash
   merve build --classifier sentiment/v1.0.3
   ```
   - Validates current code matches tag
   - Adds container labels with all commit info
   - Warns if code has changed since tag

3. **Reproduce Exact Build**: Checkout and rebuild
   ```bash
   git checkout sentiment/v1.0.3
   merve build --classifier sentiment
   ```

#### Multi-Classifier Tagging

Each classifier is tagged independently:
```bash
merve tag --classifier sentiment patch  # v1.0.3
merve tag --classifier intent minor     # v2.1.0

# View all tags
merve tag
```

---

### `build` - Build Docker Container

Build a Docker container for the classifier project with full reproducibility tracking.

**Build-once / deploy-many (default for multi-classifier repos):** `merve build` without `--classifier` builds **one commit image** tagged `<repo>:<git-sha>` and `<repo>:latest` that bundles every classifier — the classifier is chosen at deploy/run time via `MLSERVER_CLASSIFIER`. Passing `--classifier` on a multi-classifier repo still validates a full version tag, but the selection is ignored for the image content (the commit image always bundles all classifiers). Use `--per-classifier-image` (requires `--classifier`) as the escape hatch to bake one image per classifier when conflicting dependencies cannot share an image. Single-classifier repos always build a single image.

The `--classifier` parameter accepts a simple name or a full version tag (canonical or legacy):
- Simple: `--classifier sentiment`
- Canonical tag: `--classifier sentiment/v1.0.0`
- Legacy tag: `--classifier sentiment-v1.0.0-mlserver-b5dff2a`

When using a full tag, the build validates that your current code matches the tag's expected commits and warns on mismatches.

#### Syntax
```bash
merve build [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--classifier`, `-c` | Classifier to build (simple name, or full tag `name/vX.Y.Z` / legacy `name-vX.Y.Z-mlserver-hash`) | Auto-detect |
| `--path`, `-C` | Path to classifier project | `.` |
| `--config` | Config file to use | Auto-detected |
| `--registry` | Container registry URL | None |
| `--tag-prefix` | Tag prefix for container names | None |
| `--build-arg` | Build arguments (`key=value`, repeatable) | None |
| `--no-cache` | Do not use cache when building | `false` |
| `--platform` | Target platform for the image, e.g. `linux/amd64` (single platform only; cross-architecture builds need BuildKit with binfmt/QEMU or `docker buildx`) | Host platform |
| `--verbose`, `-v` | Verbose output | `false` |
| `--force` | Skip validation prompts and continue with build | `false` |
| `--per-classifier-image` | Escape hatch: build one baked image per classifier instead of the single commit image (requires `--classifier`) | `false` |

#### Examples

**Simple Build:**
```bash
# Single-classifier repo: build the classifier image
# Multi-classifier repo: build ONE commit image bundling all classifiers
merve build

# Escape hatch: bake a single classifier into its own image
merve build --per-classifier-image --classifier sentiment
```

**Build with a Full Version Tag:**
```bash
# Build exact tagged version (with validation)
merve build --classifier sentiment/v1.0.3
```

**Validation Warnings** (a legacy `-mlserver-<hash>` tag additionally validates the MLServer commit, which canonical tags no longer encode):
```bash
# If code doesn't match tag
merve build --classifier sentiment-v1.0.2-mlserver-old123

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
merve build --classifier sentiment-v1.0.2-mlserver-old123 --force
```

**Advanced Builds:**
```bash
# Fresh build without Docker cache
merve build --no-cache

# Custom registry and build arguments
merve build --registry gcr.io/project --build-arg PIP_INDEX_URL=https://pypi.internal
```

#### Container Labels

Built containers include comprehensive labels for full traceability (RFC 0001 D5):

```dockerfile
# Standard OCI labels
LABEL org.opencontainers.image.title="sentiment"            # plain classifier name
LABEL org.opencontainers.image.description="ML classifier: sentiment"
LABEL org.opencontainers.image.source="https://github.com/org/repo"
LABEL org.opencontainers.image.revision="c5f99978a3b2..."   # full project git commit
LABEL org.opencontainers.image.version="1.0.3"
LABEL org.opencontainers.image.created="2026-07-04T10:30:00Z"

# Custom merve labels
LABEL dev.merve.classifier="sentiment"
LABEL dev.merve.mlserver_version="0.5.0"
LABEL dev.merve.mlserver_commit="b5dff2a"

# Legacy labels (kept for one release for dashboard continuity)
LABEL com.classifier.name="sentiment"
LABEL com.classifier.version="1.0.3"
LABEL com.classifier.git_tag="sentiment/v1.0.3"
LABEL com.classifier.git_commit="c5f9997"
LABEL com.classifier.git_branch="main"
LABEL com.mlserver.version="0.5.0"
LABEL com.mlserver.commit="b5dff2a"
```

> **Commit images** (multi-classifier build-once default) bundle every classifier, so no single release version applies: their `org.opencontainers.image.version` is the **short git commit**; release versions are applied later as registry tag aliases by `merve push`.

#### Reproducibility

Container labels enable exact reproduction:
```bash
# Inspect container to get tag
docker inspect my-container | grep com.classifier.git_tag

# Checkout exact version
git checkout sentiment/v1.0.3

# Rebuild identically
merve build --classifier sentiment
```

---

### `push` - Push Container

Push a built container to a registry. Requires a tagged commit for the specific classifier (or `--force`).

**Multi-classifier repos (build-once/deploy-many):** `push --classifier X` does **not** rebuild anything. It validates that HEAD carries the canonical git tag `X/vN.N.N` (with `--force`, it falls back to the classifier's latest release tag), then applies the release as **registry tag aliases on the already-built commit image** — `<repo>:X-vN.N.N` and `<repo>:X-latest`, both resolving to the same image digest. Single-classifier repos keep the classic per-image push.

#### Syntax
```bash
merve push [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--registry`, `-r` | Container registry URL | **Required** |
| `--classifier`, `-c` | Classifier to push (required for multi-classifier configs) | None |
| `--path`, `-C` | Path to classifier project | `.` |
| `--tag-prefix` | Tag prefix for container names | None |
| `--force`, `-f` | Force push even if not on tagged commit or tag exists | `false` |

> **`--version-source` was removed (RFC 0001 D3).** Git tags are the canonical version source, so the pushed version always comes from the classifier's release tag. Passing the flag exits with code 2 and a message pointing at this change.

#### Examples
```bash
# Multi-classifier repo: apply the release aliases on the commit image (no rebuild)
merve push --registry gcr.io/project --classifier sentiment

# Force push (uses the classifier's latest release tag if HEAD is untagged)
merve push --registry gcr.io/project --classifier sentiment --force
```

---

### `images` - List Docker Images

List Docker images for the classifier project. For multi-classifier configs, use `--classifier` to filter.

#### Syntax
```bash
merve images [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-C` | Path to classifier project | `.` |
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

Remove Docker images for the classifier project. For multi-classifier repos, `--classifier` restricts removal to that classifier's per-classifier/alias images.

#### Syntax
```bash
merve clean [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-C` | Path to classifier project | `.` |
| `--classifier`, `-c` | Only remove images for this classifier | None |
| `--force`, `-f` | Force removal without confirmation | `false` |

#### Examples
```bash
# Interactive clean
merve clean

# Remove without confirmation
merve clean --force

# Only remove one classifier's images
merve clean --classifier sentiment
```

---

### `run` - Run Docker Container

Run a built Docker container for the classifier locally. On a multi-classifier repo, `run --classifier X` runs the commit image and passes `-e MLSERVER_CLASSIFIER=X` so the container serves the selected classifier.

#### Syntax
```bash
merve run [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--classifier`, `-c` | Classifier to run (required for multi-classifier configs) | None |
| `--path`, `-C` | Path to classifier project | `.` |
| `--port`, `-p` | Port to expose the container on | `8000` |
| `--version` | Specific version to run | latest |
| `--detach`, `-d` | Run container in background | `false` |
| `--name` | Container name | Auto-generated |
| `--env`, `-e` | Environment variables (`KEY=value`, repeatable) | None |
| `--volume` | Volume mounts (`host:container`, repeatable; **long-only** — `-v` means `--verbose` elsewhere and is rejected here with a pointer) | None |

#### Examples
```bash
# Run latest build on port 8000
merve run --classifier sentiment

# Run a specific version in the background
merve run --classifier sentiment --version 1.0.2 --detach

# Custom port, env vars, and mounts
merve run -c sentiment -p 9000 -e LOG_LEVEL=DEBUG --volume ./models:/app/models
```

---

### `list-classifiers` - List Available Models

Show all classifiers in a multi-classifier configuration.

#### Syntax
```bash
merve list-classifiers [CONFIG] [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `CONFIG` | Path to multi-classifier config file | Auto-detect (`mlserver.yaml`) |
| `--path`, `-C` | Path to classifier project | `.` |
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

Show Merve status and system info (local environment overview, Docker availability, config detection).

#### Syntax
```bash
merve status [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-C` | Path to classifier project | `.` |
| `--json` | Output as JSON (machine-readable) | `false` |

#### Examples
```bash
merve status
merve status --json
```

---

## Project Setup Commands

### `init` - Initialize a New Project

Initialize a new MLServer classifier project. Creates all necessary files:
- `mlserver.yaml` (configuration file)
- `<predictor>.py` (skeleton predictor class)
- `.github/workflows/ml-classifier-container-build.yml` (CI/CD workflow)
- `.gitignore` (Python/ML project gitignore)
- `AGENTS.md` (operating guide for coding agents and humans)

Existing files are never overwritten unless `--force` is used.

#### Syntax
```bash
merve init [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-C` | Path to initialize project in | `.` |
| `--classifier`, `-c` | Classifier name | Directory name |
| `--predictor-file` | Name of predictor Python file (without `.py`) | Derived from classifier |
| `--predictor-class` | Name of predictor class | Derived from classifier |
| `--no-github` | Skip GitHub Actions workflow creation | `false` |
| `--force`, `-f` | Overwrite existing files | `false` |

#### Examples
```bash
merve init --classifier sentiment-analyzer
merve init --classifier my-model --predictor-file custom_predictor
merve init --no-github
```

---

### `init-github` - Initialize CI/CD Workflow

Initialize a GitHub Actions workflow for automated container builds. Sets up automated building and publishing of containers to a container registry when canonical `<classifier>/vX.Y.Z` version tags (created with `merve tag`) are pushed.

This command:
- Creates `.github/workflows/ml-classifier-container-build.yml`
- Configures automated Docker builds on tag push
- Sets up container publishing to GitHub Container Registry
- Auto-detects your GitHub repository information

Note: `merve init` already creates this workflow; run `init-github` separately only to add CI/CD to an existing project.

#### Syntax
```bash
merve init-github [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-C` | Path to classifier project | `.` |
| `--python-version` | Python version for CI/CD workflow | `3.11` |
| `--registry` | Container registry | `ghcr.io` |
| `--force`, `-f` | Overwrite existing workflow files | `false` |

#### Examples
```bash
merve init-github
merve init-github --python-version 3.12 --registry ghcr.io
```

---

### `init-agents` - Generate/Refresh AGENTS.md

Generate or refresh `AGENTS.md` — the operating guide for coding agents and humans working in a classifier repo. The content is generated from a template shipped with the installed merve version (never hand-written) and carries a template-version stamp, so it can be regenerated after upgrades instead of drifting. It auto-detects single vs multi-classifier config and lists the classifiers, and `merve doctor` checks the stamp for staleness.

Note: `merve init` already creates `AGENTS.md` for new projects; run `init-agents` separately to refresh an existing repo (use `--force` after upgrading merve to pick up template changes).

#### Syntax
```bash
merve init-agents [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--path`, `-C` | Path to classifier project | `.` |
| `--force`, `-f` | Overwrite existing AGENTS.md | `false` |

#### Examples
```bash
merve init-agents
merve init-agents --force
merve init-agents -C ./my-project
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
merve validate [CONFIG] [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `CONFIG` | Path to config file | Auto-detect (`mlserver.yaml`) |
| `--path`, `-C` | Path to classifier project | `.` |
| `--strict`, `-s` | Fail on warnings | `false` |
| `--check-imports/--no-check-imports` | Check predictor imports | `check-imports` |
| `--verbose`, `-v` | Show detailed output | `false` |
| `--json` | Output as JSON (machine-readable) | `false` |

Passing a file path validates that named file (e.g. `merve validate staging.yaml`). Unknown configuration keys are reported as warnings. `--json` emits a `valid` flag plus per-check results and exits non-zero when invalid.

#### Examples
```bash
# Validate auto-detected config
merve validate

# Validate a specific config with verbose output
merve validate mlserver.yaml -v

# Strict mode - fail on any warning
merve validate --strict

# Skip import checking
merve validate mlserver.yaml --no-check-imports
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
merve doctor [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--verbose`, `-v` | Show detailed diagnostics | `false` |
| `--path`, `-C` | Project path to diagnose | `.` |
| `--json` | Output as JSON (machine-readable) | `false` |

`--json` emits a single JSON document — `success`, the full `checks` list (each with `name`, `status`, `message`, `suggestion`, `details`), `recommendations`, and a `summary` of pass/warn/fail/skip counts — and keeps the same exit code as human mode (`1` when a check fails, else `0`).

#### Examples
```bash
merve doctor
merve doctor -v
merve doctor --path ./my-project
merve doctor --json
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
merve test [options]
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
merve test --data '{"feature1": 1.5, "feature2": 2.0}'

# Test with JSON file
merve test --file sample_request.json

# Test another server / endpoint
merve test --url http://localhost:8080 --endpoint /predict_proba
```

---

### `schema` - Generate JSON Schema

Generate a JSON schema for `mlserver.yaml` configuration files, enabling IDE autocompletion and validation.

#### Syntax
```bash
merve schema [options]
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
merve schema

# Save schema with setup instructions
merve schema -o .mlserver/schema.json --setup

# Full VSCode setup
merve schema -o .mlserver/schema.json --vscode --setup

# Generate for multi-classifier configs only
merve schema --type multi -o schema.json
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
| `MLSERVER_CLASSIFIER` | Classifier to serve from a multi-classifier config — the **deploy-time selector** for build-once commit images. Honored by `merve serve` and the uvicorn app factory; `merve run --classifier X` passes it to the container; baked in only by `--per-classifier-image` builds. Precedence: `--classifier` flag > env var > `default_classifier`. An invalid value is a hard startup error listing the available classifiers |
| `MLSERVER_CONFIG_FILE` | Config file name recorded in response metadata (set in built containers) |
| `MLSERVER_GIT_COMMIT` / `MLSERVER_GIT_TAG` / `MLSERVER_GIT_BRANCH` / `MLSERVER_GIT_URL` | Classifier git metadata overrides (baked into containers at build time) |
| `MLSERVER_API_COMMIT` / `MLSERVER_API_TAG` / `MLSERVER_API_BRANCH` | MLServer tool git metadata overrides (baked into containers at build time) |
| `MLSERVER_SOURCE_PATH` / `MLSERVER_BUILD_TIME` | MLServer source path and build timestamp (baked into containers at build time) |

There are no `MLSERVER_HOST`, `MLSERVER_PORT`, `MLSERVER_WORKERS`, `MLSERVER_METRICS`, or similar per-setting override variables — use the YAML config or CLI flags instead.

> **Removed (RFC 0001 D12):** the old `GlobalSettings` singleton and `global_config.yaml` no longer exist. There are no `MLSERVER_GLOBAL_CONFIG`, `MLSERVER_LOG_FILE`, `MLSERVER_PID_FILE`, `MLSERVER_VENV_PATH`, or `MLSERVER_EXEC_PATH` variables. Package defaults live in `mlserver/defaults.py`; if a stray `global_config.yaml` is found in the working directory the server logs a one-time "no longer read" warning. Configure everything through `mlserver.yaml` plus the environment variables above.

## Configuration File Detection

The CLI resolves the configuration file as follows:

1. Explicitly provided path (e.g. `merve serve my-config.yaml`)
2. `mlserver.yaml` in the current directory

If neither is found, the command exits with an error asking you to create `mlserver.yaml` or pass a path.

## Advanced Usage

### Multi-Classifier Deployment

```bash
# List available classifiers
merve list-classifiers mlserver.yaml

# Serve a specific classifier
merve serve mlserver.yaml --classifier production

# Serve multiple classifiers as separate processes
merve serve mlserver.yaml --classifier staging --port 8001 &
merve serve mlserver.yaml --classifier production --port 8002 &
```

### Development Workflow

```bash
# Start with auto-reload
merve serve --reload --log-level DEBUG

# Validate the configuration
merve validate -v

# Diagnose the environment
merve doctor

# Send a test request
merve test --data '{"feature1": 1.5, "feature2": 2.0}'
```

### Production Deployment

```bash
# Tag a release
merve tag --classifier sentiment patch

# Build the container (multi-classifier repos: ONE commit image for all classifiers)
merve build

# Push to registry (multi-classifier repos: applies the release aliases, no rebuild)
merve push --registry gcr.io/project --classifier sentiment

# Deploy to Kubernetes (commit images: set MLSERVER_CLASSIFIER in the pod spec)
kubectl apply -f deployment.yaml
```

## Debugging

### Verbose Output

```bash
# Debug logging
merve serve --log-level DEBUG

# Verbose validation / diagnostics
merve validate -v
merve doctor -v
```

### Configuration Validation

```bash
# Validate auto-detected config
merve validate

# Validate a specific file
merve validate mlserver.yaml
```

## Common Issues

### Port Already in Use

```bash
# Use different port
merve serve --port 8001

# Kill existing process
lsof -ti:8000 | xargs kill -9
```

### Module Not Found

```bash
# Verify predictor imports
merve validate --check-imports

# Full environment diagnosis
merve doctor -v
```

### Permission Denied

```bash
# Use sudo for system ports
sudo merve serve --port 80

# Or use high port
merve serve --port 8080
```

## Tips and Tricks

### Quick Commands

```bash
# Serve with defaults
merve serve

# Quick status check
merve status

# List classifiers in the config
merve list-classifiers

# Smoke-test a running server
merve test
```

### Aliases

Add to your shell profile:

```bash
alias mls='merve serve'
alias mlb='merve build'
alias mlp='merve push'
alias mlv='merve validate'
```

### Docker Integration

```bash
# Build and run locally (run passes -e MLSERVER_CLASSIFIER=sentiment on commit images)
merve build
merve run --classifier sentiment --port 8000

# Or run the image directly (commit images need the classifier selector)
docker run -p 8000:8000 -e MLSERVER_CLASSIFIER=sentiment my-repo:latest
```
