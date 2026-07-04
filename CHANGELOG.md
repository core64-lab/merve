# Changelog

All notable changes to Merve (formerly "MLServer FastAPI Wrapper") will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- Release builds from a git-installed merve now pin the framework via its immutable release tag on the same git source (`pip install "git+<url>@vX.Y.Z"`) instead of `merve==X.Y.Z` — merve is not published on PyPI, so the index pin produced Dockerfiles that could not build (RFC 0001 §8 A8). Index installs still pin `merve==X.Y.Z`.

---

## [0.5.0] - 2026-07-04

RFC 0001 Wave 2: the breaking batch — rename, tag format, image strategy, and
flag cleanup — plus the post-audit fixes that make the documented behavior real,
and the agent-facing surface from RFC 0002.
**Upgrade steps: [docs/migration-0.5.md](docs/migration-0.5.md).**

### Breaking
- Renamed the distribution and command to **`merve`**; `mlserver` is a deprecated alias for one transition release (it prints a stderr notice). The `mlserver` module/import path, `mlserver.yaml`, and the `MLSERVER_*` environment variables are unchanged.
- CLI short flags: `-p` is `--port` only; `--path`/`-C` is the project directory on every path-taking command; `run`'s volume flag is `--volume` (long-only); `-v` is `--verbose` only. Removed spellings (`-p` as path on `version`/`init`/`init-github`/`doctor`, `-v` as volume on `run`) exit with code 2 **and a pointer to the replacement** — never a bare "No such option".
- `merve tag` now writes the canonical `<classifier>/vX.Y.Z` tag format. Legacy `<classifier>-vX.Y.Z[-mlserver-<hash>]` tags remain readable forever. Generated CI workflows (v3) trigger on `*/v*`.
- Build-once/deploy-many is the default for multi-classifier repos: `merve build` (no `--classifier`) produces one commit image (`<repo>:<git-sha>` + `<repo>:latest`, no baked classifier); `merve push --classifier X` applies registry tag aliases (`<repo>:X-vN.N.N`, `<repo>:X-latest`) on that same image after validating the git tag — no rebuild. `--per-classifier-image` (requires `--classifier`) restores one baked image per classifier.
- Removed `push --version-source`: passing it exits with code 2 and a pointer — the pushed version always comes from git tags (the tag at HEAD, or with `--force` the classifier's latest release tag).
- Removed the `GlobalSettings` singleton and `global_config.yaml`.

### Added
- **AGENTS.md scaffolding** (RFC 0002 A1/A2): `merve init` now also generates a version-stamped `AGENTS.md` operating guide (single/multi-classifier aware) into new projects; the new `merve init-agents [--force]` command (re)generates it for existing repos; `merve doctor` reports a missing or stale AGENTS.md (advisory — never a failure).
- **`merve doctor --json`** (RFC 0002 A3): machine-readable diagnosis — per-check `status`/`message`/`suggestion`/`details`, recommendations, and a pass/warn/fail/skip summary; JSON on stdout, exit codes identical to human mode.
- **Agent discovery docs** (RFC 0002 A4): `llms.txt` at the repo root and `docs/agent-guide.md` ("Driving Merve from an Agent") documenting the CLI-as-tool-interface contract: one JSON document on stdout, diagnostics on stderr, stable exit codes (0/1/2), and errors that carry the fix.
- `MLSERVER_CLASSIFIER` is now honored by `merve serve` and the uvicorn app factory (deploy-time classifier selection on commit images). Precedence: `--classifier` flag > `MLSERVER_CLASSIFIER` env > config `default_classifier`. An invalid env value fails loudly, listing the available classifiers.
- `/healthz` returns HTTP 503 `{"status": "loading", "model": null}` until the predictor is loaded, then 200 `{"status": "ok", "model": ...}` — readiness probes no longer read "ok" for a model that cannot serve.
- The served OpenAPI spec now carries request-body examples on the prediction endpoints (top-level shape first, deprecated wrapper last) and the `ProbaResponse` schema for `/predict_proba`.
- `clean --classifier`/`-c` restricts image removal to one classifier's per-classifier/alias images.
- `build --platform` (single target platform) and `build --per-classifier-image`.
- Generated Dockerfiles pin clean release versions via `pip install merve==X.Y.Z` and skip the wheel build (D16). Only strict `X.Y.Z` versions count as releases — dev/rc/alpha/beta/post/local versions use the wheel path with a loud non-reproducibility warning.

### Changed
- Image labels: standard OCI annotations `org.opencontainers.image.{title,description,source,revision,version,created}` plus `dev.merve.{classifier,mlserver_version,mlserver_commit}`; the legacy `com.mlserver.*`/`com.classifier.*` labels are kept for one release. Commit images set `org.opencontainers.image.version` to the short git commit (they bundle all classifiers, so no single release version applies) and `title` to the plain classifier/repo name.
- `classifier.version` in `mlserver.yaml` now logs its deprecation warning at config load (once per process); the field is display-only and never feeds builds, tags, or pushes.
- `doctor` runs its predictor-import check through the production import path (same isolation as the server).
- Remaining `mlserver` command spellings in generated CI workflows and CLI output renamed to `merve`.

---

## [0.4.0] - 2026-07-04

RFC 0001 Waves 0–1: stabilization, guardrails, and backward-compatible
additions. Everything deprecated here keeps working through 0.5 with a warning.

### Deprecated

| Deprecated | Replacement | Removal target |
|------------|-------------|----------------|
| Request `{"payload": {...}}` wrapper (warns once per process) | Top-level keys: `records` / `instances` / `ndarray` / `inputs` / `features` | 1.0 |
| `api.response_format: custom` and `api.extract_values` (load-time warnings) | `standard` or `passthrough`; return the desired structure from the predictor | 1.0 |
| `classifier.version` in `mlserver.yaml` (display-only; load-time warning since 0.5.0) | Git tags via `merve tag <major\|minor\|patch>` | None scheduled — stays display-only |
| `mlserver` command alias (effective with the 0.5.0 rename; stderr notice) | `merve` | 0.6.0 (one transition release) |

### Added
- Dual request shapes (top-level keys alongside the deprecated wrapper).
- A `Predictor` Protocol and a `"module:ClassName"` predictor spec. Import isolation: user modules load under the `merve._user.*` namespace without deleting foreign `sys.modules` entries; the project directory is appended (never front-inserted) to `sys.path` for sibling imports.
- `api.retry_after_seconds`, `--json` output on read commands, OCI-standard image labels, and two-stage (multi-stage) Dockerfiles.
- CI workflow, docs-drift gate, ruff/mypy config, and a tracked CHANGELOG (Wave 0).

### Stabilization sprint (2026-07-03)

A bug-fix and cleanup pass across the server, CLI, and build tooling (see RFC 0001).

#### Added
- `observability.log_payloads` is now actually implemented — request/response payloads are logged when enabled.
- Warnings for unknown top-level configuration keys, catching typos in `mlserver.yaml`.
- `max_concurrent_predictions: 0` disables concurrency limiting entirely.
- OCI-relevant build metadata fixes: generated Dockerfiles now emit port-aware `EXPOSE` and `HEALTHCHECK` instructions.

#### Changed
- Coverage gate moved from pytest `addopts` to `make ci-test`, so plain `pytest` runs no longer fail on the coverage threshold.
- Removed the `nbformat` dependency.
- `catboost` and `scikit-learn` moved out of the core dependencies into the `[ml]` extra.
- `examples/` is no longer shipped in built wheels.
- Documentation overhauled to match the actual implementation.
- Historical planning documents archived under `docs/archive/`.

#### Fixed
- Request metrics were never recorded (middleware initialization-order bug).
- Model warmup never ran (missing `numpy` import).
- Structured logging crashed when log calls passed `exc_info`.
- `NameError` in the JSON-serialization fallback path.
- `serve --reload` exited immediately instead of serving.
- `validate` ignored the config-file argument it was given.
- The CLI clobbered the YAML `log_level` even when no `--log-level` flag was passed.
- Stale `_version_info.py` embedded the wrong mlserver commit in tags and metadata.
- Registry manifest check always raised `ValueError`.
- `push` was broken for multi-classifier repositories, and partial push failures incorrectly exited with code 0.
- List-format multi-classifier configurations failed to load.
- Multi-worker mode crashed in the app-factory setup.
- `/info` no longer races on `os.chdir`.
- Feature-order cache is now thread-safe.
- `reload_settings()` was a no-op.

---

## [0.3.0] - 2025-10-27 - Hierarchical Versioning Release

### Major Features

#### Hierarchical Git Tagging System
- **Complete reproducibility** with `<classifier>-v<X.X.X>-mlserver-<hash>` tag format
- Captures both classifier code commit AND MLServer tool commit
- Enables rebuilding identical containers months or years later
- Full traceability from production back to exact source code

#### Version Management CLI
- `mlserver tag` - Display version status for all classifiers
- `mlserver tag --classifier <name> <patch|minor|major>` - Create hierarchical tags with semantic versioning
- Beautiful rich table output showing version status, MLServer commits, and recommendations
- Automatic version bumping (patch: 1.0.0 → 1.0.1, minor: 1.0.0 → 1.1.0, major: 1.0.0 → 2.0.0)

#### Enhanced Container Labels
- 17 OCI-compliant container labels
- Hierarchical git tag embedded in every container
- Complete version metadata (classifier version, commits, MLServer version)
- Enables audit trails and compliance tracking

### 🔧 Enhancements

#### CLI Improvements
- Enhanced `mlserver build` with validation warnings when code doesn't match tag
- Enhanced `mlserver version --detailed` showing MLServer tool information
- Rich CLI output with colored status indicators and formatted tables
- Helpful next-step recommendations after operations

#### Multi-Classifier Support
- Independent version tracking per classifier in multi-classifier repositories
- Status table shows all classifiers with their versions and MLServer commits
- Each classifier can be tagged and versioned independently

#### Git Integration
- Automatic MLServer commit detection from installed package
- Support for git-based installations (editable mode)
- Intelligent repository detection and metadata extraction
- Clean git tag management with hierarchical format

### 📚 Documentation

#### Comprehensive Documentation Updates
- **CLI Reference**: Complete documentation of tag, build, and version commands
- **Deployment Guide**: GitHub Actions workflows for CI/CD integration (3 workflow options)
- **Development Guide**: Version management in development workflow with practical scenarios
- **Documentation Index**: Updated navigation with versioning tasks

#### New Documentation Sections
- Hierarchical tag format explanation and examples
- Migration path from simple tags to hierarchical format
- Reproducibility testing workflows
- Version bumping guidelines (when to use patch/minor/major)
- Common version management scenarios with solutions

#### Code Examples
- 47+ runnable code examples across documentation
- 3 production-ready GitHub Actions workflows
- 12+ visual CLI output examples
- 6+ scenario walkthroughs (hotfixes, migration, etc.)

### 🐛 Bug Fixes

#### Multi-Classifier Detection
- Fixed multi-classifier config detection to support both `dict` and `list` formats
- Enhanced `list_available_classifiers()` to parse both configuration styles

#### Test Improvements
- Fixed test assertions for `mlserver` command (was `ml_server`)
- Added 6 new tests for `validate_push_readiness()`
- Added container label validation tests

### 📦 New Files

#### Core Implementation
- `mlserver/version_control.py` - Complete version control and git integration (782 lines)
- `mlserver/_version_info.py` - Version information module
- `UPDATED_VERSIONING.md` - Version strategy documentation

#### Testing
- `tests/unit/test_version_control.py` - 21 unit tests for version control
- `tests/integration/test_hierarchical_versioning.py` - 20 integration tests
- `tests/integration/test_container_labels_versioning.py` - 5 container label tests
- `TEST_RESULTS_PHASE1_2.md` through `TEST_RESULTS_PHASE5.md` - Test reports
- `TEST_REVIEW_PHASE5.md` - Comprehensive test quality review
- `TEST_IMPROVEMENT_BACKLOG.md` - Prioritized test improvement roadmap

#### Documentation
- `DOCUMENTATION_UPDATES_PHASE6.md` - Phase 6 documentation summary
- `PHASE7_INTEGRATION_TEST_REPORT.md` - Integration testing report
- Updated `docs/cli-reference.md`, `docs/deployment.md`, `docs/development.md`, `docs/INDEX.md`

### ⚙️ Technical Details

#### Version Control System
- GitVersionManager class for all version control operations
- Hierarchical tag parsing with regex validation
- MLServer commit hash detection from package installation
- Tag status tracking with commit counting
- Push readiness validation (uncommitted changes detection)

#### Container Integration
- Enhanced `generate_container_labels()` with hierarchical tag support
- 17 labels including classifier, MLServer, and OCI standard labels
- Git information embedding (commit, branch, tag, URL)
- MLServer tool metadata (version, commit, installation source)

#### Git Operations
- `parse_hierarchical_tag()` - Extract components from hierarchical tags
- `get_mlserver_commit_hash()` - Detect MLServer installation commit
- `tag_version()` - Create hierarchical tags with validation
- `get_all_classifiers_tag_status()` - Status for all classifiers
- `validate_push_readiness()` - Pre-push validation

### 🧪 Testing

#### Test Coverage
- **Unit Tests**: 21 new tests for version control
- **Integration Tests**: 25 new tests for hierarchical versioning and container labels
- **Coverage**: version_control.py at 86%, container.py hierarchical features at 90%+
- **Total Tests**: 46 new tests (45 passing, 1 with known mock issue)

#### Test Categories
- Tag parsing and validation
- Version bumping (patch, minor, major)
- MLServer commit detection
- Multi-classifier scenarios
- Container label generation
- Git integration workflows

### 📊 Performance

- Tag creation: < 1 second
- Container builds: ~45 seconds (includes wheel build from source)
- CLI commands: < 0.5 second response time
- Status table generation: < 0.5 second for multi-classifier repos

### ⚠️ Known Issues

#### Issue #1: `mlserver version` Command in Multi-Classifier Repos
- **Severity**: High (blocks useful CLI command)
- **Description**: `mlserver version --detailed` fails in multi-classifier repositories
- **Root Cause**: `load_classifier_metadata()` looks for `'classifier'` key, not `'classifiers'`
- **Workaround**: Use `mlserver tag` to see MLServer commit information
- **Status**: Documented, workaround available

#### Issue #2: `mlserver images` Command in Multi-Classifier Repos
- **Severity**: Medium (informational command)
- **Description**: `mlserver images` fails in multi-classifier repositories
- **Root Cause**: Same as Issue #1
- **Workaround**: Use `docker images | grep <classifier-name>` directly
- **Status**: Documented, workaround available

### 🔄 Migration Guide

#### For New Projects
Start using hierarchical tags immediately:
```bash
# Create first tag
mlserver tag --classifier your-classifier-name patch

# Build with tag
mlserver build --classifier your-classifier-name

# Push tags to remote
git push --tags
```

#### For Existing Projects
Migrate gradually:
1. Continue using existing tags (fully backward compatible)
2. Start using `mlserver tag` for new versions
3. Update CI/CD to handle hierarchical tag format
4. Benefit from complete reproducibility going forward

No breaking changes - existing tags and containers continue to work.

### 📈 Statistics

- **Code Added**: ~1,600 lines of production code
- **Tests Added**: 46 new tests
- **Documentation Added**: ~816 lines across 4 files
- **Coverage Improvement**: version_control.py from 0% to 86%
- **Pass Rate**: 98% (45/46 tests)

### 🎯 Use Cases

#### Perfect For
- Production ML deployments requiring audit trails
- Teams needing to reproduce historical model versions
- Multi-classifier repositories with independent versioning
- CI/CD pipelines with automatic builds from tags
- Compliance and regulatory requirements for traceability

#### Benefits
- **Complete Reproducibility**: Rebuild exact container from any historical tag
- **Audit Trail**: Full traceability from production to source code
- **Multi-Classifier**: Independent version management per classifier
- **CI/CD Integration**: Automatic builds triggered by tags
- **Developer Experience**: Simple CLI commands, clear visual feedback
- **Container Metadata**: All version info embedded in container labels
- **Backward Compatible**: Existing projects work without changes

### 🙏 Credits

**Development**: 7 phases of implementation (October 2025)
**Testing**: 46 new tests, 3 test reports
**Documentation**: 4 major documentation files updated

---

## [0.2.0] - 2025-01-16 - Previous Features

*(Previous changelog entries preserved here)*

### Features
- AI-powered initialization with `mlserver ainit`
- Multi-classifier repository support
- Comprehensive observability with Prometheus and structured logging
- Flexible input adapters (records, ndarray, auto)
- Container build workflow
- Process-based scaling
- Load testing framework

### Documentation
- Complete API reference
- Configuration guide
- Architecture overview
- Examples and tutorials
- Development guide

---

## [0.1.0] - Initial Release

### Initial Features
- FastAPI-based ML server
- Dynamic predictor loading
- YAML configuration
- Basic metrics and logging
- Docker containerization
- CORS support
- Health check endpoints

---

## Upgrade Guide

### From 0.2.0 to 0.3.0

#### No Breaking Changes
All existing functionality continues to work. The hierarchical versioning system is additive.

#### Optional: Adopt Hierarchical Versioning

**Benefits**:
- Complete reproducibility
- Better version tracking
- Enhanced container labels
- CI/CD integration

**Steps**:
1. Update to 0.3.0: `pip install --upgrade mlserver-fastapi-wrapper`
2. Check current status: `mlserver tag`
3. Create first hierarchical tag: `mlserver tag --classifier <name> patch`
4. Update CI/CD (optional): Use new GitHub Actions workflows from docs
5. Continue normal workflow

**Timeline**: Migrate at your own pace - no urgency, fully backward compatible

---

## Support

- **Issues**: https://github.com/core64-lab/merve/issues
- **Documentation**: https://github.com/core64-lab/merve/tree/main/docs
- **Examples**: https://github.com/core64-lab/merve/tree/main/examples

---

**Note**: 0.4.0 and 0.5.0 are cut in this changelog; their git tags (and the
corresponding package releases) are pending.
