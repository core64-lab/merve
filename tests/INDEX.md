# Test Suite Index and Evolution Plan

## Overview

This document serves as the central index for the MLServer test suite. It tracks test coverage, organization, and evolution plans. **Always update this document when modifying functionality or adding tests.**

## Current Test Status (as of 2026-07-03)

Coverage numbers and pass/fail counts are **not** tracked in this document - they go stale. Get the live numbers instead:

- **Test results**: `pytest tests/`
- **Coverage**: `pytest tests/ --cov=mlserver --cov-report=term-missing`
- **Machine-readable coverage**: `make coverage-export` (writes `coverage.json`)

## Test Organization

### Unit Tests (`tests/unit/`)

**24 test files** (as of 2026-07-03 - run `ls tests/unit/` for the current list) covering:

- Adapters and input handling: `test_adapters.py`, `test_auto_detect.py`, `test_response_handling.py`
- Configuration: `test_config.py`, `test_config_extended.py`, `test_config_simplification.py`, `test_settings.py`, `test_path_resolution.py`, `test_validation.py`
- CLI and project tooling: `test_cli_basic.py`, `test_init_project.py`, `test_doctor.py`, `test_schema_generator.py`
- Server and runtime: `test_server_unit.py`, `test_concurrency_limiter.py`, `test_predictor_loader.py`, `test_errors.py`, `test_logging_conf.py`, `test_metrics.py`
- Multi-classifier support: `test_multi_classifier.py`
- Containers and versioning: `test_container.py`, `test_github_actions.py`, `test_version.py`, `test_version_control.py`

### Integration Tests (`tests/integration/`)

**7 test files** (as of 2026-07-03 - run `ls tests/integration/` for the current list) covering: API endpoints (`test_server_endpoints.py`), observability (`test_observability.py`), concurrency control (`test_concurrency_simplified.py`), complex response formats (`test_complex_response_formats.py`), container labels and hierarchical versioning (`test_container_labels_versioning.py`, `test_hierarchical_versioning.py`), and the end-to-end CLI workflow.

### Load Tests (`tests/load/`)

| File | Purpose | Status |
|------|---------|--------|
| `locustfile.py` | Load testing with Locust | ✅ Ready |

### Test Fixtures (`tests/fixtures/`)

| File | Purpose | Status |
|------|---------|--------|
| `mock_predictor.py` | Mock ML predictor with delay | ✅ Working |
| `mock_classifier_repo.py` | Mock multi-classifier repository | ✅ Working |
| `concurrency_test_config.yaml` | Config for concurrency tests | ✅ Working |

## Recent Fixes Applied

### Configuration Fixes
- ✅ Added required `classifier` and `api` fields to test configs
- ✅ Changed adapter from "records" to "auto" for broader format support
- ✅ Fixed feature ordering in test configurations

### Endpoint Fixes
- ✅ Updated URLs from the old versioned classifier-prefixed paths to flat `/predict`
- ✅ Fixed payload format to use `{"payload": {"records": [...]}}` wrapper
- ✅ Fixed health endpoint assertions (returns "ok" not "healthy")

### Test Infrastructure
- ✅ Renamed TestPredictor to MockPredictor to avoid pytest collection issues
- ✅ Fixed test fixtures to properly initialize predictor state
- ✅ Added proper lifespan handling in test apps

### Version Control Features
- ✅ Added `mlserver tag <major|minor|patch>` command for semantic versioning
- ✅ Implemented safe_push_container with registry tag validation
- ✅ Enhanced /info endpoint to show git-based version information
- ✅ Added comprehensive tests for version control functionality

## Test Evolution Plan

### Phase 1: Core Server Tests ✅ COMPLETED
**File**: `mlserver/server.py` (Current: 72% coverage)
- [x] Fix endpoint tests for unified URLs
- [x] Fix payload format issues
- [ ] Test new /info endpoint with full metadata
- [ ] Test predictor loading with module resolution
- [ ] Test lifespan events (startup/shutdown)
- [ ] Test error handling and fallbacks
- [ ] Test CORS configuration
- [ ] Test concurrency limiter integration

### Phase 2: Adapters Tests ✅ MOSTLY COMPLETE
**File**: `mlserver/adapters.py` (Current: 93% coverage)
- [x] Test RecordsAdapter with various input formats
- [x] Test NdarrayAdapter with nested arrays
- [x] Test AutoAdapter detection logic
- [x] Test error cases (malformed input)
- [x] Test feature ordering cache

### Phase 3: Configuration Tests ✅ GOOD
**File**: `mlserver/config.py` (Current: 94% coverage)
- [x] Test configuration validation
- [ ] Test multi-classifier config loading
- [ ] Test global config inheritance
- [ ] Test environment variable overrides

### Phase 4: CLI Tests (Priority: HIGH)
**File**: `mlserver/cli.py` (the Typer-based CLI - the only CLI)
- [ ] Test CLI commands
- [ ] Test config file auto-detection
- [ ] Test multi-classifier selection
- [ ] Test build/push/clean commands

### Phase 5: Multi-Classifier Tests (Priority: HIGH)
**File**: `mlserver/multi_classifier.py`
- [ ] Test loading multi-classifier configs
- [ ] Test classifier extraction
- [ ] Test default classifier selection
- [ ] Test config merging

### Phase 6: Integration Tests (Priority: MEDIUM)
**Files**: `tests/integration/`
- [x] Fix MockPredictor references
- [x] Update endpoint URLs in all tests
- [ ] Add tests for /info and /status endpoints
- [ ] Test complete multi-classifier flow
- [ ] Test concurrent predictions with limiter
- [ ] Test metrics collection accuracy

### Phase 7: Container Tests ✅ COMPLETED (2026-07-03, W0.4)
**File**: `mlserver/container.py` (unit tests: `tests/unit/test_container.py`)
- [x] Fix Docker detection tests
- [x] Test wheel building (`_build_mlserver_wheel`, wheel cleanup vs user-wheel preservation)
- [x] Test Dockerfile generation (COPY generation, config port, classifier_name fallback, wheel install)
- [x] Test image building/pushing (build_container, push_container tag construction incl. partial failure, list/remove images)
- All 16 stale skipped tests rewritten or deleted; all docker interactions mocked (no daemon needed)

## Test Commands

### Run All Tests
```bash
pytest tests/
```

### Run with Coverage
```bash
pytest tests/ --cov=mlserver --cov-report=term-missing
```

### Run Specific Test File
```bash
pytest tests/unit/test_server_unit.py -v
```

### Run Integration Tests Only
```bash
pytest tests/integration/ -v
```

### Run Load Tests
```bash
cd tests/load && locust -f locustfile.py --host http://localhost:8000
```

## Testing Guidelines

### Skip Policy (RFC 0001 / D21)

- Every `@pytest.mark.skip` / `skipif` must carry a **reason and a date** (e.g. `reason="... (2026-07-03)"`).
- Any skip older than **one sprint** is rewritten against the current API or deleted - permanently-skipped tests are not allowed to accumulate.
- The only acceptable long-lived skips are environment-gated ones (e.g. tests requiring a live docker daemon), which must state the gating condition in the reason.
- Applied 2026-07-03: the 16 stale skips in `tests/unit/test_container.py` were rewritten or deleted (W0.4).

### When Adding New Functionality

1. **Before Implementation**:
   - Check this INDEX.md for existing test patterns
   - Plan test cases for the new feature

2. **During Implementation**:
   - Write tests alongside the code
   - Ensure tests cover both success and error cases

3. **After Implementation**:
   - Update this INDEX.md with new test information
   - Run full test suite to ensure no regressions
   - Update coverage metrics in this document

### Test Naming Conventions

- Unit tests: `test_<component>_<functionality>`
- Integration tests: `test_<workflow>_<scenario>`
- Fixtures: `<type>_<purpose>` (e.g., `mock_predictor`, `sample_config`)

### Common Test Patterns

#### Testing API Endpoints
```python
async def test_predict_endpoint(async_client):
    payload = {"payload": {"records": [{"f1": 1.0}]}}
    response = await async_client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
```

#### Testing with Mock Predictor
```python
@pytest.fixture
def mock_config():
    return AppConfig(
        predictor=PredictorConfig(
            module="tests.fixtures.mock_predictor",
            class_name="MockPredictor"
        ),
        # ... other config
    )
```

#### Testing Configuration
```python
def test_config_validation():
    with pytest.raises(ValidationError):
        AppConfig(server={"port": -1})  # Invalid port
```

## Known Issues and TODOs

### High Priority
- [x] Fix remaining container.py test failures (done 2026-07-03: skip rewrite, W0.4)
- [ ] Add comprehensive CLI tests
- [ ] Test multi-classifier functionality end-to-end

### Medium Priority
- [ ] Improve concurrency control test coverage
- [ ] Add performance regression tests
- [ ] Test model hot-reloading scenarios

### Low Priority
- [ ] Add property-based tests for adapters
- [ ] Create test data generators

## Maintenance Notes

- **Test Data**: Keep test payloads small and focused
- **Mocking**: Use `MockPredictor` for predictable test behavior
- **Async Tests**: Use `pytest-asyncio` fixtures for async endpoints
- **Cleanup**: Ensure tests clean up resources (files, ports, etc.)

---

**Remember**: This document should be the first stop when working with tests. Keep it updated!