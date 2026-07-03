# Merve

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-897%20passing-brightgreen.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-64%25-yellow.svg)](#testing)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)

> **M**odel s**erve**r - Wrap any Python predictor class into a production-ready FastAPI inference API using a simple YAML configuration file.

## What This Does

You have a Python class with `predict()` and/or `predict_proba()` methods. This tool:

1. Wraps it in a FastAPI server with `/predict` and `/predict_proba` endpoints
2. Adds Prometheus metrics, structured logging, and health checks
3. Handles input validation and format conversion automatically
4. Provides Docker containerization with version tracking
5. Generates GitHub Actions workflows for automated CI/CD builds

## Installation

```bash
pip install git+https://github.com/core64-lab/merve.git
```

For development:
```bash
git clone https://github.com/core64-lab/merve.git
cd merve
pip install -e ".[dev]"
```

## Quick Start

### 1. Create Your Predictor

```python
# mlserver_predictor.py
import joblib

class MyPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, data):
        # data: list of dicts or numpy array
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)
```

### 2. Create mlserver.yaml

```yaml
predictor:
  module: mlserver_predictor
  class_name: MyPredictor
  init_kwargs:
    model_path: ./model.pkl

classifier:
  name: my-classifier
  version: 1.0.0
```

That's the minimal configuration. Server defaults to `0.0.0.0:8000`.

### 3. Start Server

```bash
merve serve
```

### 4. Make Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"payload": {"records": [{"feature1": 1.0, "feature2": 2.0}]}}'

# Batch prediction (same endpoint)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"payload": {"records": [
    {"feature1": 1.0, "feature2": 2.0},
    {"feature1": 3.0, "feature2": 4.0}
  ]}}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Model predictions (single or batch) |
| `/predict_proba` | POST | Probability predictions |
| `/healthz` | GET | Health check |
| `/info` | GET | Server and model metadata |
| `/status` | GET | Detailed status information |
| `/metrics` | GET | Prometheus metrics |

## Configuration

### Minimal Configuration

```yaml
predictor:
  module: mlserver_predictor
  class_name: MyPredictor
  init_kwargs:
    model_path: ./model.pkl

classifier:
  name: my-classifier
  version: 1.0.0
```

### Full Configuration

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 1
  log_level: INFO
  cors:
    allow_origins: []

predictor:
  module: mlserver_predictor
  class_name: MyPredictor
  init_kwargs:
    model_path: ./model.pkl

classifier:
  name: my-classifier
  version: 1.0.0
  description: My ML classifier

api:
  adapter: records           # records (default) | ndarray | auto
  feature_order: [col1, col2]  # or path to JSON file
  thread_safe_predict: false
  max_concurrent_predictions: 1  # 0 disables the limiter
  warmup_on_start: true
  endpoints:
    predict: true
    predict_proba: true

observability:
  metrics: true
  structured_logging: true
  correlation_ids: true
  log_payloads: false
```

### Multi-Classifier Configuration

Serve multiple models from one repository:

```yaml
server:
  host: 0.0.0.0
  port: 8000

classifiers:
  sentiment:
    predictor:
      module: sentiment_predictor
      class_name: SentimentPredictor
      init_kwargs:
        model_path: ./models/sentiment.pkl
    classifier:
      name: sentiment
      version: 1.0.0

  fraud:
    predictor:
      module: fraud_predictor
      class_name: FraudPredictor
      init_kwargs:
        model_path: ./models/fraud.pkl
    classifier:
      name: fraud
      version: 2.0.0
```

Run a specific classifier:
```bash
merve serve --classifier sentiment
merve build --classifier sentiment
```

## CLI Commands

```
merve serve [config.yaml]           Start the server
merve build --classifier <name>     Build Docker container
merve tag <patch|minor|major> -c <name>   Create version tag
merve push --classifier <name>      Push to container registry
merve run --classifier <name>       Run container locally
merve images                        List built images
merve clean --classifier <name>     Remove built images
merve list-classifiers              List classifiers in config
merve version [--json]              Show version info
merve status                        Show system status
merve validate                      Validate configuration
merve doctor                        Diagnose common issues
merve test                          Test against running server
merve init                          Initialize new project
merve init-github                   Generate GitHub Actions workflow
merve schema                        Generate JSON schema for mlserver.yaml
```

## Docker Containerization

### Build and Run

```bash
# Build container
merve build --classifier my-classifier

# Run locally
merve run --classifier my-classifier

# Or manually
docker run -p 8000:8000 my-repo/my-classifier:latest
```

### Version Tagging

Create hierarchical tags that track both classifier and mlserver versions:

```bash
# Create patch version bump (1.0.0 -> 1.0.1)
merve tag patch --classifier my-classifier

# Push to trigger GitHub Actions
git push --tags
```

Tag format: `<classifier>-v<version>-mlserver-<commit>`

Example: `my-classifier-v1.0.1-mlserver-abc123d`

## GitHub Actions CI/CD

Initialize the workflow:

```bash
merve init-github
```

This creates `.github/workflows/ml-classifier-container-build.yml` which:

1. Triggers on hierarchical tags
2. Installs the exact merve version from the tag
3. Builds and tests the container
4. Pushes to GHCR or ECR

Configure registry in `mlserver.yaml`:

```yaml
deployment:
  registry:
    type: ghcr    # or ecr
    namespace: your-org
```

## Input Formats

All requests wrap the input in a `payload` object. The default adapter is `records`; set `api.adapter: auto` to opt in to auto-detection of the input format.

**Records (list of dicts, default):**
```json
{"payload": {"records": [{"age": 25, "income": 50000}]}}
```

**ndarray (nested lists, requires `api.adapter: ndarray` or `auto`):**
```json
{"payload": {"ndarray": [[25, 50000]]}}
```

The ndarray adapter also accepts an `"inputs"` key in place of `"ndarray"`. Force a specific format with `api.adapter: records` or `api.adapter: ndarray`.

## Observability

Built-in observability at no extra configuration:

- **Prometheus metrics** at `/metrics`
- **Structured JSON logging** with correlation IDs
- **Health checks** at `/healthz`
- **Correlation IDs** attached to structured logs

Example Prometheus + Grafana setup in `monitoring/` directory.

## Requirements

- Python 3.9+
- Docker (for containerization)
- Git (for version tagging)

## Architecture

```
Request -> FastAPI -> InputAdapter -> Predictor.predict() -> Response
                           |
                    Metrics + Logging
```

## Examples

See `examples/` directory for complete working examples:

- Single classifier setup
- Multi-classifier repository
- Custom preprocessing
- Model ensembles

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=mlserver --cov-report=term-missing

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
```

Current status: **897 tests passing, 0 failing**, **64% coverage**

## Troubleshooting

```bash
# Validate configuration
merve validate

# Diagnose environment issues
merve doctor

# Check server status
merve status
```

Common issues:

- **Import errors**: Ensure predictor module is in Python path
- **Memory issues**: Reduce `server.workers` (each loads full model)
- **Slow first request**: Enable `api.warmup_on_start: true`

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.

## Alternatives

- MLflow Models - Full ML lifecycle platform
- BentoML - Feature-rich model serving
- TorchServe / TensorFlow Serving - Framework-specific

This tool focuses on simplicity: wrap any Python predictor with minimal configuration, no framework lock-in.
