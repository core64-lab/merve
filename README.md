# mlserver-fastapi-wrapper

Wrap any Python predictor class into a production-ready FastAPI inference API using a simple YAML configuration file.

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
mlserver serve
```

### 4. Make Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"feature1": 1.0, "feature2": 2.0}]}'

# Batch prediction (same endpoint)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [
    {"feature1": 1.0, "feature2": 2.0},
    {"feature1": 3.0, "feature2": 4.0}
  ]}'
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
  adapter: auto              # auto | records | ndarray
  feature_order: [col1, col2]  # or path to JSON file
  thread_safe_predict: false
  max_concurrent_predictions: 1
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
mlserver serve --classifier sentiment
mlserver build --classifier sentiment
```

## CLI Commands

```
mlserver serve [config.yaml]           Start the server
mlserver build --classifier <name>     Build Docker container
mlserver tag <patch|minor|major> -c <name>   Create version tag
mlserver push --classifier <name>      Push to container registry
mlserver run --classifier <name>       Run container locally
mlserver images                        List built images
mlserver clean --classifier <name>     Remove built images
mlserver list-classifiers              List classifiers in config
mlserver version [--json]              Show version info
mlserver status                        Show system status
mlserver validate                      Validate configuration
mlserver doctor                        Diagnose common issues
mlserver test                          Test against running server
mlserver init                          Initialize new project
mlserver init-github                   Generate GitHub Actions workflow
```

## Docker Containerization

### Build and Run

```bash
# Build container
mlserver build --classifier my-classifier

# Run locally
mlserver run --classifier my-classifier

# Or manually
docker run -p 8000:8000 my-repo/my-classifier:latest
```

### Version Tagging

Create hierarchical tags that track both classifier and mlserver versions:

```bash
# Create patch version bump (1.0.0 -> 1.0.1)
mlserver tag patch --classifier my-classifier

# Push to trigger GitHub Actions
git push --tags
```

Tag format: `<classifier>-v<version>-mlserver-<commit>`

Example: `my-classifier-v1.0.1-mlserver-abc123d`

## GitHub Actions CI/CD

Initialize the workflow:

```bash
mlserver init-github
```

This creates `.github/workflows/ml-classifier-container-build.yml` which:

1. Triggers on hierarchical tags
2. Installs the exact mlserver version from the tag
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

The API auto-detects input format:

**Records (list of dicts):**
```json
{"instances": [{"age": 25, "income": 50000}]}
```

**ndarray (nested lists):**
```json
{"instances": [[25, 50000]]}
```

Force a specific format with `api.adapter: records` or `api.adapter: ndarray`.

## Observability

Built-in observability at no extra configuration:

- **Prometheus metrics** at `/metrics`
- **Structured JSON logging** with correlation IDs
- **Health checks** at `/healthz`
- **Request tracing** via X-Correlation-ID header

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

## Troubleshooting

```bash
# Validate configuration
mlserver validate

# Diagnose environment issues
mlserver doctor

# Check server status
mlserver status
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
