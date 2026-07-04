# Merve

[![CI](https://github.com/core64-lab/merve/actions/workflows/ci.yml/badge.svg)](https://github.com/core64-lab/merve/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Coverage floor](https://img.shields.io/badge/coverage-%E2%89%A575%25%20enforced%20in%20CI-brightgreen.svg)](.github/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **M**odel s**erve**r — wrap any Python predictor class into a production-ready
> FastAPI inference API with one YAML file. No base classes, no model registry,
> no platform. Your code stays yours.

```python
# my_predictor.py — this is ALL merve asks of your code
class MyPredictor:
    def predict(self, X): ...
    def predict_proba(self, X): ...   # optional
```

```yaml
# mlserver.yaml
predictor:
  module: my_predictor
  class_name: MyPredictor

classifier:
  name: my-classifier
```

```bash
merve serve      # → POST /predict on :8000, with metrics, logs, and health checks
```

## Why merve instead of …?

Most serving stacks make you adopt *their* world: a base class, a packaging
format, a registry, an operator, or a whole platform. Merve inverts that — it
adapts to a plain Python class and stays out of the way.

|  | **merve** | MLflow Models | BentoML | Seldon MLServer | KServe / Seldon Core | FastAPI by hand |
|---|---|---|---|---|---|---|
| Predictor contract | any class with `predict()` | pyfunc flavor / logged model | `bentoml.Service` + runners | subclass their runtime | wrap in an InferenceService CRD | you write everything |
| Config | one `mlserver.yaml` | MLmodel + env files | `bentofile.yaml` + build step | model-settings.json | Kubernetes YAML + operator | n/a |
| Packaging | plain Docker image, auto-generated two-stage Dockerfile | `mlflow models build-docker` | bento archive → image | their image | container + CRDs | yours |
| Multi-model monorepo with per-model release tags | **built in** (`<classifier>/vX.Y.Z` git tags, build-once/deploy-many) | registry versions, separate from your git history | one bento per service | one model-settings each | one CRD per model | yours to invent |
| Infra required | any place a container runs | any | any (more for Yatai) | any | **Kubernetes + operator** | any |
| Metrics, structured logs, correlation IDs, health/readiness | **on by default** | partial | partial | partial | via platform | yours to build |
| Concurrency/backpressure semantics | explicit: bounded predictions, `503` + `Retry-After` | opaque | configurable runners | configurable | platform-level | yours |
| Lock-in if you leave | delete one YAML | re-export models | rewrite services | rewrite runtime class | unwind CRDs/operator | none |

**Where merve wins:**

- **Zero-adoption predictor contract.** No inheritance, no decorators, no
  artifact format. If it has `predict()`, it serves. Leaving merve costs you
  one YAML file.
- **A release trail your auditors can read.** Every deployment maps to a
  canonical git tag `<classifier>/vX.Y.Z` in *your* repo — not to an entry in
  a registry database. `merve tag patch --classifier fraud` is the whole
  release ceremony.
- **Build once, deploy many.** One image per git commit bundles every
  classifier in the repo; the model is chosen at deploy time via
  `MLSERVER_CLASSIFIER`. Releases are registry tag aliases on the *same
  digest* — no rebuilds, no image sprawl, byte-identical rollbacks.
- **Boring containers.** Two-stage Dockerfiles (compilers never reach the
  runtime layer), OCI provenance labels, release-pinned framework installs.
  They run on Kubernetes, ECS, Compose, or a lone VM — no operator, no CRDs,
  no control plane.
- **Observability without wiring.** Prometheus `/metrics`, structured JSON
  logs with correlation IDs, health/readiness, model warmup — from the same
  one YAML file.
- **An honest scaling story.** Predictions are bounded
  (`max_concurrent_predictions: 1` by default); overload returns `503` with a
  configurable `Retry-After` so orchestrators can react. Scale with replicas,
  not with hidden thread pools.

**Where merve does not compete (on purpose):** experiment tracking and model
registries (use MLflow — merve happily serves artifacts you tracked there),
inference graphs/canaries/explainers on Kubernetes (Seldon Core, KServe),
serverless GPU autoscaling (cloud platforms), and full lakehouse lifecycles
(Databricks). Merve is the thin, inspectable serving layer — not the platform.

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

### 1. Write (or keep) your predictor

```python
# my_predictor.py
import joblib

class MyPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)
```

Optional hooks merve will use if present: `load()` (called once at startup;
a failure aborts boot so a broken model never reports ready) and `close()`
(called at shutdown).

### 2. Describe it

```yaml
# mlserver.yaml
predictor:
  module: my_predictor
  class_name: MyPredictor
  init_kwargs:
    model_path: ./model.pkl

classifier:
  name: my-classifier
```

The compact spec `predictor: "my_predictor:MyPredictor"` works too.

### 3. Serve and call it

```bash
merve serve
```

```bash
# Single or batch — same endpoint, top-level keys
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [
    {"feature1": 1.0, "feature2": 2.0},
    {"feature1": 3.0, "feature2": 4.0}
  ]}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predictions (single or batch) |
| `/predict_proba` | POST | Class probabilities |
| `/healthz` | GET | `200 ok` once the model is loaded; `503 loading` before that |
| `/info` | GET | Model + deployment metadata (git commit, tag, versions) |
| `/status` | GET | Detailed status |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | OpenAPI UI with request examples |

### Input formats

Send input keys at the **top level** of the request body:

```json
{"records":  [{"age": 25, "income": 50000}]}   // records adapter (default)
{"ndarray":  [[25, 50000]]}                    // ndarray adapter
{"features": {"age": 25, "income": 50000}}     // single record
```

`instances` (records) and `inputs` (ndarray) are accepted aliases. The legacy
`{"payload": {...}}` wrapper still works but is **deprecated** (one warning
per process; removal targeted for 1.0).

## Configuration

Everything beyond the minimal file is optional:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 1                     # scale with replicas, not workers (see below)

predictor:
  module: my_predictor
  class_name: MyPredictor
  init_kwargs:
    model_path: ./model.pkl

classifier:
  name: my-classifier
  description: My ML classifier

api:
  adapter: records               # records (default) | ndarray | auto
  feature_order: [col1, col2]    # or a path to a JSON file
  warmup_on_start: true
  max_concurrent_predictions: 1  # 0 disables the limiter
  retry_after_seconds: 5         # Retry-After header on 503
  thread_safe_predict: false
  endpoints:
    predict: true
    predict_proba: true

observability:
  metrics: true
  structured_logging: true
  correlation_ids: true
  log_payloads: false            # privacy: opt-in
```

Validate any config with `merve validate`; diagnose environments with
`merve doctor`.

### The concurrency stance

Model inference is CPU-bound; queueing requests inside one process only hides
overload. Merve therefore **bounds concurrent predictions** (default: 1) and
answers excess load with `503` + `Retry-After` so load balancers and
autoscalers can do their job. Scale by adding container replicas. If you raise
`server.workers` instead, note that each process keeps its own metrics
registry, so `/metrics` scrapes sample one worker (merve warns about this at
startup).

## Multi-model repositories

Serve many models from one repo, each with its own release trail:

```yaml
# mlserver.yaml
default_classifier: sentiment

classifiers:
  sentiment:
    predictor: {module: sentiment_predictor, class_name: SentimentPredictor}
    classifier: {name: sentiment}
  fraud:
    predictor: {module: fraud_predictor, class_name: FraudPredictor}
    classifier: {name: fraud}
```

```bash
merve serve --classifier fraud      # pick one locally
merve list-classifiers              # see what's defined
```

### Build once, deploy many

```bash
merve build                          # ONE image per git commit: <repo>:<sha>, <repo>:latest
merve tag patch --classifier fraud   # canonical git tag: fraud/v1.0.1
merve push --classifier fraud --registry ghcr.io/acme
#   → tags the SAME image digest as acme/<repo>:fraud-v1.0.1 (no rebuild)

docker run -e MLSERVER_CLASSIFIER=fraud -p 8000:8000 <repo>:latest
#   → the env var selects the model at deploy time; unknown names fail fast
```

The commit image bundles every classifier; a release is a registry tag alias
on that digest. Rollbacks are exact, storage is deduplicated, and the git tag
`fraud/v1.0.1` pins precisely which commit serves in production. Classifiers
with conflicting dependencies can opt out via
`merve build --per-classifier-image --classifier <name>`.

### CI/CD

```bash
merve init-github
```

generates a GitHub Actions workflow that triggers on `<classifier>/vX.Y.Z`
tags, builds the commit image once, smoke-tests the released classifier, and
pushes the release aliases to GHCR or ECR (configure under
`deployment.registry` in `mlserver.yaml`).

## CLI

```
merve serve [config.yaml]              Start the server (-C <dir> for project dir)
merve validate                         Validate configuration
merve doctor                           Diagnose environment issues
merve test --data '{"f1": 1.5}'        Smoke-test a running server
merve build                            Build the commit image (--per-classifier-image to opt out)
merve run --classifier <name>          Run the image locally with the model selected
merve push --classifier <name> -r <registry>   Apply release aliases (no rebuild)
merve tag <patch|minor|major> -c <name>        Create canonical release tag
merve images / merve clean             List / remove built images
merve version [--json]                 Version info
merve status [--json]                  System status
merve list-classifiers [--json]        Classifiers in the config
merve init / merve init-github         Scaffold a project / CI workflow
merve schema                           JSON schema for mlserver.yaml (IDE support)
```

Read commands take `--json` for machine-readable output; exit codes are
stable (`0` ok, `1` failure, `2` usage). The old `mlserver` command still
works as a deprecated alias for one transition release.

## Observability

- **Prometheus** metrics at `/metrics` (request rates, latencies, sample
  counts, batch sizes)
- **Structured JSON logs** with per-request correlation IDs
- **Health/readiness**: `/healthz` returns `503` until the model has loaded
- Example Prometheus + Grafana stack in `monitoring/`

## Requirements

- Python 3.9+ (CI exercises 3.11 and 3.13)
- Docker for containerization, Git for release tagging

## Testing

```bash
pytest tests/                          # full suite
pytest tests/ --cov=mlserver           # with coverage (CI enforces ≥75%)
make typecheck                         # mypy (advisory)
```

See `tests/INDEX.md` for suite organization and `docs/INDEX.md` for the full
documentation set, including the [architecture](docs/architecture.md),
[configuration reference](docs/configuration.md), and the
[0.5 migration guide](docs/migration-0.5.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
