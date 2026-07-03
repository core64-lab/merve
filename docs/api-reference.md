# API Reference

## Overview

MLServer provides a unified REST API for ML model inference without version or classifier names in URLs. All endpoints return JSON responses and support CORS for cross-origin requests (when configured).

## Base URL Structure

```
http://<host>:<port>/
```

No versioning or classifier names in paths - clean, unified endpoints. Each deployment serves exactly one classifier at the root path.

## Core Endpoints

### Prediction Endpoints

#### `POST /predict`
Prediction endpoint for single and batch inference requests. Send one record for a single prediction or many records for a batch — there is no separate batch endpoint (the former batch endpoint was folded into `/predict`).

**Request Body** — all requests use the `payload` wrapper:
```json
{
  "payload": {
    "records": [...]
  }
}
```

The key inside `payload` depends on the input format: `records` (or `instances`) for lists of feature objects, `features` for a single feature object, `ndarray` (or `inputs`) for nested arrays.

**Input Formats**:
1. **Records Format** (JSON objects, default adapter):
```json
{
  "payload": {
    "records": [
      {"age": 25, "sex": "male", "fare": 72.5},
      {"age": 31, "sex": "female", "fare": 15.0}
    ]
  }
}
```

2. **Ndarray Format** (nested arrays, requires `api.adapter: ndarray` or `auto`):
```json
{
  "payload": {
    "ndarray": [[25, 1, 72.5]]
  }
}
```

3. **Auto-Detection** (only when `api.adapter: auto` is configured):
- Detects records vs. ndarray based on the payload structure
- The default adapter is `records`; auto-detection is opt-in

**Response**:
```json
{
  "predictions": [0, 1],
  "time_ms": 12.5,
  "predictor_class": "CatBoostPredictor",
  "metadata": {
    "project": "mlserver-repo",
    "classifier": "catboost-survival",
    "predictor_class": "CatBoostPredictor",
    "predictor_module": "predictor_catboost",
    "config_file": "mlserver.yaml",
    "git_commit": "abc123",
    "git_tag": "catboost-survival-v1.0.0-mlserver-b5dff2a",
    "deployed_at": "2025-01-18T10:00:00Z",
    "mlserver_version": "0.3.2",
    "mlserver_api_commit": "b5dff2a",
    "mlserver_api_tag": null
  }
}
```

**Status Codes**:
- `200`: Success
- `400`: Input parsing failed
- `422`: Request validation error (malformed body)
- `500`: Prediction error
- `503`: Concurrency limit reached (see [Concurrency Control](#concurrency-control))

---

#### `POST /predict_proba`
Probability predictions for classification models.

**Request Body**: Same as `/predict`

**Response**:
```json
{
  "probabilities": [[0.8, 0.2]],
  "time_ms": 14.8,
  "classes": null,
  "metadata": {
    "project": "mlserver-repo",
    "classifier": "catboost-survival",
    "predictor_class": "CatBoostPredictor",
    "predictor_module": "predictor_catboost",
    "config_file": "mlserver.yaml",
    "git_commit": "abc123",
    "git_tag": "catboost-survival-v1.0.0-mlserver-b5dff2a",
    "deployed_at": "2025-01-18T10:00:00Z",
    "mlserver_version": "0.3.2",
    "mlserver_api_commit": "b5dff2a",
    "mlserver_api_tag": null
  }
}
```

**Status Codes**: Same as `/predict`, plus:
- `501`: The loaded predictor does not implement `predict_proba`

---

### Metadata Endpoints

#### `GET /info`
Complete model and API metadata.

**Response**:
```json
{
  "project": "mlserver-models",
  "classifier": "catboost-survival",
  "description": "Titanic survival prediction model",
  "predictor_class": "CatBoostPredictor",
  "deployed_at": "2025-01-18T10:30:00Z",
  "classifier_repository": {
    "repository": "mlserver-models",
    "commit": "abc123def",
    "tag": "catboost-survival-v1.0.0-mlserver-b5dff2a",
    "branch": "main",
    "dirty": false
  },
  "api_service": {
    "api_commit": "def456",
    "api_tag": null,
    "api_branch": "main",
    "api_dirty": false
  },
  "endpoints": {
    "predict": "/predict",
    "predict_proba": "/predict_proba",
    "info": "/info",
    "health": "/healthz",
    "metrics": "/metrics"
  }
}
```

---

### Health & Status Endpoints

#### `GET /healthz`
Basic health check endpoint for liveness (and readiness) probes.

**Response**:
```json
{
  "status": "ok",
  "model": "CatBoostPredictor"
}
```

`model` is the predictor class name, or `null` before the predictor is loaded.

---

#### `GET /status`
Prediction availability status, reflecting the concurrency limiter.

**Response**:
```json
{
  "prediction_slots_available": true,
  "active_predictions": 0,
  "max_concurrent_predictions": 1,
  "concurrency_control_enabled": true
}
```

When concurrency control is disabled (`api.max_concurrent_predictions: 0`), `max_concurrent_predictions` is `null` and `concurrency_control_enabled` is `false`.

---

### Observability Endpoints

#### `GET /metrics`
Prometheus-formatted metrics endpoint. The path is configurable via `observability.metrics_endpoint` (default `/metrics`).

**Response** (text/plain):
```
# HELP mlserver_requests_total Total number of requests
# TYPE mlserver_requests_total counter
mlserver_requests_total{endpoint="/predict",method="POST",model="CatBoostPredictor",status_code="200"} 195
mlserver_requests_total{endpoint="/predict",method="POST",model="CatBoostPredictor",status_code="400"} 5

# HELP mlserver_request_duration_seconds Request duration in seconds
# TYPE mlserver_request_duration_seconds histogram
mlserver_request_duration_seconds_bucket{endpoint="/predict",method="POST",model="CatBoostPredictor",le="0.1"} 100
mlserver_request_duration_seconds_sum{endpoint="/predict",method="POST",model="CatBoostPredictor"} 45.2
mlserver_request_duration_seconds_count{endpoint="/predict",method="POST",model="CatBoostPredictor"} 200

# HELP mlserver_predictions_total Total number of predictions made (output samples)
# TYPE mlserver_predictions_total counter
mlserver_predictions_total{endpoint="/predict",model="CatBoostPredictor"} 950
```

The full metric set (names and labels) is documented in [Observability Features](./observability.md#available-metrics).

---

### Documentation Endpoints

#### `GET /docs`
Interactive OpenAPI documentation (Swagger UI).

#### `GET /redoc`
Alternative API documentation (ReDoc format).

#### `GET /openapi.json`
OpenAPI schema in JSON format.

---

## Request Headers

### Standard Headers

| Header | Description | Example |
|--------|-------------|---------|
| `Content-Type` | Request body format | `application/json` |
| `Accept` | Response format | `application/json` |

Correlation IDs are generated server-side per request (when `observability.correlation_ids: true`) and appear in the structured logs; there is no request/response correlation header.

### CORS Headers

Handled for cross-origin requests when `server.cors.allow_origins` is non-empty:
- `Access-Control-Allow-Origin`
- `Access-Control-Allow-Methods`
- `Access-Control-Allow-Headers`

---

## Error Responses

### Standard Error Format

Errors use the FastAPI error model — a single `detail` field:

```json
{
  "detail": "Input parsing failed. Please check your input format."
}
```

Request-body validation errors (HTTP 422) use FastAPI's standard structure, where `detail` is a list of validation issues.

### Error Status Codes

| HTTP Status | Meaning | Example `detail` |
|-------------|---------|------------------|
| 400 | Input parsing failed (bad payload contents) | `"Input parsing failed. Please check your input format."` |
| 422 | Request validation error (malformed body) | List of Pydantic validation errors |
| 500 | Prediction error | `"Prediction failed. Please contact support if the issue persists."` |
| 501 | Predictor has no `predict_proba` | `"Probability prediction not available for this model."` |
| 503 | Concurrency limit reached | `"Server is currently processing another prediction. Please retry later."` |

With `--log-level DEBUG`, 400 responses include additional parsing detail.

---

## Concurrency Control

Configured via `api.max_concurrent_predictions`:
```yaml
api:
  max_concurrent_predictions: 1  # Default: one prediction at a time
                                 # 0 disables concurrency limiting entirely
```

When the limit is reached, additional prediction requests are rejected immediately with **HTTP 503** and a `Retry-After: 5` header:

```json
{
  "detail": "Server is currently processing another prediction. Please retry later."
}
```

This is designed for Kubernetes pod scaling: overloaded pods reject fast so the load balancer retries against another replica. Use `GET /status` to observe slot availability.

---

## Input Adapters

All request bodies use the `payload` wrapper regardless of adapter.

### Records Adapter (Default)
Expects JSON objects with named features:
```yaml
# Config
api:
  adapter: "records"
  feature_order: ["age", "sex", "fare"]  # Optional ordering
```

```json
{"payload": {"records": [{"age": 25, "sex": "male", "fare": 72.5}]}}
```

### Ndarray Adapter
Expects nested arrays:
```yaml
# Config
api:
  adapter: "ndarray"
```

```json
{"payload": {"ndarray": [[25, 1, 72.5]]}}
```

### Auto Adapter
Automatically detects the format (opt-in — the default adapter is `records`):
```yaml
# Config
api:
  adapter: "auto"
```

```json
{"payload": {"records": [{"age": 25}]}}
{"payload": {"ndarray": [[25, 1]]}}
```

---

## Response Formats

### Standard Prediction Response (`PredictResponse`)
```json
{
  "predictions": [0, 1, 0],
  "time_ms": 12.5,
  "predictor_class": "CatBoostPredictor",
  "metadata": { "...": "see metadata fields below" }
}
```

### Probability Response (`ProbaResponse`)
```json
{
  "probabilities": [
    [0.8, 0.2],
    [0.3, 0.7]
  ],
  "time_ms": 14.8,
  "classes": null,
  "metadata": { "...": "see metadata fields below" }
}
```

### Metadata Fields

Every prediction response can include a `metadata` object with these fields:

| Field | Description |
|-------|-------------|
| `project` | Auto-detected project/repository name |
| `classifier` | Classifier name |
| `predictor_class` | Predictor class name |
| `predictor_module` | Module file containing the predictor |
| `config_file` | Configuration file used |
| `git_commit` | Classifier git commit hash |
| `git_tag` | Git tag if on a tagged commit |
| `deployed_at` | Deployment timestamp |
| `mlserver_version` | MLServer package version |
| `mlserver_api_commit` | MLServer tool git commit |
| `mlserver_api_tag` | MLServer tool git tag |

### Custom / Passthrough Formats

With `api.response_format: custom`, the prediction output is returned in a flexible `result` field; with `passthrough`, the predictor output is returned unmodified (no wrapper, no metadata). See the [Configuration Guide](./configuration.md#response-formats).

---

## Security

### CORS Configuration

```yaml
server:
  cors:
    allow_origins: ["https://app.example.com"]  # Empty list = CORS disabled
    allow_methods: ["GET", "POST"]
    allow_headers: ["Content-Type"]
    allow_credentials: false
```

### Request Validation

- Type validation via Pydantic (`payload` must be a JSON object)
- Optional feature validation against `api.feature_order` (missing/extra features produce clear 400 errors)

---

## Performance Optimization

### Feature Ordering

Provide `api.feature_order` to ensure deterministic column ordering for the records adapter (it can also be a path to a JSON file with the feature list). The resolved ordering is cached.

### Warmup

With `api.warmup_on_start: true` (default), the server runs a dummy prediction at startup to initialize model internals and reduce first-request latency (requires `feature_order` to synthesize input).

### Connection Pooling

Keep-alive connections supported:
```bash
curl -H "Connection: keep-alive" http://localhost:8000/predict
```

---

## Examples

### Python Client

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "payload": {
            "records": [{"age": 25, "sex": "male", "fare": 72.5}]
        }
    }
)
print(response.json()["predictions"])

# Batch prediction (multiple records in one /predict call)
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "payload": {
            "records": [
                {"age": 25, "sex": "male", "fare": 72.5},
                {"age": 30, "sex": "female", "fare": 15.0}
            ]
        }
    }
)
print(response.json()["predictions"])

# Get model info
response = requests.get("http://localhost:8000/info")
print(response.json()["classifier"])
```

### cURL Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"payload": {"records": [{"age": 25, "sex": "male", "fare": 72.5}]}}'

# Probability prediction
curl -X POST http://localhost:8000/predict_proba \
  -H "Content-Type: application/json" \
  -d '{"payload": {"records": [{"age": 25, "sex": "male", "fare": 72.5}]}}'

# Model info
curl http://localhost:8000/info

# Health check
curl http://localhost:8000/healthz

# Concurrency status
curl http://localhost:8000/status

# Metrics
curl http://localhost:8000/metrics
```

### JavaScript/TypeScript

```typescript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    payload: {
      records: [{ age: 25, sex: 'male', fare: 72.5 }]
    }
  })
});
const result = await response.json();
console.log(result.predictions);
```

---

## Troubleshooting

### Common Issues

**404 Not Found**
- Ensure no `/v1/` or classifier-name prefix in the URL — endpoints are mounted at the root
- Check endpoint spelling
- Verify server is running

**400 Bad Request**
- Ensure the body uses the `payload` wrapper: `{"payload": {"records": [...]}}`
- Check feature names match model expectations
- Ensure data format matches adapter configuration (`records` vs `ndarray`)

**422 Unprocessable Entity**
- The request body is not valid JSON or `payload` is missing/not an object

**501 Not Implemented**
- `/predict_proba` was called but the predictor does not implement `predict_proba`

**503 Service Unavailable**
- Another prediction is in flight and `api.max_concurrent_predictions` is reached
- Honor the `Retry-After: 5` header and retry, check `/status`, add replicas, or set `max_concurrent_predictions: 0` to disable limiting
