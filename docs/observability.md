# Observability Features

## Overview

MLServer provides comprehensive observability features including Prometheus metrics, structured logging, correlation IDs, and health monitoring.

## Metrics

### Prometheus Integration

MLServer exposes metrics at the `/metrics` endpoint (path configurable via `observability.metrics_endpoint`) in Prometheus format.

> **⚠️ Multi-worker caveat**: with `server.workers > 1`, each worker process keeps its own in-process Prometheus registry — `prometheus_client` multiprocess mode is **not** configured. A scrape of `/metrics` samples whichever worker answers the request, so counters will appear to jump around. For accurate metrics, run `workers: 1` per container and scale horizontally with pod replicas.

#### Available Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `mlserver_requests_total` | Counter | Total number of requests | `method`, `endpoint`, `status_code`, `model` |
| `mlserver_request_duration_seconds` | Histogram | Request duration in seconds | `method`, `endpoint`, `model` |
| `mlserver_prediction_duration_seconds` | Histogram | Model prediction duration in seconds | `model`, `endpoint` |
| `mlserver_predictions_total` | Counter | Total number of predictions made (output samples) | `model`, `endpoint` |
| `mlserver_input_samples_total` | Counter | Total number of input samples received | `model`, `endpoint` |
| `mlserver_active_requests` | Gauge | Number of active requests | `model` |
| `mlserver_batch_size` | Histogram | Distribution of prediction batch sizes | `model`, `endpoint` |
| `mlserver_model_info` | Gauge | Model information (set to 1 when the model is loaded) | `model`, `version` |

Notes:
- The `model` label carries the predictor class name (e.g. `CatBoostPredictor`).
- Requests to `/healthz` and the metrics endpoint itself are excluded from request metrics to reduce noise.
- There is no dedicated error counter — derive error rates from `mlserver_requests_total` filtered by `status_code`.

#### Configuration

```yaml
# mlserver.yaml
observability:
  metrics: true
  metrics_endpoint: "/metrics"
```

#### Custom Metrics

Add custom metrics in your predictor:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define custom metrics
prediction_cache_hits = Counter(
    'prediction_cache_hits_total',
    'Number of cache hits'
)

model_confidence = Histogram(
    'model_confidence_score',
    'Model confidence distribution',
    buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

class CachedPredictor:
    def predict(self, X):
        # Check cache
        if self.is_cached(X):
            prediction_cache_hits.inc()
            return self.get_from_cache(X)

        # Make prediction
        predictions = self.model.predict(X)
        confidence = self.model.predict_proba(X).max(axis=1)

        # Record confidence
        for conf in confidence:
            model_confidence.observe(conf)

        return predictions
```

### Grafana Dashboards

Pre-built dashboards for monitoring MLServer.

#### Request Metrics Dashboard

```json
{
  "dashboard": {
    "title": "MLServer Request Metrics",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(mlserver_requests_total[5m])"
        }]
      },
      {
        "id": 2,
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(mlserver_requests_total{status_code=~'4..|5..'}[5m])"
        }]
      },
      {
        "id": 3,
        "title": "Latency Percentiles",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(mlserver_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(mlserver_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(mlserver_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      }
    ]
  }
}
```

#### Model Performance Dashboard

```json
{
  "dashboard": {
    "title": "Model Performance",
    "panels": [
      {
        "title": "Prediction Throughput (requests)",
        "targets": [{
          "expr": "rate(mlserver_requests_total{endpoint='/predict'}[1m])"
        }]
      },
      {
        "title": "Prediction Throughput (samples)",
        "targets": [{
          "expr": "rate(mlserver_predictions_total[1m])"
        }]
      },
      {
        "title": "Prediction Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(mlserver_prediction_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Prediction Errors",
        "targets": [{
          "expr": "rate(mlserver_requests_total{endpoint='/predict', status_code=~'5..'}[5m])"
        }]
      },
      {
        "title": "Batch Size P95",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(mlserver_batch_size_bucket[5m]))"
        }]
      },
      {
        "title": "Active Requests",
        "targets": [{
          "expr": "mlserver_active_requests"
        }]
      }
    ]
  }
}
```

## Logging

### Structured Logging

MLServer uses structured JSON logging for better log aggregation and analysis.

#### Configuration

```yaml
server:
  log_level: "INFO"        # DEBUG, INFO, WARNING, ERROR, CRITICAL
  logger:
    timestamp: false       # Include timestamps (default: false — container runtimes add their own)
    structured: true       # JSON output format
    show_tasks: false      # Include asyncio task names
    format: null           # Custom log format string (overrides other settings)

observability:
  structured_logging: true   # Emit request/response log events
  correlation_ids: true      # Attach a correlation_id to all logs of a request
  log_payloads: false        # Log request payload + response at INFO (privacy!)
```

`observability.log_payloads: true` logs the incoming request payload and the outgoing response at INFO level — useful for debugging, but be mindful of sensitive data in production.

#### Log Format

Each log line is a JSON object. Base fields are `level`, `logger`, and `message`; `correlation_id` is added when `correlation_ids` is enabled, and `timestamp` only when `server.logger.timestamp: true`. Request lifecycle events add structured fields:

```json
{"level": "INFO", "logger": "mlserver.request", "message": "Request started", "correlation_id": "0b0e8a7e-4f6e-4a9e-9a1e-1f2d3c4b5a69", "event": "request_start", "method": "POST", "path": "/predict"}
{"level": "INFO", "logger": "mlserver.response", "message": "Request completed", "correlation_id": "0b0e8a7e-4f6e-4a9e-9a1e-1f2d3c4b5a69", "event": "request_complete", "status_code": 200, "duration_ms": 45.2}
```

On errors, the log entry additionally carries the exception details:

```json
{"level": "ERROR", "logger": "mlserver.server", "message": "Prediction error: could not convert string to float", "correlation_id": "0b0e8a7e-4f6e-4a9e-9a1e-1f2d3c4b5a69", "exception": "Traceback (most recent call last): ..."}
```

#### Custom Logging

```python
import logging

logger = logging.getLogger("my_predictor")

class LoggingPredictor:
    def predict(self, X):
        logger.info(
            "prediction_started",
            extra={"model": self.__class__.__name__, "input_shape": X.shape}
        )

        start_time = time.time()
        try:
            predictions = self.model.predict(X)

            logger.info(
                "prediction_completed",
                extra={
                    "duration_ms": (time.time() - start_time) * 1000,
                    "output_shape": predictions.shape,
                }
            )

            return predictions

        except Exception as e:
            logger.error(
                "prediction_failed",
                exc_info=True,
                extra={"error_type": type(e).__name__}
            )
            raise
```

Extra fields passed via `extra=` are merged into the JSON log entry, and the correlation ID of the current request is attached automatically.

### Log Aggregation

#### ELK Stack Integration

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/mlserver/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "mlserver-%{+yyyy.MM.dd}"
```

#### Fluentd Configuration

```ruby
# fluent.conf
<source>
  @type tail
  path /var/log/mlserver/*.log
  pos_file /var/log/fluentd/mlserver.pos
  tag mlserver.*
  format json
</source>

<match mlserver.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix mlserver
</match>
```

## Request Tracking

### Correlation IDs

Every request gets a unique correlation ID for tracking across log lines.

```yaml
observability:
  correlation_ids: true
```

The correlation ID is generated server-side per request and attached to every log entry emitted while handling that request (via a context variable). It is not propagated via HTTP headers; to correlate with downstream services, log your own identifiers from within the predictor.

### OpenTelemetry Integration

MLServer has no built-in tracing, but you can instrument your predictor with OpenTelemetry:

```python
# your_project/tracing.py — user-level integration example
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument predictions
class TracedPredictor:
    def predict(self, X):
        with tracer.start_as_current_span("predict") as span:
            span.set_attribute("input.shape", str(X.shape))

            # Preprocessing span
            with tracer.start_as_current_span("preprocess"):
                X_processed = self.preprocess(X)

            # Model inference span
            with tracer.start_as_current_span("model_inference"):
                predictions = self.model.predict(X_processed)

            span.set_attribute("output.shape", str(predictions.shape))
            return predictions
```

## Health Monitoring

### Health Endpoints

MLServer exposes two built-in endpoints for health monitoring:

- **`GET /healthz`** — returns `200 {"status": "ok", "model": "<PredictorClass>"}` once the predictor is loaded. **Until then it returns `503 {"status": "loading", "model": null}`** (not-ready semantics), so readiness probes never read "ok" for a model that cannot serve. Use it for liveness **and** readiness probes.
- **`GET /status`** — returns concurrency-limiter state: `{"prediction_slots_available": ..., "active_predictions": ..., "max_concurrent_predictions": ..., "concurrency_control_enabled": ...}`. Useful for observing prediction slot saturation (e.g. before routing traffic).

These are the only health endpoints — there are no separate readiness or startup probe paths, so point all Kubernetes probes at `/healthz`; the 503-while-loading answer makes that safe for readiness and startup probes.

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: mlserver
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

        # For slow-loading models, point the startup probe at /healthz too
        startupProbe:
          httpGet:
            path: /healthz
            port: 8000
          failureThreshold: 30
          periodSeconds: 10
```

## Performance Monitoring

### Request Profiling

```python
import cProfile
import pstats
import io

class ProfilingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["path"] == "/profile":
            profiler = cProfile.Profile()
            profiler.enable()

            await self.app(scope, receive, send)

            profiler.disable()
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)

            # Log or return profile data
            logger.info("profile_data", extra={"profile": stream.getvalue()})
        else:
            await self.app(scope, receive, send)
```

### Memory Monitoring

```python
import psutil
import gc

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()

    def get_memory_stats(self):
        """Get current memory statistics."""
        return {
            "rss_mb": self.process.memory_info().rss / 1024 / 1024,
            "vms_mb": self.process.memory_info().vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }

    def check_memory_pressure(self, threshold_percent=80):
        """Check if memory usage is high."""
        if self.process.memory_percent() > threshold_percent:
            logger.warning(
                "high_memory_usage",
                extra=self.get_memory_stats()
            )
            # Trigger garbage collection
            gc.collect()
            return True
        return False
```

## Alerting

### Prometheus Alert Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: mlserver
    rules:
      - alert: HighErrorRate
        expr: rate(mlserver_requests_total{status_code=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on {{ $labels.instance }}"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(mlserver_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency on {{ $labels.instance }}"
          description: "P95 latency is {{ $value }} seconds"

      - alert: ModelInfoMissing
        # mlserver_model_info is set to 1 at startup once the model is loaded;
        # absence of the series means the server is down or the model never loaded.
        expr: absent(mlserver_model_info) == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "No mlserver_model_info series — model not loaded or server down"

      - alert: ConcurrencyRejections
        expr: rate(mlserver_requests_total{status_code="503"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Prediction requests being rejected (503) on {{ $labels.instance }}"
          description: "Concurrency limit reached — consider adding replicas"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 / 1024 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Using {{ $value }} GB of memory"
```

### AlertManager Configuration

```yaml
# alertmanager.yml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'slack-notifications'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#ml-alerts'
        title: 'MLServer Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Monitoring Best Practices

### 1. Set Up Baseline Metrics

```python
# Collect baseline metrics during normal operation
baseline = {
    "p50_latency_ms": 25,
    "p95_latency_ms": 100,
    "error_rate": 0.001,
    "throughput_rps": 100
}

# Alert on deviations
if current_p95 > baseline["p95_latency_ms"] * 2:
    alert("Latency doubled from baseline")
```

### 2. Monitor Model Drift

```python
class DriftMonitor:
    def __init__(self, reference_distribution):
        self.reference = reference_distribution

    def check_drift(self, current_data):
        """Check for distribution drift using KS test."""
        from scipy import stats

        drift_scores = {}
        for feature in self.reference.keys():
            statistic, p_value = stats.ks_2samp(
                self.reference[feature],
                current_data[feature]
            )
            drift_scores[feature] = {
                "statistic": statistic,
                "p_value": p_value,
                "drifted": p_value < 0.05
            }

        # Log drift metrics
        for feature, scores in drift_scores.items():
            drift_detected.labels(feature=feature).set(
                1 if scores["drifted"] else 0
            )

        return drift_scores
```

### 3. Business Metrics

```python
# Track business-relevant metrics
business_value = Gauge(
    'model_business_value_dollars',
    'Estimated business value from predictions'
)

predictions_by_class = Counter(
    'predictions_by_class_total',
    'Predictions grouped by class',
    ['predicted_class']
)

class BusinessMetricsPredictor:
    def predict(self, X):
        predictions = self.model.predict(X)

        # Track predictions by class
        for pred in predictions:
            predictions_by_class.labels(predicted_class=str(pred)).inc()

        # Estimate business value
        value = self.calculate_business_value(predictions)
        business_value.set(value)

        return predictions
```

## Troubleshooting

### High Latency Issues

1. **Check metrics endpoint**:
```bash
curl http://localhost:8000/metrics | grep duration
```

2. **Analyze slow requests**:
```python
# Add timing to predictor
import time

class TimedPredictor:
    def predict(self, X):
        timings = {}

        start = time.time()
        X_processed = self.preprocess(X)
        timings['preprocess_ms'] = (time.time() - start) * 1000

        start = time.time()
        predictions = self.model.predict(X_processed)
        timings['inference_ms'] = (time.time() - start) * 1000

        logger.info("prediction_timings", extra=timings)
        return predictions
```

### Memory Leaks

1. **Monitor memory growth**:
```bash
# Watch memory usage
watch -n 1 'ps aux | grep mlserver'
```

2. **Profile memory**:
```python
from memory_profiler import profile

@profile
def predict(self, X):
    return self.model.predict(X)
```

### Metric Collection Issues

1. **Verify Prometheus scraping**:
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

2. **Test metrics endpoint**:
```bash
# Should return Prometheus format
curl -v http://localhost:8000/metrics
```

3. **Counters look inconsistent between scrapes**: check `server.workers` — with more than one worker each scrape samples a single process (see the multi-worker caveat above).
