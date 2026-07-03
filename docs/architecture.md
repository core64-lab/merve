# Architecture Overview

## System Architecture

MLServer follows a modular, plugin-based architecture designed for flexibility and performance.

```
┌─────────────────────────────────────────────────────────┐
│                     Client Applications                  │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP/REST
┌─────────────────▼───────────────────────────────────────┐
│                    FastAPI Application                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │              API Endpoints Layer                  │  │
│  │  /predict  /predict_proba  /info  /status        │  │
│  │  /healthz  /metrics                              │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Middleware Stack                     │  │
│  │  CORS │ Metrics │ Logging │ Rate Limiting       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                  Core Components                         │
│  ┌─────────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │   Adapter   │ │ Predictor│ │  Concurrency     │    │
│  │   Manager   │ │  Loader  │ │    Limiter       │    │
│  └─────────────┘ └──────────┘ └──────────────────┘    │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                 Predictor Plugin Layer                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ CatBoost │ │  Random  │ │ XGBoost  │ │  Custom  │  │
│  │Predictor │ │  Forest  │ │Predictor │ │Predictor │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Core Components

### FastAPI Application

The main application layer built on FastAPI:

```python
# mlserver/server.py
app = FastAPI(
    title=config.server.title,
    lifespan=lifespan  # Manages startup/shutdown
)
```

**Responsibilities**:
- Request routing
- Response serialization
- OpenAPI documentation
- Async request handling

### Configuration System

```python
# mlserver/config.py (simplified)
class AppConfig(BaseModel):
    server: ServerConfig
    predictor: PredictorConfig
    observability: ObservabilityConfig
    api: Optional[ApiConfig]
    classifier: Optional[Dict[str, Any]]
    build: Optional[BuildConfig]
    deployment: Optional[DeploymentConfig]
```

**Features**:
- Pydantic validation
- Warnings for unknown/typo'd config keys at load time
- Multi-classifier support (`MultiClassifierConfig` in `mlserver/multi_classifier.py`)

### Predictor Loader

Dynamic module loading system:

```python
# mlserver/predictor_loader.py
def load_predictor(config: PredictorConfig):
    # Resolve module path
    module_path = resolve_module_path(config.module)
    
    # Dynamic import
    module = importlib.import_module(module_path)
    
    # Instantiate predictor
    predictor_class = getattr(module, config.class_name)
    return predictor_class(**config.init_kwargs)
```

### Input Adapters

Input adaptation is function-based, not class-based:

```python
# mlserver/adapters.py
def to_ndarray(payload: dict, adapter: str, feature_order: Optional[List[str]]) -> np.ndarray:
    """Convert a request payload to a numpy array based on the configured adapter."""
```

**Adapter Modes** (`api.adapter`):
- **records** (default): JSON objects with named features (`payload.records` / `payload.instances` / `payload.features`)
- **ndarray**: Nested arrays (`payload.ndarray` / `payload.inputs`)
- **auto**: Format inferred from the payload structure (`_infer_adapter_type`)

Feature ordering for records is cached per feature-name set (`_get_cached_feature_order`).

### Concurrency Control

```python
# mlserver/concurrency_limiter.py (simplified)
class PredictionSemaphore:
    """Limit concurrent predictions; overflow is rejected immediately."""

    def __init__(self, max_concurrent: int = 1, timeout: float = 0):
        self._semaphore = threading.Semaphore(max_concurrent)

class PredictionLimiter:
    """Context manager: acquires the semaphore or raises HTTP 503 (Retry-After: 5)."""
```

## Request Flow

### 1. Request Reception
```
Client → nginx/LB → FastAPI → Middleware Stack
```

### 2. Preprocessing
```python
# Request validation (Pydantic) — body uses the payload wrapper
req = PredictRequest(payload={"records": [...]})

# Correlation ID generation (ObservabilityMiddleware)
correlation_id = set_correlation_id()

# Structured request log
log_request(method="POST", path="/predict", correlation_id=correlation_id)
```

### 3. Adaptation
```python
# Convert input format
input_array = to_ndarray(
    req.payload,
    adapter=config.api.adapter,
    feature_order=feature_order,
)
```

### 4. Prediction
```python
# Concurrency control: acquire a slot or fail fast with 503
with PredictionLimiter(prediction_semaphore):
    # Thread-safe prediction (if api.thread_safe_predict)
    predictions = predictor_wrapper.predict(input_array)
```

### 5. Response
```python
# Format response
response = PredictResponse(
    predictions=_tolist(predictions),
    time_ms=duration_ms,
    predictor_class=predictor_wrapper.name,
    metadata=metadata,  # project, classifier, git info, deployed_at, ...
)

# Return with metrics
return response
```

## Process Model

### Multi-Process Architecture

```
┌─────────────────────────────────────────────┐
│              Main Process                    │
│         (Uvicorn Server Manager)            │
└───────────────┬─────────────────────────────┘
                │ fork()
    ┌───────────┼───────────┬──────────┐
    ▼           ▼           ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │
│(Process)│ │(Process)│ │(Process)│ │(Process)│
└─────────┘ └─────────┘ └─────────┘ └─────────┘
```

**Benefits**:
- True parallelism (no GIL)
- Process isolation
- Container-friendly
- Fault tolerance

### Worker Lifecycle

```python
# Each worker process:
1. Fork from main
2. Load configuration
3. Initialize predictor
4. Start FastAPI app
5. Handle requests
6. Graceful shutdown
```

## Performance Optimizations

### 1. Feature Ordering Cache

```python
# mlserver/adapters.py — feature order is resolved once per feature-name set
def _get_cached_feature_order(records, config_order=None) -> List[str]:
    """Cache the column ordering so repeated requests skip re-sorting keys."""
```

The cache can be inspected/cleared via `get_cache_info()` / `clear_feature_cache()`.

### 2. Numpy Conversions

```python
# mlserver/adapters.py — records are converted to numpy in one pass
def _records_to_numpy_fast(records: List[Dict], feature_order: List[str]) -> np.ndarray:
    ...
```

### 3. Model Warmup

```python
# With api.warmup_on_start (default: true), a dummy prediction runs at startup
# so model internals are initialized before the first real request.
```

### 4. Metrics Caching

```python
# /metrics output is cached in 5-second windows to keep scrape cost low.
```

## Plugin System

### Predictor Interface

```python
class BasePredictor(ABC):
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def predict_proba(self, X):
        """Optional: Return probabilities"""
        raise NotImplementedError
```

### Custom Predictor Example

```python
class CustomPredictor(BasePredictor):
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def predict(self, X):
        return self.model.predict(X)
```

### Registration

```yaml
# mlserver.yaml
predictor:
  module: custom_predictor
  class_name: CustomPredictor
  init_kwargs:
    model_path: "./model.pkl"
```

## Scaling Strategies

### Vertical Scaling

```yaml
# Increase workers
server:
  workers: 16  # More processes (note: /metrics samples one worker per scrape)

api:
  max_concurrent_predictions: 0  # 0 disables per-process concurrency limiting
```

### Horizontal Scaling

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 10  # Multiple pods
  template:
    spec:
      containers:
      - name: mlserver
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
```

### Load Balancing

```nginx
# nginx.conf
upstream mlserver {
    least_conn;
    server mlserver1:8000;
    server mlserver2:8000;
    server mlserver3:8000;
}
```

## Security Architecture

### Request Validation

```python
# mlserver/schemas.py — requests use the payload wrapper
class PredictRequest(BaseModel):
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Prediction input data: 'records' (list of dicts) or 'ndarray' (2D array)"
    )
```

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors.allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)
```

### Concurrency Limiting

```python
# Semaphore-based limiting (mlserver/server.py)
if config.api.max_concurrent_predictions > 0:
    prediction_limiter = PredictionSemaphore(
        max_concurrent=config.api.max_concurrent_predictions,
        timeout=0,  # immediate rejection (503 + Retry-After: 5)
    )
```

## Monitoring Architecture

### Metrics Collection

```python
# mlserver/metrics.py — MetricsCollector (simplified)
request_duration = Histogram(
    "mlserver_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint", "model"],
)

# ObservabilityMiddleware tracks each request (skipping /healthz and /metrics)
# and records duration, status_code and active-request gauges.
```

See [Observability Features](./observability.md#available-metrics) for the full metric list.

### Health Checks

```python
@app.get("/healthz")
def health():
    # Liveness/readiness probe — reports the loaded predictor class
    return HealthResponse(status="ok", model=predictor.name if predictor else None)

@app.get("/status")
def prediction_status():
    # Concurrency-limiter visibility (slots available, active predictions)
    return {...}
```

## Error Handling

### Sanitized Errors

```python
# mlserver/server.py — failures are logged in full, clients get a sanitized detail
try:
    predictions = app.state.predictor.predict(X)
except Exception as e:
    logging.error(f"Prediction error: {e}", exc_info=True)
    raise HTTPException(
        status_code=500,
        detail="Prediction failed. Please contact support if the issue persists."
    )
```

All error responses use the FastAPI shape `{"detail": "..."}`.

### Circuit Breaker Pattern (illustrative — not built in)

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.is_open = False
    
    async def call(self, func, *args):
        if self.is_open:
            raise ServiceUnavailable()
        
        try:
            result = await func(*args)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.is_open = True
            raise
```

## Deployment Architecture

### Container Structure

```dockerfile
# Multi-stage build
FROM python:3.9-slim as builder
# Build dependencies

FROM python:3.9-slim
# Runtime only
COPY --from=builder /app /app
CMD ["merve", "serve"]
```

### Kubernetes Architecture

```yaml
# Service mesh integration
apiVersion: v1
kind: Service
metadata:
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
```

## Performance Benchmarks

> **Note**: the numbers below are illustrative, meant to show the shape of the latency/throughput profile — they are not measured benchmarks of this codebase. Actual figures depend entirely on your model and hardware.

### Latency Profile

| Component | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Request Parse | 1ms | 2ms | 5ms |
| Adaptation | 2ms | 5ms | 10ms |
| Prediction | 20ms | 50ms | 100ms |
| Response | 1ms | 2ms | 5ms |
| **Total** | **24ms** | **59ms** | **120ms** |

### Throughput

| Workers | RPS | CPU | Memory |
|---------|-----|-----|--------|
| 1 | 100 | 25% | 200MB |
| 4 | 400 | 90% | 800MB |
| 8 | 750 | 95% | 1.6GB |
| 16 | 1200 | 95% | 3.2GB |

### Scalability

```
Throughput vs Workers
1500 RPS │     ╱─────
         │    ╱
1000 RPS │   ╱
         │  ╱
500 RPS  │ ╱
         │╱
0 RPS    └────────────
         0  4  8  12  16
            Workers
```