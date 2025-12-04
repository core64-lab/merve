from __future__ import annotations
import threading
import time
from typing import Optional, Any, Union
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry
from fastapi import Request, Response


def count_samples(data: Any) -> int:
    """Count the number of samples in prediction input or output.

    Handles various data types:
    - numpy.ndarray: shape[0] or 1 for scalars
    - pandas.DataFrame: len(df)
    - pandas.Series: len(series)
    - list/tuple: len() for nested lists, 1 for flat values
    - dict: len() if it looks like a batch, 1 otherwise
    - scalar: 1

    Args:
        data: Input or output data from prediction

    Returns:
        Number of samples (minimum 1)
    """
    if data is None:
        return 1

    # numpy array
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                return 1
            return data.shape[0] if len(data.shape) > 0 else 1
    except ImportError:
        pass

    # pandas DataFrame or Series
    try:
        import pandas as pd
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return len(data)
    except ImportError:
        pass

    # list or tuple
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return 1
        # Check if it's a list of samples (nested) vs flat array
        if isinstance(data[0], (list, tuple, dict)):
            return len(data)
        # Could be a flat array representing one sample
        return len(data) if len(data) > 0 else 1

    # dict - check if it looks like a batch
    if isinstance(data, dict):
        # If dict contains 'predictions', 'instances', 'records' etc, count those
        for key in ['predictions', 'instances', 'records', 'results', 'outputs']:
            if key in data and isinstance(data[key], (list, tuple)):
                return len(data[key])
        # If dict values are all same-length lists, it might be columnar format
        values = list(data.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            lengths = [len(v) for v in values]
            if len(set(lengths)) == 1:  # All same length
                return lengths[0]
        # Single dict = single sample
        return 1

    # Scalar or unknown type
    return 1


class MetricsCollector:
    def __init__(self, model_name: Optional[str] = None, registry: Optional[CollectorRegistry] = None):
        self.model_name = model_name or "unknown"
        self._registry = registry or REGISTRY

        # Request metrics
        self.request_count = Counter(
            "mlserver_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status_code", "model"],
            registry=self._registry
        )

        self.request_duration = Histogram(
            "mlserver_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint", "model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self._registry
        )

        # Prediction-specific metrics
        self.prediction_duration = Histogram(
            "mlserver_prediction_duration_seconds",
            "Model prediction duration in seconds",
            ["model", "endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self._registry
        )

        self.prediction_count = Counter(
            "mlserver_predictions_total",
            "Total number of predictions made (output samples)",
            ["model", "endpoint"],
            registry=self._registry
        )

        # Input samples counter (tracks batch sizes)
        self.input_samples = Counter(
            "mlserver_input_samples_total",
            "Total number of input samples received",
            ["model", "endpoint"],
            registry=self._registry
        )

        # System metrics
        self.active_requests = Gauge(
            "mlserver_active_requests",
            "Number of active requests",
            ["model"],
            registry=self._registry
        )

        # Batch size histogram for monitoring batch patterns
        self.batch_size = Histogram(
            "mlserver_batch_size",
            "Distribution of prediction batch sizes",
            ["model", "endpoint"],
            buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self._registry
        )

        # Model info
        self.model_info = Gauge(
            "mlserver_model_info",
            "Model information",
            ["model", "version"],
            registry=self._registry
        )
        self.model_info.labels(model=self.model_name, version="1.0").set(1)

    def track_request(self, request: Request, response: Response, duration: float):
        """Track general request metrics"""
        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)

        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            model=self.model_name
        ).inc()

        self.request_duration.labels(
            method=method,
            endpoint=endpoint,
            model=self.model_name
        ).observe(duration)

    def track_prediction(self, endpoint: str, duration: float,
                         input_samples: int = 1, output_samples: int = 1):
        """Track prediction-specific metrics.

        Args:
            endpoint: The API endpoint path
            duration: Prediction duration in seconds
            input_samples: Number of input samples (batch size)
            output_samples: Number of output samples (predictions made)
        """
        self.prediction_duration.labels(
            model=self.model_name,
            endpoint=endpoint
        ).observe(duration)

        # Track output samples (predictions made)
        self.prediction_count.labels(
            model=self.model_name,
            endpoint=endpoint
        ).inc(output_samples)

        # Track input samples
        self.input_samples.labels(
            model=self.model_name,
            endpoint=endpoint
        ).inc(input_samples)

        # Track batch size distribution
        self.batch_size.labels(
            model=self.model_name,
            endpoint=endpoint
        ).observe(input_samples)

    def inc_active_requests(self):
        """Increment active requests counter"""
        self.active_requests.labels(model=self.model_name).inc()

    def dec_active_requests(self):
        """Decrement active requests counter"""
        self.active_requests.labels(model=self.model_name).dec()

    def generate_metrics(self) -> str:
        """Generate Prometheus metrics in text format"""
        return generate_latest()

    def get_content_type(self) -> str:
        """Get the content type for metrics response"""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance with thread safety
_metrics_collector: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def init_metrics(model_name: Optional[str] = None) -> MetricsCollector:
    """Initialize global metrics collector (thread-safe singleton).

    Uses double-checked locking pattern to ensure thread safety while
    minimizing lock contention after initialization.
    """
    global _metrics_collector
    # Fast path: already initialized
    if _metrics_collector is not None:
        return _metrics_collector

    with _metrics_lock:
        # Double-check inside lock to prevent race condition
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector(model_name)
        return _metrics_collector


def get_metrics() -> Optional[MetricsCollector]:
    """Get global metrics collector instance (thread-safe)."""
    # Reading a reference is atomic in Python, no lock needed for get
    return _metrics_collector


def reset_metrics() -> None:
    """Reset global metrics collector (for testing purposes).

    Thread-safe reset of the singleton instance.
    Also clears the Prometheus registry to avoid duplicate metric errors.
    """
    global _metrics_collector
    with _metrics_lock:
        if _metrics_collector is not None:
            # Unregister all metrics from the default registry
            collectors_to_remove = []
            for collector in list(REGISTRY._collector_to_names.keys()):
                # Check if this collector belongs to our metrics
                names = REGISTRY._collector_to_names.get(collector, [])
                if any(name.startswith('mlserver_') for name in names):
                    collectors_to_remove.append(collector)

            for collector in collectors_to_remove:
                try:
                    REGISTRY.unregister(collector)
                except Exception:
                    pass  # Ignore errors during cleanup

        _metrics_collector = None


def create_test_registry() -> CollectorRegistry:
    """Create an isolated registry for testing.

    Returns a new CollectorRegistry that won't conflict with
    the default REGISTRY, useful for parallel test execution.
    """
    return CollectorRegistry()