import pytest
import time
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from fastapi import Request, Response
from prometheus_client import REGISTRY, CollectorRegistry

from mlserver.metrics import (
    MetricsCollector,
    init_metrics,
    get_metrics,
    reset_metrics,
    count_samples,
    create_test_registry
)


class TestCountSamples:
    """Test the count_samples function for various input types."""

    def test_none_returns_one(self):
        """None should return 1."""
        assert count_samples(None) == 1

    def test_scalar_returns_one(self):
        """Scalar values should return 1."""
        assert count_samples(42) == 1
        assert count_samples(3.14) == 1
        assert count_samples("hello") == 1
        assert count_samples(True) == 1

    # numpy arrays
    def test_numpy_1d_array(self):
        """1D numpy array returns its length."""
        arr = np.array([1, 2, 3, 4, 5])
        assert count_samples(arr) == 5

    def test_numpy_2d_array(self):
        """2D numpy array returns number of rows."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        assert count_samples(arr) == 3

    def test_numpy_scalar(self):
        """0D numpy array (scalar) returns 1."""
        arr = np.array(42)
        assert count_samples(arr) == 1

    # pandas DataFrames and Series
    def test_pandas_dataframe(self):
        """DataFrame returns number of rows."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert count_samples(df) == 3

    def test_pandas_series(self):
        """Series returns its length."""
        series = pd.Series([1, 2, 3, 4])
        assert count_samples(series) == 4

    def test_pandas_empty_dataframe(self):
        """Empty DataFrame returns 0."""
        df = pd.DataFrame()
        assert count_samples(df) == 0

    # Lists and tuples
    def test_list_of_dicts(self):
        """List of dicts (records format) returns count."""
        data = [{'a': 1}, {'a': 2}, {'a': 3}]
        assert count_samples(data) == 3

    def test_list_of_lists(self):
        """List of lists (2D array format) returns row count."""
        data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert count_samples(data) == 4

    def test_flat_list(self):
        """Flat list of values returns its length."""
        data = [1, 2, 3, 4, 5]
        assert count_samples(data) == 5

    def test_empty_list(self):
        """Empty list returns 1."""
        assert count_samples([]) == 1

    def test_tuple_of_dicts(self):
        """Tuple of dicts returns count."""
        data = ({'a': 1}, {'a': 2})
        assert count_samples(data) == 2

    # Dictionaries
    def test_dict_with_predictions_key(self):
        """Dict with 'predictions' key counts those."""
        data = {'predictions': [1, 2, 3, 4]}
        assert count_samples(data) == 4

    def test_dict_with_instances_key(self):
        """Dict with 'instances' key counts those."""
        data = {'instances': [{'f1': 1}, {'f1': 2}]}
        assert count_samples(data) == 2

    def test_dict_with_records_key(self):
        """Dict with 'records' key counts those."""
        data = {'records': [{'a': 1}, {'a': 2}, {'a': 3}]}
        assert count_samples(data) == 3

    def test_dict_columnar_format(self):
        """Dict in columnar format counts rows."""
        data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        assert count_samples(data) == 3

    def test_single_dict(self):
        """Single dict without special keys returns 1."""
        data = {'feature1': 1.0, 'feature2': 2.0}
        assert count_samples(data) == 1


class TestMetricsCollector:
    """Test MetricsCollector functionality"""

    def test_init_with_model_name(self):
        collector = MetricsCollector("TestModel")
        assert collector.model_name == "TestModel"

    def test_init_without_model_name(self):
        collector = MetricsCollector()
        assert collector.model_name == "unknown"

    def test_metrics_creation(self):
        collector = MetricsCollector("TestModel")

        # Test that all metrics are created
        assert collector.request_count is not None
        assert collector.request_duration is not None
        assert collector.prediction_duration is not None
        assert collector.prediction_count is not None
        assert collector.active_requests is not None
        assert collector.model_info is not None

    def test_track_request(self):
        collector = MetricsCollector("TestModel")

        # Mock request and response
        request = Mock(spec=Request)
        request.url.path = "/predict"
        request.method = "POST"

        response = Mock(spec=Response)
        response.status_code = 200

        duration = 0.1

        # Track request
        collector.track_request(request, response, duration)

        # Verify metrics were updated
        # Note: We can't easily check the exact values due to Prometheus client internals
        # but we can verify the methods were called without error

    def test_track_prediction(self):
        collector = MetricsCollector("TestModel")

        endpoint = "/predict"
        duration = 0.05
        input_samples = 2
        output_samples = 2

        # Track prediction with new signature
        collector.track_prediction(endpoint, duration,
                                   input_samples=input_samples,
                                   output_samples=output_samples)

        # Should complete without error

    def test_active_requests_counter(self):
        collector = MetricsCollector("TestModel")

        # Test increment
        collector.inc_active_requests()

        # Test decrement
        collector.dec_active_requests()

        # Should complete without error

    def test_generate_metrics(self):
        collector = MetricsCollector("TestModel")

        # Generate some metrics
        metrics_output = collector.generate_metrics()

        assert isinstance(metrics_output, bytes)
        assert b"mlserver_" in metrics_output
        assert b"TestModel" in metrics_output

    def test_get_content_type(self):
        collector = MetricsCollector("TestModel")
        content_type = collector.get_content_type()
        assert "text/plain" in content_type

    def test_metrics_with_labels(self):
        collector = MetricsCollector("MyModel")

        # Mock request with different endpoints
        request1 = Mock(spec=Request)
        request1.url.path = "/predict"
        request1.method = "POST"

        request2 = Mock(spec=Request)
        request2.url.path = "/predict_proba"
        request2.method = "POST"

        response = Mock(spec=Response)
        response.status_code = 200

        # Track different requests
        collector.track_request(request1, response, 0.1)
        collector.track_request(request2, response, 0.2)

        # Track predictions with different endpoints
        collector.track_prediction("/predict", 0.05, input_samples=1, output_samples=1)
        collector.track_prediction("/predict_proba", 0.08, input_samples=1, output_samples=1)

        # Generate metrics and check for labels
        metrics_output = collector.generate_metrics()
        assert b"/predict" in metrics_output
        assert b"/predict_proba" in metrics_output
        assert b"MyModel" in metrics_output

    def test_error_response_tracking(self):
        collector = MetricsCollector("TestModel")

        request = Mock(spec=Request)
        request.url.path = "/predict"
        request.method = "POST"

        response = Mock(spec=Response)
        response.status_code = 500  # Error response

        # Should handle error responses
        collector.track_request(request, response, 0.1)

    def test_different_http_methods(self):
        collector = MetricsCollector("TestModel")

        methods = ["GET", "POST", "PUT"]
        for method in methods:
            request = Mock(spec=Request)
            request.url.path = "/test"
            request.method = method

            response = Mock(spec=Response)
            response.status_code = 200

            collector.track_request(request, response, 0.1)

    def test_histogram_buckets(self):
        collector = MetricsCollector("TestModel")

        # Test different durations to hit different buckets
        durations = [0.001, 0.01, 0.1, 1.0, 5.0]

        request = Mock(spec=Request)
        request.url.path = "/predict"
        request.method = "POST"

        response = Mock(spec=Response)
        response.status_code = 200

        for duration in durations:
            collector.track_request(request, response, duration)
            collector.track_prediction("/predict", duration, input_samples=1, output_samples=1)

    def test_multiple_samples_prediction(self):
        collector = MetricsCollector("TestModel")

        # Track prediction with multiple samples
        collector.track_prediction("/batch_predict", 0.5, input_samples=10, output_samples=10)

        metrics_output = collector.generate_metrics()
        assert b"batch_predict" in metrics_output


class TestMetricsModule:
    """Test module-level functions"""

    def test_init_metrics(self):
        collector = init_metrics("TestModel")
        assert isinstance(collector, MetricsCollector)
        assert collector.model_name == "TestModel"

    def test_get_metrics_after_init(self):
        init_metrics("TestModel")
        collector = get_metrics()
        assert isinstance(collector, MetricsCollector)
        assert collector.model_name == "TestModel"

    def test_get_metrics_before_init(self):
        # Reset global collector
        import mlserver.metrics
        mlserver.metrics._metrics_collector = None

        collector = get_metrics()
        assert collector is None

    def test_reinit_metrics(self):
        # Initialize first time
        collector1 = init_metrics("Model1")
        assert collector1.model_name == "Model1"

        # Clear registry before reinitializing to avoid duplicate metrics
        from prometheus_client import REGISTRY
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass

        # Reset global collector
        from mlserver import metrics
        metrics._metrics_collector = None

        # Initialize again with different name
        collector2 = init_metrics("Model2")
        assert collector2.model_name == "Model2"

        # Should get the new one
        current = get_metrics()
        assert current.model_name == "Model2"


class TestMetricsIntegration:
    """Test metrics in realistic scenarios"""

    def test_complete_request_cycle(self):
        collector = MetricsCollector("IntegrationTest")

        # Simulate a complete request
        request = Mock(spec=Request)
        request.url.path = "/predict"
        request.method = "POST"

        response = Mock(spec=Response)
        response.status_code = 200

        # Track active request
        collector.inc_active_requests()

        # Track prediction
        prediction_duration = 0.05
        collector.track_prediction("/predict", prediction_duration, input_samples=2, output_samples=2)

        # Track complete request
        total_duration = 0.1
        collector.track_request(request, response, total_duration)

        # Decrement active requests
        collector.dec_active_requests()

        # Generate metrics
        metrics_output = collector.generate_metrics()
        assert b"mlserver_requests_total" in metrics_output
        assert b"mlserver_request_duration_seconds" in metrics_output
        assert b"mlserver_prediction_duration_seconds" in metrics_output
        assert b"mlserver_predictions_total" in metrics_output

    def test_concurrent_requests_simulation(self):
        collector = MetricsCollector("ConcurrentTest")

        # Simulate multiple concurrent requests
        for i in range(5):
            collector.inc_active_requests()

        # Process requests
        for i in range(5):
            request = Mock(spec=Request)
            request.url.path = f"/predict_{i}"
            request.method = "POST"

            response = Mock(spec=Response)
            response.status_code = 200

            collector.track_prediction(f"/predict_{i}", 0.1, input_samples=1, output_samples=1)
            collector.track_request(request, response, 0.15)
            collector.dec_active_requests()

    def test_error_scenarios(self):
        collector = MetricsCollector("ErrorTest")

        # Test various error scenarios
        error_codes = [400, 422, 500]

        for code in error_codes:
            request = Mock(spec=Request)
            request.url.path = "/predict"
            request.method = "POST"

            response = Mock(spec=Response)
            response.status_code = code

            collector.track_request(request, response, 0.1)

        # Should track different status codes
        metrics_output = collector.generate_metrics()
        assert b"400" in metrics_output or b"422" in metrics_output or b"500" in metrics_output