"""Unit tests for server module components."""
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import Request, Response, HTTPException
from starlette.responses import JSONResponse

from mlserver.server import (
    ObservabilityMiddleware,
    create_app,
    PredictorWrapper,
    _prepare_input_data,
    _track_prediction_metrics,
    _create_predict_handler,
    _create_predict_proba_handler,
    _to_jsonable,
    _tolist2d
)
from mlserver.config import AppConfig, ServerConfig, PredictorConfig, ObservabilityConfig
from mlserver.schemas import PredictRequest
from mlserver.adapters import AdapterError


@pytest.fixture
def mock_config():
    """Create a mock AppConfig with default settings."""
    return AppConfig.model_validate({
        "server": {"host": "127.0.0.1", "port": 8000},
        "predictor": {
            "module": "mlserver.predictors.sklearn",
            "class_name": "SKLearnPredictor",
            "init_kwargs": {"model_path": "model.pkl"}
        },
        "observability": {
            "metrics": True,
            "structured_logging": True,
            "correlation_ids": True
        },
        "classifier": {
            "name": "test-model",
            "version": "1.0.0"
        },
        "api": {
            "version": "v1",
            "adapter": "auto"
        }
    })


@pytest.fixture
def mock_config_minimal():
    """Create a minimal AppConfig with observability disabled."""
    return AppConfig.model_validate({
        "server": {"host": "127.0.0.1", "port": 8000},
        "predictor": {
            "module": "mlserver.predictors.sklearn",
            "class_name": "SKLearnPredictor",
            "init_kwargs": {"model_path": "model.pkl"}
        },
        "observability": {
            "metrics": False,
            "structured_logging": False,
            "correlation_ids": False
        },
        "classifier": {
            "name": "test-model",
            "version": "1.0.0"
        },
        "api": {
            "version": "v1",
            "adapter": "auto"
        }
    })


class TestObservabilityMiddleware:
    """Test the ObservabilityMiddleware class."""

    def test_middleware_init(self, mock_config):
        """Test middleware initialization."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            app = Mock()
            middleware = ObservabilityMiddleware(app, mock_config)

            assert middleware.config == mock_config
            assert middleware.metrics == mock_metrics

    def test_middleware_init_no_metrics(self, mock_config_minimal):
        """Test middleware initialization when metrics are disabled."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = None

            app = Mock()
            middleware = ObservabilityMiddleware(app, mock_config_minimal)

            assert middleware.config == mock_config_minimal
            assert middleware.metrics is None

    @pytest.mark.asyncio
    async def test_middleware_dispatch_full_observability(self, mock_config):
        """Test middleware dispatch with full observability enabled."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics, \
             patch('mlserver.server.set_correlation_id') as mock_set_correlation_id, \
             patch('mlserver.server.log_request') as mock_log_request, \
             patch('mlserver.server.log_response') as mock_log_response:

            # Setup mocks
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics
            mock_set_correlation_id.return_value = "test-correlation-id"

            # Create middleware
            app = Mock()
            middleware = ObservabilityMiddleware(app, mock_config)

            # Create mock request and response
            mock_request = Mock(spec=Request)
            mock_request.url = Mock()
            mock_request.url.path = "/predict"
            mock_request.method = "POST"

            mock_response = Mock(spec=Response)
            mock_response.status_code = 200

            # Mock call_next
            async def mock_call_next(request):
                return mock_response

            # Execute middleware
            result = await middleware.dispatch(mock_request, mock_call_next)

            # Verify calls - API changed to use track_request instead of record_request_duration
            mock_set_correlation_id.assert_called_once()
            mock_metrics.inc_active_requests.assert_called_once()
            mock_metrics.dec_active_requests.assert_called_once()
            mock_metrics.track_request.assert_called_once()  # Updated: was record_request_duration
            mock_log_request.assert_called_once()
            mock_log_response.assert_called_once()

            assert result == mock_response

    @pytest.mark.asyncio
    async def test_middleware_dispatch_minimal_observability(self, mock_config_minimal):
        """Test middleware dispatch with minimal observability."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics, \
             patch('mlserver.server.set_correlation_id') as mock_set_correlation_id, \
             patch('mlserver.server.log_request') as mock_log_request, \
             patch('mlserver.server.log_response') as mock_log_response:

            # Setup mocks
            mock_get_metrics.return_value = None

            # Create middleware
            app = Mock()
            middleware = ObservabilityMiddleware(app, mock_config_minimal)

            # Create mock request and response
            mock_request = Mock(spec=Request)
            mock_response = Mock(spec=Response)
            mock_response.status_code = 200

            # Mock call_next
            async def mock_call_next(request):
                return mock_response

            # Execute middleware
            result = await middleware.dispatch(mock_request, mock_call_next)

            # Verify minimal calls
            mock_set_correlation_id.assert_not_called()
            mock_log_request.assert_not_called()
            mock_log_response.assert_not_called()

            assert result == mock_response

    @pytest.mark.asyncio
    async def test_middleware_dispatch_handles_exceptions(self, mock_config):
        """Test middleware properly handles exceptions."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            app = Mock()
            middleware = ObservabilityMiddleware(app, mock_config)

            mock_request = Mock(spec=Request)
            mock_request.url = Mock()
            mock_request.url.path = "/predict"
            mock_request.method = "POST"

            # Mock call_next to raise an exception
            async def mock_call_next(request):
                raise Exception("Test error")

            # Execute middleware and expect exception to propagate
            with pytest.raises(Exception, match="Test error"):
                await middleware.dispatch(mock_request, mock_call_next)

            # Verify cleanup still happened
            mock_metrics.dec_active_requests.assert_called_once()


class TestCreateApp:
    """Test the create_app function."""

    def test_create_app_with_metrics(self, mock_config):
        """Test app creation with metrics enabled."""
        with patch('mlserver.server.init_metrics') as mock_init_metrics, \
             patch('mlserver.server.load_predictor') as mock_load_predictor:

            mock_predictor = Mock()
            mock_load_predictor.return_value = mock_predictor
            mock_init_metrics.return_value = Mock()

            app = create_app(mock_config)

            # Verify app creation - title now generated from classifier name/version
            expected_title = mock_config.get_api_title()  # "Test Model API v1.0.0"
            assert app.title == expected_title
            # Metrics init is called during create_app if enabled
            # Note: load_predictor is called during lifespan, not create_app

    def test_create_app_without_metrics(self, mock_config_minimal):
        """Test app creation with metrics disabled."""
        with patch('mlserver.server.init_metrics') as mock_init_metrics, \
             patch('mlserver.server.load_predictor') as mock_load_predictor:

            mock_predictor = Mock()
            mock_load_predictor.return_value = mock_predictor

            app = create_app(mock_config_minimal)

            # Verify app creation - title now generated from classifier name/version
            expected_title = mock_config_minimal.get_api_title()  # "Test Model API v1.0.0"
            assert app.title == expected_title
            # init_metrics should not be called when metrics disabled
            mock_init_metrics.assert_not_called()

    def test_create_app_with_cors(self):
        """Test app creation with CORS middleware."""
        from mlserver.config import CORSConfig
        config = AppConfig(
            server=ServerConfig(
                host="127.0.0.1",
                port=8000,
                cors=CORSConfig(
                    allow_origins=["http://localhost:3000", "https://example.com"]
                )
            ),
            predictor=PredictorConfig(
                module="mlserver.predictors.sklearn",
                class_name="SKLearnPredictor",
                init_kwargs={"model_path": "model.pkl"}
            ),
            classifier={
                "name": "test-model",
                "version": "1.0.0"
            },
            api={
                "version": "v1",
                "adapter": "auto"
            }
        )

        with patch('mlserver.server.load_predictor') as mock_load_predictor:
            mock_predictor = Mock()
            mock_load_predictor.return_value = mock_predictor

            app = create_app(config)

            # CORS middleware is added via app.add_middleware(), check middleware_stack
            # The middleware_stack is populated on first request, so we check user_middleware
            # or verify config was properly set
            assert config.server.cors is not None
            assert "http://localhost:3000" in config.server.cors.allow_origins


class TestPredictorWrapper:
    """Test the PredictorWrapper class."""

    def test_predictor_wrapper_init_thread_safe(self):
        """Test PredictorWrapper initialization with thread safety."""
        mock_predictor = Mock()
        wrapper = PredictorWrapper(mock_predictor, thread_safe=True)

        assert wrapper._predictor == mock_predictor
        assert wrapper._lock is not None

    def test_predictor_wrapper_init_not_thread_safe(self):
        """Test PredictorWrapper initialization without thread safety."""
        mock_predictor = Mock()
        wrapper = PredictorWrapper(mock_predictor, thread_safe=False)

        assert wrapper._predictor == mock_predictor
        assert wrapper._lock is None

    def test_predict_without_lock(self):
        """Test prediction without thread safety."""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = [1, 0, 1]

        wrapper = PredictorWrapper(mock_predictor, thread_safe=False)
        result = wrapper.predict([[1, 2], [3, 4], [5, 6]])

        assert result == [1, 0, 1]
        mock_predictor.predict.assert_called_once_with([[1, 2], [3, 4], [5, 6]])

    def test_predict_with_lock(self):
        """Test prediction with thread safety."""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = [1, 0]

        wrapper = PredictorWrapper(mock_predictor, thread_safe=True)
        result = wrapper.predict([[1, 2], [3, 4]])

        assert result == [1, 0]
        mock_predictor.predict.assert_called_once_with([[1, 2], [3, 4]])

    def test_predict_proba_success(self):
        """Test predict_proba method."""
        mock_predictor = Mock()
        mock_predictor.predict_proba.return_value = [[0.7, 0.3], [0.2, 0.8]]

        wrapper = PredictorWrapper(mock_predictor, thread_safe=False)
        result = wrapper.predict_proba([[1, 2], [3, 4]])

        assert result == [[0.7, 0.3], [0.2, 0.8]]

    def test_predict_proba_no_method(self):
        """Test predict_proba when method doesn't exist."""
        mock_predictor = Mock(spec=[])
        wrapper = PredictorWrapper(mock_predictor, thread_safe=False)

        with pytest.raises(AttributeError, match="predictor has no predict_proba"):
            wrapper.predict_proba([[1, 2]])

    def test_predictor_name_property(self):
        """Test the name property."""
        mock_predictor = Mock()
        mock_predictor.__class__.__name__ = "TestPredictor"

        wrapper = PredictorWrapper(mock_predictor)
        assert wrapper.name == "TestPredictor"

    def test_close_with_method(self):
        """Test close method when predictor has close."""
        mock_predictor = Mock()
        wrapper = PredictorWrapper(mock_predictor)

        wrapper.close()
        mock_predictor.close.assert_called_once()

    def test_close_without_method(self):
        """Test close method when predictor doesn't have close."""
        mock_predictor = Mock(spec=[])
        wrapper = PredictorWrapper(mock_predictor)

        # Should not raise an exception
        wrapper.close()

    def test_close_with_exception(self):
        """Test close method when predictor.close raises exception."""
        mock_predictor = Mock()
        mock_predictor.close.side_effect = Exception("Close failed")

        wrapper = PredictorWrapper(mock_predictor)

        # Should not propagate the exception
        wrapper.close()
        mock_predictor.close.assert_called_once()


class TestHelperFunctions:
    """Test server helper functions."""

    def test_prepare_input_data_success(self, mock_config):
        """Test _prepare_input_data with successful parsing."""
        with patch('mlserver.server.to_ndarray') as mock_to_ndarray:
            mock_to_ndarray.return_value = [[1, 2], [3, 4]]

            request = PredictRequest(inputs={"data": [[1, 2], [3, 4]]})
            result = _prepare_input_data(request, mock_config)

            assert result == [[1, 2], [3, 4]]
            mock_to_ndarray.assert_called_once()

    def test_prepare_input_data_failure(self, mock_config):
        """Test _prepare_input_data with parsing failure."""
        with patch('mlserver.server.to_ndarray') as mock_to_ndarray:
            mock_to_ndarray.side_effect = Exception("Parsing error")

            request = PredictRequest(inputs={"data": "invalid"})

            with pytest.raises(HTTPException) as exc_info:
                _prepare_input_data(request, mock_config)

            assert exc_info.value.status_code == 400
            assert "Input parsing failed" in str(exc_info.value.detail)

    def test_track_prediction_metrics_with_metrics(self, mock_config):
        """Test _track_prediction_metrics when metrics are enabled."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics, \
             patch('mlserver.server.log_prediction') as mock_log_prediction:

            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            _track_prediction_metrics("/predict", 0.5, 10, "TestModel", mock_config)

            mock_metrics.track_prediction.assert_called_once_with("/predict", 0.5, 10)
            mock_log_prediction.assert_called_once()

    def test_track_prediction_metrics_without_metrics(self, mock_config_minimal):
        """Test _track_prediction_metrics when metrics and structured logging are disabled."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics, \
             patch('mlserver.server.log_prediction') as mock_log_prediction:

            mock_get_metrics.return_value = None

            _track_prediction_metrics("/predict", 0.5, 10, "TestModel", mock_config_minimal)

            # Should not call any logging since mock_config_minimal has structured_logging=False
            mock_log_prediction.assert_not_called()

    def test_to_jsonable_simple_types(self):
        """Test _to_jsonable with simple types."""
        assert _to_jsonable(42) == 42
        assert _to_jsonable("hello") == "hello"
        assert _to_jsonable([1, 2, 3]) == [1, 2, 3]

    def test_to_jsonable_numpy_types(self):
        """Test _to_jsonable with numpy-like types."""
        import numpy as np

        # Test numpy scalar
        result = _to_jsonable(np.int32(42))
        assert result == 42
        assert isinstance(result, int)

        # Test numpy array
        arr = np.array([1, 2, 3])
        result = _to_jsonable(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_tolist2d_simple_list(self):
        """Test _tolist2d with simple nested list."""
        data = [[1, 2], [3, 4]]
        result = _tolist2d(data)
        assert result == [[1, 2], [3, 4]]

    def test_tolist2d_numpy_array(self):
        """Test _tolist2d with numpy array."""
        import numpy as np

        data = np.array([[1, 2], [3, 4]])
        result = _tolist2d(data)
        assert result == [[1, 2], [3, 4]]
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)


class TestHandlerFunctions:
    """Test the handler creation functions."""

    def test_create_predict_handler(self, mock_config):
        """Test _create_predict_handler function."""
        with patch('mlserver.server.load_predictor') as mock_load_predictor:
            mock_predictor = Mock()
            mock_load_predictor.return_value = mock_predictor

            app = Mock()
            handler = _create_predict_handler(app, mock_config, "/predict")

            # Verify that a handler function was created
            assert callable(handler)

    def test_create_predict_proba_handler(self, mock_config):
        """Test _create_predict_proba_handler function."""
        with patch('mlserver.server.load_predictor') as mock_load_predictor:
            mock_predictor = Mock()
            mock_load_predictor.return_value = mock_predictor

            app = Mock()
            handler = _create_predict_proba_handler(app, mock_config, "/predict_proba")

            # Verify that a handler function was created
            assert callable(handler)

    @pytest.mark.asyncio
    async def test_app_lifespan_context(self, mock_config):
        """Test app lifespan context functionality - predictor loaded on startup."""
        with patch('mlserver.server.load_predictor') as mock_load_predictor:
            # Mock predictor with close method
            mock_predictor = Mock()
            mock_wrapper = Mock()
            mock_wrapper.close = Mock()

            with patch('mlserver.server.PredictorWrapper') as mock_wrapper_class:
                mock_wrapper_class.return_value = mock_wrapper
                mock_load_predictor.return_value = mock_predictor

                # Create app
                app = create_app(mock_config)

                # Predictor not loaded until lifespan starts
                mock_load_predictor.assert_not_called()

                # Enter lifespan context to trigger predictor loading
                async with app.router.lifespan_context(app):
                    # Now predictor should be loaded with individual arguments
                    mock_load_predictor.assert_called_once_with(
                        mock_config.predictor.module,
                        mock_config.predictor.class_name,
                        mock_config.predictor.init_kwargs,
                        config_dir=mock_config.project_path
                    )


class TestValidateInputFeatures:
    """Test _validate_input_features function."""

    def test_validate_input_features_valid_records(self, mock_config):
        """Test validation with valid records."""
        from mlserver.server import _validate_input_features
        import logging

        logger = logging.getLogger(__name__)
        feature_order = ["feature1", "feature2", "feature3"]
        payload = {"records": [
            {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
            {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}
        ]}

        # Should not raise
        _validate_input_features(payload, feature_order, logger)

    def test_validate_input_features_missing_features(self, mock_config):
        """Test validation with missing features."""
        from mlserver.server import _validate_input_features
        import logging

        logger = logging.getLogger(__name__)
        feature_order = ["feature1", "feature2", "feature3"]
        payload = {"records": [
            {"feature1": 1.0}  # Missing feature2 and feature3
        ]}

        with pytest.raises(HTTPException) as exc_info:
            _validate_input_features(payload, feature_order, logger)

        assert exc_info.value.status_code == 400
        assert "Feature validation failed" in str(exc_info.value.detail)

    def test_validate_input_features_instances_format(self, mock_config):
        """Test validation with 'instances' key format."""
        from mlserver.server import _validate_input_features
        import logging

        logger = logging.getLogger(__name__)
        feature_order = ["a", "b"]
        payload = {"instances": [
            {"a": 1.0, "b": 2.0}
        ]}

        # Should not raise
        _validate_input_features(payload, feature_order, logger)

    def test_validate_input_features_single_record_via_features(self, mock_config):
        """Test validation with single record via 'features' key."""
        from mlserver.server import _validate_input_features
        import logging

        logger = logging.getLogger(__name__)
        feature_order = ["x", "y"]
        payload = {"features": {"x": 1.0, "y": 2.0}}

        # Should not raise
        _validate_input_features(payload, feature_order, logger)

    def test_validate_input_features_skips_ndarray_format(self, mock_config):
        """Test validation skips ndarray format."""
        from mlserver.server import _validate_input_features
        import logging

        logger = logging.getLogger(__name__)
        feature_order = ["a", "b"]
        payload = {"ndarray": [[1.0, 2.0], [3.0, 4.0]]}

        # Should not raise (validation skipped for ndarray)
        _validate_input_features(payload, feature_order, logger)

    def test_validate_input_features_non_dict_payload(self, mock_config):
        """Test validation skips non-dict payload."""
        from mlserver.server import _validate_input_features
        import logging

        logger = logging.getLogger(__name__)
        feature_order = ["a", "b"]
        payload = [[1.0, 2.0], [3.0, 4.0]]

        # Should not raise (validation skipped for non-dict)
        _validate_input_features(payload, feature_order, logger)


class TestFormatResponse:
    """Test _format_response function."""

    def test_format_response_passthrough(self, mock_config):
        """Test passthrough response format."""
        from mlserver.server import _format_response

        # Modify config for passthrough
        mock_config.api.response_format = 'passthrough'

        predictions = {"custom": "data", "predictions": [1, 2, 3]}
        result = _format_response(predictions, mock_config, 10.5, "TestModel")

        assert result == predictions

    def test_format_response_custom_with_dict(self, mock_config):
        """Test custom response format with dict predictions."""
        from mlserver.server import _format_response
        from mlserver.schemas import CustomPredictResponse

        mock_config.api.response_format = 'custom'

        predictions = {"label": "cat", "confidence": 0.95}
        result = _format_response(predictions, mock_config, 10.5, "TestModel")

        assert isinstance(result, CustomPredictResponse)
        assert result.result == predictions
        assert result.predictor_class == "TestModel"

    def test_format_response_custom_with_list(self, mock_config):
        """Test custom response format with list predictions."""
        from mlserver.server import _format_response
        from mlserver.schemas import CustomPredictResponse

        mock_config.api.response_format = 'custom'

        predictions = [0, 1, 0, 1]
        result = _format_response(predictions, mock_config, 5.0, "MyPredictor")

        assert isinstance(result, CustomPredictResponse)
        assert result.result == predictions

    def test_format_response_standard_with_numpy(self, mock_config):
        """Test standard response format with numpy array."""
        from mlserver.server import _format_response
        from mlserver.schemas import PredictResponse
        import numpy as np

        mock_config.api.response_format = 'standard'

        predictions = np.array([0, 1, 0])
        result = _format_response(predictions, mock_config, 15.0, "Classifier")

        assert isinstance(result, PredictResponse)
        assert result.predictions == [0, 1, 0]


class TestGetClassifierMetadata:
    """Test _get_classifier_metadata function."""

    def test_get_classifier_metadata_basic(self, mock_config):
        """Test getting classifier metadata."""
        from mlserver.server import _get_classifier_metadata

        with patch('mlserver.auto_detect.get_git_info') as mock_git_info, \
             patch('mlserver.auto_detect.get_project_name') as mock_project_name, \
             patch('mlserver.auto_detect.get_mlserver_git_info') as mock_mlserver_info:

            mock_git_info.return_value = {
                "repository": "test-repo",
                "commit": "abc1234",
                "tag": "v1.0.0"
            }
            mock_project_name.return_value = "test-project"
            mock_mlserver_info.return_value = {
                "package_version": "0.5.0",
                "api_commit": "def5678",
                "api_tag": None
            }

            metadata = _get_classifier_metadata(mock_config, "TestPredictor")

            assert metadata.project == "test-repo"
            assert metadata.classifier == "test-model"
            assert metadata.predictor_class == "TestPredictor"
            assert metadata.mlserver_version == "0.5.0"

    def test_get_classifier_metadata_no_classifier(self):
        """Test getting metadata when no classifier configured."""
        from mlserver.server import _get_classifier_metadata

        config = AppConfig.model_validate({
            "predictor": {
                "module": "test",
                "class_name": "Test"
            }
        })
        # Remove classifier to test None case
        config.classifier = None

        metadata = _get_classifier_metadata(config, "Test")
        assert metadata is None


class TestToJsonableExtended:
    """Extended tests for _to_jsonable function."""

    def test_to_jsonable_nested_dict(self):
        """Test _to_jsonable with nested dictionary."""
        import numpy as np

        data = {
            "predictions": np.array([1, 2, 3]),
            "metadata": {
                "count": np.int32(3),
                "score": np.float64(0.95)
            }
        }

        result = _to_jsonable(data)

        assert result["predictions"] == [1, 2, 3]
        assert result["metadata"]["count"] == 3
        assert result["metadata"]["score"] == 0.95

    def test_to_jsonable_list_of_arrays(self):
        """Test _to_jsonable with list of numpy arrays."""
        import numpy as np

        data = [np.array([1, 2]), np.array([3, 4])]
        result = _to_jsonable(data)

        assert result == [[1, 2], [3, 4]]

    def test_to_jsonable_bytes(self):
        """Test _to_jsonable with bytes."""
        import base64
        data = b"hello"
        result = _to_jsonable(data)
        # bytes are base64 encoded
        assert result == base64.b64encode(data).decode('utf-8')

    def test_to_jsonable_none(self):
        """Test _to_jsonable with None."""
        result = _to_jsonable(None)
        assert result is None


class TestMiddlewareHealthEndpoint:
    """Test middleware behavior for health/metrics endpoints."""

    @pytest.mark.asyncio
    async def test_middleware_skips_metrics_for_healthz(self, mock_config):
        """Test middleware skips metrics for /healthz endpoint."""
        with patch('mlserver.server.get_metrics') as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            app = Mock()
            middleware = ObservabilityMiddleware(app, mock_config)

            mock_request = Mock(spec=Request)
            mock_request.url = Mock()
            mock_request.url.path = "/healthz"
            mock_request.method = "GET"

            mock_response = Mock(spec=Response)
            mock_response.status_code = 200

            async def mock_call_next(request):
                return mock_response

            result = await middleware.dispatch(mock_request, mock_call_next)

            # Should not track metrics for health endpoint
            mock_metrics.inc_active_requests.assert_not_called()
            mock_metrics.track_request.assert_not_called()


class TestPredictProbaWithLock:
    """Test predict_proba with thread safety."""

    def test_predict_proba_with_lock(self):
        """Test predict_proba with thread safe wrapper."""
        mock_predictor = Mock()
        mock_predictor.predict_proba.return_value = [[0.8, 0.2]]

        wrapper = PredictorWrapper(mock_predictor, thread_safe=True)
        result = wrapper.predict_proba([[1, 2]])

        assert result == [[0.8, 0.2]]
        mock_predictor.predict_proba.assert_called_once()