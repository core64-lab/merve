from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .adapters import to_ndarray
from .auto_detect import (
    get_deployed_timestamp,
    get_simplified_info_response,
)
from .concurrency_limiter import PredictionLimiter, PredictionSemaphore
from .config import AppConfig
from .errors import PredictorError
from .logging_conf import log_prediction, log_request, log_response, set_correlation_id
from .metrics import count_samples, get_metrics, init_metrics
from .predictor_loader import load_predictor
from .schemas import (
    ClassifierMetadataResponse,
    CustomPredictResponse,
    HealthResponse,
    PredictResponse,
    ProbaResponse,
    predict_request_openapi_examples,
)

logger = logging.getLogger(__name__)

# Once-per-process guard for the legacy {"payload": {...}} wrapper warning (RFC 0001 D10)
_payload_wrapper_warned = False


def _resolve_request_payload(body: Any) -> dict:
    """Resolve the prediction payload from either accepted request shape (RFC 0001 D10).

    Canonical shape: input keys ('records', 'instances', 'ndarray', 'inputs',
    'features') at the top level of the body. Legacy shape: the same data
    wrapped as {"payload": {...}} — deprecated, logs one warning per process,
    removal targeted for 1.0. When both are present the wrapper wins.
    """
    global _payload_wrapper_warned

    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object",
        )

    wrapper = body.get("payload")
    if isinstance(wrapper, dict):
        if not _payload_wrapper_warned:
            _payload_wrapper_warned = True
            logger.warning(
                'The {"payload": {...}} request wrapper is deprecated; send '
                "'records'/'instances'/'ndarray'/'inputs'/'features' as top-level "
                "keys instead. Wrapper removal is targeted for 1.0 (RFC 0001 D10)."
            )
        return wrapper

    if "payload" in body:
        raise HTTPException(
            status_code=400,
            detail=(
                "'payload' must be a JSON object when using the legacy wrapper "
                f"(got {type(wrapper).__name__}). Prefer top-level keys: "
                "'records', 'instances', 'ndarray', 'inputs' or 'features'."
            ),
        )

    return body


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: AppConfig):
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next):
        # Skip metrics for health and metrics endpoints to reduce overhead
        skip_metrics = request.url.path in ["/healthz", self.config.observability.metrics_endpoint]

        # Look up the metrics collector per-request (cheap global read).
        # It is initialized in the app lifespan, which runs AFTER middleware
        # construction - caching it in __init__ would always capture None.
        metrics = get_metrics() if self.config.observability.metrics else None

        # Set correlation ID if enabled
        correlation_id = None
        if self.config.observability.correlation_ids:
            correlation_id = set_correlation_id()

        start_time = time.perf_counter()

        # Track active requests only if metrics enabled and not skipped
        if metrics and not skip_metrics:
            metrics.inc_active_requests()

        # Log request start
        if self.config.observability.structured_logging:
            log_request(method=request.method, path=request.url.path, correlation_id=correlation_id)

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time

            # Track metrics only if enabled and not skipped
            if metrics and not skip_metrics:
                metrics.track_request(request, response, duration)
                metrics.dec_active_requests()

            # Log response
            if self.config.observability.structured_logging:
                log_response(status_code=response.status_code, duration_ms=duration * 1000)

            return response

        except Exception as e:
            duration = time.perf_counter() - start_time

            if metrics and not skip_metrics:
                metrics.dec_active_requests()

            if self.config.observability.structured_logging:
                log_response(status_code=500, duration_ms=duration * 1000, error=str(e))
            raise


class PredictorWrapper:
    def __init__(self, predictor: Any, thread_safe: bool = False):
        self._predictor = predictor
        self._lock = threading.Lock() if thread_safe else None

    def predict(self, X):
        if self._lock:
            with self._lock:
                return self._predictor.predict(X)
        return self._predictor.predict(X)

    def predict_proba(self, X):
        if not hasattr(self._predictor, "predict_proba"):
            raise AttributeError("predictor has no predict_proba")
        if self._lock:
            with self._lock:
                return self._predictor.predict_proba(X)
        return self._predictor.predict_proba(X)

    @property
    def name(self) -> str:
        return type(self._predictor).__name__

    def close(self):
        if hasattr(self._predictor, "close"):
            try:
                self._predictor.close()
            except Exception:
                pass


def _prepare_input_data(payload: dict, config: AppConfig):
    """Parse and prepare input data for prediction.

    ``payload`` is the resolved input data (see _resolve_request_payload).
    Includes optional feature schema validation when feature_order is configured,
    providing clearer error messages for missing/extra features.
    """
    # Debug logging for request details
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Incoming request payload type: {type(payload)}")
        logger.debug(f"Incoming request payload: {payload}")
        logger.debug(f"Adapter configuration: {config.api.adapter}")
        logger.debug(f"Feature order: {config.api.feature_order}")

    try:
        # Resolve feature_order from file if needed
        base_path = (
            Path(config.project_path_internal) if config.project_path_internal else Path.cwd()
        )
        feature_order = config.api.get_resolved_feature_order(base_path=base_path)

        # Optional: Validate features before parsing (provides better error messages)
        # Only applies to records adapter with configured feature_order
        if feature_order and config.api.adapter in ("records", "auto"):
            _validate_input_features(payload, feature_order, logger)

        return to_ndarray(
            payload,
            adapter=config.api.adapter,
            feature_order=feature_order,
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log full traceback in DEBUG mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Input parsing failed with detailed traceback:")
        else:
            logger.error(f"Input parsing error: {e}")

        # In DEBUG mode, include more detail in the response
        if logger.isEnabledFor(logging.DEBUG):
            raise HTTPException(
                status_code=400,
                detail=f"Input parsing failed: {str(e)}. Check server logs for full traceback.",
            ) from e
        else:
            raise HTTPException(
                status_code=400, detail="Input parsing failed. Please check your input format."
            ) from e


def _validate_input_features(payload: dict, feature_order: list, logger) -> None:
    """Validate input features match expected schema.

    Provides clear error messages when features are missing or unexpected.
    """
    from .validation import FeatureSchemaValidator

    # Extract records from various payload formats
    records = None
    if isinstance(payload, dict):
        records = payload.get("records") or payload.get("instances")
        if records is None and "features" in payload:
            # Single record via 'features' key
            records = [payload["features"]]
        elif records is None and not any(k in payload for k in ["ndarray", "inputs"]):
            # Treat as single record if no known keys
            records = [payload]

    if not records or not isinstance(records, list):
        # Not a records format, skip validation
        return

    # Validate records
    validator = FeatureSchemaValidator(feature_order)
    all_valid, results = validator.validate_records(records)

    if not all_valid:
        # Build helpful error message
        invalid_results = [r for r in results if not r.valid]

        # Collect all missing features across records
        all_missing = set()
        for r in invalid_results:
            all_missing.update(r.missing_features)

        if len(invalid_results) == 1:
            error_msg = invalid_results[0].to_error_message()
        else:
            error_msg = validator.get_validation_summary(results)

        # Log details
        logger.warning(f"Feature validation failed: {error_msg}")

        # Provide suggestion
        suggestion = f"Expected features: {', '.join(feature_order[:5])}"
        if len(feature_order) > 5:
            suggestion += f" ... and {len(feature_order) - 5} more"

        raise HTTPException(
            status_code=400, detail=f"Feature validation failed: {error_msg}. {suggestion}"
        )


def _track_prediction_metrics(
    endpoint_path: str,
    duration_seconds: float,
    input_samples: int,
    output_samples: int,
    model_name: str,
    config: AppConfig,
) -> None:
    """Track prediction metrics and logging.

    Args:
        endpoint_path: API endpoint path
        duration_seconds: Prediction duration in seconds
        input_samples: Number of input samples (batch size)
        output_samples: Number of output samples (predictions)
        model_name: Name of the predictor/model
        config: Application configuration
    """
    # Track prediction metrics only if enabled
    if config.observability.metrics:
        metrics = get_metrics()
        if metrics:
            metrics.track_prediction(
                endpoint_path,
                duration_seconds,
                input_samples=input_samples,
                output_samples=output_samples,
            )

    # Log prediction
    if config.observability.structured_logging:
        log_prediction(
            model_name=model_name,
            duration_ms=duration_seconds * 1000,
            sample_count=output_samples,
            batch_size=input_samples,
        )


def _log_payload(endpoint_path: str, request_payload: Any, response: Any) -> None:
    """Log request/response payloads for a prediction.

    Only called when observability.log_payloads is enabled. Payloads may
    contain sensitive data, so this is opt-in and disabled by default.
    Uses the standard logging path so correlation IDs are included by the
    structured formatter.

    Args:
        endpoint_path: API endpoint path
        request_payload: Raw request payload dict
        response: Response object (pydantic model or JSON-safe structure)
    """
    import logging

    from pydantic import BaseModel

    if isinstance(response, BaseModel):
        response_data = response.model_dump()
    else:
        response_data = _to_jsonable(response)

    logging.getLogger("mlserver.payload").info(
        "Prediction payload",
        extra={
            "event": "payload",
            "endpoint": endpoint_path,
            "payload": _to_jsonable(request_payload),
            "response": response_data,
        },
    )


def _create_predict_handler(
    app: FastAPI,
    config: AppConfig,
    endpoint_path: str,
    prediction_limiter: Optional[PredictionSemaphore] = None,
):
    """Create a predict endpoint handler with concurrency control."""

    def predict(body: Optional[dict[str, Any]] = None):
        payload = _resolve_request_payload(body if body is not None else {})
        # Use prediction limiter if configured
        if prediction_limiter:
            with PredictionLimiter(
                prediction_limiter, retry_after_seconds=config.api.retry_after_seconds
            ):
                return _execute_prediction(app, config, endpoint_path, payload)
        else:
            return _execute_prediction(app, config, endpoint_path, payload)

    return predict


def _get_classifier_metadata(
    config: AppConfig, predictor_class_name: str = None, config_file_name: str = None
) -> Optional[ClassifierMetadataResponse]:
    """Get simplified classifier metadata for response."""
    if not config.classifier:
        return None

    # Use auto-detection for git info
    import os

    from .auto_detect import get_git_info, get_mlserver_git_info, get_project_name

    git_data = get_git_info(config.project_path or ".")
    mlserver_info = get_mlserver_git_info()

    # Get config_file from environment if not provided (for containerized deployments)
    if not config_file_name:
        config_file_name = os.environ.get("MLSERVER_CONFIG_FILE")

    metadata = ClassifierMetadataResponse(
        project=git_data.get("repository") or get_project_name(config.project_path or "."),
        classifier=config.classifier.get("name", "unknown"),
        predictor_class=predictor_class_name,
        predictor_module=config.predictor.module if config.predictor else None,
        config_file=config_file_name,
        git_commit=git_data.get("commit"),
        git_tag=git_data.get("tag"),
        deployed_at=None,  # Will be set at runtime
        mlserver_version=mlserver_info.get("package_version", "unknown"),
        mlserver_api_commit=mlserver_info.get("api_commit"),
        mlserver_api_tag=mlserver_info.get("api_tag"),
    )

    # Note: deployed_at will be added at runtime from app.state.deployed_at

    return metadata


def _format_response(
    predictions, config: AppConfig, timing_ms: float, model_name: str, metadata=None
):
    """Format response based on configuration.

    Args:
        predictions: Raw predictions from the predictor
        config: Application configuration
        timing_ms: Time taken for prediction in milliseconds
        model_name: Name of the predictor/model
        metadata: Optional classifier metadata

    Returns:
        Formatted response based on response_format configuration
    """
    response_format = config.api.response_format

    # Passthrough format - return exactly what predictor returned
    if response_format == "passthrough":
        return predictions

    # Convert to JSON-serializable format
    json_safe = _to_jsonable(predictions)

    # Custom format - flexible structure with result field
    if response_format == "custom":
        # For dictionaries, include the whole structure
        if isinstance(json_safe, dict):
            # Extract predictions if configured or if the dict contains a 'predictions' key
            predictions_list = None
            if config.api.extract_values:
                predictions_list = list(json_safe.values())
            elif "predictions" in json_safe:
                # Use the predictions field if it exists in the response
                predictions_list = (
                    json_safe["predictions"]
                    if isinstance(json_safe["predictions"], list)
                    else [json_safe["predictions"]]
                )

            response = CustomPredictResponse(
                result=json_safe,
                predictions=predictions_list,
                time_ms=timing_ms,
                predictor_class=model_name,
                metadata=metadata,
            )
            return response
        else:
            # For non-dict responses, wrap in result field
            return CustomPredictResponse(
                result=json_safe,
                predictions=json_safe if isinstance(json_safe, list) else [json_safe],
                time_ms=timing_ms,
                predictor_class=model_name,
                metadata=metadata,
            )

    # Standard format (default) - backward compatible
    # Special handling for dict responses
    if isinstance(json_safe, dict):
        # In standard mode with dict, we need to handle it specially
        # to maintain backward compatibility
        if config.api.extract_values:
            # Extract values for predictions field
            predictions_list = list(json_safe.values())
        else:
            # Use the dict as a single prediction item
            predictions_list = [json_safe]

        return PredictResponse(
            predictions=predictions_list,
            time_ms=timing_ms,
            predictor_class=model_name,
            metadata=metadata,
        )

    # Standard format for list/array responses
    if isinstance(json_safe, list):
        return PredictResponse(
            predictions=json_safe, time_ms=timing_ms, predictor_class=model_name, metadata=metadata
        )

    # Single value response - wrap in list for consistency
    return PredictResponse(
        predictions=[json_safe], time_ms=timing_ms, predictor_class=model_name, metadata=metadata
    )


def _execute_prediction(app: FastAPI, config: AppConfig, endpoint_path: str, payload: dict):
    """Execute the actual prediction."""
    start_time = time.perf_counter()

    X = _prepare_input_data(payload, config)

    # Count input samples before prediction
    input_sample_count = count_samples(X)

    try:
        predictions = app.state.predictor.predict(X)
    except Exception as e:
        # Log full error internally, return sanitized message to client
        logging.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please contact support if the issue persists.",
        ) from e

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Count output samples using the improved counter
    output_sample_count = count_samples(predictions)

    _track_prediction_metrics(
        endpoint_path,
        duration_ms / 1000,
        input_samples=input_sample_count,
        output_samples=output_sample_count,
        model_name=app.state.predictor.name,
        config=config,
    )

    # Include cached metadata if available
    metadata = getattr(app.state, "metadata", None)

    # Add deployed_at timestamp to metadata
    if metadata:
        deployed_at = getattr(app.state, "deployed_at", None)
        if deployed_at:
            metadata.deployed_at = deployed_at

    # Use the new formatting function
    response = _format_response(
        predictions, config, duration_ms, app.state.predictor.name, metadata
    )

    # Log request/response payloads if enabled (privacy-sensitive, opt-in)
    if config.observability.log_payloads:
        _log_payload(endpoint_path, payload, response)

    return response


def _create_predict_proba_handler(
    app: FastAPI,
    config: AppConfig,
    endpoint_path: str,
    prediction_limiter: Optional[PredictionSemaphore] = None,
):
    """Create a predict_proba endpoint handler with concurrency control."""

    def predict_proba(body: Optional[dict[str, Any]] = None):
        payload = _resolve_request_payload(body if body is not None else {})
        # Use prediction limiter if configured
        if prediction_limiter:
            with PredictionLimiter(
                prediction_limiter, retry_after_seconds=config.api.retry_after_seconds
            ):
                return _execute_predict_proba(app, config, endpoint_path, payload)
        else:
            return _execute_predict_proba(app, config, endpoint_path, payload)

    return predict_proba


def _execute_predict_proba(app: FastAPI, config: AppConfig, endpoint_path: str, payload: dict):
    """Execute the actual probability prediction."""
    start_time = time.perf_counter()

    X = _prepare_input_data(payload, config)

    # Count input samples before prediction
    input_sample_count = count_samples(X)

    try:
        probabilities = app.state.predictor.predict_proba(X)
    except AttributeError as e:
        raise HTTPException(
            status_code=501, detail="Probability prediction not available for this model."
        ) from e
    except Exception as e:
        # Log full error internally, return sanitized message to client
        logging.error(f"Predict_proba error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Probability prediction failed. Please contact support if the issue persists.",
        ) from e

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Count output samples
    output_sample_count = count_samples(probabilities)

    _track_prediction_metrics(
        endpoint_path,
        duration_ms / 1000,
        input_samples=input_sample_count,
        output_samples=output_sample_count,
        model_name=app.state.predictor.name,
        config=config,
    )

    # Include cached metadata if available
    metadata = getattr(app.state, "metadata", None)

    # Add deployed_at timestamp to metadata
    if metadata:
        deployed_at = getattr(app.state, "deployed_at", None)
        if deployed_at:
            metadata.deployed_at = deployed_at

    # Create ProbaResponse with metadata
    response = ProbaResponse(
        probabilities=_tolist2d(probabilities),
        time_ms=duration_ms,
        classes=None,  # Could be populated if predictor provides class names
        predictor_class=app.state.predictor.name,
        metadata=metadata,
    )

    # Log request/response payloads if enabled (privacy-sensitive, opt-in)
    if config.observability.log_payloads:
        _log_payload(endpoint_path, payload, response)

    return response


def _prediction_openapi_extra() -> dict[str, Any]:
    """OpenAPI operation overrides for the prediction endpoints (RFC 0001 D10).

    The handlers accept a raw ``dict`` body (top-level and legacy wrapped
    shapes), so the request schema and examples are injected explicitly —
    top-level form first, the deprecated wrapper last.
    """
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object", "title": "PredictionInput"},
                    "examples": predict_request_openapi_examples(),
                }
            },
        }
    }


def _register_endpoint(
    app: FastAPI,
    endpoint_name: str,
    handler,
    base_path: str,
    response_model=None,
    openapi_extra: Optional[dict] = None,
) -> None:
    """Register a versioned endpoint."""
    # Only register versioned endpoint
    if base_path:
        path = f"{base_path}/{endpoint_name}"
    else:
        # Fallback to root level if no base path (should not happen in modern config)
        path = f"/{endpoint_name}"
    app.post(path, response_model=response_model, openapi_extra=openapi_extra)(handler)


def _register_prediction_endpoints(
    app: FastAPI, config: AppConfig, prediction_limiter: Optional[PredictionSemaphore] = None
) -> None:
    """Register versioned prediction endpoints with optional concurrency control."""
    base_path = config.get_base_path()

    # Determine response model based on configuration
    response_format = config.api.response_format
    if response_format == "custom":
        response_model = CustomPredictResponse
    elif response_format == "passthrough":
        response_model = None  # No validation for passthrough
    else:
        response_model = PredictResponse  # Default standard format

    openapi_extra = _prediction_openapi_extra()

    # Register predict endpoint
    if config.is_endpoint_enabled("predict"):
        endpoint_path = f"{base_path}/predict" if base_path else "/predict"
        predict_handler = _create_predict_handler(app, config, endpoint_path, prediction_limiter)
        _register_endpoint(
            app, "predict", predict_handler, base_path, response_model, openapi_extra
        )

    # Note: batch_predict endpoint removed - /predict already handles batches naturally

    # Register predict_proba endpoint
    if config.is_endpoint_enabled("predict_proba"):
        endpoint_path = f"{base_path}/predict_proba" if base_path else "/predict_proba"
        predict_proba_handler = _create_predict_proba_handler(
            app, config, endpoint_path, prediction_limiter
        )
        # _execute_predict_proba always returns a ProbaResponse (the response
        # format setting shapes /predict only), so the schema is unconditional
        proba_model = None if response_format == "passthrough" else ProbaResponse
        _register_endpoint(
            app, "predict_proba", predict_proba_handler, base_path, proba_model, openapi_extra
        )


def create_app(config: AppConfig, config_file_name: str = None) -> FastAPI:
    predictor_wrapper: Optional[PredictorWrapper] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal predictor_wrapper
        startup_logger = logging.getLogger(__name__)

        # Load predictor once at startup
        # Pass config_dir for intelligent module resolution
        config_dir = config.project_path if config.project_path else None
        predictor = load_predictor(
            config.predictor.module,
            config.predictor.class_name,
            config.predictor.init_kwargs,
            config_dir=config_dir,
        )

        # Optional load() hook (RFC 0001 D13): called exactly once after
        # construction, before the predictor is marked ready. A failure here
        # aborts startup so the pod never reports ready with a broken model.
        load_hook = getattr(predictor, "load", None)
        if callable(load_hook):
            startup_logger.info(f"Calling {type(predictor).__name__}.load() before serving...")
            try:
                load_hook()
            except Exception as e:
                raise PredictorError(
                    message=f"Predictor load() failed for {type(predictor).__name__}: {e}",
                    suggestion=(
                        "load() runs once at startup after construction; check the "
                        "artifact paths and dependencies it uses. The server did not "
                        "start and the predictor was not marked ready."
                    ),
                ) from e

        predictor_wrapper = PredictorWrapper(predictor, thread_safe=config.api.thread_safe_predict)
        app.state.predictor = predictor_wrapper

        # Cache metadata at startup
        predictor_class_name = config.predictor.class_name if config.predictor else None
        app.state.metadata = _get_classifier_metadata(
            config, predictor_class_name, config_file_name
        )

        # Cache deployment timestamp at startup
        app.state.deployed_at = get_deployed_timestamp()

        # Initialize metrics if enabled
        if config.observability.metrics:
            model_version = config.classifier.get("version") if config.classifier else None
            init_metrics(predictor_wrapper.name, model_version=model_version)

        # Model warmup: run a dummy prediction to initialize model internals
        # This reduces latency on the first real prediction request
        if config.api.warmup_on_start:
            startup_logger.info("Warming up model...")
            try:
                warmup_start = time.perf_counter()
                warmup_data = _create_warmup_data(config)
                if warmup_data is not None:
                    # Run warmup prediction (result is discarded)
                    _ = predictor_wrapper.predict(warmup_data)
                    warmup_duration = time.perf_counter() - warmup_start
                    startup_logger.info(f"Model warmup complete in {warmup_duration:.3f}s")
                elif config.api.feature_order is None:
                    startup_logger.info("Model warmup skipped (no feature_order configured)")
                else:
                    startup_logger.warning(
                        "Model warmup skipped (feature_order configured but could not be resolved)"
                    )
            except Exception as e:
                # Warmup failure is non-fatal - log and continue
                startup_logger.warning(f"Model warmup failed (non-fatal): {e}")

        yield
        # Shutdown
        predictor_wrapper.close()

    app = FastAPI(title=config.get_api_title(), lifespan=lifespan)

    # Per-process Prometheus registries do not aggregate across workers (RFC 0001 D14)
    if config.server.workers > 1 and config.observability.metrics:
        logger.warning(
            f"workers={config.server.workers} with Prometheus metrics enabled: each "
            "worker process keeps its own metrics registry, so /metrics scrapes "
            "sample only one worker. Recommended: workers: 1 per container and "
            "scale horizontally (RFC 0001 D14)."
        )

    # Add observability middleware
    app.add_middleware(ObservabilityMiddleware, config=config)

    # CORS
    if config.server.cors:
        c = config.server.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins=c.allow_origins,
            allow_credentials=c.allow_credentials,
            allow_methods=c.allow_methods,
            allow_headers=c.allow_headers,
        )

    # Create prediction limiter if concurrency control is enabled
    prediction_limiter = None
    if config.api.max_concurrent_predictions > 0:
        prediction_limiter = PredictionSemaphore(
            max_concurrent=config.api.max_concurrent_predictions,
            timeout=0,  # Immediate rejection for Kubernetes pod scaling
        )
    # Exposed for introspection/tests (e.g. asserting Retry-After behavior)
    app.state.prediction_limiter = prediction_limiter

    @app.get("/healthz", response_model=HealthResponse)
    def health():
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            # Not ready: the predictor has not finished loading (RFC 0001 D13).
            # Under uvicorn this state is only observable while startup is in
            # flight, but probes and harnesses mounting the app without its
            # lifespan must not read "ok" for a model that cannot serve.
            return JSONResponse(
                status_code=503,
                content=HealthResponse(status="loading", model=None).model_dump(),
            )
        return HealthResponse(status="ok", model=predictor.name)

    @app.get("/info")
    def info():
        """Get simplified classifier information with auto-detected metadata."""
        predictor = getattr(app.state, "predictor", None)
        deployed_at = getattr(app.state, "deployed_at", None)

        # Use the new simplified info response
        info_response = get_simplified_info_response(
            config.model_dump(),
            predictor.name if predictor else "unknown",
            config.project_path or ".",
        )

        # Use the cached deployed_at timestamp from startup
        if deployed_at:
            info_response["deployed_at"] = deployed_at

        # Keep the endpoints section consistent
        info_response["endpoints"] = {
            "predict": "/predict" if config.is_endpoint_enabled("predict") else None,
            "predict_proba": (
                "/predict_proba" if config.is_endpoint_enabled("predict_proba") else None
            ),
            "info": "/info",
            "health": "/healthz",
            "metrics": (
                config.observability.metrics_endpoint if config.observability.metrics else None
            ),
        }

        return info_response

    @app.get("/status")
    def prediction_status():
        """Get current prediction availability status."""
        if prediction_limiter:
            return {
                "prediction_slots_available": prediction_limiter.is_available,
                "active_predictions": prediction_limiter.active_predictions,
                "max_concurrent_predictions": config.api.max_concurrent_predictions,
                "concurrency_control_enabled": True,
            }
        else:
            return {
                "prediction_slots_available": True,
                "active_predictions": 0,
                "max_concurrent_predictions": None,
                "concurrency_control_enabled": False,
            }

    # Add metrics endpoint if enabled
    if config.observability.metrics:
        # Cache metrics for 5 seconds to reduce CPU load during monitoring scrapes
        @lru_cache(maxsize=1)
        def _get_cached_metrics(timestamp_key: int):
            """Cache metrics generation keyed by 5-second intervals."""
            metrics_collector = get_metrics()
            if metrics_collector:
                return Response(
                    content=metrics_collector.generate_metrics(),
                    media_type=metrics_collector.get_content_type(),
                )
            return Response(content="# No metrics available\n", media_type="text/plain")

        @app.get(config.observability.metrics_endpoint)
        def metrics():
            # Cache key updates every 5 seconds
            cache_key = int(time.time() // 5)
            return _get_cached_metrics(cache_key)

    # Register versioned prediction endpoints
    _register_prediction_endpoints(app, config, prediction_limiter)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _create_warmup_data(config: AppConfig) -> Optional[np.ndarray]:
    """Create minimal warmup data for model initialization.

    Generates a single row of zeros based on feature_order configuration.
    This triggers model initialization (e.g., loading weights, JIT compilation)
    without affecting actual predictions.

    Args:
        config: Application configuration with feature_order

    Returns:
        numpy array with shape (1, n_features) or None if no feature_order

    Raises:
        Exception: If feature_order resolution fails (caller treats warmup
            failures as non-fatal and logs them)
    """
    # Try to get feature order from config
    base_path = Path(config.project_path) if config.project_path else None
    feature_order = config.api.get_resolved_feature_order(base_path=base_path)

    if feature_order:
        # Create single row of zeros with correct number of features
        return np.zeros((1, len(feature_order)))

    # If no feature_order, try to infer from predictor if it has n_features attribute
    # This is a best-effort approach for models without explicit feature_order
    return None


def _to_jsonable(x, _depth=0):
    """Convert any Python object to JSON-serializable format.

    Handles pandas, numpy, datetime types, and nested structures with comprehensive
    type coverage to prevent serialization errors in production.

    Args:
        x: Object to convert
        _depth: Current recursion depth (internal use)

    Returns:
        JSON-serializable representation of x

    Raises:
        RecursionError: If recursion depth exceeds maximum limit
    """
    import logging

    _logger = logging.getLogger(__name__)

    # Maximum recursion depth to prevent stack overflow
    _MAX_RECURSION_DEPTH = 50

    # Recursion depth check
    if _depth > _MAX_RECURSION_DEPTH:
        _logger.warning(f"Max recursion depth {_MAX_RECURSION_DEPTH} exceeded in _to_jsonable")
        raise RecursionError(f"Maximum recursion depth exceeded: {_MAX_RECURSION_DEPTH}")

    # Fast path: None
    if x is None:
        return None

    # Fast path: Basic JSON-serializable types
    if isinstance(x, (str, int, float, bool)):
        return x

    # === Pandas temporal types (MUST be before numpy!) ===
    try:
        import pandas as pd

        # pd.Timestamp - Convert to ISO 8601 string
        if isinstance(x, pd.Timestamp):
            return x.isoformat()

        # pd.NaT - Not a Time (pandas null for timestamps)
        # Check type name first to avoid calling pd.isna() on arrays
        if hasattr(x, "__class__") and x.__class__.__name__ == "NaTType":
            return None

        # pd.Timedelta - Convert to seconds
        if isinstance(x, pd.Timedelta):
            return x.total_seconds()

    except ImportError:
        pass

    # === Pandas collection types ===
    try:
        import pandas as pd

        # DataFrame - Convert to list of records
        if isinstance(x, pd.DataFrame):
            # Recursively convert to handle nested types
            records = x.to_dict("records")
            return _to_jsonable(records, _depth + 1)

        # Series - Convert to list
        if isinstance(x, pd.Series):
            # Recursively convert to handle nested types
            return _to_jsonable(x.tolist(), _depth + 1)

        # Index - Convert to list
        if isinstance(x, pd.Index):
            return _to_jsonable(x.tolist(), _depth + 1)

    except ImportError:
        pass

    # === Python datetime types ===
    try:
        from datetime import date, datetime, time, timedelta

        # datetime.datetime - ISO 8601 with timezone
        if isinstance(x, datetime):
            return x.isoformat()

        # datetime.date - YYYY-MM-DD
        if isinstance(x, date):
            return x.isoformat()

        # datetime.time - HH:MM:SS
        if isinstance(x, time):
            return x.isoformat()

        # datetime.timedelta - Total seconds
        if isinstance(x, timedelta):
            return x.total_seconds()

    except ImportError:
        pass

    # === NumPy types ===
    try:
        import numpy as np

        # np.datetime64 - MUST check before np.generic!
        if isinstance(x, np.datetime64):
            # Convert to ISO string via pandas for consistency
            import pandas as pd

            return pd.Timestamp(x).isoformat()

        # np.timedelta64 - Convert to seconds
        if isinstance(x, np.timedelta64):
            # Convert to pandas Timedelta for consistent handling
            import pandas as pd

            return pd.Timedelta(x).total_seconds()

        # np.ndarray - Convert to nested list
        if isinstance(x, np.ndarray):
            # Recursively convert to handle nested types
            return _to_jsonable(x.tolist(), _depth + 1)

        # np.generic - Catches all numpy scalar types (int64, float64, bool_, etc.)
        if isinstance(x, np.generic):
            return x.item()

    except ImportError:
        pass

    # === Python collection types (recursive) ===

    # Dictionary - Recursively convert keys and values
    if isinstance(x, dict):
        return {_to_jsonable(k, _depth + 1): _to_jsonable(v, _depth + 1) for k, v in x.items()}

    # List/tuple - Recursively convert items
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(item, _depth + 1) for item in x]

    # Set/frozenset - Convert to sorted list
    if isinstance(x, (set, frozenset)):
        try:
            return sorted(_to_jsonable(list(x), _depth + 1))
        except TypeError:
            # If items aren't sortable, return unsorted
            return _to_jsonable(list(x), _depth + 1)

    # === Other common types ===

    # decimal.Decimal - Convert to float
    try:
        from decimal import Decimal

        if isinstance(x, Decimal):
            return float(x)
    except ImportError:
        pass

    # bytes - Convert to base64 string
    if isinstance(x, bytes):
        import base64

        return base64.b64encode(x).decode("ascii")

    # === Fallback ===
    # If we reach here, the type isn't explicitly handled
    # Log a warning and attempt str conversion
    type_name = type(x).__name__
    _logger.warning(
        f"Unhandled type in _to_jsonable: {type_name}. "
        f"Attempting str() conversion. Value: {str(x)[:100]}"
    )

    try:
        # Attempt to convert to string as last resort
        return str(x)
    except Exception as e:
        # If even str() fails, log error and return placeholder
        _logger.error(
            f"Failed to convert {type_name} to JSON-serializable format: {e}. "
            f"Returning placeholder string."
        )
        return f"<Unserializable: {type_name}>"


def _tolist2d(arr):
    try:
        import numpy as _np

        if isinstance(arr, _np.ndarray):
            return arr.tolist()
    except Exception:
        pass
    # assume already nested list
    return arr


# Factory function for uvicorn when using workers
def app() -> FastAPI:
    """Factory function to create the FastAPI app for uvicorn workers."""
    import os

    import yaml

    from .config import AppConfig
    from .multi_classifier import (
        detect_multi_classifier_config,
        extract_single_classifier_config,
        load_multi_classifier_config,
    )

    # Try to find config file
    config_paths = [os.environ.get("MLSERVER_CONFIG_PATH"), "mlserver.yaml"]

    config_file = None
    for path in config_paths:
        if path and Path(path).exists():
            config_file = Path(path)
            break

    if not config_file:
        raise RuntimeError(
            "No configuration file found. Please specify mlserver.yaml or set MLSERVER_CONFIG_PATH"
        )

    # Check for multi-classifier config
    if detect_multi_classifier_config(str(config_file)):
        mc = load_multi_classifier_config(str(config_file))

        if not mc.classifiers:
            raise RuntimeError(f"No classifiers defined in multi-classifier config: {config_file}")

        # Resolve classifier name: MLSERVER_CLASSIFIER env var (deploy-time
        # selection on commit images, RFC 0001 D4), then default_classifier,
        # then first. An invalid env value is a hard error — a typo must not
        # silently serve a different model.
        classifier_name = os.environ.get("MLSERVER_CLASSIFIER")
        if classifier_name:
            if classifier_name not in mc.classifiers:
                raise RuntimeError(
                    f"MLSERVER_CLASSIFIER={classifier_name!r} does not match any "
                    f"classifier in {config_file}. "
                    f"Available: {', '.join(sorted(mc.classifiers))}"
                )
        elif mc.default_classifier and mc.default_classifier in mc.classifiers:
            classifier_name = mc.default_classifier
        else:
            classifier_name = next(iter(mc.classifiers))

        cfg = extract_single_classifier_config(mc, classifier_name)
    else:
        # Single classifier config
        with open(config_file) as f:
            raw = yaml.safe_load(f)
        cfg = AppConfig.model_validate(raw)

    cfg.set_project_path(str(config_file.parent))

    return create_app(cfg, config_file_name=config_file.name)
