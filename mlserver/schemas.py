
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model: Optional[str] = None


class PredictRequest(BaseModel):
    """Request payload for predictions.

    Preferred shape (RFC 0001 D10): send the input data keys at the TOP LEVEL
    of the request body. Which key to use depends on the adapter:
    - records adapter:  {"records": [{"feature1": v1, "feature2": v2}, ...]}
                        (alias: "instances"; single record: {"features": {...}})
    - ndarray adapter:  {"ndarray": [[v1, v2], [v3, v4], ...]}
                        (alias: "inputs")

    Legacy shape (deprecated, removal targeted for 1.0): the same data wrapped
    in a "payload" object, e.g. {"payload": {"records": [...]}}. Both shapes
    behave identically; when both are present the wrapper wins. Using the
    wrapper logs a deprecation warning once per server process.
    """
    # Legacy wrapper field - prediction endpoints parse the raw body, so top-level
    # keys need no schema field here; this model documents the wrapped form.
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "DEPRECATED wrapper around the prediction input data. Prefer sending "
            "'records'/'instances'/'ndarray'/'inputs'/'features' as top-level keys."
        )
    )

    model_config = ConfigDict(
        extra="allow",  # top-level shapes ('records', 'ndarray', ...) are valid bodies
        json_schema_extra={
            "examples": [
                {
                    "description": "Records format, top-level (for adapter='records')",
                    "value": {
                        "records": [
                            {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8},
                            {"feature1": 2.1, "feature2": 1.7, "feature3": 1.2}
                        ]
                    }
                },
                {
                    "description": "Ndarray format, top-level (for adapter='ndarray')",
                    "value": {
                        "ndarray": [
                            [1.5, 2.3, 0.8],
                            [2.1, 1.7, 1.2]
                        ]
                    }
                },
                {
                    "description": "Single record prediction, top-level",
                    "value": {
                        "features": {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8}
                    }
                },
                {
                    "description": "Legacy wrapped format (deprecated, removal targeted for 1.0)",
                    "value": {
                        "payload": {
                            "records": [
                                {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8}
                            ]
                        }
                    }
                }
            ]
        }
    )


# BatchPredictRequest removed - /predict already handles both single and batch predictions
# Just use PredictRequest with multiple records in the payload


class ClassifierMetadataResponse(BaseModel):
    """Simplified metadata included in responses."""
    project: str = Field(description="Auto-detected project/repository name")
    classifier: str = Field(description="Classifier name")
    predictor_class: Optional[str] = Field(None, description="Predictor class name")
    predictor_module: Optional[str] = Field(None, description="Module file containing predictor")
    config_file: Optional[str] = Field(None, description="Configuration file used")
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    git_tag: Optional[str] = Field(None, description="Git tag if on tagged commit")
    deployed_at: Optional[str] = Field(None, description="Deployment timestamp")
    mlserver_version: Optional[str] = Field(None, description="MLServer package version")
    mlserver_api_commit: Optional[str] = Field(None, description="MLServer API git commit")
    mlserver_api_tag: Optional[str] = Field(None, description="MLServer API git tag")


class PredictResponse(BaseModel):
    """Response from prediction endpoints."""
    predictions: list[Any] = Field(description="Model predictions for each input record")
    time_ms: float = Field(description="Time taken for prediction in milliseconds")
    predictor_class: Optional[str] = Field(None, description="Name of the predictor class used")
    metadata: Optional[ClassifierMetadataResponse] = Field(
        None, description="Comprehensive classifier metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predictions": [0, 1, 0],
                    "time_ms": 12.5,
                    "predictor_class": "CatBoostPredictor",
                    "metadata": {
                        "project": "mlserver-repo",
                        "classifier": "catboost-survival",
                        "predictor_class": "CatBoostPredictor",
                        "git_commit": "abc1234",
                        "mlserver_version": "0.3.0"
                    }
                },
                {
                    "predictions": ["class_a", "class_b", "class_a"],
                    "time_ms": 8.3,
                    "predictor_class": "TextClassifierPredictor"
                }
            ]
        }
    )


class ProbaResponse(BaseModel):
    """Response from predict_proba endpoints."""
    probabilities: list[list[float]] = Field(
        description="Class probabilities for each input record"
    )
    time_ms: float = Field(description="Time taken for prediction in milliseconds")
    classes: Optional[list[str]] = Field(
        None, description="Class labels corresponding to probability columns"
    )
    predictor_class: Optional[str] = Field(
        None, description="Name of the predictor class used"
    )
    metadata: Optional[ClassifierMetadataResponse] = Field(
        None, description="Comprehensive classifier metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "probabilities": [
                        [0.75, 0.25],
                        [0.10, 0.90],
                        [0.60, 0.40]
                    ],
                    "time_ms": 15.2,
                    "classes": ["negative", "positive"],
                    "predictor_class": "CatBoostPredictor"
                }
            ]
        }
    )


class CustomPredictResponse(BaseModel):
    """Flexible response supporting arbitrary structures for custom predictors."""
    result: Any = Field(description="Prediction result (any JSON-serializable structure)")
    predictions: Optional[list[Any]] = Field(
        None,
        description="Optional extracted predictions for compatibility"
    )
    time_ms: float = Field(description="Time taken for prediction in milliseconds")
    predictor_class: Optional[str] = Field(None, description="Name of the predictor class used")
    metadata: Optional[ClassifierMetadataResponse] = Field(
        None,
        description="Classifier version and metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "result": {
                        "a": [1, 2, 3, 4, 5],
                        "b": {
                            "c": [1, 2, 3],
                            "d": [4, 5, 6]
                        }
                    },
                    "time_ms": 16.4,
                    "predictor_class": "CustomPredictor"
                },
                {
                    "result": {
                        "predictions": [0, 1, 0],
                        "confidence": [0.95, 0.87, 0.92],
                        "features_used": ["feature1", "feature2", "feature3"]
                    },
                    "time_ms": 12.5,
                    "predictor_class": "AdvancedClassifier",
                    "metadata": {
                        "project": "mlserver-repo",
                        "classifier": "advanced-classifier",
                        "predictor_class": "AdvancedClassifier",
                        "git_commit": "abc1234",
                        "mlserver_version": "0.3.0"
                    }
                }
            ]
        }
    )
