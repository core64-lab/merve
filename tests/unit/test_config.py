import logging

import pytest
from pydantic import ValidationError

from mlserver import defaults
from mlserver.config import (
    ApiConfig,
    AppConfig,
    BuildConfig,
    CORSConfig,
    ObservabilityConfig,
    PredictorConfig,
    ServerConfig,
)


class TestServerConfig:
    """Test ServerConfig validation and defaults"""

    def test_default_values(self):
        config = ServerConfig()
        assert config.title == "ML Server"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "INFO"
        assert config.workers == 1
        assert config.cors is None

    def test_custom_values(self):
        config = ServerConfig(
            title="Custom Server", host="127.0.0.1", port=9000, log_level="DEBUG", workers=4
        )
        assert config.title == "Custom Server"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.log_level == "DEBUG"
        assert config.workers == 4

    def test_cors_config(self):
        cors_config = CORSConfig(allow_origins=["http://localhost:3000"], allow_credentials=True)
        config = ServerConfig(cors=cors_config)
        assert config.cors.allow_origins == ["http://localhost:3000"]
        assert config.cors.allow_credentials is True

    def test_invalid_port(self):
        with pytest.raises(ValidationError):
            ServerConfig(port=-1)

    def test_invalid_workers(self):
        with pytest.raises(ValidationError):
            ServerConfig(workers=0)


class TestCORSConfig:
    """Test CORS configuration"""

    def test_default_values(self):
        config = CORSConfig()
        # Security fix: Default to no CORS instead of wildcard
        assert config.allow_origins == []
        assert config.allow_methods == ["GET", "POST"]
        assert config.allow_headers == ["Content-Type"]
        assert config.allow_credentials is False

    def test_custom_values(self):
        config = CORSConfig(
            allow_origins=["http://localhost:3000", "https://app.example.com"],
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
            allow_credentials=True,
        )
        assert config.allow_origins == ["http://localhost:3000", "https://app.example.com"]
        assert config.allow_methods == ["GET", "POST"]
        assert config.allow_headers == ["Authorization", "Content-Type"]
        assert config.allow_credentials is True


class TestPredictorConfig:
    """Test PredictorConfig validation"""

    def test_minimal_config(self):
        config = PredictorConfig(module="my_module", class_name="MyPredictor")
        assert config.module == "my_module"
        assert config.class_name == "MyPredictor"
        assert config.init_kwargs == {}

    def test_with_init_kwargs(self):
        init_kwargs = {"model_path": "/path/to/model.pkl", "batch_size": 32, "device": "cuda"}
        config = PredictorConfig(
            module="my_module", class_name="MyPredictor", init_kwargs=init_kwargs
        )
        assert config.init_kwargs == init_kwargs

    def test_missing_module(self):
        with pytest.raises(ValidationError):
            PredictorConfig(class_name="MyPredictor")

    def test_missing_class_name(self):
        with pytest.raises(ValidationError):
            PredictorConfig(module="my_module")


class TestApiConfig:
    """Test ApiConfig validation and defaults"""

    def test_default_values(self):
        config = ApiConfig()
        assert config.version == "v1"
        assert config.adapter == "records"
        assert config.feature_order is None
        assert config.endpoints["predict"] is True
        # Note: batch_predict was removed - /predict handles both single and batch
        assert config.endpoints["predict_proba"] is True
        assert config.thread_safe_predict is False

    def test_custom_values(self):
        feature_order = ["feature1", "feature2", "feature3"]
        config = ApiConfig(
            version="v2",
            adapter="ndarray",
            feature_order=feature_order,
            endpoints={
                "predict": True,
                # Note: batch_predict was removed - /predict handles both single and batch
                "predict_proba": False,
            },
            thread_safe_predict=True,
        )
        assert config.version == "v2"
        assert config.adapter == "ndarray"
        assert config.feature_order == feature_order
        assert config.endpoints["predict"] is True
        assert config.endpoints["predict_proba"] is False
        assert config.thread_safe_predict is True


class TestObservabilityConfig:
    """Test ObservabilityConfig validation and defaults"""

    def test_default_values(self):
        config = ObservabilityConfig()
        assert config.metrics is True
        assert config.metrics_endpoint == "/metrics"
        assert config.structured_logging is True
        assert config.log_payloads is False
        assert config.correlation_ids is True

    def test_custom_values(self):
        config = ObservabilityConfig(
            metrics=False,
            metrics_endpoint="/custom-metrics",
            structured_logging=False,
            log_payloads=True,
            correlation_ids=False,
        )
        assert config.metrics is False
        assert config.metrics_endpoint == "/custom-metrics"
        assert config.structured_logging is False
        assert config.log_payloads is True
        assert config.correlation_ids is False

    def test_custom_metrics_endpoint(self):
        config = ObservabilityConfig(metrics_endpoint="/api/v1/metrics")
        assert config.metrics_endpoint == "/api/v1/metrics"


class TestAppConfig:
    """Test main AppConfig validation"""

    def test_minimal_config(self):
        config = AppConfig(
            predictor=PredictorConfig(module="test_module", class_name="TestPredictor"),
            classifier={"name": "test-classifier", "version": "1.0.0"},
            api=ApiConfig(),
        )
        # Should use defaults for server, observability
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.predictor, PredictorConfig)
        assert isinstance(config.api, ApiConfig)
        assert isinstance(config.observability, ObservabilityConfig)

    def test_full_config(self):
        server_config = ServerConfig(title="Test Server", port=9000)
        predictor_config = PredictorConfig(
            module="test_module", class_name="TestPredictor", init_kwargs={"param": "value"}
        )
        api_config = ApiConfig(adapter="ndarray", feature_order=["f1", "f2"])
        observability_config = ObservabilityConfig(metrics=False, structured_logging=False)

        config = AppConfig(
            server=server_config,
            predictor=predictor_config,
            classifier={"name": "test-classifier", "version": "1.0.0"},
            api=api_config,
            observability=observability_config,
        )

        assert config.server.title == "Test Server"
        assert config.server.port == 9000
        assert config.predictor.module == "test_module"
        assert config.api.adapter == "ndarray"
        assert config.observability.metrics is False

    def test_missing_predictor(self):
        with pytest.raises(ValidationError):
            AppConfig()

    def test_classifier_defaults_when_missing(self):
        """Test that classifier gets smart defaults when not provided."""
        config = AppConfig(predictor=PredictorConfig(module="test", class_name="Test"))
        # Should auto-generate classifier metadata
        assert config.classifier is not None
        assert "name" in config.classifier
        assert "version" in config.classifier

    def test_valid_adapters(self):
        for adapter in ["records", "ndarray", "auto"]:
            config = AppConfig(
                predictor=PredictorConfig(module="test", class_name="Test"),
                classifier={"name": "test-classifier", "version": "1.0.0"},
                api=ApiConfig(adapter=adapter),
            )
            assert config.api.adapter == adapter


class TestConfigFromDict:
    """Test configuration creation from dictionary (YAML-like)"""

    def test_from_dict_basic(self):
        config_dict = {
            "server": {"title": "Test API", "port": 8080},
            "predictor": {"module": "my_predictor", "class_name": "Predictor"},
            "classifier": {"name": "test-classifier", "version": "1.0.0"},
            "api": {
                "adapter": "records",
                "endpoints": {
                    "predict": True,
                    # Note: batch_predict was removed - /predict handles both single and batch
                    "predict_proba": True,
                },
            },
            "observability": {"metrics": True, "structured_logging": False},
        }

        config = AppConfig.model_validate(config_dict)
        assert config.server.title == "Test API"
        assert config.server.port == 8080
        assert config.predictor.module == "my_predictor"
        assert config.api.adapter == "records"
        assert config.api.endpoints["predict_proba"] is True
        assert config.observability.metrics is True
        assert config.observability.structured_logging is False

    def test_from_dict_with_defaults(self):
        config_dict = {
            "predictor": {"module": "my_predictor", "class_name": "Predictor"},
            "classifier": {"name": "test-classifier", "version": "1.0.0"},
            "api": {},
        }

        config = AppConfig.model_validate(config_dict)
        # Should fill in defaults
        assert config.server.title == "ML Server"
        assert config.server.port == 8000
        assert config.api.adapter == "records"
        assert config.observability.metrics is True

    def test_from_dict_nested_validation_error(self):
        config_dict = {
            "server": {
                "port": "invalid"  # Should be int
            },
            "predictor": {"module": "test", "class_name": "Test"},
            "classifier": {"name": "test-classifier", "version": "1.0.0"},
            "api": {},
        }

        with pytest.raises(ValidationError) as exc_info:
            AppConfig.model_validate(config_dict)
        assert "port" in str(exc_info.value)

    def test_cors_from_dict(self):
        config_dict = {
            "server": {
                "cors": {"allow_origins": ["http://localhost:3000"], "allow_credentials": True}
            },
            "predictor": {"module": "test", "class_name": "Test"},
            "classifier": {"name": "test-classifier", "version": "1.0.0"},
            "api": {},
        }

        config = AppConfig.model_validate(config_dict)
        assert config.server.cors.allow_origins == ["http://localhost:3000"]
        assert config.server.cors.allow_credentials is True


class TestResponseFormatDeprecations:
    """RFC 0001 D11: deprecated response options warn at config load time."""

    def _deprecation_records(self, caplog):
        return [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "DeprecationWarning" in r.getMessage()
        ]

    def test_custom_response_format_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            config = ApiConfig(response_format="custom")
        assert config.response_format == "custom"  # still accepted, only deprecated
        records = self._deprecation_records(caplog)
        assert len(records) == 1
        assert "response_format" in records[0].getMessage()
        assert "custom" in records[0].getMessage()

    def test_extract_values_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            config = ApiConfig(extract_values=True)
        assert config.extract_values is True  # still accepted, only deprecated
        records = self._deprecation_records(caplog)
        assert len(records) == 1
        assert "extract_values" in records[0].getMessage()

    def test_both_deprecated_options_warn_once_each_per_load(self, caplog):
        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            ApiConfig(response_format="custom", extract_values=True)
        records = self._deprecation_records(caplog)
        assert len(records) == 2

    def test_defaults_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            ApiConfig()
            ApiConfig(response_format="standard")
            ApiConfig(response_format="passthrough")
        assert self._deprecation_records(caplog) == []

    def test_warning_via_full_app_config_load(self, caplog):
        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            AppConfig.model_validate(
                {
                    "predictor": {"module": "m", "class_name": "C"},
                    "api": {"response_format": "custom"},
                }
            )
        assert len(self._deprecation_records(caplog)) == 1


class TestDefaultsModule:
    """RFC 0001 D12: config defaults come from mlserver/defaults.py (env-overridable)."""

    def test_server_config_uses_defaults_module(self):
        config = ServerConfig()
        assert config.host == defaults.DEFAULT_HOST
        assert config.port == defaults.DEFAULT_PORT
        assert config.log_level == defaults.DEFAULT_LOG_LEVEL
        assert config.workers == defaults.DEFAULT_WORKERS

    def test_server_defaults_honor_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MLSERVER_DEFAULT_HOST", "127.0.0.9")
        monkeypatch.setenv("MLSERVER_DEFAULT_PORT", "9123")
        monkeypatch.setenv("MLSERVER_LOG_LEVEL", "DEBUG")

        config = ServerConfig()
        assert config.host == "127.0.0.9"
        assert config.port == 9123
        assert config.log_level == "DEBUG"

    def test_invalid_port_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MLSERVER_DEFAULT_PORT", "not-a-port")
        assert ServerConfig().port == defaults.DEFAULT_PORT

    def test_build_config_uses_defaults_module(self):
        assert BuildConfig().base_image == defaults.DEFAULT_BASE_IMAGE

    def test_no_settings_module_dependency(self):
        """config.py must not import the removed GlobalSettings singleton."""
        import inspect

        import mlserver.config as config_module

        source = inspect.getsource(config_module)
        assert "get_settings" not in source
        assert "from .settings" not in source


class TestGlobalConfigYamlWarning:
    """RFC 0001 D12: presence of a legacy global_config.yaml warns once per process."""

    MINIMAL = {"predictor": {"module": "m", "class_name": "C"}}

    def _records(self, caplog):
        return [
            r for r in caplog.records if "global_config.yaml is no longer read" in r.getMessage()
        ]

    def test_warns_once_when_global_config_present(self, tmp_path, monkeypatch, caplog):
        import mlserver.config as config_module

        (tmp_path / "global_config.yaml").write_text("server: {}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_module, "_GLOBAL_CONFIG_WARNING_EMITTED", False)

        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            AppConfig.model_validate(self.MINIMAL)
            AppConfig.model_validate(self.MINIMAL)  # second load: no repeat warning

        records = self._records(caplog)
        assert len(records) == 1
        assert "RFC 0001 D12" in records[0].getMessage()

    def test_no_warning_without_global_config(self, tmp_path, monkeypatch, caplog):
        import mlserver.config as config_module

        monkeypatch.chdir(tmp_path)  # clean CWD without global_config.yaml
        monkeypatch.setattr(config_module, "_GLOBAL_CONFIG_WARNING_EMITTED", False)

        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            AppConfig.model_validate(self.MINIMAL)

        assert self._records(caplog) == []
        # flag stays unset so a later appearance of the file would still warn
        assert config_module._GLOBAL_CONFIG_WARNING_EMITTED is False


class TestClassifierVersionDeprecation:
    """RFC 0001 D3: classifier.version is display-only and warns at load."""

    def test_warns_once_per_process_when_version_set(self, caplog, monkeypatch):
        import logging

        import mlserver.config as config_module

        monkeypatch.setattr(config_module, "_CLASSIFIER_VERSION_WARNING_EMITTED", False)
        raw = {
            "predictor": {"module": "m", "class_name": "C"},
            "classifier": {"name": "x", "version": "1.0.0"},
        }
        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            AppConfig.model_validate(raw)
            AppConfig.model_validate(raw)  # second load must NOT warn again

        hits = [r for r in caplog.records if "display-only" in r.getMessage()]
        assert len(hits) == 1
        msg = hits[0].getMessage()
        assert "git tags" in msg
        assert "RFC 0001 D3" in msg

    def test_no_warning_without_explicit_version(self, caplog, monkeypatch):
        import logging

        import mlserver.config as config_module

        monkeypatch.setattr(config_module, "_CLASSIFIER_VERSION_WARNING_EMITTED", False)
        with caplog.at_level(logging.WARNING, logger="mlserver.config"):
            # No classifier section at all: the auto-generated default version
            # (added in model_post_init) must not trigger the warning.
            AppConfig.model_validate({"predictor": {"module": "m", "class_name": "C"}})
            # Classifier section without a version key: also silent.
            AppConfig.model_validate(
                {
                    "predictor": {"module": "m", "class_name": "C"},
                    "classifier": {"name": "x"},
                }
            )

        assert not [r for r in caplog.records if "display-only" in r.getMessage()]


class TestConcurrencyDefaults:
    """RFC 0001 D14: the documented concurrency defaults, asserted directly."""

    def test_max_concurrent_predictions_defaults_to_one(self):
        from mlserver.config import ApiConfig

        assert ApiConfig().max_concurrent_predictions == 1

    def test_retry_after_defaults_to_five_seconds(self):
        from mlserver.config import ApiConfig

        assert ApiConfig().retry_after_seconds == 5
