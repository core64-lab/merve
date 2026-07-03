"""Tests for the Predictor protocol, string predictor specs, import isolation,
and the optional load() startup lifecycle (RFC 0001, D13)."""
import sys

import pytest
from pydantic import ValidationError

from mlserver.config import AppConfig
from mlserver.errors import PredictorError
from mlserver.predictor import Predictor
from mlserver.predictor_loader import USER_MODULE_NAMESPACE, load_predictor


class TestPredictorProtocol:
    """Runtime-checkable Predictor protocol sanity."""

    def test_class_with_predict_satisfies_protocol(self):
        class MyPredictor:
            def predict(self, X):
                return [0] * len(X)

        assert isinstance(MyPredictor(), Predictor)

    def test_class_without_predict_fails_protocol(self):
        class NotAPredictor:
            def transform(self, X):
                return X

        assert not isinstance(NotAPredictor(), Predictor)

    def test_optional_methods_not_required_by_protocol(self):
        """predict_proba/load/close are optional - predict alone suffices."""

        class MinimalPredictor:
            def predict(self, X):
                return X

        obj = MinimalPredictor()
        assert isinstance(obj, Predictor)
        assert not hasattr(obj, "predict_proba")
        assert not hasattr(obj, "load")

    def test_full_featured_predictor_satisfies_protocol(self):
        class FullPredictor:
            def load(self):
                pass

            def predict(self, X):
                return X

            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)

            def close(self):
                pass

        assert isinstance(FullPredictor(), Predictor)

    def test_protocol_is_runtime_checkable_class_level(self):
        # runtime_checkable protocols support issubclass on the class too
        class MyPredictor:
            def predict(self, X):
                return X

        assert issubclass(MyPredictor, Predictor)


class TestStringPredictorSpec:
    """predictor: "module:ClassName" string spec (RFC 0001 D13)."""

    def test_valid_string_spec(self):
        config = AppConfig.model_validate({"predictor": "my_predictor:MyPredictor"})
        assert config.predictor.module == "my_predictor"
        assert config.predictor.class_name == "MyPredictor"
        assert config.predictor.init_kwargs == {}

    def test_dotted_module_string_spec(self):
        config = AppConfig.model_validate(
            {"predictor": "mypackage.predictors.catboost:CatBoostPredictor"}
        )
        assert config.predictor.module == "mypackage.predictors.catboost"
        assert config.predictor.class_name == "CatBoostPredictor"

    def test_string_spec_strips_whitespace(self):
        config = AppConfig.model_validate({"predictor": " my_predictor : MyPredictor "})
        assert config.predictor.module == "my_predictor"
        assert config.predictor.class_name == "MyPredictor"

    def test_missing_colon_rejected(self):
        with pytest.raises(ValidationError, match="module:ClassName"):
            AppConfig.model_validate({"predictor": "my_predictor.MyPredictor"})

    @pytest.mark.parametrize("bad_spec", [":MyPredictor", "my_predictor:", ":", "  :  "])
    def test_empty_parts_rejected(self, bad_spec):
        with pytest.raises(ValidationError, match="module:ClassName"):
            AppConfig.model_validate({"predictor": bad_spec})

    def test_legacy_mapping_spec_still_works(self):
        """Regression: the two-field mapping form is unchanged."""
        config = AppConfig.model_validate({
            "predictor": {
                "module": "my_predictor",
                "class_name": "MyPredictor",
                "init_kwargs": {"model_path": "model.pkl"},
            }
        })
        assert config.predictor.module == "my_predictor"
        assert config.predictor.class_name == "MyPredictor"
        assert config.predictor.init_kwargs == {"model_path": "model.pkl"}


class TestImportIsolation:
    """File-based predictors import under merve._user.* - no stdlib shadowing."""

    def test_predictor_file_named_json_loads_and_stdlib_survives(self, tmp_path):
        (tmp_path / "json.py").write_text(
            "import json  # absolute import must reach the stdlib, not this file\n"
            "\n"
            "class JsonNamedPredictor:\n"
            "    def predict(self, X):\n"
            "        return [json.dumps({'n': len(X)})]\n"
        )

        predictor = load_predictor("json", "JsonNamedPredictor", {}, config_dir=str(tmp_path))
        assert predictor.predict([1, 2, 3]) == ['{"n": 3}']

        # stdlib json must be fully intact afterwards
        import json
        assert json.loads(json.dumps({"a": 1})) == {"a": 1}
        assert hasattr(sys.modules["json"], "JSONDecoder")

        # user module lives only under the isolation namespace
        assert f"{USER_MODULE_NAMESPACE}.json" in sys.modules
        assert type(predictor).__module__ == f"{USER_MODULE_NAMESPACE}.json"

    def test_predictor_file_named_types_loads_and_stdlib_survives(self, tmp_path):
        (tmp_path / "types.py").write_text(
            "class TypesNamedPredictor:\n"
            "    def predict(self, X):\n"
            "        return [42] * len(X)\n"
        )

        predictor = load_predictor("types", "TypesNamedPredictor", {}, config_dir=str(tmp_path))
        assert predictor.predict([0, 0]) == [42, 42]

        # stdlib types must be fully intact afterwards
        import types
        assert hasattr(types, "ModuleType")
        assert hasattr(sys.modules["types"], "ModuleType")
        assert type(predictor).__module__ == f"{USER_MODULE_NAMESPACE}.types"

    def test_no_sys_path_mutation(self, tmp_path):
        (tmp_path / "clean_predictor.py").write_text(
            "class CleanPredictor:\n"
            "    def predict(self, X):\n"
            "        return X\n"
        )
        path_before = list(sys.path)

        load_predictor("clean_predictor", "CleanPredictor", {}, config_dir=str(tmp_path))

        assert sys.path == path_before
        assert str(tmp_path) not in sys.path
        assert str(tmp_path.resolve()) not in sys.path

    def test_import_error_in_user_file_cleans_namespace(self, tmp_path):
        (tmp_path / "broken_predictor.py").write_text("raise RuntimeError('boom at import')\n")

        with pytest.raises(PredictorError, match="broken_predictor"):
            load_predictor("broken_predictor", "Anything", {}, config_dir=str(tmp_path))

        # a failed exec must not leave a half-initialized module registered
        assert f"{USER_MODULE_NAMESPACE}.broken_predictor" not in sys.modules


class TestLoadLifecycle:
    """Optional load() hook: called exactly once at startup, failures abort."""

    @staticmethod
    def _write_predictor(tmp_path, body: str, name: str = "lifecycle_predictor") -> None:
        (tmp_path / f"{name}.py").write_text(body)

    @staticmethod
    def _make_config(tmp_path, class_name: str) -> AppConfig:
        config = AppConfig.model_validate({
            "predictor": {"module": "lifecycle_predictor", "class_name": class_name},
            "classifier": {"name": "lifecycle-test", "version": "1.0.0"},
            "api": {"warmup_on_start": False},
            "observability": {"metrics": False, "structured_logging": False},
        })
        config.set_project_path(str(tmp_path))
        return config

    async def test_load_called_exactly_once_at_startup(self, tmp_path):
        from mlserver.server import create_app

        self._write_predictor(tmp_path, (
            "class LoadTrackingPredictor:\n"
            "    def __init__(self):\n"
            "        self.load_calls = 0\n"
            "    def load(self):\n"
            "        self.load_calls += 1\n"
            "    def predict(self, X):\n"
            "        return [0] * len(X)\n"
        ))
        config = self._make_config(tmp_path, "LoadTrackingPredictor")
        app = create_app(config)

        async with app.router.lifespan_context(app):
            assert app.state.predictor._predictor.load_calls == 1

    async def test_load_called_before_first_prediction_including_warmup(self, tmp_path):
        from mlserver.server import create_app

        self._write_predictor(tmp_path, (
            "class OrderedPredictor:\n"
            "    def __init__(self):\n"
            "        self.loaded = False\n"
            "    def load(self):\n"
            "        self.loaded = True\n"
            "    def predict(self, X):\n"
            "        assert self.loaded, 'predict() ran before load()'\n"
            "        return [1] * len(X)\n"
        ))
        config = AppConfig.model_validate({
            "predictor": {"module": "lifecycle_predictor", "class_name": "OrderedPredictor"},
            "classifier": {"name": "lifecycle-test", "version": "1.0.0"},
            # warmup ON: the warmup prediction must run after load()
            "api": {"warmup_on_start": True, "feature_order": ["f1", "f2"]},
            "observability": {"metrics": False, "structured_logging": False},
        })
        config.set_project_path(str(tmp_path))
        app = create_app(config)

        async with app.router.lifespan_context(app):
            assert app.state.predictor._predictor.loaded is True

    async def test_load_failure_aborts_startup(self, tmp_path):
        from mlserver.server import create_app

        self._write_predictor(tmp_path, (
            "class FailingLoadPredictor:\n"
            "    def load(self):\n"
            "        raise RuntimeError('artifact missing')\n"
            "    def predict(self, X):\n"
            "        return [0] * len(X)\n"
        ))
        config = self._make_config(tmp_path, "FailingLoadPredictor")
        app = create_app(config)

        with pytest.raises(PredictorError, match="load\\(\\) failed"):
            async with app.router.lifespan_context(app):
                pass  # pragma: no cover - startup must fail before entry

        # predictor must NOT be marked ready
        assert getattr(app.state, "predictor", None) is None

    async def test_predictor_without_load_starts_normally(self, tmp_path):
        """Regression: load() stays optional."""
        from mlserver.server import create_app

        self._write_predictor(tmp_path, (
            "class PlainPredictor:\n"
            "    def predict(self, X):\n"
            "        return [7] * len(X)\n"
        ))
        config = self._make_config(tmp_path, "PlainPredictor")
        app = create_app(config)

        async with app.router.lifespan_context(app):
            assert app.state.predictor.name == "PlainPredictor"
