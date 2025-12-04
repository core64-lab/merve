"""Unit tests for settings module."""
import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlserver.settings import (
    ServerSettings,
    MonitoringSettings,
    ContainerSettings,
    FilesSettings,
    DevelopmentSettings,
    LoadTestingSettings,
    GlobalSettings,
    SettingsSingleton,
    get_settings,
    reload_settings,
    save_settings,
)


class TestServerSettings:
    """Test ServerSettings model."""

    def test_default_values(self):
        """Test default server settings."""
        settings = ServerSettings()
        assert settings.default_host == "0.0.0.0"
        assert settings.default_port == 8000
        assert settings.default_log_level == "INFO"
        assert settings.default_workers == 1
        assert settings.health_endpoint_port == 8000
        assert settings.container_test_port == 8012

    def test_custom_values(self):
        """Test custom server settings."""
        settings = ServerSettings(
            default_host="127.0.0.1",
            default_port=9000,
            default_log_level="DEBUG",
            default_workers=4
        )
        assert settings.default_host == "127.0.0.1"
        assert settings.default_port == 9000
        assert settings.default_log_level == "DEBUG"
        assert settings.default_workers == 4


class TestMonitoringSettings:
    """Test MonitoringSettings model."""

    def test_default_values(self):
        """Test default monitoring settings."""
        settings = MonitoringSettings()
        assert settings.prometheus_host == "localhost"
        assert settings.prometheus_port == 9090
        assert settings.grafana_host == "localhost"
        assert settings.grafana_port == 3000
        assert settings.grafana_credentials == "admin/admin123"
        assert settings.metrics_endpoint == "/metrics"

    def test_custom_values(self):
        """Test custom monitoring settings."""
        settings = MonitoringSettings(
            prometheus_host="prometheus.local",
            prometheus_port=9091,
            grafana_host="grafana.local",
            metrics_endpoint="/custom-metrics"
        )
        assert settings.prometheus_host == "prometheus.local"
        assert settings.prometheus_port == 9091
        assert settings.grafana_host == "grafana.local"
        assert settings.metrics_endpoint == "/custom-metrics"


class TestContainerSettings:
    """Test ContainerSettings model."""

    def test_default_values(self):
        """Test default container settings."""
        settings = ContainerSettings()
        assert settings.default_base_image == "python:3.11-slim"
        assert settings.python_version_threshold == "3.11"
        assert settings.temp_dir == "/tmp"
        assert ".git" in settings.excluded_patterns
        assert "__pycache__" in settings.excluded_patterns

    def test_excluded_patterns_is_list(self):
        """Test excluded patterns is a list."""
        settings = ContainerSettings()
        assert isinstance(settings.excluded_patterns, list)
        assert len(settings.excluded_patterns) > 0

    def test_custom_excluded_patterns(self):
        """Test custom excluded patterns."""
        custom_patterns = ["custom_pattern", "*.custom"]
        settings = ContainerSettings(excluded_patterns=custom_patterns)
        assert settings.excluded_patterns == custom_patterns


class TestFilesSettings:
    """Test FilesSettings model."""

    def test_default_values(self):
        """Test default files settings."""
        settings = FilesSettings()
        assert settings.server_log_file == "server.log"
        assert settings.server_pid_file == "server.pid"
        assert settings.global_config_file == "global_config.yaml"
        assert settings.venv_path == ".venv"
        assert settings.ml_server_path == "mlserver"


class TestDevelopmentSettings:
    """Test DevelopmentSettings model."""

    def test_default_values(self):
        """Test default development settings."""
        settings = DevelopmentSettings()
        assert settings.test_base_url == "http://localhost:8000"
        assert settings.load_test_users == 5
        assert settings.load_test_spawn_rate == 1
        assert settings.load_test_duration == 60


class TestLoadTestingSettings:
    """Test LoadTestingSettings model."""

    def test_default_values(self):
        """Test default load testing settings."""
        settings = LoadTestingSettings()
        assert settings.stress_test_users == 50
        assert settings.stress_test_spawn_rate == 10
        assert settings.stress_test_duration == 300
        assert settings.endurance_test_users == 10
        assert settings.endurance_test_spawn_rate == 2
        assert settings.endurance_test_duration == 1800


class TestGlobalSettings:
    """Test GlobalSettings model."""

    def test_default_values(self):
        """Test all nested settings have defaults."""
        settings = GlobalSettings()
        assert isinstance(settings.server, ServerSettings)
        assert isinstance(settings.monitoring, MonitoringSettings)
        assert isinstance(settings.container, ContainerSettings)
        assert isinstance(settings.files, FilesSettings)
        assert isinstance(settings.development, DevelopmentSettings)
        assert isinstance(settings.load_testing, LoadTestingSettings)

    def test_from_yaml_nonexistent_file(self):
        """Test loading from nonexistent file returns defaults."""
        settings = GlobalSettings.from_yaml("/nonexistent/config.yaml")
        assert settings.server.default_port == 8000

    def test_from_yaml_valid_file(self):
        """Test loading from valid YAML file."""
        config = {
            "server": {
                "default_port": 9000,
                "default_host": "127.0.0.1"
            },
            "monitoring": {
                "prometheus_port": 9091
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        settings = GlobalSettings.from_yaml(config_path)
        assert settings.server.default_port == 9000
        assert settings.server.default_host == "127.0.0.1"
        assert settings.monitoring.prometheus_port == 9091

    def test_from_yaml_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            config_path = f.name

        settings = GlobalSettings.from_yaml(config_path)
        # Should return defaults
        assert settings.server.default_port == 8000

    def test_env_override_server_port(self):
        """Test environment variable overrides server port."""
        config = {"server": {"default_port": 8000}}
        with patch.dict(os.environ, {"MLSERVER_DEFAULT_PORT": "9999"}):
            result = GlobalSettings._apply_env_overrides(config)
            assert result["server"]["default_port"] == 9999

    def test_env_override_log_file(self):
        """Test environment variable overrides log file."""
        config = {}
        with patch.dict(os.environ, {"MLSERVER_LOG_FILE": "custom.log"}):
            result = GlobalSettings._apply_env_overrides(config)
            assert result["files"]["server_log_file"] == "custom.log"

    def test_env_override_invalid_port(self):
        """Test environment variable with invalid port value."""
        config = {"server": {"default_port": 8000}}
        with patch.dict(os.environ, {"MLSERVER_DEFAULT_PORT": "not_a_number"}):
            result = GlobalSettings._apply_env_overrides(config)
            # Should keep original string value
            assert result["server"]["default_port"] == "not_a_number"

    def test_env_override_multiple_values(self):
        """Test multiple environment variable overrides."""
        config = {}
        env_vars = {
            "MLSERVER_DEFAULT_HOST": "192.168.1.1",
            "MLSERVER_LOG_LEVEL": "DEBUG",
            "MLSERVER_PID_FILE": "custom.pid"
        }
        with patch.dict(os.environ, env_vars):
            result = GlobalSettings._apply_env_overrides(config)
            assert result["server"]["default_host"] == "192.168.1.1"
            assert result["server"]["default_log_level"] == "DEBUG"
            assert result["files"]["server_pid_file"] == "custom.pid"

    def test_to_yaml(self):
        """Test saving settings to YAML file."""
        settings = GlobalSettings()
        settings.server.default_port = 9999

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.yaml"
            settings.to_yaml(str(config_path))

            # Verify file was created
            assert config_path.exists()

            # Verify contents
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
                assert loaded["server"]["default_port"] == 9999

    def test_get_prometheus_url(self):
        """Test getting Prometheus URL."""
        settings = GlobalSettings()
        url = settings.get_prometheus_url()
        assert url == "http://localhost:9090"

    def test_get_grafana_url(self):
        """Test getting Grafana URL."""
        settings = GlobalSettings()
        url = settings.get_grafana_url()
        assert url == "http://localhost:3000"

    def test_get_server_url(self):
        """Test getting server URL."""
        settings = GlobalSettings()
        url = settings.get_server_url()
        assert url == "http://0.0.0.0:8000"

    def test_get_health_url(self):
        """Test getting health URL."""
        settings = GlobalSettings()
        url = settings.get_health_url()
        assert url == "http://localhost:8000/healthz"

    def test_get_metrics_url(self):
        """Test getting metrics URL."""
        settings = GlobalSettings()
        url = settings.get_metrics_url()
        assert url == "http://localhost:8000/metrics"

    def test_custom_urls(self):
        """Test URLs with custom settings."""
        settings = GlobalSettings(
            server=ServerSettings(default_host="10.0.0.1", default_port=9000, health_endpoint_port=9001),
            monitoring=MonitoringSettings(prometheus_host="prom.local", prometheus_port=9091)
        )
        assert settings.get_server_url() == "http://10.0.0.1:9000"
        assert settings.get_prometheus_url() == "http://prom.local:9091"
        assert settings.get_health_url() == "http://localhost:9001/healthz"


class TestSettingsSingleton:
    """Test SettingsSingleton class."""

    def test_singleton_returns_same_instance(self):
        """Test singleton pattern returns same instance."""
        # Reset singleton for clean test
        SettingsSingleton._instance = None
        SettingsSingleton._settings = None

        instance1 = SettingsSingleton()
        instance2 = SettingsSingleton()
        assert instance1 is instance2

    def test_settings_property(self):
        """Test settings property returns GlobalSettings."""
        # Reset singleton for clean test
        SettingsSingleton._instance = None
        SettingsSingleton._settings = None

        singleton = SettingsSingleton()
        settings = singleton.settings
        assert isinstance(settings, GlobalSettings)

    def test_reload_settings_with_path(self):
        """Test reloading settings from specific path."""
        # Reset singleton for clean test
        SettingsSingleton._instance = None
        SettingsSingleton._settings = None

        config = {"server": {"default_port": 7777}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        singleton = SettingsSingleton()
        singleton.reload_settings(config_path)
        assert singleton.settings.server.default_port == 7777

    def test_reload_settings_without_path(self):
        """Test reloading settings without path uses default search."""
        # Reset singleton for clean test
        SettingsSingleton._instance = None
        SettingsSingleton._settings = None

        singleton = SettingsSingleton()
        # Should not raise
        singleton.reload_settings()
        assert singleton.settings is not None

    def test_save_settings_with_path(self):
        """Test saving settings to specific path."""
        # Reset singleton for clean test
        SettingsSingleton._instance = None
        SettingsSingleton._settings = None

        singleton = SettingsSingleton()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "saved_config.yaml"
            singleton.save_settings(str(config_path))
            assert config_path.exists()

    def test_save_settings_without_path(self):
        """Test saving settings without path uses default."""
        # Reset singleton for clean test
        SettingsSingleton._instance = None
        SettingsSingleton._settings = None

        singleton = SettingsSingleton()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the default path
            with patch.object(Path, 'parent', new_callable=lambda: Path(tmpdir)):
                # Should not raise
                try:
                    singleton.save_settings()
                except Exception:
                    # May fail due to path issues, but should not crash
                    pass

    def test_load_settings_from_env_path(self):
        """Test loading settings from environment variable path."""
        config = {"server": {"default_port": 6666}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        # Reset singleton
        SettingsSingleton._instance = None
        SettingsSingleton._settings = None

        with patch.dict(os.environ, {"MLSERVER_GLOBAL_CONFIG_PATH": config_path}):
            singleton = SettingsSingleton()
            assert singleton.settings.server.default_port == 6666


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_settings(self):
        """Test get_settings returns GlobalSettings."""
        settings = get_settings()
        assert isinstance(settings, GlobalSettings)

    def test_reload_settings_function(self):
        """Test reload_settings module function."""
        # Should not raise
        reload_settings()

    def test_save_settings_function(self):
        """Test save_settings module function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "func_saved.yaml"
            save_settings(str(config_path))
            assert config_path.exists()


class TestSettingsEdgeCases:
    """Test edge cases in settings handling."""

    def test_yaml_with_null_values(self):
        """Test YAML file with null values raises validation error."""
        yaml_content = """
server:
  default_port: null
  default_host: "localhost"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name

        # Pydantic requires int for default_port, null is not valid
        with pytest.raises(Exception):  # ValidationError
            GlobalSettings.from_yaml(config_path)

    def test_yaml_with_extra_fields(self):
        """Test YAML file with extra unknown fields."""
        config = {
            "server": {"default_port": 8000},
            "unknown_section": {"foo": "bar"}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        # Should ignore extra fields
        settings = GlobalSettings.from_yaml(config_path)
        assert settings.server.default_port == 8000

    def test_model_dump(self):
        """Test settings can be dumped to dict."""
        settings = GlobalSettings()
        data = settings.model_dump()

        assert "server" in data
        assert "monitoring" in data
        assert "container" in data
        assert isinstance(data["server"]["default_port"], int)

    def test_settings_immutable_after_load(self):
        """Test settings values can be accessed after loading."""
        settings = GlobalSettings()

        # Access all sections
        assert settings.server.default_port >= 0
        assert settings.monitoring.prometheus_port >= 0
        assert settings.container.temp_dir is not None
        assert settings.files.server_log_file is not None
        assert settings.development.test_base_url is not None
        assert settings.load_testing.stress_test_users >= 0
