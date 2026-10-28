"""Environment and config loading. No network calls here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ENVS = ("dev", "demo", "prod")

META_TEXT_MODEL = "muse-spark-1.3"
META_CONTRIBUTOR_MODEL = "muse-spark-1.3-contributor"
SAM_MODEL = "sam-3.1"
VOICE_MODEL = "muse-voice-transcribe-1.0"
IMAGE_MODEL = "muse-image-1.0"
GLIMMER_MODEL = "muse-glimmer-30b"


class ConfigError(RuntimeError):
    """Configuration is missing or unsafe; the app must not start."""


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() == "1"


@dataclass(frozen=True)
class Settings:
    app_env: str = "dev"
    mock_models: bool = False
    use_contributor_model: bool = False
    repo_root: Path = REPO_ROOT
    config_dir: Path = field(default=REPO_ROOT / "config")
    var_dir: Path = field(default=REPO_ROOT / "var")

    @property
    def data_dir(self) -> Path:
        # Mock runs get their own ledger and cache so fake spend never mixes with real spend.
        return self.var_dir / "mock" if self.mock_models else self.var_dir

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "ledger.db"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache.db"

    @property
    def kill_file(self) -> Path:
        return self.repo_root / ".KILL"

    def meta_text_model(self) -> str:
        if self.app_env == "dev" and self.use_contributor_model:
            return META_CONTRIBUTOR_MODEL
        return META_TEXT_MODEL


def validate(settings: Settings) -> Settings:
    if settings.app_env not in APP_ENVS:
        raise ConfigError(f"APP_ENV must be one of {APP_ENVS}, got {settings.app_env!r}.")
    if settings.mock_models and settings.app_env in ("demo", "prod"):
        raise ConfigError(
            f"MOCK_MODELS=1 is not allowed when APP_ENV={settings.app_env}. "
            "Mocks are for automated tests only."
        )
    return settings


def load_settings(read_env_file: bool = True) -> Settings:
    if read_env_file:
        load_dotenv(REPO_ROOT / ".env", override=False)
    return validate(
        Settings(
            app_env=os.environ.get("APP_ENV", "dev").strip().lower(),
            mock_models=_flag("MOCK_MODELS"),
            use_contributor_model=_flag("USE_CONTRIBUTOR_MODEL"),
        )
    )


def load_yaml(config_dir: Path, name: str) -> dict[str, Any]:
    path = config_dir / name
    if not path.is_file():
        raise ConfigError(f"Missing config file {path}.")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping.")
    return data


def load_budget(config_dir: Path) -> dict[str, Any]:
    # Deliberately no fallback to budget.example.yaml: example caps are not the developer's real caps.
    if not (config_dir / "budget.yaml").is_file():
        raise ConfigError(
            "config/budget.yaml is missing. Run `make setup` to create it from "
            "config/budget.example.yaml, then review the caps before making any paid call."
        )
    return load_yaml(config_dir, "budget.yaml")
