import shutil
from pathlib import Path

import pytest
import yaml
from physicslens.costguard import CostGuardClient
from physicslens.costguard.mock import MockSDK
from physicslens.settings import Settings

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"

TEST_BUDGET = {
    "meta_api": {"hard_cap_usd": 1.00, "daily_cap_usd": 0.50},
    "glimmer": {"hard_cap_usd": 0.00, "daily_cap_usd": 0.00},
    "baseten": {"hard_cap_usd": 0.50, "daily_cap_usd": 0.20},
    "together": {"hard_cap_usd": 0.50, "daily_cap_usd": 0.20},
    "per_run_cap_usd": 0.20,
    "smoke_cap_usd": 0.10,
    "judge_daily_cap_usd": 0.30,
    "public_daily_cap_usd": 0.05,
    "public_per_ip_daily_analyses": 3,
    "warn_at_fraction": 0.75,
}


@pytest.fixture(autouse=True)
def _no_kill_env(monkeypatch):
    monkeypatch.delenv("PHYSICSLENS_KILL", raising=False)
    # Importing the app loads the developer's .env; tests choose their reasoning provider explicitly.
    for name in ("REASONING_PROVIDER", "GLIMMER_BASE_URL", "PHYSICSLENS_CACHE_ONLY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    config = tmp_path / "config"
    config.mkdir()
    for name in ("models.yaml", "prices.yaml", "limits.yaml"):
        shutil.copy(REPO_CONFIG / name, config / name)
    # Most tests script the model's replies in order; the single-frame locate call is tested on its own.
    limits = yaml.safe_load((config / "limits.yaml").read_text())
    limits.setdefault("scene", {})["locate_single_image"] = False
    limits.setdefault("agent", {})["hypothesis_samples"] = 1
    (config / "limits.yaml").write_text(yaml.safe_dump(limits))
    (config / "budget.yaml").write_text(yaml.safe_dump(TEST_BUDGET))
    return Settings(app_env="dev", mock_models=True, repo_root=tmp_path, config_dir=config, var_dir=tmp_path / "var")


class Harness:
    def __init__(self, settings: Settings):
        self.sdks: dict[str, MockSDK] = {}
        self.sleeps: list[float] = []

        def factory(provider, base_url, api_key_env, timeout):
            return self.sdks.setdefault(provider, MockSDK(provider))

        self.client = CostGuardClient(settings, sdk_factory=factory, sleep=self.sleeps.append)

    def sdk(self, provider: str = "meta_api") -> MockSDK:
        return self.sdks.setdefault(provider, MockSDK(provider))


@pytest.fixture
def harness(settings) -> Harness:
    return Harness(settings)
