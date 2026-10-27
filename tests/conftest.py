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
    "baseten": {"hard_cap_usd": 1.00, "daily_cap_usd": 0.50},
    "glimmer": {"hard_cap_usd": 0.00, "daily_cap_usd": 0.00},
    "modal": {"hard_cap_usd": 1.00, "daily_cap_usd": 0.50},
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


@pytest.fixture
def settings(tmp_path) -> Settings:
    config = tmp_path / "config"
    config.mkdir()
    for name in ("models.yaml", "prices.yaml", "limits.yaml"):
        shutil.copy(REPO_CONFIG / name, config / name)
    (config / "budget.yaml").write_text(yaml.safe_dump(TEST_BUDGET))
    return Settings(app_env="dev", mock_models=True, repo_root=tmp_path, config_dir=config, var_dir=tmp_path / "var")


def set_baseten_price(settings: Settings, input_price: float = 1.0, output_price: float = 2.0) -> None:
    path = settings.config_dir / "prices.yaml"
    prices = yaml.safe_load(path.read_text())
    prices["baseten"]["meta-models/muse-spark-open-1026"] = {"input": input_price, "output": output_price}
    path.write_text(yaml.safe_dump(prices))


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
