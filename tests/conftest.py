import os
import shutil
import tempfile
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
    "judge_total_caps_usd": {"meta_api": 0.30, "baseten": 0.30},
    "public_daily_cap_usd": 0.05,
    "public_per_ip_daily_analyses": 3,
    "warn_at_fraction": 0.75,
}


def pytest_configure(config):
    """Before any test module imports the app: a private config folder with test-only caps.

    The app refuses to start without config/budget.yaml (it must never fall back to example caps). Tests should neither
    depend on a developer's real caps nor fail on a fresh clone that has not run `make setup` yet.
    """
    # Runtime data (mock ledger, cache, runs) per test process: parallel workers must not share one ledger.
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    os.environ["PHYSICSLENS_VAR_DIR"] = tempfile.mkdtemp(prefix=f"physicslens-test-var-{worker}-")
    if os.environ.get("PHYSICSLENS_CONFIG_DIR"):
        return
    tmp = Path(tempfile.mkdtemp(prefix="physicslens-test-config-"))
    for name in ("models.yaml", "prices.yaml", "limits.yaml"):
        shutil.copy(REPO_CONFIG / name, tmp / name)
    (tmp / "budget.yaml").write_text(yaml.safe_dump(TEST_BUDGET))
    os.environ["PHYSICSLENS_CONFIG_DIR"] = str(tmp)


@pytest.fixture(autouse=True)
def _no_kill_env(monkeypatch):
    monkeypatch.delenv("PHYSICSLENS_KILL", raising=False)
    # Importing the app loads the developer's .env; tests choose their reasoning provider explicitly. Set to empty,
    # not deleted: when a test is the first to import the app (common under pytest-xdist), .env would put a deleted
    # variable straight back, but it never overrides one that is present.
    for name in ("REASONING_PROVIDER", "GLIMMER_BASE_URL", "PHYSICSLENS_CACHE_ONLY"):
        monkeypatch.setenv(name, "")


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


@pytest.fixture(autouse=True)
def _isolated_law_registry():
    """custom.register() adds the agent's laws to the global MODELS (and FREE_MODELS): restore both after every test,
    so a law one test wrote cannot change another test's candidates, whatever order or worker runs them."""
    from physicslens.physics.custom import FREE_MODELS
    from physicslens.physics.models import MODELS
    saved_models, saved_free = dict(MODELS), dict(FREE_MODELS)
    yield
    MODELS.clear()
    MODELS.update(saved_models)
    FREE_MODELS.clear()
    FREE_MODELS.update(saved_free)


# Tests over 10 s on one core (the slowest 30 of a parallel run, measured; re-measure with --durations=30).
# `make test-fast` skips them; `make test` and CI run everything.
SLOW_TESTS = {
    "test_a_law_never_ends_worse_than_its_special_case",
    "test_both_frictions_from_the_truth_reach_the_noise_floor",
    "test_capture_timestamps_set_the_time_base",
    "test_chaotic_swing_is_fitted_and_flagged",
    "test_damped_pendulum_recovers_g_and_damping",
    "test_damped_spring",
    "test_door_closer_is_a_damped_torsional_spring",
    "test_dry_friction_spring_is_told_from_viscous_damping",
    "test_friction_law_identified",
    "test_live_prediction_is_stored_before_it_happens_and_then_judged",
    "test_live_session_switches_to_the_new_regime",
    "test_live_stream_tracks_fits_and_predicts",
    "test_no_design_without_a_tie",
    "test_no_trend_when_nothing_changes",
    "test_one_law_written_twice_is_not_a_tie",
    "test_pipeline_double_analysis",
    "test_pipeline_records_the_referee_in_the_trail_and_result",
    "test_predictions_from_the_first_half_land_in_their_window",
    "test_sindy_is_hidden_when_its_forecast_is_far_worse",
    "test_slow_motion_factor_puts_the_physics_in_real_time",
    "test_small_swing_ties_and_the_requested_large_swing_separates_the_laws",
    "test_spring_with_both_frictions_needs_both_terms",
    "test_study_review_fits_the_recordings_together",
    "test_sun_planets_and_moon_reproduce_the_2029_flyby",
    "test_tapped_family_fits_at_once_and_the_agent_still_proposes",
    "test_the_law_survives_when_the_alternative_does_not_forecast_better",
    "test_trend_catches_slowly_rising_damping",
    "test_trend_endpoint_caches_metrics",
    "test_undamped_pendulum_prefers_simplest_model",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.originalname in SLOW_TESTS or item.name in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
