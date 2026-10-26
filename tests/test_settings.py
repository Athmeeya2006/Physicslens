import pytest
from physicslens.settings import ConfigError, Settings, load_budget, validate


@pytest.mark.parametrize("env", ["demo", "prod"])
def test_mock_models_refused_outside_dev(env):
    with pytest.raises(ConfigError, match="MOCK_MODELS=1 is not allowed"):
        validate(Settings(app_env=env, mock_models=True))


def test_unknown_app_env_refused():
    with pytest.raises(ConfigError):
        validate(Settings(app_env="staging"))


def test_missing_budget_points_to_make_setup(tmp_path):
    (tmp_path / "budget.example.yaml").write_text("per_run_cap_usd: 99\n")
    with pytest.raises(ConfigError, match="make setup"):
        load_budget(tmp_path)


def test_contributor_model_only_in_dev():
    assert Settings(app_env="dev", use_contributor_model=True).meta_text_model() == "muse-spark-1.3-contributor"
    assert Settings(app_env="demo", use_contributor_model=True).meta_text_model() == "muse-spark-1.3"


def test_mock_mode_uses_separate_ledger(tmp_path):
    real = Settings(var_dir=tmp_path)
    mock = Settings(var_dir=tmp_path, mock_models=True)
    assert real.ledger_path != mock.ledger_path
