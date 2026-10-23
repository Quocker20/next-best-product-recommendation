from nbp.config import DataConfig, load_config
from nbp.paths import CONFIGS, ROOT


def test_paths_exist():
    assert ROOT.is_dir()
    assert CONFIGS.is_dir()


def test_config_loads():
    cfg = load_config(DataConfig, CONFIGS / "data.yaml")
    assert cfg.max_history == 20
