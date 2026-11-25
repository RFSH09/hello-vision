from pathlib import Path

from hello_vision.config import FridgeConfig

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads() -> None:
    cfg = FridgeConfig.load(ROOT / "configs" / "example_fridge.yaml")
    assert cfg.fridge_id == "demo-fridge" and cfg.cameras[0].kind == "file"
    assert cfg.storage.retention_days == 30
