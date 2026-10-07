import crypto_volatility
from crypto_volatility.utils.config import load_config


def test_version() -> None:
    """Test that the package version is correct."""
    assert crypto_volatility.__version__ == "0.1.0"


def test_load_config():
    config = load_config()

    assert "data" in config
    assert "assets" in config["data"]
    assert "BTC" in config["data"]["assets"]
    assert "ETH" in config["data"]["assets"]
    assert "SOL" in config["data"]["assets"]
    assert "BNB" in config["data"]["assets"]