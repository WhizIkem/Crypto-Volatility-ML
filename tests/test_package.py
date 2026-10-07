import crypto_volatility
from crypto_volatility.utils.config import load_config


# Test suite for the crypto_volatility package
def test_version() -> None:
    """Test that the package version is correct."""
    assert crypto_volatility.__version__ == "0.1.0"


# Test loading the configuration file
def test_load_config():
    config = load_config()

    assert "data" in config
    assert "symbols" in config["data"]
    assert "BTC" in config["data"]["symbols"]
    assert "ETH" in config["data"]["symbols"]
    assert "SOL" in config["data"]["symbols"]
    assert "BNB" in config["data"]["symbols"]
