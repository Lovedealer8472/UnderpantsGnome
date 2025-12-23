"""
Exchange Factory - Creates exchange instances based on configuration.
"""
from typing import Optional, Union
from .base import ExchangeBase
from .binance_futures import BinanceFuturesExchange
from .bybit import BybitExchange
from .okx import OKXExchange
from .kraken import KrakenExchange
from .bitget import BitgetExchange
from .mexc import MEXCExchange
from .governed import GovernedExchange
from ..api_governor import ApiGovernor, ApiGovernorConfig
from ..logger import get_logger


def create_exchange(config=None) -> ExchangeBase:
    """
    Create exchange instance based on configuration.
    
    Args:
        config: Configuration object with EXCHANGE setting (optional)
        
    Returns:
        Exchange instance (one of the supported exchanges)
        
    Raises:
        ValueError: If exchange type is not supported
    """
    logger = get_logger("ExchangeFactory")
    
    # Get exchange type from config (default to binance_futures)
    if config:
        exchange_type = getattr(config, 'EXCHANGE', 'binance_futures').lower()
    else:
        from ..config import EXCHANGE
        exchange_type = EXCHANGE.lower()
    
    logger.info(f"Creating exchange: {exchange_type}")

    # Create raw exchange first (pass config through so credentials/settings are consistent)
    if exchange_type == "binance_futures":
        raw = BinanceFuturesExchange(config)
    elif exchange_type == "bybit":
        raw = BybitExchange(config)
    elif exchange_type == "okx":
        raw = OKXExchange(config)
    elif exchange_type == "kraken":
        raw = KrakenExchange(config)
    elif exchange_type == "bitget":
        raw = BitgetExchange(config)
    elif exchange_type == "mexc_futures" or exchange_type == "mexc":
        raw = MEXCExchange(config)
    else:
        supported = "binance_futures, bybit, okx, kraken, bitget, mexc_futures"
        raise ValueError(f"Unsupported exchange: {exchange_type}. Supported: {supported}")

    # Wrap with governor ("single gateway") unless explicitly disabled
    enabled = True
    if config is not None:
        enabled = bool(getattr(config, "API_GOVERNOR_ENABLED", True))

    if not enabled:
        return raw

    cfg = ApiGovernorConfig(
        target_weight_per_min=int(getattr(config, "API_TARGET_WEIGHT_PER_MIN", 1800) if config else 1800),
        burst=int(getattr(config, "API_BURST", 60) if config else 60),
        exit_token_reserve=int(getattr(config, "API_EXIT_TOKEN_RESERVE", 3) if config else 3),
        max_inflight=int(getattr(config, "API_MAX_INFLIGHT", 25) if config else 25),
        call_timeout_sec=float(getattr(config, "API_CALL_TIMEOUT_SEC", 3.0) if config else 3.0),
    )
    governor = ApiGovernor(cfg)
    return GovernedExchange(raw, governor)

