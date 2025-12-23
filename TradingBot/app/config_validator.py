"""
Configuration validator for trading bot settings.
Validates critical configuration values and provides warnings/errors.

CRITICAL: NO print() statements - all output goes to logger.
"""

import sys
from typing import List, Tuple, Optional


def validate_config() -> Tuple[List[str], List[str]]:
    """
    Validate configuration and return lists of warnings and errors.
    
    Returns:
        Tuple of (warnings, errors)
    """
    warnings = []
    errors = []
    
    try:
        from .config import (
            DRY_RUN, EXCHANGE, MAX_CONCURRENT_POS, MAX_CAPITAL_PER_POS,
            MAX_LEVERAGE, ACCOUNT_BAL, MIN_STOP_DISTANCE_PCT,
            MAX_RISK_PER_TRADE_PCT
        )
        
        # Optional: Try to import signal threshold (may not exist in all configs)
        try:
            from .config import SIGNAL_MIN_SCORE_ENTRY
        except ImportError:
            SIGNAL_MIN_SCORE_ENTRY = 65  # Default value
        
        # Exchange validation
        valid_exchanges = ['binance_futures', 'bybit', 'okx', 'kraken', 'bitget', 'mexc_futures', 'binance', 'mexc']
        if EXCHANGE.lower() not in valid_exchanges:
            errors.append(f"Invalid EXCHANGE: {EXCHANGE}. Must be one of {valid_exchanges}")
        
        # Position limits
        if MAX_CONCURRENT_POS < 1:
            errors.append(f"MAX_CONCURRENT_POS must be >= 1, got {MAX_CONCURRENT_POS}")
        elif MAX_CONCURRENT_POS > 10:
            warnings.append(f"MAX_CONCURRENT_POS is high ({MAX_CONCURRENT_POS}). Risk of overexposure.")
        
        # Capital per position
        if MAX_CAPITAL_PER_POS is not None:
            if MAX_CAPITAL_PER_POS <= 0 or MAX_CAPITAL_PER_POS > 1:
                errors.append(f"MAX_CAPITAL_PER_POS must be between 0 and 1, got {MAX_CAPITAL_PER_POS}")
            elif MAX_CAPITAL_PER_POS > 0.25:
                warnings.append(f"MAX_CAPITAL_PER_POS is high ({MAX_CAPITAL_PER_POS*100:.1f}%). Risk of overexposure.")
        
        # Leverage
        if MAX_LEVERAGE is not None:
            if MAX_LEVERAGE < 1:
                errors.append(f"MAX_LEVERAGE must be >= 1, got {MAX_LEVERAGE}")
            elif MAX_LEVERAGE > 20:
                warnings.append(f"MAX_LEVERAGE is very high ({MAX_LEVERAGE}x). Extreme risk.")
            elif MAX_LEVERAGE > 10:
                warnings.append(f"MAX_LEVERAGE is high ({MAX_LEVERAGE}x). High risk.")
        
        # Account balance
        if ACCOUNT_BAL is not None:
            if ACCOUNT_BAL <= 0:
                errors.append(f"ACCOUNT_BAL must be > 0, got {ACCOUNT_BAL}")
        
        # Stop distance
        if MIN_STOP_DISTANCE_PCT is not None:
            if MIN_STOP_DISTANCE_PCT <= 0:
                errors.append(f"MIN_STOP_DISTANCE_PCT must be > 0, got {MIN_STOP_DISTANCE_PCT*100:.2f}%")
            elif MIN_STOP_DISTANCE_PCT < 0.003:  # Less than 0.3%
                warnings.append(f"MIN_STOP_DISTANCE_PCT is very tight ({MIN_STOP_DISTANCE_PCT*100:.2f}%). May hit stops frequently.")
        
        # Risk per trade
        if MAX_RISK_PER_TRADE_PCT is not None:
            if MAX_RISK_PER_TRADE_PCT <= 0:
                errors.append(f"MAX_RISK_PER_TRADE_PCT must be > 0, got {MAX_RISK_PER_TRADE_PCT}")
            elif MAX_RISK_PER_TRADE_PCT > 3.0:
                warnings.append(f"MAX_RISK_PER_TRADE_PCT is very high ({MAX_RISK_PER_TRADE_PCT}%). Extreme risk.")
        
        # Signal score threshold
        if SIGNAL_MIN_SCORE_ENTRY is not None:
            if SIGNAL_MIN_SCORE_ENTRY < 0 or SIGNAL_MIN_SCORE_ENTRY > 100:
                errors.append(f"SIGNAL_MIN_SCORE_ENTRY must be between 0 and 100, got {SIGNAL_MIN_SCORE_ENTRY}")
            elif SIGNAL_MIN_SCORE_ENTRY < 60:
                warnings.append(f"SIGNAL_MIN_SCORE_ENTRY is low ({SIGNAL_MIN_SCORE_ENTRY}). May take poor quality signals.")
        
        # Mode warnings
        if not DRY_RUN:
            warnings.append("DRY_RUN is disabled. Bot will trade with REAL MONEY.")
        
    except ImportError as e:
        errors.append(f"Failed to import config: {e}")
    except Exception as e:
        errors.append(f"Validation failed: {e}")
    
    return warnings, errors


def validate_and_exit_on_error():
    """
    Validate configuration and exit if errors found.
    
    CRITICAL: During startup, console output is OK (before UI takes over).
    """
    warnings, errors = validate_config()
    
    if not warnings and not errors:
        # Success - no output needed
        return
    
    # STARTUP PHASE: Console output is OK
    # (UI hasn't started yet, so we can print)
    
    if warnings:
        print(f"[CONFIG] ⚠ {len(warnings)} warning(s):")
        for i, msg in enumerate(warnings, 1):
            print(f"  {i}. {msg}")
    
    if errors:
        print(f"[CONFIG] ✗ {len(errors)} error(s):")
        for i, msg in enumerate(errors, 1):
            print(f"  {i}. {msg}")
        print("\n[FATAL] Invalid configuration. Please fix errors and restart.")
        print("=" * 80)
        sys.exit(1)
    
    if warnings and not errors:
        print("[CONFIG] Configuration has warnings but will continue...")
        print("=" * 80)
