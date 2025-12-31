"""
Integration module to add ML Confidence filtering to position manager.

This file patches the position manager to check ML confidence before entering.
"""

from typing import Optional, Dict, Any
from .ml_confidence_filter import get_confidence_filter, ConfidenceResult
from .ml_confidence_config import ALLOW_ENTRY_WITHOUT_ML_CONFIDENCE, USE_ML_CONFIDENCE_FILTER


def apply_ml_confidence_filter(
    symbol: str,
    signal_data: Dict[str, Any],
    position_count: int
) -> tuple[bool, Optional[str]]:
    """
    Apply ML confidence filter to signal.
    
    Returns:
        (passed: bool, reason: Optional[str])
        - passed=True: Signal passed confidence check, can enter
        - passed=False: reason explains why rejected
    """
    # BYPASS: If ML confidence filter is disabled, always pass
    if not USE_ML_CONFIDENCE_FILTER:
        return True, None
    
    confidence_filter = get_confidence_filter()
    
    # Extract model probabilities from signal data
    # These come from the ML scorer's output
    ml_data = signal_data.get('ml_data', {})
    
    xgb_prob = ml_data.get('xgboost_prob')
    lgb_prob = ml_data.get('lightgbm_prob')
    rf_prob = ml_data.get('random_forest_prob')
    
    signal_score = signal_data.get('final_score', 0)
    
    # BYPASS: If no probabilities and fallback is allowed, pass
    if xgb_prob is None and lgb_prob is None and rf_prob is None:
        if ALLOW_ENTRY_WITHOUT_ML_CONFIDENCE:
            # Allow entry using score-based filtering (ML probabilities not available)
            return True, None
        # else: fall through to rejection below
    
    # Check confidence
    result = confidence_filter.check_confidence(
        symbol=symbol,
        xgboost_prob=xgb_prob,
        lightgbm_prob=lgb_prob,
        random_forest_prob=rf_prob,
        position_count=position_count,
        signal_score=signal_score
    )
    
    # Log the decision
    confidence_filter.log_confidence_decision(symbol, result, signal_score)
    
    # Return pass/fail
    if result.passed:
        return True, None
    else:
        return False, f"ml_confidence:{result.reason}"

