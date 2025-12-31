"""
ML Confidence-Based Entry Filter

Instead of percentile/score filtering, only enter trades when the ML model
is CONFIDENT it will win. Uses prediction probabilities from ensemble models.

Key Principle: "Only pick guaranteed winners, not top-ranked trash"
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Result of confidence check."""
    passed: bool
    confidence: float  # 0.0-1.0
    reason: str
    min_required: float
    position_count: int


class MLConfidenceFilter:
    """
    Filter entries based on ML model's confidence in prediction.
    
    Uses prediction probabilities from ensemble (XGBoost, LightGBM, RandomForest)
    to determine how confident the models are that a trade will win.
    
    Only enters when ALL models agree with high confidence.
    """
    
    def __init__(self, min_base_confidence: float = 0.80):
        """
        Initialize confidence filter.
        
        Args:
            min_base_confidence: Base minimum confidence (80% when 0 positions)
        """
        self.min_base_confidence = min_base_confidence
        
        # Adaptive thresholds based on position count
        # Higher position count = require higher confidence
        self.confidence_by_position_count = {
            0: 0.80,  # Cold market: take good opportunities
            1: 0.85,  # 1 position: normal filter
            2: 0.90,  # 2 positions: getting strict
            3: 0.95,  # Full/near-full: only slam dunks
        }
    
    def get_min_confidence(self, position_count: int) -> float:
        """Get minimum required confidence for current position count."""
        if position_count in self.confidence_by_position_count:
            return self.confidence_by_position_count[position_count]
        # For positions > 3, use maximum (0.95)
        return self.confidence_by_position_count[3]
    
    def check_confidence(
        self,
        symbol: str,
        xgboost_prob: Optional[float],
        lightgbm_prob: Optional[float],
        random_forest_prob: Optional[float],
        position_count: int,
        signal_score: float
    ) -> ConfidenceResult:
        """
        Check if models are confident enough to enter.
        
        Args:
            symbol: Trading pair
            xgboost_prob: XGBoost prediction probability (0-1)
            lightgbm_prob: LightGBM prediction probability (0-1)
            random_forest_prob: RandomForest prediction probability (0-1)
            position_count: Current open positions
            signal_score: Original signal score (for logging)
        
        Returns:
            ConfidenceResult with passed/failed and reasoning
        """
        min_required = self.get_min_confidence(position_count)
        
        # Collect available probabilities
        probs = []
        if xgboost_prob is not None:
            probs.append(xgboost_prob)
        if lightgbm_prob is not None:
            probs.append(lightgbm_prob)
        if random_forest_prob is not None:
            probs.append(random_forest_prob)
        
        # No probabilities available - cannot enter
        if not probs:
            return ConfidenceResult(
                passed=False,
                confidence=0.0,
                reason="no_model_probabilities",
                min_required=min_required,
                position_count=position_count
            )
        
        # Calculate ensemble confidence (average of available models)
        confidence = sum(probs) / len(probs)
        
        # Check if all models agree (low variance = high confidence)
        model_variance = max(probs) - min(probs) if len(probs) > 1 else 0.0
        
        # Models must agree (variance < 10%) AND average confidence high enough
        models_agree = model_variance < 0.10
        passes_threshold = confidence >= min_required
        
        if not models_agree:
            return ConfidenceResult(
                passed=False,
                confidence=confidence,
                reason=f"models_disagree (variance={model_variance:.3f})",
                min_required=min_required,
                position_count=position_count
            )
        
        if not passes_threshold:
            return ConfidenceResult(
                passed=False,
                confidence=confidence,
                reason=f"low_confidence ({confidence:.3f} < {min_required:.3f})",
                min_required=min_required,
                position_count=position_count
            )
        
        # PASSED: All models agree and confidence is high
        return ConfidenceResult(
            passed=True,
            confidence=confidence,
            reason=f"high_confidence ({confidence:.3f} >= {min_required:.3f}, variance={model_variance:.3f})",
            min_required=min_required,
            position_count=position_count
        )
    
    def log_confidence_decision(
        self,
        symbol: str,
        result: ConfidenceResult,
        signal_score: float
    ) -> None:
        """Log confidence decision for debugging."""
        status = "ACCEPT" if result.passed else "REJECT"
        logger.info(
            f"[ML_CONFIDENCE] {status} {symbol} | "
            f"Confidence: {result.confidence:.3f} | "
            f"Required: {result.min_required:.3f} | "
            f"Positions: {result.position_count} | "
            f"Reason: {result.reason} | "
            f"Signal Score: {signal_score:.1f}"
        )


# Singleton instance
_confidence_filter = None


def get_confidence_filter() -> MLConfidenceFilter:
    """Get or create singleton confidence filter."""
    global _confidence_filter
    if _confidence_filter is None:
        _confidence_filter = MLConfidenceFilter()
    return _confidence_filter

