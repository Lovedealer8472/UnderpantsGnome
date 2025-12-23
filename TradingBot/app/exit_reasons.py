"""
Centralized Exit Reasons

All exit reasons used throughout the bot must be defined here.
This ensures consistency and makes it easy to track exit patterns.
"""

from enum import Enum
from typing import Dict, Set


class ExitReason(str, Enum):
    """
    Canonical exit reasons for position exits.
    
    Format: CATEGORY_SUBCATEGORY_DETAIL
    Examples:
    - stop_loss_hit
    - take_profit_hit
    - prs_full_exit_25
    - prs_scale_out_45
    - stale_position_timeout
    """
    
    # ────────────────────────────────────────────────────────────────
    # STOP-LOSS / TAKE-PROFIT EXITS
    # ────────────────────────────────────────────────────────────────
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    TRAILING_STOP_HIT = "trailing_stop_hit"
    ATR_TRAILING_STOP_HIT = "atr_trailing_stop_hit"
    TRAILING_PARTIAL_1R = "trailing_partial_1r"
    TRAILING_PARTIAL_2R = "trailing_partial_2r"
    
    # ────────────────────────────────────────────────────────────────
    # PRS (POSITION RECOVERY SCORE) EXITS
    # ────────────────────────────────────────────────────────────────
    PRS_FULL_EXIT = "prs_full_exit"  # Base reason (score appended in code)
    PRS_SCALE_OUT = "prs_scale_out"  # Base reason (score appended in code)
    
    # Format: prs_full_exit_<score> or prs_scale_out_<score>
    # Example: prs_full_exit_25, prs_scale_out_45
    
    # ────────────────────────────────────────────────────────────────
    # TIME-BASED EXITS
    # ────────────────────────────────────────────────────────────────
    STALE_POSITION_TIMEOUT = "stale_position_timeout"
    MAX_AGE_EXCEEDED = "max_age_exceeded"
    STALE_90MIN_EXIT = "stale_90min_exit"
    STALE_DRAWDOWN_RESUME = "stale_drawdown_resume"
    
    # ────────────────────────────────────────────────────────────────
    # RISK MANAGEMENT EXITS
    # ────────────────────────────────────────────────────────────────
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"
    MAX_POSITIONS_REACHED = "max_positions_reached"
    CORRELATION_BLOCK = "correlation_block"
    DRAWDOWN_CIRCUIT_BREAKER = "drawdown_circuit_breaker"
    
    # ────────────────────────────────────────────────────────────────
    # BREAK-EVEN / RESCUE PROTOCOLS
    # ────────────────────────────────────────────────────────────────
    BERP_BREAKEVEN = "berp_breakeven"
    BERP_TIMEOUT = "berp_timeout"
    BREAKEVEN_RESCUE = "breakeven_rescue"
    
    # ────────────────────────────────────────────────────────────────
    # R-BASED EXIT SYSTEM
    # ────────────────────────────────────────────────────────────────
    R_SCALP_TP = "r_scalp_tp"
    R_SCALP_TIME_STOP = "r_scalp_time_stop"
    R_STANDARD_PARTIAL_TP = "r_standard_partial_tp"
    R_STANDARD_TRAILING_STOP = "r_standard_trailing_stop"
    R_STANDARD_TIME_STOP = "r_standard_time_stop"
    R_RUNNER_PARTIAL_TP = "r_runner_partial_tp"
    R_RUNNER_TRAILING_STOP = "r_runner_trailing_stop"
    R_RUNNER_TIME_STOP = "r_runner_time_stop"
    R_BOREDOM_EXIT = "r_boredom_exit"
    
    # ────────────────────────────────────────────────────────────────
    # MSX (MULTI-STAGE EXIT) SYSTEM
    # ────────────────────────────────────────────────────────────────
    MSX_STAGE1_INVALIDATION = "msx_stage1_invalidation"
    MSX_PARTIAL_SCALP = "msx_partial_scalp"
    MSX_PARTIAL_STANDARD = "msx_partial_standard"
    MSX_PARTIAL_RUNNER = "msx_partial_runner"
    MSX_TIME_STOP = "msx_time_stop"
    
    # ────────────────────────────────────────────────────────────────
    # SPREAD / MARKET CONDITION EXITS
    # ────────────────────────────────────────────────────────────────
    WIDE_SPREAD_EXIT = "wide_spread_exit"
    VOLATILITY_SPIKE_EXIT = "volatility_spike_exit"
    LIQUIDITY_EXIT = "liquidity_exit"
    
    # ────────────────────────────────────────────────────────────────
    # REPLACEMENT / SIGNAL MANAGEMENT
    # ────────────────────────────────────────────────────────────────
    REPLACED_BY_BETTER_SIGNAL = "replaced_by_better_signal"
    SCORE_AWARE_REPLACEMENT = "score_aware_replacement"
    
    # ────────────────────────────────────────────────────────────────
    # MANUAL / EXTERNAL EXITS
    # ────────────────────────────────────────────────────────────────
    MANUAL_EXIT = "manual_exit"
    LLM_OVERRIDE = "llm_override"
    EXTERNAL_SIGNAL = "external_signal"
    
    # ────────────────────────────────────────────────────────────────
    # ERROR / SAFETY EXITS
    # ────────────────────────────────────────────────────────────────
    EXCHANGE_ERROR = "exchange_error"
    DATA_STALE = "data_stale"
    INVALID_POSITION = "invalid_position"
    SAFETY_EXIT = "safety_exit"
    
    # ────────────────────────────────────────────────────────────────
    # LEGACY / FALLBACK
    # ────────────────────────────────────────────────────────────────
    UNKNOWN = "unknown"
    CLOSED = "closed"  # Generic close (should be replaced with specific reason)
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def is_prs_reason(cls, reason: str) -> bool:
        """Check if reason is PRS-related."""
        return reason.startswith("prs_")
    
    @classmethod
    def is_r_based_reason(cls, reason: str) -> bool:
        """Check if reason is R-based exit system."""
        return reason.startswith("r_")
    
    @classmethod
    def is_msx_reason(cls, reason: str) -> bool:
        """Check if reason is MSX system."""
        return reason.startswith("msx_")
    
    @classmethod
    def normalize_reason(cls, reason: str) -> str:
        """
        Normalize a reason string to a canonical ExitReason value.
        
        Handles legacy reasons and variations.
        Returns the normalized reason string.
        """
        if not reason:
            return cls.UNKNOWN.value
        
        reason_lower = reason.lower()
        
        # Map common variations
        reason_map: Dict[str, str] = {
            "stop_loss": cls.STOP_LOSS_HIT.value,
            "sl": cls.STOP_LOSS_HIT.value,
            "take_profit": cls.TAKE_PROFIT_HIT.value,
            "tp": cls.TAKE_PROFIT_HIT.value,
            "trailing_stop": cls.TRAILING_STOP_HIT.value,
            "manual": cls.MANUAL_EXIT.value,
            "timeout": cls.STALE_POSITION_TIMEOUT.value,
            "stale": cls.STALE_POSITION_TIMEOUT.value,
            "closed": cls.CLOSED.value,
        }
        
        # Check exact match first
        if reason_lower in reason_map:
            return reason_map[reason_lower]
        
        # Check prefix matches for PRS, R-based, MSX
        if reason_lower.startswith("prs_"):
            if "full_exit" in reason_lower:
                return cls.PRS_FULL_EXIT.value
            elif "scale" in reason_lower:
                return cls.PRS_SCALE_OUT.value
        
        if reason_lower.startswith("r_"):
            # R-based reasons are already canonical
            return reason
        
        if reason_lower.startswith("msx_"):
            # MSX reasons are already canonical
            return reason
        
        # Check if it matches any enum value
        for exit_reason in cls:
            if exit_reason.value.lower() == reason_lower:
                return exit_reason.value
        
        # Return original if no match (might be a dynamic reason like "prs_full_exit_25")
        return reason


# Convenience sets for filtering
PRS_REASONS: Set[str] = {
    ExitReason.PRS_FULL_EXIT.value,
    ExitReason.PRS_SCALE_OUT.value,
}

R_BASED_REASONS: Set[str] = {
    ExitReason.R_SCALP_TP.value,
    ExitReason.R_SCALP_TIME_STOP.value,
    ExitReason.R_STANDARD_PARTIAL_TP.value,
    ExitReason.R_STANDARD_TRAILING_STOP.value,
    ExitReason.R_STANDARD_TIME_STOP.value,
    ExitReason.R_RUNNER_PARTIAL_TP.value,
    ExitReason.R_RUNNER_TRAILING_STOP.value,
    ExitReason.R_RUNNER_TIME_STOP.value,
    ExitReason.R_BOREDOM_EXIT.value,
}

STOP_LOSS_REASONS: Set[str] = {
    ExitReason.STOP_LOSS_HIT.value,
    ExitReason.TRAILING_STOP_HIT.value,
    ExitReason.ATR_TRAILING_STOP_HIT.value,
}

TAKE_PROFIT_REASONS: Set[str] = {
    ExitReason.TAKE_PROFIT_HIT.value,
    ExitReason.R_SCALP_TP.value,
    ExitReason.R_STANDARD_PARTIAL_TP.value,
    ExitReason.R_RUNNER_PARTIAL_TP.value,
    ExitReason.TRAILING_PARTIAL_1R.value,
    ExitReason.TRAILING_PARTIAL_2R.value,
}

