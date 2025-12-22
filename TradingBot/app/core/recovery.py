"""
Recovery Module - Safe, rate-limited PRS (Position Recovery Score) engine.
Handles position recovery evaluation and actions with proper guardrails.
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..config import PRS_MIN_AGE_MIN
from ..logger import get_logger
from ..exit_manager import ExitManager

logger = get_logger("RecoveryModule")


@dataclass
class PRSState:
    """PRS state for a position."""
    last_action: Optional[str] = None  # "full_exit", "scale_out", or None
    last_update: float = 0.0
    cooldown: float = 180.0  # 3 minutes cooldown


@dataclass
class PRSAction:
    """PRS action recommendation."""
    action_type: str  # "full_exit", "scale_out", or None
    reason: str
    exit_size_ratio: float = 1.0  # 1.0 = full exit, 0.5 = 50% exit
    score: float = 0.0


class RecoveryModule:
    """
    Safe, rate-limited PRS engine.
    
    Responsibilities:
    - Evaluate position recovery score
    - Respect cooldown periods
    - Act only on fresh indicators
    - Take ONE action per deterioration event
    - Reset on recovery
    - Log once per action
    - Never block or destabilize exit engine
    """
    
    def __init__(self, exit_manager: ExitManager):
        """
        Initialize recovery module.
        
        Args:
            exit_manager: Exit manager for recovery score computation
        """
        self.exit_manager = exit_manager
        self.logger = get_logger("RecoveryModule")
        
        # Statistics
        self.prs_evaluations = 0
        self.prs_actions_taken = 0
        self.prs_skipped_stale = 0
        self.prs_skipped_cooldown = 0
    
    def evaluate_position(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        now: float,
        trend5: int = 0,
        trend15: int = 0,
        vol_regime: str = "normal"
    ) -> Optional[PRSAction]:
        """
        Evaluate position recovery and return action if needed.
        
        Args:
            symbol: Trading symbol
            position: Position dictionary
            current_price: Current market price
            now: Current timestamp
            trend5: 5m trend direction
            trend15: 15m trend direction
            vol_regime: Volatility regime
        
        Returns:
            PRSAction if action needed, None otherwise
        """
        entry_time = position.get('entry_time', now)
        age_minutes = (now - entry_time) / 60.0 if entry_time > 0 else 0.0
        
        # Guardrail: PRS only active after minimum age
        if age_minutes < PRS_MIN_AGE_MIN:
            return None
        
        # Guardrail: Check data freshness
        price_age = now - position.get('last_price_update', entry_time)
        if price_age > 300:  # 5 minutes stale
            self.prs_skipped_stale += 1
            self.logger.debug(f"[PRS] {symbol}: Skipping - price data stale ({price_age:.0f}s)")
            return None
        
        # Guardrail: Check position size
        position_size = position.get('size', 0)
        if position_size <= 0:
            return None
        
        # Guardrail: Skip if position is too small (noise)
        min_position_size = 0.001  # Minimum position size threshold
        if position_size < min_position_size:
            return None
        
        # Compute recovery score
        recovery_score = self.exit_manager.compute_recovery_score(
            position, current_price, trend5=trend5, trend15=trend15, vol_regime=vol_regime
        )
        
        # Store recovery score for UI/logging
        position['recovery_score'] = recovery_score
        position['age_minutes'] = age_minutes
        
        self.prs_evaluations += 1
        
        # Get or create PRS state
        prs_state_dict = position.setdefault('prs_state', {})
        prs_state = PRSState(
            last_action=prs_state_dict.get('last_action'),
            last_update=prs_state_dict.get('last_update', 0),
            cooldown=prs_state_dict.get('cooldown', 180.0)
        )
        
        # Check cooldown
        time_since_last = now - prs_state.last_update
        if time_since_last < prs_state.cooldown:
            self.prs_skipped_cooldown += 1
            return None
        
        # Determine action based on recovery score
        action = None
        
        # ONE-SHOT FULL EXIT (PRS < 30)
        if recovery_score < 30 and prs_state.last_action != "full_exit":
            action = PRSAction(
                action_type="full_exit",
                reason=f"prs_full_exit_{int(recovery_score)}",
                exit_size_ratio=1.0,
                score=recovery_score
            )
            prs_state.last_action = "full_exit"
            prs_state.last_update = now
        
        # ONE-SHOT SCALE OUT (PRS < 50) - DISABLED if DISABLE_PARTIAL_EXITS
        elif recovery_score < 50 and prs_state.last_action != "scale_out":
            # Check if partial exits are disabled
            from ..config import DISABLE_PARTIAL_EXITS
            if DISABLE_PARTIAL_EXITS:
                # Force full exit instead of scale-out (prevents dust positions)
                action = PRSAction(
                    action_type="full_exit",
                    reason=f"prs_full_exit_{int(recovery_score)}_no_partials",
                    exit_size_ratio=1.0,
                    score=recovery_score
                )
                prs_state.last_action = "full_exit"
            else:
                # Ensure minimum position size after scale-out
                min_size_after_scale = position_size * 0.1
                if position_size > min_size_after_scale:
                    action = PRSAction(
                        action_type="scale_out",
                        reason=f"prs_scale_out_{int(recovery_score)}",
                        exit_size_ratio=0.5,
                        score=recovery_score
                    )
                    prs_state.last_action = "scale_out"
                else:
                    # Position too small for scale-out, force full exit
                    action = PRSAction(
                        action_type="full_exit",
                        reason=f"prs_full_exit_{int(recovery_score)}_too_small",
                        exit_size_ratio=1.0,
                        score=recovery_score
                    )
                    prs_state.last_action = "full_exit"
            prs_state.last_update = now
        
        # RESET PRS WHEN RECOVERED
        elif recovery_score >= 50:
            prs_state.last_action = None
        
        # Update PRS state in position
        prs_state_dict['last_action'] = prs_state.last_action
        prs_state_dict['last_update'] = prs_state.last_update
        prs_state_dict['cooldown'] = prs_state.cooldown
        
        if action:
            self.prs_actions_taken += 1
            self.logger.info(
                f"[PRS] {symbol}: {action.action_type} | score={recovery_score:.1f} | "
                f"age={age_minutes:.1f}m"
            )
        
        return action
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get PRS statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'prs_evaluations': self.prs_evaluations,
            'prs_actions_taken': self.prs_actions_taken,
            'prs_skipped_stale': self.prs_skipped_stale,
            'prs_skipped_cooldown': self.prs_skipped_cooldown
        }

