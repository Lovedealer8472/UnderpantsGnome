"""
Learner Integration - TURBO EDITION

Enhanced with data-driven filters from 2M+ trade analysis:
- Volatility gate (ATR > 1.5% = profitable, < 1% = losing)
- Streak awareness (39% win rate after wins vs 33% baseline)
- Spread filter (tight spread + high vol = +0.097R)
- SL/TP adjustments (2x wider SL, 1.5x wider TP)
"""

import os
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import random
from collections import deque

def is_learner_enabled() -> bool:
    return os.environ.get("LEARNER_ENABLED", "false").lower() in ("true", "1", "yes")

# DATA-DRIVEN THRESHOLDS (from 2M+ trade analysis)
MIN_ATR_PCT = 1.5      # Minimum volatility to trade
MAX_SPREAD_BPS = 20    # Maximum spread to trade
STREAK_BOOST_THRESHOLD = 2  # After 2 wins, boost confidence


class LearnerIntegration:
    _instance = None
    
    def __init__(self):
        self.enabled = is_learner_enabled()
        self.weights = None
        self.introspector = None
        self.recent_outcomes = {}
        self.global_recent_outcomes = deque(maxlen=20)
        self.stats = {
            'trades_approved': 0,
            'trades_rejected_volatility': 0,
            'trades_rejected_spread': 0,
            'streak_boosts_applied': 0,
        }
        if self.enabled:
            self._initialize()
    
    def _initialize(self):
        from .weights import AdaptiveWeights
        from .introspector import TradeIntrospector
        self.weights = AdaptiveWeights()
        self.introspector = TradeIntrospector(verbose=False)
        total_samples = self.weights.global_weights.sample_count
        n_contexts = len(self.weights.weights)
        print(f"[LEARNER] TURBO enabled: {n_contexts} contexts, {total_samples} samples")
        print(f"[LEARNER] Gates: ATR>{MIN_ATR_PCT}%, Spread<{MAX_SPREAD_BPS}bps")
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        cls._instance = None
    
    def check_volatility_gate(self, atr_pct: float) -> Tuple[bool, str]:
        if atr_pct < MIN_ATR_PCT:
            self.stats['trades_rejected_volatility'] += 1
            return False, f"low_vol:ATR={atr_pct:.1f}%<{MIN_ATR_PCT}%"
        return True, "vol_ok"
    
    def check_spread_gate(self, spread_bps: float) -> Tuple[bool, str]:
        if spread_bps > MAX_SPREAD_BPS:
            self.stats['trades_rejected_spread'] += 1
            return False, f"wide_spread:{spread_bps:.0f}bps>{MAX_SPREAD_BPS}bps"
        return True, "spread_ok"
    
    def get_streak_bonus(self, symbol: str = None) -> Tuple[float, int]:
        consecutive_wins = 0
        for outcome in reversed(list(self.global_recent_outcomes)):
            if outcome:
                consecutive_wins += 1
            else:
                break
        if consecutive_wins >= STREAK_BOOST_THRESHOLD:
            self.stats['streak_boosts_applied'] += 1
            bonus = min(1.3, 1.0 + 0.1 * (consecutive_wins - 1))
            return bonus, consecutive_wins
        return 1.0, 0
    
    def record_outcome(self, symbol: str, won: bool):
        if symbol not in self.recent_outcomes:
            self.recent_outcomes[symbol] = deque(maxlen=5)
        self.recent_outcomes[symbol].append(won)
        self.global_recent_outcomes.append(won)
    
    def get_context_key(self, rsi: float, volatility: str, trend: str) -> str:
        if rsi < 30:
            rsi_bucket = "oversold"
        elif rsi < 45:
            rsi_bucket = "neutral_low"
        elif rsi < 55:
            rsi_bucket = "neutral"
        elif rsi < 70:
            rsi_bucket = "neutral_high"
        else:
            rsi_bucket = "overbought"
        return f"{rsi_bucket}_{volatility}_{trend}"
    
    def get_trade_adjustments(self, rsi: float, volatility: str, trend: str,
                              entry_price: float, default_sl: float, default_tp: float,
                              atr_pct: float = None, spread_bps: float = None,
                              symbol: str = None, **kwargs) -> Dict[str, Any]:
        if not self.enabled or not self.weights:
            return {"entry_delay_bars": 0, "stop_loss": default_sl, "take_profit": default_tp,
                    "skip": False, "confidence": 0.0, "context_key": "disabled",
                    "sl_multiplier": 1.0, "tp_multiplier": 1.0}
        
        # TURBO: Quality gates
        if atr_pct is not None:
            vol_ok, vol_reason = self.check_volatility_gate(atr_pct)
            if not vol_ok:
                return {"skip": True, "skip_reason": vol_reason, "stop_loss": default_sl,
                        "take_profit": default_tp, "confidence": 1.0, "context_key": "vol_gate",
                        "sl_multiplier": 1.0, "tp_multiplier": 1.0}
        
        if spread_bps is not None:
            spread_ok, spread_reason = self.check_spread_gate(spread_bps)
            if not spread_ok:
                return {"skip": True, "skip_reason": spread_reason, "stop_loss": default_sl,
                        "take_profit": default_tp, "confidence": 1.0, "context_key": "spread_gate",
                        "sl_multiplier": 1.0, "tp_multiplier": 1.0}
        
        self.stats['trades_approved'] += 1
        
        # Determine volatility bucket
        if atr_pct is not None:
            if atr_pct < 1.0:
                volatility = "low"
            elif atr_pct < 2.5:
                volatility = "medium"
            else:
                volatility = "high"
        
        context_key = self.get_context_key(rsi, volatility, trend)
        context_weights = self.weights.get_weights(context_key)
        
        sl_distance = abs(entry_price - default_sl)
        tp_distance = abs(entry_price - default_tp)
        is_long = default_sl < entry_price
        
        # TURBO: Streak bonus on TP
        streak_bonus, wins = self.get_streak_bonus(symbol)
        tp_mult = context_weights.tp_multiplier * streak_bonus
        
        if is_long:
            adjusted_sl = entry_price - (sl_distance * context_weights.sl_multiplier)
            adjusted_tp = entry_price + (tp_distance * tp_mult)
        else:
            adjusted_sl = entry_price + (sl_distance * context_weights.sl_multiplier)
            adjusted_tp = entry_price - (tp_distance * tp_mult)
        
        should_skip = random.random() < context_weights.skip_probability
        
        return {"entry_delay_bars": int(round(context_weights.entry_delay_bars)),
                "stop_loss": adjusted_sl, "take_profit": adjusted_tp,
                "skip": should_skip, "skip_probability": context_weights.skip_probability,
                "confidence": context_weights.confidence, "context_key": context_key,
                "sl_multiplier": context_weights.sl_multiplier, "tp_multiplier": tp_mult,
                "streak_bonus": streak_bonus}
    
    def should_skip_trade(self, rsi: float, volatility: str, trend: str,
                          atr_pct: float = None, spread_bps: float = None,
                          symbol: str = None, **kwargs) -> Tuple[bool, float, str]:
        if not self.enabled:
            return False, 0.0, "learner_disabled"
        
        # TURBO gates
        if atr_pct is not None:
            vol_ok, vol_reason = self.check_volatility_gate(atr_pct)
            if not vol_ok:
                return True, 1.0, vol_reason
        
        if spread_bps is not None:
            spread_ok, spread_reason = self.check_spread_gate(spread_bps)
            if not spread_ok:
                return True, 1.0, spread_reason
        
        if not self.weights:
            return False, 0.0, "no_weights"
        
        context_key = self.get_context_key(rsi, volatility, trend)
        context_weights = self.weights.get_weights(context_key)
        
        if context_weights.confidence < 0.2:
            return False, 0.0, "low_confidence"
        
        should_skip = random.random() < context_weights.skip_probability
        if should_skip:
            return True, context_weights.skip_probability, f"learned_skip:{context_key}"
        return False, context_weights.skip_probability, "no_skip"
    
    def record_trade_outcome(self, trade: Dict, ohlcv, entry_bar_idx: int):
        if not self.enabled:
            return
        symbol = trade.get('symbol', 'UNKNOWN')
        won = trade.get('won', False)
        self.record_outcome(symbol, won)
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.enabled or not self.weights:
            return {"enabled": False}
        return {"enabled": True, "global_samples": self.weights.global_weights.sample_count,
                "contexts_learned": len(self.weights.weights),
                "turbo_stats": self.stats}
    
    def report(self) -> str:
        if not self.enabled:
            return "[LEARNER] Disabled"
        lines = [f"[LEARNER] TURBO: approved={self.stats['trades_approved']}, "
                 f"rej_vol={self.stats['trades_rejected_volatility']}, "
                 f"rej_spread={self.stats['trades_rejected_spread']}, "
                 f"streak_boosts={self.stats['streak_boosts_applied']}"]
        return "\n".join(lines)


def get_learner():
    return LearnerIntegration.get_instance()

def is_learner_enabled():
    return os.environ.get("LEARNER_ENABLED", "false").lower() in ("true", "1", "yes")

def get_trade_adjustments(rsi, volatility, trend, entry_price, default_sl, default_tp):
    return get_learner().get_trade_adjustments(rsi, volatility, trend, entry_price, default_sl, default_tp)
