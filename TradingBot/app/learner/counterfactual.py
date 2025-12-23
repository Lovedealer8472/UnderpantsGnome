"""
Counterfactual Simulator - Tests "what if" scenarios for each trade.

For each trade, simulates:
- Different entry timings (what if I entered N bars earlier/later?)
- Different stop-loss levels (what if SL was tighter/wider?)
- Different take-profit levels (what if TP was more/less ambitious?)
- Extended holds (what if I held longer?)
- Skip decision (should I have skipped this trade entirely?)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CounterfactualResult:
    """Result of a single counterfactual simulation."""
    
    cf_type: str  # "entry_offset", "sl_mult", "tp_mult", "hold_bars", "skip"
    cf_value: float  # The counterfactual parameter value
    
    # Outcome
    pnl: float
    pnl_pct: float
    r_multiple: float
    won: bool
    
    # Execution details
    entry_price: float
    exit_price: float
    exit_reason: str  # "sl_hit", "tp_hit", "time_exit", "skipped"
    bars_held: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cf_type": self.cf_type,
            "cf_value": float(self.cf_value),
            "pnl": float(self.pnl),
            "pnl_pct": float(self.pnl_pct),
            "r_multiple": float(self.r_multiple),
            "won": bool(self.won),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "exit_reason": self.exit_reason,
            "bars_held": int(self.bars_held),
        }


@dataclass
class CounterfactualAnalysis:
    """Complete analysis of all counterfactuals for a trade."""
    
    # The actual trade outcome
    actual_pnl: float
    actual_r: float
    actual_won: bool
    
    # All counterfactual results
    entry_offsets: List[CounterfactualResult] = field(default_factory=list)
    sl_variations: List[CounterfactualResult] = field(default_factory=list)
    tp_variations: List[CounterfactualResult] = field(default_factory=list)
    hold_variations: List[CounterfactualResult] = field(default_factory=list)
    skip_result: Optional[CounterfactualResult] = None
    
    # Best alternatives found
    best_entry_offset: Optional[CounterfactualResult] = None
    best_sl: Optional[CounterfactualResult] = None
    best_tp: Optional[CounterfactualResult] = None
    best_hold: Optional[CounterfactualResult] = None
    
    # Summary
    optimal_pnl: float = 0.0
    improvement_potential: float = 0.0  # How much better could it have been
    should_have_skipped: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "actual_pnl": self.actual_pnl,
            "actual_r": self.actual_r,
            "actual_won": self.actual_won,
            "optimal_pnl": self.optimal_pnl,
            "improvement_potential": self.improvement_potential,
            "should_have_skipped": self.should_have_skipped,
            "best_entry_offset": self.best_entry_offset.to_dict() if self.best_entry_offset else None,
            "best_sl": self.best_sl.to_dict() if self.best_sl else None,
            "best_tp": self.best_tp.to_dict() if self.best_tp else None,
            "best_hold": self.best_hold.to_dict() if self.best_hold else None,
        }


class CounterfactualSimulator:
    """
    Simulates alternative trade outcomes with full hindsight.
    
    Given a trade and the surrounding market data, tests:
    - What if entry was earlier/later?
    - What if SL/TP was different?
    - What if we held longer?
    - Should we have skipped?
    """
    
    # Default counterfactual parameters
    ENTRY_OFFSETS = list(range(-10, 11))  # -10 to +10 bars
    SL_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    TP_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    HOLD_BARS = [10, 20, 50, 100, 200]  # How long to hold if ignoring SL/TP
    
    def __init__(
        self,
        entry_offsets: Optional[List[int]] = None,
        sl_multipliers: Optional[List[float]] = None,
        tp_multipliers: Optional[List[float]] = None,
        hold_bars: Optional[List[int]] = None,
        max_hold_bars: int = 500,
    ):
        self.entry_offsets = entry_offsets or self.ENTRY_OFFSETS
        self.sl_multipliers = sl_multipliers or self.SL_MULTIPLIERS
        self.tp_multipliers = tp_multipliers or self.TP_MULTIPLIERS
        self.hold_bars = hold_bars or self.HOLD_BARS
        self.max_hold_bars = max_hold_bars
    
    def analyze_trade(
        self,
        trade: Dict[str, Any],
        ohlcv: np.ndarray,  # Shape: (N, 5) - open, high, low, close, volume
        entry_bar_idx: int,
        fees_pct: float = 0.1,  # Total round-trip fees as percentage
    ) -> CounterfactualAnalysis:
        """
        Analyze a trade with all counterfactuals.
        
        Args:
            trade: Trade dict with entry_price, exit_price, side, sl, tp, etc.
            ohlcv: Full OHLCV data around the trade
            entry_bar_idx: Index in ohlcv where original entry occurred
            fees_pct: Round-trip fees as percentage (0.1 = 0.1%)
        
        Returns:
            CounterfactualAnalysis with all alternatives tested
        """
        side = trade.get("side", "long")
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]
        original_sl = trade.get("stop_loss", entry_price * (0.98 if side == "long" else 1.02))
        original_tp = trade.get("take_profit", entry_price * (1.04 if side == "long" else 0.96))
        
        # Calculate original R (risk per share)
        original_r = abs(entry_price - original_sl)
        if original_r == 0:
            original_r = entry_price * 0.01  # Default 1% risk
        
        # Actual outcome
        if side == "long":
            actual_pnl_pct = (exit_price - entry_price) / entry_price * 100 - fees_pct
        else:
            actual_pnl_pct = (entry_price - exit_price) / entry_price * 100 - fees_pct
        
        actual_r = actual_pnl_pct / (original_r / entry_price * 100) if original_r > 0 else 0
        actual_won = actual_pnl_pct > 0
        
        analysis = CounterfactualAnalysis(
            actual_pnl=actual_pnl_pct,
            actual_r=actual_r,
            actual_won=actual_won,
        )
        
        # Test entry timing offsets
        for offset in self.entry_offsets:
            result = self._simulate_entry_offset(
                ohlcv, entry_bar_idx, offset, side,
                original_sl, original_tp, original_r, fees_pct
            )
            if result:
                analysis.entry_offsets.append(result)
                if analysis.best_entry_offset is None or result.pnl > analysis.best_entry_offset.pnl:
                    analysis.best_entry_offset = result
        
        # Test SL variations
        for mult in self.sl_multipliers:
            result = self._simulate_sl_variation(
                ohlcv, entry_bar_idx, side, entry_price,
                original_sl, original_tp, mult, original_r, fees_pct
            )
            if result:
                analysis.sl_variations.append(result)
                if analysis.best_sl is None or result.pnl > analysis.best_sl.pnl:
                    analysis.best_sl = result
        
        # Test TP variations
        for mult in self.tp_multipliers:
            result = self._simulate_tp_variation(
                ohlcv, entry_bar_idx, side, entry_price,
                original_sl, original_tp, mult, original_r, fees_pct
            )
            if result:
                analysis.tp_variations.append(result)
                if analysis.best_tp is None or result.pnl > analysis.best_tp.pnl:
                    analysis.best_tp = result
        
        # Test extended holds (ignore SL/TP)
        for hold in self.hold_bars:
            result = self._simulate_hold(
                ohlcv, entry_bar_idx, side, entry_price,
                hold, original_r, fees_pct
            )
            if result:
                analysis.hold_variations.append(result)
                if analysis.best_hold is None or result.pnl > analysis.best_hold.pnl:
                    analysis.best_hold = result
        
        # Test skip (PnL = 0, no risk)
        analysis.skip_result = CounterfactualResult(
            cf_type="skip",
            cf_value=1.0,
            pnl=0.0,
            pnl_pct=0.0,
            r_multiple=0.0,
            won=False,
            entry_price=0,
            exit_price=0,
            exit_reason="skipped",
            bars_held=0,
        )
        
        # Find optimal and calculate improvement potential
        all_pnls = [actual_pnl_pct]
        if analysis.best_entry_offset:
            all_pnls.append(analysis.best_entry_offset.pnl)
        if analysis.best_sl:
            all_pnls.append(analysis.best_sl.pnl)
        if analysis.best_tp:
            all_pnls.append(analysis.best_tp.pnl)
        if analysis.best_hold:
            all_pnls.append(analysis.best_hold.pnl)
        all_pnls.append(0.0)  # Skip option
        
        analysis.optimal_pnl = max(all_pnls)
        analysis.improvement_potential = analysis.optimal_pnl - actual_pnl_pct
        analysis.should_have_skipped = (0.0 > actual_pnl_pct and 
                                         0.0 >= analysis.optimal_pnl)
        
        return analysis
    
    def _simulate_entry_offset(
        self,
        ohlcv: np.ndarray,
        original_entry_idx: int,
        offset: int,
        side: str,
        original_sl: float,
        original_tp: float,
        original_r: float,
        fees_pct: float,
    ) -> Optional[CounterfactualResult]:
        """Simulate entering at a different bar."""
        new_entry_idx = original_entry_idx + offset
        
        # Check bounds
        if new_entry_idx < 0 or new_entry_idx >= len(ohlcv):
            return None
        
        # Entry at close of the offset bar
        entry_price = ohlcv[new_entry_idx, 3]  # Close
        
        # Adjust SL/TP relative to new entry
        sl_distance = abs(original_sl - ohlcv[original_entry_idx, 3])
        tp_distance = abs(original_tp - ohlcv[original_entry_idx, 3])
        
        if side == "long":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        # Simulate trade from new entry
        exit_price, exit_reason, bars_held = self._simulate_trade(
            ohlcv, new_entry_idx, side, entry_price, sl, tp
        )
        
        if exit_price is None:
            return None
        
        # Calculate PnL
        if side == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100 - fees_pct
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100 - fees_pct
        
        r_mult = pnl_pct / (sl_distance / entry_price * 100) if sl_distance > 0 else 0
        
        return CounterfactualResult(
            cf_type="entry_offset",
            cf_value=float(offset),
            pnl=pnl_pct,
            pnl_pct=pnl_pct,
            r_multiple=r_mult,
            won=pnl_pct > 0,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            bars_held=bars_held,
        )
    
    def _simulate_sl_variation(
        self,
        ohlcv: np.ndarray,
        entry_idx: int,
        side: str,
        entry_price: float,
        original_sl: float,
        original_tp: float,
        sl_mult: float,
        original_r: float,
        fees_pct: float,
    ) -> Optional[CounterfactualResult]:
        """Simulate with different stop-loss."""
        sl_distance = abs(entry_price - original_sl) * sl_mult
        
        if side == "long":
            new_sl = entry_price - sl_distance
        else:
            new_sl = entry_price + sl_distance
        
        exit_price, exit_reason, bars_held = self._simulate_trade(
            ohlcv, entry_idx, side, entry_price, new_sl, original_tp
        )
        
        if exit_price is None:
            return None
        
        if side == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100 - fees_pct
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100 - fees_pct
        
        r_mult = pnl_pct / (sl_distance / entry_price * 100) if sl_distance > 0 else 0
        
        return CounterfactualResult(
            cf_type="sl_mult",
            cf_value=sl_mult,
            pnl=pnl_pct,
            pnl_pct=pnl_pct,
            r_multiple=r_mult,
            won=pnl_pct > 0,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            bars_held=bars_held,
        )
    
    def _simulate_tp_variation(
        self,
        ohlcv: np.ndarray,
        entry_idx: int,
        side: str,
        entry_price: float,
        original_sl: float,
        original_tp: float,
        tp_mult: float,
        original_r: float,
        fees_pct: float,
    ) -> Optional[CounterfactualResult]:
        """Simulate with different take-profit."""
        tp_distance = abs(entry_price - original_tp) * tp_mult
        
        if side == "long":
            new_tp = entry_price + tp_distance
        else:
            new_tp = entry_price - tp_distance
        
        exit_price, exit_reason, bars_held = self._simulate_trade(
            ohlcv, entry_idx, side, entry_price, original_sl, new_tp
        )
        
        if exit_price is None:
            return None
        
        sl_distance = abs(entry_price - original_sl)
        if side == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100 - fees_pct
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100 - fees_pct
        
        r_mult = pnl_pct / (sl_distance / entry_price * 100) if sl_distance > 0 else 0
        
        return CounterfactualResult(
            cf_type="tp_mult",
            cf_value=tp_mult,
            pnl=pnl_pct,
            pnl_pct=pnl_pct,
            r_multiple=r_mult,
            won=pnl_pct > 0,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            bars_held=bars_held,
        )
    
    def _simulate_hold(
        self,
        ohlcv: np.ndarray,
        entry_idx: int,
        side: str,
        entry_price: float,
        hold_bars: int,
        original_r: float,
        fees_pct: float,
    ) -> Optional[CounterfactualResult]:
        """Simulate holding for a fixed number of bars (ignore SL/TP)."""
        exit_idx = min(entry_idx + hold_bars, len(ohlcv) - 1)
        
        if exit_idx <= entry_idx:
            return None
        
        exit_price = ohlcv[exit_idx, 3]  # Close
        bars_held = exit_idx - entry_idx
        
        if side == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100 - fees_pct
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100 - fees_pct
        
        r_mult = pnl_pct / (original_r / entry_price * 100) if original_r > 0 else 0
        
        return CounterfactualResult(
            cf_type="hold_bars",
            cf_value=float(hold_bars),
            pnl=pnl_pct,
            pnl_pct=pnl_pct,
            r_multiple=r_mult,
            won=pnl_pct > 0,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason="time_exit",
            bars_held=bars_held,
        )
    
    def _simulate_trade(
        self,
        ohlcv: np.ndarray,
        entry_idx: int,
        side: str,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> Tuple[Optional[float], str, int]:
        """
        Simulate a trade through OHLCV data.
        
        Returns:
            (exit_price, exit_reason, bars_held)
        """
        for i in range(entry_idx + 1, min(entry_idx + self.max_hold_bars, len(ohlcv))):
            high = ohlcv[i, 1]
            low = ohlcv[i, 2]
            close = ohlcv[i, 3]
            
            if side == "long":
                # Check SL hit (low touches SL)
                if low <= sl:
                    return sl, "sl_hit", i - entry_idx
                # Check TP hit (high touches TP)
                if high >= tp:
                    return tp, "tp_hit", i - entry_idx
            else:  # short
                # Check SL hit (high touches SL)
                if high >= sl:
                    return sl, "sl_hit", i - entry_idx
                # Check TP hit (low touches TP)
                if low <= tp:
                    return tp, "tp_hit", i - entry_idx
        
        # Time exit at max hold
        final_idx = min(entry_idx + self.max_hold_bars, len(ohlcv) - 1)
        return ohlcv[final_idx, 3], "time_exit", final_idx - entry_idx
