"""
Adaptive Weights System.

Maintains and updates weights that control trading behavior.
Weights are adjusted gradually based on lessons from trade introspection.

The key insight: Don't just find "best parameters" but learn
CONTEXT-SPECIFIC adjustments:
- "In high volatility, use wider stops"
- "In overbought RSI, wait before entering"
- "In low volume, skip more trades"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path
import math


@dataclass
class ContextWeights:
    """Weights for a specific market context."""
    
    # Entry timing
    entry_delay_bars: float = 0.0  # How many bars to wait (learned)
    
    # SL/TP multipliers
    sl_multiplier: float = 1.0  # Multiply default SL by this
    tp_multiplier: float = 1.0  # Multiply default TP by this
    
    # Skip probability
    skip_probability: float = 0.0  # 0-1, probability of skipping trade
    
    # Confidence (how much data supports these weights)
    sample_count: int = 0
    confidence: float = 0.0  # 0-1, based on sample count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_delay_bars": self.entry_delay_bars,
            "sl_multiplier": self.sl_multiplier,
            "tp_multiplier": self.tp_multiplier,
            "skip_probability": self.skip_probability,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextWeights":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AdaptiveWeights:
    """
    Central weight management with gradient-based updates.
    
    Maintains context-specific weights that are nudged based on
    lessons learned from trade introspection.
    """
    
    # Learning rates (how fast to adapt)
    ENTRY_DELAY_LR = 0.1  # Learning rate for entry delay
    SL_MULT_LR = 0.05     # Learning rate for SL multiplier
    TP_MULT_LR = 0.05     # Learning rate for TP multiplier
    SKIP_LR = 0.02        # Learning rate for skip probability
    
    # Confidence calculation
    MIN_SAMPLES_FOR_CONFIDENCE = 10
    FULL_CONFIDENCE_SAMPLES = 100
    
    # Bounds
    MAX_ENTRY_DELAY = 10
    MIN_SL_MULT = 0.5
    MAX_SL_MULT = 3.0
    MIN_TP_MULT = 0.5
    MAX_TP_MULT = 3.0
    MAX_SKIP_PROB = 0.35  # Cap at 35% to avoid over-skipping
    
    def __init__(self, store_path: Optional[str] = None):
        if store_path is None:
            base = Path(os.environ.get("LEARNER_DATA_DIR", "data/learner"))
            base.mkdir(parents=True, exist_ok=True)
            store_path = str(base / "weights.json")
        
        self.store_path = Path(store_path)
        
        # Weights by context key (e.g., "overbought_high_down")
        self.weights: Dict[str, ContextWeights] = {}
        
        # Global weights (used when no context-specific weight exists)
        self.global_weights = ContextWeights()
        
        self._load()
    
    def _load(self) -> None:
        """Load weights from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    data = json.load(f)
                
                self.global_weights = ContextWeights.from_dict(data.get("global", {}))
                
                for key, w in data.get("contexts", {}).items():
                    self.weights[key] = ContextWeights.from_dict(w)
                
                print(f"[LEARNER] Loaded weights for {len(self.weights)} contexts")
            except Exception as e:
                print(f"[LEARNER] Warning: Failed to load weights: {e}")
    
    def save(self) -> None:
        """Save weights to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "global": self.global_weights.to_dict(),
            "contexts": {k: v.to_dict() for k, v in self.weights.items()},
        }
        
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_weights(self, context_key: str) -> ContextWeights:
        """
        Get weights for a context.
        
        Returns context-specific weights if available and confident,
        otherwise falls back to global or default.
        """
        if context_key in self.weights:
            ctx_w = self.weights[context_key]
            # Use context weights if we have at least 10 samples (10% confidence)
            # This allows learning to kick in earlier while still being somewhat reliable
            if ctx_w.confidence >= 0.10:
                return ctx_w
        
        # Fall back to global
        if self.global_weights.confidence > 0.05:
            return self.global_weights
        
        # Default (no learning yet)
        return ContextWeights()
    
    def update_from_lesson(self, lesson: "Lesson") -> None:
        """
        Update weights based on a lesson.
        
        This is the core learning function. It nudges weights
        in the direction indicated by the lesson.
        """
        from .lessons import Lesson
        
        context_key = lesson.context.get_bucket_key()
        
        # Get or create context weights
        if context_key not in self.weights:
            self.weights[context_key] = ContextWeights()
        
        w = self.weights[context_key]
        w.sample_count += 1
        
        # Update confidence
        w.confidence = min(1.0, w.sample_count / self.FULL_CONFIDENCE_SAMPLES)
        
        # Adaptive learning rate based on confidence (learn faster early)
        confidence_factor = 2.0 - w.confidence  # Higher when less confident
        
        # Apply gradient updates based on lesson type
        if lesson.optimal_action == "entry_offset":
            # Nudge entry delay toward optimal
            target = lesson.optimal_value
            delta = (target - w.entry_delay_bars) * self.ENTRY_DELAY_LR * confidence_factor
            w.entry_delay_bars = self._clamp(
                w.entry_delay_bars + delta,
                -self.MAX_ENTRY_DELAY,
                self.MAX_ENTRY_DELAY
            )
        
        elif lesson.optimal_action == "sl_mult":
            # Nudge SL multiplier toward optimal
            target = lesson.optimal_value
            delta = (target - w.sl_multiplier) * self.SL_MULT_LR * confidence_factor
            w.sl_multiplier = self._clamp(
                w.sl_multiplier + delta,
                self.MIN_SL_MULT,
                self.MAX_SL_MULT
            )
        
        elif lesson.optimal_action == "tp_mult":
            # Nudge TP multiplier toward optimal
            target = lesson.optimal_value
            delta = (target - w.tp_multiplier) * self.TP_MULT_LR * confidence_factor
            w.tp_multiplier = self._clamp(
                w.tp_multiplier + delta,
                self.MIN_TP_MULT,
                self.MAX_TP_MULT
            )
        
        elif lesson.optimal_action == "hold_bars":
            # Holding longer was better - translate to wider TP
            # If optimal hold was 100+ bars, increase TP multiplier
            hold_bars = lesson.optimal_value
            if hold_bars >= 50:
                # Longer hold = wider TP target
                tp_boost = 1.0 + (hold_bars / 200) * 0.5  # 50 bars -> 1.125x, 200 bars -> 1.5x
                delta = (tp_boost - w.tp_multiplier) * self.TP_MULT_LR * 0.5 * confidence_factor
                w.tp_multiplier = self._clamp(
                    w.tp_multiplier + delta,
                    self.MIN_TP_MULT,
                    self.MAX_TP_MULT
                )
        
        elif lesson.optimal_action == "skip":
            # Increase skip probability
            delta = self.SKIP_LR * confidence_factor
            w.skip_probability = self._clamp(
                w.skip_probability + delta,
                0.0,
                self.MAX_SKIP_PROB
            )
        
        elif lesson.optimal_action == "none":
            # Trade was good or no better alternative
            if lesson.actual_won:
                # Decrease skip probability (reinforce taking good trades)
                w.skip_probability = self._clamp(
                    w.skip_probability - self.SKIP_LR * 0.5 * confidence_factor,
                    0.0,
                    self.MAX_SKIP_PROB
                )
        
        # Also update global weights (with smaller learning rate)
        self._update_global(lesson, confidence_factor * 0.3)
        
        # Save after each update
        self.save()
    
    def _update_global(self, lesson: "Lesson", lr_factor: float) -> None:
        """Update global weights (slower, for fallback)."""
        w = self.global_weights
        w.sample_count += 1
        w.confidence = min(1.0, w.sample_count / self.FULL_CONFIDENCE_SAMPLES)
        
        if lesson.optimal_action == "entry_offset":
            target = lesson.optimal_value
            delta = (target - w.entry_delay_bars) * self.ENTRY_DELAY_LR * lr_factor
            w.entry_delay_bars = self._clamp(
                w.entry_delay_bars + delta,
                -self.MAX_ENTRY_DELAY,
                self.MAX_ENTRY_DELAY
            )
        
        elif lesson.optimal_action == "sl_mult":
            target = lesson.optimal_value
            delta = (target - w.sl_multiplier) * self.SL_MULT_LR * lr_factor
            w.sl_multiplier = self._clamp(
                w.sl_multiplier + delta,
                self.MIN_SL_MULT,
                self.MAX_SL_MULT
            )
        
        elif lesson.optimal_action == "tp_mult":
            target = lesson.optimal_value
            delta = (target - w.tp_multiplier) * self.TP_MULT_LR * lr_factor
            w.tp_multiplier = self._clamp(
                w.tp_multiplier + delta,
                self.MIN_TP_MULT,
                self.MAX_TP_MULT
            )
        
        elif lesson.optimal_action == "skip":
            w.skip_probability = self._clamp(
                w.skip_probability + self.SKIP_LR * lr_factor,
                0.0,
                self.MAX_SKIP_PROB
            )
    
    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to bounds."""
        return max(min_val, min(max_val, value))
    
    def apply_to_signal(
        self,
        context_key: str,
        entry_price: float,
        default_sl: float,
        default_tp: float,
    ) -> Dict[str, Any]:
        """
        Apply learned weights to modify a trade setup.
        
        Returns:
            Dict with adjusted parameters and whether to skip.
        """
        w = self.get_weights(context_key)
        
        # Calculate adjusted values
        sl_distance = abs(entry_price - default_sl)
        tp_distance = abs(entry_price - default_tp)
        
        is_long = default_sl < entry_price
        
        if is_long:
            adjusted_sl = entry_price - (sl_distance * w.sl_multiplier)
            adjusted_tp = entry_price + (tp_distance * w.tp_multiplier)
        else:
            adjusted_sl = entry_price + (sl_distance * w.sl_multiplier)
            adjusted_tp = entry_price - (tp_distance * w.tp_multiplier)
        
        # Decide skip based on probability
        import random
        should_skip = random.random() < w.skip_probability
        
        return {
            "entry_delay_bars": int(round(w.entry_delay_bars)),
            "stop_loss": adjusted_sl,
            "take_profit": adjusted_tp,
            "sl_multiplier": w.sl_multiplier,
            "tp_multiplier": w.tp_multiplier,
            "skip": should_skip,
            "skip_probability": w.skip_probability,
            "confidence": w.confidence,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all learned weights."""
        return {
            "global": self.global_weights.to_dict(),
            "contexts": {
                k: {
                    **v.to_dict(),
                    "key": k,
                }
                for k, v in sorted(
                    self.weights.items(),
                    key=lambda x: x[1].sample_count,
                    reverse=True
                )[:20]  # Top 20 by sample count
            },
            "total_contexts": len(self.weights),
            "total_samples": sum(w.sample_count for w in self.weights.values()),
        }
    
    def report(self) -> str:
        """Generate human-readable report of learned weights."""
        lines = [
            "=" * 60,
            "ADAPTIVE WEIGHTS REPORT",
            "=" * 60,
            "",
            f"Global Weights (confidence: {self.global_weights.confidence:.2f}):",
            f"  Entry Delay: {self.global_weights.entry_delay_bars:.1f} bars",
            f"  SL Multiplier: {self.global_weights.sl_multiplier:.2f}x",
            f"  TP Multiplier: {self.global_weights.tp_multiplier:.2f}x",
            f"  Skip Probability: {self.global_weights.skip_probability:.1%}",
            "",
            f"Context-Specific Weights ({len(self.weights)} contexts):",
        ]
        
        # Sort by sample count
        sorted_contexts = sorted(
            self.weights.items(),
            key=lambda x: x[1].sample_count,
            reverse=True
        )
        
        for key, w in sorted_contexts[:10]:  # Top 10
            lines.append(f"\n  {key} ({w.sample_count} samples, {w.confidence:.0%} conf):")
            if abs(w.entry_delay_bars) > 0.5:
                lines.append(f"    Entry: Wait {w.entry_delay_bars:.1f} bars")
            if abs(w.sl_multiplier - 1.0) > 0.1:
                lines.append(f"    SL: {w.sl_multiplier:.2f}x {'wider' if w.sl_multiplier > 1 else 'tighter'}")
            if abs(w.tp_multiplier - 1.0) > 0.1:
                lines.append(f"    TP: {w.tp_multiplier:.2f}x {'wider' if w.tp_multiplier > 1 else 'tighter'}")
            if w.skip_probability > 0.1:
                lines.append(f"    Skip: {w.skip_probability:.0%} of trades")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def reset(self, context_key: Optional[str] = None) -> None:
        """Reset weights (all or specific context)."""
        if context_key:
            if context_key in self.weights:
                del self.weights[context_key]
        else:
            self.weights = {}
            self.global_weights = ContextWeights()
        self.save()
