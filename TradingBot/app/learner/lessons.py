"""
Lesson Extraction and Storage.

Extracts actionable lessons from counterfactual analysis:
- "In this context, I should have waited 3 bars"
- "In this context, SL was too tight"
- "In this context, I should have skipped"
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path


@dataclass
class MarketContext:
    """Market conditions at time of trade."""
    
    # Trend indicators
    trend: str = "unknown"  # "up", "down", "sideways"
    trend_strength: float = 0.0  # 0-1
    
    # Momentum
    rsi: float = 50.0
    rsi_bucket: str = "neutral"  # "oversold", "neutral", "overbought"
    momentum_1h: float = 0.0
    momentum_4h: float = 0.0
    momentum_24h: float = 0.0
    
    # Volatility
    volatility: str = "medium"  # "low", "medium", "high"
    atr_pct: float = 0.0
    
    # Volume
    volume_ratio: float = 1.0  # Current vs average
    
    # Price action
    distance_from_support: float = 0.0
    distance_from_resistance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": self.trend,
            "trend_strength": float(self.trend_strength),
            "rsi": float(self.rsi),
            "rsi_bucket": self.rsi_bucket,
            "momentum_1h": float(self.momentum_1h),
            "momentum_4h": float(self.momentum_4h),
            "momentum_24h": float(self.momentum_24h),
            "volatility": self.volatility,
            "atr_pct": float(self.atr_pct),
            "volume_ratio": float(self.volume_ratio),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketContext":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def get_bucket_key(self) -> str:
        """Get a key for bucketing similar contexts."""
        return f"{self.rsi_bucket}_{self.volatility}_{self.trend}"


@dataclass
class Lesson:
    """A single lesson learned from a trade."""
    
    # Trade identification
    trade_id: str
    symbol: str
    timestamp: datetime
    
    # What happened
    actual_pnl: float
    actual_r: float
    actual_won: bool
    
    # What should have happened
    optimal_action: str  # "entry_offset", "sl_mult", "tp_mult", "hold", "skip", "none"
    optimal_value: float  # The optimal parameter value
    optimal_pnl: float
    improvement: float  # optimal_pnl - actual_pnl
    
    # Context (for pattern matching)
    context: MarketContext = field(default_factory=MarketContext)
    
    # Derived insight
    insight: str = ""  # Human-readable lesson
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "actual_pnl": float(self.actual_pnl),
            "actual_r": float(self.actual_r),
            "actual_won": bool(self.actual_won),  # Convert numpy bool
            "optimal_action": self.optimal_action,
            "optimal_value": float(self.optimal_value),
            "optimal_pnl": float(self.optimal_pnl),
            "improvement": float(self.improvement),
            "context": self.context.to_dict(),
            "insight": self.insight,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Lesson":
        context = MarketContext.from_dict(d.get("context", {}))
        ts = d.get("timestamp", datetime.now())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            trade_id=d["trade_id"],
            symbol=d["symbol"],
            timestamp=ts,
            actual_pnl=d["actual_pnl"],
            actual_r=d["actual_r"],
            actual_won=d["actual_won"],
            optimal_action=d["optimal_action"],
            optimal_value=d["optimal_value"],
            optimal_pnl=d["optimal_pnl"],
            improvement=d["improvement"],
            context=context,
            insight=d.get("insight", ""),
        )


class LessonExtractor:
    """Extracts lessons from counterfactual analysis."""
    
    # Thresholds for considering something "significantly better"
    MIN_IMPROVEMENT_PCT = 0.5  # Must be at least 0.5% better
    MIN_IMPROVEMENT_R = 0.2    # Must be at least 0.2R better
    
    def extract(
        self,
        trade: Dict[str, Any],
        analysis: "CounterfactualAnalysis",  # From counterfactual.py
        context: MarketContext,
    ) -> Lesson:
        """
        Extract the key lesson from a trade's counterfactual analysis.
        
        Determines what the optimal action would have been and by how much
        it would have improved the outcome.
        """
        from .counterfactual import CounterfactualAnalysis
        
        trade_id = f"{trade.get('symbol', 'UNK')}_{trade.get('entry_time', 0)}"
        
        # Find the best alternative
        best_action = "none"
        best_value = 0.0
        best_pnl = analysis.actual_pnl
        
        # Check entry timing
        if analysis.best_entry_offset and analysis.best_entry_offset.pnl > best_pnl + self.MIN_IMPROVEMENT_PCT:
            best_action = "entry_offset"
            best_value = analysis.best_entry_offset.cf_value
            best_pnl = analysis.best_entry_offset.pnl
        
        # Check SL
        if analysis.best_sl and analysis.best_sl.pnl > best_pnl + self.MIN_IMPROVEMENT_PCT:
            best_action = "sl_mult"
            best_value = analysis.best_sl.cf_value
            best_pnl = analysis.best_sl.pnl
        
        # Check TP
        if analysis.best_tp and analysis.best_tp.pnl > best_pnl + self.MIN_IMPROVEMENT_PCT:
            best_action = "tp_mult"
            best_value = analysis.best_tp.cf_value
            best_pnl = analysis.best_tp.pnl
        
        # Check hold
        if analysis.best_hold and analysis.best_hold.pnl > best_pnl + self.MIN_IMPROVEMENT_PCT:
            best_action = "hold_bars"
            best_value = analysis.best_hold.cf_value
            best_pnl = analysis.best_hold.pnl
        
        # Check skip (only if we lost and skipping was best)
        if analysis.should_have_skipped and 0 > analysis.actual_pnl:
            if 0 > best_pnl or abs(0 - best_pnl) < self.MIN_IMPROVEMENT_PCT:
                best_action = "skip"
                best_value = 1.0
                best_pnl = 0.0
        
        improvement = best_pnl - analysis.actual_pnl
        
        # Generate insight
        insight = self._generate_insight(
            trade, analysis, context, best_action, best_value, improvement
        )
        
        return Lesson(
            trade_id=trade_id,
            symbol=trade.get("symbol", "UNK"),
            timestamp=datetime.fromtimestamp(trade.get("entry_time", 0)) if trade.get("entry_time") else datetime.now(),
            actual_pnl=analysis.actual_pnl,
            actual_r=analysis.actual_r,
            actual_won=analysis.actual_won,
            optimal_action=best_action,
            optimal_value=best_value,
            optimal_pnl=best_pnl,
            improvement=improvement,
            context=context,
            insight=insight,
        )
    
    def _generate_insight(
        self,
        trade: Dict[str, Any],
        analysis: "CounterfactualAnalysis",
        context: MarketContext,
        action: str,
        value: float,
        improvement: float,
    ) -> str:
        """Generate a human-readable insight from the lesson."""
        ctx_key = context.get_bucket_key()
        
        if action == "none":
            if analysis.actual_won:
                return f"Good trade. No better alternative found in {ctx_key} context."
            else:
                return f"Loss was unavoidable in {ctx_key} context."
        
        elif action == "entry_offset":
            if value > 0:
                return f"In {ctx_key}: Wait {int(value)} bars before entering (+{improvement:.1f}%)"
            else:
                return f"In {ctx_key}: Enter {int(-value)} bars earlier (+{improvement:.1f}%)"
        
        elif action == "sl_mult":
            if value > 1:
                return f"In {ctx_key}: Use wider SL ({value:.1f}x) (+{improvement:.1f}%)"
            else:
                return f"In {ctx_key}: Use tighter SL ({value:.1f}x) (+{improvement:.1f}%)"
        
        elif action == "tp_mult":
            if value > 1:
                return f"In {ctx_key}: Use wider TP ({value:.1f}x) (+{improvement:.1f}%)"
            else:
                return f"In {ctx_key}: Use tighter TP ({value:.1f}x) (+{improvement:.1f}%)"
        
        elif action == "hold_bars":
            return f"In {ctx_key}: Hold for {int(value)} bars instead of exiting (+{improvement:.1f}%)"
        
        elif action == "skip":
            return f"In {ctx_key}: Should have skipped this trade (+{improvement:.1f}%)"
        
        return f"Action: {action}, Value: {value}, Improvement: {improvement:.1f}%"


class LessonStore:
    """
    Persistent storage for lessons.
    
    Stores lessons and provides aggregation methods for learning.
    """
    
    def __init__(self, store_path: Optional[str] = None):
        if store_path is None:
            base = Path(os.environ.get("LEARNER_DATA_DIR", "data/learner"))
            base.mkdir(parents=True, exist_ok=True)
            store_path = str(base / "lessons.jsonl")
        
        self.store_path = Path(store_path)
        self.lessons: List[Lesson] = []
        self._load()
    
    def _load(self) -> None:
        """Load lessons from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    for line in f:
                        if line.strip():
                            self.lessons.append(Lesson.from_dict(json.loads(line)))
                print(f"[LEARNER] Loaded {len(self.lessons)} lessons")
            except Exception as e:
                print(f"[LEARNER] Warning: Failed to load lessons: {e}")
    
    def add(self, lesson: Lesson) -> None:
        """Add a lesson and persist."""
        self.lessons.append(lesson)
        
        # Append to file
        with open(self.store_path, "a") as f:
            f.write(json.dumps(lesson.to_dict()) + "\n")
    
    def get_lessons_by_context(self, context_key: str) -> List[Lesson]:
        """Get all lessons for a given context bucket."""
        return [l for l in self.lessons if l.context.get_bucket_key() == context_key]
    
    def get_aggregate_by_action(self, action: str) -> Dict[str, Any]:
        """Get aggregate statistics for a specific action type."""
        relevant = [l for l in self.lessons if l.optimal_action == action]
        if not relevant:
            return {"count": 0}
        
        values = [l.optimal_value for l in relevant]
        improvements = [l.improvement for l in relevant]
        
        return {
            "count": len(relevant),
            "avg_value": sum(values) / len(values),
            "avg_improvement": sum(improvements) / len(improvements),
            "max_improvement": max(improvements),
        }
    
    def get_context_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by context bucket."""
        summary = {}
        
        for lesson in self.lessons:
            key = lesson.context.get_bucket_key()
            if key not in summary:
                summary[key] = {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_improvement": 0.0,
                    "skip_recommended": 0,
                    "entry_delay_sum": 0.0,
                    "entry_delay_count": 0,
                    "sl_mult_sum": 0.0,
                    "sl_mult_count": 0,
                }
            
            s = summary[key]
            s["count"] += 1
            if lesson.actual_won:
                s["wins"] += 1
            else:
                s["losses"] += 1
            s["total_improvement"] += lesson.improvement
            
            if lesson.optimal_action == "skip":
                s["skip_recommended"] += 1
            elif lesson.optimal_action == "entry_offset":
                s["entry_delay_sum"] += lesson.optimal_value
                s["entry_delay_count"] += 1
            elif lesson.optimal_action == "sl_mult":
                s["sl_mult_sum"] += lesson.optimal_value
                s["sl_mult_count"] += 1
        
        # Calculate averages
        for key, s in summary.items():
            s["win_rate"] = s["wins"] / s["count"] if s["count"] > 0 else 0
            s["avg_improvement"] = s["total_improvement"] / s["count"] if s["count"] > 0 else 0
            s["skip_rate"] = s["skip_recommended"] / s["count"] if s["count"] > 0 else 0
            s["avg_entry_delay"] = s["entry_delay_sum"] / s["entry_delay_count"] if s["entry_delay_count"] > 0 else 0
            s["avg_sl_mult"] = s["sl_mult_sum"] / s["sl_mult_count"] if s["sl_mult_count"] > 0 else 1.0
        
        return summary
    
    def clear(self) -> None:
        """Clear all lessons."""
        self.lessons = []
        if self.store_path.exists():
            self.store_path.unlink()
