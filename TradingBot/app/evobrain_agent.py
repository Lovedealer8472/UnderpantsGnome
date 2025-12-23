"""
EvoBrain - AI-Driven Evolutionary Optimization Agent
Hybrid system combining algorithmic evolution with LLM-driven strategic mutations.
"""

import json
import copy
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

try:
    from app.llm_optimizer import LLMOptimizer, BacktestSummary, TradeSummary
except ImportError:
    LLMOptimizer = None
    BacktestSummary = None
    TradeSummary = None


@dataclass
class EvolutionContext:
    """Context for evolutionary decision-making"""
    generation: int
    stagnation_count: int
    champion_fitness: float
    population_avg_fitness: float
    hall_of_fame_size: int
    recent_improvements: List[float]  # Last 5 fitness improvements


@dataclass
class MutationDecision:
    """Structured mutation output"""
    mutations: Dict[str, Any]  # Parameter changes
    persona_switch: Optional[str]  # New persona name or None
    exploration_mode: str  # CONSERVATIVE, NORMAL, AGGRESSIVE, CHAOS
    reasoning: str
    confidence: float
    method: str  # "ai_driven" or "algorithmic"
    json_output: Dict[str, Any]  # Machine-readable format


class EvoBrainAgent:
    """
    EvoBrain - Evolutionary optimization agent.
    Makes strategic mutation decisions using AI when beneficial.
    Falls back to algorithmic mutations for routine cases.
    """
    
    def __init__(
        self,
        llm_optimizer: Optional[LLMOptimizer] = None,
        use_ai_threshold: float = 0.7,  # Use AI if confidence > threshold
        ai_trigger_stagnation: int = 5,  # Use AI after N generations without improvement
        ai_trigger_champion: bool = True,  # Use AI for Hall of Fame entries
    ):
        self.llm_optimizer = llm_optimizer
        self.use_ai_threshold = use_ai_threshold
        self.ai_trigger_stagnation = ai_trigger_stagnation
        self.ai_trigger_champion = ai_trigger_champion
    
    def decide_mutation(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        context: EvolutionContext,
        exploration_mode: str = "NORMAL",
        mutate_persona_func=None,  # Pass the mutate_persona function
        champion_config: Dict[str, Any] = None  # Pass the champion config (optional)
    ) -> MutationDecision:
        """
        Decide on mutations for a config.
        Returns structured MutationDecision with JSON output.
        """
        # Determine if AI should be used
        use_ai = self._should_use_ai(context, metrics)
        
        if use_ai and self.llm_optimizer:
            return self._ai_driven_mutation(config, metrics, context, exploration_mode, champion_config)
        else:
            return self._algorithmic_mutation(
                config, metrics, context, exploration_mode, mutate_persona_func
            )
    
    def _should_use_ai(
        self,
        context: EvolutionContext,
        metrics: Dict[str, Any]
    ) -> bool:
        """Determine if AI-driven mutation should be used"""
        
        # Always use AI for champions (Hall of Fame entries)
        if self.ai_trigger_champion and context.hall_of_fame_size > 0:
            return True
        
        # Use AI if stagnation detected
        if context.stagnation_count >= self.ai_trigger_stagnation:
            return True
        
        # Use AI if performance is poor (needs strategic intervention)
        pf = metrics.get('profit_factor', 0)
        if pf < 0.9:
            return True
        
        # Use AI if drawdown is extreme
        dd = abs(metrics.get('max_drawdown_pct', 0))
        if dd > 20:
            return True
        
        return False
    
    def _ai_driven_mutation(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        context: EvolutionContext,
        exploration_mode: str,
        champion_config: Dict[str, Any] = None
    ) -> MutationDecision:
        """Generate AI-driven mutation"""
        
        # Convert metrics to BacktestSummary format
        summary = self._convert_metrics_to_summary(metrics)
        
        # Get LLM suggestions
        try:
            llm_result = self.llm_optimizer.analyze_backtest(
                summary=summary,
                current_config=config,
                iteration=context.generation,
                history=None,  # Could add history tracking
                champion_config=champion_config  # Pass champion config for context
            )
            
            adjustments = llm_result.get('adjustments', {})
            reasoning = llm_result.get('reasoning', 'AI-driven strategic adjustment')
            confidence = llm_result.get('confidence', 0.5)
        except Exception as e:
            # Fallback to algorithmic if AI fails
            print(f"[EVOBRAIN] AI mutation failed: {e}, falling back to algorithmic")
            return self._algorithmic_mutation(
                config, metrics, context, exploration_mode, None
            )
        
        # Apply adjustments to config (handle relative/absolute)
        mutated_config = self._apply_adjustments(config, adjustments)
        
        # AI-driven tier analysis: Remove unprofitable tiers
        tier_pnl = metrics.get('tier_pnl', {})
        if tier_pnl:
            current_tiers = mutated_config.get('allowed_tiers', ['unicorn'])
            profitable_tiers = [tier for tier, pnl in tier_pnl.items() if pnl > 0]
            losing_tiers = [tier for tier, pnl in tier_pnl.items() if pnl < 0]
            
            # Remove losing tiers if AI didn't already suggest it
            if losing_tiers and 'allowed_tiers' not in adjustments:
                new_tiers = [t for t in current_tiers if t not in losing_tiers]
                if new_tiers:  # Don't remove all tiers
                    mutated_config['allowed_tiers'] = new_tiers
                    reasoning += f" | Removed losing tiers: {losing_tiers}"
        
        # Determine persona switch (if in CHAOS mode)
        persona_switch = None
        if exploration_mode == "CHAOS" and random.random() < 0.3:
            persona_switch = self._suggest_persona_switch(metrics, config)
        
        # Build JSON output
        json_output = {
            "mutations": {
                "min_score": mutated_config.get('min_score'),
                "min_momentum_pct": mutated_config.get('min_momentum_pct'),
                "sl_R": mutated_config.get('sl_R'),
                "tp_R": mutated_config.get('tp_R'),
                "allowed_tiers": mutated_config.get('allowed_tiers')
            },
            "persona_switch": persona_switch,
            "exploration_mode": exploration_mode,
            "reasoning": reasoning,
            "confidence": confidence,
            "generation": context.generation,
            "method": "ai_driven",
            "context": {
                "stagnation_count": context.stagnation_count,
                "champion_fitness": context.champion_fitness,
                "population_avg_fitness": context.population_avg_fitness
            }
        }
        
        return MutationDecision(
            mutations=json_output["mutations"],
            persona_switch=persona_switch,
            exploration_mode=exploration_mode,
            reasoning=reasoning,
            confidence=confidence,
            method="ai_driven",
            json_output=json_output
        )
    
    def _algorithmic_mutation(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        context: EvolutionContext,
        exploration_mode: str,
        mutate_persona_func=None
    ) -> MutationDecision:
        """Generate algorithmic mutation (fast, cheap)"""
        
        # Use existing mutate_persona logic if provided
        # PHASE 1: Pass metrics for Evolutionary Apex adaptive mutation
        if mutate_persona_func:
            # Try calling with metrics (Evolutionary Apex), fallback to old signature
            try:
                mutated_config = mutate_persona_func(config, exploration_mode, metrics)
            except TypeError:
                # Old signature doesn't accept metrics
                mutated_config = mutate_persona_func(config, exploration_mode)
        else:
            # Fallback: simple mutation
            mutated_config = copy.deepcopy(config)
        
        # PRIME DIRECTIVE reasoning based on metrics
        reasoning_parts = []
        pf = metrics.get('profit_factor', 0)
        wr = metrics.get('win_rate', 0)
        trades = metrics.get('total_trades', 0)
        avg_r = metrics.get('avg_R', 0)
        max_r = metrics.get('max_R', 0)
        target_trades = 5475
        
        if pf < 1.0:
            reasoning_parts.append("PRIME DIRECTIVE: Unprofitable - ESCAPE NOW (exploring aggressively)")
        elif avg_r < 0.5 or max_r < 1.5:
            reasoning_parts.append("PRIME DIRECTIVE: Low R-multiples - boost runners for tail profits")
        elif trades < target_trades * 0.5:
            reasoning_parts.append("PRIME DIRECTIVE: Low trades - aggressively increasing frequency for faster compounding")
        elif trades < target_trades:
            reasoning_parts.append("PRIME DIRECTIVE: Moderate trades - increasing frequency for faster compounding")
        elif wr < 50:
            reasoning_parts.append("PRIME DIRECTIVE: Low win rate - adjusting thresholds for better edge extraction")
        else:
            reasoning_parts.append("PRIME DIRECTIVE: Algorithmic exploration for profit maximization")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "PRIME DIRECTIVE: Routine profit-focused mutation"
        
        json_output = {
            "mutations": {
                "min_score": mutated_config.get('min_score'),
                "min_momentum_pct": mutated_config.get('min_momentum_pct'),
                "sl_R": mutated_config.get('sl_R'),
                "tp_R": mutated_config.get('tp_R'),
                "allowed_tiers": mutated_config.get('allowed_tiers')
            },
            "persona_switch": None,
            "exploration_mode": exploration_mode,
            "reasoning": reasoning,
            "confidence": 0.5,
            "generation": context.generation,
            "method": "algorithmic"
        }
        
        return MutationDecision(
            mutations=json_output["mutations"],
            persona_switch=None,
            exploration_mode=exploration_mode,
            reasoning=reasoning,
            confidence=0.5,
            method="algorithmic",
            json_output=json_output
        )
    
    def _convert_metrics_to_summary(self, metrics: Dict[str, Any]) -> BacktestSummary:
        """Convert backtest metrics to BacktestSummary format"""
        if BacktestSummary is None:
            raise ImportError("BacktestSummary not available")
        
        # Convert tier_pnl to tier_performance format
        tier_pnl = metrics.get('tier_pnl', {})
        tier_counts = metrics.get('tier_counts', {})
        tier_performance = {}
        
        for tier in tier_pnl.keys():
            count = tier_counts.get(tier, 0)
            pnl_pct = tier_pnl.get(tier, 0) * 100
            tier_performance[tier] = {
                'count': count,
                'pnl_pct': pnl_pct
            }
        
        return BacktestSummary(
            total_trades=metrics.get('total_trades', 0),
            win_rate=metrics.get('win_rate', 0),
            profit_factor=metrics.get('profit_factor', 0),
            total_pnl_pct=metrics.get('total_pnl', 0) * 100,
            max_drawdown_pct=abs(metrics.get('max_drawdown_pct', 0)),
            avg_r=metrics.get('avg_R', 0),
            max_r=metrics.get('max_R', 0),
            worst_r=metrics.get('worst_R', 0),
            sl_hit_pct=metrics.get('sl_hit_pct', 0),
            tp_hit_pct=metrics.get('tp_hit_pct', 0),
            avg_trade_duration_sec=metrics.get('avg_trade_duration_sec', 0),
            tier_performance=tier_performance,
            symbol_performance=metrics.get('symbol_performance', {}),
            regime_performance=metrics.get('regime_performance', {}),
            outlier_trades=[],  # Could extract from metrics
            r_distribution=metrics.get('r_distribution', {}),
            execution_quality={}
        )
    
    def _apply_adjustments(
        self,
        config: Dict[str, Any],
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply LLM adjustments to config (handle relative/absolute)"""
        mutated = copy.deepcopy(config)
        
        # Use LLMOptimizer validation if available
        if self.llm_optimizer:
            try:
                validated = self.llm_optimizer._validate_adjustments(adjustments, config)
                for key, value in validated.items():
                    if key in mutated:
                        mutated[key] = value
                return mutated
            except Exception:
                pass  # Fall through to manual application
        
        # Manual application (fallback)
        for key, value in adjustments.items():
            if key == 'allowed_tiers':
                if isinstance(value, list):
                    mutated[key] = value
                continue
            
            if key not in mutated:
                continue
            
            current_val = mutated[key]
            
            # Handle relative adjustments (e.g., "+5", "+0.1")
            if isinstance(value, str):
                if value.startswith('+'):
                    try:
                        delta = float(value[1:])
                        mutated[key] = current_val + delta
                    except ValueError:
                        pass
                elif value.startswith('-'):
                    try:
                        delta = float(value[1:])
                        mutated[key] = current_val - delta
                    except ValueError:
                        pass
                else:
                    try:
                        mutated[key] = float(value)
                    except ValueError:
                        pass
            else:
                # Absolute value
                mutated[key] = value
        
        return mutated
    
    def _suggest_persona_switch(
        self,
        metrics: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> Optional[str]:
        """Suggest persona switch based on performance"""
        # Simple heuristic: switch if performance is poor
        pf = metrics.get('profit_factor', 0)
        trades = metrics.get('total_trades', 0)
        
        if pf < 0.8 and trades < 100:
            # Low trades, low PF - try more aggressive persona
            return "LightningStrike"
        elif pf > 1.5 and trades > 500:
            # High trades, good PF - try defensive persona
            return "GrimBadgerV2"
        
        return None

