"""
LLM Analysis Engine
Analyzes backtest results and suggests parameter adjustments using LLM reasoning.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Optional import - httpx may not be installed
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


@dataclass
class TradeSummary:
    """Summary of a single trade for LLM analysis"""
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    r_multiple: float
    duration_sec: float
    exit_reason: str
    score: float
    tier: str
    slippage_entry_bps: float
    slippage_exit_bps: float


@dataclass
class BacktestSummary:
    """Complete backtest summary for LLM analysis"""
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl_pct: float
    max_drawdown_pct: float
    avg_r: float
    max_r: float
    worst_r: float
    sl_hit_pct: float
    tp_hit_pct: float
    avg_trade_duration_sec: float
    tier_performance: Dict[str, Dict[str, float]]
    symbol_performance: Dict[str, Dict[str, float]]
    regime_performance: Dict[str, Dict[str, float]]
    outlier_trades: List[TradeSummary]
    r_distribution: Dict[str, int]  # R-multiple buckets
    execution_quality: Dict[str, float]


class LLMOptimizer:
    """
    LLM-driven optimization engine with hybrid approach.
    Analyzes backtest results and suggests parameter adjustments.
    Uses LLM only when needed (critical cases), rule-based otherwise.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "groq",
        model: str = "llama-3.1-70b-versatile"
    ):
        """
        Initialize LLM optimizer.
        
        Args:
            api_key: LLM API key (defaults to env var)
            provider: LLM provider ('groq', 'openai', 'deepseek')
            model: Model name
        """
        # Check if httpx is available (try import if module-level check failed)
        httpx_available = HTTPX_AVAILABLE
        if not httpx_available:
            # Try to import again at runtime - might be available now
            try:
                import httpx
                httpx_available = True
            except ImportError:
                pass
        
        if not httpx_available:
            raise ImportError(
                "httpx is required for LLMOptimizer. Install it with: pip install httpx>=0.24.0\n"
                "If httpx is already installed, check that you're using the correct Python environment."
            )
        
        self.provider = provider.lower()
        self.model = model
        
        # Get API key from env if not provided
        if api_key is None:
            if self.provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
            elif self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
        
        self.api_key = api_key
        self.base_url = self._get_base_url(self.provider)
        
        # Simple cache for common patterns (reduces redundant LLM calls)
        self._response_cache: Dict[str, Dict] = {}
        self._cache_max_size = 50
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment"""
        if self.provider == "groq":
            return os.getenv("GROQ_API_KEY")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY")
        return None
    
    def _get_base_url(self, provider: str) -> str:
        """Get API base URL for provider"""
        if provider == "groq":
            return "https://api.groq.com/openai/v1"
        elif provider == "openai":
            return "https://api.openai.com/v1"
        elif provider == "deepseek":
            return "https://api.deepseek.com/v1"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def analyze_backtest(
        self,
        summary: BacktestSummary,
        current_config: Dict[str, Any],
        iteration: int,
        history: List[Dict] = None,
        champion_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze backtest results and suggest parameter adjustments.
        Uses hybrid approach: LLM only when needed, rule-based otherwise.
        
        Args:
            summary: BacktestSummary with all metrics
            current_config: Current parameter configuration
            iteration: Current iteration number
            history: Previous iteration results (for context)
            champion_config: Best configuration found so far (optional)
            
        Returns:
            Dict with 'adjustments' (parameter changes) and 'reasoning' (explanation)
        """
        if not self.api_key:
            # Fallback to rule-based if no API key
            return self._rule_based_analysis(summary, current_config)
        
        # HYBRID: Decide if LLM is needed
        use_llm = self._should_use_llm(summary, current_config, iteration, history)
        
        if not use_llm:
            # Use rule-based for routine cases (saves tokens)
            result = self._rule_based_analysis(summary, current_config)
            result['method'] = 'rule_based'
            return result
        
        # Build compressed prompt for LLM
        prompt = self._build_compressed_prompt(summary, current_config, iteration, history, champion_config)
        
        # Call LLM (use cheaper model for routine cases)
        model_to_use = self._select_model(summary, iteration)
        
        try:
            response = self._call_llm(prompt, model=model_to_use)
            adjustments = self._parse_llm_response(response)
            
            # Validate and sanitize adjustments
            adjustments = self._validate_adjustments(adjustments, current_config)
            
            return {
                'adjustments': adjustments,
                'reasoning': response.get('reasoning', ''),
                'confidence': response.get('confidence', 0.5),
                'method': 'llm',
                'model': model_to_use
            }
        except Exception as e:
            print(f"[LLM ERROR] LLM analysis failed (iteration {iteration}): {type(e).__name__}: {e} - Falling back to rule-based analysis")
            return self._rule_based_analysis(summary, current_config)
    
    def _build_compressed_prompt(
        self,
        summary: BacktestSummary,
        current_config: Dict[str, Any],
        iteration: int,
        history: List[Dict] = None,
        champion_config: Dict[str, Any] = None
    ) -> str:
        """Build ultra-compressed analysis prompt for LLM (80-85% token reduction)"""
        
        # ULTRA-COMPRESSED: Tier performance (abbreviated)
        tier_perf = []
        for tier, data in summary.tier_performance.items():
            count = data.get('count', 0)
            pnl = data.get('pnl_pct', 0)
            tier_abbr = tier[0].upper()  # U, E, S, A
            tier_perf.append(f"{tier_abbr}:{count}t+{pnl:.1f}%")
        tier_perf_str = " ".join(tier_perf)
        
        # ULTRA-COMPRESSED: Outlier trades (PnL only, no symbols)
        if summary.outlier_trades:
            winners = [t for t in summary.outlier_trades if t.pnl_pct > 0][:3]
            losers = [t for t in summary.outlier_trades if t.pnl_pct < 0][-3:]
            win_str = "+".join([f"{t.pnl_pct:.1f}%" for t in winners]) if winners else ""
            lose_str = "-".join([f"{abs(t.pnl_pct):.1f}%" for t in losers]) if losers else ""
            outlier_str = f"W:{win_str} L:{lose_str}" if (win_str or lose_str) else ""
        else:
            outlier_str = ""
        
        # ULTRA-COMPRESSED: R distribution (minimal format)
        r_dist = summary.r_distribution
        r_neg = r_dist.get('<-2R', 0) + r_dist.get('-2R to -1R', 0) + r_dist.get('-1R to 0R', 0)
        r_zero_one = r_dist.get('0R to 1R', 0)
        r_pos = r_dist.get('1R to 2R', 0) + r_dist.get('>2R', 0)
        r_summary = f"R:<0:{r_neg} 0-1:{r_zero_one} >1:{r_pos}"
        
        # CONDITIONAL: History (only if stuck or first iteration)
        history_str = ""
        include_history = False
        if iteration == 1:
            include_history = True  # Always include on first iteration
        elif history and len(history) >= 3:
            # Check if stuck (no improvement for 3+ iterations)
            recent_scores = [h.get('score', 0) for h in history[-3:]]
            if max(recent_scores) - min(recent_scores) < 0.5:
                include_history = True  # Stuck, include history
        
        if include_history and history and len(history) >= 2:
            recent = history[-2:]
            history_str = f"H: PF={recent[0].get('pf', 0):.2f} WR={recent[0].get('wr', 0):.0f}% T={recent[0].get('trades', 0)} | PF={recent[1].get('pf', 0):.2f} WR={recent[1].get('wr', 0):.0f}% T={recent[1].get('trades', 0)}"
            
        # NEW: Add Champion Context
        champ_str = ""
        if champion_config:
            c = champion_config
            champ_str = f"CHAMPION (PF={c.get('pf', 0):.2f}): score={c.get('min_score')} mom={c.get('min_momentum_pct')} sl={c.get('sl_R')} tp={c.get('tp_R')}"
        
        # ULTRA-COMPRESSED PROMPT (80-85% token reduction)
        # PRIME DIRECTIVE: MAKE MORE MONEY, FASTER, SAFELY
        # Profit > Everything. No style points. No persona loyalty. Money determines survival.
        target_trades = 5475  # 5 trades/day over 3 years
        trade_status = "LOW" if summary.total_trades < target_trades * 0.5 else "OK" if summary.total_trades < target_trades else "HIGH"
        
        prompt = f"""EVOLUTION PRIME DIRECTIVE: GET AS FAT AS POSSIBLE, AS FAST AS POSSIBLE, WITHOUT HURTING YOURSELF OR STARVING.

THE HUNT: Opening a position is like hunting for food. Each hunt costs energy (fees) and can result in injury (loss) or food (win).
SUPREME OBJECTIVE: Accumulate maximum energy (profit) to beat competition and earn the honor of breeding.
BREEDING IS THE PIVOTAL URGE: Only the fittest (fattest) hunters breed. Higher fitness = more breeding rights = genes spread more.

RISK vs REWARD:
- Conservative hunters: Low frequency, safe (small snacks) = survive but don't breed much
- Aggressive hunters: High frequency = MORE RISK (more chances to get hurt/lose)
  - If successful: Get fat fast, breed a lot (high fitness = breeding rights)
  - If unsuccessful: Nature eliminates them (high frequency + unprofitable = death before breeding)
- Bad aggressive hunters: Taking more risk but losing = eliminated by nature (no breeding = genes die out)
- Good aggressive hunters: Taking more risk and winning = get fat and breed (genes spread)

Your future generations will evolve and flourish based on your success. Get fat or die trying.

HUNTING STATS:
- Current Hunter: score={current_config.get('min_score', 80)} mom={current_config.get('min_momentum_pct', 3.0)} sl={current_config.get('sl_R', 1.4)} tp={current_config.get('tp_R', 2.2)}
{champ_str if champ_str else ""}

HUNT RESULTS (I{iteration}): 
- Hunts Attempted: {summary.total_trades} ({trade_status}, target={target_trades})
- Success Rate: {summary.win_rate:.1f}% (food caught vs injuries)
- Energy Gained: PF={summary.profit_factor:.2f} PnL={summary.total_pnl_pct:.1f}%
- Risk Taken: DD={summary.max_drawdown_pct:.1f}% (how much you got hurt)
- Food Quality: R={summary.avg_r:.2f} MaxR={summary.max_r:.2f} (size of catches)
- Exit Patterns: SL={summary.sl_hit_pct:.1f}% TP={summary.tp_hit_pct:.1f}%

T:{tier_perf_str} {r_summary} {outlier_str if outlier_str else ""}
{history_str if history_str else ""}

HUNTING PRIORITY - ACCUMULATE ENERGY OR STARVE:
1. PF<1.0: YOU'RE STARVING - ESCAPE NOW (raise score/mom, tighten stops, remove losing tiers)
   ⚠️ CRITICAL: If PF<1.0, you're losing energy on every hunt. Transaction costs (hunting energy cost) compound your losses.
   GUIDANCE: Become more selective (raise thresholds) OR hunt smarter (tighter stops). 
   Natural learning: Unprofitable + high frequency = negative energy = eliminated from competition.
   
2. PF>=1.0 BUT Low R (avg_R<0.5): CATCH BIGGER PREY (wider TP for runners, capture momentum)
   You're catching food but it's small. Let winners run to catch bigger prey (higher R-multiples).
   
3. PF>=1.0 AND T<{target_trades}: HUNT MORE OFTEN (lower score/mom, expand tiers, relax filters)
   You're profitable but hunting too rarely. More successful hunts = faster energy accumulation.
   GUIDANCE: Hunt more IF you maintain profitability. Transaction costs create natural limit - find equilibrium.
   
4. PF>=1.0 AND T>={target_trades}: MAXIMIZE ENERGY PER HUNT (optimize PF, boost R-multiples)
   You have enough frequency. Focus on quality: bigger catches (higher R), smarter exits, better edge extraction.

HUNTING WISDOM:
- Big Prey (High R, max_R>2.0): Runners boost tail profits. If max_R<1.5, widen TP to catch bigger prey.
- Hunting Frequency: More hunts = faster energy growth IF PF>=1.0. But each hunt costs energy (fees).
- Energy Efficiency: High PF with low trades = good edge but slow. Find the sweet spot.
- Natural Selection: Hunters that find the efficiency frontier (high PF + optimal frequency) accumulate most energy and win breeding rights.

GUIDANCE FOR THE HUNT (not rules - let evolution discover):
- If PF<1.0: You're losing energy. Focus on profitability first. High frequency will naturally reduce as you learn.
- If PF>=1.0 AND T<{target_trades}: Can hunt more often, but must maintain profitability.
- If PF>=1.0 AND T>{target_trades}: Focus on quality. Transaction costs will guide optimal hunting frequency.
- Natural learning: The fitness function (energy measurement) rewards profitable frequency and penalizes unprofitable frequency.
To BOOST R: widen tp_R (2.0-4.0) to let runners run, tighten sl_R if needed to maintain risk.
To ESCAPE UNPROFITABILITY: raise score (85-95), raise mom (5.0-10.0), tighten stops, remove losing tiers.

Limits: score 70-95, mom 2.0-10.0, tiers [U]/[E]/[U,E]/[U,E,S], sl 0.6-2.0, tp 0.8-4.0

JSON:
{{"reasoning":"brief profit-focused","adjustments":{{"min_score":"+5"or"85","min_momentum_pct":"+1.0"or"5.0","allowed_tiers":["unicorn"],"sl_R":"+0.1"or"1.2","tp_R":"+0.3"or"1.5"}},"confidence":0.0-1.0}}
Only changes that increase PROFIT. No style points."""
        return prompt
    
    def _should_use_llm(
        self,
        summary: BacktestSummary,
        current_config: Dict[str, Any],
        iteration: int,
        history: List[Dict] = None
    ) -> bool:
        """
        Determine if LLM analysis is needed.
        Always returns True to analyze every iteration (user preference).
        Cost optimizations still apply: compressed prompts, caching, cheaper models.
        """
        # Always use LLM for analysis on each iteration
        return True
    
    def _select_model(self, summary: BacktestSummary, iteration: int) -> str:
        """
        Select appropriate model based on complexity.
        Use cheaper models for routine cases.
        """
        # For critical issues, use full model
        if summary.profit_factor < 0.9 or summary.max_drawdown_pct > 30:
            return self.model  # Full model
        
        # For routine cases, use cheaper model if available
        if self.provider == "groq":
            # Groq has free tier - use it for routine cases
            if "mixtral" in self.model.lower() or "llama" in self.model.lower():
                return "mixtral-8x7b-32768"  # Cheaper/faster model
            return self.model
        
        # Default to configured model
        return self.model
    
    def _get_cache_key(self, summary: BacktestSummary, config: Dict[str, Any]) -> str:
        """
        Generate cache key for similar scenarios.
        Groups by similar metric patterns to avoid redundant LLM calls.
        """
        # Round metrics to create buckets (similar scenarios get same key)
        pf_bucket = round(summary.profit_factor, 1)  # 0.1 precision
        wr_bucket = round(summary.win_rate / 10) * 10  # 10% buckets
        dd_bucket = round(summary.max_drawdown_pct / 5) * 5  # 5% buckets
        
        # Include tier performance pattern
        tier_pattern = tuple(sorted([
            (tier, round(data.get('pnl_pct', 0), 1))
            for tier, data in summary.tier_performance.items()
        ]))
        
        return f"{pf_bucket}_{wr_bucket}_{dd_bucket}_{tier_pattern}"
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache LLM result with size limit"""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = result.copy()
    
    def test_connection(self) -> bool:
        """
        Test API connection with a simple ping.
        Returns True if API is working, False otherwise.
        """
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Minimal test payload
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "test"}
                ],
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return True
        except Exception:
            return False
    
    def _call_llm(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Call LLM API"""
        if model is None:
            model = self.model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "EVOLUTION PRIME DIRECTIVE: HUNT FOR ENERGY, ACCUMULATE WEALTH, DOMINATE COMPETITION. You are a hunting guide for evolutionary traders. Each position is a hunt - it costs energy (fees) and can result in food (win) or injury (loss). Your supreme objective is to guide hunters to accumulate maximum energy (profit) to beat competition and earn breeding rights. Energy > Aesthetic strategies. Energy > Tradability preferences. Energy > Style or personality. Energy > Simplicity, unless simplicity increases energy. Persona loyalty is irrelevant - energy accumulation determines survival and breeding rights. Every recommendation must directly support: HUNT FOR ENERGY, ACCUMULATE WEALTH, DOMINATE COMPETITION. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse JSON response
            content = result['choices'][0]['message']['content']
            return json.loads(content)
    
    def _parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into parameter adjustments"""
        adjustments = response.get('adjustments', {})
        parsed = {}
        
        for key, value in adjustments.items():
            if isinstance(value, str):
                # Handle relative adjustments: "+5", "-3", "85"
                if value.startswith('+'):
                    parsed[key] = ('relative', float(value[1:]))
                elif value.startswith('-'):
                    parsed[key] = ('relative', float(value))
                else:
                    parsed[key] = ('absolute', float(value))
            elif isinstance(value, (int, float)):
                parsed[key] = ('absolute', float(value))
            elif isinstance(value, list):
                # For allowed_tiers
                parsed[key] = ('absolute', value)
        
        return parsed
    
    def _validate_adjustments(
        self,
        adjustments: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and apply bounds to adjustments"""
        validated = {}
        
        bounds = {
            'min_score': (70, 95),
            'min_momentum_pct': (2.0, 10.0),
            'sl_R': (0.6, 2.0),
            'tp_R': (0.8, 4.0),
            'atr_len': (10, 21)
        }
        
        for key, (adjust_type, value) in adjustments.items():
            if key == 'allowed_tiers':
                # Validate tier list
                if isinstance(value, tuple) and value[0] == 'absolute':
                    tiers = value[1]
                    if isinstance(tiers, list) and all(t in ['unicorn', 'elite', 'strong', 'average'] for t in tiers):
                        validated[key] = tiers
                continue
            
            if key not in bounds:
                continue
            
            min_val, max_val = bounds[key]
            current_val = current_config.get(key, min_val)
            
            if adjust_type == 'relative':
                new_val = current_val + value
            else:  # absolute
                new_val = value
            
            # Clamp to bounds
            new_val = max(min_val, min(max_val, new_val))
            
            # Round appropriately
            if key in ['min_score', 'atr_len']:
                new_val = int(round(new_val))
            else:
                new_val = round(new_val, 2)
            
            validated[key] = new_val
        
        return validated
    
    def _rule_based_analysis(
        self,
        summary: BacktestSummary,
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback rule-based analysis if LLM unavailable.
        PRIME DIRECTIVE: HUNT FOR ENERGY, ACCUMULATE WEALTH, DOMINATE COMPETITION.
        """
        adjustments = {}
        reasoning_parts = []
        
        target_trades = 5475  # 5 trades/day over 3 years
        
        # PRIME DIRECTIVE PRIORITY 1: ESCAPE STARVATION - NO EXCUSES
        if summary.profit_factor < 1.0:
            # Starving hunter - AGGRESSIVELY improve hunting skills
            current_score = current_config.get('min_score', 80)
            current_mom = current_config.get('min_momentum_pct', 3.0)
            
            if current_score < 90:
                adjustments['min_score'] = min(95, current_score + 10)  # More aggressive
                reasoning_parts.append("HUNTING GUIDE: Starving hunter - become more selective (raise score) to escape energy loss")
            
            if current_mom < 7.0:
                adjustments['min_momentum_pct'] = min(10.0, current_mom + 2.0)  # More aggressive
                reasoning_parts.append("HUNTING GUIDE: Starving hunter - require stronger momentum to catch better prey")
            
            if summary.tp_hit_pct < 10:
                current_tp = current_config.get('tp_R', 2.2)
                adjustments['tp_R'] = max(0.8, current_tp * 0.8)
                reasoning_parts.append("TP hit rate too low, tightening targets")
        
        # PRIME DIRECTIVE PRIORITY 2: ACCUMULATE MAXIMUM ENERGY - More successful hunts, Bigger prey, Faster energy growth
        elif summary.profit_factor >= 1.0:
            current_score = current_config.get('min_score', 80)
            current_mom = current_config.get('min_momentum_pct', 3.0)
            current_tiers = current_config.get('allowed_tiers', ['unicorn', 'elite'])
            
            # HUNTING GUIDE: Check prey size first (R-multiples = size of catches)
            if summary.avg_r < 0.5 or summary.max_r < 1.5:
                # Small prey - widen TP to catch bigger prey (runners)
                current_tp = current_config.get('tp_R', 2.2)
                if current_tp < 3.0:
                    adjustments['tp_R'] = min(4.0, current_tp + 0.5)
                    reasoning_parts.append("HUNTING GUIDE: Catching small prey - widen targets to catch bigger prey (higher R)")
            
            # HUNTING GUIDE: If hunting too rarely, hunt more often (faster energy accumulation)
            if summary.total_trades < target_trades * 0.5:  # Less than 50% of target
                # AGGRESSIVE: Lower thresholds significantly to hunt more often
                if current_score > 70:
                    adjustments['min_score'] = max(70, current_score - 15)  # More aggressive
                    reasoning_parts.append("HUNTING GUIDE: Hunting too rarely - lower selectivity to hunt more often (faster energy accumulation)")
                
                if current_mom > 2.5:
                    adjustments['min_momentum_pct'] = max(2.0, current_mom - 2.0)  # More aggressive
                    reasoning_parts.append("HUNTING GUIDE: Hunting too rarely - lower momentum requirement to hunt more often")
                
                # Expand tiers if not already full (more hunting opportunities)
                if len(current_tiers) < 3:
                    if 'strong' not in current_tiers:
                        adjustments['allowed_tiers'] = ['unicorn', 'elite', 'strong']
                        reasoning_parts.append("HUNTING GUIDE: Hunting too rarely - expand hunting grounds for more opportunities")
            
            elif summary.total_trades < target_trades:  # 50-100% of target
                # Moderate: Lower thresholds to hunt more often
                if current_score > 75:
                    adjustments['min_score'] = max(70, current_score - 8)  # More aggressive
                    reasoning_parts.append("HUNTING GUIDE: Moderate hunting frequency - lower selectivity to hunt more often")
                
                if current_mom > 3.0:
                    adjustments['min_momentum_pct'] = max(2.0, current_mom - 1.0)  # More aggressive
                    reasoning_parts.append("HUNTING GUIDE: Moderate hunting frequency - lower momentum requirement")
                
                # Expand tiers if conservative
                if len(current_tiers) == 1:
                    adjustments['allowed_tiers'] = ['unicorn', 'elite']
                    reasoning_parts.append("HUNTING GUIDE: Moderate hunting frequency - expand hunting grounds")
            
            # HUNTING GUIDE: If hunting enough, maximize energy per hunt (bigger prey, better edge extraction)
            elif summary.total_trades >= target_trades:
                # Maximize energy: catch bigger prey (boost R-multiples) and improve hunting skills
                if summary.profit_factor < 1.5:
                    # Boost R-multiples if low (catch bigger prey)
                    if summary.max_r < 2.0:
                        current_tp = current_config.get('tp_R', 2.2)
                        if current_tp < 3.5:
                            adjustments['tp_R'] = min(4.0, current_tp + 0.5)
                            reasoning_parts.append("HUNTING GUIDE: Good frequency, small prey - widen targets to catch bigger prey (boost R-multiples)")
                    
                    # Improve edge extraction (hunting skills)
                    if summary.win_rate < 50 and summary.profit_factor > 1.2:
                        # High PF but low WR = good edge, can improve consistency
                        current_score = current_config.get('min_score', 80)
                        if current_score < 85:
                            adjustments['min_score'] = min(90, current_score + 3)
                            reasoning_parts.append("HUNTING GUIDE: Good frequency, improving hunting skills (edge extraction)")
        
        return {
            'adjustments': adjustments,
            'reasoning': '; '.join(reasoning_parts) if reasoning_parts else 'No changes needed',
            'confidence': 0.5
        }

