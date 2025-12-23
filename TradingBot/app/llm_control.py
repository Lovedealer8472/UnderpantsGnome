import os, time, json, csv
from collections import deque
from datetime import datetime
from .config import CONTROL_PATH, ENVELOPE_PATH, TUNING_LOG, safe_read_json
from .regime import RegimeManager, TradingRegime

class LlmController:
    def __init__(self):
        self.last_control_patch = 0
        self.recent_info = deque(maxlen=12)
        self.recent_errors = deque(maxlen=12)
        self.last_advisor_check = 0
        self.advisor_interval = 60  # Generate suggestions every 60 seconds (reduced frequency)
        self.last_regime_check = 0
        self.regime_check_interval = 300  # Check regime every 5 minutes (reduced frequency)
        self.regime_manager = RegimeManager()
        self.last_regime_switch_time = 0.0  # Track last regime switch time
        self.min_regime_duration = 600  # Minimum 10 minutes before allowing regime switch

    def apply_params(self, mode:str, params:dict, comment:str):
        envelopes = safe_read_json(ENVELOPE_PATH, {})
        changed = {}
        for k, v in params.items():
            lo, hi = envelopes.get(k, [None, None])
            if lo is not None and (v < lo or v > hi):
                continue
            old = os.getenv(k, str(v))
            os.environ[k] = str(v)
            changed[k] = (old, v)

        with open(TUNING_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.utcnow().isoformat(), mode, comment, json.dumps(changed)])

        self.recent_info.append({
            'message': f"Applied mode {mode} ({comment})",
            'timestamp': time.time(),
            'priority': 70
        })

    async def check_control_patch(self, bot):
        """Check for manual control patch file updates with improved error handling."""
        if not os.path.exists(CONTROL_PATH):
            return
        
        try:
            ts = os.path.getmtime(CONTROL_PATH)
            if ts <= self.last_control_patch:
                return
            
            data = safe_read_json(CONTROL_PATH, {})
            if not data:
                return
            
            mode = data.get("mode", "")
            params = data.get("params", {})
            comment = data.get("comment", "")
            
            # Check for LLM regime override
            regime_override = data.get("regime_override")
            regime_override_reason = data.get("regime_override_reason", "")
            
            if regime_override:
                try:
                    # Validate regime override
                    override_regime = TradingRegime(regime_override)
                    current_regime = self.regime_manager.current_regime
                    
                    if override_regime != current_regime:
                        # Apply LLM override
                        self.regime_manager.set_regime(override_regime, f"LLM Override: {regime_override_reason}")
                        bot.current_regime = override_regime.value
                        bot.regime_since = time.time()
                        self.last_regime_switch_time = time.time()
                        
                        # Apply regime-specific parameters
                        regime_config = self.regime_manager.get_current_config()
                        self.apply_regime_config(bot, regime_config)
                        
                        self.recent_info.append({
                            'message': f"LLM Regime Override: {override_regime.value} ({regime_override_reason})",
                            'timestamp': time.time(),
                            'priority': 95
                        })
                        
                        # Clear the override from control patch to prevent re-applying
                        data.pop("regime_override", None)
                        data.pop("regime_override_reason", None)
                        try:
                            with open(CONTROL_PATH, 'w') as f:
                                json.dump(data, f, indent=2)
                        except Exception as e:
                            self.recent_errors.append(f"Failed to write control patch: {e}")
                            
                except (ValueError, AttributeError) as e:
                    error_msg = f"Invalid regime override: {regime_override} ({e})"
                    self.recent_errors.append(error_msg)
                    try:
                        from .logger import get_logger
                        get_logger().warning(f"Control patch error: {error_msg}")
                    except ImportError:
                        pass
            
            # Apply parameter changes
            if mode or params:
                try:
                    self.apply_params(mode, params, comment)
                    bot.current_mode = mode or bot.current_mode
                    bot.mode_since = time.time()
                except Exception as e:
                    error_msg = f"Failed to apply params: {e}"
                    self.recent_errors.append(error_msg)
                    try:
                        from .logger import get_logger
                        get_logger().warning(f"Control patch error: {error_msg}")
                    except ImportError:
                        pass
            
            self.last_control_patch = ts
            
        except Exception as e:
            error_msg = f"Error checking control patch: {e}"
            self.recent_errors.append(error_msg)
            try:
                from .logger import get_logger
                get_logger().error(f"Control patch error: {error_msg}")
            except ImportError:
                pass

    def generate_advisor_suggestions(self, bot):
        """
        Generate critical advisor suggestions only (simplified).
        Focuses on critical alerts and important status updates.
        """
        now = time.time()
        if now - self.last_advisor_check < self.advisor_interval:
            return
        
        self.last_advisor_check = now
        
        # Get critical metrics only
        equity = bot.equity_now()
        start_equity = getattr(bot, "start_equity", equity)
        equity_change = ((equity - start_equity) / start_equity) * 100 if start_equity > 0 else 0
        
        win_count = getattr(bot, "win_count", 0)
        loss_count = getattr(bot, "loss_count", 0)
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        gross_win = getattr(bot, "gross_win", 0.0)
        gross_loss = getattr(bot, "gross_loss", 0.0)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 0
        
        current_positions = len(getattr(bot, "positions", {}))
        from .config import MAX_CONCURRENT_POS
        max_pos = MAX_CONCURRENT_POS
        
        suggestions = []
        
        # CRITICAL: Circuit breaker
        circuit_breaker_triggered = getattr(bot, "circuit_breaker_triggered", False)
        if circuit_breaker_triggered:
            suggestions.append(("CIRCUIT BREAKER ACTIVE: Trading halted due to drawdown limit. Manual intervention required.", 100))
        
        # CRITICAL: Significant drawdown
        elif equity_change < -5.0:
            suggestions.append(("CRITICAL: Significant drawdown ({:.1f}%) - Reducing risk exposure".format(equity_change), 98))
        elif equity_change < -2.0:
            suggestions.append(("WARNING: Moderate drawdown ({:.1f}%) - Monitoring closely".format(equity_change), 85))
        
        # IMPORTANT: Poor performance
        if total_trades >= 10:
            if win_rate < 35 and profit_factor < 0.7:
                suggestions.append(("WARNING: Poor performance - Win rate {:.1f}%, PF {:.2f}".format(win_rate, profit_factor), 90))
            elif win_rate < 40:
                suggestions.append(("Low win rate: {:.1f}% - Entry signals may need strengthening".format(win_rate), 80))
        
        # IMPORTANT: Position capacity
        if current_positions >= max_pos:
            suggestions.append(("Max positions reached: {}/{} - At capacity".format(current_positions, max_pos), 88))
        
        # Remove old suggestions (older than 2 minutes)
        recent_info_with_times = []
        for info in self.recent_info:
            if isinstance(info, dict):
                if now - info.get('timestamp', 0) < 120:
                    recent_info_with_times.append(info)
            else:
                recent_info_with_times.append({'message': info, 'timestamp': now - 60})
        
        # Add new suggestions (up to 2 critical ones)
        existing_messages = {item.get('message', item) if isinstance(item, dict) else item for item in recent_info_with_times}
        
        added = 0
        for message, priority in suggestions:
            if added >= 2:  # Limit to 2 critical suggestions
                break
            if message not in existing_messages:
                recent_info_with_times.append({
                    'message': message,
                    'timestamp': now,
                    'priority': priority
                })
                existing_messages.add(message)
                added += 1
        
        # Update recent_info
        self.recent_info = deque(recent_info_with_times, maxlen=12)
    
    def check_and_switch_regime(self, bot):
        """
        Check market conditions and switch trading regime if appropriate.
        Includes minimum duration check to prevent frequent switching.
        
        HYBRID APPROACH:
        - Rule-based switching is the default (stable, predictable)
        - LLM can override via control_patch.json if it feels it's prudent
        - LLM override takes precedence over rule-based switching
        - Minimum regime duration enforced to prevent oscillation
        """
        # Check for active LLM override first
        if os.path.exists(CONTROL_PATH):
            try:
                data = safe_read_json(CONTROL_PATH, {})
                if data and data.get("regime_override"):
                    # LLM override is active, skip rule-based switching
                    return
            except Exception:
                pass  # Continue with rule-based check if file read fails
        
        # Throttle: Only check every 5 minutes (reduced frequency)
        now = time.time()
        if now - self.last_regime_check < self.regime_check_interval:
            return
        self.last_regime_check = now
        
        # Enforce minimum regime duration (prevent frequent switching)
        time_since_last_switch = now - self.last_regime_switch_time
        if time_since_last_switch < self.min_regime_duration:
            return  # Too soon to switch again
        
        # Get market conditions
        volatility_regime = getattr(bot, "volatility_regime", "Normal")
        spread_regime = getattr(bot, "spread_regime", "Normal")
        btc_trend = getattr(bot, "btc_trend", 0.0)
        
        # Get performance metrics
        win_count = getattr(bot, "win_count", 0)
        loss_count = getattr(bot, "loss_count", 0)
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades) if total_trades > 0 else 0.5
        
        gross_win = getattr(bot, "gross_win", 0.0)
        gross_loss = getattr(bot, "gross_loss", 0.0)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 1.0
        
        equity = bot.equity_now()
        start_equity = getattr(bot, "start_equity", equity)
        drawdown_pct = ((equity - start_equity) / start_equity * 100) if start_equity > 0 else 0
        
        # Calculate average hold time from recent trades
        recent_trades = list(getattr(bot, "recent_trades", []) or [])
        if recent_trades:
            hold_times = []
            for trade in recent_trades[-10:]:  # Last 10 trades
                entry_time = trade.get('entry_time', 0)
                exit_time = trade.get('exit_time', 0)
                if entry_time > 0 and exit_time > entry_time:
                    hold_times.append(exit_time - entry_time)
            avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0.0
        else:
            avg_hold_time = 0.0
        
        # Calculate signal quality
        signal_stats = getattr(bot, "signal_stats", {})
        signals_generated = signal_stats.get("signals_generated", 0)
        signals_blocked = signal_stats.get("signals_blocked", 0)
        total_signals = signals_generated + signals_blocked
        signal_quality = (signals_generated / total_signals) if total_signals > 0 else 0.5
        
        # Check if regime should be switched (rule-based)
        try:
            switch_result = self.regime_manager.should_switch_regime(
                volatility_regime=volatility_regime,
                spread_regime=spread_regime,
                btc_trend=btc_trend,
                win_rate=win_rate,
                profit_factor=profit_factor,
                drawdown_pct=drawdown_pct,
                avg_hold_time=avg_hold_time,
                signal_quality=signal_quality
            )
            
            if switch_result:
                new_regime, reason = switch_result
                self.regime_manager.set_regime(new_regime, f"Rule-based: {reason}")
                bot.current_regime = new_regime.value
                bot.regime_since = time.time()
                self.last_regime_switch_time = time.time()
                
                self.recent_info.append({
                    'message': f"Regime switch: {new_regime.value} ({reason})",
                    'timestamp': now,
                    'priority': 90
                })
                
                # Apply regime-specific parameters
                regime_config = self.regime_manager.get_current_config()
                self.apply_regime_config(bot, regime_config)
                
        except Exception as e:
            error_msg = f"Error in regime switch check: {e}"
            self.recent_errors.append(error_msg)
            try:
                from .logger import get_logger
                get_logger().warning(f"Regime switch error: {error_msg}")
            except ImportError:
                pass
    
    def apply_regime_config(self, bot, regime_config):
        """Apply regime-specific configuration to bot."""
        # Update config values via environment variables (they'll be read on next access)
        import os
        os.environ['MIN_SIGNAL_STRENGTH'] = str(regime_config.min_signal_strength)
        os.environ['MIN_SIGNAL_SCORE'] = str(regime_config.min_signal_score)
        os.environ['MIN_SPREAD_BPS'] = str(regime_config.min_spread_bps)
        os.environ['MAX_SPREAD_BPS'] = str(regime_config.max_spread_bps)
        os.environ['MIN_VOLUME_24H'] = str(regime_config.min_volume_24h)
        os.environ['MAX_LATENCY_MS'] = str(regime_config.max_latency_ms)
        os.environ['MAX_CONCURRENT_POS'] = str(regime_config.max_concurrent_positions)
        os.environ['MAX_CONCURRENT_POS_MIN'] = str(regime_config.max_concurrent_positions_min)
        os.environ['MAX_CONCURRENT_POS_MAX'] = str(regime_config.max_concurrent_positions_max)
        
        # Store regime config in bot for easy access
        bot.regime_config = regime_config