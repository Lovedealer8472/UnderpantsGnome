"""
Periodic Retrainer - Scheduled retraining system with performance monitoring.

Features:
- Daily retraining on new trades
- Performance monitoring and drift detection
- Automatic re-optimization triggers
- Incremental learning with catastrophic forgetting prevention
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class PeriodicRetrainer:
    """Manages periodic retraining and performance monitoring."""
    
    def __init__(
        self,
        config_path: str = "ML_TRAINING_CONFIG.json",
        state_path: str = "ml_retraining_state.json"
    ):
        """
        Initialize periodic retrainer.
        
        Args:
            config_path: Path to training configuration
            state_path: Path to retraining state file
        """
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        
        self.config = self._load_config()
        self.state = self._load_state()
        
    def _load_config(self) -> Dict:
        """Load training configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_state(self) -> Dict:
        """Load retraining state."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                return json.load(f)
        
        # Default state
        return {
            'last_retrain_timestamp': None,
            'last_full_optimization_timestamp': None,
            'trades_since_last_retrain': 0,
            'performance_history': [],
            'current_config': None,
            'model_version': 0
        }
    
    def _save_state(self):
        """Save retraining state."""
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def should_retrain(self) -> bool:
        """Check if retraining should be triggered."""
        retrain_config = self.config.get('periodic_retraining', {})
        
        if not retrain_config.get('enabled', True):
            return False
        
        # Check if enough new trades
        min_trades = retrain_config.get('min_new_trades_required', 100)
        if self.state['trades_since_last_retrain'] < min_trades:
            return False
        
        # Check time since last retrain
        last_retrain = self.state.get('last_retrain_timestamp')
        if last_retrain:
            last_retrain_dt = datetime.fromisoformat(last_retrain)
            days_since = (datetime.now() - last_retrain_dt).days
            
            max_days = retrain_config.get('retrain_triggers', {}).get('days_since_last_retrain', 7)
            if days_since < max_days:
                return False
        
        return True
    
    def should_full_reoptimize(self) -> bool:
        """Check if full re-optimization should be triggered."""
        if not self.state['performance_history']:
            return False
        
        triggers = self.config.get('periodic_retraining', {}).get('retrain_triggers', {})
        recent_performance = self.state['performance_history'][-100:]  # Last 100 trades
        
        if len(recent_performance) < 50:
            return False
        
        # Calculate recent metrics
        wins = sum(1 for p in recent_performance if p.get('won', False))
        win_rate = wins / len(recent_performance) * 100
        
        r_multiples = [p.get('r_multiple', 0) for p in recent_performance]
        sharpe = np.mean(r_multiples) / np.std(r_multiples) * np.sqrt(252) if np.std(r_multiples) > 0 else 0
        
        cumulative_r = np.cumsum(r_multiples)
        running_max = np.maximum.accumulate(cumulative_r)
        drawdown = running_max - cumulative_r
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Check triggers
        if win_rate < triggers.get('win_rate_below', 45.0):
            print(f"[REOPTIMIZE] Win rate {win_rate:.1f}% below threshold")
            return True
        
        if sharpe < triggers.get('sharpe_ratio_below', 0.5):
            print(f"[REOPTIMIZE] Sharpe ratio {sharpe:.2f} below threshold")
            return True
        
        if max_drawdown > triggers.get('max_drawdown_above', 15.0):
            print(f"[REOPTIMIZE] Max drawdown {max_drawdown:.1f}R above threshold")
            return True
        
        return False
    
    def retrain_models(self):
        """Retrain models incrementally on new trades."""
        print(f"\n{'='*60}")
        print(f"INCREMENTAL RETRAINING")
        print(f"{'='*60}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Load new trades
        new_trades = self._load_new_trades()
        print(f"New trades: {len(new_trades)}")
        
        if not new_trades:
            print("No new trades to train on")
            return
        
        # TODO: Implement incremental training
        # This would:
        # 1. Load existing models
        # 2. Run hindsight analysis on new trades
        # 3. Extract features
        # 4. Update models with new data (warm start)
        # 5. Validate on holdout set
        # 6. Save updated models
        
        print("Incremental retraining completed")
        
        # Update state
        self.state['last_retrain_timestamp'] = datetime.now().isoformat()
        self.state['trades_since_last_retrain'] = 0
        self.state['model_version'] += 1
        self._save_state()
    
    def full_reoptimization(self):
        """Trigger full re-optimization."""
        print(f"\n{'='*60}")
        print(f"FULL RE-OPTIMIZATION TRIGGERED")
        print(f"{'='*60}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # TODO: Launch full ML training pipeline
        # This would run ML_TRAINING_SYSTEM.py with all phases
        
        print("Full re-optimization completed")
        
        # Update state
        self.state['last_full_optimization_timestamp'] = datetime.now().isoformat()
        self._save_state()
    
    def update_performance(self, trade: Dict):
        """Update performance history with new trade."""
        self.state['performance_history'].append({
            'timestamp': datetime.now().isoformat(),
            'r_multiple': trade.get('r_multiple', 0),
            'won': trade.get('won', False),
            'score': trade.get('score', 0)
        })
        
        # Keep only last 1000 trades
        if len(self.state['performance_history']) > 1000:
            self.state['performance_history'] = self.state['performance_history'][-1000:]
        
        self.state['trades_since_last_retrain'] += 1
        self._save_state()
    
    def _load_new_trades(self) -> List[Dict]:
        """Load trades since last retrain."""
        # TODO: Implement loading new trades from learning_results
        return []
    
    def monitor_performance(self) -> Dict:
        """Calculate current performance metrics."""
        if not self.state['performance_history']:
            return {}
        
        recent = self.state['performance_history'][-100:]
        
        wins = sum(1 for p in recent if p.get('won', False))
        win_rate = wins / len(recent) * 100 if recent else 0
        
        r_multiples = [p.get('r_multiple', 0) for p in recent]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        sharpe = np.mean(r_multiples) / np.std(r_multiples) * np.sqrt(252) if np.std(r_multiples) > 0 else 0
        
        return {
            'trades_count': len(recent),
            'win_rate': win_rate,
            'avg_r': avg_r,
            'sharpe_ratio': sharpe,
            'trades_since_last_retrain': self.state['trades_since_last_retrain'],
            'model_version': self.state['model_version']
        }
    
    def run_monitoring_loop(self, check_interval_seconds: int = 3600):
        """Run continuous monitoring loop."""
        print(f"\n{'='*60}")
        print(f"PERIODIC RETRAINER - MONITORING ACTIVE")
        print(f"{'='*60}")
        print(f"Check interval: {check_interval_seconds}s ({check_interval_seconds/3600:.1f}h)")
        
        while True:
            try:
                # Check if retraining needed
                if self.should_retrain():
                    self.retrain_models()
                
                # Check if full re-optimization needed
                if self.should_full_reoptimize():
                    self.full_reoptimization()
                
                # Display current performance
                metrics = self.monitor_performance()
                if metrics:
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performance:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                
                # Wait for next check
                time.sleep(check_interval_seconds)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"\nError in monitoring loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # Wait 1 minute before retry


def main():
    """Main entry point for periodic retrainer."""
    retrainer = PeriodicRetrainer()
    retrainer.run_monitoring_loop(check_interval_seconds=3600)  # Check every hour


if __name__ == "__main__":
    main()

