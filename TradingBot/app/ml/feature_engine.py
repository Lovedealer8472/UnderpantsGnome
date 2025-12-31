"""
Feature Engineering - Extract comprehensive features from trades and market data.

Extracts 80+ features across multiple categories:
- Entry features: Technical indicators, price patterns, volume
- Market context: BTC correlation, volatility regime, time features
- Exit features: Position metrics, market changes, risk metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import math


class FeatureEngine:
    """Extracts features from trades and market data."""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, enhanced_trade: Dict) -> Dict[str, float]:
        """
        Extract all features from an enhanced trade (with hindsight).
        
        Args:
            enhanced_trade: Trade with hindsight analysis
            
        Returns:
            Dictionary of feature_name -> value
        """
        trade = enhanced_trade['original_trade']
        hindsight = enhanced_trade.get('hindsight', {})
        
        features = {}
        
        # Entry features from trade
        features.update(self._extract_entry_features(trade))
        
        # Hindsight features
        features.update(self._extract_hindsight_features(hindsight))
        
        # Time features
        features.update(self._extract_time_features(trade))
        
        # Derived features
        features.update(self._extract_derived_features(trade, hindsight))
        
        return features
    
    def _extract_entry_features(self, trade: Dict) -> Dict[str, float]:
        """Extract features from trade entry."""
        entry_features = trade.get('entry_features', {})
        
        features = {
            # Technical indicators
            'rsi': entry_features.get('rsi', 50.0),
            'spread_bps': entry_features.get('spread_bps', 0.0),
            'vol_bps': entry_features.get('vol_bps', 0.0),
            'atr_pct': entry_features.get('atr_pct', 0.01) * 100,  # Convert to percentage
            
            # Price momentum
            'pct_change_24h': entry_features.get('pct_change_24h', 0.0),
            'pct_change_1h': entry_features.get('pct_change_1h', 0.0),
            'pct_change_4h': entry_features.get('pct_change_4h', 0.0),
            'momentum_gate': entry_features.get('momentum_gate', 0.0),
            
            # Signal quality
            'score': trade.get('score', 50.0),
            'strength': entry_features.get('strength', 0.5),
            
            # Signal type (one-hot encoded)
            'signal_type_momentum': 1.0 if entry_features.get('signal_type') == 'momentum' else 0.0,
            'signal_type_mean_reversion': 1.0 if entry_features.get('signal_type') == 'mean_reversion' else 0.0,
            'signal_type_breakout': 1.0 if entry_features.get('signal_type') == 'breakout' else 0.0,
            
            # Side
            'side_long': 1.0 if trade.get('side') == 'long' else 0.0,
            'side_short': 1.0 if trade.get('side') == 'short' else 0.0,
        }
        
        # RSI-based features
        rsi = features['rsi']
        features['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
        features['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
        features['rsi_extreme'] = 1.0 if rsi < 20 or rsi > 80 else 0.0
        features['rsi_neutral'] = 1.0 if 40 <= rsi <= 60 else 0.0
        features['rsi_distance_from_50'] = abs(rsi - 50.0)
        
        # Momentum features
        mom_24h = features['pct_change_24h']
        features['momentum_strong'] = 1.0 if abs(mom_24h) > 5.0 else 0.0
        features['momentum_moderate'] = 1.0 if 2.0 < abs(mom_24h) <= 5.0 else 0.0
        features['momentum_weak'] = 1.0 if abs(mom_24h) <= 2.0 else 0.0
        features['momentum_direction'] = 1.0 if mom_24h > 0 else -1.0
        
        # Volatility features
        atr = features['atr_pct']
        features['volatility_high'] = 1.0 if atr > 3.0 else 0.0
        features['volatility_normal'] = 1.0 if 1.0 <= atr <= 3.0 else 0.0
        features['volatility_low'] = 1.0 if atr < 1.0 else 0.0
        
        # Spread quality
        spread = features['spread_bps']
        features['spread_tight'] = 1.0 if spread < 10 else 0.0
        features['spread_normal'] = 1.0 if 10 <= spread < 30 else 0.0
        features['spread_wide'] = 1.0 if spread >= 30 else 0.0
        
        return features
    
    def _extract_hindsight_features(self, hindsight: Dict) -> Dict[str, float]:
        """Extract features from hindsight analysis."""
        if not hindsight.get('data_available', False):
            return {
                'mfe_r': 0.0,
                'mae_r': 0.0,
                'optimal_r': 0.0,
                'quality_score': 50.0,
                'should_have_entered': 0.5,
            }
        
        features = {
            'mfe_r': hindsight.get('mfe_r', 0.0),
            'mae_r': hindsight.get('mae_r', 0.0),
            'optimal_r': hindsight.get('optimal_r_multiple', 0.0),
            'quality_score': hindsight.get('quality_score', 50.0),
            'should_have_entered': int(hindsight.get('should_have_entered', False)),  # Convert to int (0 or 1)
            'ideal_sl_atr_mult': hindsight.get('ideal_sl_atr_mult', 1.5),
            'ideal_tp_r': hindsight.get('ideal_tp_r', 2.0),
        }
        
        # MFE categories
        mfe = features['mfe_r']
        features['mfe_excellent'] = 1.0 if mfe > 3.0 else 0.0
        features['mfe_good'] = 1.0 if 2.0 < mfe <= 3.0 else 0.0
        features['mfe_moderate'] = 1.0 if 1.0 < mfe <= 2.0 else 0.0
        features['mfe_poor'] = 1.0 if mfe <= 1.0 else 0.0
        
        # MAE categories
        mae = features['mae_r']
        features['mae_low'] = 1.0 if mae < 0.5 else 0.0
        features['mae_moderate'] = 1.0 if 0.5 <= mae < 1.0 else 0.0
        features['mae_high'] = 1.0 if mae >= 1.0 else 0.0
        
        # Risk/reward ratio
        if mae > 0:
            features['mfe_mae_ratio'] = mfe / mae
        else:
            features['mfe_mae_ratio'] = mfe * 10  # High ratio if no adverse excursion
        
        # Price action features
        for timeframe in ['5m', '15m', '1h', '4h']:
            pa = hindsight.get(f'price_action_{timeframe}', {})
            if pa.get('available', False):
                features[f'pa_{timeframe}_pct_change'] = pa.get('pct_change', 0.0)
                features[f'pa_{timeframe}_high_pct'] = pa.get('high_pct', 0.0)
                features[f'pa_{timeframe}_low_pct'] = pa.get('low_pct', 0.0)
                features[f'pa_{timeframe}_volatility'] = pa.get('volatility', 0.0)
            else:
                features[f'pa_{timeframe}_pct_change'] = 0.0
                features[f'pa_{timeframe}_high_pct'] = 0.0
                features[f'pa_{timeframe}_low_pct'] = 0.0
                features[f'pa_{timeframe}_volatility'] = 0.0
        
        return features
    
    def _extract_time_features(self, trade: Dict) -> Dict[str, float]:
        """Extract time-based features."""
        entry_time_ms = trade.get('entry_time_ms', 0)
        entry_time = datetime.fromtimestamp(entry_time_ms / 1000.0)
        
        hour = entry_time.hour
        day_of_week = entry_time.weekday()
        
        features = {
            # Hour (cyclical encoding)
            'hour_sin': math.sin(2 * math.pi * hour / 24),
            'hour_cos': math.cos(2 * math.pi * hour / 24),
            
            # Day of week (cyclical encoding)
            'dow_sin': math.sin(2 * math.pi * day_of_week / 7),
            'dow_cos': math.cos(2 * math.pi * day_of_week / 7),
            
            # Trading session
            'session_asia': 1.0 if 0 <= hour < 8 else 0.0,
            'session_europe': 1.0 if 8 <= hour < 16 else 0.0,
            'session_us': 1.0 if 16 <= hour < 24 else 0.0,
            
            # Weekend
            'is_weekend': 1.0 if day_of_week >= 5 else 0.0,
        }
        
        return features
    
    def _extract_derived_features(self, trade: Dict, hindsight: Dict) -> Dict[str, float]:
        """Extract derived/interaction features."""
        features = {}
        
        # Trade duration
        duration_sec = trade.get('duration_sec', 0)
        features['duration_hours'] = duration_sec / 3600.0
        features['duration_short'] = 1.0 if duration_sec < 1800 else 0.0  # < 30 min
        features['duration_medium'] = 1.0 if 1800 <= duration_sec < 7200 else 0.0  # 30min - 2h
        features['duration_long'] = 1.0 if duration_sec >= 7200 else 0.0  # > 2h
        
        # Actual outcome
        r_multiple = trade.get('r_multiple', 0.0)
        won = trade.get('won', False)
        
        features['r_multiple'] = r_multiple
        features['won'] = int(won)  # Convert to int (0 or 1)
        features['r_positive'] = 1.0 if r_multiple > 0 else 0.0
        features['r_excellent'] = 1.0 if r_multiple > 2.0 else 0.0
        features['r_good'] = 1.0 if 1.0 < r_multiple <= 2.0 else 0.0
        features['r_breakeven'] = 1.0 if -0.2 <= r_multiple <= 0.2 else 0.0
        features['r_loss'] = 1.0 if r_multiple < -0.2 else 0.0
        
        # Exit reason
        reason = trade.get('reason', 'unknown')
        features['exit_take_profit'] = 1.0 if 'take_profit' in reason else 0.0
        features['exit_stop_loss'] = 1.0 if 'stop_loss' in reason else 0.0
        features['exit_time'] = 1.0 if 'time' in reason else 0.0
        features['exit_trailing'] = 1.0 if 'trail' in reason else 0.0
        
        # Interaction features
        entry_features = trade.get('entry_features', {})
        rsi = entry_features.get('rsi', 50.0)
        momentum = entry_features.get('pct_change_24h', 0.0)
        
        # RSI + Momentum interaction
        features['rsi_momentum_aligned'] = 1.0 if (rsi > 50 and momentum > 0) or (rsi < 50 and momentum < 0) else 0.0
        
        # Score + Volatility interaction
        score = trade.get('score', 50.0)
        atr = entry_features.get('atr_pct', 0.01) * 100
        features['score_volatility_product'] = (score / 100.0) * (atr / 3.0)
        
        return features
    
    def extract_features_batch(self, enhanced_trades: List[Dict]) -> pd.DataFrame:
        """
        Extract features from multiple trades and return as DataFrame.
        
        Args:
            enhanced_trades: List of enhanced trades with hindsight
            
        Returns:
            DataFrame with features
        """
        feature_dicts = []
        
        for trade in enhanced_trades:
            features = self.extract_features(trade)
            feature_dicts.append(features)
        
        df = pd.DataFrame(feature_dicts)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        # Handle any NaN values
        df = df.fillna(0.0)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis."""
        groups = {
            'technical': [f for f in self.feature_names if any(x in f for x in ['rsi', 'atr', 'spread', 'vol_bps'])],
            'momentum': [f for f in self.feature_names if 'momentum' in f or 'pct_change' in f],
            'hindsight': [f for f in self.feature_names if any(x in f for x in ['mfe', 'mae', 'optimal', 'quality'])],
            'time': [f for f in self.feature_names if any(x in f for x in ['hour', 'dow', 'session', 'weekend'])],
            'outcome': [f for f in self.feature_names if any(x in f for x in ['r_multiple', 'won', 'exit'])],
            'derived': [f for f in self.feature_names if any(x in f for x in ['duration', 'aligned', 'product', 'ratio'])],
        }
        return groups

