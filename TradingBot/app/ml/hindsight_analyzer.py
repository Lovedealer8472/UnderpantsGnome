"""
Hindsight Analysis Engine - Learn from what COULD have been done.

This module analyzes historical trades with perfect hindsight to determine:
- Should this trade have been taken?
- What was the optimal exit point?
- What position size would have maximized ROI?
- What R:R ratio would have been ideal?
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone


class HindsightAnalyzer:
    """Analyzes trades with hindsight to extract optimal outcomes."""
    
    def __init__(self, ohlcv_data_dir: Path):
        """
        Initialize hindsight analyzer.
        
        Args:
            ohlcv_data_dir: Directory containing OHLCV CSV files
        """
        self.ohlcv_data_dir = ohlcv_data_dir
        self.ohlcv_cache = {}  # Cache loaded OHLCV data
        
    def load_ohlcv_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a symbol.
        
        Args:
            symbol: Symbol like "BTC/USDT:USDT"
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        if symbol in self.ohlcv_cache:
            return self.ohlcv_cache[symbol]
        
        # Convert symbol format: "BTC/USDT:USDT" -> "BTC_USDT_USDT_5m.csv"
        symbol_clean = symbol.replace('/', '_').replace(':', '_')
        csv_path = self.ohlcv_data_dir / f"{symbol_clean}_5m.csv"
        
        if not csv_path.exists():
            return None
        
        try:
            # Try reading with headers first
            df = pd.read_csv(csv_path)
            
            # Check if file has headers
            if 'timestamp' not in df.columns and 'time' not in df.columns:
                # No headers - read again with column names
                df = pd.read_csv(csv_path, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            
            df = df.set_index('timestamp')
            self.ohlcv_cache[symbol] = df
            return df
        except Exception as e:
            print(f"Error loading OHLCV for {symbol}: {e}")
            return None
    
    def analyze_trade(self, trade: Dict) -> Dict:
        """
        Analyze a single trade with hindsight.
        
        Args:
            trade: Trade dictionary from trades.jsonl
            
        Returns:
            Enhanced trade with hindsight analysis
        """
        symbol = trade.get('symbol')
        entry_time_ms = trade.get('entry_time_ms')
        exit_time_ms = trade.get('exit_time_ms')
        entry_price = trade.get('entry_price')
        exit_price = trade.get('exit_price')
        side = trade.get('side')
        r_multiple = trade.get('r_multiple', 0)
        won = trade.get('won', False)
        
        # Load OHLCV data
        ohlcv = self.load_ohlcv_for_symbol(symbol)
        
        if ohlcv is None:
            # Can't do hindsight without price data
            return {
                'original_trade': trade,
                'hindsight': {
                    'data_available': False,
                    'should_have_entered': won,  # Fallback to actual outcome
                    'quality_score': 50.0  # Neutral
                }
            }
        
        # Convert timestamps
        entry_time = pd.to_datetime(entry_time_ms, unit='ms')
        exit_time = pd.to_datetime(exit_time_ms, unit='ms')
        
        # Get price data during trade
        trade_data = ohlcv[(ohlcv.index >= entry_time) & (ohlcv.index <= exit_time)]
        
        if trade_data.empty:
            return {
                'original_trade': trade,
                'hindsight': {
                    'data_available': False,
                    'should_have_entered': won,
                    'quality_score': 50.0
                }
            }
        
        # Calculate MFE and MAE
        mfe_r, mae_r = self._calculate_excursions(
            trade_data, entry_price, side, trade.get('entry_features', {}).get('atr_pct', 0.01)
        )
        
        # Find optimal exit
        optimal_exit_price, optimal_exit_time, optimal_r = self._find_optimal_exit(
            trade_data, entry_price, side, trade.get('entry_features', {}).get('atr_pct', 0.01)
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            r_multiple, mfe_r, mae_r, optimal_r, won
        )
        
        # Determine if trade should have been taken
        should_have_entered = self._should_have_entered(
            optimal_r, mfe_r, mae_r, quality_score
        )
        
        # Calculate ideal parameters
        ideal_sl_atr_mult, ideal_tp_r = self._calculate_ideal_params(
            mfe_r, mae_r, optimal_r
        )
        
        # Get price action after entry (for feature engineering)
        price_action_5m = self._get_price_action_after_entry(ohlcv, entry_time, minutes=5)
        price_action_15m = self._get_price_action_after_entry(ohlcv, entry_time, minutes=15)
        price_action_1h = self._get_price_action_after_entry(ohlcv, entry_time, minutes=60)
        price_action_4h = self._get_price_action_after_entry(ohlcv, entry_time, minutes=240)
        
        hindsight = {
            'data_available': True,
            'should_have_entered': should_have_entered,
            'optimal_exit_price': optimal_exit_price,
            'optimal_exit_time': optimal_exit_time.timestamp() * 1000 if optimal_exit_time else None,
            'optimal_r_multiple': optimal_r,
            'mfe_r': mfe_r,  # Maximum favorable excursion
            'mae_r': mae_r,  # Maximum adverse excursion
            'ideal_sl_atr_mult': ideal_sl_atr_mult,
            'ideal_tp_r': ideal_tp_r,
            'quality_score': quality_score,
            'price_action_5m': price_action_5m,
            'price_action_15m': price_action_15m,
            'price_action_1h': price_action_1h,
            'price_action_4h': price_action_4h,
        }
        
        return {
            'original_trade': trade,
            'hindsight': hindsight
        }
    
    def _calculate_excursions(
        self, 
        trade_data: pd.DataFrame, 
        entry_price: float, 
        side: str, 
        atr_pct: float
    ) -> Tuple[float, float]:
        """Calculate maximum favorable and adverse excursions in R."""
        if trade_data.empty:
            return 0.0, 0.0
        
        risk_amount = entry_price * atr_pct
        
        if side == 'long':
            # MFE: highest high during trade
            max_price = trade_data['high'].max()
            mfe_r = (max_price - entry_price) / risk_amount if risk_amount > 0 else 0
            
            # MAE: lowest low during trade
            min_price = trade_data['low'].min()
            mae_r = (entry_price - min_price) / risk_amount if risk_amount > 0 else 0
        else:  # short
            # MFE: lowest low during trade
            min_price = trade_data['low'].min()
            mfe_r = (entry_price - min_price) / risk_amount if risk_amount > 0 else 0
            
            # MAE: highest high during trade
            max_price = trade_data['high'].max()
            mae_r = (max_price - entry_price) / risk_amount if risk_amount > 0 else 0
        
        return max(0, mfe_r), max(0, mae_r)
    
    def _find_optimal_exit(
        self,
        trade_data: pd.DataFrame,
        entry_price: float,
        side: str,
        atr_pct: float
    ) -> Tuple[float, pd.Timestamp, float]:
        """Find the optimal exit point (highest profit before significant reversal)."""
        if trade_data.empty:
            return entry_price, None, 0.0
        
        risk_amount = entry_price * atr_pct
        best_r = -999
        best_price = entry_price
        best_time = trade_data.index[0]
        
        for idx, row in trade_data.iterrows():
            if side == 'long':
                current_r = (row['close'] - entry_price) / risk_amount if risk_amount > 0 else 0
            else:
                current_r = (entry_price - row['close']) / risk_amount if risk_amount > 0 else 0
            
            if current_r > best_r:
                best_r = current_r
                best_price = row['close']
                best_time = idx
        
        return best_price, best_time, best_r
    
    def _calculate_quality_score(
        self,
        actual_r: float,
        mfe_r: float,
        mae_r: float,
        optimal_r: float,
        won: bool
    ) -> float:
        """
        Calculate trade quality score (0-100).
        
        High quality = high MFE, low MAE, good actual outcome
        """
        score = 50.0  # Base
        
        # MFE component (0-25 points)
        if mfe_r > 3.0:
            score += 25
        elif mfe_r > 2.0:
            score += 20
        elif mfe_r > 1.0:
            score += 15
        elif mfe_r > 0.5:
            score += 10
        
        # MAE component (-15 to 0 points)
        if mae_r > 2.0:
            score -= 15
        elif mae_r > 1.5:
            score -= 10
        elif mae_r > 1.0:
            score -= 5
        
        # Outcome component (0-25 points)
        if won:
            if actual_r > 2.0:
                score += 25
            elif actual_r > 1.0:
                score += 20
            elif actual_r > 0.5:
                score += 15
            else:
                score += 10
        else:
            if actual_r > -0.5:
                score += 5  # Small loss is okay
            elif actual_r > -1.0:
                score -= 5
            else:
                score -= 15  # Big loss is bad
        
        # Efficiency component (0-10 points): Did we capture the opportunity?
        if optimal_r > 0:
            efficiency = actual_r / optimal_r if optimal_r > 0 else 0
            score += efficiency * 10
        
        return max(0, min(100, score))
    
    def _should_have_entered(
        self,
        optimal_r: float,
        mfe_r: float,
        mae_r: float,
        quality_score: float
    ) -> bool:
        """Determine if trade should have been taken with hindsight."""
        # Good opportunity: high MFE, low MAE, positive optimal R
        if optimal_r > 1.0 and mfe_r > 1.5 and mae_r < 1.0:
            return True
        
        # Decent opportunity: positive optimal R, reasonable risk
        if optimal_r > 0.5 and mae_r < 1.5:
            return True
        
        # Quality-based decision
        if quality_score > 65:
            return True
        
        return False
    
    def _calculate_ideal_params(
        self,
        mfe_r: float,
        mae_r: float,
        optimal_r: float
    ) -> Tuple[float, float]:
        """Calculate ideal SL and TP parameters."""
        # Ideal SL: Just beyond MAE
        ideal_sl_atr_mult = max(1.0, min(3.0, mae_r * 1.2))
        
        # Ideal TP: Capture most of MFE
        ideal_tp_r = max(1.0, min(5.0, mfe_r * 0.8))
        
        return ideal_sl_atr_mult, ideal_tp_r
    
    def _get_price_action_after_entry(
        self,
        ohlcv: pd.DataFrame,
        entry_time: pd.Timestamp,
        minutes: int
    ) -> Dict:
        """Get price action stats after entry for specified time window."""
        end_time = entry_time + pd.Timedelta(minutes=minutes)
        window_data = ohlcv[(ohlcv.index >= entry_time) & (ohlcv.index <= end_time)]
        
        if window_data.empty:
            return {'available': False}
        
        entry_price = window_data.iloc[0]['close']
        
        return {
            'available': True,
            'pct_change': ((window_data.iloc[-1]['close'] - entry_price) / entry_price * 100) if len(window_data) > 0 else 0,
            'high_pct': ((window_data['high'].max() - entry_price) / entry_price * 100),
            'low_pct': ((window_data['low'].min() - entry_price) / entry_price * 100),
            'volume_avg': window_data['volume'].mean(),
            'volatility': window_data['close'].pct_change().std() * 100
        }
    
    def analyze_trades_batch(self, trades: List[Dict], progress_callback=None) -> List[Dict]:
        """
        Analyze multiple trades with hindsight.
        
        Args:
            trades: List of trade dictionaries
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of enhanced trades with hindsight
        """
        results = []
        total = len(trades)
        
        for i, trade in enumerate(trades):
            if progress_callback and i % 100 == 0:
                progress_callback(i, total)
            
            enhanced_trade = self.analyze_trade(trade)
            results.append(enhanced_trade)
        
        if progress_callback:
            progress_callback(total, total)
        
        return results

