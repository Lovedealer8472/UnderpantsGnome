"""
OHLCV Simulator - Validate configurations on historical price data.

Simulates realistic trading with:
- Bar-by-bar replay of 5m OHLCV data
- Realistic slippage and fees
- Stop-loss and take-profit execution
- Trailing stops
- Time-based exits
- Partial position closes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class SimulatedTrade:
    """Represents a simulated trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    r_multiple: float
    won: bool
    exit_reason: str
    duration_bars: int
    entry_score: float


class OHLCVSimulator:
    """Simulates trading on historical OHLCV data."""
    
    def __init__(
        self,
        ohlcv_data_dir: Path,
        config: Dict,
        fee_rate: float = 0.0004,  # 0.04% taker fee
        slippage_bps: float = 2.0   # 2 bps slippage
    ):
        """
        Initialize OHLCV simulator.
        
        Args:
            ohlcv_data_dir: Directory containing OHLCV CSV files
            config: Trading configuration to test
            fee_rate: Trading fee rate (e.g., 0.0004 = 0.04%)
            slippage_bps: Slippage in basis points
        """
        self.ohlcv_data_dir = Path(ohlcv_data_dir)
        self.config = config
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_bps / 10000.0
        
        self.ohlcv_cache = {}
        
    def load_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load OHLCV data for symbol."""
        if symbol in self.ohlcv_cache:
            return self.ohlcv_cache[symbol]
        
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
    
    def simulate_trade(
        self,
        symbol: str,
        entry_time: pd.Timestamp,
        side: str,
        entry_price: float,
        atr_pct: float,
        entry_score: float
    ) -> Optional[SimulatedTrade]:
        """
        Simulate a single trade.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            side: 'long' or 'short'
            entry_price: Entry price
            atr_pct: ATR as percentage (e.g., 0.015 = 1.5%)
            entry_score: Entry signal score
            
        Returns:
            SimulatedTrade or None if simulation failed
        """
        ohlcv = self.load_ohlcv(symbol)
        if ohlcv is None:
            return None
        
        # Get data after entry
        trade_data = ohlcv[ohlcv.index >= entry_time]
        if trade_data.empty:
            return None
        
        # Calculate stop-loss and take-profit
        sl_atr_mult = self.config.get('SL_ATR_MULTIPLIER', 1.5)
        risk_amount = entry_price * atr_pct * sl_atr_mult
        
        # Determine exit profile based on score
        if entry_score >= self.config.get('R_EXIT_RUNNER_SCORE_MIN', 70):
            # Runner profile
            tp_r = self.config.get('R_RUNNER_MAX_R_NORMAL', 4.0)
            time_stop_bars = self.config.get('R_RUNNER_TIME_STOP_BARS', 288)
            trail_start_r = self.config.get('R_RUNNER_TRAIL_START_R', 1.5)
            trail_atr_mult = self.config.get('R_RUNNER_TRAIL_ATR_MULT_NORMAL', 2.0)
        elif entry_score >= self.config.get('R_EXIT_STANDARD_SCORE_MIN', 55):
            # Standard profile
            tp_r = self.config.get('R_STANDARD_MAX_R', 3.0)
            time_stop_bars = self.config.get('R_STANDARD_TIME_STOP_BARS', 72)
            trail_start_r = self.config.get('R_STANDARD_TRAIL_START_R', 1.5)
            trail_atr_mult = self.config.get('R_STANDARD_TRAIL_ATR_MULT', 1.5)
        else:
            # Scalp profile
            tp_r = self.config.get('R_SCALP_TP_R', 1.0)
            time_stop_bars = self.config.get('R_SCALP_TIME_STOP_BARS', 24)
            trail_start_r = None  # No trailing for scalps
            trail_atr_mult = None
        
        # Calculate SL/TP prices
        if side == 'long':
            stop_loss = entry_price - risk_amount
            take_profit = entry_price + (risk_amount * tp_r)
        else:
            stop_loss = entry_price + risk_amount
            take_profit = entry_price - (risk_amount * tp_r)
        
        # Simulate bar by bar
        trailing_stop = None
        max_favorable_r = 0
        
        for i, (timestamp, bar) in enumerate(trade_data.iterrows()):
            if i >= time_stop_bars:
                # Time stop hit
                exit_price = bar['close']
                exit_time = timestamp
                exit_reason = 'time_stop'
                break
            
            # Check stop-loss and take-profit
            if side == 'long':
                # Check SL (use low of bar)
                if trailing_stop is not None:
                    if bar['low'] <= trailing_stop:
                        exit_price = trailing_stop
                        exit_time = timestamp
                        exit_reason = 'trailing_stop'
                        break
                elif bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_time = timestamp
                    exit_reason = 'stop_loss'
                    break
                
                # Check TP (use high of bar)
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_time = timestamp
                    exit_reason = 'take_profit'
                    break
                
                # Update trailing stop
                current_r = (bar['close'] - entry_price) / risk_amount if risk_amount > 0 else 0
                max_favorable_r = max(max_favorable_r, current_r)
                
                if trail_start_r is not None and current_r >= trail_start_r:
                    # Activate trailing stop
                    trail_distance = entry_price * atr_pct * trail_atr_mult
                    new_trailing_stop = bar['close'] - trail_distance
                    if trailing_stop is None or new_trailing_stop > trailing_stop:
                        trailing_stop = new_trailing_stop
            
            else:  # short
                # Check SL (use high of bar)
                if trailing_stop is not None:
                    if bar['high'] >= trailing_stop:
                        exit_price = trailing_stop
                        exit_time = timestamp
                        exit_reason = 'trailing_stop'
                        break
                elif bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_time = timestamp
                    exit_reason = 'stop_loss'
                    break
                
                # Check TP (use low of bar)
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_time = timestamp
                    exit_reason = 'take_profit'
                    break
                
                # Update trailing stop
                current_r = (entry_price - bar['close']) / risk_amount if risk_amount > 0 else 0
                max_favorable_r = max(max_favorable_r, current_r)
                
                if trail_start_r is not None and current_r >= trail_start_r:
                    # Activate trailing stop
                    trail_distance = entry_price * atr_pct * trail_atr_mult
                    new_trailing_stop = bar['close'] + trail_distance
                    if trailing_stop is None or new_trailing_stop < trailing_stop:
                        trailing_stop = new_trailing_stop
        else:
            # Reached end of data without exit
            exit_price = trade_data.iloc[-1]['close']
            exit_time = trade_data.index[-1]
            exit_reason = 'end_of_data'
            i = len(trade_data) - 1
        
        # Apply slippage
        if side == 'long':
            exit_price = exit_price * (1 - self.slippage_pct)
        else:
            exit_price = exit_price * (1 + self.slippage_pct)
        
        # Calculate R-multiple
        if side == 'long':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        
        # Account for fees
        total_fees = (entry_price + exit_price) * self.fee_rate
        pnl -= total_fees
        
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0
        won = r_multiple > 0
        
        return SimulatedTrade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            r_multiple=r_multiple,
            won=won,
            exit_reason=exit_reason,
            duration_bars=i + 1,
            entry_score=entry_score
        )
    
    def simulate_trades_from_history(
        self,
        trades: List[Dict],
        max_trades: Optional[int] = None
    ) -> List[SimulatedTrade]:
        """
        Simulate trades from historical trade list.
        
        Args:
            trades: List of historical trades
            max_trades: Maximum number of trades to simulate
            
        Returns:
            List of simulated trades
        """
        simulated = []
        
        for i, trade in enumerate(trades):
            if max_trades and i >= max_trades:
                break
            
            if (i + 1) % 100 == 0:
                print(f"  Simulated {i+1}/{min(len(trades), max_trades or len(trades))} trades...")
            
            entry_time = pd.to_datetime(trade['entry_time_ms'], unit='ms')
            entry_features = trade.get('entry_features', {})
            
            sim_trade = self.simulate_trade(
                symbol=trade['symbol'],
                entry_time=entry_time,
                side=trade['side'],
                entry_price=trade['entry_price'],
                atr_pct=entry_features.get('atr_pct', 0.015),
                entry_score=trade.get('score', 50.0)
            )
            
            if sim_trade:
                simulated.append(sim_trade)
        
        return simulated
    
    def calculate_metrics(self, trades: List[SimulatedTrade]) -> Dict[str, float]:
        """Calculate performance metrics from simulated trades."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_r': 0,
                'profit_factor': 0,
                'roi_per_day': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        r_multiples = [t.r_multiple for t in trades]
        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        
        # Basic metrics
        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_r = np.mean(r_multiples)
        
        # Profit factor
        gross_wins = sum(t.r_multiple for t in wins)
        gross_losses = abs(sum(t.r_multiple for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
        
        # ROI per day
        if trades:
            start_time = min(t.entry_time for t in trades)
            end_time = max(t.exit_time for t in trades)
            days_active = (end_time - start_time).total_seconds() / 86400.0
            total_r = sum(r_multiples)
            roi_per_day = total_r / days_active if days_active > 0 else 0
        else:
            roi_per_day = 0
        
        # Sharpe ratio
        if len(r_multiples) > 1:
            sharpe_ratio = np.mean(r_multiples) / np.std(r_multiples) * np.sqrt(252) if np.std(r_multiples) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_r = np.cumsum(r_multiples)
        running_max = np.maximum.accumulate(cumulative_r)
        drawdown = running_max - cumulative_r
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_r': avg_r,
            'profit_factor': profit_factor,
            'roi_per_day': roi_per_day,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_r': sum(r_multiples),
            'gross_wins': gross_wins,
            'gross_losses': gross_losses
        }

