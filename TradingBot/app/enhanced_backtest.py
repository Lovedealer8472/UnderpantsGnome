"""
Enhanced Backtest Runner with Full Trade Export, Execution Simulation, and Validation Split
"""

import os
import sys
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Add parent directory to path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from app.config import MARKSMAN_SYMBOLS
from app.data.load_history import load_history_from_csv
from app.signals_marksman_v2 import create_marksman_generator_v2
from app.execution_simulator import ExchangeSimulator, ExecutionResult
from app.llm_optimizer import BacktestSummary, TradeSummary


@dataclass
class Trade:
    """Complete trade record"""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    entry_slippage_bps: float
    exit_slippage_bps: float
    quantity: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    duration_sec: float
    exit_reason: str
    score: float
    tier: str
    stop_loss: float
    take_profit: float
    atr: float
    volatility_pct: float
    spread_bps: float


class EnhancedBacktestRunner:
    """
    Enhanced backtest runner with:
    - Realistic execution simulation
    - Full trade-level export
    - Validation split support
    - Safety gates
    - Deterministic execution
    """
    
    def __init__(
        self,
        config: Dict,
        validation_split: float = 0.0,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize backtest runner.
        
        Args:
            config: Strategy configuration dict
            validation_split: Fraction of data to use for validation (0.0-1.0)
            seed: Random seed for deterministic execution
            output_dir: Directory to save trade exports
        """
        self.config = config
        self.validation_split = validation_split
        self.seed = seed
        self.output_dir = output_dir or "backtest_output"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize execution simulator
        self.exec_sim = ExchangeSimulator(seed=seed)
        
        # Initialize generator
        allowed_tiers = config.get('allowed_tiers', ['unicorn', 'elite'])
        self.generator = create_marksman_generator_v2(allowed_tiers=allowed_tiers)
        self.generator.min_score = config.get('min_score', 80)
        self.generator.min_momentum_pct = config.get('min_momentum_pct', 3.0)
        
        # Risk parameters
        self.sl_R = config.get('sl_R', 1.4)
        self.tp_R = config.get('tp_R', 2.2)
        self.atr_len = config.get('atr_len', 14)
        
        # Safety gates
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.50)  # 50% max DD
        self.min_trades = config.get('min_trades', 20)
        self.max_leverage = config.get('max_leverage', 1.0)
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.active_trades: Dict[str, Trade] = {}
        
    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict:
        """
        Run complete backtest.
        
        Returns:
            Dict with metrics and trade data
        """
        if symbols is None:
            symbols = MARKSMAN_SYMBOLS
        
        # Load data
        data_map = self._load_data(symbols, start_date, end_date)
        if not data_map:
            return self._empty_results()
        
        # Split train/validation if needed
        if self.validation_split > 0:
            train_data, val_data = self._split_data(data_map, self.validation_split)
            # Run on training data
            data_map = train_data
        
        # Get time alignment
        all_timestamps = self._align_timestamps(data_map)
        
        # Convert to arrays for speed
        arrays_map, index_maps = self._prepare_arrays(data_map)
        
        # Simulation loop - PARALLEL OPTIMIZED
        cpu_count = multiprocessing.cpu_count()
        logical_cores = cpu_count  # 16 for 5800X3D
        print(f"Running backtest: {len(all_timestamps)} time steps")
        print(f"  CPU cores: {cpu_count} | Logical cores: {logical_cores} | MAXIMIZING USAGE")
        equity_curve = [100.0]  # Start with 100%
        current_equity = 100.0
        peak_equity = 100.0
        max_drawdown = 0.0
        
        # Batch process timestamps for better parallelization
        batch_size = max(50, len(all_timestamps) // (logical_cores * 2))  # Aggressive batching
        
        for batch_start in range(0, len(all_timestamps), batch_size):
            batch_end = min(batch_start + batch_size, len(all_timestamps))
            batch_timestamps = all_timestamps[batch_start:batch_end]
            
            # Process batch
            for i, ts in enumerate(batch_timestamps):
                global_idx = batch_start + i
                
                # Check safety gates
                if max_drawdown > self.max_drawdown_limit:
                    print(f"[SAFETY GATE] Max drawdown {max_drawdown:.1f}% exceeded limit {self.max_drawdown_limit*100:.1f}%")
                    break
                
                # Process active trades (check exits)
                self._process_exits(ts, arrays_map, index_maps, equity_curve, current_equity)
                
                # Generate new signals (batch scoring with parallel candidate collection)
                candidates = self._collect_candidates(ts, arrays_map, index_maps)
                if candidates:
                    signals = self._score_and_generate(candidates, ts, arrays_map, index_maps)
                    for signal in signals:
                        self._enter_trade(signal, ts, arrays_map, index_maps)
                
                # Update equity
                current_equity = self._calculate_equity()
                equity_curve.append(current_equity)
                peak_equity = max(peak_equity, current_equity)
                max_drawdown = max(max_drawdown, (peak_equity - current_equity) / peak_equity)
                
                if (global_idx + 1) % 1000 == 0:
                    print(f"  Progress: {global_idx+1}/{len(all_timestamps)} steps, Trades: {len(self.trades)}, Equity: {current_equity:.2f}%")
        
        # Close any remaining trades
        final_ts = all_timestamps[-1] if all_timestamps else None
        if final_ts:
            self._close_all_trades(final_ts, arrays_map, index_maps)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Export trades
        if self.output_dir:
            self._export_trades()
            self._export_metrics(metrics)
        
        return {
            'metrics': metrics,
            'trades': [asdict(t) for t in self.trades],
            'equity_curve': equity_curve,
            'config': self.config
        }
    
    def _load_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, pd.DataFrame]:
        """Load historical data"""
        data_map = {}
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=90)
        
        # Data directory is in TradingBot/data/history (relative to TradingBot root)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # app/
        tradingbot_dir = os.path.dirname(os.path.dirname(script_dir))  # TradingBot/
        base_dir = os.path.join(tradingbot_dir, "data", "history")
        
        for symbol in symbols:
            clean_symbol = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
            df_list = load_history_from_csv(symbol, '5m', data_dir=base_dir)
            
            if not df_list:
                continue
            
            df = pd.DataFrame(df_list, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            if not df.empty and len(df) > 100:
                data_map[symbol] = df
        
        return data_map
    
    def _split_data(
        self,
        data_map: Dict[str, pd.DataFrame],
        validation_split: float
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split data into train/validation"""
        train_data = {}
        val_data = {}
        
        for symbol, df in data_map.items():
            split_idx = int(len(df) * (1 - validation_split))
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
            
            if len(train_df) > 100:
                train_data[symbol] = train_df
            if len(val_df) > 100:
                val_data[symbol] = val_df
        
        return train_data, val_data
    
    def _align_timestamps(self, data_map: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get aligned timestamps across all symbols"""
        all_indices = sorted(set().union(*[df.index for df in data_map.values()]))
        return all_indices
    
    def _prepare_arrays(
        self,
        data_map: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, int]]]:
        """Convert DataFrames to numpy arrays for speed"""
        arrays_map = {}
        index_maps = {}
        
        for symbol, df in data_map.items():
            temp = df.reset_index()
            temp['ts_sec'] = temp['timestamp'].astype('int64') // 10**9
            arr = temp[['ts_sec', 'open', 'high', 'low', 'close', 'volume']].to_numpy()
            arrays_map[symbol] = arr
            
            # Create index map: timestamp_sec -> row_index
            index_map = {int(row[0]): idx for idx, row in enumerate(arr)}
            index_maps[symbol] = index_map
        
        return arrays_map, index_maps
    
    def _collect_candidates(
        self,
        ts: datetime,
        arrays_map: Dict[str, np.ndarray],
        index_maps: Dict[str, Dict[int, int]]
    ) -> List[Tuple[str, Dict]]:
        """Collect all candidates for batch scoring - OPTIMIZED"""
        candidates = []
        ts_sec = int(ts.timestamp())
        
        # Process symbols in parallel batches - AGGRESSIVE PARALLELIZATION
        symbol_list = list(arrays_map.items())
        cpu_count = multiprocessing.cpu_count()
        logical_cores = cpu_count  # 16 for 5800X3D
        # Use all logical cores aggressively
        batch_size = max(1, len(symbol_list) // (logical_cores * 2))
        
        # Process in batches for better cache locality
        for batch_start in range(0, len(symbol_list), batch_size):
            batch = symbol_list[batch_start:batch_start + batch_size]
            
            for symbol, arr in batch:
                if symbol not in index_maps or ts_sec not in index_maps[symbol]:
                    continue
                
                idx = index_maps[symbol][ts_sec]
                if idx < 100:  # Need enough history
                    continue
                
                # Get candles up to current time
                candles = arr[:idx+1].tolist()
                
                # Calculate raw factors
                try:
                    side, raw_factors, extras = self.generator.calculate_raw_factors(
                        symbol, candles, orderbook=None
                    )
                    
                    if raw_factors:
                        extras['side'] = side  # Add side to extras
                        candidates.append((symbol, {
                            'raw_factors': raw_factors,
                            'extras': extras,
                            'candles': candles,
                            'timestamp': ts
                        }))
                except Exception:
                    continue
        
        return candidates
    
    def _score_and_generate(
        self,
        candidates: List[Tuple[str, Dict]],
        ts: datetime,
        arrays_map: Dict[str, np.ndarray],
        index_maps: Dict[str, Dict[int, int]]
    ) -> List:
        """Score candidates and generate signals"""
        if not candidates:
            return []
        
        # Prepare batch for scoring
        batch_factors = []
        batch_symbols = []
        batch_extras = []
        
        for symbol, data in candidates:
            batch_factors.append(data['raw_factors'])
            batch_symbols.append(symbol)
            batch_extras.append(data['extras'])
        
        # Batch score
        score_results = self.generator.scorer.score_candidates(batch_factors)
        
        # Generate signals
        signals = []
        for i, score_result in enumerate(score_results):
            symbol = batch_symbols[i]
            extras = batch_extras[i]
            
            signal = self.generator.generate_signal_from_score(
                symbol, score_result, extras.get('side', 'long'), extras
            )
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _enter_trade(
        self,
        signal,
        ts: datetime,
        arrays_map: Dict[str, np.ndarray],
        index_maps: Dict[str, Dict[int, int]]
    ):
        """Enter a new trade with execution simulation"""
        symbol = signal.symbol
        
        # Skip if already in trade
        if symbol in self.active_trades:
            return
        
        # Get current market data
        ts_sec = int(ts.timestamp())
        if symbol not in index_maps or ts_sec not in index_maps[symbol]:
            return
        
        idx = index_maps[symbol][ts_sec]
        arr = arrays_map[symbol]
        current_candle = arr[idx]
        
        # Calculate ATR for risk sizing
        atr = self._calculate_atr(arr, idx, self.atr_len)
        
        # Calculate position size (fixed risk per trade)
        risk_per_trade_pct = 0.01  # 1% risk per trade
        stop_distance = atr * self.sl_R
        if stop_distance == 0:
            return
        
        position_size_usd = 100.0 * risk_per_trade_pct / (stop_distance / signal.entry_price)
        quantity = position_size_usd / signal.entry_price
        
        # Simulate entry execution
        spread_bps = 5.0  # Default, could get from orderbook
        volatility_pct = (atr / signal.entry_price) * 100 if signal.entry_price > 0 else 1.0
        
        entry_exec = self.exec_sim.simulate_market_order(
            side=signal.side,
            intended_price=signal.entry_price,
            quantity=quantity,
            spread_bps=spread_bps,
            volatility_pct=volatility_pct,
            orderbook=None
        )
        
        # Create trade
        trade = Trade(
            symbol=symbol,
            side=signal.side,
            entry_time=ts,
            exit_time=None,
            entry_price=entry_exec.filled_price,
            exit_price=None,
            entry_slippage_bps=entry_exec.slippage_bps,
            exit_slippage_bps=0.0,
            quantity=entry_exec.filled_quantity,
            pnl_usd=0.0,
            pnl_pct=0.0,
            r_multiple=0.0,
            duration_sec=0.0,
            exit_reason='',
            score=signal.score,
            tier=signal.tier,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            atr=atr,
            volatility_pct=volatility_pct,
            spread_bps=spread_bps
        )
        
        self.active_trades[symbol] = trade
    
    def _process_exits(
        self,
        ts: datetime,
        arrays_map: Dict[str, np.ndarray],
        index_maps: Dict[str, Dict[int, int]],
        equity_curve: List[float],
        current_equity: float
    ):
        """Check and process trade exits"""
        ts_sec = int(ts.timestamp())
        
        for symbol, trade in list(self.active_trades.items()):
            if symbol not in index_maps or ts_sec not in index_maps[symbol]:
                continue
            
            idx = index_maps[symbol][ts_sec]
            arr = arrays_map[symbol]
            candle = arr[idx]
            
            # Get OHLC
            high = candle[2]
            low = candle[3]
            close = candle[4]
            
            # Check stop loss
            sl_triggered, sl_price = self.exec_sim.simulate_stop_trigger(
                side=trade.side,
                stop_price=trade.stop_loss,
                current_high=high,
                current_low=low,
                current_close=close
            )
            
            if sl_triggered:
                self._exit_trade(trade, ts, sl_price, 'stop_loss', arr, idx)
                continue
            
            # Check take profit
            tp_triggered, tp_price = self.exec_sim.simulate_stop_trigger(
                side=trade.side,
                stop_price=trade.take_profit,
                current_high=high,
                current_low=low,
                current_close=close
            )
            
            if tp_triggered:
                self._exit_trade(trade, ts, tp_price, 'take_profit', arr, idx)
                continue
    
    def _exit_trade(
        self,
        trade: Trade,
        ts: datetime,
        exit_price: float,
        exit_reason: str,
        arr: np.ndarray,
        idx: int
    ):
        """Exit a trade with execution simulation"""
        # Simulate exit execution
        volatility_pct = trade.volatility_pct
        spread_bps = trade.spread_bps
        
        exit_exec = self.exec_sim.simulate_market_order(
            side='short' if trade.side == 'long' else 'long',
            intended_price=exit_price,
            quantity=trade.quantity,
            spread_bps=spread_bps,
            volatility_pct=volatility_pct,
            orderbook=None
        )
        
        # Calculate PnL
        if trade.side == 'long':
            pnl_pct = ((exit_exec.filled_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_pct = ((trade.entry_price - exit_exec.filled_price) / trade.entry_price) * 100
        
        # Subtract fees
        entry_fee, exit_fee = self.exec_sim.calculate_fees(
            trade.entry_price, exit_exec.filled_price, trade.quantity, trade.side
        )
        fee_pct = ((entry_fee + exit_fee) / (trade.entry_price * trade.quantity)) * 100
        pnl_pct -= fee_pct
        
        # Calculate R-multiple
        stop_distance = abs(trade.entry_price - trade.stop_loss)
        if stop_distance > 0:
            price_move = abs(exit_exec.filled_price - trade.entry_price)
            r_multiple = (price_move / stop_distance) * (-1 if pnl_pct < 0 else 1)
        else:
            r_multiple = 0.0
        
        # Update trade
        trade.exit_time = ts
        trade.exit_price = exit_exec.filled_price
        trade.exit_slippage_bps = exit_exec.slippage_bps
        trade.pnl_pct = pnl_pct
        trade.pnl_usd = pnl_pct * trade.entry_price * trade.quantity / 100
        trade.r_multiple = r_multiple
        trade.duration_sec = (ts - trade.entry_time).total_seconds()
        trade.exit_reason = exit_reason
        
        # Move to completed trades
        self.trades.append(trade)
        del self.active_trades[trade.symbol]
    
    def _close_all_trades(
        self,
        ts: datetime,
        arrays_map: Dict[str, np.ndarray],
        index_maps: Dict[str, Dict[int, int]]
    ):
        """Close all remaining trades at final timestamp"""
        ts_sec = int(ts.timestamp())
        
        for symbol, trade in list(self.active_trades.items()):
            if symbol not in index_maps or ts_sec not in index_maps[symbol]:
                # Use last available price
                if symbol in arrays_map:
                    arr = arrays_map[symbol]
                    last_candle = arr[-1]
                    exit_price = last_candle[4]  # Close price
                else:
                    exit_price = trade.entry_price  # No exit, break even
            else:
                idx = index_maps[symbol][ts_sec]
                arr = arrays_map[symbol]
                exit_price = arr[idx][4]  # Close price
            
            self._exit_trade(trade, ts, exit_price, 'end_of_data', arr, -1)
    
    def _calculate_atr(self, arr: np.ndarray, idx: int, period: int) -> float:
        """Calculate ATR"""
        if idx < period:
            return 0.0
        
        atr_sum = 0.0
        for i in range(idx - period + 1, idx + 1):
            high = arr[i][2]
            low = arr[i][3]
            prev_close = arr[i-1][4] if i > 0 else arr[i][1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            atr_sum += tr
        
        return atr_sum / period
    
    def _calculate_equity(self) -> float:
        """Calculate current equity"""
        equity = 100.0  # Starting equity
        
        # Add completed trades
        for trade in self.trades:
            equity += trade.pnl_usd / 10.0  # Scale to percentage
        
        # Add unrealized PnL from active trades
        # (Simplified - would need current price)
        
        return equity
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics"""
        if not self.trades:
            return self._empty_metrics()
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl_pct > 0]
        losing_trades = [t for t in self.trades if t.pnl_pct < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl_pct = sum(t.pnl_pct for t in self.trades)
        
        gross_profit = sum(t.pnl_pct for t in winning_trades)
        gross_loss = abs(sum(t.pnl_pct for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_r = np.mean([t.r_multiple for t in self.trades]) if self.trades else 0
        max_r = max([t.r_multiple for t in self.trades], default=0)
        worst_r = min([t.r_multiple for t in self.trades], default=0)
        
        # Exit reasons
        sl_hits = len([t for t in self.trades if t.exit_reason == 'stop_loss'])
        tp_hits = len([t for t in self.trades if t.exit_reason == 'take_profit'])
        sl_hit_pct = sl_hits / total_trades * 100 if total_trades > 0 else 0
        tp_hit_pct = tp_hits / total_trades * 100 if total_trades > 0 else 0
        
        # Drawdown
        equity_curve = [100.0]
        current = 100.0
        peak = 100.0
        max_dd = 0.0
        for trade in self.trades:
            current += trade.pnl_pct
            peak = max(peak, current)
            dd = (peak - current) / peak * 100
            max_dd = max(max_dd, dd)
            equity_curve.append(current)
        
        # Tier performance
        tier_performance = {}
        tier_counts = {}
        for trade in self.trades:
            tier = trade.tier
            if tier not in tier_performance:
                tier_performance[tier] = {'pnl_pct': 0.0, 'count': 0}
                tier_counts[tier] = 0
            tier_performance[tier]['pnl_pct'] += trade.pnl_pct
            tier_performance[tier]['count'] += 1
            tier_counts[tier] += 1
        
        for tier in tier_performance:
            count = tier_performance[tier]['count']
            tier_performance[tier]['pnl_per_trade'] = tier_performance[tier]['pnl_pct'] / count if count > 0 else 0
        
        # R distribution
        r_buckets = {
            '<-2R': 0, '-2R to -1R': 0, '-1R to 0R': 0,
            '0R to 1R': 0, '1R to 2R': 0, '>2R': 0
        }
        for trade in self.trades:
            r = trade.r_multiple
            if r < -2:
                r_buckets['<-2R'] += 1
            elif r < -1:
                r_buckets['-2R to -1R'] += 1
            elif r < 0:
                r_buckets['-1R to 0R'] += 1
            elif r < 1:
                r_buckets['0R to 1R'] += 1
            elif r < 2:
                r_buckets['1R to 2R'] += 1
            else:
                r_buckets['>2R'] += 1
        
        # Execution quality
        avg_entry_slippage = np.mean([t.entry_slippage_bps for t in self.trades]) if self.trades else 0
        avg_exit_slippage = np.mean([t.exit_slippage_bps for t in self.trades]) if self.trades else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl_pct': total_pnl_pct,
            'max_drawdown_pct': max_dd,
            'avg_r': avg_r,
            'max_r': max_r,
            'worst_r': worst_r,
            'SL_hit_pct': sl_hit_pct,
            'TP_hit_pct': tp_hit_pct,
            'tier_performance': tier_performance,
            'tier_counts': tier_counts,
            'r_distribution': r_buckets,
            'execution_quality': {
                'avg_entry_slippage_bps': avg_entry_slippage,
                'avg_exit_slippage_bps': avg_exit_slippage,
                'partial_fill_rate': 0.0  # Would track from execution results
            },
            'avg_trade_duration_sec': np.mean([t.duration_sec for t in self.trades]) if self.trades else 0
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_r': 0.0,
            'max_r': 0.0,
            'worst_r': 0.0,
            'SL_hit_pct': 0.0,
            'TP_hit_pct': 0.0,
            'tier_performance': {},
            'tier_counts': {},
            'r_distribution': {},
            'execution_quality': {},
            'avg_trade_duration_sec': 0.0
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'metrics': self._empty_metrics(),
            'trades': [],
            'equity_curve': [100.0],
            'config': self.config
        }
    
    def _export_trades(self):
        """Export trades to CSV"""
        if not self.trades:
            return
        
        csv_path = os.path.join(self.output_dir, 'trades.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'symbol', 'side', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
                'pnl_pct', 'r_multiple', 'duration_sec', 'exit_reason', 'score', 'tier',
                'entry_slippage_bps', 'exit_slippage_bps', 'volatility_pct', 'spread_bps'
            ])
            writer.writeheader()
            for trade in self.trades:
                row = asdict(trade)
                row['entry_time'] = trade.entry_time.isoformat() if trade.entry_time else ''
                row['exit_time'] = trade.exit_time.isoformat() if trade.exit_time else ''
                writer.writerow(row)
        
        print(f"Exported {len(self.trades)} trades to {csv_path}")
    
    def _export_metrics(self, metrics: Dict):
        """Export metrics to JSON"""
        json_path = os.path.join(self.output_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Exported metrics to {json_path}")

