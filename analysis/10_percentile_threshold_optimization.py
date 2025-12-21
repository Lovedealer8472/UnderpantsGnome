"""
Percentile Threshold Optimization
=================================
Test different SIGNAL_PERCENTILE_THRESHOLD values to find optimal filtering.
Simulates percentile filtering on historical trades to find best threshold.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    stream_all_trades, print_section, save_report
)
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np

def simulate_percentile_filtering(trades, percentile_threshold):
    """
    Simulate percentile filtering on a rolling window of signals.
    
    Args:
        trades: List of trades with scores
        percentile_threshold: Float 0.0-1.0 (e.g., 0.90 = top 10%)
    
    Returns:
        Filtered trades that would have passed the percentile filter
    """
    if percentile_threshold <= 0.0:
        return trades  # No filtering
    
    # Sort trades by timestamp (assume they're in order)
    sorted_trades = sorted(trades, key=lambda t: t.get('timestamp', 0))
    
    # Rolling window for percentile calculation
    window_size = 100  # Match SIGNAL_HISTORY_SIZE
    filtered_trades = []
    
    for i, trade in enumerate(sorted_trades):
        # Need at least window_size trades before filtering
        if i < window_size:
            # Early trades: use all (building history)
            filtered_trades.append(trade)
            continue
        
        # Get recent scores for percentile calculation
        recent_trades = sorted_trades[max(0, i - window_size):i]
        recent_scores = [t.get('score', 0) for t in recent_trades if t.get('score', 0) > 0]
        
        if len(recent_scores) < 20:
            # Not enough history - allow trade
            filtered_trades.append(trade)
            continue
        
        # Calculate percentile threshold
        recent_scores_sorted = sorted(recent_scores, reverse=True)
        top_percent = 1.0 - percentile_threshold  # Convert to "top X%"
        threshold_index = int(len(recent_scores_sorted) * top_percent) - 1
        threshold_index = max(0, min(threshold_index, len(recent_scores_sorted) - 1))
        threshold_score = recent_scores_sorted[threshold_index]
        
        # Check if this trade's score is above threshold
        trade_score = trade.get('score', 0)
        if trade_score >= threshold_score:
            filtered_trades.append(trade)
    
    return filtered_trades

def analyze_percentile_threshold():
    print_section("PERCENTILE THRESHOLD OPTIMIZATION")
    
    # Load from decision logs (has signal scores) or trades
    BASE_DIR = Path(r"C:\UniRabbit_Live")
    LOGS_DIR = BASE_DIR / "logs"
    
    print("Loading signal data from decision logs...")
    all_signals = []
    decision_files = sorted(LOGS_DIR.glob("decisions_*.jsonl"))
    
    if decision_files:
        for filepath in decision_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            # Include all signals (approved and rejected) to build proper percentile distribution
                            if entry.get('final_score'):
                                all_signals.append({
                                    'score': entry.get('final_score', 0),
                                    'won': entry.get('won', False),
                                    'pnl_dollars': entry.get('pnl', entry.get('pnl_dollars', 0)),
                                    'r_multiple': entry.get('r_multiple', 0),
                                    'timestamp': entry.get('timestamp', 0),
                                    'symbol': entry.get('symbol', ''),
                                    'side': entry.get('side', ''),
                                    'approved': entry.get('action') == 'approved'
                                })
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")
    
    # Fallback to trades if decision logs don't have enough data
    if len(all_signals) < 1000:
        print(f"  Only {len(all_signals)} signals from decision logs, trying trades...")
        all_trades = list(stream_all_trades())
        if all_trades:
            all_signals = [{
                'score': t.get('score', 0),
                'won': t.get('won', False),
                'pnl_dollars': t.get('pnl_dollars', 0),
                'r_multiple': t.get('r_multiple', 0),
                'timestamp': t.get('timestamp', 0),
                'symbol': t.get('symbol', ''),
                'side': t.get('side', ''),
                'approved': True  # Trades are all approved
            } for t in all_trades if t.get('score', 0) > 0]
    
    total_signals = len(all_signals)
    print(f"Loaded {total_signals:,} signals\n")
    
    if total_signals < 1000:
        print("⚠️  WARNING: Need at least 1000 signals for reliable analysis")
        print(f"   Found: {total_signals:,} signals")
        print(f"   Decision files: {len(decision_files)}")
        return
    
    # For analysis, only use approved signals (actual trades taken)
    all_trades = [s for s in all_signals if s.get('approved', True)]
    print(f"  Approved trades: {len(all_trades):,}\n")
    
    # Test different percentile thresholds
    thresholds_to_test = [0.0, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    threshold_labels = {
        0.0: "No Filter",
        0.75: "Top 25%",
        0.80: "Top 20%",
        0.85: "Top 15%",
        0.90: "Top 10%",
        0.95: "Top 5%",
        0.99: "Top 1%"
    }
    
    results = {}
    
    print("Testing percentile thresholds...")
    print("=" * 100)
    print(f"{'Threshold':<12} {'Label':<12} {'Trades':>10} {'% of Total':>12} {'Win Rate':>10} {'Avg R':>10} {'Total PnL':>12} {'PF':>8} {'Sharpe':>8}")
    print("=" * 100)
    
    for threshold in thresholds_to_test:
        # Simulate filtering
        filtered_trades = simulate_percentile_filtering(all_trades, threshold)
        
        if len(filtered_trades) == 0:
            continue
        
        # Calculate metrics
        wins = sum(1 for t in filtered_trades if t.get('won', False))
        losses = len(filtered_trades) - wins
        win_rate = (wins / len(filtered_trades) * 100) if filtered_trades else 0
        
        pnl_list = [t.get('pnl_dollars', 0) for t in filtered_trades]
        r_multiples = [t.get('r_multiple', 0) for t in filtered_trades]
        
        avg_pnl = statistics.mean(pnl_list) if pnl_list else 0
        total_pnl = sum(pnl_list)
        avg_r = statistics.mean(r_multiples) if r_multiples else 0
        
        # Profit factor
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified - using R multiples)
        if len(r_multiples) > 1:
            sharpe = statistics.mean(r_multiples) / statistics.stdev(r_multiples) if statistics.stdev(r_multiples) > 0 else 0
        else:
            sharpe = 0
        
        pct_of_total = (len(filtered_trades) / total_trades * 100) if total_trades > 0 else 0
        
        results[threshold] = {
            'trades': len(filtered_trades),
            'pct_of_total': pct_of_total,
            'win_rate': win_rate,
            'avg_r': avg_r,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'avg_pnl': avg_pnl
        }
        
        label = threshold_labels.get(threshold, f"Top {int((1-threshold)*100)}%")
        print(f"{threshold:<12.2f} {label:<12} {len(filtered_trades):>10,} {pct_of_total:>11.1f}% {win_rate:>9.1f}% {avg_r:>9.2f} ${total_pnl:>11,.2f} {profit_factor:>7.2f} {sharpe:>7.2f}")
    
    print("=" * 100)
    
    # Find optimal threshold
    print_section("OPTIMAL THRESHOLD ANALYSIS")
    
    # Score based on multiple factors (weighted)
    best_threshold = None
    best_score = -float('inf')
    
    print("\nScoring thresholds (weighted: Win Rate 40%, Profit Factor 30%, Sharpe 20%, Avg R 10%):")
    print("-" * 80)
    
    for threshold, data in results.items():
        # Normalized scores (0-1)
        max_wr = max(r['win_rate'] for r in results.values())
        max_pf = max(r['profit_factor'] for r in results.values())
        max_sharpe = max(abs(r['sharpe']) for r in results.values()) or 1
        max_avg_r = max(r['avg_r'] for r in results.values()) or 1
        
        wr_score = (data['win_rate'] / max_wr) if max_wr > 0 else 0
        pf_score = (data['profit_factor'] / max_pf) if max_pf > 0 else 0
        sharpe_score = (abs(data['sharpe']) / max_sharpe) if max_sharpe > 0 else 0
        r_score = (data['avg_r'] / max_avg_r) if max_avg_r > 0 else 0
        
        # Weighted composite score
        composite_score = (
            wr_score * 0.40 +
            pf_score * 0.30 +
            sharpe_score * 0.20 +
            r_score * 0.10
        )
        
        if composite_score > best_score:
            best_score = composite_score
            best_threshold = threshold
        
        label = threshold_labels.get(threshold, f"Top {int((1-threshold)*100)}%")
        print(f"{label:<12} | Score: {composite_score:.3f} | WR: {data['win_rate']:.1f}% | PF: {data['profit_factor']:.2f} | Sharpe: {data['sharpe']:.2f}")
    
    print("-" * 80)
    best_label = threshold_labels.get(best_threshold, f"Top {int((1-best_threshold)*100)}%")
    best_data = results[best_threshold]
    print(f"\n🏆 OPTIMAL THRESHOLD: {best_threshold:.2f} ({best_label})")
    print(f"   Trades: {best_data['trades']:,} ({best_data['pct_of_total']:.1f}% of total)")
    print(f"   Win Rate: {best_data['win_rate']:.1f}%")
    print(f"   Profit Factor: {best_data['profit_factor']:.2f}")
    print(f"   Avg R: {best_data['avg_r']:.2f}")
    print(f"   Total PnL: ${best_data['total_pnl']:,.2f}")
    print(f"   Sharpe: {best_data['sharpe']:.2f}")
    
    # Generate report
    report = ["# Percentile Threshold Optimization\n\n"]
    report.append(f"**Total Trades Analyzed:** {total_trades:,}\n\n")
    report.append("## Results by Threshold\n\n")
    report.append("| Threshold | Label | Trades | % of Total | Win Rate | Avg R | Total PnL | Profit Factor | Sharpe |\n")
    report.append("|-----------|-------|--------|------------|----------|-------|-----------|---------------|--------|\n")
    
    for threshold in sorted(results.keys()):
        data = results[threshold]
        label = threshold_labels.get(threshold, f"Top {int((1-threshold)*100)}%")
        report.append(
            f"| {threshold:.2f} | {label} | {data['trades']:,} | {data['pct_of_total']:.1f}% | "
            f"{data['win_rate']:.1f}% | {data['avg_r']:.2f} | ${data['total_pnl']:,.2f} | "
            f"{data['profit_factor']:.2f} | {data['sharpe']:.2f} |\n"
        )
    
    report.append(f"\n## Optimal Threshold\n\n")
    report.append(f"**Recommended:** `{best_threshold:.2f}` ({best_label})\n\n")
    report.append(f"- Trades: {best_data['trades']:,} ({best_data['pct_of_total']:.1f}% of total)\n")
    report.append(f"- Win Rate: {best_data['win_rate']:.1f}%\n")
    report.append(f"- Profit Factor: {best_data['profit_factor']:.2f}\n")
    report.append(f"- Avg R: {best_data['avg_r']:.2f}\n")
    report.append(f"- Total PnL: ${best_data['total_pnl']:,.2f}\n")
    report.append(f"- Sharpe: {best_data['sharpe']:.2f}\n")
    
    save_report("10_percentile_threshold_optimization", "".join(report))
    
    print(f"\n✅ Report saved to: analysis/reports/10_percentile_threshold_optimization.md")

if __name__ == "__main__":
    analyze_percentile_threshold()

