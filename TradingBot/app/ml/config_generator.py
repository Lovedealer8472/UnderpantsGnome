"""
Config Generator - Generate optimal config files from training results.

Creates deployable configuration files with:
- Optimal parameters from training
- Expected performance metrics
- Deployment instructions
- Version tracking
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class ConfigGenerator:
    """Generates configuration files from training results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize config generator.
        
        Args:
            output_dir: Directory to save generated configs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_optimal_config(
        self,
        parameters: Dict,
        expected_performance: Dict,
        training_stats: Dict,
        validation_results: Optional[list] = None
    ) -> Path:
        """
        Generate optimal configuration file.
        
        Args:
            parameters: Optimal parameter values
            expected_performance: Expected performance metrics
            training_stats: Training statistics
            validation_results: Optional validation results
            
        Returns:
            Path to generated config file
        """
        config = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "training_stats": training_stats,
            "parameters": parameters,
            "expected_performance": expected_performance,
            "validation_results": validation_results or [],
            "deployment": {
                "recommended_steps": [
                    "1. Deploy to paper trading (DRY_RUN=1)",
                    "2. Monitor for 7-14 days",
                    "3. Compare actual vs. expected performance",
                    "4. If validated, deploy to live trading",
                    "5. Enable periodic retraining"
                ],
                "monitoring_thresholds": {
                    "min_win_rate": expected_performance.get('win_rate', 50) * 0.9,
                    "min_sharpe_ratio": expected_performance.get('sharpe_ratio', 1.0) * 0.8,
                    "max_drawdown": expected_performance.get('max_drawdown', 10) * 1.2
                }
            }
        }
        
        config_path = self.output_dir / "OPTIMAL_ML_CONFIG.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Generated optimal config: {config_path}")
        return config_path
    
    def generate_deployment_script(
        self,
        config_path: Path,
        bot_config_path: str = "TradingBot/app/config.py"
    ) -> Path:
        """
        Generate deployment script to apply config.
        
        Args:
            config_path: Path to optimal config
            bot_config_path: Path to bot config file
            
        Returns:
            Path to deployment script
        """
        script_content = f"""#!/usr/bin/env python3
\"\"\"
Deployment Script - Apply optimal ML config to trading bot.

Generated: {datetime.now().isoformat()}
Config: {config_path}
\"\"\"

import json
from pathlib import Path

def apply_config():
    # Load optimal config
    with open('{config_path}', 'r') as f:
        optimal_config = json.load(f)
    
    parameters = optimal_config['parameters']
    
    print("Optimal Configuration:")
    print("=" * 60)
    for key, value in parameters.items():
        print(f"  {{key}}: {{value}}")
    
    print("\\nExpected Performance:")
    print("=" * 60)
    for key, value in optimal_config['expected_performance'].items():
        print(f"  {{key}}: {{value:.4f}}")
    
    print("\\nDeployment Steps:")
    print("=" * 60)
    for step in optimal_config['deployment']['recommended_steps']:
        print(f"  {{step}}")
    
    print("\\nTo apply these parameters:")
    print("1. Manually update {bot_config_path}")
    print("2. Or use environment variables")
    print("3. Restart the bot")
    
    # Generate environment variable export commands
    print("\\nEnvironment Variables:")
    print("=" * 60)
    for key, value in parameters.items():
        if isinstance(value, bool):
            value = "1" if value else "0"
        print(f"export {{key}}={{value}}")

if __name__ == "__main__":
    apply_config()
"""
        
        script_path = self.output_dir / "deploy_optimal_config.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        print(f"Generated deployment script: {script_path}")
        return script_path
    
    def generate_comparison_report(
        self,
        current_config: Dict,
        optimal_config: Dict,
        current_performance: Dict,
        expected_performance: Dict
    ) -> Path:
        """
        Generate comparison report between current and optimal configs.
        
        Args:
            current_config: Current configuration
            optimal_config: Optimal configuration
            current_performance: Current performance metrics
            expected_performance: Expected performance with optimal config
            
        Returns:
            Path to comparison report
        """
        report_lines = [
            "# Configuration Comparison Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Comparison",
            "",
            "| Metric | Current | Optimal | Improvement |",
            "|--------|---------|---------|-------------|"
        ]
        
        for metric in ['roi_per_day', 'sharpe_ratio', 'win_rate', 'profit_factor']:
            current_val = current_performance.get(metric, 0)
            optimal_val = expected_performance.get(metric, 0)
            improvement = ((optimal_val - current_val) / current_val * 100) if current_val > 0 else 0
            
            report_lines.append(
                f"| {metric} | {current_val:.4f} | {optimal_val:.4f} | {improvement:+.1f}% |"
            )
        
        report_lines.extend([
            "",
            "## Parameter Changes",
            "",
            "| Parameter | Current | Optimal | Change |",
            "|-----------|---------|---------|--------|"
        ])
        
        all_params = set(current_config.keys()) | set(optimal_config.keys())
        for param in sorted(all_params):
            current_val = current_config.get(param, "N/A")
            optimal_val = optimal_config.get(param, "N/A")
            
            if current_val != optimal_val:
                change = "✓ Changed"
            else:
                change = "- Same"
            
            report_lines.append(
                f"| {param} | {current_val} | {optimal_val} | {change} |"
            )
        
        report_lines.extend([
            "",
            "## Recommendations",
            "",
            "1. Review parameter changes carefully",
            "2. Test in paper trading for 7-14 days",
            "3. Monitor performance vs. expected metrics",
            "4. Gradually roll out if validated",
            ""
        ])
        
        report_path = self.output_dir / "CONFIG_COMPARISON.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Generated comparison report: {report_path}")
        return report_path
    
    def generate_monitoring_dashboard_config(
        self,
        expected_performance: Dict
    ) -> Path:
        """
        Generate monitoring dashboard configuration.
        
        Args:
            expected_performance: Expected performance metrics
            
        Returns:
            Path to dashboard config
        """
        dashboard_config = {
            "monitoring": {
                "metrics": [
                    {
                        "name": "roi_per_day",
                        "expected": expected_performance.get('roi_per_day', 0),
                        "alert_threshold_low": expected_performance.get('roi_per_day', 0) * 0.8,
                        "alert_threshold_high": expected_performance.get('roi_per_day', 0) * 1.2
                    },
                    {
                        "name": "sharpe_ratio",
                        "expected": expected_performance.get('sharpe_ratio', 0),
                        "alert_threshold_low": expected_performance.get('sharpe_ratio', 0) * 0.7,
                        "alert_threshold_high": expected_performance.get('sharpe_ratio', 0) * 1.3
                    },
                    {
                        "name": "win_rate",
                        "expected": expected_performance.get('win_rate', 0),
                        "alert_threshold_low": expected_performance.get('win_rate', 0) * 0.85,
                        "alert_threshold_high": expected_performance.get('win_rate', 0) * 1.15
                    },
                    {
                        "name": "max_drawdown",
                        "expected": expected_performance.get('max_drawdown', 0),
                        "alert_threshold_high": expected_performance.get('max_drawdown', 0) * 1.5
                    }
                ],
                "alert_channels": [
                    "console",
                    "log_file"
                ],
                "check_interval_seconds": 3600
            }
        }
        
        dashboard_path = self.output_dir / "monitoring_dashboard_config.json"
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        print(f"Generated monitoring config: {dashboard_path}")
        return dashboard_path

