"""
ML Configuration
================

Unified ML configuration for all ML systems:
- ML model paths and settings
- ML training parameters
- ML optimization settings
- Feature extraction configs
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


def env(name: str, default: Any = None, cast: Optional[type] = None) -> Any:
    """Get environment variable with optional type casting."""
    value = os.getenv(name, default)
    if cast and value is not None:
        try:
            return cast(value)
        except (ValueError, TypeError):
            return default
    return value


# ===========================
# ML MODEL PATHS
# ===========================

# Primary model directory (XGBoost signal scorer)
ML_MODEL_DIR_PRIMARY = env("ML_MODEL_DIR", "G:/ML data/models")
ML_MODEL_DIR_FALLBACK = env("ML_MODEL_DIR_FALLBACK", str(Path(__file__).parent.parent.parent / "data" / "ml_models"))

# ML Training System output directory
ML_TRAINING_OUTPUT_DIR = env("ML_TRAINING_OUTPUT_DIR", "ml_training_output")

# Evolution ML components directory
ML_EVOLUTION_DIR = env("ML_EVOLUTION_DIR", "TradingBot/persona_optimization")


# ===========================
# ML SIGNAL SCORER SETTINGS
# ===========================

# Model file patterns
ML_MODEL_FILE_PATTERN = env("ML_MODEL_FILE_PATTERN", "best_model_XGBoost_*.joblib")
ML_SCALER_FILE_PATTERN = env("ML_SCALER_FILE_PATTERN", "scaler_*.joblib")
ML_FEATURES_FILE_PATTERN = env("ML_FEATURES_FILE_PATTERN", "features_*.json")

# Prediction cache settings
ML_PREDICTION_CACHE_TTL = env("ML_PREDICTION_CACHE_TTL", "5.0", float)  # 5 seconds

# Score interpretation thresholds
ML_SCORE_LOW = env("ML_SCORE_LOW", "30", float)  # Below this = low probability
ML_SCORE_AVERAGE = env("ML_SCORE_AVERAGE", "45", float)  # Average probability
ML_SCORE_GOOD = env("ML_SCORE_GOOD", "55", float)  # Good trades
ML_SCORE_EXCELLENT = env("ML_SCORE_EXCELLENT", "70", float)  # Excellent trades


# ===========================
# ML TRAINING SYSTEM SETTINGS
# ===========================

# Training configuration file
ML_TRAINING_CONFIG_FILE = env("ML_TRAINING_CONFIG_FILE", "ML_TRAINING_CONFIG.json")

# Default training parameters
ML_TRAINING_MAX_TRADES = env("ML_TRAINING_MAX_TRADES", "10000", int)
ML_TRAINING_TEST_SIZE = env("ML_TRAINING_TEST_SIZE", "0.15", float)
ML_TRAINING_VAL_SIZE = env("ML_TRAINING_VAL_SIZE", "0.15", float)

# NSGA-II optimization parameters
ML_NSGA_POPULATION_SIZE = env("ML_NSGA_POPULATION_SIZE", "100", int)
ML_NSGA_GENERATIONS = env("ML_NSGA_GENERATIONS", "50", int)
ML_NSGA_MUTATION_RATE = env("ML_NSGA_MUTATION_RATE", "0.15", float)
ML_NSGA_CROSSOVER_RATE = env("ML_NSGA_CROSSOVER_RATE", "0.8", float)

# Simulation parameters
ML_SIMULATION_MAX_TRADES = env("ML_SIMULATION_MAX_TRADES", "5000", int)
ML_TOP_CONFIGS_TO_VALIDATE = env("ML_TOP_CONFIGS_TO_VALIDATE", "10", int)


# ===========================
# ML EVOLUTION COMPONENTS SETTINGS
# ===========================

# Experience replay settings
ML_EXPERIENCE_REPLAY_CAPACITY = env("ML_EXPERIENCE_REPLAY_CAPACITY", "10000", int)
ML_EXPERIENCE_REPLAY_FILE = env("ML_EXPERIENCE_REPLAY_FILE", "experience_replay.pkl")

# Fitness predictor settings
ML_FITNESS_PREDICTOR_FILE = env("ML_FITNESS_PREDICTOR_FILE", "fitness_predictor.pkl")
ML_FITNESS_PREDICTOR_MODEL = env("ML_FITNESS_PREDICTOR_MODEL", "xgboost")  # xgboost, random_forest

# Feature extraction settings
ML_FEATURE_EXTRACTION_CACHE_SIZE = env("ML_FEATURE_EXTRACTION_CACHE_SIZE", "10000", int)


# ===========================
# ML FEATURE EXTRACTION
# ===========================

# Parameter ranges for normalization (used by evolution ML)
ML_PARAMETER_RANGES = {
    "min_score": (60, 95),
    "min_momentum_pct": (0.5, 8.0),
    "sl_R": (0.5, 2.0),
    "tp_R": (0.8, 4.0),
    "atr_len": (10, 21),
}


# ===========================
# ML SYSTEM STATUS
# ===========================

def get_ml_config_summary() -> Dict[str, Any]:
    """Get summary of ML configuration."""
    return {
        'model_dirs': {
            'primary': ML_MODEL_DIR_PRIMARY,
            'fallback': ML_MODEL_DIR_FALLBACK,
        },
        'training': {
            'config_file': ML_TRAINING_CONFIG_FILE,
            'output_dir': ML_TRAINING_OUTPUT_DIR,
            'max_trades': ML_TRAINING_MAX_TRADES,
        },
        'optimization': {
            'population_size': ML_NSGA_POPULATION_SIZE,
            'generations': ML_NSGA_GENERATIONS,
            'mutation_rate': ML_NSGA_MUTATION_RATE,
            'crossover_rate': ML_NSGA_CROSSOVER_RATE,
        },
        'evolution': {
            'experience_replay_capacity': ML_EXPERIENCE_REPLAY_CAPACITY,
            'fitness_predictor_model': ML_FITNESS_PREDICTOR_MODEL,
        },
    }

