"""
ML Confidence Entry Configuration

Configuration for the new ML Confidence-Based entry filter system.
"""

import os
from typing import Dict

# Enable/disable ML confidence filtering
USE_ML_CONFIDENCE_FILTER = os.getenv("USE_ML_CONFIDENCE_FILTER", "1") in ("1", "true", "TRUE")

# Base confidence thresholds (0.0 - 1.0) - ML-OPTIMIZED for 93% AUC models
# These are adaptive: they increase as positions fill up
# Calibrated based on actual model probability distributions (most signals: 40-60%)
ML_CONFIDENCE_IDLE = float(os.getenv("ML_CONFIDENCE_IDLE", "0.50"))        # When 0 positions (50% = breakeven)
ML_CONFIDENCE_NORMAL = float(os.getenv("ML_CONFIDENCE_NORMAL", "0.55"))    # When 1 position (55% = slight edge)
ML_CONFIDENCE_CROWDED = float(os.getenv("ML_CONFIDENCE_CROWDED", "0.60"))  # When 2 positions (60% = profitable)
ML_CONFIDENCE_FULL = float(os.getenv("ML_CONFIDENCE_FULL", "0.70"))        # When 3+ positions (70% = high confidence)

# Model agreement threshold: models must agree within this variance
# 0.10 = models can differ by up to 10% and still be considered "agreeing"
ML_CONFIDENCE_AGREEMENT_THRESHOLD = float(os.getenv("ML_CONFIDENCE_AGREEMENT_THRESHOLD", "0.10"))

# Confidence thresholds by position count (for easy override)
ML_CONFIDENCE_THRESHOLDS: Dict[int, float] = {
    0: ML_CONFIDENCE_IDLE,      # 0 positions
    1: ML_CONFIDENCE_NORMAL,     # 1 position
    2: ML_CONFIDENCE_CROWDED,    # 2 positions
    3: ML_CONFIDENCE_FULL,       # 3+ positions
}

# Fallback behavior when models don't have probabilities
# If True: allow entry with score-based filtering
# If False: reject entry (safer, requires ML probabilities)
# NOTE: Set to True because current ML models have feature mismatch (73 vs 20 features)
ALLOW_ENTRY_WITHOUT_ML_CONFIDENCE = os.getenv("ALLOW_ENTRY_WITHOUT_ML_CONFIDENCE", "1") in ("1", "true")

print(f"""
[ML_CONFIDENCE_FILTER] Configuration:
  Enabled: {USE_ML_CONFIDENCE_FILTER}
  Confidence thresholds:
    - Idle (0 pos): {ML_CONFIDENCE_IDLE:.0%}
    - Normal (1 pos): {ML_CONFIDENCE_NORMAL:.0%}
    - Crowded (2 pos): {ML_CONFIDENCE_CROWDED:.0%}
    - Full (3+ pos): {ML_CONFIDENCE_FULL:.0%}
  Agreement threshold: {ML_CONFIDENCE_AGREEMENT_THRESHOLD:.1%} variance
  Fallback (no ML prob): {ALLOW_ENTRY_WITHOUT_ML_CONFIDENCE}
""")

