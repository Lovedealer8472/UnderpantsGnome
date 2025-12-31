#!/usr/bin/env python3
"""
Hybrid ML Scorer - XGBoost + Random Forest + LightGBM Ensemble
==============================================================

Replaces single XGBoost with proper ensemble for live predictions.
Uses weighted voting across 3 models for more robust signal scoring.

Performance benefits:
- More consistent predictions (reduced overfitting)
- Better handling of different market regimes
- Lower prediction variance across symbols
"""

import json
import os
import time
import math
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

# ML libraries
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class HybridMLScorer:
    """
    Hybrid ML Scorer using ensemble of XGBoost, Random Forest, and LightGBM.
    
    Weights:
    - XGBoost: 0.4 (best AUC)
    - LightGBM: 0.4 (fast, robust)
    - Random Forest: 0.2 (interpretable, conservative)
    """
    
    MODEL_PATHS = [
        # Latest training output with LIVE-AVAILABLE features (highest priority)
        Path(__file__).parent.parent.parent / "ml_training_output" / "20251229_200117" / "models",
        # Legacy paths (will fail feature match but kept for fallback)
        Path(__file__).parent.parent / "data" / "ml_models",
    ]
    
    # LIVE-AVAILABLE FEATURES ONLY (12 features matching trained models)
    # These are features we can compute at signal time without post-trade data
    REQUIRED_FEATURES = [
        "rsi",              # RSI at signal time
        "adx",              # ADX at signal time  
        "side_long",        # 1 if long, 0 if short
        "hour_sin",         # Cyclical hour encoding
        "hour_cos",
        "dow_sin",          # Cyclical day-of-week encoding
        "dow_cos",
        "rsi_oversold",     # RSI < 30
        "rsi_overbought",   # RSI > 70
        "rsi_neutral",      # 30 <= RSI <= 70
        "adx_strong",       # ADX > 25 (trending)
        "adx_weak",         # ADX < 20 (ranging)
    ]
    
    # Ensemble weights
    ENSEMBLE_WEIGHTS = {
        'xgboost': 0.4,
        'lightgbm': 0.4,
        'random_forest': 0.2,
    }
    
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.models = {
            'xgboost': None,
            'lightgbm': None,
            'random_forest': None,
        }
        self.scaler = None
        self.feature_names = None
        self._loaded = False
        self._load_error = None
        
        # Cache
        self._prediction_cache = {}
        self._cache_ttl = 5.0
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.model_availability = {
            'xgboost': False,
            'lightgbm': False,
            'random_forest': False,
        }
    
    def _find_model_files(self) -> Dict[str, Optional[Path]]:
        """Find ensemble model files."""
        found = {
            'xgboost': None,
            'lightgbm': None,
            'random_forest': None,
            'scaler': None,
        }
        
        search_dirs = [self.model_dir] if self.model_dir else self.MODEL_PATHS
        
        for search_dir in search_dirs:
            if not Path(search_dir).exists():
                continue
            
            # XGBoost - check both .pkl and .joblib
            if found['xgboost'] is None:
                for ext in ['*.pkl', '*.joblib']:
                    for f in Path(search_dir).glob(f"*xgboost*{ext[1:]}"):
                        found['xgboost'] = f
                        break
                    if found['xgboost']:
                        break
            
            # LightGBM - check both .pkl and .joblib
            if found['lightgbm'] is None:
                for ext in ['*.pkl', '*.joblib']:
                    for f in Path(search_dir).glob(f"*lightgbm*{ext[1:]}"):
                        found['lightgbm'] = f
                        break
                    if found['lightgbm']:
                        break
            
            # Random Forest - check both .pkl and .joblib
            if found['random_forest'] is None:
                for ext in ['*.pkl', '*.joblib']:
                    for f in Path(search_dir).glob(f"*random*forest*{ext[1:]}"):
                        found['random_forest'] = f
                        break
                    if found['random_forest']:
                        break
            
            # Scaler - check both .pkl and .joblib
            if found['scaler'] is None:
                for ext in ['*.pkl', '*.joblib']:
                    for f in Path(search_dir).glob(f"*scaler*{ext[1:]}"):
                        found['scaler'] = f
                        break
                    if found['scaler']:
                        break
        
        return found
    
    def _load_model(self):
        """Load all ensemble models."""
        if self._loaded:
            return
        
        self._loaded = True
        files = self._find_model_files()
        
        # Load scaler
        if files['scaler'] and JOBLIB_AVAILABLE:
            try:
                self.scaler = joblib.load(files['scaler'])
            except Exception as e:
                self._load_error = f"Scaler load error: {e}"
        
        # Load XGBoost
        if files['xgboost'] and XGBOOST_AVAILABLE and JOBLIB_AVAILABLE:
            try:
                self.models['xgboost'] = joblib.load(files['xgboost'])
                self.model_availability['xgboost'] = True
            except Exception:
                pass
        
        # Load LightGBM
        if files['lightgbm'] and LIGHTGBM_AVAILABLE and JOBLIB_AVAILABLE:
            try:
                self.models['lightgbm'] = joblib.load(files['lightgbm'])
                self.model_availability['lightgbm'] = True
            except Exception:
                pass
        
        # Load Random Forest
        if files['random_forest'] and SKLEARN_AVAILABLE and JOBLIB_AVAILABLE:
            try:
                self.models['random_forest'] = joblib.load(files['random_forest'])
                self.model_availability['random_forest'] = True
            except Exception:
                pass
        
        # Set feature names if not set
        if not self.feature_names:
            self.feature_names = self.REQUIRED_FEATURES
    
    def _extract_features(self, symbol: str, pct_change_24h: float, volume_24h: float,
                         spread_bps: float, orderbook: Optional[Dict] = None,
                         indicators: Optional[Dict] = None, latency_ms: float = 0.0,
                         btc_trend: Optional[float] = None, funding_rate: Optional[float] = None,
                         side: str = None) -> Dict:
        """
        Extract LIVE-AVAILABLE features for ML prediction.
        
        These 12 features match what our models were trained on:
        - rsi, adx, side_long
        - hour_sin, hour_cos, dow_sin, dow_cos  
        - rsi_oversold, rsi_overbought, rsi_neutral
        - adx_strong, adx_weak
        """
        features = {}
        
        # Get RSI and ADX from indicators
        if indicators:
            rsi = indicators.get('rsi', 50.0)
            adx = indicators.get('adx', 20.0)
        else:
            rsi = 50.0
            adx = 20.0
        
        # Core indicators
        features['rsi'] = rsi
        features['adx'] = adx
        
        # Side encoding (infer from pct_change if not provided)
        if side is not None:
            is_long = side.lower() == 'long'
        else:
            is_long = pct_change_24h > 0  # Momentum-based inference
        features['side_long'] = 1.0 if is_long else 0.0
        
        # Time features (cyclical encoding)
        now = datetime.now(timezone.utc)
        hour = now.hour / 24.0
        dow = now.weekday() / 7.0
        features['hour_sin'] = math.sin(2 * math.pi * hour)
        features['hour_cos'] = math.cos(2 * math.pi * hour)
        features['dow_sin'] = math.sin(2 * math.pi * dow)
        features['dow_cos'] = math.cos(2 * math.pi * dow)
        
        # RSI zones (binary indicators)
        features['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
        features['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
        features['rsi_neutral'] = 1.0 if 30 <= rsi <= 70 else 0.0
        
        # ADX zones (trend strength indicators)
        features['adx_strong'] = 1.0 if adx > 25 else 0.0
        features['adx_weak'] = 1.0 if adx < 20 else 0.0
        
        return features
    
    def score_signal(self, symbol: str, pct_change_24h: float, volume_24h: float,
                    spread_bps: float, orderbook: Optional[Dict] = None,
                    indicators: Optional[Dict] = None, latency_ms: float = 0.0,
                    btc_trend: Optional[float] = None, funding_rate: Optional[float] = None,
                    side: str = None
                    ) -> Tuple[float, Dict]:
        """
        Score signal using ensemble of ML models.
        
        Args:
            side: 'long' or 'short' - if provided, improves prediction accuracy
        
        Returns (score, components).
        """
        
        # Ensure models loaded
        if not self._loaded:
            self._load_model()
        
        # Cache check (include side in cache key)
        cache_key = f"{symbol}:{pct_change_24h:.4f}:{spread_bps:.2f}:{side or 'auto'}"
        now = time.time()
        if cache_key in self._prediction_cache:
            cache_time, cached_result = self._prediction_cache[cache_key]
            if (now - cache_time) < self._cache_ttl:
                return cached_result
        
        try:
            start_time = time.time()
            
            # Extract features (using new 12-feature set)
            features = self._extract_features(
                symbol, pct_change_24h, volume_24h, spread_bps,
                orderbook, indicators, latency_ms,
                btc_trend, funding_rate, side
            )
            
            # Create feature vector
            X = np.array([[features.get(f, 0.0) for f in self.REQUIRED_FEATURES]], dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale if available
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Ensemble prediction
            predictions = {}
            weights = []
            
            # XGBoost
            if self.model_availability['xgboost'] and self.models['xgboost']:
                try:
                    pred = self.models['xgboost'].predict_proba(X_scaled)[0, 1]
                    predictions['xgboost'] = float(pred)
                    weights.append(self.ENSEMBLE_WEIGHTS['xgboost'])
                except Exception as e:
                    logger.debug(f"[ML_SCORE] XGBoost prediction failed: {e}")
            
            # LightGBM
            if self.model_availability['lightgbm'] and self.models['lightgbm']:
                try:
                    pred = self.models['lightgbm'].predict_proba(X_scaled)[0, 1]
                    predictions['lightgbm'] = float(pred)
                    weights.append(self.ENSEMBLE_WEIGHTS['lightgbm'])
                except Exception as e:
                    logger.debug(f"[ML_SCORE] LightGBM prediction failed: {e}")
            
            # Random Forest
            if self.model_availability['random_forest'] and self.models['random_forest']:
                try:
                    pred = self.models['random_forest'].predict_proba(X_scaled)[0, 1]
                    predictions['random_forest'] = float(pred)
                    weights.append(self.ENSEMBLE_WEIGHTS['random_forest'])
                except Exception as e:
                    logger.debug(f"[ML_SCORE] RandomForest prediction failed: {e}")
            
            # If no models available, fallback
            if not predictions:
                return self._fallback_score(pct_change_24h, spread_bps, indicators)
            
            # Weighted ensemble
            pred_values = list(predictions.values())
            if weights:
                # Normalize weights
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                win_prob = np.average(pred_values, weights=normalized_weights)
            else:
                win_prob = np.mean(pred_values)
            
            score = float(win_prob * 100)
            score = max(0, min(100, score))
            
            # Track performance
            self.prediction_count += 1
            self.total_prediction_time += (time.time() - start_time)
            
            components = {
                'total': score,
                'ensemble_prob': float(win_prob),
                'models_used': len(predictions),
                'model_scores': predictions,
                'atr_pct': features.get('atr_pct', 0),
                'rsi': features.get('rsi', 50),
                'signal_direction': features.get('signal_direction', 1),
                'fallback': False,
                'model': f"Ensemble_{len(predictions)}"
            }
            
            result = (score, components)
            self._prediction_cache[cache_key] = (now, result)
            return result
            
        except Exception as e:
            return self._fallback_score(pct_change_24h, spread_bps, indicators)
    
    def _fallback_score(self, pct_change_24h: float, spread_bps: float,
                       indicators: Optional[Dict] = None) -> Tuple[float, Dict]:
        """Fallback scoring if models unavailable."""
        
        # Heuristic: momentum, volatility, spread
        base_score = 50.0  # Start neutral
        
        # Momentum bonus
        if pct_change_24h > 0.05:  # > 5% positive
            base_score += 15
        elif pct_change_24h > 0.02:  # > 2% positive
            base_score += 8
        elif pct_change_24h < -0.05:  # > 5% negative (good for shorts)
            base_score += 12
        elif pct_change_24h < -0.02:  # > 2% negative
            base_score += 6
        
        # Spread penalty (less harsh)
        if spread_bps > 50:
            base_score -= 8
        elif spread_bps > 30:
            base_score -= 3
        
        # Volatility bonus (from ATR if available)
        if indicators:
            atr_pct = indicators.get('atr_pct', 0.01)
            atr_pct = atr_pct * 100 if atr_pct < 1 else atr_pct
            if atr_pct >= 2.0:
                base_score += 10
            elif atr_pct >= 1.0:
                base_score += 5
            
            # RSI extremes
            rsi = indicators.get('rsi', 50)
            if rsi > 70 or rsi < 30:
                base_score += 5
        
        score = max(35, min(85, base_score))  # Was max(20, min(80, base_score))
        
        return (score, {
            'total': score,
            'fallback': True,
            'reason': 'ML models unavailable',
            'model': 'FALLBACK'
        })
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if not self._loaded:
            self._load_model()
        
        if not any(self.model_availability.values()):
            return {
                'status': 'error',
                'error': 'No models loaded',
                'models_tried': list(self.model_availability.keys())
            }
        
        avg_time = (self.total_prediction_time / self.prediction_count * 1000
                   if self.prediction_count > 0 else 0)
        
        loaded_models = [k for k, v in self.model_availability.items() if v]
        
        return {
            'status': 'loaded',
            'model': f"Ensemble_{len(loaded_models)}",
            'models': loaded_models,
            'weights': {k: v for k, v in self.ENSEMBLE_WEIGHTS.items() if k in loaded_models},
            'predictions': self.prediction_count,
            'avg_prediction_ms': avg_time
        }


# Singleton
_default_scorer = None

def get_hybrid_scorer():
    """Get or create default scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = HybridMLScorer()
    return _default_scorer

