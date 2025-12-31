"""
Unified ML Framework
====================

Single framework that unifies all ML systems:
- Signal scoring (MLScorer)
- Parameter optimization (ML Training System)
- Evolution learning (ML Components)

This is the single source of truth for all ML operations.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

# Import ML components
from ..ml_scorer import MLScorer

# Try to import hybrid ML scorer (ensemble)
try:
    from ..hybrid_ml_scorer import HybridMLScorer
    HAS_HYBRID_ML = True
except ImportError:
    HAS_HYBRID_ML = False
    HybridMLScorer = None

# Try to import hindsight ML scorer (Master Hindsight System)
try:
    from .hindsight_scorer import HindsightMLScorer
    HAS_HINDSIGHT_ML = True
except ImportError:
    HAS_HINDSIGHT_ML = False
    HindsightMLScorer = None

# Try to import evolution ML components (optional)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from persona_optimization.ml_components import (
        FeatureExtractor as EvolutionFeatureExtractor,
        ExperienceReplay,
        FitnessPredictor
    )
    HAS_EVOLUTION_ML = True
except ImportError:
    HAS_EVOLUTION_ML = False
    EvolutionFeatureExtractor = None
    ExperienceReplay = None
    FitnessPredictor = None

# Try to import training system ML components (optional)
try:
    from .hindsight_analyzer import HindsightAnalyzer
    from .feature_engine import FeatureEngine
    from .ensemble import EnsembleTrainer
    from .multi_objective import NSGAII
    from .ohlcv_simulator import OHLCVSimulator
    HAS_TRAINING_ML = True
except ImportError:
    HAS_TRAINING_ML = False
    HindsightAnalyzer = None
    FeatureEngine = None
    EnsembleTrainer = None
    NSGAII = None
    OHLCVSimulator = None


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages ML model versions, loading, and fallbacks."""
    
    def __init__(self, model_base_dir: Optional[Path] = None):
        self.model_base_dir = model_base_dir or Path("data/ml_models")
        self.model_base_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models = {}
        self._model_versions = {}
    
    def register_model(self, name: str, model: Any, version: str = "latest"):
        """Register a loaded model."""
        self._loaded_models[name] = model
        self._model_versions[name] = version
        logger.info(f"[ModelRegistry] Registered model '{name}' version '{version}'")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a registered model."""
        return self._loaded_models.get(name)
    
    def get_version(self, name: str) -> Optional[str]:
        """Get model version."""
        return self._model_versions.get(name)


class UnifiedFeatureExtractor:
    """Unified feature extraction for all ML systems."""
    
    def __init__(self):
        self.evolution_extractor = None
        if HAS_EVOLUTION_ML:
            try:
                self.evolution_extractor = EvolutionFeatureExtractor()
            except Exception as e:
                logger.warning(f"[UnifiedFeatureExtractor] Failed to init evolution extractor: {e}")
        
        self.training_extractor = None
        if HAS_TRAINING_ML:
            try:
                self.training_extractor = FeatureEngine()
            except Exception as e:
                logger.warning(f"[UnifiedFeatureExtractor] Failed to init training extractor: {e}")
    
    def extract_signal_features(self, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for signal scoring (used by MLScorer)."""
        # MLScorer has its own feature extraction built-in
        # This is a placeholder for future unified feature extraction
        return signal_data
    
    def extract_config_features(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from config for evolution/training."""
        if self.evolution_extractor:
            try:
                return self.evolution_extractor.extract_config_features(config)
            except Exception as e:
                logger.warning(f"[UnifiedFeatureExtractor] Evolution extractor failed: {e}")
        
        # Fallback: basic feature extraction
        return {
            'min_score': config.get('min_score', 70),
            'sl_R': config.get('sl_R', 0.7),
            'tp_R': config.get('tp_R', 1.5),
        }
    
    def extract_trade_features(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from trade data for training."""
        if self.training_extractor:
            try:
                return self.training_extractor.extract_features(trade_data)
            except Exception as e:
                logger.warning(f"[UnifiedFeatureExtractor] Training extractor failed: {e}")
        
        # Fallback: basic feature extraction
        return trade_data


class UnifiedMLFramework:
    """
    Unified ML Framework - Single source of truth for all ML operations.
    
    Components:
    - signal_scorer: MLScorer for signal quality prediction
    - parameter_optimizer: ML Training System for config optimization
    - evolution_learner: ML Components for evolution learning
    - feature_extractor: Unified feature extraction
    - model_registry: Model versioning and management
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize unified ML framework."""
        logger.info("[UnifiedMLFramework] Initializing unified ML framework...")
        
        # Model registry
        model_base = Path(model_dir) if model_dir else None
        self.model_registry = ModelRegistry(model_base)
        
        # Feature extractor
        self.feature_extractor = UnifiedFeatureExtractor()
        
        # Signal scorer (always available - core functionality)
        logger.info("[UnifiedMLFramework] Initializing signal scorer...")
        
        # PRIORITY 1: Try Hindsight ML (Master Hindsight System) - BEST
        if HAS_HINDSIGHT_ML:
            try:
                self.signal_scorer = HindsightMLScorer(model_dir=model_dir)
                if self.signal_scorer.enabled:
                    logger.info(f"[UnifiedMLFramework] ✓✓✓ Using HindsightMLScorer (Master Hindsight System)")
                    self.model_registry.register_model("signal_scorer", self.signal_scorer, "hindsight_v1")
                else:
                    raise Exception("Hindsight ML not enabled")
            except Exception as e:
                logger.warning(f"[UnifiedMLFramework] Hindsight ML unavailable: {e}")
                # Skip HybridMLScorer (has fallback issues), go straight to MLScorer
                logger.warning("[UnifiedMLFramework] Skipping HybridMLScorer (fallback issues), using MLScorer directly")
                self.signal_scorer = MLScorer(model_dir=model_dir)
                self.model_registry.register_model("signal_scorer", self.signal_scorer, "v4")
        # PRIORITY 2: Try Hybrid ML (Ensemble) - GOOD
        elif HAS_HYBRID_ML:
            try:
                self.signal_scorer = HybridMLScorer(model_dir=model_dir)
                logger.info(f"[UnifiedMLFramework] ✓ Using HybridMLScorer (Ensemble: XGB + LGB + RF)")
                self.model_registry.register_model("signal_scorer", self.signal_scorer, "hybrid_v1")
            except Exception as e:
                logger.error(f"[UnifiedMLFramework] HybridMLScorer FAILED, falling back to MLScorer: {e}")
                self.signal_scorer = MLScorer(model_dir=model_dir)
                self.model_registry.register_model("signal_scorer", self.signal_scorer, "v4")
        # PRIORITY 3: Fallback to basic MLScorer
        else:
            logger.warning("[UnifiedMLFramework] Hindsight not available, using MLScorer (XGBoost only)")
            self.signal_scorer = MLScorer(model_dir=model_dir)
            self.model_registry.register_model("signal_scorer", self.signal_scorer, "v4")
        
        # Parameter optimizer (training system - optional)
        self.parameter_optimizer = None
        if HAS_TRAINING_ML:
            try:
                logger.info("[UnifiedMLFramework] Training ML components available")
                # Lazy initialization - only create when needed
                self._training_components = {
                    'hindsight_analyzer': HindsightAnalyzer,
                    'feature_engine': FeatureEngine,
                    'ensemble_trainer': EnsembleTrainer,
                    'optimizer': NSGAII,
                    'simulator': OHLCVSimulator,
                }
            except Exception as e:
                logger.warning(f"[UnifiedMLFramework] Training ML components failed: {e}")
                self._training_components = None
        else:
            logger.info("[UnifiedMLFramework] Training ML components not available")
            self._training_components = None
        
        # Evolution learner (evolution components - optional)
        self.evolution_learner = None
        if HAS_EVOLUTION_ML:
            try:
                logger.info("[UnifiedMLFramework] Evolution ML components available")
                # Lazy initialization
                self._evolution_components = {
                    'feature_extractor': EvolutionFeatureExtractor,
                    'experience_replay': ExperienceReplay,
                    'fitness_predictor': FitnessPredictor,
                }
            except Exception as e:
                logger.warning(f"[UnifiedMLFramework] Evolution ML components failed: {e}")
                self._evolution_components = None
        else:
            logger.info("[UnifiedMLFramework] Evolution ML components not available")
            self._evolution_components = None
        
        logger.info("[UnifiedMLFramework] Initialization complete")
    
    def get_signal_scorer(self) -> MLScorer:
        """Get signal scorer (always available)."""
        return self.signal_scorer
    
    def get_parameter_optimizer(self) -> Optional[Dict[str, Any]]:
        """Get parameter optimizer components (lazy initialization)."""
        if not self._training_components:
            return None
        
        if self.parameter_optimizer is None:
            self.parameter_optimizer = {
                'hindsight_analyzer': self._training_components['hindsight_analyzer'](),
                'feature_engine': self._training_components['feature_engine'](),
                'ensemble_trainer': self._training_components['ensemble_trainer'](),
                'optimizer': self._training_components['optimizer'],
                'simulator': self._training_components['simulator'](),
            }
            logger.info("[UnifiedMLFramework] Parameter optimizer initialized")
        
        return self.parameter_optimizer
    
    def get_evolution_learner(self) -> Optional[Dict[str, Any]]:
        """Get evolution learner components (lazy initialization)."""
        if not self._evolution_components:
            return None
        
        if self.evolution_learner is None:
            self.evolution_learner = {
                'feature_extractor': self._evolution_components['feature_extractor'](),
                'experience_replay': self._evolution_components['experience_replay'](),
                'fitness_predictor': self._evolution_components['fitness_predictor'](),
            }
            logger.info("[UnifiedMLFramework] Evolution learner initialized")
        
        return self.evolution_learner
    
    def score_signal(
        self,
        symbol: str,
        pct_change_24h: float,
        volume_24h: float,
        spread_bps: float,
        orderbook: Optional[Dict] = None,
        indicators: Optional[Dict] = None,
        latency_ms: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Score a trading signal using ML.
        
        This is the primary interface for signal scoring.
        """
        return self.signal_scorer.score_signal(
            symbol=symbol,
            pct_change_24h=pct_change_24h,
            volume_24h=volume_24h,
            spread_bps=spread_bps,
            orderbook=orderbook,
            indicators=indicators,
            latency_ms=latency_ms
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get framework status and availability."""
        return {
            'signal_scorer': {
                'available': True,
                'loaded': self.signal_scorer._loaded if hasattr(self.signal_scorer, '_loaded') else False,
                'error': self.signal_scorer._load_error if hasattr(self.signal_scorer, '_load_error') else None,
            },
            'parameter_optimizer': {
                'available': HAS_TRAINING_ML,
                'initialized': self.parameter_optimizer is not None,
            },
            'evolution_learner': {
                'available': HAS_EVOLUTION_ML,
                'initialized': self.evolution_learner is not None,
            },
            'feature_extractor': {
                'available': True,
                'evolution_extractor': self.feature_extractor.evolution_extractor is not None,
                'training_extractor': self.feature_extractor.training_extractor is not None,
            },
        }


# Global instance (lazy initialization)
_global_framework: Optional[UnifiedMLFramework] = None


def get_unified_ml_framework(model_dir: Optional[str] = None) -> UnifiedMLFramework:
    """Get or create global unified ML framework instance."""
    global _global_framework
    if _global_framework is None:
        _global_framework = UnifiedMLFramework(model_dir=model_dir)
    return _global_framework


def reset_unified_ml_framework():
    """Reset global framework (for testing)."""
    global _global_framework
    _global_framework = None

