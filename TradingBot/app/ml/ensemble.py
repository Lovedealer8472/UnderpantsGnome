"""
Ensemble ML Models - Train and combine multiple models for robust predictions.

Models:
- XGBoost: Gradient boosting (currently used)
- Random Forest: Ensemble, robust to overfitting
- LightGBM: Fast gradient boosting
- CatBoost: Excellent with categorical features

Prediction targets:
1. Entry Quality: Should we enter? (0-100 score)
2. Exit Timing: When to exit? (time in bars)
3. Position Sizing: How much to risk? (% of equity)
4. R:R Optimizer: What SL/TP to use? (ATR multiples)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available (optional). System will use 3 models instead of 4.")
    print("  To install CatBoost on Windows, you need Visual Studio C++ Build Tools.")
    print("  Download from: https://visualstudio.microsoft.com/downloads/")


class EnsembleTrainer:
    """Train ensemble of ML models for trading predictions."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize ensemble trainer.
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_weights = {
            'xgboost': 0.3,
            'random_forest': 0.2,
            'lightgbm': 0.3,
            'catboost': 0.2
        }
        
    def train_entry_quality_predictor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Dict:
        """
        Train models to predict entry quality (should_have_entered).
        
        Args:
            X: Feature DataFrame
            y: Target (should_have_entered: 0 or 1)
            test_size: Test set size
            val_size: Validation set size
            
        Returns:
            Training metrics
        """
        print("\n" + "="*60)
        print("Training Entry Quality Predictor (Classification)")
        print("="*60)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Positive rate - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")
        
        self.feature_names = X.columns.tolist()
        metrics = {}
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            print("\n[1/4] Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='auc'
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred_val = xgb_model.predict_proba(X_val)[:, 1]
            y_pred_test = xgb_model.predict_proba(X_test)[:, 1]
            
            metrics['xgboost'] = {
                'val_auc': roc_auc_score(y_val, y_pred_val),
                'test_auc': roc_auc_score(y_test, y_pred_test),
                'val_acc': accuracy_score(y_val, (y_pred_val > 0.5).astype(int)),
                'test_acc': accuracy_score(y_test, (y_pred_test > 0.5).astype(int))
            }
            self.models['entry_quality_xgboost'] = xgb_model
            print(f"  Val AUC: {metrics['xgboost']['val_auc']:.4f}, Test AUC: {metrics['xgboost']['test_auc']:.4f}")
        
        # Train Random Forest
        if SKLEARN_AVAILABLE:
            print("\n[2/4] Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            y_pred_val = rf_model.predict_proba(X_val)[:, 1]
            y_pred_test = rf_model.predict_proba(X_test)[:, 1]
            
            metrics['random_forest'] = {
                'val_auc': roc_auc_score(y_val, y_pred_val),
                'test_auc': roc_auc_score(y_test, y_pred_test),
                'val_acc': accuracy_score(y_val, (y_pred_val > 0.5).astype(int)),
                'test_acc': accuracy_score(y_test, (y_pred_test > 0.5).astype(int))
            }
            self.models['entry_quality_random_forest'] = rf_model
            print(f"  Val AUC: {metrics['random_forest']['val_auc']:.4f}, Test AUC: {metrics['random_forest']['test_auc']:.4f}")
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n[3/4] Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred_val = lgb_model.predict_proba(X_val)[:, 1]
            y_pred_test = lgb_model.predict_proba(X_test)[:, 1]
            
            metrics['lightgbm'] = {
                'val_auc': roc_auc_score(y_val, y_pred_val),
                'test_auc': roc_auc_score(y_test, y_pred_test),
                'val_acc': accuracy_score(y_val, (y_pred_val > 0.5).astype(int)),
                'test_acc': accuracy_score(y_test, (y_pred_test > 0.5).astype(int))
            }
            self.models['entry_quality_lightgbm'] = lgb_model
            print(f"  Val AUC: {metrics['lightgbm']['val_auc']:.4f}, Test AUC: {metrics['lightgbm']['test_auc']:.4f}")
        
        # Train CatBoost
        if CATBOOST_AVAILABLE:
            print("\n[4/4] Training CatBoost...")
            cat_model = cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            y_pred_val = cat_model.predict_proba(X_val)[:, 1]
            y_pred_test = cat_model.predict_proba(X_test)[:, 1]
            
            metrics['catboost'] = {
                'val_auc': roc_auc_score(y_val, y_pred_val),
                'test_auc': roc_auc_score(y_test, y_pred_test),
                'val_acc': accuracy_score(y_val, (y_pred_val > 0.5).astype(int)),
                'test_acc': accuracy_score(y_test, (y_pred_test > 0.5).astype(int))
            }
            self.models['entry_quality_catboost'] = cat_model
            print(f"  Val AUC: {metrics['catboost']['val_auc']:.4f}, Test AUC: {metrics['catboost']['test_auc']:.4f}")
        
        # Ensemble prediction
        print("\n[Ensemble] Combining models...")
        ensemble_pred_val = self._ensemble_predict(X_val, 'entry_quality')
        ensemble_pred_test = self._ensemble_predict(X_test, 'entry_quality')
        
        metrics['ensemble'] = {
            'val_auc': roc_auc_score(y_val, ensemble_pred_val),
            'test_auc': roc_auc_score(y_test, ensemble_pred_test),
            'val_acc': accuracy_score(y_val, (ensemble_pred_val > 0.5).astype(int)),
            'test_acc': accuracy_score(y_test, (ensemble_pred_test > 0.5).astype(int))
        }
        print(f"  Ensemble Val AUC: {metrics['ensemble']['val_auc']:.4f}, Test AUC: {metrics['ensemble']['test_auc']:.4f}")
        
        return metrics
    
    def train_optimal_r_predictor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Dict:
        """
        Train models to predict optimal R-multiple.
        
        Args:
            X: Feature DataFrame
            y: Target (optimal_r_multiple)
            test_size: Test set size
            val_size: Validation set size
            
        Returns:
            Training metrics
        """
        print("\n" + "="*60)
        print("Training Optimal R Predictor (Regression)")
        print("="*60)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Target stats - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
        
        metrics = {}
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            print("\n[1/4] Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred_val = xgb_model.predict(X_val)
            y_pred_test = xgb_model.predict(X_test)
            
            metrics['xgboost'] = {
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'val_r2': r2_score(y_val, y_pred_val),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            self.models['optimal_r_xgboost'] = xgb_model
            print(f"  Val RMSE: {metrics['xgboost']['val_rmse']:.4f}, Test RMSE: {metrics['xgboost']['test_rmse']:.4f}")
        
        # Train Random Forest
        if SKLEARN_AVAILABLE:
            print("\n[2/4] Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            y_pred_val = rf_model.predict(X_val)
            y_pred_test = rf_model.predict(X_test)
            
            metrics['random_forest'] = {
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'val_r2': r2_score(y_val, y_pred_val),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            self.models['optimal_r_random_forest'] = rf_model
            print(f"  Val RMSE: {metrics['random_forest']['val_rmse']:.4f}, Test RMSE: {metrics['random_forest']['test_rmse']:.4f}")
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n[3/4] Training LightGBM...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred_val = lgb_model.predict(X_val)
            y_pred_test = lgb_model.predict(X_test)
            
            metrics['lightgbm'] = {
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'val_r2': r2_score(y_val, y_pred_val),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            self.models['optimal_r_lightgbm'] = lgb_model
            print(f"  Val RMSE: {metrics['lightgbm']['val_rmse']:.4f}, Test RMSE: {metrics['lightgbm']['test_rmse']:.4f}")
        
        # Train CatBoost
        if CATBOOST_AVAILABLE:
            print("\n[4/4] Training CatBoost...")
            cat_model = cb.CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            y_pred_val = cat_model.predict(X_val)
            y_pred_test = cat_model.predict(X_test)
            
            metrics['catboost'] = {
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'val_r2': r2_score(y_val, y_pred_val),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            self.models['optimal_r_catboost'] = cat_model
            print(f"  Val RMSE: {metrics['catboost']['val_rmse']:.4f}, Test RMSE: {metrics['catboost']['test_rmse']:.4f}")
        
        return metrics
    
    def _ensemble_predict(self, X: pd.DataFrame, task: str) -> np.ndarray:
        """Make ensemble prediction by combining model outputs."""
        predictions = []
        weights = []
        
        for model_name, weight in self.model_weights.items():
            model_key = f"{task}_{model_name}"
            if model_key in self.models:
                model = self.models[model_key]
                if 'entry_quality' in task:
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions.append(pred)
                weights.append(weight)
        
        if not predictions:
            return np.zeros(len(X))
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def save_models(self, timestamp: str):
        """Save all trained models."""
        print(f"\nSaving models to {self.output_dir}...")
        
        for model_name, model in self.models.items():
            model_path = self.output_dir / f"{model_name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            print(f"  Saved: {model_path.name}")
        
        # Save feature names
        features_path = self.output_dir / f"features_{timestamp}.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"  Saved: {features_path.name}")
        
        # Save model weights
        weights_path = self.output_dir / f"ensemble_weights_{timestamp}.json"
        with open(weights_path, 'w') as f:
            json.dump(self.model_weights, f)
        print(f"  Saved: {weights_path.name}")
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get feature importance from all models."""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                
                importance_dict[model_name] = {
                    'features': [self.feature_names[i] for i in indices],
                    'importances': [float(importances[i]) for i in indices]
                }
        
        return importance_dict

