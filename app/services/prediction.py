import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import os
import joblib
import sys
from app.services.transformers import (
    PreprocessorMining,
    FeatureSelector,
    PriorityEncoder,
    OutlierRemover,
    DuplicateRemover,
    MissingValueHandler,
    WeatherCategoryEncoder,
    DateFormatter,
    PreprocessorShipping,
    MiningPipelineWrapper
)

# Register class alias for joblib
sys.modules['__main__'].PreprocessorMining = PreprocessorMining
sys.modules['__main__'].FeatureSelector = FeatureSelector
sys.modules['__main__'].PriorityEncoder = PriorityEncoder
sys.modules['__main__'].OutlierRemover = OutlierRemover
sys.modules['__main__'].DuplicateRemover = DuplicateRemover
sys.modules['__main__'].MissingValueHandler = MissingValueHandler
sys.modules['__main__'].WeatherCategoryEncoder = WeatherCategoryEncoder
sys.modules['__main__'].DateFormatter = DateFormatter
sys.modules['__main__'].PreprocessorShipping = PreprocessorShipping
sys.modules['__main__'].MiningPipelineWrapper = MiningPipelineWrapper


class PredictionService:
    """Service untuk handle predictions menggunakan trained models"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize prediction service dengan load models"""
        self.models_dir = models_dir
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        """Load semua trained models"""
        try:
            # Load individual models
            self.models['priority'] = joblib.load(os.path.join(self.models_dir, 'model_priority.pkl'))
            self.models['mining'] = joblib.load(os.path.join(self.models_dir, 'model_mine.pkl'))
            self.models['shipping'] = joblib.load(os.path.join(self.models_dir, 'model_ship.pkl'))
            
            # Load pipelines (optional - jika ada)
            try:
                self.models['priority_pipeline'] = joblib.load(os.path.join(self.models_dir, 'priority_pipeline_complete.pkl'))
                self.models['mining_pipeline'] = joblib.load(os.path.join(self.models_dir, 'mining_pipeline_complete.pkl'))
                self.models['shipping_pipeline'] = joblib.load(os.path.join(self.models_dir, 'shipping_pipeline_complete.pkl'))
            except FileNotFoundError:
                print("⚠️ Pipeline files not found, using models only")
            
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict_mining(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Predict mining operations
        
        Args:
            data: List of mining plan dictionaries
            
        Returns:
            DataFrame with predictions
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure date is datetime
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_datetime(df['plan_date'])
        
        # Encode priority (sesuai notebook)
        priority_map = {"High": 3, "Medium": 2, "Low": 1}
        df['priority_score'] = df['priority_flag'].map(priority_map)
        
        # Feature names for priority prediction
        priority_features = [
            "planned_production_ton",
            "hauling_distance_km",
            "priority_score",
            "precipitation_mm",
            "wind_speed_kmh",
            "cloud_cover_pct",
            "temp_day"
        ]
        
        # Prepare features for priority prediction
        X_priority = df[priority_features].copy()
        X_priority = X_priority.fillna(X_priority.mean())
        
        # Predict AI priority (output: 0, 1, 2)
        ai_priority_scores = self.models['priority'].predict(X_priority)
        df['ai_priority_score'] = ai_priority_scores
        
        # Map priority scores to flags
        priority_map_reverse = {2: "High", 1: "Medium", 0: "Low"}
        df['ai_priority_flag'] = df['ai_priority_score'].map(priority_map_reverse)
        
        # Feature names for mining prediction (dengan AI priority)
        mining_features = [
            "planned_production_ton",
            "hauling_distance_km",
            "ai_priority_score",
            "precipitation_mm",
            "wind_speed_kmh",
            "cloud_cover_pct",
            "temp_day"
        ]
        
        # Prepare features for mining prediction
        X_mine = df[mining_features].copy()
        X_mine = X_mine.fillna(X_mine.mean())
        
        # Predict mining outputs (MultiOutputRegressor: 3 outputs)
        mining_predictions = self.models['mining'].predict(X_mine)
        
        # Extract predictions (sesuai notebook)
        df['predicted_production_ton'] = mining_predictions[:, 0]  # adjusted_production
        df['efficiency_factor'] = mining_predictions[:, 1]         # efficiency_factor
        df['cycle_delay_min'] = mining_predictions[:, 2]           # cycle_delay_min
        
        # Calculate gaps and metrics
        df['production_gap_ton'] = df['planned_production_ton'] - df['predicted_production_ton']
        df['production_gap_pct'] = (df['production_gap_ton'] / df['planned_production_ton']) * 100
        
        # Calculate confidence scores
        df['confidence_score'] = df['efficiency_factor'].clip(0, 1)
        
        # Determine risk level (sesuai notebook)
        df['risk_level'] = df.apply(self._calculate_mining_risk, axis=1)
        
        # Weather impact (sesuai notebook)
        df['weather_impact'] = df.apply(self._classify_weather_impact, axis=1)
        
        return df
    
    def predict_shipping(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Predict shipping operations
        
        Args:
            data: List of shipping plan dictionaries
            
        Returns:
            DataFrame with predictions
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure date is datetime
        if 'eta_date' in df.columns:
            df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        # Feature names for shipping prediction
        shipping_features = [
            "planned_volume_ton",
            "loading_rate_tph",
            "precipitation_mm",
            "wind_speed_kmh",
            "temp_day",
            "cloud_cover_pct"
        ]
        
        # Prepare features
        X_ship = df[shipping_features].copy()
        X_ship = X_ship.fillna(X_ship.mean())
        
        # Predict shipping outputs (MultiOutputRegressor: 3 outputs)
        shipping_predictions = self.models['shipping'].predict(X_ship)
        
        # Extract predictions (sesuai notebook)
        df['predicted_loading_hours'] = shipping_predictions[:, 0]    # adjusted_loading_hours
        df['loading_efficiency'] = shipping_predictions[:, 1]         # loading_efficiency
        df['predicted_demurrage_cost'] = shipping_predictions[:, 2]   # demurrage_cost_usd
        
        # Calculate confidence scores
        df['confidence_score'] = df['loading_efficiency'].clip(0, 1)
        
        # Determine risk level (sesuai notebook)
        df['risk_level'] = df.apply(self._calculate_shipping_risk, axis=1)
        
        # Determine status
        df['status'] = df['loading_efficiency'].apply(
            lambda x: "SUSPENDED" if x == 0 else "OPERATIONAL"
        )
        
        # Weather impact
        df['weather_impact'] = df.apply(self._classify_shipping_weather, axis=1)
        
        # Recommended action
        df['recommended_action'] = df.apply(self._recommend_shipping_action, axis=1)
        
        return df
    
    @staticmethod
    def _calculate_mining_risk(row) -> str:
        """Calculate risk level for mining (sesuai notebook)"""
        if row['efficiency_factor'] < 0.7 or row['precipitation_mm'] > 20:
            return 'HIGH'
        elif row['efficiency_factor'] < 0.85 or row['precipitation_mm'] > 10:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @staticmethod
    def _calculate_shipping_risk(row) -> str:
        """Calculate risk level for shipping (sesuai notebook)"""
        if row['predicted_demurrage_cost'] > 50000:
            return 'HIGH'
        elif row['predicted_demurrage_cost'] > 20000:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @staticmethod
    def _classify_weather_impact(row) -> str:
        """Classify weather impact for mining (sesuai notebook)"""
        rain = row['precipitation_mm']
        wind = row['wind_speed_kmh']
        
        if rain > 20 or wind > 40:
            return "SEVERE"
        elif rain > 10 or wind > 25:
            return "MODERATE"
        elif rain > 5 or wind > 15:
            return "LIGHT"
        else:
            return "MINIMAL"
    
    @staticmethod
    def _classify_shipping_weather(row) -> str:
        """Classify weather impact for shipping (sesuai notebook)"""
        wind = row['wind_speed_kmh']
        rain = row['precipitation_mm']
        
        if wind > 40 or rain > 30:
            return "SEVERE"
        elif wind > 30 or rain > 20:
            return "MODERATE"
        elif wind > 20 or rain > 10:
            return "LIGHT"
        else:
            return "MINIMAL"
    
    @staticmethod
    def _recommend_shipping_action(row) -> str:
        """Recommend action for shipping (sesuai notebook)"""
        if row['status'] == 'SUSPENDED':
            return "POSTPONE LOADING - Weather unsafe"
        elif row['predicted_demurrage_cost'] > 100000:
            return "URGENT - High demurrage risk, expedite loading"
        elif row['predicted_demurrage_cost'] > 50000:
            return "MONITOR - Moderate demurrage risk"
        else:
            return "PROCEED - Normal operations"
    
    def is_healthy(self) -> Dict[str, bool]:
        """Check if all models are loaded"""
        return {
            'priority': 'priority' in self.models,
            'mining': 'mining' in self.models,
            'shipping': 'shipping' in self.models,
            'priority_pipeline': 'priority_pipeline' in self.models,
            'mining_pipeline': 'mining_pipeline' in self.models,
            'shipping_pipeline': 'shipping_pipeline' in self.models
        }