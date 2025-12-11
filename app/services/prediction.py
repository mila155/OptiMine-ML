import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import os
import joblib
import sys
from app.services.weather import WeatherService
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
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        try:
            self.models['priority'] = joblib.load(os.path.join(self.models_dir, 'model_priority.pkl'))
            self.models['mining'] = joblib.load(os.path.join(self.models_dir, 'model_mine.pkl'))
            self.models['shipping'] = joblib.load(os.path.join(self.models_dir, 'model_ship.pkl'))
            
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
        df = pd.DataFrame(data)
        
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_datetime(df['plan_date']).dt.date
        
        weather_rows = []
        for i, row in df.iterrows():
            weather = WeatherService.fetch_weather(
                lat=row["latitude"],
                lon=row["longitude"],
                target_date=row["plan_date"]
            )
            weather_rows.append(weather)
        
        weather_df = pd.DataFrame(weather_rows)
        df = pd.concat([df, weather_df], axis=1)

        priority_map = {"High": 3, "Medium": 2, "Low": 1}
        df['priority_score'] = df['priority_flag'].map(priority_map)
        
        priority_features = [
            "planned_production_ton",
            "hauling_distance_km",
            "priority_score",
            "precipitation_mm",
            "wind_speed_kmh",
            "cloud_cover_pct",
            "temp_day"
        ]
        
        X_priority = df[priority_features].copy()
        X_priority = X_priority.fillna(X_priority.mean())
        
        ai_priority_scores = self.models['priority'].predict(X_priority)
        df['ai_priority_score'] = ai_priority_scores
        
        priority_map_reverse = {2: "High", 1: "Medium", 0: "Low"}
        df['ai_priority_flag'] = df['ai_priority_score'].map(priority_map_reverse)
        
        mining_features = [
            "planned_production_ton",
            "hauling_distance_km",
            "ai_priority_score",
            "precipitation_mm",
            "wind_speed_kmh",
            "cloud_cover_pct",
            "temp_day"
        ]
        
        X_mine = df[mining_features].copy()
        X_mine = X_mine.fillna(X_mine.mean())
        
        mining_predictions = self.models['mining'].predict(X_mine)
        
        df['predicted_production_ton'] = mining_predictions[:, 0]  
        df['efficiency_factor'] = mining_predictions[:, 1]         
        df['cycle_delay_min'] = mining_predictions[:, 2]           
        
        df['production_gap_ton'] = df['planned_production_ton'] - df['predicted_production_ton']
        df['production_gap_pct'] = (df['production_gap_ton'] / df['planned_production_ton']) * 100
        
        df['confidence_score'] = df['efficiency_factor'].clip(0, 1)
        
        df['risk_level'] = df.apply(self._calculate_mining_risk, axis=1)
        
        df['weather_impact'] = df.apply(self._classify_weather_impact, axis=1)
        
        return df
    
    def predict_shipping(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        
        if 'eta_date' in df.columns:
            df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        shipping_features = [
            "planned_volume_ton",
            "loading_rate_tph",
            "precipitation_mm",
            "wind_speed_kmh",
            "temp_day",
            "cloud_cover_pct"
        ]
        
        X_ship = df[shipping_features].copy()
        X_ship = X_ship.fillna(X_ship.mean())
        
        shipping_predictions = self.models['shipping'].predict(X_ship)
        
        df['predicted_loading_hours'] = shipping_predictions[:, 0]    
        df['loading_efficiency'] = shipping_predictions[:, 1]         
        df['predicted_demurrage_cost'] = shipping_predictions[:, 2]   
        
        df['confidence_score'] = df['loading_efficiency'].clip(0, 1)
        
        df['risk_level'] = df.apply(self._calculate_shipping_risk, axis=1)
        
        df['status'] = df['loading_efficiency'].apply(
            lambda x: "SUSPENDED" if x == 0 else "OPERATIONAL"
        )
        
        df['weather_impact'] = df.apply(self._classify_shipping_weather, axis=1)
        
        df['recommended_action'] = df.apply(self._recommend_shipping_action, axis=1)
        
        return df
    
    @staticmethod
    def _calculate_mining_risk(row) -> str:
        if row['efficiency_factor'] < 0.7 or row['precipitation_mm'] > 20:
            return 'HIGH'
        elif row['efficiency_factor'] < 0.85 or row['precipitation_mm'] > 10:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @staticmethod
    def _calculate_shipping_risk(row) -> str:
        if row['predicted_demurrage_cost'] > 50000:
            return 'HIGH'
        elif row['predicted_demurrage_cost'] > 20000:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @staticmethod
    def _classify_weather_impact(row) -> str:
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
        if row['status'] == 'SUSPENDED':
            return "POSTPONE LOADING - Weather unsafe"
        elif row['predicted_demurrage_cost'] > 100000:
            return "URGENT - High demurrage risk, expedite loading"
        elif row['predicted_demurrage_cost'] > 50000:
            return "MONITOR - Moderate demurrage risk"
        else:
            return "PROCEED - Normal operations"
    
    def is_healthy(self) -> Dict[str, bool]:
        return {
            'priority': 'priority' in self.models,
            'mining': 'mining' in self.models,
            'shipping': 'shipping' in self.models,
            'priority_pipeline': 'priority_pipeline' in self.models,
            'mining_pipeline': 'mining_pipeline' in self.models,
            'shipping_pipeline': 'shipping_pipeline' in self.models
        }
