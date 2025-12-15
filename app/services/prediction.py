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

        for i, row in df.iterrows():
            if ("latitude" not in row) or pd.isna(row["latitude"]):
                raise ValueError(f"Missing latitude for plan_id={row.get('plan_id')}")
            if ("longitude" not in row) or pd.isna(row["longitude"]):
                raise ValueError(f"Missing longitude for plan_id={row.get('plan_id')}")
            
        weather_rows = []
        for i, row in df.iterrows():
            weather = WeatherService.fetch_weather(
                lat=row["latitude"],
                lon=row["longitude"],
                target_date=row["plan_date"]
            )
            weather_rows.append(weather)

        df['original_priority_flag'] = df['priority_flag']
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
        
        raw_predicted_production = mining_predictions[:, 0]
        df['predicted_production_ton'] = np.maximum(
            df['planned_production_ton'] * 0.05,  
            np.minimum(
                df['planned_production_ton'] * 1.3,  
                raw_predicted_production
            )
        )
        raw_efficiency = mining_predictions[:, 1]
        df['efficiency_factor'] = np.clip(raw_efficiency, 0.1, 1.0)                 
        raw_delay = mining_predictions[:, 2]
        df['cycle_delay_min'] = np.clip(raw_delay, 0, 180)
        
        df['production_gap_ton'] = df['planned_production_ton'] - df['predicted_production_ton']
        df['production_gap_pct'] = (df['production_gap_ton'] / df['planned_production_ton']) * 100
        
        df['confidence_score'] = df['efficiency_factor']
        
        df['risk_level'] = df.apply(self._calculate_mining_risk, axis=1)
        
        df['weather_impact'] = df.apply(self._classify_weather_impact, axis=1)

        negative_production = (df['predicted_production_ton'] < 0).sum()
        if negative_production > 0:
            print(f"WARNING: {negative_production} rows still have negative production after clipping!")
        
        low_efficiency = (df['efficiency_factor'] < 0.3).sum()
        if low_efficiency > 0:
            print(f"INFO: {low_efficiency} rows have efficiency < 0.3 (possible extreme weather)")
        
        extreme_gap = (df['production_gap_pct'].abs() > 50).sum()
        if extreme_gap > 0:
            print(f"INFO: {extreme_gap} rows have production gap > 50%")        
        
        return df
    
    def predict_shipping(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        
        if 'eta_date' in df.columns:
            df['eta_date'] = pd.to_datetime(df['eta_date'])

        weather_defaults = {
            "precipitation_mm": 0.0,
            "wind_speed_kmh": 10.0,
            "temp_day": 25.0,
            "cloud_cover_pct": 30.0
        }

        for col, default in weather_defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
                
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
        
        raw_loading_hours = shipping_predictions[:, 0]
        expected_base_hours = df['planned_volume_ton'] / df['loading_rate_tph']
        df['predicted_loading_hours'] = np.maximum(
            expected_base_hours * 0.5,   
            np.minimum(
                expected_base_hours * 5.0,  
                raw_loading_hours
            )
        )
        raw_efficiency = shipping_predictions[:, 1]
        df['loading_efficiency'] = np.clip(raw_efficiency, 0.0, 1.0)       
        raw_demurrage = shipping_predictions[:, 2]
        df['predicted_demurrage_cost'] = np.maximum(0, raw_demurrage)  
        
        df['confidence_score'] = df['loading_efficiency']
        
        df['risk_level'] = df.apply(self._calculate_shipping_risk, axis=1)
        
        df['status'] = df['loading_efficiency'].apply(
            lambda x: "SUSPENDED" if x == 0 else "OPERATIONAL"
        )
        
        df['weather_impact'] = df.apply(self._classify_shipping_weather, axis=1)
        
        df['recommended_action'] = df.apply(self._recommend_shipping_action, axis=1)

        negative_demurrage = (df['predicted_demurrage_cost'] < 0).sum()
        if negative_demurrage > 0:
            print(f"WARNING: {negative_demurrage} rows had negative demurrage (fixed to 0)")
        
        zero_efficiency = (df['loading_efficiency'] == 0).sum()
        if zero_efficiency > 0:
            print(f"INFO: {zero_efficiency} vessels marked as SUSPENDED (0% efficiency)")
        
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
