"""
Database models (jika diperlukan untuk future development)
Saat ini menggunakan file-based models
"""

from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

class BaseModel:
    """Base model untuk semua predictions"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
    
    def load(self):
        """Load model dari file"""
        raise NotImplementedError
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions"""
        raise NotImplementedError


class MiningModel(BaseModel):
    """Mining operations model"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.feature_names = [
            "planned_production_ton",
            "hauling_distance_km",
            "ai_priority_score",
            "precipitation_mm",
            "wind_speed_kmh",
            "cloud_cover_pct",
            "temp_day"
        ]


class ShippingModel(BaseModel):
    """Shipping operations model"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.feature_names = [
            "planned_volume_ton",
            "loading_rate_tph",
            "precipitation_mm",
            "wind_speed_kmh",
            "temp_day",
            "cloud_cover_pct"
        ]


class PriorityModel(BaseModel):
    """Priority classification model"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.feature_names = [
            "planned_production_ton",
            "hauling_distance_km",
            "priority_score",
            "precipitation_mm",
            "wind_speed_kmh",
            "cloud_cover_pct",
            "temp_day"
        ]