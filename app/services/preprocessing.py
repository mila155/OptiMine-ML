"""
Data preprocessing service
Handles data cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class PreprocessingService:
    """Service untuk preprocessing data sebelum prediction"""
    
    def __init__(self):
        self.priority_map = {"High": 3, "Medium": 2, "Low": 1}
        self.priority_reverse_map = {2: "High", 1: "Medium", 0: "Low"}
    
    def preprocess_mining_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess mining data
        
        Args:
            data: List of mining plan dictionaries
            
        Returns:
            Preprocessed DataFrame
        """
        df = pd.DataFrame(data)
        
        # Convert dates
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_datetime(df['plan_date'])
        
        # Encode priority
        df['priority_score'] = df['priority_flag'].map(self.priority_map)
        
        # Fill missing values
        numeric_columns = [
            'precipitation_mm', 'wind_speed_kmh', 
            'cloud_cover_pct', 'temp_day'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0)
        
        # Add derived features
        df['weather_severity_score'] = self._calculate_weather_severity(df)
        df['distance_category'] = pd.cut(
            df['hauling_distance_km'], 
            bins=[0, 5, 10, 20, float('inf')],
            labels=['Very Short', 'Short', 'Medium', 'Long']
        )
        
        return df
    
    def preprocess_shipping_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess shipping data
        
        Args:
            data: List of shipping plan dictionaries
            
        Returns:
            Preprocessed DataFrame
        """
        df = pd.DataFrame(data)
        
        # Convert dates
        if 'eta_date' in df.columns:
            df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        # Fill missing values
        numeric_columns = [
            'precipitation_mm', 'wind_speed_kmh',
            'cloud_cover_pct', 'temp_day'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0)
        
        # Add derived features
        df['weather_severity_score'] = self._calculate_weather_severity(df)
        df['expected_loading_hours'] = df['planned_volume_ton'] / df['loading_rate_tph']
        df['volume_category'] = pd.cut(
            df['planned_volume_ton'],
            bins=[0, 30000, 50000, 70000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        return df
    
    @staticmethod
    def _calculate_weather_severity(df: pd.DataFrame) -> pd.Series:
        """
        Calculate weather severity score
        
        Args:
            df: DataFrame with weather columns
            
        Returns:
            Weather severity score (0-100)
        """
        rain_score = (df['precipitation_mm'] / 50) * 40  # Max 40 points
        wind_score = (df['wind_speed_kmh'] / 60) * 40   # Max 40 points
        cloud_score = (df['cloud_cover_pct'] / 100) * 20 # Max 20 points
        
        total_score = rain_score + wind_score + cloud_score
        return total_score.clip(0, 100)
    
    def validate_mining_input(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate mining input data
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = [
            'plan_id', 'plan_date', 'pit_id', 
            'planned_production_ton', 'hauling_distance_km', 'priority_flag'
        ]
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate ranges
        if data['planned_production_ton'] <= 0:
            return False, "planned_production_ton must be positive"
        
        if data['hauling_distance_km'] < 0:
            return False, "hauling_distance_km cannot be negative"
        
        if data['priority_flag'] not in ['High', 'Medium', 'Low']:
            return False, "priority_flag must be High, Medium, or Low"
        
        return True, ""
    
    def validate_shipping_input(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate shipping input data
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = [
            'shipment_id', 'vessel_name', 'assigned_jetty',
            'eta_date', 'planned_volume_ton', 'loading_rate_tph'
        ]
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate ranges
        if data['planned_volume_ton'] <= 0:
            return False, "planned_volume_ton must be positive"
        
        if data['loading_rate_tph'] <= 0:
            return False, "loading_rate_tph must be positive"
        
        return True, ""
    
    def extract_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Extract time-based features from date column
        
        Args:
            df: DataFrame with date column
            date_column: Name of date column
            
        Returns:
            DataFrame with additional time features
        """
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_of_month'] = df[date_column].dt.day
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        
        return df