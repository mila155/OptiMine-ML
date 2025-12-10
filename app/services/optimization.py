"""
Optimization service
Handles schedule optimization and recommendations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

class OptimizationService:
    """Service untuk optimization dan scheduling"""
    
    def __init__(self):
        self.optimization_strategies = {
            'cost_minimization': self._optimize_cost,
            'throughput_maximization': self._optimize_throughput,
            'risk_minimization': self._optimize_risk,
            'balanced': self._optimize_balanced
        }
    
    def optimize_mining_schedule(
        self, 
        predictions: pd.DataFrame,
        strategy: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Optimize mining schedule based on predictions
        
        Args:
            predictions: DataFrame with mining predictions
            strategy: Optimization strategy to use
            
        Returns:
            Optimization results with recommendations
        """
        if strategy not in self.optimization_strategies:
            strategy = 'balanced'
        
        # Sort by priority and efficiency
        optimized = predictions.copy()
        optimized['optimization_score'] = self._calculate_optimization_score(
            optimized, strategy
        )
        optimized = optimized.sort_values('optimization_score', ascending=False)
        
        # Generate recommendations
        recommendations = []
        for idx, row in optimized.head(10).iterrows():
            rec = {
                'plan_id': row['plan_id'],
                'date': str(row['plan_date']),
                'pit_id': row['pit_id'],
                'priority': row['ai_priority_flag'],
                'reason': self._generate_recommendation_reason(row),
                'expected_output_ton': float(row['predicted_production_ton']),
                'confidence': float(row['confidence_score'])
            }
            recommendations.append(rec)
        
        # Calculate metrics
        total_planned = optimized['planned_production_ton'].sum()
        total_predicted = optimized['predicted_production_ton'].sum()
        avg_efficiency = optimized['efficiency_factor'].mean()
        
        return {
            'strategy': strategy,
            'total_plans': len(optimized),
            'metrics': {
                'total_planned_ton': float(total_planned),
                'total_predicted_ton': float(total_predicted),
                'achievement_rate': float(total_predicted / total_planned * 100),
                'avg_efficiency': float(avg_efficiency)
            },
            'top_recommendations': recommendations,
            'high_risk_days': int((optimized['risk_level'] == 'HIGH').sum())
        }
    
    def optimize_shipping_schedule(
        self,
        predictions: pd.DataFrame,
        strategy: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Optimize shipping schedule based on predictions
        
        Args:
            predictions: DataFrame with shipping predictions
            strategy: Optimization strategy to use
            
        Returns:
            Optimization results with recommendations
        """
        optimized = predictions.copy()
        optimized['optimization_score'] = self._calculate_shipping_score(
            optimized, strategy
        )
        optimized = optimized.sort_values('optimization_score', ascending=False)
        
        # Generate recommendations
        recommendations = []
        for idx, row in optimized.head(10).iterrows():
            rec = {
                'shipment_id': row['shipment_id'],
                'vessel': row['vessel_name'],
                'jetty': row['assigned_jetty'],
                'eta': str(row['eta_date']),
                'action': row['recommended_action'],
                'demurrage_risk_usd': float(row['predicted_demurrage_cost']),
                'confidence': float(row['confidence_score'])
            }
            recommendations.append(rec)
        
        # Calculate metrics
        total_volume = optimized['planned_volume_ton'].sum()
        total_demurrage = optimized['predicted_demurrage_cost'].sum()
        avg_efficiency = optimized['loading_efficiency'].mean()
        
        return {
            'strategy': strategy,
            'total_shipments': len(optimized),
            'metrics': {
                'total_volume_ton': float(total_volume),
                'total_demurrage_usd': float(total_demurrage),
                'avg_loading_efficiency': float(avg_efficiency),
                'vessels_at_risk': int((optimized['risk_level'] == 'HIGH').sum())
            },
            'top_recommendations': recommendations,
            'suspended_operations': int((optimized['status'] == 'SUSPENDED').sum())
        }
    
    def _calculate_optimization_score(
        self, 
        df: pd.DataFrame, 
        strategy: str
    ) -> pd.Series:
        """Calculate optimization score for mining"""
        if strategy == 'cost_minimization':
            # Prioritize short distance, low delay
            score = (
                (1 / (df['hauling_distance_km'] + 1)) * 40 +
                (1 / (df['cycle_delay_min'] + 1)) * 30 +
                df['efficiency_factor'] * 30
            )
        
        elif strategy == 'throughput_maximization':
            # Prioritize high production, high efficiency
            score = (
                (df['predicted_production_ton'] / df['predicted_production_ton'].max()) * 50 +
                df['efficiency_factor'] * 50
            )
        
        elif strategy == 'risk_minimization':
            # Prioritize low risk, good weather
            risk_score = {'LOW': 100, 'MEDIUM': 60, 'HIGH': 20}
            score = (
                df['risk_level'].map(risk_score) * 0.5 +
                (1 - df['precipitation_mm'] / 50) * 30 +
                df['efficiency_factor'] * 20
            )
        
        else:  # balanced
            score = (
                df['efficiency_factor'] * 30 +
                (df['predicted_production_ton'] / df['predicted_production_ton'].max()) * 30 +
                (1 / (df['cycle_delay_min'] + 1)) * 20 +
                (df['ai_priority_score'] / 2) * 20
            )
        
        return score
    
    def _calculate_shipping_score(
        self,
        df: pd.DataFrame,
        strategy: str
    ) -> pd.Series:
        """Calculate optimization score for shipping"""
        if strategy == 'cost_minimization':
            # Minimize demurrage
            max_demurrage = df['predicted_demurrage_cost'].max()
            score = (
                (1 - df['predicted_demurrage_cost'] / (max_demurrage + 1)) * 60 +
                df['loading_efficiency'] * 40
            )
        
        elif strategy == 'throughput_maximization':
            # Maximize volume
            score = (
                (df['planned_volume_ton'] / df['planned_volume_ton'].max()) * 50 +
                df['loading_efficiency'] * 50
            )
        
        elif strategy == 'risk_minimization':
            risk_score = {'LOW': 100, 'MEDIUM': 60, 'HIGH': 20}
            score = (
                df['risk_level'].map(risk_score) * 0.6 +
                (1 - df['wind_speed_kmh'] / 60) * 40
            )
        
        else:  # balanced
            max_demurrage = df['predicted_demurrage_cost'].max()
            score = (
                df['loading_efficiency'] * 35 +
                (1 - df['predicted_demurrage_cost'] / (max_demurrage + 1)) * 35 +
                (df['planned_volume_ton'] / df['planned_volume_ton'].max()) * 30
            )
        
        return score
    
    @staticmethod
    def _generate_recommendation_reason(row) -> str:
        """Generate human-readable recommendation reason"""
        reasons = []
        
        if row['ai_priority_flag'] == 'High':
            reasons.append("High priority operation")
        
        if row['efficiency_factor'] > 0.9:
            reasons.append("Excellent efficiency expected")
        
        if row['risk_level'] == 'LOW':
            reasons.append("Low operational risk")
        
        if row['weather_impact'] == 'MINIMAL':
            reasons.append("Favorable weather conditions")
        
        return " | ".join(reasons) if reasons else "Standard operation"
    
    def generate_daily_plan(
        self,
        predictions: pd.DataFrame,
        target_date: str
    ) -> Dict[str, Any]:
        """
        Generate optimized daily operational plan
        
        Args:
            predictions: DataFrame with predictions
            target_date: Date to generate plan for
            
        Returns:
            Daily operational plan
        """
        target = pd.to_datetime(target_date)
        daily_data = predictions[predictions['plan_date'] == target]
        
        if daily_data.empty:
            return {'error': 'No data for specified date'}
        
        # Sort by priority and efficiency
        daily_data = daily_data.sort_values(
            ['ai_priority_score', 'efficiency_factor'],
            ascending=[False, False]
        )
        
        plan = {
            'date': target_date,
            'total_operations': len(daily_data),
            'sequence': [],
            'summary': {
                'total_planned_ton': float(daily_data['planned_production_ton'].sum()),
                'total_predicted_ton': float(daily_data['predicted_production_ton'].sum()),
                'avg_efficiency': float(daily_data['efficiency_factor'].mean()),
                'high_priority_count': int((daily_data['ai_priority_flag'] == 'High').sum())
            }
        }
        
        for idx, row in daily_data.iterrows():
            operation = {
                'sequence': len(plan['sequence']) + 1,
                'plan_id': row['plan_id'],
                'pit_id': row['pit_id'],
                'priority': row['ai_priority_flag'],
                'target_ton': float(row['planned_production_ton']),
                'expected_ton': float(row['predicted_production_ton']),
                'efficiency': float(row['efficiency_factor']),
                'risk': row['risk_level']
            }
            plan['sequence'].append(operation)
        
        return plan