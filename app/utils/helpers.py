"""
Helper functions untuk berbagai operasi
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

def convert_to_probabilities(mining_summary, shipping_summary):
    """Convert predictions to probability scores"""
    mining_probs = mining_summary.copy()

    # Production success probability
    mining_probs['production_success_prob'] = mining_probs['avg_efficiency']

    # On-time probability
    max_delay = mining_probs['avg_delay_min'].max()
    if max_delay > 0:
        mining_probs['on_time_prob'] = 1 - (mining_probs['avg_delay_min'] / max_delay)
    else:
        mining_probs['on_time_prob'] = 1.0

    # Weather safety probability
    mining_probs['weather_safe_prob'] = np.where(
        mining_probs['avg_rain_mm'] > 20, 0.3,
        np.where(mining_probs['avg_rain_mm'] > 10, 0.6,
                 np.where(mining_probs['avg_rain_mm'] > 5, 0.8, 0.95))
    )

    # Risk level probability
    risk_map = {'LOW': 0.9, 'MEDIUM': 0.6, 'HIGH': 0.3}
    mining_probs['low_risk_prob'] = mining_probs['risk_level'].map(risk_map)

    # SHIPPING
    shipping_probs = shipping_summary.copy()

    # Loading success probability
    shipping_probs['loading_success_prob'] = shipping_probs['avg_efficiency']

    # No demurrage probability
    max_demurrage = shipping_probs['total_demurrage_cost_usd'].max()
    if max_demurrage > 0:
        shipping_probs['no_demurrage_prob'] = 1 - (
            shipping_probs['total_demurrage_cost_usd'] / max_demurrage
        )
    else:
        shipping_probs['no_demurrage_prob'] = 1.0

    # Weather safety
    shipping_probs['weather_safe_prob'] = np.where(
        shipping_probs['max_wind_kmh'] > 40, 0.2,
        np.where(shipping_probs['max_wind_kmh'] > 30, 0.5,
                 np.where(shipping_probs['max_wind_kmh'] > 20, 0.7, 0.95))
    )

    # Risk level probability
    shipping_probs['low_risk_prob'] = shipping_probs['risk_level'].map(risk_map)

    return mining_probs, shipping_probs


def prepare_optimization_data(mining_probs, shipping_probs, combined_summary):
    """Prepare data structure for optimization"""
    daily_ops = []

    for idx, row in combined_summary.iterrows():
        date_str = str(row['date'])

        mine_data = mining_probs[mining_probs['date'] == row['date']]
        ship_data = shipping_probs[shipping_probs['date'] == row['date']]

        if not mine_data.empty:
            mine_data = mine_data.iloc[0]
        if not ship_data.empty:
            ship_data = ship_data.iloc[0]

        daily_entry = {
            "date": date_str,
            "day": idx + 1,
            "mining": {
                "planned_production_ton": float(row.get('planned_production_ton', 0)),
                "predicted_production_ton": float(row.get('predicted_production_ton', 0)),
                "probabilities": {
                    "production_success": float(mine_data.get('production_success_prob', 0.5)) if not mine_data.empty else 0.5,
                    "on_time_delivery": float(mine_data.get('on_time_prob', 0.5)) if not mine_data.empty else 0.5,
                    "weather_safe": float(mine_data.get('weather_safe_prob', 0.5)) if not mine_data.empty else 0.5,
                    "low_risk": float(mine_data.get('low_risk_prob', 0.5)) if not mine_data.empty else 0.5
                },
                "weather": {
                    "rain_mm": float(mine_data.get('avg_rain_mm', 0)) if not mine_data.empty else 0,
                    "wind_kmh": float(mine_data.get('max_wind_kmh', 0)) if not mine_data.empty else 0
                }
            },
            "shipping": {
                "total_volume_ton": float(row.get('total_volume_ton', 0)),
                "vessel_count": int(row.get('vessel_count', 0)),
                "probabilities": {
                    "loading_success": float(ship_data.get('loading_success_prob', 0.5)) if not ship_data.empty else 0.5,
                    "no_demurrage": float(ship_data.get('no_demurrage_prob', 0.5)) if not ship_data.empty else 0.5,
                    "weather_safe": float(ship_data.get('weather_safe_prob', 0.5)) if not ship_data.empty else 0.5
                },
                "demurrage_cost": float(ship_data.get('total_demurrage_cost_usd', 0)) if not ship_data.empty else 0
            },
            "financials": {
                "potential_revenue": float(row.get('potential_revenue', 0)),
                "total_cost": float(row.get('total_cost', 0)),
                "base_profit": float(row.get('base_profit', 0))
            },
            "balance": {
                "supply_demand_balance": float(row.get('supply_demand_balance', 0)),
                "balance_status": str(row.get('balance_status', 'UNKNOWN'))
            }
        }
        daily_ops.append(daily_entry)

    return {"daily_operations": daily_ops}


def calculate_financial_metrics(
    production_ton: float,
    price_per_ton: float = 50.0,
    cost_per_ton: float = 30.0
) -> Dict[str, float]:
    """
    Calculate financial metrics
    
    Args:
        production_ton: Production volume
        price_per_ton: Selling price per ton
        cost_per_ton: Cost per ton
        
    Returns:
        Financial metrics dictionary
    """
    revenue = production_ton * price_per_ton
    cost = production_ton * cost_per_ton
    profit = revenue - cost
    margin = (profit / revenue * 100) if revenue > 0 else 0
    
    return {
        'revenue': revenue,
        'cost': cost,
        'profit': profit,
        'margin_pct': margin
    }


def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format number as currency"""
    return f"{currency} {amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage"""
    return f"{value:.{decimals}f}%"


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics"""
    if not data:
        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range"""
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return start <= end
    except:
        return False


def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of dates in range"""
    try:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return [date.strftime('%Y-%m-%d') for date in dates]
    except:
        return []