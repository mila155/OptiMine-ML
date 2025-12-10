from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date

# ==================== MINING SCHEMAS ====================

class MiningPlanInput(BaseModel):
    """Input schema untuk single mining plan"""
    plan_id: str
    plan_date: date
    pit_id: str
    destination_rom: str  
    planned_production_ton: float
    hauling_distance_km: float
    priority_flag: str  # High, Medium, Low
    precipitation_mm: Optional[float] = 0
    wind_speed_kmh: Optional[float] = 0
    cloud_cover_pct: Optional[float] = 0
    temp_day: Optional[float] = 25
    
    class Config:
        json_schema_extra = {
            "example": {
                "plan_id": "MP1001",
                "plan_date": "2025-12-10",
                "pit_id": "PIT_TUTUPAN",
                "destination_rom": "ROM_CENTRAL",
                "planned_production_ton": 8500,
                "hauling_distance_km": 12,
                "priority_flag": "High",
                "precipitation_mm": 5.2,
                "wind_speed_kmh": 15.3,
                "cloud_cover_pct": 60,
                "temp_day": 28
            }
        }

class MiningPlanBatchInput(BaseModel):
    """Batch input untuk multiple mining plans"""
    plans: List[MiningPlanInput]

class MiningPredictionOutput(BaseModel):
    """Output schema untuk mining prediction"""
    plan_id: str
    plan_date: date
    pit_id: str
    planned_production_ton: float
    predicted_production_ton: float
    production_gap_ton: float
    production_gap_pct: float
    efficiency_factor: float
    cycle_delay_min: float
    ai_priority_flag: str
    ai_priority_score: int
    original_priority_flag: str
    confidence_score: float
    risk_level: str
    weather_impact: str

class MiningSummaryOutput(BaseModel):
    """Summary output untuk mining operations"""
    period: str
    total_days: int
    total_planned_production_ton: float
    total_predicted_production_ton: float
    avg_efficiency: float
    high_risk_days: int
    daily_summary: List[Dict[str, Any]]
    ai_summary: Optional[str] = None

# ==================== SHIPPING SCHEMAS ====================

class ShippingPlanInput(BaseModel):
    """Input schema untuk single shipping plan"""
    shipment_id: str
    vessel_name: str
    assigned_jetty: str
    eta_date: date
    planned_volume_ton: float
    loading_rate_tph: float
    precipitation_mm: Optional[float] = 0
    wind_speed_kmh: Optional[float] = 0
    cloud_cover_pct: Optional[float] = 0
    temp_day: Optional[float] = 25
    
    class Config:
        json_schema_extra = {
            "example": {
                "shipment_id": "SH7001",
                "vessel_name": "MV-OCEAN",
                "assigned_jetty": "JTY-01",
                "eta_date": "2025-12-10",
                "planned_volume_ton": 45000,
                "loading_rate_tph": 1200,
                "precipitation_mm": 3.5,
                "wind_speed_kmh": 22.5,
                "cloud_cover_pct": 45,
                "temp_day": 27
            }
        }

class ShippingPlanBatchInput(BaseModel):
    """Batch input untuk multiple shipping plans"""
    plans: List[ShippingPlanInput]

class ShippingPredictionOutput(BaseModel):
    """Output schema untuk shipping prediction"""
    shipment_id: str
    vessel_name: str
    assigned_jetty: str
    eta_date: date
    planned_volume_ton: float
    predicted_loading_hours: float
    loading_efficiency: float
    predicted_demurrage_cost: float
    confidence_score: float
    risk_level: str
    status: str
    weather_impact: str
    recommended_action: str

class ShippingSummaryOutput(BaseModel):
    """Summary output untuk shipping operations"""
    period: str
    total_days: int
    total_volume_ton: float
    total_vessels: int
    total_demurrage_cost: float
    avg_loading_efficiency: float
    high_risk_days: int
    daily_summary: List[Dict[str, Any]]
    route_recommendations: Optional[Dict[str, Any]] = None
    ai_summary: Optional[str] = None

# ==================== OPTIMIZATION SCHEMAS ====================

class OptimizationStrategy(BaseModel):
    """Single optimization strategy"""
    plan_id: int
    plan_name: str
    strategy_description: str
    optimized_schedule: List[Dict[str, Any]]
    financial_impact: Dict[str, float]
    implementation_steps: List[str]
    strengths: List[str]
    limitations: List[str]
    justification: Optional[str] = None

class MiningOptimizationOutput(BaseModel):
    """Output untuk mining optimization"""
    plan_type: str
    generated_at: datetime
    executive_summary: Dict[str, Any]
    recommendations: List[OptimizationStrategy]

class ShippingOptimizationOutput(BaseModel):
    """Output untuk shipping optimization"""
    plan_type: str
    generated_at: datetime
    executive_summary: Dict[str, Any]
    recommendations: List[OptimizationStrategy]

class CombinedOptimizationOutput(BaseModel):
    """Combined optimization output"""
    mining_plan: MiningOptimizationOutput
    shipping_plan: ShippingOptimizationOutput
    timestamp: datetime

# ==================== HEALTH CHECK ====================

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    api_version: str

# ==================== ERROR RESPONSE ====================

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime