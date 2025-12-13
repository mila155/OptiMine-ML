from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date

# ==================== MINING SCHEMAS ====================

class MiningPlanInput(BaseModel):
    plan_id: str
    plan_date: date
    pit_id: str
    destination_rom: str  
    planned_production_ton: float
    hauling_distance_km: float
    priority_flag: str  # High, Medium, Low
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "plan_id": "MP1001",
                "plan_date": "2025-12-10",
                "pit_id": "PIT001",
                "destination_rom": "ROM_CENTRAL",
                "planned_production_ton": 8500,
                "hauling_distance_km": 12,
                "priority_flag": "High",
                "latitude": -0.2345,
                "longitude": 116.9876
            }
        }

class MiningPlanBatchInput(BaseModel):
    plans: List[MiningPlanInput]

class MiningPredictionOutput(BaseModel):
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
    role: str
    focus: str
    period: str
    total_days: int
    total_planned_production_ton: float
    total_predicted_production_ton: float
    production_gap_pct: float
    avg_efficiency: float
    avg_hauling_distance: float
    avg_rain: float
    max_wind: float
    high_risk_days: int
    daily_summary: List[Dict[str, Any]]
    ai_summary: Optional[str]

# ==================== SHIPPING SCHEMAS ====================

class ShippingPlanInput(BaseModel):
    shipment_id: str
    vessel_name: str
    assigned_jetty: str
    eta_date: date
    planned_volume_ton: float
    loading_rate_tph: float
    status: Optional[str] = None
    priority_flag: Optional[str] = None
    rom_id: str
    rom_lat: float
    rom_lon: float
    jetty_id: str
    jetty_lat: Optional[float] = None
    jetty_lon: Optional[float] = None
    
class ShippingPlanBatchInput(BaseModel):
    plans: List[ShippingPlanInput]

class ShippingPredictionOutput(BaseModel):
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
    role: str
    focus: str
    period: str
    total_days: int
    total_volume_ton: float
    total_vessels: int
    total_demurrage_cost: float
    avg_loading_efficiency: float
    high_risk_days: int
    avg_rain: float
    max_wind: float
    daily_summary: List[Dict[str, Any]]
    route_recommendations: Dict[str, Any]
    ai_summary: Optional[str] = None
    ai_notes: Optional[Dict[str, Any]] = None

# ==================== OPTIMIZATION SCHEMAS ====================

class OptimizationStrategy(BaseModel):
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
    plan_type: str
    generated_at: datetime
    executive_summary: Dict[str, Any]
    recommendations: List[OptimizationStrategy]

class ShippingOptimizationOutput(BaseModel):
    plan_type: str
    generated_at: datetime
    executive_summary: Dict[str, Any]
    recommendations: List[OptimizationStrategy]

class CombinedOptimizationOutput(BaseModel):
    mining_plan: MiningOptimizationOutput
    shipping_plan: ShippingOptimizationOutput
    timestamp: datetime

# ==================== HEALTH CHECK ====================

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    api_version: str

# ==================== RAG QUERY ====================

class PredictionRequest(BaseModel):
    features: list

class RAGQuery(BaseModel):
    query: str


# ==================== ERROR RESPONSE ====================

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime
