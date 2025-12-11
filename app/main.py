import pandas as pd
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List
from app.services.prediction import PredictionService
import os

from app.schemas import (
    MiningPlanInput, MiningPlanBatchInput, MiningPredictionOutput, MiningSummaryOutput,
    ShippingPlanInput, ShippingPlanBatchInput, ShippingPredictionOutput, ShippingSummaryOutput,
    HealthCheck, ErrorResponse
)

app = FastAPI(
    title="OptiMine API",
    description="AI-powered Mining & Shipping Operations Optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_dir = os.getenv("MODELS_DIR", "models")
prediction_service = PredictionService(models_dir=models_dir)

# ==================== HEALTH CHECK ====================

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Welcome to OptiMine API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    models_status = prediction_service.is_healthy()
    
    return HealthCheck(
        status="healthy" if all(models_status.values()) else "degraded",
        timestamp=datetime.now(),
        models_loaded=models_status,
        api_version="1.0.0"
    )

# ==================== MINING ENDPOINTS ====================

@app.post("/mining/upload-csv", tags=["Mining"])
async def upload_csv_for_mining(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        records = df.to_dict(orient="records")

        result_df = prediction_service.predict_mining(records)

        return result_df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CSV processing failed: {str(e)}"
        )


@app.post("/mining/predict", response_model=List[MiningPredictionOutput], tags=["Mining"])
async def predict_mining_single(plan: MiningPlanInput):
    try:
        data = [plan.model_dump()]
        
        result_df = prediction_service.predict_mining(data)
        
        response = []
        for _, row in result_df.iterrows():
            response.append(MiningPredictionOutput(
                plan_id=row['plan_id'],
                plan_date=row['plan_date'],
                pit_id=row['pit_id'],
                planned_production_ton=float(row['planned_production_ton']),
                predicted_production_ton=float(row['predicted_production_ton']),
                production_gap_ton=float(row['production_gap_ton']),
                production_gap_pct=float(row['production_gap_pct']),
                efficiency_factor=float(row['efficiency_factor']),
                cycle_delay_min=float(row['cycle_delay_min']),
                ai_priority_flag=row['ai_priority_flag'],
                ai_priority_score=int(row['ai_priority_score']),
                original_priority_flag=row['priority_flag'],
                confidence_score=float(row['confidence_score']),
                risk_level=row['risk_level'],
                weather_impact=row['weather_impact']
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/mining/predict/batch", response_model=List[MiningPredictionOutput], tags=["Mining"])
async def predict_mining_batch(batch: MiningPlanBatchInput):
    try:
        data = [plan.model_dump() for plan in batch.plans]
        
        result_df = prediction_service.predict_mining(data)
        
        response = []
        for _, row in result_df.iterrows():
            response.append(MiningPredictionOutput(
                plan_id=row['plan_id'],
                plan_date=row['plan_date'],
                pit_id=row['pit_id'],
                planned_production_ton=float(row['planned_production_ton']),
                predicted_production_ton=float(row['predicted_production_ton']),
                production_gap_ton=float(row['production_gap_ton']),
                production_gap_pct=float(row['production_gap_pct']),
                efficiency_factor=float(row['efficiency_factor']),
                cycle_delay_min=float(row['cycle_delay_min']),
                ai_priority_flag=row['ai_priority_flag'],
                ai_priority_score=int(row['ai_priority_score']),
                original_priority_flag=row['priority_flag'],
                confidence_score=float(row['confidence_score']),
                risk_level=row['risk_level'],
                weather_impact=row['weather_impact']
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/mining/summary", response_model=MiningSummaryOutput, tags=["Mining"])
async def get_mining_summary(batch: MiningPlanBatchInput):
    try:
        data = [plan.model_dump() for plan in batch.plans]
        
        result_df = prediction_service.predict_mining(data)
        
        summary = {
            "period": f"{result_df['plan_date'].min()} to {result_df['plan_date'].max()}",
            "total_days": len(result_df['plan_date'].unique()),
            "total_planned_production_ton": float(result_df['planned_production_ton'].sum()),
            "total_predicted_production_ton": float(result_df['predicted_production_ton'].sum()),
            "avg_efficiency": float(result_df['efficiency_factor'].mean()),
            "high_risk_days": int((result_df['risk_level'] == 'HIGH').sum()),
            "daily_summary": result_df.groupby('plan_date').agg({
                'planned_production_ton': 'sum',
                'predicted_production_ton': 'sum',
                'efficiency_factor': 'mean',
                'risk_level': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN'
            }).reset_index().to_dict('records')
        }
        
        return MiningSummaryOutput(**summary)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary generation failed: {str(e)}"
        )

# ==================== SHIPPING ENDPOINTS ====================

@app.post("/shipping/predict", response_model=List[ShippingPredictionOutput], tags=["Shipping"])
async def predict_shipping_single(plan: ShippingPlanInput):
    try:
        data = [plan.model_dump()]
        
        result_df = prediction_service.predict_shipping(data)
        
        response = []
        for _, row in result_df.iterrows():
            response.append(ShippingPredictionOutput(
                shipment_id=row['shipment_id'],
                vessel_name=row['vessel_name'],
                assigned_jetty=row['assigned_jetty'],
                eta_date=row['eta_date'],
                planned_volume_ton=float(row['planned_volume_ton']),
                predicted_loading_hours=float(row['predicted_loading_hours']),
                loading_efficiency=float(row['loading_efficiency']),
                predicted_demurrage_cost=float(row['predicted_demurrage_cost']),
                confidence_score=float(row['confidence_score']),
                risk_level=row['risk_level'],
                status=row['status'],
                weather_impact=row['weather_impact'],
                recommended_action=row['recommended_action']
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/shipping/predict/batch", response_model=List[ShippingPredictionOutput], tags=["Shipping"])
async def predict_shipping_batch(batch: ShippingPlanBatchInput):
    try:
        data = [plan.model_dump() for plan in batch.plans]
        
        result_df = prediction_service.predict_shipping(data)
        
        response = []
        for _, row in result_df.iterrows():
            response.append(ShippingPredictionOutput(
                shipment_id=row['shipment_id'],
                vessel_name=row['vessel_name'],
                assigned_jetty=row['assigned_jetty'],
                eta_date=row['eta_date'],
                planned_volume_ton=float(row['planned_volume_ton']),
                predicted_loading_hours=float(row['predicted_loading_hours']),
                loading_efficiency=float(row['loading_efficiency']),
                predicted_demurrage_cost=float(row['predicted_demurrage_cost']),
                confidence_score=float(row['confidence_score']),
                risk_level=row['risk_level'],
                status=row['status'],
                weather_impact=row['weather_impact'],
                recommended_action=row['recommended_action']
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/shipping/summary", response_model=ShippingSummaryOutput, tags=["Shipping"])
async def get_shipping_summary(batch: ShippingPlanBatchInput):
    try:
        data = [plan.model_dump() for plan in batch.plans]

        result_df = prediction_service.predict_shipping(data)

        summary = {
            "period": f"{result_df['eta_date'].min()} to {result_df['eta_date'].max()}",
            "total_days": len(result_df['eta_date'].unique()),
            "total_volume_ton": float(result_df['planned_volume_ton'].sum()),
            "total_vessels": len(result_df),
            "total_demurrage_cost": float(result_df['predicted_demurrage_cost'].sum()),
            "avg_loading_efficiency": float(result_df['loading_efficiency'].mean()),
            "high_risk_days": int((result_df['risk_level'] == 'HIGH').sum()),
            "daily_summary": result_df.groupby('eta_date').agg({
                'planned_volume_ton': 'sum',
                'predicted_loading_hours': 'mean',
                'loading_efficiency': 'mean',
                'risk_level': lambda x: x.mode()[0] if len(x.mode()) > 0 else "UNKNOWN"
            }).reset_index().to_dict('records'),
            "route_recommendations": {},
            "ai_summary": None
        }

        return ShippingSummaryOutput(**summary)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Shipping summary generation failed: {str(e)}"
        )

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
