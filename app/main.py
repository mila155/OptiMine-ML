import io
import os
import pandas as pd
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.services.prediction import PredictionService
from app.schemas import (
    PredictionRequest, RAGQuery,
    MiningPlanInput, MiningPlanBatchInput, MiningPredictionOutput, MiningSummaryOutput,
    ShippingPlanInput, ShippingPlanBatchInput, ShippingPredictionOutput, ShippingSummaryOutput,
    HealthCheck
)

from app.services.optimization import (
    generate_top3_mining_plans,
    generate_top3_shipping_plans
)

from app.rag.rag_engine import RAGEngine

# ==================== SAFE RAG ENGINE ====================

class SafeRAGEngine:
    def __init__(self, docs_path="app/rag/documents"):
        try:
            self.engine = RAGEngine(docs_path)
            self.ready = True
            print("✅ RAG Engine initialized successfully.")
        except Exception as e:
            print("⚠️ Failed to initialize RAG Engine:", e)
            self.engine = None
            self.ready = False

    def get_context(self, query: str, k: int = 5):
        if self.engine and self.ready:
            return self.engine.get_context(query, k=k)
        else:
            return "No document context available."

DOCS_PATH = "app/rag/documents"
rag_engine = SafeRAGEngine(DOCS_PATH)

# ==================== FASTAPI SETUP ====================

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

# ==================== ROUTES ==========================

@app.get("/")
async def home():
    return {"message": "API is running successfully!"}

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    models_status = prediction_service.is_healthy()
    return HealthCheck(
        status="healthy" if all(models_status.values()) and rag_engine.ready else "degraded",
        timestamp=datetime.now(),
        models_loaded=models_status,
        rag_status="healthy" if rag_engine.ready else "degraded",
        api_version="1.0.0"
    )

# ==================== PREDICTION ========================

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        prediction = prediction_service.predict(payload.features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RAG QUERY =========================

@app.post("/rag/query")
def rag_query(payload: RAGQuery):
    try:
        context = rag_engine.get_context(payload.query)
        return {
            "query": payload.query,
            "context": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MINING ENDPOINTS ===================

@app.post("/mining/upload-csv", tags=["Mining"])
async def upload_csv_for_mining(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        records = df.to_dict(orient="records")
        result_df = prediction_service.predict_mining(records)
        return result_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")

@app.post("/mining/predict", response_model=List[MiningPredictionOutput], tags=["Mining"])
async def predict_mining_single(plan: MiningPlanInput):
    try:
        data = [plan.model_dump()]
        result_df = prediction_service.predict_mining(data)
        response = [
            MiningPredictionOutput(**{
                **row.to_dict(),
                'planned_production_ton': float(row['planned_production_ton']),
                'predicted_production_ton': float(row['predicted_production_ton']),
                'production_gap_ton': float(row['production_gap_ton']),
                'production_gap_pct': float(row['production_gap_pct']),
                'efficiency_factor': float(row['efficiency_factor']),
                'cycle_delay_min': float(row['cycle_delay_min']),
                'ai_priority_score': int(row['ai_priority_score']),
                'confidence_score': float(row['confidence_score'])
            }) for _, row in result_df.iterrows()
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/mining/predict/batch", response_model=List[MiningPredictionOutput], tags=["Mining"])
async def predict_mining_batch(batch: MiningPlanBatchInput):
    try:
        data = [plan.model_dump() for plan in batch.plans]
        result_df = prediction_service.predict_mining(data)
        response = [
            MiningPredictionOutput(**{
                **row.to_dict(),
                'planned_production_ton': float(row['planned_production_ton']),
                'predicted_production_ton': float(row['predicted_production_ton']),
                'production_gap_ton': float(row['production_gap_ton']),
                'production_gap_pct': float(row['production_gap_pct']),
                'efficiency_factor': float(row['efficiency_factor']),
                'cycle_delay_min': float(row['cycle_delay_min']),
                'ai_priority_score': int(row['ai_priority_score']),
                'confidence_score': float(row['confidence_score'])
            }) for _, row in result_df.iterrows()
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

# ==================== SHIPPING ENDPOINTS ===================

@app.post("/shipping/predict", response_model=List[ShippingPredictionOutput], tags=["Shipping"])
async def predict_shipping_single(plan: ShippingPlanInput):
    try:
        data = [plan.model_dump()]
        result_df = prediction_service.predict_shipping(data)
        response = [
            ShippingPredictionOutput(**{
                **row.to_dict(),
                'planned_volume_ton': float(row['planned_volume_ton']),
                'predicted_loading_hours': float(row['predicted_loading_hours']),
                'loading_efficiency': float(row['loading_efficiency']),
                'predicted_demurrage_cost': float(row['predicted_demurrage_cost']),
                'confidence_score': float(row['confidence_score'])
            }) for _, row in result_df.iterrows()
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/shipping/predict/batch", response_model=List[ShippingPredictionOutput], tags=["Shipping"])
async def predict_shipping_batch(batch: ShippingPlanBatchInput):
    try:
        data = [plan.model_dump() for plan in batch.plans]
        result_df = prediction_service.predict_shipping(data)
        response = [
            ShippingPredictionOutput(**{
                **row.to_dict(),
                'planned_volume_ton': float(row['planned_volume_ton']),
                'predicted_loading_hours': float(row['predicted_loading_hours']),
                'loading_efficiency': float(row['loading_efficiency']),
                'predicted_demurrage_cost': float(row['predicted_demurrage_cost']),
                'confidence_score': float(row['confidence_score'])
            }) for _, row in result_df.iterrows()
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Shipping summary generation failed: {str(e)}")

# ==================== OPTIMIZATION ======================

@app.post("/mining/optimize", tags=["Optimization"])
async def optimize_mining(batch: MiningPlanBatchInput):
    try:
        data = [p.model_dump() for p in batch.plans]
        pred_df = prediction_service.predict_mining(data)
        result = generate_top3_mining_plans(pred_df, config={
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.2,
            "max_tokens": 1024,
            "rag_engine": rag_engine
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mining optimization failed: {str(e)}")

@app.post("/shipping/optimize", tags=["Optimization"])
async def optimize_shipping(batch: ShippingPlanBatchInput):
    try:
        data = [p.model_dump() for p in batch.plans]
        pred_df = prediction_service.predict_shipping(data)
        result = generate_top3_shipping_plans(pred_df, config={
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.2,
            "max_tokens": 1024,
            "rag_engine": rag_engine
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shipping optimization failed: {str(e)}")

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

# ==================== RUN SERVER ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)