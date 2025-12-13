import io
import os
import pandas as pd
from datetime import datetime
from typing import List

from fastapi import APIRouter, APIRouter, FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.services.prediction import PredictionService
from app.services.llm_service import LLMService
llm_service = LLMService()

from app.rag.vectorstore import VectorStore
from app.rag.rag_engine import RAGEngineSafe
from app.services.rag_services import RAGService
from pydantic import BaseModel
from app.services.weather import WeatherService
weather_service = WeatherService()

from app.services.route_service import RouteService
route_service = RouteService()

from app.services.jetty_locations import JETTY_LOCATIONS
import logging

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

from app.rag.rag_engine import RAGEngineSafe

# ==================== SAFE RAG ENGINE ====================

class SafeRAGEngine:
    def __init__(self, docs_path="app/rag/documents"):
        try:
            self.engine = RAGEngineSafe(docs_path)
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
        df = prediction_service.predict_shipping(data)

        df["eta_date"] = pd.to_datetime(df["eta_date"])

        required_cols = [
            "rom_id", "rom_lat", "rom_lon","jetty_id", "eta_date","planned_volume_ton",
            "predicted_loading_hours","predicted_demurrage_cost","loading_efficiency","risk_level"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required column '{col}'"
                )
        
        rom_lat = float(df.iloc[0]["rom_lat"])
        rom_lon = float(df.iloc[0]["rom_lon"])
        rom_id = df.iloc[0]["rom_id"]
        selected_jetty = df.iloc[0]["jetty_id"]
        
        if selected_jetty not in JETTY_LOCATIONS:
            raise HTTPException(400, f"Jetty '{selected_jetty}' not found")

        sel_coord = JETTY_LOCATIONS[selected_jetty]
        sel_dist, sel_dur = route_service.compute_route(
            rom_lat, rom_lon,
            sel_coord["lat"], sel_coord["lon"]
        )

        nearest_id, nearest_dist, nearest_dur = None, float("inf"), None
        for jid, coord in JETTY_LOCATIONS.items():
            d, dur = route_service.compute_route(
                rom_lat, rom_lon,
                coord["lat"], coord["lon"]
            )
            if d is not None and d < nearest_dist:
                nearest_id, nearest_dist, nearest_dur = jid, d, dur
     
        df["precipitation_mm"] = 0.0  
        df["wind_speed_kmh"] = 0.0    
        df["temp_day"] = 25.0         
        df["cloud_cover_pct"] = 50.0  

        weather_rows = []
        for d in df["eta_date"].dt.date.unique():
            w = WeatherService.fetch_weather(
                lat=sel_coord["lat"],
                lon=sel_coord["lon"],
                target_date=d
            )
            weather_rows.append(w)

        avg_rain = float(pd.DataFrame(weather_rows)["precipitation_mm"].mean())
        max_wind = float(pd.DataFrame(weather_rows)["wind_speed_kmh"].max())
                
        # ===================== HAULING SUMMARY =====================
        hauling_summary = {
            "rom_id": rom_id,
            "selected_jetty": selected_jetty,
            "distance_km": sel_dist,
            "duration_min": sel_dur,
            "recommended_nearest_jetty": {
                "jetty_id": nearest_id,
                "distance_km": nearest_dist,
                "duration_min": nearest_dur
            },
            "avg_rain_mm": avg_rain,
            "max_wind_kmh": max_wind
        }

        # ===================== SHIPPING SUMMARY =====================
        shipping_summary = {
            "total_volume_ton": float(df["planned_volume_ton"].sum()),
            "total_vessels": int(len(df)),
            "avg_loading_efficiency": float(df["loading_efficiency"].mean()),
            "total_demurrage_cost": float(df["predicted_demurrage_cost"].sum()),
            "high_risk_vessels": int((df["risk_level"] == "HIGH").sum()),
            "daily_summary": (
                df.groupby(df["eta_date"].dt.date)
                .agg({
                    "planned_volume_ton": "sum",
                    "predicted_loading_hours": "mean",
                    "predicted_demurrage_cost": "sum",
                    "risk_level": lambda x: x.mode()[0] if len(x.mode()) else "UNKNOWN"
                })
                .reset_index()
                .rename(columns={"eta_date": "date"})
                .to_dict("records")
            )
        }

        integration_insight = (
            "Hauling distance and jetty selection directly impact vessel "
            "waiting time and demurrage exposure, especially during high wind "
            "and rainfall conditions at the jetty."
        )

        ai_summary = llm_service.summarize_shipping({
            "hauling": hauling_summary,
            "shipping": shipping_summary
        })

        return {
            "role": "Shipping Planner",
            "focus": "ROM-to-Jetty Hauling & Vessel Loading Coordination",
            "period": f"{df['eta_date'].min().date()} to {df['eta_date'].max().date()}",
            "total_days": int(df["eta_date"].nunique()),
            "hauling_summary": hauling_summary,
            "shipping_summary": shipping_summary,
            "route_recommendations": hauling_summary,
            "integration_insight": integration_insight,
            "ai_summary": ai_summary
        }

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

# ==================== CHATBOT ====================

chatbot_service = ChatbotService()

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chatbot"]
)
def chat_endpoint(req: ChatRequest):
    """
    Endpoint utama chatbot OptiMine
    """
    try:
        result = chatbot_service.handle_chat(
            session_id=req.session_id,
            message=req.message,
            context_type=req.context_type,
            context_payload=req.context_payload,
            role=req.role
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources"),
            timestamp=result["timestamp"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
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

# ==================== RUN SERVER ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)