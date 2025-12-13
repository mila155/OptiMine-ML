import io
import os
import pandas as pd

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List, Optional, Dict, Any
from app.rag import rag_engine
from app.services import llm
from app.services import rag_services
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
from app.services.jetty_locations import JETTY_COORDINATES

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

DOCS_PATH = "app/rag/documents"

rag_engine = RAGEngineSafe(DOCS_PATH)

rag_service = RAGService(DOCS_PATH)

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

# ==================== CHATBOT ====================

@app.post("/chat", tags=["Chatbot"])
async def chat_with_rag(query: str):
    try:
        docs = rag_services.retrieve_context(query, top_k=5)

        prompt = rag_services.build_prompt(
            user_query=query,
            docs=docs
        )

        answer = llm.ask(prompt)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {"id": d["id"], "score": d["score"]}
                for d in docs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# ==================== MINING ENDPOINTS ====================

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
        weather_data = []

        for _, row in result_df.iterrows():
            w = weather_service.fetch_weather(
                lat=row["latitude"],
                lon=row["longitude"],
                target_date=row["plan_date"]
            )
            weather_data.append(w)
            
        weather_df = pd.DataFrame(weather_data)
        for col in ["temp_day", "wind_speed_kmh", "precipitation_mm", "cloud_cover_pct"]:
            weather_df[col] = weather_df[col].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )
            
        weather_cols = ["temp_day", "wind_speed_kmh", "precipitation_mm", "cloud_cover_pct"]

        for col in weather_cols:
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])
                
        result_df = pd.concat([result_df.reset_index(drop=True), weather_df], axis=1)

        total_planned = result_df['planned_production_ton'].sum()
        total_predicted = result_df['predicted_production_ton'].sum()
        production_gap_pct = (
            (total_planned - total_predicted) / total_planned * 100
            if total_planned > 0 else 0
        )
        
        summary = {
            "role": "Mining Planner",
            "focus": "Pit-to-ROM Operations & Production Optimization",
            "period": f"{result_df['plan_date'].min()} to {result_df['plan_date'].max()}",
            "total_days": len(result_df['plan_date'].unique()),
            "total_planned_production_ton":  float(total_planned),
            "total_predicted_production_ton": float(total_predicted),
            "production_gap_pct": float(production_gap_pct),
            "avg_efficiency": float(result_df['efficiency_factor'].mean()),
            "avg_hauling_distance": float(result_df["hauling_distance_km"].mean()),
            "avg_rain": float(result_df["precipitation_mm"].mean()),
            "max_wind": float(result_df["wind_speed_kmh"].max()),
            "high_risk_days": int((result_df['risk_level'] == 'HIGH').sum()),           
        }

        daily_rows = []
        grouped = result_df.groupby("plan_date")

        for i, (date, g) in enumerate(grouped, start=1):
            planned = g['planned_production_ton'].sum()
            predicted = g['predicted_production_ton'].sum()
            gap_pct = (planned - predicted) / planned * 100 if planned > 0 else 0

            daily_rows.append({
                "day": i,
                "date": date,
                "active_pits": ", ".join(sorted(g["pit_id"].unique())),
                "planned_production_ton": float(planned),
                "predicted_production_ton": float(predicted),
                "gap_pct": float(gap_pct),
                "efficiency_factor": float(g["efficiency_factor"].mean()),
                "rain_mm": float(g["precipitation_mm"].mean()),
                "wind_kmh": float(g["wind_speed_kmh"].max()),
                "risk_level": g["risk_level"].mode()[0] if len(g["risk_level"].mode()) else "UNKNOWN"
            })

        summary["daily_summary"] = daily_rows        
        summary["ai_summary"] = llm_service.summarize_mining(summary)
        
        return MiningSummaryOutput(**summary)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary generation failed: {str(e)}"
        )

# ==================== SHIPPING ENDPOINTS ===================

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
        
        required_cols = ["rom_id", "rom_lat", "rom_lon", "jetty_id", "eta_date"]
        for col in required_cols:
            if col not in result_df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required column '{col}' from backend web JSON"
                )

        result_df["eta_date"] = pd.to_datetime(result_df["eta_date"])
        
        rom_lat = float(result_df.iloc[0]["rom_lat"])
        rom_lon = float(result_df.iloc[0]["rom_lon"])
        rom_id = result_df.iloc[0]["rom_id"]

        selected_jetty_id = result_df.iloc[0]["jetty_id"]
        if selected_jetty_id not in JETTY_LOCATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Jetty '{selected_jetty_id}' not found in jetty_locations.py"
            )

        # ================= USER SELECTED JETTY =================
        sel_coord = JETTY_LOCATIONS[selected_jetty_id]
        jetty_lat = sel_coord["lat"]
        jetty_lon = sel_coord["lon"]
        
        sel_dist, sel_dur = route_service.compute_route(
            rom_lat, rom_lon,
            jetty_lat, jetty_lon
        )
        
        nearest_id, nearest_dist, nearest_dur = None, float("inf"), None        
        for jid, coord in JETTY_LOCATIONS.items():
            d, dur = route_service.compute_route(
                rom_lat, rom_lon,
                coord["lat"], coord["lon"]
            )
            if d is not None and d < nearest_dist:
                nearest_id, nearest_dist, nearest_dur = jid, d, dur

        result_df["rain_mm"] = 0.0
        result_df["wind_kmh"] = 0.0

        for d in result_df["eta_date"].dt.date.unique():
            w = WeatherService.fetch_weather(
                lat=jetty_lat,
                lon=jetty_lon,
                target_date=d
            )

            mask = result_df["eta_date"].dt.date == d
            result_df.loc[mask, "rain_mm"] = w["precipitation_mm"]
            result_df.loc[mask, "wind_kmh"] = w["wind_speed_kmh"]

        daily_summary = (
            result_df
            .groupby(result_df["eta_date"].dt.date)
            .agg({
                "planned_volume_ton": "sum",
                "predicted_loading_hours": "mean",
                "loading_efficiency": "mean",
                "predicted_demurrage_cost": "sum",
                "rain_mm": "mean",
                "wind_kmh": "max",
                "risk level": lambda x: x.mode() [0] if len(x.mode()) else "UNKNOWN"
            })
            .reset_index()
            .rename(columns={"eta_date": "date"})
            .to_dict("records")
        )

        summary = {
            "role": "Shipping Planner",
            "focus": "ROM-to-Jetty Hauling & Vessel Loading",
            "period": f"{result_df['eta_date'].min()} to {result_df['eta_date'].max()}",
            "total_days": int(result_df["eta_date"].nunique()),
            "total_volume_ton": float(result_df["planned_volume_ton"].sum()),
            "total_vessels": int(len(result_df)),
            "total_demurrage_cost": float(result_df["predicted_demurrage_cost"].sum()),
            "avg_loading_efficiency": float(result_df["loading_efficiency"].mean()),
            "high_risk_days": int((result_df["risk_level"] == "HIGH").sum()),
            "avg_rain": float(result_df["rain_mm"].mean()),
            "max_wind": float(result_df["wind_kmh"].max()),
            "daily_summary": daily_summary
        }
        
        route_recommendations = {
            "rom_id": rom_id,
            "user_selected_jetty": {
                "jetty_id": selected_jetty_id,
                "distance_km": sel_dist,
                "duration_min": sel_dur
            },
            "recommended_nearest_jetty": {
                "jetty_id": nearest_id,
                "distance_km": nearest_dist,
                "duration_min": nearest_dur
            }
        }

        ai_summary = llm_service.summarize_shipping(summary)

        return ShippingSummaryOutput(**summary)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Shipping summary generation failed: {str(e)}"
        )

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
