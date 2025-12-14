import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
from app.services.llm import call_groq
from app.services.llm_service import LLMService
llm_service = LLMService()

try:
    from app.services.optimization_ai import (
        generate_strengths_ai,
        generate_limitations_ai,
        generate_steps_ai
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    
    def generate_strengths_ai(plan_data, domain="mining", config=None):
        return ["Kekuatan rencana tidak tersedia (AI module missing)"]
    
    def generate_limitations_ai(plan_data, domain="mining", config=None):
        return ["Keterbatasan rencana tidak tersedia (AI module missing)"]
        
    def generate_steps_ai(plan_data, domain="mining", config=None):
        return ["Langkah implementasi tidak tersedia (AI module missing)"]


def _generate_ai_justification(plan_context: Dict, domain: str, config: Dict) -> str:
    if not AI_AVAILABLE:
        return "Justifikasi tidak tersedia (AI module missing)."
        
    prompt = f"""
    Bertindaklah sebagai Senior Operations Manager di bidang {domain}.
    
    Analisis rencana optimasi berikut dan berikan JUSTIFIKASI NARATIF (2 paragraf) 
    mengapa rencana ini direkomendasikan. Gunakan Bahasa Indonesia yang profesional.
    
    DATA RENCANA:
    - Nama Strategi: {plan_context.get('strategy')}
    - Deskripsi: {plan_context.get('description')}
    - Impact Finansial: {json.dumps(plan_context.get('financial', {}), indent=2)}
    
    Fokus pada trade-off antara risiko dan keuntungan.
    """
    try:
        return call_groq(prompt, config).strip()
    except Exception:
        return "Gagal menghasilkan justifikasi AI."

class OptimizationService:
    def __init__(self):
        # Placeholder strategies (logic dihandle di fungsi generate_top3)
        pass

# ==================== MINING OPTIMIZATION  ====================

def generate_top3_mining_plans(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Generating Top 3 Mining Plans...")
    
    try:
        df = predictions.copy()
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_datetime(df['plan_date'])
        else:
            df['plan_date'] = pd.date_range(start=datetime.now().date(), periods=len(df), freq='D')
        
        executive_summary = _make_executive_summary_mining(df)
        
        strategies = [
            {"id": 1, "name": "Conservative Plan", "prod_multiplier": 0.90, "risk_threshold": 0.7, "desc": "Minimalkan risiko operasional"},
            {"id": 2, "name": "Balanced Plan", "prod_multiplier": 1.00, "risk_threshold": 0.6, "desc": "Seimbangkan target & risiko"},
            {"id": 3, "name": "Aggressive Plan", "prod_multiplier": 1.10, "risk_threshold": 0.5, "desc": "Maksimalkan volume produksi"}
        ]
        
        recommendations = []
        
        for s in strategies:
            optimized_schedule = []
            total_baseline_cost = 0.0
            total_optimized_cost = 0.0
            total_risk_score = 0.0
            
            for idx, row in df.iterrows():
                conf_items = []
                for col in ['efficiency_factor', 'confidence_score']:
                    if col in row and not pd.isna(row[col]): 
                        try: conf_items.append(float(row[col]))
                        except: pass
                overall_conf = float(np.mean(conf_items)) if conf_items else 0.6
                
                adj_multiplier = s['prod_multiplier']
                if overall_conf < s['risk_threshold']: adj_multiplier *= 0.8
                
                original_prod = float(row.get('planned_production_ton', 0.0))
                adjusted_prod = round(original_prod * adj_multiplier, 0)
                baseline_cost = original_prod * 30.0
                optimized_cost = adjusted_prod * 30.0
                
                total_baseline_cost += baseline_cost
                total_optimized_cost += optimized_cost
                total_risk_score += (1.0 - overall_conf)
                
                weather_info = []
                if 'precipitation_mm' in row and not pd.isna(row['precipitation_mm']):
                    weather_info.append(f"Hujan: {row['precipitation_mm']}mm")
                weather_condition = " | ".join(weather_info) if weather_info else "Normal"
                
                rationale = f"{s['name']}: {'Adjustment Risk' if overall_conf < s['risk_threshold'] else 'Normal Ops'}"

                optimized_schedule.append({
                    "date": str(row['plan_date'].date()),
                    "plan_id": row.get('plan_id', f"MP{idx+1:03d}"),
                    "original_production_ton": int(original_prod),
                    "optimized_production_ton": int(adjusted_prod),
                    "adjustment_pct": round(((adjusted_prod - original_prod) / original_prod * 100) if original_prod > 0 else 0, 2),
                    "baseline_cost_usd": round(baseline_cost, 2),
                    "optimized_cost_usd": round(optimized_cost, 2),
                    "confidence_score": round(overall_conf, 2),
                    "weather_condition": weather_condition,
                    "rationale": rationale
                })
            
            financial_impact = {
                "baseline_total_cost_usd": round(total_baseline_cost, 2),
                "optimized_total_cost_usd": round(total_optimized_cost, 2),
                "cost_savings_usd": round(total_baseline_cost - total_optimized_cost, 2),
                "avg_risk_score": round(total_risk_score / len(df), 2) if len(df) > 0 else 0.0
            }
            
            plan_context = {
                "strategy": s['name'],
                "description": s['desc'],
                "financial": financial_impact,
                "schedule_sample": optimized_schedule[:2]
            }
            
            print(f"Generating AI content for {s['name']}...")
            strengths = generate_strengths_ai(plan_context, "mining", config)
            limitations = generate_limitations_ai(plan_context, "mining", config)
            steps = generate_steps_ai(plan_context, "mining", config)
            justification = _generate_ai_justification(plan_context, "mining", config)
            
            plan_obj = {
                "plan_id": s['id'],
                "plan_name": s['name'],
                "strategy_description": s['desc'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": financial_impact,
                "implementation_steps": steps,
                "strengths": strengths,
                "limitations": limitations,
                "justification": justification
            }
            recommendations.append(plan_obj)
            
        return {
            "plan_type": "RENCANA OPTIMASI PERTAMBANGAN",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"Error top3 mining: {e}")
        return {"error": str(e)}

# ==================== SHIPPING OPTIMIZATION  ====================

def generate_top3_shipping_plans(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Generating Top 3 Shipping Plans...")
    try:
        df = predictions.copy()
        df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        # Aggregasi harian
        daily_df = df.groupby('eta_date').agg({
            'planned_volume_ton': 'sum',
            'loading_efficiency': 'mean',
            'predicted_demurrage_cost': 'sum',
            'wind_speed_kmh': 'mean'
        }).reset_index()
        
        executive_summary = _make_executive_summary_shipping(df)
        
        strategies = [
            {"id": 1, "name": "Conservative Plan", "multiplier": 0.85, "desc": "Prioritas: Hindari Demurrage"},
            {"id": 2, "name": "Balanced Plan", "multiplier": 1.00, "desc": "Prioritas: Keseimbangan Revenue & Risiko"},
            {"id": 3, "name": "Aggressive Plan", "multiplier": 1.10, "desc": "Prioritas: Max Revenue Throughput"}
        ]
        
        recommendations = []
        
        for s in strategies:
            optimized_schedule = []
            baseline_rev = 0
            opt_rev = 0
            savings = 0
            
            for i, (_, row) in enumerate(daily_df.iterrows(), 1):
                original_ton = row['planned_volume_ton']
                
                # === USE EFFICIENCY SCORE ===
                efficiency_val = float(row.get('loading_efficiency', 0.6))
                efficiency_score = round(max(0.1, min(1.0, efficiency_val)), 2)
                
                multiplier = s['multiplier']
                rationale = "Operasi normal"
                
                if "Conservative" in s['name']:
                    if efficiency_score < 0.7: 
                        multiplier = 0.85
                        rationale = f"Low Eff ({efficiency_score}): Reduced Volume"
                    else:
                        multiplier = 0.95
                        rationale = "Standard Conservative"
                
                elif "Aggressive" in s['name']:
                    if efficiency_score < 0.5:
                        multiplier = 0.95 
                        rationale = "Critical Eff: Adjusted"
                    else:
                        multiplier = 1.10
                        rationale = "Max Throughput Mode"
                
                else:  
                    if efficiency_score < 0.6:
                        multiplier = 0.90
                        rationale = f"Balanced Adj ({efficiency_score})"
                
                adjusted_ton = original_ton * multiplier
                
                b_rev = original_ton * 65
                o_rev = adjusted_ton * 65
                
                raw_demurrage = float(row['predicted_demurrage_cost'])
                dem_base = max(0.0, raw_demurrage) # Handle negative cost
                dem_save = dem_base * (1 - multiplier) if multiplier < 1 else 0
                
                baseline_rev += b_rev
                opt_rev += o_rev
                savings += dem_save
                
                optimized_schedule.append({
                    "date": row['eta_date'].strftime('%Y-%m-%d'),
                    "day": i,
                    "plan_id": f"PDS{i:04d}",
                    "original_shipping_ton": int(original_ton),
                    "optimized_shipping_ton": int(adjusted_ton),
                    "adjustment_pct": round(((adjusted_ton - original_ton) / original_ton * 100) if original_ton else 0, 2),
                    "baseline_revenue_usd": round(b_rev, 2),
                    "optimized_revenue_usd": round(o_rev, 2),
                    "demurrage_cost_usd": round(raw_demurrage, 2),
                    "efficiency_score": efficiency_score,
                    "confidence_score": efficiency_score, 
                    "weather_condition": f"Angin: {round(row['wind_speed_kmh'], 1)}km/j",
                    "rationale": rationale
                })
            
            financial_impact = {
                "baseline_total_revenue_usd": round(baseline_rev, 2),
                "optimized_total_revenue_usd": round(opt_rev, 2),
                "revenue_change_usd": round(opt_rev - baseline_rev, 2),
                "demurrage_savings_usd": round(savings, 2),
                "avg_risk_score": round(1 - daily_df['loading_efficiency'].mean(), 2)
            }
            
            plan_context = {
                "strategy": s['name'],
                "description": s['desc'],
                "financial": financial_impact,
                "schedule_sample": optimized_schedule[:2]
            }
            
            print(f"Generating AI content for {s['name']}...")
            strengths = generate_strengths_ai(plan_context, "shipping", config)
            limitations = generate_limitations_ai(plan_context, "shipping", config)
            steps = generate_steps_ai(plan_context, "shipping", config)
            justification = _generate_ai_justification(plan_context, "shipping", config)
            
            plan_obj = {
                "plan_id": s['id'],
                "plan_name": s['name'],
                "strategy_description": s['desc'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": financial_impact,
                "implementation_steps": steps,
                "strengths": strengths,
                "limitations": limitations,
                "justification": justification
            }
            recommendations.append(plan_obj)
            
        return {
            "plan_type": "RENCANA OPTIMASI PENGIRIMAN",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"Error top3 shipping: {e}")
        return {"error": str(e)}

# ==================== REGENERATE MINING OPTIMIZATION ====================

def generate_custom_mining_plan(predictions: pd.DataFrame, params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = predictions.copy()
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_datetime(df['plan_date'])
                
        s = {
            "id": 99, 
            "name": params.get("strategy_name", "Custom Plan"),
            "prod_multiplier": params.get("prod_multiplier", 1.0),
            "risk_threshold": params.get("risk_threshold", 0.6),
            "description": params.get("description", "User defined strategy")
        }

        optimized_schedule = []
        total_baseline_cost = 0.0
        total_optimized_cost = 0.0
        total_risk_score = 0.0

        for idx, row in df.iterrows():
            conf_items = []
            for col in ['efficiency_factor', 'confidence_score']:
                if col in row and not pd.isna(row[col]):
                    try: conf_items.append(float(row[col]))
                    except: pass
            overall_conf = float(np.mean(conf_items)) if conf_items else 0.6
            
            adj_multiplier = s['prod_multiplier']
            if overall_conf < s['risk_threshold']:
                adj_multiplier *= 0.8  
            
            original_prod = float(row.get('planned_production_ton', 0.0))
            adjusted_prod = round(original_prod * adj_multiplier, 0)

            baseline_cost = original_prod * 30.0
            optimized_cost = adjusted_prod * 30.0
            total_baseline_cost += baseline_cost
            total_optimized_cost += optimized_cost
            total_risk_score += (1.0 - overall_conf)

            weather_info = []
            if 'precipitation_mm' in row: weather_info.append(f"Hujan: {row['precipitation_mm']}mm")
            weather_condition = " | ".join(weather_info) if weather_info else "Normal"

            optimized_schedule.append({
                "date": str(row['plan_date'].date()),
                "plan_id": row.get('plan_id', f"CUST{idx}"),
                "original_production_ton": int(original_prod),
                "optimized_production_ton": int(adjusted_prod),
                "adjustment_pct": round(((adjusted_prod - original_prod) / original_prod * 100) if original_prod > 0 else 0, 2),
                "baseline_cost_usd": round(baseline_cost, 2),
                "optimized_cost_usd": round(optimized_cost, 2),
                "confidence_score": round(overall_conf, 2),
                "weather_condition": weather_condition,
                "rationale": f"Custom Adjustment ({adj_multiplier:.2f}x)"
            })

        financial_impact = {
            "baseline_total_cost_usd": round(total_baseline_cost, 2),
            "optimized_total_cost_usd": round(total_optimized_cost, 2),
            "cost_savings_usd": round(total_baseline_cost - total_optimized_cost, 2),
            "avg_risk_score": round(total_risk_score / len(df), 2) if len(df) > 0 else 0.0
        }

        plan_context = {
            "strategy": s['name'],
            "description": s['description'],
            "financial": financial_impact,
            "schedule_sample": optimized_schedule[:2] 
        }

        if AI_AVAILABLE:
            print("Generating Custom Mining Plan AI analysis...")
            strengths = generate_strengths_ai(plan_context, "mining", config)
            limitations = generate_limitations_ai(plan_context, "mining", config)
            steps = generate_steps_ai(plan_context, "mining", config)
            justification = _generate_ai_justification(plan_context, "mining", config)
        else:
            strengths = ["Fitur AI tidak tersedia"]
            limitations = ["Fitur AI tidak tersedia"]
            steps = ["Fitur AI tidak tersedia"]
            justification = "Rencana dibuat berdasarkan parameter input user."

        plan_obj = {
            "plan_id": 99,
            "plan_name": s['name'],
            "strategy_description": s['description'],
            "optimized_schedule": optimized_schedule,
            "financial_impact": financial_impact,
            "implementation_steps": steps,
            "strengths": strengths,
            "limitations": limitations,
            "justification": justification
        }

        return {
            "plan_type": "RENCANA OPTIMASI KUSTOM (PERTAMBANGAN)",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "period": "Custom Period",
                "total_days": len(optimized_schedule)
            },
            "recommendations": [plan_obj] 
        }

    except Exception as e:
        print(f"Error custom mining: {e}")
        return {"error": str(e)}

# ==================== REGENERATE SHIPPING OPTIMIZATION ====================

def generate_custom_shipping_plan(predictions: pd.DataFrame, params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = predictions.copy()
        df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        daily_df = df.groupby('eta_date').agg({
            'planned_volume_ton': 'sum',
            'loading_efficiency': 'mean',
            'predicted_demurrage_cost': 'sum',
            'wind_speed_kmh': 'mean',
            'confidence_score': 'mean'
        }).reset_index()
        
        s = {
            "id": 99,
            "name": params.get("strategy_name", "Custom Shipping"),
            "description": params.get("description", ""),
            "ship_multiplier": params.get("ship_multiplier", 1.0),
            "risk_threshold": params.get("risk_threshold", 0.6)
        }

        optimized_schedule = []
        baseline_total_revenue = 0
        optimized_total_revenue = 0
        demurrage_savings = 0

        for i, (_, row) in enumerate(daily_df.iterrows(), 1):
            original_ton = row['planned_volume_ton']            
            
            efficiency_val = float(row.get('loading_efficiency', 0.6))
            efficiency_score = round(max(0.1, min(1.0, efficiency_val)), 2)            
            
            multiplier = s['ship_multiplier']
            
            if efficiency_score < s['risk_threshold']:
                multiplier *= 0.9 
                rationale = f"Safety Cut: Eff {efficiency_score} < Threshold {s['risk_threshold']}"
            else:
                rationale = f"On Target ({multiplier}x)"

            adjusted_ton = original_ton * multiplier
            
            baseline_rev = original_ton * 65
            optimized_rev = adjusted_ton * 65
            
            raw_demurrage = float(row.get('predicted_demurrage_cost', 0))
            demurrage_base = max(0.0, raw_demurrage)
            demurrage_save = demurrage_base * (1 - multiplier) if multiplier < 1.0 else 0

            baseline_total_revenue += baseline_rev
            optimized_total_revenue += optimized_rev
            demurrage_savings += demurrage_save

            optimized_schedule.append({
                "date": row['eta_date'].strftime('%Y-%m-%d'),
                "day": i,
                "plan_id": f"CUST{i:04d}",
                "original_shipping_ton": int(original_ton),
                "optimized_shipping_ton": int(adjusted_ton),
                "adjustment_pct": round(((adjusted_ton - original_ton) / original_ton * 100) if original_ton else 0, 2),
                "baseline_revenue_usd": round(baseline_rev, 2),
                "optimized_revenue_usd": round(optimized_rev, 2),
                "demurrage_cost_usd": round(raw_demurrage, 2),
                "efficiency_score": efficiency_score,
                "confidence_score": efficiency_score,
                "rationale": rationale
            })

        financial_impact = {
            "baseline_total_revenue_usd": round(baseline_total_revenue, 2),
            "optimized_total_revenue_usd": round(optimized_total_revenue, 2),
            "revenue_change_usd": round(optimized_total_revenue - baseline_total_revenue, 2),
            "demurrage_savings_usd": round(demurrage_savings, 2),
            "avg_risk_score": round(1 - (daily_df['loading_efficiency'].mean()), 2)
        }

        plan_context = {
            "strategy": s['name'],
            "description": s['description'],
            "financial": financial_impact,
            "schedule_sample": optimized_schedule[:2]
        }

        if AI_AVAILABLE:
            print("Generating Custom Shipping Plan AI analysis...")
            strengths = generate_strengths_ai(plan_context, "shipping", config)
            limitations = generate_limitations_ai(plan_context, "shipping", config)
            steps = generate_steps_ai(plan_context, "shipping", config)
            justification = _generate_ai_justification(plan_context, "shipping", config)
        else:
            strengths = ["Custom plan"]
            limitations = ["Manual check required"]
            steps = ["Execute custom plan"]
            justification = "Generated based on user parameters."

        plan_obj = {
            "plan_id": 99,
            "plan_name": f"Custom: {s['name']}",
            "strategy_description": s['description'],
            "optimized_schedule": optimized_schedule,
            "financial_impact": financial_impact,
            "implementation_steps": steps,
            "strengths": strengths,
            "limitations": limitations,
            "justification": justification
        }    

        return {
            "plan_type": "RENCANA OPTIMASI KUSTOM (PENGIRIMAN)",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {},
            "recommendations": [plan_obj]
        }

    except Exception as e:
        return {"error": str(e)}

# ==================== HELPER FUNCTIONS ====================

def _make_executive_summary_mining(df):
    return {
        "period": f"{df['plan_date'].min()} - {df['plan_date'].max()}",
        "total_days": len(df['plan_date'].unique()),
        "total_ton": float(df['planned_production_ton'].sum()) if 'planned_production_ton' in df else 0
    }

def _make_executive_summary_shipping(df):
    return {
        "period": f"{df['eta_date'].min()} - {df['eta_date'].max()}",
        "total_days": len(df['eta_date'].unique()),
        "total_ton": float(df['planned_volume_ton'].sum())
    }
