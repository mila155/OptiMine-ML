import pandas as pd
import numpy as np
import random
import json
import time  
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from app.services.llm import call_groq
from app.services.llm_service import LLMService
llm_service = LLMService()

try:
    from app.services.optimization_ai import generate_full_analysis
    _OPT_AI_AVAILABLE  = True
except ImportError:
    _OPT_AI_AVAILABLE  = False
    def generate_full_analysis(*args):
        return {
            "strengths": ["Fitur AI tidak tersedia"],
            "limitations": ["Fitur AI tidak tersedia"],
            "implementation_steps": ["Fitur AI tidak tersedia"],
            "justification": "Modul AI tidak ditemukan."
        }

# ==================== HELPER FUNCTIONS ====================

def _generate_ai_rationale(row_data: Dict, strategy_name: str, domain: str, config: Dict) -> str:
    """Rule-based rationale (Cepat & Hemat Token)"""
    adj = row_data.get('adjustment_pct', 0)
    name = strategy_name.lower()
    
    if "conservative" in name or "saver" in name:
        if adj < 0: return f"Volume dikurangi {abs(adj)}% (Safety Mode)"
        return "Target moderat (Safety Focus)"
    elif "aggressive" in name or "max" in name or "boost" in name:
        if adj > 0: return f"Volume ditingkatkan {adj}% (Optimal Mode)"
        return "Target kapasitas maksimal"
    else:
        if adj != 0: return f"Penyesuaian taktis {adj}%"
        return "Sesuai rencana baseline"

def _make_executive_summary_mining(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "period": f"{df['plan_date'].min()} hingga {df['plan_date'].max()}",
        "total_days": int(len(df['plan_date'].unique())),
        "total_planned_production_ton": float(df['planned_production_ton'].sum()) if 'planned_production_ton' in df.columns else 0.0,
        "avg_efficiency": round(float(df['efficiency_factor'].mean()) if 'efficiency_factor' in df.columns else 0.6, 2),
        "avg_delay_min": round(float(df['cycle_delay_min'].mean()) if 'cycle_delay_min' in df.columns else 0.0, 2),
        "high_risk_days": int((df['risk_level'] == 'HIGH').sum() if 'risk_level' in df.columns else 0)
    }

def _make_executive_summary_shipping(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "period": f"{df['eta_date'].min().strftime('%Y-%m-%d %H:%M:%S')} hingga {df['eta_date'].max().strftime('%Y-%m-%d %H:%M:%S')}",
        "total_days": int(len(df['eta_date'].unique())),
        "total_planned_shipment_ton": round(float(df['planned_volume_ton'].sum()) if 'planned_volume_ton' in df.columns else float(df['total_volume_ton'].sum()), 2),
        "avg_loading_efficiency": round(float(df['loading_efficiency'].mean()) if 'loading_efficiency' in df.columns else float(df['loading_efficiency'].mean()), 2),
        "total_demurrage_cost_usd": round(float(df['predicted_demurrage_cost'].sum()) if 'predicted_demurrage_cost' in df.columns else float(df['total_demurrage_cost_usd'].sum()), 2),
        "high_risk_days": int((df['loading_efficiency'] < 0.5).sum()) 
    }

# ==================== MINING OPTIMIZATION ====================

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
            totals = {"base_cost": 0, "opt_cost": 0, "risk": 0}
            
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
                
                totals["base_cost"] += baseline_cost
                totals["opt_cost"] += optimized_cost
                totals["risk"] += (1.0 - overall_conf)
                
                weather_info = []
                if 'precipitation_mm' in row and not pd.isna(row['precipitation_mm']):
                    weather_info.append(f"Hujan: {row['precipitation_mm']}mm")
                weather_cond = " | ".join(weather_info) if weather_info else "Normal"
                
                optimized_schedule.append({
                    "date": str(row['plan_date'].date()),
                    "plan_id": f"MP{idx+1:03d}",
                    "original_production_ton": int(original_prod),
                    "optimized_production_ton": int(adjusted_prod),
                    "adjustment_pct": round(((adjusted_prod - original_prod) / original_prod * 100) if original_prod > 0 else 0, 2),
                    "baseline_cost_usd": round(baseline_cost, 2),
                    "optimized_cost_usd": round(optimized_cost, 2),
                    "confidence_score": round(overall_conf, 2),
                    "weather_condition": weather_cond,
                    "rationale": _generate_ai_rationale({'adjustment_pct': (adjusted_prod-original_prod)/original_prod*100 if original_prod else 0}, s['name'], "mining", config)
                })
            
            financial_impact = {
                "baseline_total_cost_usd": round(totals["base_cost"], 2),
                "optimized_total_cost_usd": round(totals["opt_cost"], 2),
                "cost_savings_usd": round(totals["base_cost"] - totals["opt_cost"], 2),
                "avg_risk_score": round(totals["risk"] / len(df), 2) if len(df) > 0 else 0
            }
            
            print(f"Generating AI for {s['name']}...")
            plan_context = {
                "strategy": s['name'], "description": s['desc'], "financial": financial_impact,
            }
            
            ai_analysis = generate_full_analysis(plan_context, "mining", config)
            
            time.sleep(1) 
            
            plan_obj = {
                "plan_id": s['id'],
                "plan_name": s['name'],
                "strategy_description": s['desc'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": financial_impact,
                "implementation_steps": ai_analysis["implementation_steps"],
                "strengths": ai_analysis["strengths"],
                "limitations": ai_analysis["limitations"],
                "justification": ai_analysis["justification"]
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

# ==================== SHIPPING OPTIMIZATION ====================

def generate_top3_shipping_plans(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Generating Top 3 Shipping Plans...")
    try:
        df = predictions.copy()
        df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        daily_df = df.groupby('eta_date').agg({
            'planned_volume_ton': 'sum', 'loading_efficiency': 'mean',
            'predicted_demurrage_cost': 'sum', 'wind_speed_kmh': 'mean'
        }).reset_index()
        
        executive_summary = _make_executive_summary_shipping(df)
        
        strategies = [
            {"id": 1, "name": "Conservative Plan", "multiplier": 0.85, "desc": "Prioritas: Hindari Demurrage"},
            {"id": 2, "name": "Balanced Plan", "multiplier": 1.00, "desc": "Prioritas: Keseimbangan Revenue"},
            {"id": 3, "name": "Aggressive Plan", "multiplier": 1.10, "desc": "Prioritas: Max Revenue"}
        ]
        
        recommendations = []
        for s in strategies:
            optimized_schedule = []
            totals = {"base_rev": 0, "opt_rev": 0, "saving": 0}
            
            for i, (_, row) in enumerate(daily_df.iterrows(), 1):
                orig = row['planned_volume_ton']
                eff = round(max(0.1, min(1.0, float(row['loading_efficiency']))), 2)
                
                mult = s['multiplier']
                if "Conservative" in s['name'] and eff < 0.7: mult = 0.85
                elif "Aggressive" in s['name'] and eff < 0.5: mult = 0.95
                
                adj = orig * mult
                b_rev = orig * 65
                o_rev = adj * 65
                dem_base = max(0.0, float(row['predicted_demurrage_cost']))
                dem_save = dem_base * (1 - mult) if mult < 1 else 0
                
                totals["base_rev"] += b_rev
                totals["opt_rev"] += o_rev
                totals["saving"] += dem_save
                
                rationale = _generate_ai_rationale(
                    {'adjustment_pct': (adj-orig)/orig*100 if orig else 0}, 
                    s['name'], "shipping", config
                )
                
                optimized_schedule.append({
                    "date": row['eta_date'].strftime('%Y-%m-%d'),
                    "day": i,
                    "plan_id": f"PDS{i:04d}",
                    "original_shipping_ton": int(orig),
                    "optimized_shipping_ton": int(adj),
                    "adjustment_pct": round(((adj - orig) / orig * 100) if orig else 0, 2),
                    "baseline_revenue_usd": round(b_rev, 2),
                    "optimized_revenue_usd": round(o_rev, 2),
                    "demurrage_cost_usd": round(dem_base, 2),
                    "efficiency_score": eff,
                    "confidence_score": eff,
                    "weather_condition": f"Angin: {round(row['wind_speed_kmh'], 1)}km/j",
                    "rationale": rationale
                })
            
            fin_impact = {
                "baseline_total_revenue_usd": round(totals["base_rev"], 2),
                "optimized_total_revenue_usd": round(totals["opt_rev"], 2),
                "revenue_change_usd": round(totals["opt_rev"] - totals["base_rev"], 2),
                "demurrage_savings_usd": round(totals["saving"], 2),
                "avg_risk_score": round(1 - daily_df['loading_efficiency'].mean(), 2)
            }
            
            print(f"Generating AI for {s['name']}...")
            ctx = {"strategy": s['name'], "description": s['desc'], "financial": fin_impact}
            
            ai_analysis = generate_full_analysis(ctx, "shipping", config)
            time.sleep(1) 
            
            plan_obj = {
                "plan_id": s['id'],
                "plan_name": s['name'],
                "strategy_description": s['desc'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": fin_impact,
                "implementation_steps": ai_analysis["implementation_steps"],
                "strengths": ai_analysis["strengths"],
                "limitations": ai_analysis["limitations"],
                "justification": ai_analysis["justification"]
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

# ==================== RENEGERATE MINING ====================

def generate_custom_mining_plan(predictions: pd.DataFrame, params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = predictions.copy()
        if 'plan_date' in df.columns: df['plan_date'] = pd.to_datetime(df['plan_date'])
        
        variants = [
            {"id": 91, "name": "Alt A: Efficiency Focus", "desc": "Fokus efisiensi alat.", "range": (0.92, 0.96), "thresh": 0.65},
            {"id": 92, "name": "Alt B: Weather Resilient", "desc": "Stabil saat cuaca buruk.", "range": (0.97, 1.03), "thresh": 0.70},
            {"id": 93, "name": "Alt C: Production Boost", "desc": "Peningkatan throughput.", "range": (1.04, 1.08), "thresh": 0.55}
        ]
        
        recommendations = []
        for v in variants:
            mult = round(random.uniform(*v["range"]), 2)
            s = {"id": v["id"], "name": v["name"], "description": v["desc"], "prod_multiplier": mult, "risk_threshold": v["thresh"]}

            optimized_schedule = []
            totals = {"base_cost": 0, "opt_cost": 0, "risk": 0}
            
            for idx, row in df.iterrows():
                conf = float(np.mean([float(row.get(c, 0.6)) for c in ['efficiency_factor', 'confidence_score'] if c in row]))
                adj_mult = mult
                if conf < v['thresh']: adj_mult *= 0.9
                
                orig = float(row.get('planned_production_ton', 0))
                adj = round(orig * adj_mult, 0)
                
                totals["base_cost"] += orig * 30.0
                totals["opt_cost"] += adj * 30.0
                totals["risk"] += (1.0 - conf)
                
                weather_info = []
                if 'precipitation_mm' in row: weather_info.append(f"Hujan: {row['precipitation_mm']}mm")
                
                optimized_schedule.append({
                    "date": str(row['plan_date'].date()),
                    "plan_id": f"ALT{idx}",
                    "original_production_ton": int(orig),
                    "optimized_production_ton": int(adj),
                    "adjustment_pct": round(((adj-orig)/orig*100) if orig else 0, 2),
                    "baseline_cost_usd": round(orig*30.0, 2),
                    "optimized_cost_usd": round(adj*30.0, 2),
                    "confidence_score": round(conf, 2),
                    "weather_condition": " | ".join(weather_info) if weather_info else "Normal",
                    "rationale": f"{v['name']} ({adj_mult:.2f}x)"
                })

            fin_impact = {
                "baseline_total_cost_usd": round(totals["base_cost"], 2),
                "optimized_total_cost_usd": round(totals["opt_cost"], 2),
                "cost_savings_usd": round(totals["base_cost"] - totals["opt_cost"], 2),
                "avg_risk_score": round(totals["risk"] / len(df), 2) if len(df) > 0 else 0
            }

            print(f"Generating AI for {v['name']}...")
            ctx = {"strategy": v['name'], "description": v['desc'], "financial": fin_impact}
            
            ai_analysis = generate_full_analysis(ctx, "mining", config)
            time.sleep(1)

            recommendations.append({
                "plan_id": v['id'],
                "plan_name": f"{v['name']} ({mult}x)",
                "strategy_description": v['desc'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": fin_impact,
                "implementation_steps": ai_analysis["implementation_steps"],
                "strengths": ai_analysis["strengths"],
                "limitations": ai_analysis["limitations"],
                "justification": ai_analysis["justification"]
            })

        return {
            "plan_type": "RENCANA ALTERNATIF (PERTAMBANGAN)",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": _make_executive_summary_mining(df),
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": str(e)}

# ==================== REGENERATE SHIPPING ====================

def generate_custom_shipping_plan(predictions: pd.DataFrame, params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = predictions.copy()
        df['eta_date'] = pd.to_datetime(df['eta_date'])
        daily_df = df.groupby('eta_date').agg({
            'planned_volume_ton': 'sum', 'loading_efficiency': 'mean',
            'predicted_demurrage_cost': 'sum', 'wind_speed_kmh': 'mean'
        }).reset_index()
        
        variants = [
            {"id": 91, "name": "Alt A: Demurrage Saver", "desc": "Fokus pengurangan denda.", "range": (0.88, 0.93), "thresh": 0.70},
            {"id": 92, "name": "Alt B: Quick Dispatch", "desc": "Percepatan turnover.", "range": (0.96, 1.02), "thresh": 0.60},
            {"id": 93, "name": "Alt C: Max Revenue", "desc": "Optimalisasi muatan.", "range": (1.03, 1.08), "thresh": 0.50}
        ]
        
        recommendations = []
        for v in variants:
            mult = round(random.uniform(*v["range"]), 2)
            s = {"id": v["id"], "name": v["name"], "description": v["desc"], "ship_multiplier": mult, "risk_threshold": v["thresh"]}

            optimized_schedule = []
            totals = {"base_rev": 0, "opt_rev": 0, "saving": 0}
            
            for i, (_, row) in enumerate(daily_df.iterrows(), 1):
                orig = row['planned_volume_ton']
                eff = round(max(0.1, min(1.0, float(row['loading_efficiency']))), 2)
                
                adj_mult = mult
                if eff < v['thresh']: adj_mult *= 0.9
                
                adj = orig * adj_mult
                b_rev = orig * 65
                o_rev = adj * 65
                dem_base = max(0.0, float(row['predicted_demurrage_cost']))
                dem_save = dem_base * (1 - adj_mult) if adj_mult < 1 else 0
                
                totals["base_rev"] += b_rev
                totals["opt_rev"] += o_rev
                totals["saving"] += dem_save
                
                optimized_schedule.append({
                    "date": row['eta_date'].strftime('%Y-%m-%d'),
                    "day": i,
                    "plan_id": f"ALT{i:04d}",
                    "original_shipping_ton": int(orig),
                    "optimized_shipping_ton": int(adj),
                    "adjustment_pct": round(((adj-orig)/orig*100) if orig else 0, 2),
                    "baseline_revenue_usd": round(b_rev, 2),
                    "optimized_revenue_usd": round(o_rev, 2),
                    "demurrage_cost_usd": round(dem_base, 2),
                    "efficiency_score": eff,
                    "confidence_score": eff,
                    "rationale": f"Variant ({adj_mult:.2f}x)"
                })

            fin_impact = {
                "baseline_total_revenue_usd": round(totals["base_rev"], 2),
                "optimized_total_revenue_usd": round(totals["opt_rev"], 2),
                "revenue_change_usd": round(totals["opt_rev"] - totals["base_rev"], 2),
                "demurrage_savings_usd": round(totals["saving"], 2),
                "avg_risk_score": round(1 - daily_df['loading_efficiency'].mean(), 2)
            }
            
            print(f"Generating AI for {v['name']}...")
            ctx = {"strategy": v['name'], "description": v['desc'], "financial": fin_impact, "schedule_sample": optimized_schedule[:2]}
            
            ai_analysis = generate_full_analysis(ctx, "shipping", config)
            time.sleep(1)

            recommendations.append({
                "plan_id": v['id'],
                "plan_name": f"{v['name']} ({mult}x)",
                "strategy_description": v['desc'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": fin_impact,
                "implementation_steps": ai_analysis["implementation_steps"],
                "strengths": ai_analysis["strengths"],
                "limitations": ai_analysis["limitations"],
                "justification": ai_analysis["justification"]
            })

        return {
            "plan_type": "RENCANA ALTERNATIF (PENGIRIMAN)",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": _make_executive_summary_shipping(df),
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": str(e)}
