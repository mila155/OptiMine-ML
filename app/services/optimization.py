"""
Optimization service
Handles schedule optimization and recommendations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

# --- PATCH: import AI helper generators (may not exist on startup) ---
# If app.services.optimization_ai is present, import; otherwise fallback-safe helpers are used.
try:
    from app.services.optimization_ai import (
        generate_strengths_ai,
        generate_limitations_ai,
        generate_steps_ai
    )
    _OPT_AI_AVAILABLE = True
except Exception:
    # Fallback simple implementations (safe, deterministic) so service never crashes.
    _OPT_AI_AVAILABLE = False

    def generate_strengths_ai(plan_data: Dict[str, Any], domain: str = "mining", config: Dict[str, Any] = None) -> List[str]:
        # Simple heuristic-based strengths generator (fallback)
        strengths = []
        fin = plan_data.get("financial", {}) if isinstance(plan_data, dict) else plan_data.get('financial_impact', {})
        conf_scores = [s.get('confidence_score', 0.5) for s in plan_data.get("schedule", [])] if plan_data.get("schedule") else []
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.6
        if avg_conf > 0.85:
            strengths.append("Tingkat keyakinan tinggi pada eksekusi rencana")
        else:
            strengths.append("Rencana mempertimbangkan mitigasi risiko")
        if fin and fin.get('cost_savings_usd', 0) > 0:
            strengths.append("Menunjukkan potensi penghematan biaya")
        strengths.append("Struktur rencana mudah dipahami")
        return strengths

    def generate_limitations_ai(plan_data: Dict[str, Any], domain: str = "mining", config: Dict[str, Any] = None) -> List[str]:
        # Simple heuristic-based limitations generator (fallback)
        limitations = []
        fin = plan_data.get("financial", {}) if isinstance(plan_data, dict) else plan_data.get('financial_impact', {})
        if fin and fin.get('avg_risk_score', 0) > 0.6:
            limitations.append("Skor risiko rata-rata relatif tinggi untuk beberapa hari")
        else:
            limitations.append("Perlu verifikasi lapangan dan data cuaca real-time")
        limitations.append("Model masih menggunakan asumsi biaya dasar (dummy model)")
        return limitations

    def generate_steps_ai(plan_data: Dict[str, Any], domain: str = "mining", config: Dict[str, Any] = None) -> List[str]:
        # Simple fallback steps
        steps = []
        steps.append(f"Implement {plan_data.get('strategy', plan_data.get('plan_name','strategy'))} starting Day 1")
        steps.append("Monitor kondisi operasional dan cuaca setiap hari")
        steps.append("Sesuaikan alokasi sumber daya bila risiko meningkat")
        return steps
# --- /PATCH -----------------------------------------------------------


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
                'achievement_rate': float(total_predicted / total_planned * 100) if total_planned > 0 else 0.0,
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

# Try to import RAG engine & LLM call function (fallback-safe)
try:
    from app.rag.rag_engine import RAG_ENGINE  # may be None if not initialized
except Exception:
    RAG_ENGINE = None

# call_groq should be your existing LLM wrapper (kept as-is)
try:
    from app.services.llm import call_groq 
except Exception:
    # define a fallback to avoid crash — returns simple template
    def call_groq(prompt, config):
        return "AI not available to generate detailed justification."

def _safe_get_context(query: str, k: int = 6) -> str:
    """Return RAG context if available (safe)."""
    try:
        if RAG_ENGINE:
            ctx = RAG_ENGINE.get_context(query, k=k)
            return ctx or ""
    except Exception:
        pass
    return ""

def _make_executive_summary_mining(df: pd.DataFrame) -> Dict[str, Any]:
    """Build executive summary for mining plan"""
    return {
        "period": f"{df['plan_date'].min()} hingga {df['plan_date'].max()}",
        "total_days": int(len(df['plan_date'].unique())),
        "total_planned_production_ton": float(df['planned_production_ton'].sum()),
        "avg_efficiency": round(float(df['avg_efficiency'].mean()) if 'avg_efficiency' in df.columns else float(df['efficiency_factor'].mean()), 2),
        "avg_delay_min": round(float(df['avg_delay_min'].mean()) if 'avg_delay_min' in df.columns else float(df['cycle_delay_min'].mean()), 2),
        "high_risk_days": int((df['risk_level'] == 'HIGH').sum())
    }

def _make_executive_summary_shipping(df: pd.DataFrame) -> Dict[str, Any]:
    """Build executive summary for shipping plan"""
    return {
        "period": f"{df['date'].min()} hingga {df['date'].max()}" if 'date' in df.columns else f"{df['eta_date'].min()} hingga {df['eta_date'].max()}",
        "total_days": int(len(df['eta_date'].unique())) if 'eta_date' in df.columns else int(len(df['date'].unique())),
        "total_planned_shipment_ton": float(df['planned_volume_ton'].sum()) if 'planned_volume_ton' in df.columns else float(df['total_volume_ton'].sum()),
        "avg_loading_efficiency": round(float(df['avg_efficiency'].mean()) if 'avg_efficiency' in df.columns else float(df['loading_efficiency'].mean()), 2),
        "total_demurrage_cost_usd": round(float(df['predicted_demurrage_cost'].sum()) if 'predicted_demurrage_cost' in df.columns else float(df['total_demurrage_cost_usd'].sum()), 2),
        "high_risk_days": int((df['risk_level'] == 'HIGH').sum())
    }

# -------------------------
# AI description helpers (use RAG context inside prompt)
# -------------------------
def generate_ai_description_mining(plan_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate Indonesian description for mining using AI + RAG"""
    print(f"🔍 Generating AI description for: {plan_data.get('plan_name')}")
    
    rag_context = _safe_get_context(plan_data.get('plan_name', ''), k=6)
    prompt = f"""
Gunakan konteks berikut sebagai referensi utama:
{rag_context}

Kamu adalah asisten AI yang membantu menjelaskan rencana optimasi pertambangan dalam Bahasa Indonesia.

Berdasarkan data berikut:
- Nama Plan: {plan_data.get('plan_name')}
- Deskripsi Strategi: {plan_data.get('strategy_description')}
- Baseline Cost: ${plan_data.get('financial_impact', {}).get('baseline_total_cost_usd', 0):,.0f}
- Optimized Cost: ${plan_data.get('financial_impact', {}).get('optimized_total_cost_usd', 0):,.0f}
- Cost Savings: ${plan_data.get('financial_impact', {}).get('cost_savings_usd', 0):,.0f}
- Risk Score: {plan_data.get('financial_impact', {}).get('avg_risk_score', 0):.2f}
- Kelebihan: {', '.join(plan_data.get('strengths', []))}
- Keterbatasan: {', '.join([lim.get('keterbatasan', lim) if isinstance(lim, dict) else lim for lim in plan_data.get('limitations', [])])}

Buatlah penjelasan detail dalam Bahasa Indonesia yang natural dan mudah dipahami (3-4 paragraf) yang mencakup:
1. Penjelasan singkat tentang rencana pertambangan ini dan tujuannya
2. Bagaimana rencana ini mempengaruhi operasi pertambangan harian
3. Analisis dampak finansial terhadap biaya dan potensi penghematan
4. Mengapa rencana ini direkomendasikan untuk operasi tambang

Gunakan bahasa yang profesional namun mudah dipahami. Jangan gunakan format markdown atau bullet points.
"""
    try:
        print(f"📝 Prompt length: {len(prompt)}")
        response = call_groq(prompt, config).strip()
        print(f"✅ AI Response received: {response[:100]}...")
        
        if "gagal" in response.lower() or "error" in response.lower():
            print(f"⚠️ AI returned error: {response}")
            return _generate_manual_justification(plan_data)
            
        return response
    except Exception as e:
        print(f"❌ Error in generate_ai_description_mining: {e}")
        return _generate_manual_justification(plan_data)

def _generate_manual_justification(plan_data: Dict[str, Any]) -> str:
    """Manual fallback justification"""
    plan_name = plan_data.get('plan_name', '')
    savings = plan_data.get('financial_impact', {}).get('cost_savings_usd', 0)
    
    if "Conservative" in plan_name:
        return f"Rencana konservatif berfokus pada mitigasi risiko dengan mengurangi target produksi untuk mengantisipasi kondisi buruk. {'Menghasilkan penghematan biaya.' if savings > 0 else 'Mempertahankan stabilitas operasional.'}"
    elif "Aggressive" in plan_name:
        return f"Rencana agresif meningkatkan target produksi untuk memanfaatkan kondisi optimal dengan menerima risiko lebih tinggi. {'Potensi peningkatan revenue.' if savings < 0 else 'Fokus pada pencapaian target maksimal.'}"
    else:
        return f"Rencana seimbang menjaga target produksi sambil mengoptimalkan alokasi sumber daya dan manajemen risiko."
    """Generate Indonesian description for mining using AI + RAG (safe fallback)."""
    rag_context = _safe_get_context(plan_data.get('plan_name', ''), k=6)
    prompt = f"""
Gunakan konteks berikut sebagai referensi utama:
{rag_context}

Kamu adalah asisten AI yang membantu menjelaskan rencana optimasi pertambangan dalam Bahasa Indonesia.

Berdasarkan data berikut:
- Nama Plan: {plan_data.get('plan_name')}
- Deskripsi Strategi: {plan_data.get('strategy_description')}
- Baseline Cost: ${plan_data.get('financial_impact', {}).get('baseline_total_cost_usd', 0):,.0f}
- Optimized Cost: ${plan_data.get('financial_impact', {}).get('optimized_total_cost_usd', 0):,.0f}
- Cost Savings: ${plan_data.get('financial_impact', {}).get('cost_savings_usd', 0):,.0f}
- Risk Score: {plan_data.get('financial_impact', {}).get('avg_risk_score', 0):.2f}
- Kelebihan: {', '.join(plan_data.get('strengths', []))}
- Keterbatasan: {', '.join(plan_data.get('limitations', []))}

Buatlah penjelasan detail dalam Bahasa Indonesia yang natural dan mudah dipahami (3-4 paragraf) yang mencakup:
1. Penjelasan singkat tentang rencana pertambangan ini dan tujuannya
2. Bagaimana rencana ini mempengaruhi operasi pertambangan harian
3. Analisis dampak finansial terhadap biaya dan potensi penghematan
4. Mengapa rencana ini direkomendasikan untuk operasi tambang

Gunakan bahasa yang profesional namun mudah dipahami. Jangan gunakan format markdown atau bullet points.
"""
    try:
        return call_groq(prompt, config).strip()
    except Exception as e:
        return f"Penjelasan otomatis gagal: {str(e)}"

def generate_ai_description_shipping(plan_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate Indonesian description for shipping using AI + RAG (safe fallback)."""
    rag_context = _safe_get_context(plan_data.get('plan_name', ''), k=6)
    prompt = f"""
Gunakan konteks berikut sebagai referensi utama:
{rag_context}

Kamu adalah asisten AI yang membantu menjelaskan rencana optimasi pengiriman dalam Bahasa Indonesia.

Berdasarkan data berikut:
- Nama Plan: {plan_data.get('plan_name')}
- Deskripsi Strategi: {plan_data.get('strategy_description')}
- Baseline Revenue: ${plan_data.get('financial_impact', {}).get('baseline_total_revenue_usd', 0):,.0f}
- Optimized Revenue: ${plan_data.get('financial_impact', {}).get('optimized_total_revenue_usd', 0):,.0f}
- Revenue Change: ${plan_data.get('financial_impact', {}).get('revenue_change_usd', 0):,.0f}
- Demurrage Savings: ${plan_data.get('financial_impact', {}).get('demurrage_savings_usd', 0):,.0f}
- Risk Score: {plan_data.get('financial_impact', {}).get('avg_risk_score', 0):.2f}
- Kelebihan: {', '.join(plan_data.get('strengths', []))}
- Keterbatasan: {', '.join(plan_data.get('limitations', []))}

Buatlah penjelasan detail dalam Bahasa Indonesia yang natural dan mudah dipahami (3-4 paragraf) yang mencakup:
1. Penjelasan singkat tentang rencana pengiriman ini dan tujuannya
2. Bagaimana rencana ini mempengaruhi operasi pengiriman harian
3. Analisis dampak finansial terhadap revenue dan penghematan demurrage
4. Mengapa rencana ini direkomendasikan untuk operasi pelabuhan

Gunakan bahasa yang profesional namun mudah dipahami. Jangan gunakan format markdown atau bullet points.
"""
    try:
        return call_groq(prompt, config).strip()
    except Exception as e:
        return f"Penjelasan otomatis gagal: {str(e)}"

# -------------------------
# Top3 Mining generator - PATCHED
# -------------------------
def generate_top3_mining_plans(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    df = predictions.copy()
    df['plan_date'] = pd.to_datetime(df['plan_date']) if 'plan_date' in df.columns else pd.date_range(start=datetime.now().date(), periods=len(df))
    
    executive_summary = _make_executive_summary_mining(df)

    strategies = [
        {"id": 1, "name": "Conservative Plan", "prod_multiplier": 0.90, "risk_threshold": 0.7, "description": "Meminimalkan risiko operasional akibat cuaca dan keterlambatan"},
        {"id": 2, "name": "Balanced Plan", "prod_multiplier": 1.00, "risk_threshold": 0.6, "description": "Menyeimbangkan target produksi dengan pengelolaan risiko yang efektif"},
        {"id": 3, "name": "Aggressive Plan", "prod_multiplier": 1.10, "risk_threshold": 0.5, "description": "Memaksimalkan volume produksi untuk memenuhi permintaan tinggi"}
    ]

    recommendations = []

    for s in strategies:
        optimized_schedule = []
        total_baseline_cost = 0.0
        total_optimized_cost = 0.0
        total_risk_score = 0.0

        for idx, row in df.iterrows():
            conf_items = [v for v in [row.get('efficiency_factor'), row.get('confidence_score')] if v is not None]
            overall_conf = float(np.mean(conf_items)) if conf_items else 0.6

            adj_multiplier = s['prod_multiplier'] * (0.8 if overall_conf < s['risk_threshold'] else 1.0)
            rationale = f"{s['name']}: Reduced due to risk" if overall_conf < s['risk_threshold'] else f"{s['name']}: Operating normally"

            original_prod = float(row.get('planned_production_ton', 0.0))
            adjusted_prod = round(original_prod * adj_multiplier, 0)

            baseline_cost = original_prod * 30.0
            optimized_cost = adjusted_prod * 30.0
            total_baseline_cost += baseline_cost
            total_optimized_cost += optimized_cost
            total_risk_score += (1.0 - overall_conf)

            optimized_schedule.append({
                "date": str(row['plan_date'].date()),
                "plan_id": row.get('plan_id'),
                "original_production_ton": int(original_prod),
                "optimized_production_ton": int(adjusted_prod),
                "adjustment_pct": round(((adjusted_prod - original_prod) / original_prod * 100) if original_prod > 0 else 0, 2),
                "baseline_cost_usd": round(float(baseline_cost), 2),
                "optimized_cost_usd": round(float(optimized_cost), 2),
                "confidence_score": round(overall_conf, 2),
                "weather_condition": f"Hujan (avg): {row.get('avg_rain_mm', 0)}mm, Angin (max): {row.get('max_wind_kmh', 0)}km/h",
                "rationale": rationale
            })

        financial_impact = {
            "baseline_total_cost_usd": round(total_baseline_cost, 2),
            "optimized_total_cost_usd": round(total_optimized_cost, 2),
            "cost_savings_usd": round(total_baseline_cost - total_optimized_cost, 2),
            "avg_risk_score": round(total_risk_score / len(df), 2) if len(df) > 0 else 0.0
        }

        plan_obj = {
            "plan_id": s['id'],
            "plan_name": s['name'],
            "strategy_description": s['description'],
            "optimized_schedule": optimized_schedule,
            "financial_impact": financial_impact,
            "implementation_steps": [
                "Implement " + s['name'] + " starting Day 1",
                "Monitor weather updates daily",
                "Adjust operations based on real-time conditions"
            ],
            "strengths": [],
            "limitations": []
        }

        # --- PATCH: populate strengths & limitations safely ---
        plan_data_for_ai = {"strategy": plan_obj['plan_name'], "schedule": plan_obj['optimized_schedule'], "financial": plan_obj['financial_impact']}
        try:
            plan_obj["strengths"] = generate_strengths_ai(plan_data_for_ai, domain="mining", config=config)
        except Exception:
            plan_obj["strengths"] = ["Strength example fallback"]

        try:
            plan_obj["limitations"] = generate_limitations_ai(plan_data_for_ai, domain="mining", config=config)
        except Exception:
            plan_obj["limitations"] = ["Limitation example fallback"]

        # --- PATCH: generate justification after strengths & limitations ---
        try:
            plan_obj['justification'] = generate_ai_description_mining(plan_obj, config)
        except Exception:
            plan_obj['justification'] = "AI justification fallback"

        recommendations.append(plan_obj)

    return {
        "plan_type": "RENCANA OPTIMASI PERTAMBANGAN",
        "generated_at": datetime.now().isoformat(),
        "executive_summary": executive_summary,
        "recommendations": recommendations
    }


# -------------------------
# Top3 Shipping generator - PATCHED
# -------------------------
def generate_top3_shipping_plans(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    df = predictions.copy()
    df['eta_date'] = pd.to_datetime(df['eta_date']) if 'eta_date' in df.columns else pd.date_range(start=datetime.now().date(), periods=len(df))

    executive_summary = _make_executive_summary_shipping(df)

    strategies = [
        {"id": 1, "name": "Conservative Plan", "ship_multiplier": 0.95, "risk_threshold": 0.7, "description": "Minimalkan biaya demurrage dan keterlambatan akibat cuaca"},
        {"id": 2, "name": "Balanced Plan", "ship_multiplier": 1.00, "risk_threshold": 0.6, "description": "Optimalkan revenue sambil mengelola risiko demurrage dan cuaca"},
        {"id": 3, "name": "Aggressive Plan", "ship_multiplier": 1.10, "risk_threshold": 0.5, "description": "Maksimalkan revenue pengiriman dan utilisasi kapal"}
    ]

    recommendations = []

    for s in strategies:
        optimized_schedule = []
        total_baseline_revenue = 0.0
        total_optimized_revenue = 0.0
        total_demurrage_saved = 0.0
        total_risk_score = 0.0

        for idx, row in df.iterrows():
            conf_items = [v for v in [row.get('loading_efficiency'), row.get('confidence_score')] if v is not None]
            overall_conf = float(np.mean(conf_items)) if conf_items else 0.6

            adj_multiplier = s['ship_multiplier'] * (0.85 if overall_conf < s['risk_threshold'] else 1.0)
            rationale = f"{s['name']}: Reduced due to risk" if overall_conf < s['risk_threshold'] else f"{s['name']}: Operating normally"

            original_ship = float(row.get('planned_volume_ton', row.get('total_volume_ton', 0)))
            adjusted_ship = round(original_ship * adj_multiplier, 0)

            baseline_revenue = original_ship * 65.0
            optimized_revenue = adjusted_ship * 65.0
            demurrage_cost = float(row.get('predicted_demurrage_cost', row.get('total_demurrage_cost_usd', 0)))
            demurrage_saved = demurrage_cost * (1 - (adj_multiplier if adj_multiplier < 1 else 1))

            total_baseline_revenue += baseline_revenue
            total_optimized_revenue += optimized_revenue
            total_demurrage_saved += demurrage_saved
            total_risk_score += (1.0 - overall_conf)

            optimized_schedule.append({
                "date": str(row['eta_date'].date()),
                "shipment_id": row.get('shipment_id'),
                "original_shipping_ton": int(original_ship),
                "optimized_shipping_ton": int(adjusted_ship),
                "adjustment_pct": round(((adjusted_ship - original_ship) / original_ship * 100) if original_ship > 0 else 0, 2),
                "baseline_revenue_usd": round(float(baseline_revenue), 2),
                "optimized_revenue_usd": round(float(optimized_revenue), 2),
                "demurrage_cost_usd": round(float(demurrage_cost), 2),
                "confidence_score": round(overall_conf, 2),
                "weather_condition": f"Angin (max): {row.get('max_wind_kmh', 0)}km/h",
                "rationale": rationale
            })

        financial_impact = {
            "baseline_total_revenue_usd": round(total_baseline_revenue, 2),
            "optimized_total_revenue_usd": round(total_optimized_revenue, 2),
            "revenue_change_usd": round(total_optimized_revenue - total_baseline_revenue, 2),
            "demurrage_savings_usd": round(total_demurrage_saved, 2),
            "avg_risk_score": round(total_risk_score / len(df), 2) if len(df) > 0 else 0.0
        }

        plan_obj = {
            "plan_id": s['id'],
            "plan_name": s['name'],
            "strategy_description": s['description'],
            "optimized_schedule": optimized_schedule,
            "financial_impact": financial_impact,
            "implementation_steps": [
                "Implement " + s['name'] + " starting Day 1",
                "Monitor weather updates daily",
                "Adjust berth allocation as needed"
            ],
            "strengths": [],
            "limitations": []
        }

        # --- PATCH: populate strengths & limitations safely ---
        plan_data_for_ai = {"strategy": plan_obj['plan_name'], "schedule": plan_obj['optimized_schedule'], "financial": plan_obj['financial_impact']}
        try:
            plan_obj["strengths"] = generate_strengths_ai(plan_data_for_ai, domain="shipping", config=config)
        except Exception:
            plan_obj["strengths"] = ["Strength example fallback"]

        try:
            plan_obj["limitations"] = generate_limitations_ai(plan_data_for_ai, domain="shipping", config=config)
        except Exception:
            plan_obj["limitations"] = ["Limitation example fallback"]

        # --- PATCH: generate justification after strengths & limitations ---
        try:
            plan_obj['justification'] = generate_ai_description_shipping(plan_obj, config)
        except Exception:
            plan_obj['justification'] = "AI justification fallback"

        recommendations.append(plan_obj)

    return {
        "plan_type": "RENCANA OPTIMASI PENGIRIMAN",
        "generated_at": datetime.now().isoformat(),
        "executive_summary": executive_summary,
        "recommendations": recommendations
    }
