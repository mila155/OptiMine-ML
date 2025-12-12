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
        """Enhanced fallback strengths generator"""
        strengths = []
        plan_name = plan_data.get("strategy", "")
        schedule = plan_data.get("schedule", [])
        financial = plan_data.get("financial", {})
        
        # Hitung metrics dari data
        total_days = len(schedule) if schedule else 0
        avg_conf = np.mean([s.get('confidence_score', 0.5) for s in schedule]) if schedule else 0.6
        savings = financial.get('cost_savings_usd', 0)
        risk_score = financial.get('avg_risk_score', 0.5)
        
        if "Conservative" in plan_name:
            strengths.append(f"Fokus pada mitigasi risiko dan stabilitas operasional")
            if savings > 0:
                strengths.append(f"Penghematan biaya sebesar ${savings:,.0f}")
            strengths.append(f"Mengurangi ketergantungan pada kondisi cuaca ekstrem")
            strengths.append(f"Rata-rata confidence score {avg_conf:.2f} menunjukkan prediktabilitas baik")
            
        elif "Aggressive" in plan_name:
            strengths.append(f"Maksimalisasi output produksi untuk permintaan tinggi")
            strengths.append(f"Pemanfaatan optimal kapasitas alat berat")
            if savings < 0:  # negative savings berarti peningkatan cost = investasi
                strengths.append(f"Investasi tambahan ${abs(savings):,.0f} untuk pencapaian target tinggi")
            strengths.append(f"Target produksi ditingkatkan 10% pada hari optimal")
            
        else:  # Balanced
            strengths.append(f"Keseimbangan optimal antara target produksi dan manajemen risiko")
            strengths.append(f"Fleksibilitas dalam menyesuaikan operasi harian")
            strengths.append(f"Mempertahankan efisiensi operasional rata-rata {avg_conf:.2f}")
            
        strengths.append(f"Dikalkulasi berdasarkan {total_days} hari operasi")
        return strengths

    def generate_limitations_ai(plan_data: Dict[str, Any], domain: str = "mining", config: Dict[str, Any] = None) -> List[str]:
        """Enhanced fallback limitations generator"""
        limitations = []
        plan_name = plan_data.get("strategy", "")
        schedule = plan_data.get("schedule", [])
        financial = plan_data.get("financial", {})
        
        total_days = len(schedule) if schedule else 0
        risk_score = financial.get('avg_risk_score', 0.5)
        
        if "Conservative" in plan_name:
            limitations.append("Produktivitas lebih rendah karena target yang dikurangi")
            limitations.append("Potensi underutilization sumber daya pada hari optimal")
            limitations.append("Mungkin tidak memenuhi permintaan pasar yang tinggi")
            
        elif "Aggressive" in plan_name:
            limitations.append(f"Risiko operasional lebih tinggi (skor risiko: {risk_score:.2f})")
            limitations.append("Konsumsi bahan bakar dan maintenance cost meningkat")
            limitations.append("Stres pada peralatan dan operator lebih besar")
            
        else:  # Balanced
            limitations.append("Perlu monitoring intensif untuk keseimbangan optimal")
            limitations.append("Mungkin terlalu hati-hati pada kondisi sangat baik")
            limitations.append("Margin keuntungan tidak dimaksimalkan sepenuhnya")
            
        limitations.append(f"Analisis berdasarkan {total_days} hari data historis")
        limitations.append("Menggunakan model biaya dasar ($30 per ton)")
        return limitations

    def generate_steps_ai(plan_data: Dict[str, Any], domain: str = "mining", config: Dict[str, Any] = None) -> List[str]:
        """Enhanced fallback implementation steps"""
        plan_name = plan_data.get("strategy", "")
        
        if "Conservative" in plan_name:
            return [
                "Kurangi target produksi 10-28% pada hari dengan confidence score < 0.7",
                "Prioritaskan maintenance dan safety check sebelum operasi",
                "Monitor kondisi cuaca secara real-time setiap 2 jam",
                "Siapkan backup equipment untuk antisipasi breakdown",
                "Lakukan daily briefing untuk tim tentang prioritas safety"
            ]
            
        elif "Aggressive" in plan_name:
            return [
                "Tingkatkan shift operasional dari 2 menjadi 3 shift per hari",
                "Optimalkan routing hauling untuk minimize delay",
                "Perpanjang jam operasional equipment dengan rotation crew",
                "Monitor equipment health setiap 4 jam untuk preventif maintenance",
                "Siapkan buffer stock spare parts untuk minimal downtime"
            ]
            
        else:  # Balanced
            return [
                "Implementasikan rencana sesuai jadwal produksi baseline",
                "Monitor key performance indicators (KPIs) harian",
                "Lakukan adjustment maksimal ±15% berdasarkan kondisi aktual",
                "Koordinasi dengan maintenance team untuk preventive schedule",
                "Evaluasi hasil harian dan adjust untuk hari berikutnya"
            ]
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
    # Gunakan format tanggal yang konsisten
    df['eta_date'] = pd.to_datetime(df['eta_date'])
    
    return {
        "period": f"{df['eta_date'].min().strftime('%Y-%m-%d %H:%M:%S')} hingga {df['eta_date'].max().strftime('%Y-%m-%d %H:%M:%S')}",
        "total_days": int(len(df['eta_date'].unique())),
        "total_planned_shipment_ton": round(float(df['planned_volume_ton'].sum()) if 'planned_volume_ton' in df.columns else float(df['total_volume_ton'].sum()), 2),
        "avg_loading_efficiency": round(float(df['avg_efficiency'].mean()) if 'avg_efficiency' in df.columns else float(df['loading_efficiency'].mean()), 2),
        "total_demurrage_cost_usd": round(float(df['predicted_demurrage_cost'].sum()) if 'predicted_demurrage_cost' in df.columns else float(df['total_demurrage_cost_usd'].sum()), 2),
        "high_risk_days": int((df['risk_level'] == 'HIGH').sum())
    }

# -------------------------
# Helper functions
# -------------------------
def _get_implementation_steps(plan_name: str) -> List[str]:
    if "Conservative" in plan_name:
        return [
            "Tinjau ulang target produksi harian dengan tim operasional",
            "Prioritaskan maintenance preventif untuk equipment utama",
            "Implementasikan daily safety briefing dengan fokus pada kondisi cuaca",
            "Siapkan contingency plan untuk equipment breakdown",
            "Monitor real-time production vs target setiap 4 jam"
        ]
    elif "Aggressive" in plan_name:
        return [
            "Optimalkan shift pattern menjadi 3 shift operasional",
            "Implementasikan predictive maintenance untuk minimalkan downtime",
            "Tingkatkan fuel dan spare parts inventory sebesar 20%",
            "Koordinasi intensif dengan maintenance team untuk equipment availability",
            "Setup command center untuk monitoring real-time 24/7"
        ]
    else:  # Balanced
        return [
            "Jalankan rencana produksi sesuai baseline schedule",
            "Monitor key performance indicators (KPIs) setiap shift",
            "Lakukan tactical adjustments maksimal ±15% berdasarkan kondisi lapangan",
            "Koordinasi weekly planning meeting dengan semua department",
            "Implementasikan continuous improvement process untuk operational excellence"
        ]

def _get_strengths(plan_name: str, financial_impact: Dict, schedule: List[Dict]) -> List[str]:
    strengths = []
    savings = financial_impact.get('cost_savings_usd', 0)
    risk_score = financial_impact.get('avg_risk_score', 0.5)
    total_days = len(schedule)
    
    if schedule:
        total_original = sum(d.get('original_production_ton', 0) for d in schedule)
        total_optimized = sum(d.get('optimized_production_ton', 0) for d in schedule)
        avg_confidence = np.mean([d.get('confidence_score', 0.5) for d in schedule])
    else:
        total_original = 0
        total_optimized = 0
        avg_confidence = 0.6
    
    if "Conservative" in plan_name:
        strengths.append("Fokus utama pada operational safety dan risk mitigation")
        if savings > 0:
            strengths.append(f"Potensi penghematan biaya: ${abs(savings):,.0f}")
        strengths.append("Mengurangi exposure terhadap market volatility")
        strengths.append(f"Tingkat keyakinan operasional rata-rata: {avg_confidence:.2f}")
        
    elif "Aggressive" in plan_name:
        strengths.append("Maximizes production capacity utilization")
        if total_original > 0:
            strengths.append(f"Peningkatan output: {(total_optimized/total_original*100-100):.1f}%")
        strengths.append("Capitalizes on favorable market conditions")
        strengths.append(f"Confidence level maintained at {avg_confidence:.2f}")
        
    else:  # Balanced
        strengths.append("Optimal balance between production targets and risk management")
        strengths.append("Provides operational flexibility for dynamic adjustments")
        strengths.append("Maintains consistent production flow")
        strengths.append(f"Stable confidence score: {avg_confidence:.2f}")
    
    strengths.append(f"Analysis based on {total_days}-day operational data")
    strengths.append(f"Risk exposure score: {risk_score:.2f} (scale 0-1)")
    return strengths

def _get_limitations(plan_name: str, financial_impact: Dict, total_days: int) -> List[str]:
    limitations = []
    risk_score = financial_impact.get('avg_risk_score', 0.5)
    savings = financial_impact.get('cost_savings_usd', 0)
    
    if "Conservative" in plan_name:
        limitations.append("Potential underutilization of production capacity during optimal conditions")
        limitations.append("May not meet surge demand opportunities")
        limitations.append("Higher unit cost due to reduced economies of scale")
        
    elif "Aggressive" in plan_name:
        limitations.append(f"Higher operational risk exposure ({risk_score:.2f} risk score)")
        limitations.append("Increased maintenance frequency and costs")
        limitations.append("Higher stress on equipment and personnel")
        
    else:  # Balanced
        limitations.append("May be too conservative during highly favorable conditions")
        limitations.append("Opportunity cost of not maximizing production during peaks")
        limitations.append("Requires continuous monitoring and adjustment")
    
    limitations.append(f"Based on {total_days} days of historical data projection")
    limitations.append("Using standard cost model ($30/ton operational cost)")
    limitations.append("Assumes consistent equipment availability and workforce")
    return limitations

def _generate_detailed_justification(plan_name: str, financial_impact: Dict, schedule: List[Dict]) -> str:
    """Generate detailed justification (3-4 paragraf)"""
    total_days = len(schedule)
    savings = financial_impact.get('cost_savings_usd', 0)
    baseline_cost = financial_impact.get('baseline_total_cost_usd', 0)
    optimized_cost = financial_impact.get('optimized_total_cost_usd', 0)
    risk_score = financial_impact.get('avg_risk_score', 0.5)
    
    # Hitung statistik dari schedule
    if schedule:
        total_original = sum(d.get('original_production_ton', 0) for d in schedule)
        total_optimized = sum(d.get('optimized_production_ton', 0) for d in schedule)
        avg_adjustment = np.mean([d.get('adjustment_pct', 0) for d in schedule])
        avg_confidence = np.mean([d.get('confidence_score', 0.5) for d in schedule])
        
        # Hitung hari dengan adjustment positif/negatif
        negative_days = sum(1 for d in schedule if d.get('adjustment_pct', 0) < 0)
        positive_days = sum(1 for d in schedule if d.get('adjustment_pct', 0) > 0)
        neutral_days = total_days - negative_days - positive_days
    else:
        total_original = 0
        total_optimized = 0
        avg_adjustment = 0
        avg_confidence = 0.6
        negative_days = 0
        positive_days = 0
        neutral_days = 0
    
    if "Conservative" in plan_name:
        return f"""Rencana Operasi Konservatif ini dirancang khusus untuk memastikan stabilitas dan keberlanjutan operasi pertambangan selama {total_days} hari ke depan. Strategi ini mengutamakan prinsip kehati-hatian dengan mengurangi target produksi rata-rata sebesar {abs(avg_adjustment):.1f}%, terutama pada hari-hari dengan tingkat kepastian rendah.

Dari total {total_days} hari operasi, sebanyak {negative_days} hari mengalami penurunan target produksi sebagai langkah antisipatif terhadap potensi risiko operasional. Pendekatan ini menghasilkan penghematan biaya operasional sebesar ${abs(savings):,.0f}, dari baseline ${baseline_cost:,.0f} menjadi ${optimized_cost:,.0f}. Meskipun volume produksi turun dari {total_original:,.0f} ton menjadi {total_optimized:,.0f} ton, rencana ini menjamin kontinuitas operasi dengan skor risiko hanya {risk_score:.2f} dan tingkat keyakinan rata-rata {avg_confidence:.2f}.

Rencana ini sangat direkomendasikan untuk periode dengan ketidakpastian tinggi, fluktuasi harga komoditas, atau ketika fokus utama adalah menjaga kesehatan peralatan dan keselamatan pekerja. Implementasi dapat dilakukan segera dengan penjadwalan ulang prioritas pit dan alokasi sumber daya yang lebih efisien."""
    
    elif "Aggressive" in plan_name:
        return f"""Rencana Operasi Agresif ini dirancang untuk memaksimalkan potensi produksi tambang dalam periode {total_days} hari mendatang. Strategi ini mengoptimalkan setiap kesempatan dengan meningkatkan target produksi rata-rata sebesar {avg_adjustment:.1f}%, terutama pada hari-hari dengan kondisi operasional optimal.

Dari {total_days} hari operasi, sebanyak {positive_days} hari mengalami peningkatan target produksi untuk mengejar peluang pasar. Pendekatan ini memerlukan investasi tambahan sebesar ${abs(savings):,.0f}, meningkatkan biaya operasional dari ${baseline_cost:,.0f} menjadi ${optimized_cost:,.0f}. Volume produksi meningkat signifikan dari {total_original:,.0f} ton menjadi {total_optimized:,.0f} ton, dengan menerima risiko operasional {risk_score:.2f} dan tingkat keyakinan rata-rata {avg_confidence:.2f}.

Rencana ini cocok untuk periode permintaan pasar tinggi, ketersediaan peralatan optimal, atau ketika ada target produksi kuartalan/tahunan yang harus dipenuhi. Perlu kesiapan tim maintenance dan monitoring intensif untuk memastikan keberhasilan implementasi."""
    
    else:  # Balanced Plan
        return f"""Rencana Operasi Seimbang ini menawarkan pendekatan moderat selama {total_days} hari operasi mendatang. Strategi ini menjaga target produksi baseline sambil melakukan penyesuaian dinamis berdasarkan kondisi aktual, dengan rata-rata perubahan {avg_adjustment:.1f}%.

Dari {total_days} hari operasi, {neutral_days} hari beroperasi sesuai rencana awal, sementara {positive_days} hari ditingkatkan dan {negative_days} hari dikurangi targetnya. Biaya operasional tetap di ${optimized_cost:,.0f} dengan skor risiko terkendali sebesar {risk_score:.2f}. Volume produksi total {total_optimized:,.0f} ton dihasilkan dengan tingkat keyakinan rata-rata {avg_confidence:.2f}, menunjukkan keseimbangan antara pencapaian target dan manajemen risiko.

Rencana ini ideal untuk operasi rutin dengan fluktuasi kondisi yang dapat diprediksi. Fleksibilitas dalam implementasi memungkinkan penyesuaian cepat berdasarkan perkembangan real-time di lapangan tanpa mengorbankan stabilitas operasional."""

# -------------------------
# Shipping helper functions - FIXED VERSION
# -------------------------
def _generate_shipping_steps(plan_name: str) -> List[str]:
    """Generate implementation steps for shipping plans"""
    steps = {
        "Conservative Plan": [
            "Kurangi target loading pada hari angin kencang (>30km/j)",
            "Jadwalkan kedatangan kapal dengan buffer waktu cuaca",
            "Prioritaskan keberangkatan tepat waktu daripada volume"
        ],
        "Balanced Plan": [
            "Pertahankan volume pengiriman sesuai rencana",
            "Penyesuaian minor hanya saat cuaca ekstrem",
            "Monitoring real-time dan penjadwalan ulang dinamis"
        ],
        "Aggressive Plan": [
            "Tingkatkan target loading 10%",
            "Perpanjang jam loading untuk maksimalkan throughput",
            "Terima risiko demurrage moderat untuk keuntungan volume"
        ]
    }
    return steps.get(plan_name, [
        "Implementasikan rencana mulai Hari 1",
        "Monitor kondisi cuaca dan operasional",
        "Sesuaikan operasi berdasarkan evaluasi harian"
    ])

def _generate_shipping_strengths(plan_name: str, financial_impact: Dict) -> List[str]:
    """Generate strengths for shipping plans"""
    strengths_map = {
        "Conservative Plan": [
            "Meminimalkan penalti demurrage",
            "Mengurangi keterlambatan akibat cuaca",
            "Prediktabilitas turnaround kapal lebih baik",
            "Tekanan operasional lebih rendah"
        ],
        "Balanced Plan": [
            "Mencapai target revenue",
            "Eksposur demurrage terkendali",
            "Adaptasi fleksibel terhadap kondisi",
            "Utilisasi kapasitas kapal optimal"
        ],
        "Aggressive Plan": [
            "Generasi revenue maksimum",
            "Utilisasi kapasitas kapal penuh",
            "Menangkap peluang pasar permintaan tinggi",
            "Memperkuat posisi pasar"
        ]
    }
    return strengths_map.get(plan_name, [
        "Struktur rencana yang jelas",
        "Mempertimbangkan risiko operasional",
        "Target yang terukur dan realistis"
    ])

def _generate_shipping_limitations(plan_name: str, financial_impact: Dict) -> List[Dict[str, str]]:
    """Generate limitations for shipping plans"""
    limitations_map = {
        "Conservative Plan": [
            {"keterbatasan": "Revenue lebih rendah", "deskripsi": "Volume pengiriman yang berkurang berdampak pada total revenue"},
            {"keterbatasan": "Potensi kehilangan peluang", "deskripsi": "Mungkin kehilangan peluang pengiriman di hari optimal"},
            {"keterbatasan": "Penjadwalan ulang", "deskripsi": "Mungkin perlu penjadwalan ulang kapal dan dermaga"}
        ],
        "Balanced Plan": [
            {"keterbatasan": "Memerlukan monitoring aktif", "deskripsi": "Perlu monitoring cuaca dan operasional yang intensif"},
            {"keterbatasan": "Rencana kontingensi", "deskripsi": "Perlu rencana kontingensi untuk kondisi tak terduga"},
            {"keterbatasan": "Risiko moderat", "deskripsi": "Risiko demurrage moderat masih ada dan perlu diwaspadai"}
        ],
        "Aggressive Plan": [
            {"keterbatasan": "Risiko demurrage tinggi", "deskripsi": "Potensi biaya demurrage lebih tinggi akibat target ambisius"},
            {"keterbatasan": "Eksposur cuaca meningkat", "deskripsi": "Lebih rentan terhadap gangguan cuaca karena volume tinggi"},
            {"keterbatasan": "Keterlambatan jadwal", "deskripsi": "Potensi keterlambatan jadwal kapal lebih besar"},
            {"keterbatasan": "Resource intensif", "deskripsi": "Memerlukan resource kontingensi yang kuat dan siap siaga"}
        ]
    }
    return limitations_map.get(plan_name, [
        {"keterbatasan": "Data terbatas", "deskripsi": "Berdasarkan data historis yang tersedia"},
        {"keterbatasan": "Asumsi model", "deskripsi": "Menggunakan asumsi biaya dan revenue standar"}
    ])

def _generate_shipping_justification(plan_data: Dict[str, Any]) -> str:
    """Manual fallback justification for shipping - FIXED DETAILED VERSION"""
    plan_name = plan_data.get('plan_name', '')
    financial = plan_data.get('financial_impact', {})
    savings = financial.get('demurrage_savings_usd', 0)
    revenue_change = financial.get('revenue_change_usd', 0)
    baseline_rev = financial.get('baseline_total_revenue_usd', 0)
    optimized_rev = financial.get('optimized_total_revenue_usd', 0)
    risk_score = financial.get('avg_risk_score', 0)
    
    # Ambil data schedule untuk informasi cuaca
    schedule = plan_data.get('optimized_schedule', [])
    if schedule:
        total_days = len(schedule)
        high_wind_days = sum(1 for s in schedule if "Angin:" in s.get('weather_condition', '') and float(s['weather_condition'].split(':')[1].replace('km/j', '').strip()) > 25)
    else:
        total_days = 0
        high_wind_days = 0
    
    if "Conservative" in plan_name:
        return f"""Rencana konservatif berfokus pada minimisasi biaya demurrage dan risiko keterlambatan akibat cuaca. 
Strategi ini mengurangi target loading pada hari dengan angin kencang (>25 km/jam) untuk menghindari penalti demurrage yang tinggi.
Dari {total_days} hari operasi, terdapat {high_wind_days} hari dengan kondisi angin di atas ambang batas normal.
Dengan pendekatan ini, revenue turun sebesar ${abs(revenue_change):,.0f} dari baseline ${baseline_rev:,.0f} menjadi ${optimized_rev:,.0f}, namun biaya demurrage dapat dihemat sebesar ${savings:,.0f}.
Rencana ini cocok untuk kondisi cuaca tidak menentu atau ketika stabilitas operasional lebih diprioritaskan daripada pencapaian target maksimal. Skor risiko rata-rata {risk_score:.2f} menunjukkan tingkat risiko yang dapat dikelola."""
    
    elif "Aggressive" in plan_name:
        return f"""Rencana agresif memaksimalkan volume pengiriman untuk mencapai target revenue tertinggi.
Strategi ini meningkatkan target loading hingga 10% pada hari optimal dan memperpanjang jam operasional untuk memaksimalkan throughput.
Revenue meningkat sebesar ${revenue_change:,.0f} dari baseline ${baseline_rev:,.0f} menjadi ${optimized_rev:,.0f}, dengan penghematan demurrage sebesar ${savings:,.0f}.
Dari {total_days} hari operasi, terdapat {high_wind_days} hari dengan kondisi angin di atas ambang batas normal yang memerlukan penyesuaian jadwal.
Rencana ini cocok untuk memanfaatkan permintaan pasar tinggi dengan menerima risiko demurrage moderat sebesar {risk_score:.2f} pada skala 0-1."""
    
    else:  # Balanced Plan
        return f"""Rencana seimbang menjaga keseimbangan antara pencapaian target revenue dan manajemen risiko demurrage.
Strategi ini mempertahankan volume pengiriman sesuai rencana dengan penyesuaian minor hanya pada kondisi cuaca ekstrem.
Revenue berubah sebesar ${revenue_change:,.0f} dari baseline ${baseline_rev:,.0f} menjadi ${optimized_rev:,.0f}, dengan penghematan demurrage ${savings:,.0f}.
Dari {total_days} hari operasi, operasi pengiriman tetap berjalan dengan efisiensi yang terjaga meskipun terdapat {high_wind_days} hari dengan kondisi angin di atas ambang batas.
Pendekatan ini ideal untuk operasi rutin dengan monitoring real-time dan fleksibilitas penjadwalan dinamis, dengan skor risiko rata-rata {risk_score:.2f}."""

# -------------------------
# Main function - generate_top3_mining_plans
# -------------------------
def generate_top3_mining_plans(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate top 3 mining plans with detailed analysis"""
    print(f"🚀 START generate_top3_mining_plans with {len(predictions)} rows")
    
    try:
        df = predictions.copy()
        print(f"📊 Data shape: {df.shape}")
        
        # Ensure date column exists
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_datetime(df['plan_date'])
        else:
            df['plan_date'] = pd.date_range(start=datetime.now().date(), periods=len(df), freq='D')
        
        # Buat executive summary
        executive_summary = {
            "period": f"{df['plan_date'].min()} hingga {df['plan_date'].max()}",
            "total_days": int(len(df['plan_date'].unique())),
            "total_planned_production_ton": float(df['planned_production_ton'].sum()) if 'planned_production_ton' in df.columns else 0.0,
            "avg_efficiency": round(float(df['efficiency_factor'].mean()) if 'efficiency_factor' in df.columns else 0.6, 2),
            "avg_delay_min": round(float(df['cycle_delay_min'].mean()) if 'cycle_delay_min' in df.columns else 0.0, 2),
            "high_risk_days": int((df['risk_level'] == 'HIGH').sum() if 'risk_level' in df.columns else 0)
        }
        
        print(f"📈 Executive summary: {executive_summary}")
        
        strategies = [
            {"id": 1, "name": "Conservative Plan", "prod_multiplier": 0.90, "risk_threshold": 0.7, "description": "Meminimalkan risiko operasional akibat cuaca dan keterlambatan"},
            {"id": 2, "name": "Balanced Plan", "prod_multiplier": 1.00, "risk_threshold": 0.6, "description": "Menyeimbangkan target produksi dengan pengelolaan risiko yang efektif"},
            {"id": 3, "name": "Aggressive Plan", "prod_multiplier": 1.10, "risk_threshold": 0.5, "description": "Memaksimalkan volume produksi untuk memenuhi permintaan tinggi"}
        ]
        
        recommendations = []
        
        for s in strategies:
            print(f"🔄 Processing strategy: {s['name']}")
            
            optimized_schedule = []
            total_baseline_cost = 0.0
            total_optimized_cost = 0.0
            total_risk_score = 0.0
            
            for idx, row in df.iterrows():
                # Get confidence from available columns
                conf_items = []
                for col in ['efficiency_factor', 'confidence_score']:
                    if col in row and not pd.isna(row[col]):
                        try:
                            conf_items.append(float(row[col]))
                        except:
                            pass
                
                # Use default if no confidence data
                if not conf_items:
                    overall_conf = 0.6
                else:
                    overall_conf = float(np.mean(conf_items))
                    # Ensure between 0-1
                    overall_conf = max(0.1, min(1.0, overall_conf))
                
                # Apply multiplier based on risk threshold
                adj_multiplier = s['prod_multiplier']
                if overall_conf < s['risk_threshold']:
                    adj_multiplier *= 0.8
                
                # Get production tonnage
                original_prod = float(row.get('planned_production_ton', 0.0))
                adjusted_prod = round(original_prod * adj_multiplier, 0)
                
                # Cost calculation
                baseline_cost = original_prod * 30.0
                optimized_cost = adjusted_prod * 30.0
                total_baseline_cost += baseline_cost
                total_optimized_cost += optimized_cost
                total_risk_score += (1.0 - overall_conf)
                
                # Weather information
                weather_info = []
                if 'precipitation_mm' in row and not pd.isna(row['precipitation_mm']):
                    weather_info.append(f"Hujan: {row['precipitation_mm']}mm")
                if 'wind_speed_kmh' in row and not pd.isna(row['wind_speed_kmh']):
                    weather_info.append(f"Angin: {row['wind_speed_kmh']}km/j")
                if 'weather_impact' in row and not pd.isna(row['weather_impact']):
                    weather_info.append(f"Dampak: {row['weather_impact']}")
                
                weather_condition = " | ".join(weather_info) if weather_info else "Kondisi normal"
                
                optimized_schedule.append({
                    "date": str(row['plan_date'].date()),
                    "plan_id": row.get('plan_id', f"MP{idx+1:03d}"),
                    "original_production_ton": int(original_prod),
                    "optimized_production_ton": int(adjusted_prod),
                    "adjustment_pct": round(((adjusted_prod - original_prod) / original_prod * 100) if original_prod > 0 else 0, 2),
                    "baseline_cost_usd": round(float(baseline_cost), 2),
                    "optimized_cost_usd": round(float(optimized_cost), 2),
                    "confidence_score": round(overall_conf, 2),
                    "weather_condition": weather_condition,
                    "rationale": f"{s['name']}: {'Dikurangi karena risiko tinggi' if overall_conf < s['risk_threshold'] else 'Beroperasi normal'}"
                })
            
            financial_impact = {
                "baseline_total_cost_usd": round(total_baseline_cost, 2),
                "optimized_total_cost_usd": round(total_optimized_cost, 2),
                "cost_savings_usd": round(total_baseline_cost - total_optimized_cost, 2),
                "avg_risk_score": round(total_risk_score / len(df), 2) if len(df) > 0 else 0.0
            }
            
            print(f"💰 Financial impact for {s['name']}: {financial_impact}")
            
            # Build plan object
            plan_obj = {
                "plan_id": s['id'],
                "plan_name": s['name'],
                "strategy_description": s['description'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": financial_impact,
                "implementation_steps": _get_implementation_steps(s['name']),
                "strengths": _get_strengths(s['name'], financial_impact, optimized_schedule),
                "limitations": _get_limitations(s['name'], financial_impact, len(optimized_schedule)),
                "justification": _generate_detailed_justification(s['name'], financial_impact, optimized_schedule)
            }
            
            recommendations.append(plan_obj)
            print(f"✅ Completed {s['name']}")
        
        result = {
            "plan_type": "RENCANA OPTIMASI PERTAMBANGAN",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "recommendations": recommendations
        }
        
        print(f"🎉 FINISHED: Generated {len(recommendations)} recommendations")
        return result
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR in generate_top3_mining_plans: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal safe result
        return {
            "plan_type": "RENCANA OPTIMASI PERTAMBANGAN",
            "generated_at": datetime.now().isoformat(),
            "error": f"Error generating plans: {str(e)}",
            "executive_summary": {},
            "recommendations": []
        }

# -------------------------
# Shipping plans generator
# -------------------------
def generate_top3_shipping_plans(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate top 3 shipping plans"""
    try:
        print(f"🚢 START generate_top3_shipping_plans with {len(predictions)} rows")
        df = predictions.copy()
        
        # Pastikan kolom yang diperlukan ada
        df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        # Kelompokkan per hari jika ada multiple shipments per hari
        daily_df = df.groupby('eta_date').agg({
            'planned_volume_ton': 'sum',
            'loading_efficiency': 'mean',
            'predicted_demurrage_cost': 'sum',
            'wind_speed_kmh': 'mean',
            'confidence_score': 'mean'
        }).reset_index()
        
        # Buat executive summary dengan weather summary
        executive_summary = {
            "period": f"{daily_df['eta_date'].min().strftime('%Y-%m-%d %H:%M:%S')} hingga {daily_df['eta_date'].max().strftime('%Y-%m-%d %H:%M:%S')}",
            "total_days": int(len(daily_df)),
            "total_planned_shipment_ton": round(float(daily_df['planned_volume_ton'].sum()), 2),
            "avg_loading_efficiency": round(float(daily_df['loading_efficiency'].mean()), 2),
            "total_demurrage_cost_usd": round(float(daily_df['predicted_demurrage_cost'].sum()), 2),
            "high_risk_days": int((daily_df['loading_efficiency'] < 0.5).sum()),
            "weather_summary": {
                "max_wind_speed_kmh": round(float(daily_df['wind_speed_kmh'].max()), 1),
                "high_wind_days": int((daily_df['wind_speed_kmh'] > 30).sum())
            }
        }
        
        print(f"📊 Shipping executive summary: {executive_summary}")
        
        # 3 strategi
        strategies = [
            {"id": 1, "name": "Conservative Plan", "multiplier": 0.85, "description": "Minimalkan biaya demurrage dan keterlambatan akibat cuaca"},
            {"id": 2, "name": "Balanced Plan", "multiplier": 1.00, "description": "Optimalkan revenue sambil mengelola risiko demurrage dan cuaca"},
            {"id": 3, "name": "Aggressive Plan", "multiplier": 1.10, "description": "Maksimalkan revenue pengiriman dan utilisasi kapal"}
        ]
        
        recommendations = []
        
        for s in strategies:
            print(f"🔄 Processing shipping strategy: {s['name']}")
            
            optimized_schedule = []
            baseline_total_revenue = 0
            optimized_total_revenue = 0
            demurrage_savings = 0
            
            for i, (_, row) in enumerate(daily_df.iterrows(), 1):
                original_ton = row['planned_volume_ton']
                
                # Hitung adjusted ton berdasarkan strategi dan kondisi
                if "Conservative" in s['name']:
                    if row['wind_speed_kmh'] > 25:
                        adjusted_ton = original_ton * 0.85
                        rationale = "Kurangi volume 15% karena risiko cuaca/demurrage"
                    else:
                        adjusted_ton = original_ton * 0.95
                        rationale = "Operasi pengiriman normal dengan pendekatan hati-hati"
                
                elif "Aggressive" in s['name']:
                    if row['loading_efficiency'] < 0.5:
                        adjusted_ton = original_ton * 0.935  # Kurangi 6.5%
                        rationale = "Kurangi volume 15% pada kondisi tidak optimal"
                    else:
                        adjusted_ton = original_ton * 1.10
                        rationale = "Operasi pengiriman maksimal untuk capai target tinggi"
                
                else:  # Balanced Plan
                    if row['wind_speed_kmh'] > 25 or row['loading_efficiency'] < 0.6:
                        adjusted_ton = original_ton * 0.85
                        rationale = "Kurangi volume 15% untuk antisipasi risiko"
                    else:
                        adjusted_ton = original_ton
                        rationale = "Operasi pengiriman normal sesuai rencana"
                
                # Hitung revenue (contoh: $65 per ton)
                baseline_revenue = original_ton * 65
                optimized_revenue = adjusted_ton * 65
                
                # Hitung penghematan demurrage (proporsional dengan pengurangan volume)
                demurrage_saving = row['predicted_demurrage_cost'] * (1 - (adjusted_ton / original_ton)) if original_ton > 0 else 0
                
                baseline_total_revenue += baseline_revenue
                optimized_total_revenue += optimized_revenue
                demurrage_savings += demurrage_saving
                
                optimized_schedule.append({
                    "date": row['eta_date'].strftime('%Y-%m-%d %H:%M:%S'),
                    "day": i,
                    "original_shipping_ton": round(original_ton, 0),
                    "optimized_shipping_ton": round(adjusted_ton, 0),
                    "adjustment_pct": round(((adjusted_ton - original_ton) / original_ton * 100), 2) if original_ton > 0 else 0,
                    "baseline_revenue_usd": round(baseline_revenue, 2),
                    "optimized_revenue_usd": round(optimized_revenue, 2),
                    "demurrage_cost_usd": round(row['predicted_demurrage_cost'], 2),
                    "confidence_score": round(row['confidence_score'], 2) if not pd.isna(row['confidence_score']) else 0.6,
                    "weather_condition": f"Angin: {round(row['wind_speed_kmh'], 1)}km/j",
                    "rationale": rationale
                })
            
            # Hitung financial impact
            financial_impact = {
                "baseline_total_revenue_usd": round(baseline_total_revenue, 2),
                "optimized_total_revenue_usd": round(optimized_total_revenue, 2),
                "revenue_change_usd": round(optimized_total_revenue - baseline_total_revenue, 2),
                "demurrage_savings_usd": round(demurrage_savings, 2),
                "avg_risk_score": round(1 - (daily_df['loading_efficiency'].mean()), 2)
            }
            
            print(f"💰 Financial impact for {s['name']}: {financial_impact}")
            
            # PERBAIKAN DI SINI: Hitung justification terlebih dahulu
            justification = _generate_shipping_justification({
                "plan_name": s['name'],
                "financial_impact": financial_impact,
                "optimized_schedule": optimized_schedule
            })
            
            # Buat plan object dengan justification yang sudah dihitung
            plan_obj = {
                "plan_id": s['id'],
                "plan_name": s['name'],
                "strategy_description": s['description'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": financial_impact,
                "implementation_steps": _generate_shipping_steps(s['name']),
                "strengths": _generate_shipping_strengths(s['name'], financial_impact),
                "limitations": _generate_shipping_limitations(s['name'], financial_impact),
                "justification": justification  # Gunakan yang sudah dihitung
            }
            
            recommendations.append(plan_obj)
            print(f"✅ Completed shipping plan: {s['name']}")
        
        result = {
            "plan_type": "RENCANA OPTIMASI PENGIRIMAN",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "recommendations": recommendations
        }
        
        print(f"🎉 FINISHED Shipping Plans: Generated {len(recommendations)} recommendations")
        return result
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR in generate_top3_shipping_plans: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal safe result
        return {
            "plan_type": "RENCANA OPTIMASI PENGIRIMAN",
            "generated_at": datetime.now().isoformat(),
            "error": f"Error generating shipping plans: {str(e)}",
            "executive_summary": {},
            "recommendations": []
        }
    """Generate top 3 shipping plans"""
    try:
        df = predictions.copy()
        
        # Pastikan kolom yang diperlukan ada
        df['eta_date'] = pd.to_datetime(df['eta_date'])
        
        # Kelompokkan per hari jika ada multiple shipments per hari
        daily_df = df.groupby('eta_date').agg({
            'planned_volume_ton': 'sum',
            'loading_efficiency': 'mean',
            'predicted_demurrage_cost': 'sum',
            'wind_speed_kmh': 'mean',
            'confidence_score': 'mean'
        }).reset_index()
        
        # Buat executive summary dengan weather summary
        executive_summary = {
            "period": f"{daily_df['eta_date'].min().strftime('%Y-%m-%d %H:%M:%S')} hingga {daily_df['eta_date'].max().strftime('%Y-%m-%d %H:%M:%S')}",
            "total_days": int(len(daily_df)),
            "total_planned_shipment_ton": round(float(daily_df['planned_volume_ton'].sum()), 2),
            "avg_loading_efficiency": round(float(daily_df['loading_efficiency'].mean()), 2),
            "total_demurrage_cost_usd": round(float(daily_df['predicted_demurrage_cost'].sum()), 2),
            "high_risk_days": int((daily_df['loading_efficiency'] < 0.5).sum()),
            "weather_summary": {
                "max_wind_speed_kmh": round(float(daily_df['wind_speed_kmh'].max()), 1),
                "high_wind_days": int((daily_df['wind_speed_kmh'] > 30).sum())
            }
        }
        
        # 3 strategi yang sama
        strategies = [
            {"id": 1, "name": "Conservative Plan", "multiplier": 0.85, "description": "Minimalkan biaya demurrage dan keterlambatan akibat cuaca"},
            {"id": 2, "name": "Balanced Plan", "multiplier": 1.00, "description": "Optimalkan revenue sambil mengelola risiko demurrage dan cuaca"},
            {"id": 3, "name": "Aggressive Plan", "multiplier": 1.10, "description": "Maksimalkan revenue pengiriman dan utilisasi kapal"}
        ]
        
        recommendations = []
        
        for s in strategies:
            optimized_schedule = []
            baseline_total_revenue = 0
            optimized_total_revenue = 0
            demurrage_savings = 0
            
            for i, (_, row) in enumerate(daily_df.iterrows(), 1):
                original_ton = row['planned_volume_ton']
                
                # Hitung adjusted ton berdasarkan strategi dan kondisi
                if "Conservative" in s['name']:
                    if row['wind_speed_kmh'] > 25:
                        adjusted_ton = original_ton * 0.85
                        rationale = "Kurangi volume 15% karena risiko cuaca/demurrage"
                    else:
                        adjusted_ton = original_ton * 0.95
                        rationale = "Operasi pengiriman normal dengan pendekatan hati-hati"
                
                elif "Aggressive" in s['name']:
                    if row['loading_efficiency'] < 0.5:
                        adjusted_ton = original_ton * 0.935  # Kurangi 6.5%
                        rationale = "Kurangi volume 15% pada kondisi tidak optimal"
                    else:
                        adjusted_ton = original_ton * 1.10
                        rationale = "Operasi pengiriman maksimal untuk capai target tinggi"
                
                else:  # Balanced Plan
                    if row['wind_speed_kmh'] > 25 or row['loading_efficiency'] < 0.6:
                        adjusted_ton = original_ton * 0.85
                        rationale = "Kurangi volume 15% untuk antisipasi risiko"
                    else:
                        adjusted_ton = original_ton
                        rationale = "Operasi pengiriman normal sesuai rencana"
                
                # Hitung revenue (contoh: $65 per ton)
                baseline_revenue = original_ton * 65
                optimized_revenue = adjusted_ton * 65
                
                # Hitung penghematan demurrage (proporsional dengan pengurangan volume)
                demurrage_saving = row['predicted_demurrage_cost'] * (1 - (adjusted_ton / original_ton))
                
                baseline_total_revenue += baseline_revenue
                optimized_total_revenue += optimized_revenue
                demurrage_savings += demurrage_saving
                
                optimized_schedule.append({
                    "date": row['eta_date'].strftime('%Y-%m-%d %H:%M:%S'),
                    "day": i,
                    "original_shipping_ton": round(original_ton, 0),
                    "optimized_shipping_ton": round(adjusted_ton, 0),
                    "adjustment_pct": round(((adjusted_ton - original_ton) / original_ton * 100), 2),
                    "baseline_revenue_usd": round(baseline_revenue, 2),
                    "optimized_revenue_usd": round(optimized_revenue, 2),
                    "demurrage_cost_usd": round(row['predicted_demurrage_cost'], 2),
                    "confidence_score": round(row['confidence_score'], 2),
                    "weather_condition": f"Angin: {round(row['wind_speed_kmh'], 1)}km/j",
                    "rationale": rationale
                })
            
            # Hitung financial impact
            financial_impact = {
                "baseline_total_revenue_usd": round(baseline_total_revenue, 2),
                "optimized_total_revenue_usd": round(optimized_total_revenue, 2),
                "revenue_change_usd": round(optimized_total_revenue - baseline_total_revenue, 2),
                "demurrage_savings_usd": round(demurrage_savings, 2),
                "avg_risk_score": round(1 - (daily_df['loading_efficiency'].mean()), 2)
            }
            
            # Buat plan object
            plan_obj = {
                "plan_id": s['id'],
                "plan_name": s['name'],
                "strategy_description": s['description'],
                "optimized_schedule": optimized_schedule,
                "financial_impact": financial_impact,
                "implementation_steps": _generate_shipping_steps(s['name']),
                "strengths": _generate_shipping_strengths(s['name'], financial_impact),
                "limitations": _generate_shipping_limitations(s['name'], financial_impact),
                "justification": _generate_shipping_justification(plan_obj)
            }
            
            recommendations.append(plan_obj)
        
        return {
            "plan_type": "RENCANA OPTIMASI PENGIRIMAN",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"🔥 ERROR in generate_top3_shipping_plans: {e}")
        return {
            "plan_type": "RENCANA OPTIMASI PENGIRIMAN",
            "generated_at": datetime.now().isoformat(),
            "error": f"Error generating shipping plans: {str(e)}",
            "executive_summary": {},
            "recommendations": []
        }