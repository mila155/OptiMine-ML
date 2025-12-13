from typing import Dict, Any

def build_context(context_type: str, payload: Dict[str, Any]) -> str:
    """
    context_type:
    - mining_summary
    - shipping_summary
    - mining_optimization
    - shipping_optimization
    """

    if not payload:
        return ""

    if context_type == "mining_summary":
        return f"""
MINING SUMMARY CONTEXT:
Total Planned Production: {payload.get('total_planned_production_ton')}
Total Predicted Production: {payload.get('total_predicted_production_ton')}
Average Efficiency: {payload.get('avg_efficiency')}
High Risk Days: {payload.get('high_risk_days')}
AI Summary: {payload.get('ai_summary')}
"""

    if context_type == "shipping_summary":
        return f"""
SHIPPING SUMMARY CONTEXT:
Shipping Summary: {payload.get('shipping_summary')}
Route Recommendations: {payload.get('route_recommendations')}
AI Summary: {payload.get('ai_summary')}
"""

    if context_type == "mining_optimization":
        return f"""
MINING OPTIMIZATION CONTEXT:
Executive Summary: {payload.get('executive_summary')}
Recommendations Count: {len(payload.get('recommendations', []))}
"""

    if context_type == "shipping_optimization":
        return f"""
SHIPPING OPTIMIZATION CONTEXT:
Executive Summary: {payload.get('executive_summary')}
Recommendations Count: {len(payload.get('recommendations', []))}
"""

    return ""
