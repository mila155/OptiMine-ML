import requests
import json
from datetime import date, timedelta

# API Base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_mining_prediction():
    """Test single mining prediction"""
    print("\n⛏️ Testing Mining Prediction...")
    
    payload = {
        "plan_id": "MP1001",
        "plan_date": str(date.today() + timedelta(days=1)),
        "pit_id": "PIT_TUTUPAN",
        "destination_rom": "ROM_CENTRAL",
        "planned_production_ton": 8500,
        "hauling_distance_km": 12,
        "priority_flag": "High",
        "precipitation_mm": 5.2,
        "wind_speed_kmh": 15.3,
        "cloud_cover_pct": 60,
        "temp_day": 28
    }
    
    response = requests.post(f"{BASE_URL}/mining/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Prediction Results:")
        for item in result:
            print(f"  Plan ID: {item['plan_id']}")
            print(f"  Planned: {item['planned_production_ton']:.0f} tons")
            print(f"  Predicted: {item['predicted_production_ton']:.0f} tons")
            print(f"  Gap: {item['production_gap_pct']:.1f}%")
            print(f"  AI Priority: {item['ai_priority_flag']} (was: {item['original_priority_flag']})")
            print(f"  Risk Level: {item['risk_level']}")
            print(f"  Confidence: {item['confidence_score']:.2f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_mining_batch():
    """Test batch mining prediction"""
    print("\n⛏️ Testing Batch Mining Prediction...")
    
    plans = [
        {
            "plan_id": f"MP100{i}",
            "plan_date": str(date.today() + timedelta(days=i)),
            "pit_id": "PIT_TUTUPAN",
            "destination_rom": "ROM_CENTRAL",
            "planned_production_ton": 8000 + i*500,
            "hauling_distance_km": 12,
            "priority_flag": ["High", "Medium", "Low"][i % 3],
            "precipitation_mm": 5 + i*2,
            "wind_speed_kmh": 15 + i,
            "cloud_cover_pct": 60,
            "temp_day": 28
        }
        for i in range(1, 4)
    ]
    
    payload = {"plans": plans}
    
    response = requests.post(f"{BASE_URL}/mining/predict/batch", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\n📊 Processed {len(results)} plans")
        for item in results:
            print(f"  {item['plan_id']}: {item['predicted_production_ton']:.0f} tons (Risk: {item['risk_level']})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_shipping_prediction():
    """Test single shipping prediction"""
    print("\n🚢 Testing Shipping Prediction...")
    
    payload = {
        "shipment_id": "SH7001",
        "vessel_name": "MV-OCEAN",
        "assigned_jetty": "JTY-01",
        "eta_date": str(date.today() + timedelta(days=1)),
        "planned_volume_ton": 45000,
        "loading_rate_tph": 1200,
        "precipitation_mm": 3.5,
        "wind_speed_kmh": 22.5,
        "cloud_cover_pct": 45,
        "temp_day": 27
    }
    
    response = requests.post(f"{BASE_URL}/shipping/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Prediction Results:")
        for item in result:
            print(f"  Shipment ID: {item['shipment_id']}")
            print(f"  Vessel: {item['vessel_name']}")
            print(f"  Volume: {item['planned_volume_ton']:.0f} tons")
            print(f"  Loading Hours: {item['predicted_loading_hours']:.1f}")
            print(f"  Efficiency: {item['loading_efficiency']:.2f}")
            print(f"  Demurrage Cost: ${item['predicted_demurrage_cost']:.0f}")
            print(f"  Status: {item['status']}")
            print(f"  Risk: {item['risk_level']}")
            print(f"  Action: {item['recommended_action']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_shipping_batch():
    """Test batch shipping prediction"""
    print("\n🚢 Testing Batch Shipping Prediction...")
    
    plans = [
        {
            "shipment_id": f"SH700{i}",
            "vessel_name": f"MV-VESSEL-{i}",
            "assigned_jetty": "JTY-01",
            "eta_date": str(date.today() + timedelta(days=i)),
            "planned_volume_ton": 40000 + i*5000,
            "loading_rate_tph": 1200,
            "precipitation_mm": 5 + i*5,
            "wind_speed_kmh": 20 + i*3,
            "cloud_cover_pct": 50,
            "temp_day": 27
        }
        for i in range(1, 4)
    ]
    
    payload = {"plans": plans}
    
    response = requests.post(f"{BASE_URL}/shipping/predict/batch", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\n📊 Processed {len(results)} shipments")
        for item in results:
            print(f"  {item['shipment_id']}: {item['status']} - Demurrage: ${item['predicted_demurrage_cost']:.0f} (Risk: {item['risk_level']})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print(" 🚀 OPTIMINE API TESTING")
    print("="*70)
    
    tests = [
        ("Health Check", test_health),
        ("Mining Single Prediction", test_mining_prediction),
        ("Mining Batch Prediction", test_mining_batch),
        ("Shipping Single Prediction", test_shipping_prediction),
        ("Shipping Batch Prediction", test_shipping_batch)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            results.append((name, f"❌ ERROR: {e}"))
    
    print("\n" + "="*70)
    print(" 📊 TEST RESULTS")
    print("="*70)
    for name, result in results:
        print(f"{name}: {result}")
    
    all_passed = all("PASS" in r[1] for r in results)
    print("\n" + ("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"))
    print("="*70)

if __name__ == "__main__":
    run_all_tests()