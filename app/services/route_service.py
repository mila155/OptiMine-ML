import requests
from functools import lru_cache

class RouteService:
    def __init__(self):
        self.base_url = "http://router.project-osrm.org/route/v1/driving"

    @lru_cache(maxsize=256)
    def compute_route(self, lat1, lon1, lat2, lon2):
        url = f"{self.base_url}/{lon1},{lat1};{lon2},{lat2}?overview=false"
        try:
            r = requests.get(url, timeout=8)
            data = r.json()
            route = data["routes"][0]
            return round(route["distance"] / 1000, 2), round(route["duration"] / 60, 2)
        except Exception:
            return None, None
