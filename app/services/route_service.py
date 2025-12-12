import requests

class RouteService:

    @staticmethod
    def road_distance_osrm(lat1, lon1, lat2, lon2):
        try:
            url = (
                f"http://router.project-osrm.org/route/v1/driving/"
                f"{lon1},{lat1};{lon2},{lat2}?overview=false"
            )

            r = requests.get(url, timeout=10).json()

            route = r["routes"][0]
            return route["distance"] / 1000, route["duration"] / 60

        except Exception:
            return None, None

    @staticmethod
    def compute_route(rom_lat, rom_lon, jetty_lat, jetty_lon):
        dist_km, dur_min = RouteService.road_distance_osrm(
            rom_lat, rom_lon, jetty_lat, jetty_lon
        )

        return {
            "distance_km": dist_km,
            "duration_min": dur_min
        }
