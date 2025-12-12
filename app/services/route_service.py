import requests

class RouteService:

    def road_distance_osrm(self, origin, destination):
        lat1, lon1 = origin
        lat2, lon2 = destination

        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}?overview=false"
        )

        r = requests.get(url).json()

        try:
            route = r["routes"][0]
            return route["distance"] / 1000, route["duration"] / 60
        except:
            return None, None

    def nearest_jetty(self, rom_list, jetty_list):
        results = {}

        for rom in rom_list:
            rom_name = rom["name"]
            rom_coord = (rom["latitude"], rom["longitude"])

            shortest = float("inf")
            best_jetty = None
            best_duration = None

            for jetty in jetty_list:
                jetty_name = jetty["name"]
                jetty_coord = (jetty["latitude"], jetty["longitude"])

                dist, dur = self.road_distance_osrm(rom_coord, jetty_coord)
                if dist is not None and dist < shortest:
                    shortest = dist
                    best_jetty = jetty_name
                    best_duration = dur

            results[rom_name] = {
                "nearest_jetty": best_jetty,
                "distance_km": shortest,
                "duration_min": best_duration
            }

        return results
