import requests
from datetime import date, datetime

class WeatherService:
    @staticmethod
    def fetch_weather(lat: float, lon: float, target_date: date) -> dict:
        
        date_str = target_date.strftime("%Y-%m-%d")
        today = datetime.now().date()

        if target_date < today:
            url = (
                "https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}&"
                f"start_date={date_str}&end_date={date_str}&"
                "daily=temperature_2m_mean,windspeed_10m_max,"
                "precipitation_sum,cloudcover_mean&"
                "timezone=Asia/Jakarta"
            )
        else:
            url = (
                "https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&"
                "daily=temperature_2m_mean,windspeed_10m_max,"
                "precipitation_sum,cloudcover_mean&"
                "timezone=Asia/Jakarta"
            )

        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            if "daily" not in data:
                raise ValueError("Missing daily weather data")

            daily = data["daily"]

            return {
                "temp_day": daily["temperature_2m_mean"][0],
                "wind_speed_kmh": daily["windspeed_10m_max"][0],
                "precipitation_mm": daily["precipitation_sum"][0],
                "cloud_cover_pct": daily["cloudcover_mean"][0],
            }

        except Exception:
            return {
                "temp_day": 25,
                "wind_speed_kmh": 10,
                "precipitation_mm": 0,
                "cloud_cover_pct": 30,
            }