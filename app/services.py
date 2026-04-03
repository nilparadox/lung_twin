from __future__ import annotations

import math
import os
import time
from typing import Any

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_FALLBACK_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENAQ_LOCATIONS_URL = "https://api.openaq.org/v3/locations"

HEADERS = {
    "User-Agent": "ZetaQ-AirWise-Live/4.1"
}

CACHE: dict[str, tuple[float, Any]] = {}
CACHE_TTL = 900.0


def cache_get(key: str):
    item = CACHE.get(key)
    if item is None:
        return None
    ts, value = item
    if time.time() - ts > CACHE_TTL:
        CACHE.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any):
    CACHE[key] = (time.time(), value)


def geocode_place(query: str) -> dict:
    key = f"geo::{query.strip().lower()}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    params = {
        "name": query,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    r = requests.get(GEOCODE_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    if not results:
        raise RuntimeError("Could not find that city or town.")

    item = results[0]
    result = {
        "display_name": ", ".join(
            [x for x in [item.get("name"), item.get("admin1"), item.get("country")] if x]
        ),
        "latitude": item["latitude"],
        "longitude": item["longitude"],
    }
    cache_set(key, result)
    return result


def get_weather(lat: float, lon: float) -> dict:
    key = f"weather::{lat:.4f},{lon:.4f}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto",
    }

    try:
        r = requests.get(WEATHER_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json().get("current", {})
        result = {
            "temp_c": float(data.get("temperature_2m", 30.0)),
            "humidity": float(data.get("relative_humidity_2m", 60.0)),
            "wind_speed": float(data.get("wind_speed_10m", 5.0)),
            "weather_note": None,
        }
    except Exception as e:
        print(f"[WEATHER] fallback because: {e}")
        result = {
            "temp_c": 30.0,
            "humidity": 60.0,
            "wind_speed": 5.0,
            "weather_note": "Weather fallback values used.",
        }

    cache_set(key, result)
    return result


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def get_live_air(lat: float, lon: float) -> dict:
    key = f"air::{lat:.4f},{lon:.4f}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    api_key = os.getenv("OPENAQ_API_KEY", "").strip()
    print(f"[OPENAQ] key present: {bool(api_key)}")

    if api_key:
        try:
            params = {
                "coordinates": f"{lat},{lon}",
                "radius": 25000,
                "limit": 5,
            }
            headers = dict(HEADERS)
            headers["X-API-Key"] = api_key

            print(f"[OPENAQ] searching stations near {lat}, {lon}")
            loc_res = requests.get(OPENAQ_LOCATIONS_URL, params=params, headers=headers, timeout=20)
            print(f"[OPENAQ] location status: {loc_res.status_code}")
            loc_res.raise_for_status()

            loc_json = loc_res.json()
            loc_data = loc_json.get("results", [])
            print(f"[OPENAQ] stations returned: {len(loc_data)}")

            if loc_data:
                best = None
                best_dist = 10**9

                for loc in loc_data:
                    coords = loc.get("coordinates") or {}
                    ll = coords.get("latitude")
                    lo = coords.get("longitude")
                    if ll is None or lo is None:
                        continue
                    d = _distance_km(lat, lon, ll, lo)
                    if d < best_dist:
                        best_dist = d
                        best = loc

                if best is not None:
                    loc_id = best["id"]
                    latest_url = f"{OPENAQ_LOCATIONS_URL}/{loc_id}/latest"
                    latest_res = requests.get(latest_url, headers=headers, timeout=20)
                    print(f"[OPENAQ] latest status: {latest_res.status_code}")
                    latest_res.raise_for_status()

                    latest_json = latest_res.json()
                    latest_rows = latest_json.get("results", [])
                    print(f"[OPENAQ] latest rows: {len(latest_rows)}")

                    pm25 = None
                    pm10 = None
                    no2 = None
                    o3 = None

                    for row in latest_rows:
                        param_obj = row.get("parameter") or {}
                        param = param_obj.get("name")
                        value = row.get("value")

                        if param == "pm25":
                            pm25 = value
                        elif param == "pm10":
                            pm10 = value
                        elif param == "no2":
                            no2 = value
                        elif param == "o3":
                            o3 = value

                    result = {
                        "pm25": float(pm25) if pm25 is not None else 35.0,
                        "pm10": float(pm10) if pm10 is not None else 55.0,
                        "no2": float(no2) if no2 is not None else 20.0,
                        "o3": float(o3) if o3 is not None else 30.0,
                        "source_label": "Live station data",
                        "source_note": f"Nearest OpenAQ station at ~{best_dist:.1f} km",
                    }
                    cache_set(key, result)
                    return result
            else:
                print("[OPENAQ] no stations found within 25 km")

        except Exception as e:
            print(f"[OPENAQ] fallback because: {e}")

    else:
        print("[OPENAQ] no API key loaded")

    print("[OPENAQ] using Open-Meteo fallback")
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "pm2_5,pm10,nitrogen_dioxide,ozone,european_aqi,us_aqi",
        "timezone": "auto",
    }

    r = requests.get(AIR_FALLBACK_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json().get("current", {})

    result = {
        "pm25": float(data.get("pm2_5", 40.0)),
        "pm10": float(data.get("pm10", 60.0)),
        "no2": float(data.get("nitrogen_dioxide", 20.0)),
        "o3": float(data.get("ozone", 30.0)),
        "source_label": "Modeled fallback data",
        "source_note": "OpenAQ unavailable nearby, using Open-Meteo modeled air quality.",
    }
    cache_set(key, result)
    return result
