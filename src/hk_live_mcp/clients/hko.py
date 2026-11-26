"""Hong Kong Observatory open data API."""
from typing import Literal

from .._http import get_json

BASE = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
Lang = Literal["en", "tc", "sc"]


async def current(lang: Lang = "tc", district: str | None = None) -> dict:
    d = await get_json(BASE, {"dataType": "rhrread", "lang": lang})
    temps = d.get("temperature", {}).get("data", [])
    rainfall = d.get("rainfall", {})
    rain = rainfall.get("data", [])
    rain_hits, note = [], None
    if district:
        q = district.lower()
        # Fall back to all stations if the filter matches nothing
        hits = [t for t in temps if q in t["place"].lower()]
        if not hits:
            note = f"No temperature station matches {district!r}; showing all stations."
        temps = hits or temps
        rain_hits = [r for r in rain if q in r["place"].lower()]
    humidity = d.get("humidity", {}).get("data", [])
    # "" (not {}) outside daylight hours
    uv = (d.get("uvindex") or {}).get("data", [])
    out = {
        "update_time": d.get("updateTime"),
        "temperatures": [{"place": t["place"], "celsius": t["value"]} for t in temps],
        "humidity_pct": humidity[0]["value"] if humidity else None,
        # Matched district, otherwise only the districts where it rained
        "rainfall_mm": [
            {"place": r["place"], "max": r.get("max")}
            for r in rain_hits or [r for r in rain if r.get("max")]
        ],
        "rainfall_period": {"start": rainfall.get("startTime"), "end": rainfall.get("endTime")},
        "uv_index": uv[0]["value"] if uv else None,
        "tropical_cyclone": d.get("tcmessage") or None,
        # API returns "" (not []) when there are no warnings
        "warnings": d.get("warningMessage") or [],
    }
    if note:
        out["note"] = note
    return out


async def forecast(days: int = 9, lang: Lang = "tc") -> dict:
    d = await get_json(BASE, {"dataType": "fnd", "lang": lang})
    return {
        "general_situation": d.get("generalSituation"),
        "days": [
            {
                "date": f["forecastDate"],
                "weekday": f["week"],
                "weather": f["forecastWeather"],
                "wind": f["forecastWind"],
                "min_c": f["forecastMintemp"]["value"],
                "max_c": f["forecastMaxtemp"]["value"],
                "rain_chance": f.get("PSR"),
            }
            for f in d.get("weatherForecast", [])[: max(1, min(days, 9))]
        ],
    }


async def warnings(lang: Lang = "tc") -> list[dict]:
    d = await get_json(BASE, {"dataType": "warnsum", "lang": lang})
    return [
        {"name": w.get("name"), "code": w.get("code"), "issued": w.get("issueTime")}
        for w in d.values()
    ]
