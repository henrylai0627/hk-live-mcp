"""KMB / LWB real-time ETA API (DATA.GOV.HK)."""
import re
import time
from datetime import datetime, timedelta, timezone

import httpx

from .._http import get_json

BASE = "https://data.etabus.gov.hk/v1/transport/kmb"
HKT = timezone(timedelta(hours=8))
_TTL = 24 * 3600
_RETRY_AFTER = 60
_cache: tuple[float, list[dict]] | None = None
_failed_at = 0.0


async def _stops() -> list[dict]:
    # ~6k stops; fetch once per day instead of per tool call
    global _cache, _failed_at
    if _cache and time.time() - _cache[0] < _TTL:
        return _cache[1]
    # /stop answers 403 when fetched repeatedly, so back off after a failure
    # instead of re-downloading the list on every tool call
    if time.time() - _failed_at >= _RETRY_AFTER:
        try:
            rows = (await get_json(f"{BASE}/stop")).get("data", [])
        except httpx.HTTPError:
            _failed_at = time.time()
        else:
            _cache = (time.time(), rows)
            return rows
    if _cache:  # an expired list is better than none
        return _cache[1]
    raise RuntimeError(
        "KMB stop list is temporarily unavailable (the API is rate-limiting); "
        "retry in about a minute."
    )


async def find_stop(query: str, limit: int = 10) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return []
    hits = [
        {"stop_id": s["stop"], "name_tc": s["name_tc"], "name_en": s["name_en"],
         "lat": s["lat"], "long": s["long"]}
        for s in await _stops()
        if q in s["name_en"].lower() or q in s["name_tc"]
    ]
    return hits[: max(1, limit)]


def _route_key(route: str) -> list:
    # natural order: 5 before 28, 1 before 1A
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", route)]


async def stop_eta(stop_id: str, route: str | None = None, now: datetime | None = None) -> list[dict]:
    d = await get_json(f"{BASE}/stop-eta/{stop_id.upper()}")
    now = now or datetime.now(HKT)
    seen: set[tuple] = set()
    out = []
    for e in d.get("data", []):
        if route and e["route"].upper() != route.upper():
            continue
        # API repeats the same bus once per service_type
        key = (e["route"], e.get("dir"), e.get("eta"))
        if key in seen:
            continue
        seen.add(key)
        # null ETA = no scheduled bus / service ended; keep the row so the
        # remark (e.g. "weekends only") is not lost
        minutes = None
        if e.get("eta"):
            eta = datetime.fromisoformat(e["eta"])
            minutes = max(0, round((eta - now).total_seconds() / 60))
        out.append({
            "route": e["route"],
            "dest": e["dest_tc"],
            "direction": e.get("dir"),
            "stop_seq": e.get("seq"),
            "eta": e.get("eta"),
            "minutes": minutes,
            "remark": e.get("rmk_tc") or None,
        })
    return sorted(out, key=lambda x: (_route_key(x["route"]), x["minutes"] is None, x["minutes"] or 0))
