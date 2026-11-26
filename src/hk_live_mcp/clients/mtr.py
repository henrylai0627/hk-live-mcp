"""MTR Next Train API + station list (DATA.GOV.HK)."""
import csv
import io
import time

import httpx

from .._http import get_json, get_text

SCHEDULE = "https://rt.data.gov.hk/v1/transport/mtr/getSchedule.php"
STATIONS_CSV = "https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv"
_TTL = 24 * 3600
_cache: tuple[float, list[dict]] | None = None


async def _stations() -> list[dict]:
    global _cache
    if _cache and time.time() - _cache[0] < _TTL:
        return _cache[1]
    rows = [r for r in csv.DictReader(io.StringIO(await get_text(STATIONS_CSV))) if r.get("Station Code")]
    _cache = (time.time(), rows)
    return rows


async def find_station(query: str) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return []
    seen: set[tuple[str, str]] = set()
    out = []
    for r in await _stations():
        if (q in r["English Name"].lower() or q in r["Chinese Name"]
                or q == r["Station Code"].lower()):
            key = (r["Line Code"], r["Station Code"])
            if key in seen:  # CSV lists each station once per direction
                continue
            seen.add(key)
            out.append({
                "line": r["Line Code"],
                "station": r["Station Code"],
                "name_en": r["English Name"],
                "name_tc": r["Chinese Name"],
            })
    return out


def _fmt(trains: list[dict], names: dict[str, tuple[str, str]], n: int = 4) -> list[dict]:
    out = []
    for t in trains[:n]:
        ttnt = t.get("ttnt") or ""
        row = {"dest": t["dest"], "platform": t["plat"], "time": t["time"],
               "minutes": int(ttnt) if ttnt.isdigit() else None}
        if t["dest"] in names:
            row["dest_name_en"], row["dest_name_tc"] = names[t["dest"]]
        # East Rail Line only: route "RAC" = via Racecourse, timeType A/D = arrival/departure
        if t.get("route"):
            row["route"] = t["route"]
        if t.get("timeType"):
            row["time_type"] = t["timeType"]
        out.append(row)
    return out


async def next_train(line: str, station: str) -> dict:
    line, station = line.upper(), station.upper()
    d = await get_json(SCHEDULE, {"line": line, "sta": station})
    if d.get("status") == 0:
        return {"error": d.get("message", "Service unavailable"), "url": d.get("url")}
    sched = d.get("data", {}).get(f"{line}-{station}", {})
    if "UP" not in sched and "DOWN" not in sched:
        # status is 1 but there is no schedule at all (curr_time is "-")
        return {"error": f"No real-time schedule data for {line}-{station} right now; "
                         "this does not mean no trains are running."}
    try:
        names = {r["Station Code"]: (r["English Name"], r["Chinese Name"]) for r in await _stations()}
    except httpx.HTTPError:
        names = {}  # destination codes alone are still usable
    return {
        "line": line,
        "station": station,
        "is_delayed": d.get("isdelay") == "Y",
        "up": _fmt(sched.get("UP", []), names),
        "down": _fmt(sched.get("DOWN", []), names),
        "updated": d.get("curr_time"),
    }
