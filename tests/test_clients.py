import time
from datetime import datetime

import httpx
import pytest
import respx

from hk_live_mcp.clients import hko, kmb, mtr


@respx.mock
async def test_hko_current_handles_empty_warning_string():
    respx.get(hko.BASE).mock(return_value=httpx.Response(200, json={
        "updateTime": "2026-11-10T10:02:00+08:00",
        "temperature": {"data": [{"place": "沙田", "value": 24}, {"place": "京士柏", "value": 25}]},
        "humidity": {"data": [{"place": "香港天文台", "value": 70}]},
        "warningMessage": "",
    }))
    r = await hko.current(district="沙田")
    assert r["temperatures"] == [{"place": "沙田", "celsius": 24}]
    assert r["humidity_pct"] == 70
    assert r["warnings"] == []


_STATIONS = [
    {"Line Code": "TKL", "Station Code": "POA", "English Name": "Po Lam", "Chinese Name": "寶琳"},
    {"Line Code": "TKL", "Station Code": "TKO", "English Name": "Tseung Kwan O", "Chinese Name": "將軍澳"},
]


@pytest.fixture
def stations(monkeypatch):
    monkeypatch.setattr(mtr, "_cache", (time.time(), _STATIONS))


@respx.mock
async def test_mtr_next_train_parses_schedule(stations):
    respx.get(mtr.SCHEDULE).mock(return_value=httpx.Response(200, json={
        "status": 1, "isdelay": "N", "curr_time": "2026-11-10 10:00:00",
        "data": {"TKL-TKO": {
            "UP": [{"dest": "POA", "plat": "1", "time": "2026-11-10 10:02:00", "ttnt": "2"}],
            "DOWN": [{"dest": "NOP", "plat": "2", "time": "2026-11-10 10:03:00", "ttnt": "3"}],
        }},
    }))
    r = await mtr.next_train("tkl", "tko")
    assert r["is_delayed"] is False
    assert r["up"][0]["dest"] == "POA"
    assert (r["up"][0]["dest_name_en"], r["up"][0]["dest_name_tc"]) == ("Po Lam", "寶琳")
    assert r["down"][0]["minutes"] == 3
    assert "dest_name_en" not in r["down"][0]  # NOP is not in the station list
    assert "route" not in r["up"][0]


@respx.mock
async def test_kmb_eta_filters_route_and_keeps_null_as_no_eta():
    respx.get(f"{kmb.BASE}/stop-eta/ABC").mock(return_value=httpx.Response(200, json={"data": [
        {"route": "1A", "dest_tc": "尖沙咀", "eta": "2026-11-10T10:05:00+08:00", "rmk_tc": ""},
        {"route": "1A", "dest_tc": "尖沙咀", "eta": None, "rmk_tc": ""},
        {"route": "2", "dest_tc": "蘇屋", "eta": "2026-11-10T10:01:00+08:00", "rmk_tc": ""},
    ]}))
    now = datetime.fromisoformat("2026-11-10T10:00:00+08:00")
    r = await kmb.stop_eta("ABC", route="1a", now=now)
    assert [x["minutes"] for x in r] == [5, None]


@respx.mock
async def test_kmb_stop_list_failure_is_not_refetched_on_every_call(monkeypatch):
    monkeypatch.setattr(kmb, "_cache", None)
    monkeypatch.setattr(kmb, "_failed_at", 0.0)
    route = respx.get(f"{kmb.BASE}/stop").mock(return_value=httpx.Response(403, text="<html>"))
    for _ in range(3):
        with pytest.raises(RuntimeError, match="temporarily unavailable"):
            await kmb.find_stop("旺角")
    assert route.call_count == 1


@respx.mock
async def test_kmb_stop_list_failure_falls_back_to_stale_cache(monkeypatch):
    stale = [{"stop": "S1", "name_tc": "旺角站", "name_en": "MONG KOK STATION", "lat": "1", "long": "2"}]
    monkeypatch.setattr(kmb, "_cache", (0.0, stale))  # expired
    monkeypatch.setattr(kmb, "_failed_at", 0.0)
    respx.get(f"{kmb.BASE}/stop").mock(return_value=httpx.Response(403, text="<html>"))
    assert [s["stop_id"] for s in await kmb.find_stop("旺角")] == ["S1"]


@respx.mock
async def test_kmb_eta_dedupes_service_types_and_keeps_no_eta_routes():
    respx.get(f"{kmb.BASE}/stop-eta/ABC").mock(return_value=httpx.Response(200, json={"data": [
        {"route": "5C", "dir": "O", "service_type": 1, "dest_tc": "尖沙咀碼頭",
         "eta": "2026-11-10T10:05:00+08:00", "rmk_tc": ""},
        {"route": "5C", "dir": "O", "service_type": 2, "dest_tc": "尖沙咀碼頭",
         "eta": "2026-11-10T10:05:00+08:00", "rmk_tc": ""},
        {"route": "92R", "dir": "O", "service_type": 1, "dest_tc": "油麻地",
         "eta": None, "rmk_tc": "服務只限於星期六、日及公眾假期"},
    ]}))
    now = datetime.fromisoformat("2026-11-10T10:00:00+08:00")
    r = await kmb.stop_eta("abc", now=now)  # lowercase ID is normalised
    assert [(x["route"], x["minutes"]) for x in r] == [("5C", 5), ("92R", None)]
    assert r[1]["eta"] is None
    assert r[1]["remark"] == "服務只限於星期六、日及公眾假期"


@respx.mock
async def test_mtr_next_train_reports_missing_data_as_error():
    respx.get(mtr.SCHEDULE).mock(return_value=httpx.Response(200, json={
        "sys_time": "-", "curr_time": "-", "isdelay": "N", "status": 1, "message": "successful",
        "data": {"AEL-HOK": {"curr_time": "-", "sys_time": "-"}},
    }))
    r = await mtr.next_train("AEL", "HOK")
    assert "error" in r
    assert "up" not in r


@respx.mock
async def test_mtr_next_train_keeps_east_rail_fields(stations):
    respx.get(mtr.SCHEDULE).mock(return_value=httpx.Response(200, json={
        "status": 1, "isdelay": "N", "curr_time": "2026-11-10 10:00:00",
        "data": {"EAL-SHT": {"DOWN": [
            {"dest": "ADM", "plat": "3", "time": "2026-11-10 10:02:00", "ttnt": "2",
             "route": "RAC", "timeType": "D"},
            {"dest": "ADM", "plat": "3", "time": "2026-11-10 10:06:00", "ttnt": "",
             "route": "", "timeType": "A"},
        ]}},
    }))
    down = (await mtr.next_train("EAL", "SHT"))["down"]
    assert (down[0]["route"], down[0]["time_type"]) == ("RAC", "D")
    assert "route" not in down[1]
    assert down[1]["minutes"] is None


async def test_mtr_find_station_by_code(stations):
    assert [s["station"] for s in await mtr.find_station("tko")] == ["TKO"]
    assert await mtr.find_station("tk") == []  # codes match exactly, not by substring


_RHRREAD = {
    "updateTime": "2026-11-10T10:02:00+08:00",
    "temperature": {"data": [{"place": "Sha Tin", "value": 24}, {"place": "King's Park", "value": 25}]},
    "humidity": {"data": [{"place": "Hong Kong Observatory", "value": 70}]},
    "rainfall": {
        "data": [{"place": "Sha Tin", "max": 0}, {"place": "Central & Western District", "max": 3}],
        "startTime": "2026-11-10T08:45:00+08:00", "endTime": "2026-11-10T09:45:00+08:00",
    },
    "uvindex": {"data": [{"place": "King's Park", "value": 2, "desc": "low"}]},
    "tcmessage": "",
    "warningMessage": "",
}


@respx.mock
async def test_hko_current_includes_rainfall_and_uv():
    respx.get(hko.BASE).mock(return_value=httpx.Response(200, json=_RHRREAD))
    r = await hko.current("en")
    assert r["rainfall_mm"] == [{"place": "Central & Western District", "max": 3}]  # rain only
    assert r["uv_index"] == 2
    assert r["tropical_cyclone"] is None
    assert "note" not in r
    r = await hko.current("en", "sha tin")
    assert r["rainfall_mm"] == [{"place": "Sha Tin", "max": 0}]


@respx.mock
async def test_hko_current_flags_unmatched_district():
    respx.get(hko.BASE).mock(return_value=httpx.Response(200, json={**_RHRREAD, "uvindex": ""}))
    r = await hko.current("en", "Mong Kok")
    assert len(r["temperatures"]) == 2
    assert "Mong Kok" in r["note"]
    assert r["uv_index"] is None


@respx.mock
async def test_kmb_eta_exposes_direction_and_stop_seq():
    respx.get(f"{kmb.BASE}/stop-eta/ABC").mock(return_value=httpx.Response(200, json={"data": [
        {"route": "1", "dir": "I", "seq": 25, "dest_tc": "竹園邨",
         "eta": "2026-11-10T10:05:00+08:00", "rmk_tc": ""},
    ]}))
    r = await kmb.stop_eta("ABC", now=datetime.fromisoformat("2026-11-10T10:00:00+08:00"))
    assert (r[0]["direction"], r[0]["stop_seq"]) == ("I", 25)


async def test_empty_query_and_bad_limit(stations, monkeypatch):
    stop = {"stop": "S1", "name_tc": "旺角站", "name_en": "MONG KOK STATION", "lat": "1", "long": "2"}
    monkeypatch.setattr(kmb, "_cache", (time.time(), [stop, {**stop, "stop": "S2"}]))
    assert await mtr.find_station("  ") == []
    assert await kmb.find_stop("") == []
    assert len(await kmb.find_stop("旺角", -1)) == 1


@respx.mock
async def test_kmb_eta_sorts_routes_naturally():
    eta = "2026-11-10T10:05:00+08:00"
    respx.get(f"{kmb.BASE}/stop-eta/ABC").mock(return_value=httpx.Response(200, json={"data": [
        {"route": r, "dir": "O", "dest_tc": "x", "eta": eta, "rmk_tc": ""}
        for r in ("28", "HK1", "5", "1A", "1")
    ]}))
    r = await kmb.stop_eta("ABC", now=datetime.fromisoformat("2026-11-10T10:00:00+08:00"))
    assert [x["route"] for x in r] == ["1", "1A", "5", "28", "HK1"]


@respx.mock
async def test_non_json_response_raises_readable_error():
    respx.get(hko.BASE).mock(return_value=httpx.Response(200, text="Please include valid parameters"))
    with pytest.raises(RuntimeError, match="non-JSON"):
        await hko.current("xx")


@respx.mock
async def test_empty_list_tool_result_is_visible_to_the_model():
    from hk_live_mcp.server import mcp

    respx.get(hko.BASE).mock(return_value=httpx.Response(200, json={}))
    result = await mcp.call_tool("get_weather_warnings", {})
    assert '"warnings": []' in result.content[0].text
