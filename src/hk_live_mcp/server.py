"""hk-live MCP server.

Tool docstrings are what the model reads to decide WHEN and HOW to call a tool,
so they are written for the model, not for humans.
"""
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .clients import hko, kmb, mtr
from .clients.hko import Lang

mcp = MCPServer(
    "hk-live",
    instructions="Real-time Hong Kong public data: HKO weather, MTR next train, KMB bus ETA.",
)
# Tools return dicts, never bare lists: an empty list reaches the client as an empty
# result with no text, which the model cannot tell apart from a failed call.
# Every tool is a read-only call to a public API; hint lets clients skip confirmation prompts
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


@mcp.tool(annotations=READ_ONLY)
async def get_current_weather(district: str | None = None, lang: Lang = "tc") -> dict:
    """Current Hong Kong weather from the Hong Kong Observatory: temperature by
    district, humidity, past-hour rainfall, UV index and any active warning messages.
    `rainfall_mm` lists the matched district, otherwise only districts where it
    rained; an empty list means no rain was recorded. A `note` field means the
    district matched no temperature station and all stations are returned.

    Args:
        district: Optional district/station name to filter temperatures, e.g. "Sha Tin"
            or "沙田" (must match the `lang` used).
        lang: "tc" (Traditional Chinese), "en" or "sc".
    """
    return await hko.current(lang, district)


@mcp.tool(annotations=READ_ONLY)
async def get_weather_forecast(days: int = 9, lang: Lang = "tc") -> dict:
    """Hong Kong Observatory forecast for the next 1-9 days: weather, wind,
    min/max temperature and probability of rain."""
    return await hko.forecast(days, lang)


@mcp.tool(annotations=READ_ONLY)
async def get_weather_warnings(lang: Lang = "tc") -> dict:
    """Active Hong Kong weather warnings (typhoon signals, rainstorm, hot/cold, etc.).
    `warnings` is an empty list when no warnings are in force."""
    return {"warnings": await hko.warnings(lang)}


@mcp.tool(annotations=READ_ONLY)
async def find_mtr_station(query: str) -> dict:
    """Look up MTR line and station codes by station name (English or Chinese)
    or by exact 3-letter station code.
    Call this BEFORE get_mtr_next_train if you don't already know the codes.
    A station on multiple lines returns one row per line in `stations`."""
    return {"stations": await mtr.find_station(query)}


@mcp.tool(annotations=READ_ONLY)
async def get_mtr_next_train(line: str, station: str) -> dict:
    """Next MTR trains in both directions for a station on a line.
    On the East Rail Line a train may carry `route` "RAC" (runs via Racecourse
    station) and `time_type` "A" (arrival time) or "D" (departure time).

    Args:
        line: Line code, e.g. TKL, ISL, TWL, KTL, EAL, TML, SIL, TCL, AEL, DRL.
        station: 3-letter station code, e.g. TKO, ADM. Use find_mtr_station if unsure.
    """
    return await mtr.next_train(line, station)


@mcp.tool(annotations=READ_ONLY)
async def find_kmb_stop(query: str, limit: int = 10) -> dict:
    """Search KMB bus stops by name (English or Chinese). Many stops share a name
    (one per direction), so results include coordinates to tell them apart."""
    return {"stops": await kmb.find_stop(query, limit)}


@mcp.tool(annotations=READ_ONLY)
async def get_kmb_eta(stop_id: str, route: str | None = None) -> dict:
    """Real-time KMB bus arrivals at a stop, in minutes from now.
    A row with `minutes` null is a route that serves the stop but has no bus
    scheduled right now (see `remark`). Empty `arrivals` means an unknown stop ID.
    `direction` is "O" (outbound) or "I" (inbound). `stop_seq` 1 means the bus
    starts its journey here; a bus whose `dest` is this stop is terminating
    here and cannot be boarded.

    Args:
        stop_id: KMB stop ID from find_kmb_stop.
        route: Optional route number to filter, e.g. "1A".
    """
    return {"arrivals": await kmb.stop_eta(stop_id, route)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
