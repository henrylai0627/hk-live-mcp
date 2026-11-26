"""Shared HTTP helpers. One place to tune timeouts / retries later."""
import httpx

_TIMEOUT = httpx.Timeout(10.0)
_HEADERS = {"User-Agent": "hk-live-mcp/0.1 (+https://github.com/henry5b17/hk-live-mcp)"}


async def get_json(url: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            # e.g. HKO answers 200 + an HTML page for invalid parameters
            raise RuntimeError(f"{r.url.host} returned a non-JSON response") from None


async def get_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
        r = await client.get(url)
        r.raise_for_status()
        # MTR CSV ships with a UTF-8 BOM
        return r.content.decode("utf-8-sig")
