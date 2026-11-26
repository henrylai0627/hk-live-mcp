"""Tool-selection eval: given a user query, does the model call the right tool with the right args?

Single-turn: only the FIRST tool call (or the absence of one) is graded.
Usage:
    uv run python -m evals.tool_selection --dry-run
    uv run python -m evals.tool_selection --model claude-haiku-5-5 --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from hk_live_mcp.server import mcp

CASES_PATH = Path(__file__).with_name("cases.json")
RESULTS_DIR = Path(__file__).with_name("results")
SYSTEM = "你係一個香港生活助手。需要實時數據（天氣、港鐵、巴士）就用工具；唔需要就直接答。"

ToolCall = tuple[str, dict[str, Any]]


# ---------- grading (pure functions, unit-tested) ----------

def _norm(v: Any) -> str:
    return str(v).strip().lower()


def value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):  # list = any of these is acceptable
        return any(value_matches(e, actual) for e in expected)
    if isinstance(expected, str):
        # Case-insensitive substring: "tkl" matches "TKL", "將軍澳" matches "將軍澳站"
        return actual is not None and _norm(expected) in _norm(actual)
    return expected == actual


def call_matches(expected: dict, call: ToolCall) -> bool:
    name, args = call
    if expected["tool"] != name:
        return False
    return all(value_matches(v, args.get(k)) for k, v in expected.get("args", {}).items())


def grade(case: dict, call: ToolCall | None) -> bool:
    accepted = case["expect"]  # [] means the model should answer WITHOUT a tool
    if not accepted:
        return call is None
    return call is not None and any(call_matches(e, call) for e in accepted)


# ---------- setup ----------

async def anthropic_tools() -> list[dict]:
    """Expose the real MCP tool definitions, so the eval tests exactly what clients see."""
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in await mcp.list_tools()
    ]


def validate(cases: list[dict], tools: list[dict]) -> list[str]:
    """Catch typos in cases.json before spending API money."""
    props = {t["name"]: t["input_schema"].get("properties", {}) for t in tools}
    errors, seen = [], set()
    for c in cases:
        if c["id"] in seen:
            errors.append(f"{c['id']}: duplicate id")
        seen.add(c["id"])
        for e in c["expect"]:
            if e["tool"] not in props:
                errors.append(f"{c['id']}: unknown tool {e['tool']}")
                continue
            errors += [f"{c['id']}: {e['tool']} has no arg '{k}'"
                       for k in e.get("args", {}) if k not in props[e["tool"]]]
    return errors


# ---------- run ----------

async def run_case(client, model: str, tools: list[dict], case: dict, sem: asyncio.Semaphore) -> dict:
    base = {"id": case["id"], "category": case["category"], "query": case["query"]}
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.messages.create(
                model=model, max_tokens=1024, system=SYSTEM, tools=tools,
                messages=[{"role": "user", "content": case["query"]}],
            )
        except Exception as e:  # noqa: BLE001 — one failing call shouldn't kill the whole run
            return {**base, "passed": False, "error": repr(e)}
        latency = time.perf_counter() - t0
    call = next(((b.name, b.input) for b in r.content if b.type == "tool_use"), None)
    return {
        **base,
        "passed": grade(case, call),
        "tool": call[0] if call else None,
        "args": call[1] if call else None,
        "expected": case["expect"],
        "latency_s": round(latency, 2),
        "input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
    }


def report(results: list[dict], runs: int) -> float:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])
        by_case[r["id"]].append(r)

    rate = sum(r["passed"] for r in results) / len(results)
    print(f"\nOverall pass rate: {rate:.1%}  ({sum(r['passed'] for r in results)}/{len(results)})")
    print("\nBy category:")
    for cat, ps in sorted(by_cat.items()):
        print(f"  {cat:<16} {sum(ps):>3}/{len(ps):<3} {sum(ps) / len(ps):.0%}")

    lat = [r["latency_s"] for r in results if "latency_s" in r]
    tok = [r["input_tokens"] + r["output_tokens"] for r in results if "input_tokens" in r]
    if lat:
        print(f"\nAvg latency: {sum(lat) / len(lat):.2f}s   Avg tokens/call: {sum(tok) / len(tok):.0f}")

    failures = [(cid, rs) for cid, rs in by_case.items() if not all(r["passed"] for r in rs)]
    if failures:
        print("\nFailures" + (" (passes/runs)" if runs > 1 else "") + ":")
        for cid, rs in failures:
            r = rs[-1]
            n = sum(x["passed"] for x in rs)
            flaky = f" [{n}/{runs}{' FLAKY' if 0 < n < runs else ''}]" if runs > 1 else ""
            got = r.get("error") or f"{r['tool']}({json.dumps(r['args'], ensure_ascii=False)})"
            print(f"  ✗ {cid}{flaky}: {r['query']}\n      got: {got}")
    return rate


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--runs", type=int, default=1, help="repeat each case to measure consistency")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--category", help="only run one category")
    p.add_argument("--threshold", type=float, default=0.0, help="exit 1 if pass rate below (for CI)")
    p.add_argument("--dry-run", action="store_true", help="validate cases.json only")
    a = p.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if a.category:
        cases = [c for c in cases if c["category"] == a.category]
    tools = await anthropic_tools()

    if errors := validate(cases, tools):
        print("\n".join(errors))
        return 1
    if a.dry_run:
        print(f"OK: {len(cases)} cases, {len(tools)} tools")
        return 0

    from anthropic import AsyncAnthropic  # imported late so --dry-run needs no API key

    client, sem = AsyncAnthropic(), asyncio.Semaphore(a.concurrency)
    print(f"Running {len(cases)} cases × {a.runs} on {a.model} ...")
    results = await asyncio.gather(
        *(run_case(client, a.model, tools, c, sem) for _ in range(a.runs) for c in cases)
    )
    rate = report(results, a.runs)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{a.model}.json"
    out.write_text(json.dumps(
        {"model": a.model, "runs": a.runs, "pass_rate": rate, "results": results},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(f"\nSaved → {out}")
    return 0 if rate >= a.threshold else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
