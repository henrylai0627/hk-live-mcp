"""Trajectory eval: run the full agent loop, then grade the PATH and the FINAL ANSWER.

Tools are NOT called live. Each case ships fixtures (fake tool outputs), so the ground truth
is known and the eval is deterministic, free of API flakiness, and can test edge cases
(empty results, API errors, typhoon signals) on demand.

Usage:
    uv run python -m evals.trajectory --dry-run
    uv run python -m evals.trajectory --model claude-haiku-5-5 --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.tool_selection import anthropic_tools, value_matches

CASES_PATH = Path(__file__).with_name("trajectory_cases.json")
RESULTS_DIR = Path(__file__).with_name("results")
SYSTEM = (
    "你係一個香港生活助手。今日係 2026-11-10（星期二）。"
    "需要實時數據就用工具，答案只可以根據工具返回嘅數據，唔好估。用廣東話答，簡潔。"
)
NO_FIXTURE = {"error": "no data for this call"}


# ---------- tool mocking ----------

def mock_tool(case: dict, name: str, args: dict) -> tuple[Any, bool]:
    """Return (result, matched). First fixture whose tool + args match wins."""
    for f in case.get("fixtures", []):
        if f["tool"] == name and all(value_matches(v, args.get(k)) for k, v in f.get("args", {}).items()):
            return f["result"], True
    return NO_FIXTURE, False


# ---------- agent loop ----------

async def run_agent(client, model: str, tools: list[dict], case: dict, max_turns: int) -> dict:
    messages: list[dict] = [{"role": "user", "content": case["query"]}]
    calls: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    for turn in range(1, max_turns + 1):
        r = await client.messages.create(
            model=model, max_tokens=1024, system=SYSTEM, tools=tools, messages=messages
        )
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})

        if r.stop_reason != "tool_use":
            answer = "".join(b.text for b in r.content if b.type == "text").strip()
            return {"answer": answer, "calls": calls, "turns": turn, "finished": True, **usage}

        results = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            out, matched = mock_tool(case, b.name, b.input)
            calls.append({"tool": b.name, "args": b.input, "result": out, "mocked": matched})
            results.append({
                "type": "tool_result",
                "tool_use_id": b.id,
                "content": json.dumps(out, ensure_ascii=False),
            })
        messages.append({"role": "user", "content": results})

    return {"answer": None, "calls": calls, "turns": max_turns, "finished": False, **usage}


# ---------- grading ----------

def is_subsequence(required: list, called: list[str]) -> bool:
    """Each item (a tool name, or a list = any of) must appear in order, gaps allowed."""
    it = iter(called)
    return all(any(c in (req if isinstance(req, list) else [req]) for c in it) for req in required)


def grade_trajectory(case: dict, run: dict) -> list[str]:
    exp, names = case["expect"], [c["tool"] for c in run["calls"]]
    problems = []
    if not run["finished"]:
        problems.append("hit max turns without a final answer")
    if not is_subsequence(exp.get("must_call", []), names):
        problems.append(f"must_call {exp['must_call']} not satisfied by {names}")
    if bad := [n for n in names if n in exp.get("must_not_call", [])]:
        problems.append(f"called forbidden tool(s) {bad}")
    if len(names) > exp.get("max_tool_calls", 99):
        problems.append(f"{len(names)} tool calls > max {exp['max_tool_calls']}")
    if unmocked := [c["tool"] for c in run["calls"] if not c["mocked"]]:
        problems.append(f"unexpected call(s) with no fixture: {unmocked}")
    return problems


def grade_contains(case: dict, answer: str | None) -> list[str]:
    """Cheap deterministic checks. Each item is a string or a list (any of)."""
    answer = answer or ""
    return [
        f"answer missing any of {req}"
        for req in case["expect"].get("answer_contains", [])
        if not any(s in answer for s in (req if isinstance(req, list) else [req]))
    ]


JUDGE_PROMPT = """你係一個嚴格嘅評審。根據工具返回嘅數據，評核 AI 助手嘅最終答案。

<question>{query}</question>
<tool_log>{tool_log}</tool_log>
<answer>{answer}</answer>
<required_facts>{facts}</required_facts>
<forbidden_claims>{forbidden}</forbidden_claims>

規則：
- required_facts 每一項：答案有冇清楚講到（意思對就得，唔使字眼一樣）
- forbidden_claims 每一項：答案有冇出現呢種說法
- grounded：答案入面每個具體數字、時間、地點，都可以喺 tool_log 搵到出處先算 true

只輸出 JSON，唔好有其他文字：
{{"facts": [{{"fact": "...", "ok": true}}], "violations": ["..."], "grounded": true, "reason": "一句"}}"""


def parse_judge(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate ```json fences or stray prose
    if not m:
        raise ValueError(f"judge returned no JSON: {text[:200]}")
    return json.loads(m.group(0))


async def judge(client, model: str, case: dict, run: dict) -> tuple[list[str], dict]:
    exp = case["expect"]
    if not exp.get("facts") and not exp.get("must_not_claim") and not run["calls"]:
        return [], {}
    tool_log = json.dumps(
        [{"tool": c["tool"], "args": c["args"], "result": c["result"]} for c in run["calls"]],
        ensure_ascii=False,
    )
    r = await client.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            query=case["query"], tool_log=tool_log, answer=run["answer"],
            facts=json.dumps(exp.get("facts", []), ensure_ascii=False),
            forbidden=json.dumps(exp.get("must_not_claim", []), ensure_ascii=False),
        )}],
    )
    verdict = parse_judge("".join(b.text for b in r.content if b.type == "text"))
    problems = [f"missing fact: {f['fact']}" for f in verdict.get("facts", []) if not f.get("ok")]
    problems += [f"forbidden claim: {v}" for v in verdict.get("violations", [])]
    if run["calls"] and not verdict.get("grounded", False):
        problems.append(f"not grounded: {verdict.get('reason')}")
    return problems, verdict


# ---------- run ----------

async def run_case(client, a, tools, case, sem, run: int = 0) -> dict:
    base = {"id": case["id"], "run": run, "query": case["query"]}
    async with sem:
        t0 = time.perf_counter()
        try:
            run = await run_agent(client, a.model, tools, case, a.max_turns)
            traj = grade_trajectory(case, run)
            ans = grade_contains(case, run["answer"])
            verdict: dict = {}
            judge_problems: list[str] = []
            if run["finished"] and not a.no_judge:
                judge_problems, verdict = await judge(client, a.judge_model, case, run)
                ans += judge_problems
        except Exception as e:  # noqa: BLE001 — record and keep going
            return {**base, "passed": False, "trajectory_ok": False, "answer_ok": False, "error": repr(e)}
    return {
        **base,
        "passed": not traj and not ans,
        "trajectory_ok": not traj,
        "answer_ok": not ans,
        "problems": traj + ans,
        "answer": run["answer"],
        "tools_called": [c["tool"] for c in run["calls"]],
        "turns": run["turns"],
        "input_tokens": run["input_tokens"],
        "output_tokens": run["output_tokens"],
        "latency_s": round(time.perf_counter() - t0, 2),
        "judge": verdict,
        "judge_problems": judge_problems,
        # Kept so humans can re-grade later without re-running (see evals/calibrate.py)
        "facts": case["expect"].get("facts", []),
        "must_not_claim": case["expect"].get("must_not_claim", []),
        "tool_log": [{"tool": c["tool"], "args": c["args"], "result": c["result"]} for c in run["calls"]],
    }


def report(results: list[dict]) -> float:
    n = len(results)

    def pct(key: str) -> str:
        k = sum(r.get(key, False) for r in results)
        return f"{k / n:.0%} ({k}/{n})"

    ok = [r for r in results if "turns" in r]
    print(f"\nPassed:          {pct('passed')}")
    print(f"Trajectory OK:   {pct('trajectory_ok')}")
    print(f"Answer OK:       {pct('answer_ok')}")
    if ok:
        avg = lambda f: sum(f(r) for r in ok) / len(ok)
        print(f"Avg tool calls:  {avg(lambda r: len(r['tools_called'])):.1f}   "
              f"Avg turns: {avg(lambda r: r['turns']):.1f}   "
              f"Avg tokens: {avg(lambda r: r['input_tokens'] + r['output_tokens']):.0f}   "
              f"Avg latency: {avg(lambda r: r['latency_s']):.1f}s")

    for r in (r for r in results if not r["passed"]):
        print(f"\n  ✗ {r['id']}: {r['query']}")
        if "error" in r:
            print(f"      error: {r['error']}")
            continue
        print(f"      tools: {r['tools_called']}")
        for p in r["problems"]:
            print(f"      - {p}")
        print(f"      answer: {(r['answer'] or '')[:150]}")
    return sum(r["passed"] for r in results) / n


def validate(cases: list[dict], tools: list[dict]) -> list[str]:
    props = {t["name"]: t["input_schema"].get("properties", {}) for t in tools}
    errors, seen = [], set()

    def check_name(cid: str, name: str) -> None:
        if name not in props:
            errors.append(f"{cid}: unknown tool {name}")

    for c in cases:
        if c["id"] in seen:
            errors.append(f"{c['id']}: duplicate id")
        seen.add(c["id"])
        for f in c.get("fixtures", []):
            check_name(c["id"], f["tool"])
            errors += [f"{c['id']}: {f['tool']} has no arg '{k}'"
                       for k in f.get("args", {}) if f["tool"] in props and k not in props[f["tool"]]]
        exp = c["expect"]
        for req in exp.get("must_call", []):
            for name in req if isinstance(req, list) else [req]:
                check_name(c["id"], name)
        for name in exp.get("must_not_call", []):
            check_name(c["id"], name)
    return errors


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--judge-model", default="claude-sonnet-5-5")
    p.add_argument("--no-judge", action="store_true", help="trajectory + answer_contains only")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--only", help="comma-separated case ids")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if a.only:
        ids = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in ids]
    tools = await anthropic_tools()
    if errors := validate(cases, tools):
        print("\n".join(errors))
        return 1
    if a.dry_run:
        print(f"OK: {len(cases)} cases, {len(tools)} tools")
        return 0

    from anthropic import AsyncAnthropic

    client, sem = AsyncAnthropic(), asyncio.Semaphore(a.concurrency)
    print(f"Running {len(cases)} cases × {a.runs} | agent={a.model} judge={'off' if a.no_judge else a.judge_model}")
    results = await asyncio.gather(
        *(run_case(client, a, tools, c, sem, run=i) for i in range(a.runs) for c in cases)
    )
    rate = report(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_trajectory_{a.model}.json"
    out.write_text(json.dumps({"model": a.model, "judge_model": a.judge_model, "runs": a.runs,
                               "pass_rate": rate, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out}")
    return 0 if rate >= a.threshold else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
