# hk-live-mcp 🇭🇰

MCP server that gives Claude (or any MCP client) real-time Hong Kong public data:
**HKO weather**, **MTR next train**, and **KMB bus ETA**.

> "我而家喺將軍澳，落緊雨未？下一班車去北角幾時到？"
> → Claude calls `get_current_weather` + `find_mtr_station` + `get_mtr_next_train` and answers.

## Tools

| Tool | Source | Purpose |
|---|---|---|
| `get_current_weather` | HKO `rhrread` | Temperature by district, humidity, past-hour rainfall, UV index, warnings |
| `get_weather_forecast` | HKO `fnd` | 1–9 day forecast |
| `get_weather_warnings` | HKO `warnsum` | Typhoon / rainstorm signals in force |
| `find_mtr_station` | MTR stations CSV | Name or station code → line + station code |
| `get_mtr_next_train` | MTR Next Train API | Next 4 trains each direction, with destination names |
| `find_kmb_stop` | KMB `/stop` | Name → stop ID (+ coords) |
| `get_kmb_eta` | KMB `/stop-eta` | Arrivals in minutes, filterable by route; routes with no bus due keep their remark |

All tools are read-only and need no API key.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/henrylai0627/hk-live-mcp.git
cd hk-live-mcp
```

**With [uv](https://docs.astral.sh/uv/):**

```bash
uv sync
uv run pytest
uv run mcp dev src/hk_live_mcp/server.py   # MCP Inspector in browser
```

**With plain pip** (needs pip 25.1+ for `--group`):

```bash
python3 -m venv .venv
.venv/bin/pip install -e . --group dev
.venv/bin/pytest
.venv/bin/mcp dev src/hk_live_mcp/server.py   # MCP Inspector in browser
```

### Add to Claude Code

Use absolute paths; the server is started from whatever directory Claude Code runs in.

```bash
# uv
claude mcp add hk-live -- uv --directory /ABSOLUTE/PATH/hk-live-mcp run hk-live-mcp

# pip / venv
claude mcp add hk-live -- /ABSOLUTE/PATH/hk-live-mcp/.venv/bin/hk-live-mcp
```

By default the server is registered for the current project only; add `-s user` to make it
available in every project. Check it with `claude mcp get hk-live` (expect `Status: ✔ Connected`)
and remove it with `claude mcp remove hk-live`.

### Add to Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "hk-live": {
      "command": "uv",
      "args": ["--directory", "/ABSOLUTE/PATH/hk-live-mcp", "run", "hk-live-mcp"]
    }
  }
}
```

Without uv, set `"command"` to `/ABSOLUTE/PATH/hk-live-mcp/.venv/bin/hk-live-mcp` and `"args"` to `[]`.

## Design decisions

- **Trim API responses** – raw HKO/KMB payloads are verbose; tools return only what the model
  needs, cutting tokens per call.
- **Lookup tools before action tools** – `find_*` tools resolve human names to codes so the model
  doesn't guess station/stop IDs.
- **Daily cache for static lists** – KMB has ~6k stops; fetched once per 24h, not per call.
- **Docstrings written for the model** – they say *when* to call a tool, not just what it does.

## Evals

`evals/` checks whether a model picks the **right tool with the right args** for 30 real-world
queries (Cantonese + English), using the server's actual MCP tool definitions.

| Category | What it tests |
|---|---|
| `weather_now` / `forecast` / `warnings` | Choosing between similar weather tools |
| `mtr_by_name` | Lookup-first (`find_mtr_station`) **or** correct codes directly |
| `mtr_by_code` | Passing user-given codes straight through |
| `kmb` | Resolving stop names before fetching ETA |
| `no_tool` | **Not** calling a tool for general-knowledge questions |

```bash
uv sync --group evals
uv run python -m evals.tool_selection --dry-run              # validate cases, no API cost
export ANTHROPIC_API_KEY=...
uv run python -m evals.tool_selection --model claude-haiku-5-5 --runs 3
```

Without uv: `.venv/bin/pip install --group evals`, then use `.venv/bin/python` in place of
`uv run python`.

Each run is saved to `evals/results/`. `--runs N` repeats every case to expose flaky behaviour;
`--threshold 0.9` exits non-zero for CI.

### Trajectory eval (multi-turn)

`evals/trajectory.py` runs the **full agent loop** and grades both the path and the final answer.
Tools return **fixtures** instead of live data, so ground truth is known and edge cases
(T8 typhoon, empty ETA, API outage) are reproducible.

| Layer | Check |
|---|---|
| Trajectory (code) | required tools in order, forbidden tools, max tool calls, no unmocked calls |
| Answer (code) | `answer_contains` for must-have tokens (e.g. 八號) |
| Answer (LLM judge) | required facts present, no forbidden claims, every number **grounded** in tool output |

```bash
uv run python -m evals.trajectory --dry-run
uv run python -m evals.trajectory --model claude-haiku-5-5 --runs 3
uv run python -m evals.trajectory --only t08,t09 --no-judge   # quick, cheap
```

### Judge calibration

An LLM judge can be wrong too. `evals/calibrate.py` measures how much to trust it:

1. **Blind labelling**: the judge verdict is hidden while you grade; you see the question,
   tool outputs, and answer, and type y / n
2. **Balanced sample**: judge passes and fails are mixed 50/50, so agreement isn't inflated
3. **Report**: agreement %, **Cohen's κ**, and a confusion matrix highlighting
   *judge too lenient* cases (it passed something you rejected — these hide real bugs)

```bash
uv run python -m evals.calibrate label  evals/results/<trajectory run>.json --limit 20
uv run python -m evals.calibrate report evals/results/<trajectory run>.json
```

Labels are appended to `evals/labels/*.jsonl`; the tool resumes where you stopped.

### Results

| Eval | Model | Pass rate | Avg latency | Avg tokens |
|---|---|---|---|---|
| tool selection | _fill in_ | | | |
| trajectory | _fill in_ | | | |

| Judge | Labelled | Agreement | κ |
|---|---|---|---|
| _fill in_ | | | |

## Roadmap

- [ ] v0.2 Citybus ETA, GMB
- [x] v0.3 Tool-selection evals (single-turn)
- [x] v0.4 Trajectory evals: multi-turn + LLM-as-judge grounding
- [x] v0.4.1 Judge calibration: blind human labels, agreement + Cohen's κ
- [ ] v0.5 Remote deploy (streamable HTTP on Cloud Run)

## Data sources

[DATA.GOV.HK](https://data.gov.hk) · Hong Kong Observatory · MTR Corporation · KMB.
Subject to each provider's terms of use.

## License

MIT
