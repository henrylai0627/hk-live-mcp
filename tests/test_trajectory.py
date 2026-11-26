import json
from types import SimpleNamespace as NS

from evals.tool_selection import anthropic_tools
from evals.trajectory import (
    CASES_PATH,
    grade_contains,
    grade_trajectory,
    is_subsequence,
    mock_tool,
    parse_judge,
    run_agent,
    validate,
)


def test_is_subsequence_order_and_any_of():
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])
    assert is_subsequence([["x", "b"]], ["a", "b"])
    assert is_subsequence([], [])


def test_mock_tool_matches_args_and_flags_unmocked():
    case = {"fixtures": [{"tool": "t", "args": {"line": "TKL"}, "result": 1}]}
    assert mock_tool(case, "t", {"line": "tkl"}) == (1, True)
    assert mock_tool(case, "t", {"line": "ISL"})[1] is False


def test_grade_trajectory_flags_problems():
    case = {"expect": {"must_call": ["a"], "must_not_call": ["b"], "max_tool_calls": 1}}
    run = {"finished": True, "calls": [{"tool": "b", "mocked": False}, {"tool": "a", "mocked": True}]}
    problems = " ".join(grade_trajectory(case, run))
    assert "forbidden" in problems and "> max" in problems and "no fixture" in problems


def test_grade_contains_any_of():
    case = {"expect": {"answer_contains": [["八號", "8號"]]}}
    assert grade_contains(case, "而家掛緊8號風球") == []
    assert grade_contains(case, "冇風球") != []


def test_parse_judge_tolerates_fences():
    assert parse_judge('```json\n{"grounded": true}\n```') == {"grounded": True}


class _Block(NS):
    def model_dump(self, **_):
        return {k: v for k, v in vars(self).items()}


async def test_run_agent_executes_tool_loop():
    replies = iter([
        NS(stop_reason="tool_use", usage=NS(input_tokens=10, output_tokens=5),
           content=[_Block(type="tool_use", id="1", name="get_current_weather", input={})]),
        NS(stop_reason="end_turn", usage=NS(input_tokens=20, output_tokens=8),
           content=[_Block(type="text", text="沙田 19 度")]),
    ])

    class Client:
        class messages:
            @staticmethod
            async def create(**_):
                return next(replies)

    case = {"query": "沙田幾度", "fixtures": [{"tool": "get_current_weather", "result": {"t": 19}}]}
    run = await run_agent(Client, "m", [], case, max_turns=4)
    assert run["finished"] and run["answer"] == "沙田 19 度"
    assert run["turns"] == 2 and run["input_tokens"] == 30
    assert run["calls"][0]["mocked"]


async def test_trajectory_cases_valid():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert validate(cases, await anthropic_tools()) == []
