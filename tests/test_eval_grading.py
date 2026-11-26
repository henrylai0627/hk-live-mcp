import json

from evals.tool_selection import CASES_PATH, anthropic_tools, grade, validate, value_matches


def test_value_matches_substring_and_any_of():
    assert value_matches("tkl", "TKL")
    assert value_matches("將軍澳", "將軍澳站")
    assert value_matches(["北角", "north point"], "North Point")
    assert not value_matches("TKO", None)


def test_grade_accepts_alternative_calls():
    case = {"expect": [
        {"tool": "find_mtr_station", "args": {"query": ["將軍澳", "tseung kwan o"]}},
        {"tool": "get_mtr_next_train", "args": {"line": "TKL", "station": "TKO"}},
    ]}
    assert grade(case, ("find_mtr_station", {"query": "將軍澳"}))
    assert grade(case, ("get_mtr_next_train", {"line": "tkl", "station": "tko"}))
    assert not grade(case, ("get_mtr_next_train", {"line": "TKL", "station": "POA"}))
    assert not grade(case, None)


def test_grade_no_tool_case():
    assert grade({"expect": []}, None)
    assert not grade({"expect": []}, ("get_current_weather", {}))


async def test_cases_file_matches_real_tool_schemas():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert validate(cases, await anthropic_tools()) == []
