import json

import pytest

from evals import calibrate as cal


def _r(i, judge_pass, run=0):
    return {"id": f"t{i:02}", "run": run, "query": "q", "answer": "a", "tool_log": [],
            "judge": {"reason": "x"}, "judge_problems": [] if judge_pass else ["missing fact: y"]}


def test_kappa_perfect_and_chance():
    assert cal.cohens_kappa([(True, True), (False, False)]) == pytest.approx(1.0)
    assert cal.cohens_kappa([(True, True), (True, False), (False, True), (False, False)]) == pytest.approx(0.0)
    assert cal.cohens_kappa([(True, True)]) is None  # one class only


def test_confusion_flags_lenient_judge():
    cm = cal.confusion([(False, True), (True, True), (True, False)])
    assert cm == {"both_pass": 1, "both_fail": 0, "judge_too_lenient": 1, "judge_too_strict": 1}


def test_select_sample_is_balanced_and_deterministic():
    rs = [_r(i, judge_pass=i >= 3) for i in range(20)]  # 3 fails, 17 passes
    s = cal.select_sample(rs, limit=10, seed=1)
    assert len(s) == 10
    assert sum(not cal.judge_ok(r) for r in s) == 3
    assert [cal.key(r) for r in s] == [cal.key(r) for r in cal.select_sample(rs, limit=10, seed=1)]


def test_judgeable_skips_errors_and_unjudged():
    rs = [_r(1, True), {"id": "t02", "error": "boom"}, {**_r(3, True), "judge": {}}]
    assert [r["id"] for r in cal.judgeable(rs)] == ["t01"]


def test_label_then_report_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cal, "LABELS_DIR", tmp_path / "labels")
    src = tmp_path / "run.json"
    src.write_text(json.dumps({"results": [_r(1, True), _r(2, False)]}), encoding="utf-8")

    answers = iter(["n", "too vague", "y", ""])  # whatever order the shuffle gives
    cal.label(src, limit=2, seed=0, ask=lambda _: next(answers))
    labels = cal.load_labels(cal.labels_path(src))
    assert len(labels) == 2

    # Resumable: nothing left to label
    cal.label(src, limit=2, seed=0, ask=lambda _: pytest.fail("should not ask"))

    assert cal.report(src) == 0
    assert "Agreement" in capsys.readouterr().out
