"""Calibrate the LLM judge against human labels.

"Who judges the judge?" — you label a sample of answers BLIND (judge verdict hidden),
then measure how often the judge agrees with you.

Usage:
    uv run python -m evals.calibrate label  evals/results/<trajectory run>.json --limit 20
    uv run python -m evals.calibrate report evals/results/<trajectory run>.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

LABELS_DIR = Path(__file__).with_name("labels")


# ---------- data ----------

def key(r: dict) -> str:
    return f"{r['id']}#{r.get('run', 0)}"


def judge_ok(r: dict) -> bool:
    return not r.get("judge_problems")


def judgeable(results: list[dict]) -> list[dict]:
    """Only answers the judge actually graded (skips errors, unfinished runs, --no-judge)."""
    return [r for r in results if r.get("judge") and "tool_log" in r]


def select_sample(results: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """Balance judge-pass / judge-fail so agreement isn't inflated by one dominant class,
    then shuffle so the labeller can't infer the verdict from position."""
    rng = random.Random(seed)
    fails = [r for r in results if not judge_ok(r)]
    passes = [r for r in results if judge_ok(r)]
    rng.shuffle(fails)
    rng.shuffle(passes)
    n_fail = min(len(fails), limit // 2)
    sample = fails[:n_fail] + passes[: limit - n_fail]
    if len(sample) < limit:  # not enough passes: top up with remaining fails
        sample += fails[n_fail : n_fail + limit - len(sample)]
    rng.shuffle(sample)
    return sample


def labels_path(results_path: Path) -> Path:
    return LABELS_DIR / f"{results_path.stem}.jsonl"


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["key"]: row for row in rows}  # last label wins, so re-labelling is just appending


# ---------- metrics ----------

def confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """pairs = (human_ok, judge_ok). Human is ground truth."""
    return {
        "both_pass": sum(h and j for h, j in pairs),
        "both_fail": sum(not h and not j for h, j in pairs),
        "judge_too_lenient": sum(not h and j for h, j in pairs),  # the dangerous one
        "judge_too_strict": sum(h and not j for h, j in pairs),
    }


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Agreement corrected for chance. >0.8 strong, 0.6–0.8 substantial, <0.4 don't trust the judge."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(h == j for h, j in pairs) / n
    ph, pj = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    return None if pe == 1 else (po - pe) / (1 - pe)


# ---------- label ----------

def _show(r: dict, i: int, total: int) -> None:
    w = 100
    print("\n" + "═" * w)
    print(f"[{i}/{total}] {key(r)}")
    print(f"問題：{r['query']}\n")
    print("工具結果：")
    for c in r["tool_log"]:
        out = json.dumps(c["result"], ensure_ascii=False)
        print(f"  • {c['tool']}({json.dumps(c['args'], ensure_ascii=False)})")
        print(textwrap.indent(textwrap.shorten(out, 600, placeholder=" …"), "      "))
    if not r["tool_log"]:
        print("  （冇 call tool）")
    print(f"\n答案：\n{textwrap.indent(r.get('answer') or '(空)', '  ')}\n")
    if r.get("facts"):
        print("必須包含：\n" + "\n".join(f"  ✓ {f}" for f in r["facts"]))
    if r.get("must_not_claim"):
        print("唔可以講：\n" + "\n".join(f"  ✗ {f}" for f in r["must_not_claim"]))


def label(results_path: Path, limit: int, seed: int, ask: Callable[[str], str] = input) -> int:
    results = judgeable(json.loads(results_path.read_text(encoding="utf-8"))["results"])
    out = labels_path(results_path)
    done = load_labels(out)
    todo = [r for r in select_sample(results, limit, seed) if key(r) not in done]
    if not todo:
        print(f"Nothing to label ({len(done)} already in {out}).")
        return 0

    print(f"{len(todo)} to label. Judge verdicts are hidden. y = 合格, n = 唔合格, s = skip, q = quit")
    out.parent.mkdir(exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            _show(r, i, len(todo))
            while (ans := ask("答案合格？[y/n/s/q] ").strip().lower()) not in {"y", "n", "s", "q"}:
                pass
            if ans == "q":
                break
            if ans == "s":
                continue
            note = ask("原因（optional，Enter skip）：").strip()
            f.write(json.dumps({"key": key(r), "human_ok": ans == "y", "note": note}, ensure_ascii=False) + "\n")
            f.flush()  # safe to Ctrl+C anytime
    print(f"\nSaved → {out}")
    return 0


# ---------- report ----------

def report(results_path: Path) -> int:
    by_key = {key(r): r for r in judgeable(json.loads(results_path.read_text(encoding="utf-8"))["results"])}
    labels = load_labels(labels_path(results_path))
    rows = [(lab, by_key[k]) for k, lab in labels.items() if k in by_key]
    if not rows:
        print("No labels yet. Run `label` first.")
        return 1

    pairs = [(lab["human_ok"], judge_ok(r)) for lab, r in rows]
    agree = sum(h == j for h, j in pairs) / len(pairs)
    kappa = cohens_kappa(pairs)
    cm = confusion(pairs)

    print(f"\nLabelled:   {len(pairs)}")
    print(f"Agreement:  {agree:.0%}")
    print(f"Cohen's κ:  {'n/a (all one class)' if kappa is None else f'{kappa:.2f}'}")
    print("\n                 judge PASS   judge FAIL")
    print(f"  human PASS     {cm['both_pass']:>6}       {cm['judge_too_strict']:>6}")
    print(f"  human FAIL     {cm['judge_too_lenient']:>6}       {cm['both_fail']:>6}")

    if cm["judge_too_lenient"]:
        print("\n⚠ Judge passed answers you rejected — fix these first, they hide real bugs:")
    for lab, r in rows:
        h, j = lab["human_ok"], judge_ok(r)
        if h != j:
            tag = "TOO LENIENT" if j else "TOO STRICT"
            print(f"\n  [{tag}] {key(r)}: {r['query']}")
            print(f"    answer: {textwrap.shorten(r.get('answer') or '', 140, placeholder=' …')}")
            print(f"    judge:  {r['judge_problems'] or r['judge'].get('reason')}")
            if lab.get("note"):
                print(f"    you:    {lab['note']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("label", help="blind-label a balanced sample")
    lp.add_argument("results", type=Path)
    lp.add_argument("--limit", type=int, default=20)
    lp.add_argument("--seed", type=int, default=0)
    rp = sub.add_parser("report", help="judge vs human agreement")
    rp.add_argument("results", type=Path)
    a = p.parse_args()
    return label(a.results, a.limit, a.seed) if a.cmd == "label" else report(a.results)


if __name__ == "__main__":
    sys.exit(main())
