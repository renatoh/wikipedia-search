"""
Run the agentic search over benchmark/questions.csv and record, per question:
the answer, timing (per LLM turn, per tool call, and total), tool-call counts,
and which articles the agent judged relevant.

Imports AgenticSearch directly (no HTTP / web endpoint) — one fresh agent per
question so there's no cross-question history bleed.

Usage:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --limit 10        # first 10 only
    python benchmark/run_benchmark.py --out results.csv
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make scripts/ importable so `from agents import AgenticSearch` works.
BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agents import AgenticSearch  # noqa: E402

QUESTIONS = BENCH_DIR / "questions.csv"

RESULT_FIELDS = [
    "id",
    "era",
    "complexity",
    "question",
    "answer",
    "total_seconds",
    "llm_calls",
    "llm_seconds",
    "tool_calls",
    "tool_seconds",
    "search_calls",
    "get_calls",
    "relevant_articles",
    "error",
]


def summarize_trace(trace: list[dict]) -> dict:
    """Aggregate per-action timings from an agent's last_trace."""
    llm_calls = [e for e in trace if e.get("kind") == "llm"]
    tool_calls = [e for e in trace if e.get("kind") == "tool"]
    return {
        "llm_calls": len(llm_calls),
        "llm_seconds": round(sum(e.get("seconds", 0) for e in llm_calls), 3),
        "tool_calls": len(tool_calls),
        "tool_seconds": round(sum(e.get("seconds", 0) for e in tool_calls), 3),
        "search_calls": sum(1 for e in tool_calls if e.get("name") == "wikipedia_search"),
        "get_calls": sum(1 for e in tool_calls if e.get("name") == "wikipedia_get"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="run only the first N questions")
    parser.add_argument("--out", default=None, help="output CSV path")
    parser.add_argument("--retries", type=int, default=3, help="attempts per question on transient errors")
    args = parser.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else BENCH_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    with open(QUESTIONS, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    if args.limit:
        questions = questions[: args.limit]

    print(f"running {len(questions)} question(s) → {out_path}\n")
    run_start = time.perf_counter()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        for q in questions:
            qid, question = q["id"], q["question"]
            print(f"[{qid}] {question}")
            row = {
                "id": qid,
                "era": q.get("era", ""),
                "complexity": q.get("complexity", ""),
                "question": question,
            }
            # Retry transient failures (e.g. RunPod proxy hiccups, model
            # reload races that surface as 404s). Fresh agent each attempt.
            last_err = None
            for attempt in range(1, args.retries + 1):
                try:
                    agent = AgenticSearch()  # fresh agent — no shared history
                    answer = agent.ask(question, verbose=False)
                    stats = summarize_trace(agent.last_trace)
                    row.update(
                        answer=answer,
                        total_seconds=agent.last_total_seconds,
                        relevant_articles=json.dumps(agent.relevant_articles, ensure_ascii=False),
                        error="",
                        **stats,
                    )
                    print(
                        f"     {agent.last_total_seconds:.1f}s total | "
                        f"{stats['llm_calls']} llm ({stats['llm_seconds']:.1f}s) | "
                        f"{stats['tool_calls']} tools ({stats['tool_seconds']:.1f}s)"
                        f"{f'  (attempt {attempt})' if attempt > 1 else ''}\n"
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"     attempt {attempt}/{args.retries} failed: {type(e).__name__}: {e}")
                    if attempt < args.retries:
                        time.sleep(min(2 ** attempt, 15))  # backoff: 2s, 4s, 8s...
            if last_err is not None:
                row.update(answer="", error=f"{type(last_err).__name__}: {last_err}")
                print(f"     ERROR (gave up after {args.retries}): {type(last_err).__name__}: {last_err}\n")

            writer.writerow(row)
            f.flush()  # write incrementally so a crash keeps partial results

    print(f"done. {len(questions)} question(s) in {time.perf_counter() - run_start:.1f}s")
    print(f"results: {out_path}")


if __name__ == "__main__":
    main()
