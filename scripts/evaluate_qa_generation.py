#!/usr/bin/env python3
"""Evaluate generated QA parquet files for EM/F1 and decompose behavior.

Usage:
    python scripts/evaluate_qa_generation.py --input_parquet path/to/generated.parquet
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.reward_score.qa_em import compute_answer_metrics, extract_solution  # noqa: E402


SEARCH_MODEL_RE = re.compile(r"<search>\s*([^:<]+?):")


def ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def ensure_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if value is None:
        return ""
    return str(value)


def select_scoring_response(row, response_idx, raw_responses):
    action_responses = ensure_list(row.get("action_responses"))
    if response_idx < len(action_responses):
        action_response = ensure_text(action_responses[response_idx])
        if action_response:
            return action_response
    return ensure_text(raw_responses[response_idx])


def summarize_group(rows):
    def mean(key):
        vals = [row[key] for row in rows if key in row]
        return statistics.mean(vals) if vals else 0.0

    total = len(rows)
    any_decompose = sum(1 for row in rows if row["any_decompose"]) / total if total else 0.0
    first_decompose = sum(1 for row in rows if row["first_has_decompose"]) / total if total else 0.0

    return {
        "count": total,
        "first_em": mean("first_em"),
        "first_f1": mean("first_f1"),
        "best_em": mean("best_em"),
        "best_f1": mean("best_f1"),
        "any_decompose_rate": any_decompose,
        "first_decompose_rate": first_decompose,
        "first_searches": mean("first_search_count"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    args = parser.parse_args()

    parquet_path = Path(args.input_parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    rows = []
    model_counter = Counter()

    for _, row in df.iterrows():
        responses = ensure_list(row.get("responses"))
        reward_model = row.get("reward_model", {}) or {}
        ground_truth = reward_model.get("ground_truth", {}) or {}
        data_source = row.get("data_source", "unknown")

        response_stats = []
        for response_idx, response in enumerate(responses):
            response = ensure_text(response)
            scoring_response = select_scoring_response(row, response_idx, responses)
            answer = extract_solution(scoring_response)
            em, f1 = compute_answer_metrics(answer, ground_truth)
            has_decompose = "<decompose>" in scoring_response
            search_count = scoring_response.count("<search>")
            models = [m.strip() for m in SEARCH_MODEL_RE.findall(scoring_response)]
            response_stats.append({
                "response": response,
                "scoring_response": scoring_response,
                "answer": answer,
                "em": em,
                "f1": f1,
                "has_decompose": has_decompose,
                "search_count": search_count,
                "models": models,
            })
            model_counter.update(models)

        if not response_stats:
            response_stats.append({
                "response": "",
                "answer": None,
                "em": 0.0,
                "f1": 0.0,
                "has_decompose": False,
                "search_count": 0,
                "models": [],
            })

        first = response_stats[0]
        best = max(response_stats, key=lambda item: (item["em"], item["f1"]))

        rows.append({
            "data_source": data_source,
            "first_em": first["em"],
            "first_f1": first["f1"],
            "best_em": best["em"],
            "best_f1": best["f1"],
            "first_has_decompose": first["has_decompose"],
            "any_decompose": any(item["has_decompose"] for item in response_stats),
            "first_search_count": first["search_count"],
        })

    overall = summarize_group(rows)
    print("=== OVERALL ===")
    for key, value in overall.items():
        if key == "count":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["data_source"]].append(row)

    print("\n=== PER DATA SOURCE ===")
    for source, source_rows in sorted(by_source.items()):
        summary = summarize_group(source_rows)
        print(f"[{source}]")
        for key, value in summary.items():
            if key == "count":
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:.4f}")

        d_rows = [r for r in source_rows if r["first_has_decompose"]]
        nd_rows = [r for r in source_rows if not r["first_has_decompose"]]
        if d_rows:
            print(f"  first_decompose_em: {statistics.mean(r['first_em'] for r in d_rows):.4f}")
            print(f"  first_decompose_f1: {statistics.mean(r['first_f1'] for r in d_rows):.4f}")
        if nd_rows:
            print(f"  first_non_decompose_em: {statistics.mean(r['first_em'] for r in nd_rows):.4f}")
            print(f"  first_non_decompose_f1: {statistics.mean(r['first_f1'] for r in nd_rows):.4f}")

    if model_counter:
        total_models = sum(model_counter.values())
        print("\n=== MODEL USAGE ===")
        for model, count in model_counter.most_common():
            print(f"{model}: {count} ({count / total_models:.2%})")


if __name__ == "__main__":
    main()