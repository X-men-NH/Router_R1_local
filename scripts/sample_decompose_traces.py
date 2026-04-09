#!/usr/bin/env python3
"""Sample decompose trajectories from generated QA parquet outputs.

Usage:
    python scripts/sample_decompose_traces.py --input_parquet path/to/generated.parquet
    python scripts/sample_decompose_traces.py --input_parquet path/to/generated.parquet --limit 5 --mode random
"""

from __future__ import annotations

import argparse
import random
import re
import sys
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true", help="Print the full response text instead of truncating.")
    args = parser.parse_args()

    parquet_path = Path(args.input_parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    matches = []

    for row_idx, row in df.iterrows():
        responses = ensure_list(row.get("responses"))
        reward_model = row.get("reward_model", {}) or {}
        ground_truth = reward_model.get("ground_truth", {}) or {}
        data_source = row.get("data_source", "unknown")
        prompt = ensure_text(row.get("prompt", ""))

        for response_idx, response in enumerate(responses):
            response = ensure_text(response)
            scoring_response = select_scoring_response(row, response_idx, responses)
            if "<decompose>" not in scoring_response:
                continue

            answer = extract_solution(scoring_response)
            em, f1 = compute_answer_metrics(answer, ground_truth)
            matches.append({
                "row_idx": row_idx,
                "response_idx": response_idx,
                "data_source": data_source,
                "prompt": prompt,
                "response": response,
                "scoring_response": scoring_response,
                "answer": answer,
                "em": em,
                "f1": f1,
                "search_count": scoring_response.count("<search>"),
                "models": [m.strip() for m in SEARCH_MODEL_RE.findall(scoring_response)],
                "ground_truth": ground_truth.get("target", []),
            })

    print(f"INPUT: {parquet_path}")
    print(f"TOTAL_ROWS: {len(df)}")
    print(f"DECOMPOSE_RESPONSES: {len(matches)}")

    if not matches:
        return

    if args.mode == "random":
        rng = random.Random(args.seed)
        rng.shuffle(matches)

    samples = matches[: max(args.limit, 0)]

    for idx, item in enumerate(samples, start=1):
        print("\n" + "=" * 80)
        print(f"SAMPLE {idx}")
        print(f"row_idx: {item['row_idx']}")
        print(f"response_idx: {item['response_idx']}")
        print(f"data_source: {item['data_source']}")
        print(f"EM: {item['em']:.4f}")
        print(f"F1: {item['f1']:.4f}")
        print(f"search_count: {item['search_count']}")
        print(f"models: {item['models']}")
        print(f"ground_truth: {item['ground_truth']}")
        print(f"extracted_answer: {item['answer']}")
        print("PROMPT:")
        print(item["prompt"])
        print("ACTION_RESPONSE:")
        if args.full:
            print(item["scoring_response"])
        else:
            print(item["scoring_response"][:4000])
        print("RESPONSE:")
        if args.full:
            print(item["response"])
        else:
            print(item["response"][:4000])


if __name__ == "__main__":
    main()