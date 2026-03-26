import argparse
import json
import os
import statistics
import time

from openai import OpenAI


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt-file")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: ok",
    )
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as file:
            prompt = file.read()

    client = OpenAI(
        base_url=os.environ["OPENROUTER_API_BASE"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=args.timeout,
        max_retries=0,
    )

    results = []
    for round_id in range(1, args.rounds + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            latency = time.perf_counter() - started
            content = (response.choices[0].message.content or "").strip()
            results.append(
                {
                    "round": round_id,
                    "ok": True,
                    "latency_sec": round(latency, 3),
                    "content": content,
                }
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            results.append(
                {
                    "round": round_id,
                    "ok": False,
                    "latency_sec": round(latency, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    success_latencies = [item["latency_sec"] for item in results if item["ok"]]
    summary = {
        "model": args.model,
        "rounds": args.rounds,
        "successes": sum(1 for item in results if item["ok"]),
        "failures": sum(1 for item in results if not item["ok"]),
        "avg_latency_sec": round(statistics.mean(success_latencies), 3) if success_latencies else None,
        "p50_latency_sec": round(percentile(success_latencies, 0.50), 3) if success_latencies else None,
        "p95_latency_sec": round(percentile(success_latencies, 0.95), 3) if success_latencies else None,
    }
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()