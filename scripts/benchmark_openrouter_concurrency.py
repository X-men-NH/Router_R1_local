import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def percentile(values, q):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def make_client(timeout):
    return OpenAI(
        base_url=os.environ["OPENROUTER_API_BASE"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=timeout,
        max_retries=0,
    )


def run_single_request(request_id, model, timeout, max_tokens):
    start = time.perf_counter()
    thread_name = threading.current_thread().name
    try:
        client = make_client(timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Reply with exactly: ok_{request_id}",
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        content = (response.choices[0].message.content or "").strip()
        ok = content.startswith(f"ok_{request_id}")
        return {
            "ok": ok,
            "status": "ok" if ok else "bad_content",
            "latency_sec": time.perf_counter() - start,
            "thread": thread_name,
            "content": content,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": type(exc).__name__,
            "latency_sec": time.perf_counter() - start,
            "thread": thread_name,
            "error": str(exc),
        }


def run_level(concurrency, requests_per_level, model, timeout, max_tokens):
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single_request, request_id, model, timeout, max_tokens)
            for request_id in range(requests_per_level)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - started
    latencies = [item["latency_sec"] for item in results]
    successes = [item for item in results if item["ok"]]
    failures = [item for item in results if not item["ok"]]
    failure_kinds = {}
    for item in failures:
        failure_kinds[item["status"]] = failure_kinds.get(item["status"], 0) + 1

    summary = {
        "concurrency": concurrency,
        "requests": requests_per_level,
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": len(successes) / len(results) if results else 0.0,
        "elapsed_sec": elapsed,
        "throughput_rps": len(results) / elapsed if elapsed > 0 else 0.0,
        "latency_avg_sec": statistics.mean(latencies) if latencies else None,
        "latency_p50_sec": percentile(latencies, 0.50),
        "latency_p95_sec": percentile(latencies, 0.95),
        "failure_kinds": failure_kinds,
    }
    return summary, failures[:5]


def parse_levels(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="4,8,12,16,20,24,28,32")
    parser.add_argument("--requests-multiplier", type=float, default=1.5)
    parser.add_argument("--min-requests", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--stop-on-failure-rate", type=float, default=0.15)
    parser.add_argument("--stop-on-p95", type=float, default=18.0)
    args = parser.parse_args()

    for env_name in ["OPENROUTER_API_BASE", "OPENROUTER_API_KEY"]:
        if not os.environ.get(env_name):
            raise SystemExit(f"Missing required env var: {env_name}")

    levels = parse_levels(args.levels)
    all_summaries = []
    for concurrency in levels:
        requests_per_level = max(
            args.min_requests,
            min(args.max_requests, int(round(concurrency * args.requests_multiplier))),
        )
        summary, sample_failures = run_level(
            concurrency=concurrency,
            requests_per_level=requests_per_level,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        all_summaries.append(summary)
        print(json.dumps({"summary": summary, "sample_failures": sample_failures}, ensure_ascii=True))

        failure_rate = 1.0 - summary["success_rate"]
        p95 = summary["latency_p95_sec"] or 0.0
        if failure_rate >= args.stop_on_failure_rate or p95 >= args.stop_on_p95:
            break

    safe_candidates = [
        item for item in all_summaries
        if item["success_rate"] >= 0.95 and (item["latency_p95_sec"] or 0.0) < args.stop_on_p95
    ]
    recommended = safe_candidates[-1]["concurrency"] if safe_candidates else None
    print(json.dumps({
        "recommended_concurrency": recommended,
        "tested_levels": all_summaries,
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()