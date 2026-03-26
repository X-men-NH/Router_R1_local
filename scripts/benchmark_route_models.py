import json
import os
import time
from openai import OpenAI

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-nemo",
    "google/gemma-2-27b-it",
]

PROMPT = "Reply with exactly: ok"


def test_model(model_name):
    start = time.perf_counter()
    try:
        client = OpenAI(
            base_url=os.environ["OPENROUTER_API_BASE"],
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=20,
            max_retries=0,
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0,
            max_tokens=8,
        )
        content = (resp.choices[0].message.content or "").strip()
        return {
            "model": model_name,
            "ok": content.lower().startswith("ok"),
            "latency_sec": round(time.perf_counter() - start, 3),
            "content": content,
        }
    except Exception as exc:
        return {
            "model": model_name,
            "ok": False,
            "latency_sec": round(time.perf_counter() - start, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    results = [test_model(model) for model in MODELS]
    print(json.dumps(results, ensure_ascii=False))
