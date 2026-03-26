import json
from collections import Counter
from pathlib import Path


LOG_PATH = Path("/tmp/train_3b_launch.log")
MODELS = [
    "mistralai/mistral-nemo",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemma-2-27b-it",
    "writer/palmyra-creative-122b",
    "mistralai/mixtral-8x22b-instruct",
]


def main():
    lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    counts = Counter()
    samples = []

    for index, line in enumerate(lines):
        if "Request timed out" not in line and "Reach MAX_TRIALS" not in line:
            continue

        window = "\n".join(lines[max(0, index - 3): min(len(lines), index + 4)])
        matched = next((model for model in MODELS if model in window), "UNKNOWN")
        counts[matched] += 1

        if len(samples) < 20:
            samples.append(
                {
                    "line": index + 1,
                    "model": matched,
                    "text": line.strip(),
                }
            )

    print(json.dumps({"counts": counts, "samples": samples}, ensure_ascii=False))


if __name__ == "__main__":
    main()