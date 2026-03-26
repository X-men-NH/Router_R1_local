import argparse
import json
import os

import wandb


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def fetch_history(run, history_keys, history_limit):
    if not history_keys:
        return []

    history = list(run.scan_history(keys=history_keys, page_size=100))
    if history:
        return history[-history_limit:] if history_limit >= 0 else history

    # Fallback for runs where sparse validation metrics are not returned by scan_history(keys=...).
    fallback_rows = run.history(samples=2000, pandas=False)
    filtered = []
    for row in fallback_rows:
        if not isinstance(row, dict):
            continue
        if any(key in row for key in history_keys):
            filtered.append({key: row.get(key) for key in history_keys if key in row})

    return filtered[-history_limit:] if history_limit >= 0 else filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="entity/project/run_id")
    parser.add_argument("--history-keys", nargs="*", default=[])
    parser.add_argument("--history-limit", type=int, default=10)
    args = parser.parse_args()

    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    run = api.run(args.path)

    history = fetch_history(run, args.history_keys, args.history_limit)

    payload = {
        "path": args.path,
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "project": run.project,
        "entity": run.entity,
        "url": run.url,
        "summary": to_jsonable(dict(run.summary)),
        "last_history": to_jsonable(history),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()