import json
import os

import wandb


def main():
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    viewer = api.viewer
    payload = {
        "entity": getattr(viewer, "entity", None),
        "username": getattr(viewer, "username", None),
        "teams": [getattr(team, "name", str(team)) for team in getattr(viewer, "teams", [])],
        "projects": [project.name for project in api.projects()][:100],
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()