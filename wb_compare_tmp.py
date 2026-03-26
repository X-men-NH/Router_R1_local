import wandb
from statistics import mean

ENTITY = "x1010900248-fudan-university-school-of-management"
PROJECT = "Router-R1-Official"
LATEST_ID = "5t9lpre1"
TARGET_NAME = "nh-bs32-ppo-qwen2.5-1.5b-it-em"

api = wandb.Api()
latest = api.run(f"{ENTITY}/{PROJECT}/{LATEST_ID}")

runs = api.runs(f"{ENTITY}/{PROJECT}")
cands = []
for r in runs:
    if r.id == latest.id:
        continue
    if r.name != TARGET_NAME:
        continue
    s = r.summary
    step = s.get("_step", -1)
    runtime = s.get("_runtime", None)
    state = r.state
    score = 0
    score += abs(step - 90) if isinstance(step, (int, float)) else 9999
    score += abs(runtime - 4*3600)/1000.0 if isinstance(runtime, (int, float)) else 99
    if state in ("crashed", "failed", "killed", "preempted"):
        score -= 2
    cands.append((score, r))

if not cands:
    print("NO_CANDIDATE_FOUND")
    raise SystemExit(0)

cands.sort(key=lambda x: x[0])
old = cands[0][1]

keys = [
    "actor/pg_loss","actor/ppo_kl","actor/entropy_loss",
    "critic/vf_loss","critic/vf_explained_var","critic/kl",
    "env/ratio_of_valid_action","env/finish_ratio",
    "timing_s/step"
]

def summarize_run(run):
    s = run.summary
    out = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "step": s.get("_step"),
        "runtime_sec": s.get("_runtime"),
    }
    for k in keys:
        out[k] = s.get(k)

    hist_keys = ["_step"] + keys
    rows = list(run.scan_history(keys=hist_keys))

    def window_avg(k, n=20):
        vals = [r.get(k) for r in rows[-n:] if isinstance(r.get(k), (int, float))]
        return mean(vals) if vals else None

    out["history_rows"] = len(rows)
    for k in keys:
        out[f"avg_last20::{k}"] = window_avg(k, 20)
    return out

old_sum = summarize_run(old)
new_sum = summarize_run(latest)

print("OLD_RUN", old_sum)
print("NEW_RUN", new_sum)
print("ASSESSMENT")
for k in ["critic/vf_explained_var", "env/ratio_of_valid_action", "env/finish_ratio"]:
    o = old_sum.get(f"avg_last20::{k}")
    n = new_sum.get(f"avg_last20::{k}")
    print(k, "old=", o, "new=", n)
