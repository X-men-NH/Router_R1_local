import wandb
api = wandb.Api()
run = api.run("x1010900248-fudan-university-school-of-management/Router-R1-Official/5t9lpre1")
print("RUN", run.name, run.state)
summary = run.summary
sel = [
    "_step","_runtime",
    "actor/pg_loss","actor/ppo_kl","actor/entropy_loss","actor/lr",
    "critic/vf_loss","critic/vf_explained_var","critic/kl","critic/lr",
    "env/number_of_valid_action","env/number_of_valid_route","env/ratio_of_valid_action","env/finish_ratio",
    "response_length/mean","prompt_length/mean",
    "timing_s/step","timing_s/gen","timing_s/update_actor","timing_s/update_critic"
]
print("--- SUMMARY_SELECTED ---")
for k in sel:
    v = summary.get(k, None)
    print(f"{k}: {v}")

keys = [
    "_step",
    "actor/pg_loss","actor/ppo_kl","actor/entropy_loss",
    "critic/vf_loss","critic/vf_explained_var","critic/kl",
    "env/number_of_valid_action","env/number_of_valid_route","env/ratio_of_valid_action","env/finish_ratio"
]
rows = []
for row in run.scan_history(keys=keys):
    rows.append({k: row.get(k) for k in keys})
print("HISTORY_ROWS", len(rows))
print("--- HISTORY_LAST_20 ---")
for r in rows[-20:]:
    print(r)
