"""Smoke test the forensic tool layer: does Gate-0 validity behave sanely, and
does the blind Gate-1 anomaly score respond to fake images at all?

This does NOT tune anything. It reports raw numbers on a fixed small sample so
we can see whether the tool layer is worth running the MLLM on.
"""
import json, os, sys, random
import numpy as np, yaml, cv2
sys.path.insert(0, os.path.dirname(__file__))
from forensic_tools import run_tools

cfg = yaml.safe_load(open(sys.argv[1]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
root = cfg["datasets"]["cocoglide"]["root"]
idx = json.load(open("/mnt/disk3/borui/fevi/runs/phase0/cocoglide_index.json"))
random.Random(cfg["experiment"]["seed"]).shuffle(idx)
sample = idx[:n]

tool_cfg = cfg["forensic_tools"]
grid = cfg["localization"]["grid"]
res = {t: {"real": [], "fake": []} for t in ("ela", "noise_residual", "dct_jpeg")}
valid = {t: {"real": [0, 0], "fake": [0, 0]} for t in res}
reasons = {}

for rec in sample:
    for lab, key in (("real", "real"), ("fake", "fake")):
        tools, _ = run_tools(os.path.join(root, rec[key]), tool_cfg, grid=grid)
        for t, r in tools.items():
            res[t][lab].append(r["anomaly_score"])
            valid[t][lab][0] += int(r["valid"]); valid[t][lab][1] += 1
            for rs in r["invalid_reasons"]:
                reasons[(t, rs)] = reasons.get((t, rs), 0) + 1

print(f"n pairs = {len(sample)}\n")
print(f"{'tool':<16}{'valid_real':>12}{'valid_fake':>12}{'score_real_med':>16}{'score_fake_med':>16}{'AUC':>8}")
for t in res:
    r = np.array(res[t]["real"]); f = np.array(res[t]["fake"])
    # rank-based AUC of fake-vs-real separability of the blind score
    allv = np.concatenate([r, f]); rank = allv.argsort().argsort() + 1
    auc = (rank[len(r):].sum() - len(f) * (len(f) + 1) / 2) / (len(r) * len(f))
    print(f"{t:<16}{valid[t]['real'][0]}/{valid[t]['real'][1]:<10}"
          f"{valid[t]['fake'][0]}/{valid[t]['fake'][1]:<10}"
          f"{np.median(r):>16.3f}{np.median(f):>16.3f}{auc:>8.3f}")
print("\ninvalid reasons:")
for (t, rs), c in sorted(reasons.items()):
    print(f"  {t:<16}{rs:<45}{c}")
