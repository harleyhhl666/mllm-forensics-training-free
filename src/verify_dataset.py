"""Phase 0 dataset verification for CocoGlide. No GT leakage into any Gate;
mask stats here are descriptive metadata only (tamper_ratio is a declared
explanatory variable, per the protocol)."""
import csv, json, os, sys
from collections import Counter
import numpy as np
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/disk3/borui/fevi/data/CocoGlide"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/disk3/borui/fevi/runs/phase0"
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(ROOT, "table.csv"))))
print(f"table.csv rows: {len(rows)}")

fmt = Counter()
missing = []
size_mismatch = []
mask_bad = []
recs = []

for r in rows:
    p_real = os.path.join(ROOT, r["real"])
    p_fake = os.path.join(ROOT, r["fake"])
    p_mask = os.path.join(ROOT, r["mask"])
    for p in (p_real, p_fake, p_mask):
        if not os.path.exists(p):
            missing.append(p)
    if missing and (p_real in missing or p_fake in missing or p_mask in missing):
        continue

    im_r, im_f = Image.open(p_real), Image.open(p_fake)
    im_m = Image.open(p_mask)
    fmt[("real", im_r.format, im_r.mode)] += 1
    fmt[("fake", im_f.format, im_f.mode)] += 1
    fmt[("mask", im_m.format, im_m.mode)] += 1

    if im_r.size != im_f.size or im_r.size != im_m.size:
        size_mismatch.append((r["fake"], im_r.size, im_f.size, im_m.size))

    m = np.array(im_m.convert("L"))
    uniq = np.unique(m)
    if not set(uniq.tolist()) <= {0, 255}:
        mask_bad.append((r["mask"], uniq[:6].tolist(), len(uniq)))
    tampered = int((m > 127).sum())
    tamper_ratio = tampered / m.size

    # sanity: does the fake actually differ from the real, and mostly inside mask?
    a = np.array(im_r.convert("RGB")).astype(np.int16)
    b = np.array(im_f.convert("RGB")).astype(np.int16)
    diff = np.abs(a - b).mean(axis=2)
    mbin = m > 127
    in_m = float(diff[mbin].mean()) if mbin.any() else float("nan")
    out_m = float(diff[~mbin].mean()) if (~mbin).any() else float("nan")

    recs.append(dict(
        pair_id=os.path.splitext(os.path.basename(r["fake"]))[0],
        real=r["real"], fake=r["fake"], mask=r["mask"], prompt=r["prompt"],
        width=im_r.size[0], height=im_r.size[1],
        tamper_ratio=round(tamper_ratio, 6),
        mean_abs_diff_inside_mask=round(in_m, 3),
        mean_abs_diff_outside_mask=round(out_m, 3),
    ))

print(f"missing files: {len(missing)}")
print(f"size mismatches: {len(size_mismatch)}  e.g. {size_mismatch[:3]}")
print(f"non-binary masks: {len(mask_bad)}  e.g. {mask_bad[:3]}")
print("format/mode counts:")
for k, v in sorted(fmt.items()):
    print("  ", k, v)

tr = np.array([r["tamper_ratio"] for r in recs])
ins = np.array([r["mean_abs_diff_inside_mask"] for r in recs])
outs = np.array([r["mean_abs_diff_outside_mask"] for r in recs])
print(f"\ntamper_ratio: n={len(tr)} min={tr.min():.4f} p25={np.percentile(tr,25):.4f} "
      f"med={np.median(tr):.4f} p75={np.percentile(tr,75):.4f} max={tr.max():.4f}")
print(f"  frac tamper_ratio<0.01: {(tr<0.01).mean():.3f}   <0.05: {(tr<0.05).mean():.3f}   >0.3: {(tr>0.3).mean():.3f}")
print(f"mean|diff| inside mask : med={np.median(ins):.2f}")
print(f"mean|diff| outside mask: med={np.median(outs):.2f}  (should be ~0 if edit is local)")
print(f"  pairs where outside>inside (suspect correspondence): {(outs>ins).sum()}")
print(f"  pairs with outside>2.0 (global recompression/resize): {(outs>2.0).sum()}")

json.dump(recs, open(os.path.join(OUT, "cocoglide_index.json"), "w"), indent=1)
json.dump(dict(n_pairs=len(recs), missing=len(missing), size_mismatch=size_mismatch[:20],
               non_binary_masks=len(mask_bad),
               formats={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in fmt.items()}),
          open(os.path.join(OUT, "cocoglide_verify.json"), "w"), indent=1)
print(f"\nwrote {OUT}/cocoglide_index.json and cocoglide_verify.json")
