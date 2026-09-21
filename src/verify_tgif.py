"""Phase 0 dataset verification for TGIF (sd2-sp subset, testing split).

Naming convention (from the official README, verified against the extracted tree):
  orig  : {coco_id}_orig.png                     (also _512 / _1024 crops)
  mask  : {coco_id}_mask_{bbox|segm}.png         (+ _ps_mask.png Photoshop variant)
  sd2-sp: {coco_id}_mask_{type}.png_ps_mask.png_sd2_{var}.png

CRITICAL check for this project: TGIF distinguishes 'sp' (inpainted region spliced
back into the original) from 'fr' (whole image regenerated). Only sp is expected to
leave low-level forensic traces confined to the mask. We therefore MEASURE, not
assume, the mean |diff| inside vs outside the mask. If outside-mask difference is
near zero, ELA / noise-residual are applicable in principle; if it is large, this
subset behaves like fr and the same negative result as CocoGlide would follow.

GT masks are used here only for descriptive metadata (tamper_ratio) and for this
applicability check -- never to set a Gate threshold.
"""
import json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/disk3/borui/fevi/data/TGIF"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/disk3/borui/fevi/runs/phase0"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # 0 = all
os.makedirs(OUT, exist_ok=True)

SP = os.path.join(ROOT, "sd2-sp_testing")
ORIG = os.path.join(ROOT, "orig_testing")
MASK = os.path.join(ROOT, "masks_testing")

# sd2-sp filename -> (coco_id, mask_type, variant)
PAT = re.compile(r"^(\d+)_mask_(bbox|segm)\.png_ps_mask\.png_sd2_(\d)\.png$")

cats = sorted(os.listdir(SP))
print(f"categories: {len(cats)} -> {cats}")

recs, skipped = [], Counter()
fmt = Counter()
for cat in cats:
    for fn in sorted(os.listdir(os.path.join(SP, cat))):
        m = PAT.match(fn)
        if not m:
            skipped["filename_unparsed"] += 1
            continue
        cid, mtype, var = m.group(1), m.group(2), m.group(3)
        p_fake = os.path.join(SP, cat, fn)
        p_orig = os.path.join(ORIG, cat, f"{cid}_orig.png")
        # the mask actually applied by the sp pipeline is the ps_mask variant
        p_mask = os.path.join(MASK, cat, f"{cid}_mask_{mtype}.png_ps_mask.png")
        if not os.path.exists(p_orig):
            skipped["orig_missing"] += 1; continue
        if not os.path.exists(p_mask):
            skipped["ps_mask_missing"] += 1; continue

        im_f, im_o, im_m = Image.open(p_fake), Image.open(p_orig), Image.open(p_mask)
        fmt[("fake", im_f.format, im_f.mode)] += 1
        fmt[("orig", im_o.format, im_o.mode)] += 1
        fmt[("mask", im_m.format, im_m.mode)] += 1
        if im_f.size != im_o.size or im_f.size != im_m.size:
            skipped[f"size_mismatch_f{im_f.size}_o{im_o.size}_m{im_m.size}"] += 1
            continue

        mk = np.array(im_m.convert("L"))
        uniq = np.unique(mk)
        binary = set(uniq.tolist()) <= {0, 255}
        mbin = mk > 127
        tamper_ratio = float(mbin.mean())

        a = np.array(im_o.convert("RGB")).astype(np.int16)
        b = np.array(im_f.convert("RGB")).astype(np.int16)
        d = np.abs(a - b).mean(axis=2)
        in_m = float(d[mbin].mean()) if mbin.any() else float("nan")
        out_m = float(d[~mbin].mean()) if (~mbin).any() else float("nan")

        recs.append(dict(
            pair_id=f"{cat}_{cid}_{mtype}_{var}", coco_id=cid, category=cat,
            mask_type=mtype, variant=int(var),
            fake=os.path.relpath(p_fake, ROOT), real=os.path.relpath(p_orig, ROOT),
            mask=os.path.relpath(p_mask, ROOT),
            width=im_f.size[0], height=im_f.size[1],
            tamper_ratio=round(tamper_ratio, 6), mask_binary=binary,
            mean_abs_diff_inside_mask=round(in_m, 3),
            mean_abs_diff_outside_mask=round(out_m, 3),
        ))
        if LIMIT and len(recs) >= LIMIT:
            break
    if LIMIT and len(recs) >= LIMIT:
        break

print(f"\nusable triples: {len(recs)}")
print(f"skipped: {dict(skipped)}")
print("format/mode counts:")
for k, v in sorted(fmt.items()):
    print("  ", k, v)

tr = np.array([r["tamper_ratio"] for r in recs])
ins = np.array([r["mean_abs_diff_inside_mask"] for r in recs])
outs = np.array([r["mean_abs_diff_outside_mask"] for r in recs])
nb = sum(1 for r in recs if not r["mask_binary"])
print(f"\nnon-binary masks: {nb}/{len(recs)}")
print(f"tamper_ratio: min={tr.min():.4f} p25={np.percentile(tr,25):.4f} "
      f"med={np.median(tr):.4f} p75={np.percentile(tr,75):.4f} max={tr.max():.4f}")
print(f"  <0.01: {(tr<0.01).mean():.3f}  <0.05: {(tr<0.05).mean():.3f}  "
      f"0.05-0.35: {((tr>=0.05)&(tr<0.35)).mean():.3f}  >0.35: {(tr>0.35).mean():.3f}")

print("\n*** APPLICABILITY CHECK: is this really 'spliced' (local) editing? ***")
print(f"  mean|diff| INSIDE  mask: med={np.median(ins):.2f}")
print(f"  mean|diff| OUTSIDE mask: med={np.median(outs):.4f}   "
      f"p90={np.percentile(outs,90):.4f}  max={outs.max():.3f}")
print(f"  triples with outside<0.01 (pixel-exact splice): {(outs<0.01).mean():.3f}")
print(f"  triples with outside>1.0  (global change, fr-like): {(outs>1.0).mean():.3f}")
if np.median(outs) < 0.01:
    print("  -> CONFIRMED local splice: pixels outside the mask are untouched, so")
    print("     ELA / noise-residual discontinuity at the mask boundary is")
    print("     physically meaningful here (unlike CocoGlide, median 0.24).")
else:
    print("  -> WARNING: outside-mask pixels changed; this behaves like full")
    print("     regeneration and the CocoGlide negative result likely repeats.")

print("\nper mask_type:")
for mt in ("bbox", "segm"):
    s = [r for r in recs if r["mask_type"] == mt]
    if s:
        t = np.array([r["tamper_ratio"] for r in s])
        o = np.array([r["mean_abs_diff_outside_mask"] for r in s])
        print(f"  {mt}: n={len(s)} tamper_med={np.median(t):.4f} outside_med={np.median(o):.4f}")
print(f"\ndistinct coco_ids: {len({r['coco_id'] for r in recs})}  "
      f"(variants per edit: up to 3; mask types: 2)")

json.dump(recs, open(os.path.join(OUT, "tgif_sp_index.json"), "w"), indent=1)
print(f"\nwrote {OUT}/tgif_sp_index.json")
