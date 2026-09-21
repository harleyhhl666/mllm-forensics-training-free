"""Cache ELA anomaly maps for the TGIF calibration split.

Read-out research needs the same maps scored many times under cross-validation.
Recomputing ELA each time dominates runtime, so maps are computed ONCE here and
stored as float16. Maps are downsampled to a fixed maximum side so every scorer
sees a comparable spatial sampling; the downsample factor is recorded.

No GT mask is read while producing a map. Masks are cached alongside purely for
later evaluation, in a separate array, so no scorer can accidentally touch one.
"""
import json, os, sys
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools

MAX_SIDE = 512


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    tool_cfg = cfg["forensic_tools"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    out_dir = os.path.join(cfg["experiment"]["out_root"], "ela_cache")
    os.makedirs(out_dir, exist_ok=True)

    split = json.load(open(os.path.join(cfg["experiment"]["out_root"],
                                        "splits_tgif", "split_frozen.json")))
    cal = set(split["calibration_pair_ids"])
    recs = [r for r in json.load(open(ds["index"])) if r["pair_id"] in cal]
    print(f"caching ELA maps for {len(recs)} calibration pairs (real+fake)")

    meta = []
    for n, rec in enumerate(recs):
        p = os.path.join(out_dir, rec["pair_id"] + ".npz")
        if os.path.exists(p):
            meta.append(json.load(open(p + ".json")))
            continue
        maps = {}
        for lab, key in (("real", "real"), ("fake", "fake")):
            tools, _ = run_tools(os.path.join(root, rec[key]), tool_cfg, grid=3)
            e = tools["ela"]
            m = e["map01"].astype(np.float32)
            h, w = m.shape
            s = MAX_SIDE / max(h, w)
            if s < 1.0:
                m = cv2.resize(m, (int(round(w * s)), int(round(h * s))),
                               interpolation=cv2.INTER_AREA)
            maps[lab] = m.astype(np.float16)
            maps[lab + "_valid"] = np.array([1 if e["valid"] else 0], np.int8)

        mk = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > mbt
        mkr = cv2.resize(mk.astype(np.uint8), (maps["fake"].shape[1], maps["fake"].shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        np.savez_compressed(p, real=maps["real"], fake=maps["fake"],
                            real_valid=maps["real_valid"], fake_valid=maps["fake_valid"],
                            mask_eval_only=mkr)
        d = dict(pair_id=rec["pair_id"], coco_id=rec["coco_id"], category=rec["category"],
                 mask_type=rec["mask_type"], variant=rec["variant"],
                 tamper_ratio=rec["tamper_ratio"],
                 h=int(maps["fake"].shape[0]), w=int(maps["fake"].shape[1]),
                 real_valid=bool(maps["real_valid"][0]), fake_valid=bool(maps["fake_valid"][0]))
        json.dump(d, open(p + ".json", "w"))
        meta.append(d)
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(recs)}", flush=True)

    json.dump(meta, open(os.path.join(out_dir, "cache_meta.json"), "w"), indent=1)
    nv = sum(1 for m in meta if not (m["real_valid"] and m["fake_valid"]))
    print(f"\ncached {len(meta)} pairs; pairs with an invalid ELA map: {nv}")
    print(f"max side {MAX_SIDE}; wrote {out_dir}/cache_meta.json")


if __name__ == "__main__":
    main()
