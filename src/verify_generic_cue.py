"""Verify the generic salient cue BEFORE it is admitted as a control.

Two questions, both answered on the PILOT (discovery) samples only, so the
untouched validation pool stays clean:

  1. Comparability: are the generic map's appearance statistics in the same range
     as the ELA map's? (we claim similar look/dynamic range -- verify it)
  2. Confound size: how often does the generic candidate region accidentally hit
     the GT mask, compared with the ELA candidate? A generic cue that hits GT as
     often as ELA does would be useless as a "no forensic information" control.

No decision here changes any frozen artifact.
"""
import json, os, sys
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools
from generic_cue import generic_salient_map, map_stats, candidate_region_p99


def bbox_eval(bbox, mask):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    H, W = mask.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    sub = mask[y1:y2, x1:x2]
    inter = int(sub.sum())
    barea = (x2 - x1) * (y2 - y1)
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    return dict(hit=inter > 0, precision=inter / max(barea, 1),
                iou=inter / max(barea + int(mask.sum()) - inter, 1),
                center_in_mask=bool(mask[min(cy, H - 1), min(cx, W - 1)]))


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    grid = cfg["localization"]["grid"]
    tool_cfg = cfg["forensic_tools"]
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    pilot = json.load(open(os.path.join(cfg["experiment"]["out_root"], "pilot",
                                        "pilot_frozen.json")))
    idx = {r["pair_id"]: r for r in json.load(open(ds["index"]))}
    S = pilot["samples"][:n_max]

    es, gs = [], []
    ehit, ghit, ectr, gctr, eiou, giou = [], [], [], [], [], []
    for s in S:
        rec = idx[s["pair_id"]]
        mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > mbt
        tools, rgb = run_tools(os.path.join(root, rec["fake"]), tool_cfg, grid=grid)
        r = tools["ela"]
        if not r["valid"]:
            continue
        gm, _ = generic_salient_map(rgb)
        es.append(map_stats(r["map01"]))
        gs.append(map_stats(gm))
        ee = bbox_eval(s["candidate_region"], mask)
        ge = bbox_eval(candidate_region_p99(gm), mask)
        if ee:
            ehit.append(ee["hit"]); ectr.append(ee["center_in_mask"]); eiou.append(ee["iou"])
        if ge:
            ghit.append(ge["hit"]); gctr.append(ge["center_in_mask"]); giou.append(ge["iou"])

    def col(rows, k):
        return np.array([r[k] for r in rows], float)

    print(f"n images = {len(es)}   (pilot/discovery samples only)\n")
    print("1. APPEARANCE COMPARABILITY (median over images)")
    print(f"  {'statistic':<22}{'ELA':>10}{'generic':>10}")
    for k in ("mean", "std", "p99", "dynamic_range", "near_constant_ratio"):
        print(f"  {k:<22}{np.median(col(es,k)):>10.4f}{np.median(col(gs,k)):>10.4f}")

    print("\n2. CONFOUND SIZE: does the generic cue accidentally find the edit?")
    print(f"  ELA     candidate: hit={np.mean(ehit):.3f} center_in_mask={np.mean(ectr):.3f} "
          f"median IoU={np.median(eiou):.4f}   (n={len(ehit)})")
    print(f"  generic candidate: hit={np.mean(ghit):.3f} center_in_mask={np.mean(gctr):.3f} "
          f"median IoU={np.median(giou):.4f}   (n={len(ghit)})")
    print("\n  NOTE: pilot samples were SELECTED for ELA candidate hit==True, so the")
    print("  ELA row is 1.000 by construction and is NOT a performance claim. Only")
    print("  the generic row is informative: it is the accidental hit rate of a")
    print("  saliency map on the same images.")
    if len(ghit) and np.mean(ghit) > 0.6:
        print("\n  -> WARNING: generic cue hits GT often; it is a weak 'no-information'")
        print("     control and must be reported as conservative.")


if __name__ == "__main__":
    main()
