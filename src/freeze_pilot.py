"""Freeze the pilot sample set. Tool-layer facts only; no MLLM output exists yet.

Eligibility (Gate 0 + Gate 1, in that order):
  1. ELA valid on the fake image                        (Gate 0)
  2. blind candidate region from B_region_p99 exists     (Gate 1 blind step)
  3. that region is GT-validated: hit AND center_in_mask (Gate 1 post-hoc step)
  4. localization gate informative: GT occupies <= max_gt_cells_for_gate3 cells

Requiring center_in_mask in addition to hit is deliberate and declared here: a
bare 'hit' can be a one-pixel corner overlap, which would not be a cue the model
could plausibly act on. center_in_mask means the candidate's centre lies inside the
manipulated region.

Sampling is stratified by tamper_ratio into the four pre-declared bands and, within
a band, spreads across coco_ids so no single source photograph dominates. Seed
fixed. Paired originals of the selected fakes are included automatically.
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict, Counter
import numpy as np, yaml
from PIL import Image

BANDS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]
BAND_NAMES = ["<1%", "1-3%", "3-10%", ">10%"]
READOUT = "B_region_p99"
REGION_RULES = ("otsu", "p99", "p995")


def gt_cell_count(maskf, grid, mo):
    h, w = maskf.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    n = sum(1 for i in range(grid) for j in range(grid)
            if maskf[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() >= mo)
    return max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--target", type=int, default=180)
    ap.add_argument("--per-coco-cap", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    seed = cfg["experiment"]["seed"]
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    grid = cfg["localization"]["grid"]
    mo = cfg["localization"]["gt_cell_min_overlap"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    max_gt = cfg["localization"]["max_gt_cells_for_gate3"]

    idx = {r["pair_id"]: r for r in json.load(open(ds["index"]))}
    wpath = os.path.join(out_root, "readout_tgif", "windows_test.jsonl")
    ri = REGION_RULES.index("p99")

    # ---------------- eligibility audit
    total = 0
    reasons = Counter()
    elig = []
    for line in open(wpath):
        d = json.loads(line)
        if d["tool"] != "ela" or d["label"] != "fake":
            continue
        total += 1
        if not d.get("valid"):
            reasons["gate0_tool_invalid"] += 1; continue
        blind = d.get("blind")
        if not blind:
            reasons["no_blind_output"] += 1; continue
        g = blind["region"][ri]
        if g["degenerate"] or g["bbox"] is None:
            reasons["candidate_region_degenerate"] += 1; continue
        ev = (d.get("eval") or {}).get("region", [None] * 3)[ri]
        if not ev:
            reasons["no_eval"] += 1; continue
        if not ev["hit"]:
            reasons["candidate_missed_gt"] += 1; continue
        if not ev["center_in_mask"]:
            reasons["candidate_center_outside_gt"] += 1; continue
        rec = idx[d["pair_id"]]
        mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > mbt
        ncell = gt_cell_count(mask.astype(np.float32), grid, mo)
        if ncell > max_gt:
            reasons["localization_gate_uninformative"] += 1; continue
        elig.append(dict(
            pair_id=d["pair_id"], coco_id=d["coco_id"], category=d["category"],
            mask_type=d["mask_type"], variant=d["variant"],
            tamper_ratio=d["tamper_ratio"], n_gt_cells=ncell,
            candidate_region=g["bbox"], candidate_score=float(g["contrast"]),
            candidate_area_frac=float(g["area_frac"]),
            candidate_gt_hit=True, candidate_center_in_mask=True,
            candidate_iou=float(ev["iou"]), candidate_precision=float(ev["precision"]),
            candidate_recall=float(ev["recall"]),
        ))

    print(f"test fakes examined      : {total}")
    print(f"eligible after Gate 0+1  : {len(elig)}  ({len(elig)/max(total,1):.3f})")
    print("exclusion reasons:")
    for k, v in reasons.most_common():
        print(f"  {k:<36}{v:>6}")

    def band_of(tr):
        for i, (lo, hi) in enumerate(BANDS):
            if lo <= tr < hi:
                return i
        return len(BANDS) - 1

    by_band = defaultdict(list)
    for e in elig:
        by_band[band_of(e["tamper_ratio"])].append(e)
    print("\neligible by tamper_ratio band:")
    for i, nm in enumerate(BAND_NAMES):
        print(f"  {nm:<7} n={len(by_band[i]):>5}")

    # ---------------- stratified sampling, capped per coco_id
    rng = np.random.default_rng(seed)
    per_band = args.target // len(BANDS)
    picked, used = [], Counter()
    for i in range(len(BANDS)):
        pool = sorted(by_band[i], key=lambda e: e["pair_id"])
        rng.shuffle(pool)
        pool.sort(key=lambda e: used[e["coco_id"]])     # prefer unused coco_ids
        take = []
        for e in pool:
            if len(take) >= per_band:
                break
            if used[e["coco_id"]] >= args.per_coco_cap:
                continue
            used[e["coco_id"]] += 1
            take.append(e)
        # if a band is short, top up ignoring the cap (recorded below)
        if len(take) < per_band:
            for e in pool:
                if len(take) >= per_band:
                    break
                if e in take:
                    continue
                used[e["coco_id"]] += 1
                take.append(e)
        picked += take
        print(f"  band {BAND_NAMES[i]:<7} requested {per_band}, picked {len(take)}")

    picked.sort(key=lambda e: e["pair_id"])
    # deterministic donor mapping for Control B (wrong/shuffled tool)
    ids = [e["pair_id"] for e in picked]
    half = len(ids) // 2
    donor = {pid: ids[(k + half) % len(ids)] for k, pid in enumerate(ids)}
    for e in picked:
        e["wrong_tool_donor_pair_id"] = donor[e["pair_id"]]
        assert e["wrong_tool_donor_pair_id"] != e["pair_id"]

    cats = Counter(e["category"] for e in picked)
    cocos = len({e["coco_id"] for e in picked})
    trs = np.array([e["tamper_ratio"] for e in picked])
    ious = np.array([e["candidate_iou"] for e in picked])

    out = dict(
        protocol="v2_blind_spatial_cue", readout=READOUT, seed=seed,
        target=args.target, per_coco_cap=args.per_coco_cap,
        n_pilot_fakes=len(picked), n_paired_reals=len(picked),
        n_distinct_coco_ids=cocos,
        eligibility=dict(examined=total, eligible=len(elig),
                         exclusion_reasons=dict(reasons)),
        bands=dict(zip(BAND_NAMES, [len(by_band[i]) for i in range(len(BANDS))])),
        category_counts=dict(sorted(cats.items())),
        tamper_ratio_summary=dict(
            min=float(trs.min()), p25=float(np.percentile(trs, 25)),
            median=float(np.median(trs)), p75=float(np.percentile(trs, 75)),
            max=float(trs.max())),
        candidate_iou_summary=dict(
            median=float(np.median(ious)), p25=float(np.percentile(ious, 25)),
            p75=float(np.percentile(ious, 75))),
        eligibility_rule=("gate0 ela valid AND B_region_p99 non-degenerate AND "
                          "candidate hit GT AND candidate center in GT AND "
                          f"gt cells <= {max_gt}"),
        control_b_rule=("donor = pilot id rotated by half the sorted pilot list; "
                        "donor ELA map resized to target image size; fixed before running"),
        windows_file_sha1=hashlib.sha1(open(wpath, "rb").read()).hexdigest(),
        samples=picked,
    )
    od = os.path.join(out_root, "pilot")
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "pilot_frozen.json")
    if os.path.exists(p):
        print(f"\nREFUSING TO OVERWRITE existing frozen pilot at {p}")
        sys.exit(1)
    json.dump(out, open(p, "w"), indent=1)

    print(f"\npilot fakes: {len(picked)}  (+{len(picked)} paired reals) "
          f"from {cocos} coco_ids")
    print(f"tamper_ratio: min={trs.min():.4f} med={np.median(trs):.4f} max={trs.max():.4f}")
    print(f"candidate IoU: med={np.median(ious):.3f}")
    print(f"categories: {dict(sorted(cats.items()))}")
    print(f"\nfrozen -> {p}")


if __name__ == "__main__":
    main()
