"""Phase V0: audit the TGIF validation split and freeze the confirmatory set.

Runs the checks PROTOCOL_v4_VALIDATION_FROZEN.md requires, in order, and refuses to
freeze anything if a structural check fails:

  1. files present and extracted; naming identical to the testing split
  2. triple completeness (fake / paired original / ps_mask)
  3. image format; mask binarization at >127 still applicable
  4. coco_id independence: validation ∩ all previously used coco_ids == empty
  5. local-splice check (mean |diff| outside mask), same as testing-split Phase 0
  6. ELA pipeline runs unmodified -> eligibility statistics
  7. donor mapping, frozen with assertions

Nothing here loads a model. GT is read only for post-hoc eligibility evaluation,
never for candidate generation.
"""
import argparse, hashlib, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools
from readout_precompute import region_contrast, bbox_eval

BANDS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]
BNAMES = ["<1%", "1-3%", "3-10%", ">10%"]
PAT = re.compile(r"^(\d+)_mask_(bbox|segm)\.png_ps_mask\.png_sd2_(\d)\.png$")
RULE = "p99"


def sha1_file(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_of(t):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= t < hi:
            return i
    return len(BANDS) - 1


def gt_cells_count(maskf, grid, mo):
    h, w = maskf.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    return max(1, sum(1 for i in range(grid) for j in range(grid)
                      if maskf[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() >= mo))


def assign_donors(samples):
    ids = sorted(s["pair_id"] for s in samples)
    cid = {s["pair_id"]: s["coco_id"] for s in samples}
    n, half, out = len(ids), len(ids) // 2, {}
    for k, pid in enumerate(ids):
        for step in range(n):
            c = ids[(k + half + step) % n]
            if c != pid and cid[c] != cid[pid]:
                out[pid] = c
                break
        else:
            raise RuntimeError(f"no donor for {pid}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--root", default="/mnt/disk3/borui/fevi/data/TGIF")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze", action="store_true",
                    help="write validation_v4_frozen.json (only if all checks pass)")
    ap.add_argument("--sample-per-band", type=int, default=0,
                    help="stratified sample size per tamper band (0 = use all eligible)")
    ap.add_argument("--per-coco-cap", type=int, default=2,
                    help="max samples contributed by one coco_id")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    grid = cfg["localization"]["grid"]
    mo = cfg["localization"]["gt_cell_min_overlap"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    max_gt = cfg["localization"]["max_gt_cells_for_gate3"]
    tool_cfg = cfg["forensic_tools"]

    SP = os.path.join(args.root, "sd2-sp_validation")
    OR = os.path.join(args.root, "orig_validation")
    MK = os.path.join(args.root, "masks_validation")

    print("=" * 80)
    print("CHECK 1 — files present, structure matches the testing split")
    missing_dirs = [d for d in (SP, OR, MK) if not os.path.isdir(d)]
    if missing_dirs:
        print(f"  FAIL: missing extracted directories: {missing_dirs}")
        print("  The validation split has not been downloaded/extracted yet.")
        sys.exit(2)
    cats = sorted(os.listdir(SP))
    print(f"  sd2-sp_validation categories: {len(cats)} -> {cats}")
    for d, nm in ((SP, "sd2-sp"), (OR, "orig"), (MK, "masks")):
        n = sum(len(fs) for _, _, fs in os.walk(d))
        print(f"  {nm:<8} files: {n}")

    print("\n" + "=" * 80)
    print("CHECK 2+3 — triple completeness, formats, mask binarization")
    recs, skipped, fmt = [], Counter(), Counter()
    nonbinary = 0
    for cat in cats:
        for fn in sorted(os.listdir(os.path.join(SP, cat))):
            m = PAT.match(fn)
            if not m:
                skipped["filename_unparsed"] += 1
                continue
            cidv, mtype, var = m.group(1), m.group(2), m.group(3)
            p_f = os.path.join(SP, cat, fn)
            p_o = os.path.join(OR, cat, f"{cidv}_orig.png")
            p_m = os.path.join(MK, cat, f"{cidv}_mask_{mtype}.png_ps_mask.png")
            if not os.path.exists(p_o):
                skipped["orig_missing"] += 1; continue
            if not os.path.exists(p_m):
                skipped["ps_mask_missing"] += 1; continue
            imf, imo, imm = Image.open(p_f), Image.open(p_o), Image.open(p_m)
            fmt[("fake", imf.format, imf.mode)] += 1
            fmt[("orig", imo.format, imo.mode)] += 1
            fmt[("mask", imm.format, imm.mode)] += 1
            if imf.size != imo.size or imf.size != imm.size:
                skipped["size_mismatch"] += 1; continue
            mk = np.array(imm.convert("L"))
            if not set(np.unique(mk).tolist()) <= {0, 255}:
                nonbinary += 1
            mb = mk > mbt
            a = np.array(imo.convert("RGB")).astype(np.int16)
            b = np.array(imf.convert("RGB")).astype(np.int16)
            df = np.abs(a - b).mean(axis=2)
            recs.append(dict(
                pair_id=f"{cat}_{cidv}_{mtype}_{var}", coco_id=cidv, category=cat,
                mask_type=mtype, variant=int(var),
                fake=os.path.relpath(p_f, args.root), real=os.path.relpath(p_o, args.root),
                mask=os.path.relpath(p_m, args.root),
                tamper_ratio=round(float(mb.mean()), 6),
                diff_in=round(float(df[mb].mean()) if mb.any() else float("nan"), 3),
                diff_out=round(float(df[~mb].mean()) if (~mb).any() else float("nan"), 3),
            ))
            if args.limit and len(recs) >= args.limit:
                break
        if args.limit and len(recs) >= args.limit:
            break
    print(f"  usable triples: {len(recs)}   skipped: {dict(skipped)}")
    for k, v in sorted(fmt.items()):
        print(f"    {k}: {v}")
    print(f"  non-binary masks (feathered): {nonbinary}/{len(recs)} "
          f"-> '>{mbt}' binarization applies, same as testing split")

    print("\n" + "=" * 80)
    print("CHECK 4 — coco_id independence (MANDATORY GATE)")
    used = set(json.load(open(os.path.join(out_root, "used_coco_ids.json"))))
    vc = {r["coco_id"] for r in recs}
    overlap = vc & used
    print(f"  validation coco_ids: {len(vc)}")
    print(f"  previously used coco_ids: {len(used)}")
    print(f"  INTERSECTION: {len(overlap)}")
    if overlap:
        print(f"  FAIL — overlap present, e.g. {sorted(overlap)[:10]}")
        print("  STOPPING. Reporting rather than continuing.")
        sys.exit(3)
    print("  PASS: validation coco_id set is disjoint from every prior experiment")

    print("\n" + "=" * 80)
    print("CHECK 5 — local-splice property (must match testing split)")
    do = np.array([r["diff_out"] for r in recs], float)
    di = np.array([r["diff_in"] for r in recs], float)
    print(f"  mean|diff| OUTSIDE mask: median={np.nanmedian(do):.4f} "
          f"p90={np.nanpercentile(do,90):.4f} max={np.nanmax(do):.3f}")
    print(f"  mean|diff| INSIDE  mask: median={np.nanmedian(di):.2f}")
    print(f"  fr-like (outside>1.0): {float(np.nanmean(do>1.0)):.3f}   "
          f"(testing split: 0.000, median 0.0030)")

    print("\n" + "=" * 80)
    print("CHECK 6 — ELA pipeline unmodified; eligibility statistics")
    elig, reasons = [], Counter()
    for n, r in enumerate(recs):
        tools, _ = run_tools(os.path.join(args.root, r["fake"]), tool_cfg, grid=grid)
        ela = tools.get("ela")
        if ela is None or not ela["valid"]:
            reasons["gate0_tool_invalid"] += 1; continue
        g = region_contrast(ela["map01"], RULE)
        if g["degenerate"] or g["bbox"] is None:
            reasons["candidate_region_degenerate"] += 1; continue
        mask = np.array(Image.open(os.path.join(args.root, r["mask"])).convert("L")) > mbt
        ev = bbox_eval(g["bbox"], mask)
        if not ev:
            reasons["no_eval"] += 1; continue
        if not ev["hit"]:
            reasons["candidate_missed_gt"] += 1; continue
        if not ev["center_in_mask"]:
            reasons["candidate_center_outside_gt"] += 1; continue
        nc = gt_cells_count(mask.astype(np.float32), grid, mo)
        if nc > max_gt:
            reasons["localization_gate_uninformative"] += 1; continue
        e = dict(r)
        e.update(n_gt_cells=nc, candidate_region=g["bbox"],
                 candidate_score=float(g["contrast"]),
                 candidate_area_frac=float(g["area_frac"]),
                 candidate_gt_hit=True, candidate_center_in_mask=True,
                 candidate_iou=float(ev["iou"]),
                 candidate_precision=float(ev["precision"]),
                 candidate_recall=float(ev["recall"]))
        elig.append(e)
        if (n + 1) % 200 == 0:
            print(f"    {n+1}/{len(recs)} processed, {len(elig)} eligible", flush=True)

    print(f"\n  total fakes examined : {len(recs)}")
    print(f"  ELIGIBLE             : {len(elig)}  ({len(elig)/max(len(recs),1):.3f})")
    for k, v in reasons.most_common():
        print(f"    {k:<34}{v:>6}")

    ec = {e["coco_id"] for e in elig}
    tb = Counter(band_of(e["tamper_ratio"]) for e in elig)
    cl = Counter(e["coco_id"] for e in elig)
    trs = np.array([e["tamper_ratio"] for e in elig], float)
    ious = np.array([e["candidate_iou"] for e in elig], float)
    print(f"\n  unique coco_ids      : {len(ec)}   max pairs/coco_id: "
          f"{max(cl.values()) if cl else 0}")
    print(f"  pairs per coco_id    : {dict(sorted(Counter(cl.values()).items()))}")
    print("  tamper_ratio bands:")
    for i, nm in enumerate(BNAMES):
        sub = [e for e in elig if band_of(e["tamper_ratio"]) == i]
        print(f"    {nm:<7}{tb.get(i,0):>5}  ({len({e['coco_id'] for e in sub})} coco_ids)")
    print(f"  tamper_ratio: min={trs.min():.4f} med={np.median(trs):.4f} "
          f"max={trs.max():.4f}" if len(trs) else "  no eligible samples")
    print(f"  candidate IoU median : {np.median(ious):.4f}" if len(ious) else "")
    print(f"  categories           : {dict(sorted(Counter(e['category'] for e in elig).items()))}")
    print(f"  mask types           : {dict(Counter(e['mask_type'] for e in elig))}")
    pr = sum(1 for e in elig if os.path.exists(os.path.join(args.root, e["real"])))
    print(f"  paired originals available: {pr}/{len(elig)}")

    print("\n" + "=" * 80)
    print("CHECK 7 — donor mapping")

    # ---- Stratified sampling, applied BEFORE donor assignment so the frozen donor
    # map refers exactly to the set that will be run. Rule fixed in advance:
    # equal n per frozen tamper band; within a band prefer coco_ids not yet used,
    # cap per coco_id; deterministic given the experiment seed.
    sample_note = "all eligible (no sampling)"
    if args.sample_per_band:
        rng = np.random.default_rng(cfg["experiment"]["seed"])
        by_band = defaultdict(list)
        for e in elig:
            by_band[band_of(e["tamper_ratio"])].append(e)
        picked, used_c = [], Counter()
        for i, nm in enumerate(BNAMES):
            pool = sorted(by_band[i], key=lambda e: e["pair_id"])
            rng.shuffle(pool)
            pool.sort(key=lambda e: used_c[e["coco_id"]])
            take = []
            for e in pool:
                if len(take) >= args.sample_per_band:
                    break
                if used_c[e["coco_id"]] >= args.per_coco_cap:
                    continue
                used_c[e["coco_id"]] += 1
                take.append(e)
            short = args.sample_per_band - len(take)
            if short > 0:                      # band exhausted under the cap
                for e in pool:
                    if len(take) >= args.sample_per_band:
                        break
                    if e in take:
                        continue
                    used_c[e["coco_id"]] += 1
                    take.append(e)
            print(f"  band {nm:<7} requested {args.sample_per_band}, "
                  f"picked {len(take)}" + ("  (CAP RELAXED)" if short > 0 else ""))
            picked += take
        elig = sorted(picked, key=lambda e: e["pair_id"])
        sample_note = (f"stratified: {args.sample_per_band}/band, "
                       f"per_coco_cap={args.per_coco_cap}, seed={cfg['experiment']['seed']}")
        cl = Counter(e["coco_id"] for e in elig)
        tb = Counter(band_of(e["tamper_ratio"]) for e in elig)
        ec = {e["coco_id"] for e in elig}
        trs = np.array([e["tamper_ratio"] for e in elig], float)
        ious = np.array([e["candidate_iou"] for e in elig], float)
        print(f"  SAMPLED SET: n={len(elig)}  unique coco_ids={len(ec)}  "
              f"max pairs/coco_id={max(cl.values())}")
        print(f"  bands: {dict((BNAMES[i], tb.get(i,0)) for i in range(len(BANDS)))}")
        print(f"  tamper_ratio: min={trs.min():.4f} med={np.median(trs):.4f} "
              f"max={trs.max():.4f}   candidate IoU med={np.median(ious):.4f}")
        print(f"  categories: {dict(sorted(Counter(e['category'] for e in elig).items()))}")
        print(f"  mask types: {dict(Counter(e['mask_type'] for e in elig))}")

    donors = assign_donors(elig)
    cidm = {e["pair_id"]: e["coco_id"] for e in elig}
    bad = [p for p, d in donors.items() if cidm[d] == cidm[p]]
    slf = [p for p, d in donors.items() if d == p]
    print(f"  donors assigned: {len(donors)}   same-coco: {len(bad)}   self: {len(slf)}")
    if bad or slf:
        print("  FAIL: donor rule violated"); sys.exit(4)

    n_inf_a = len(elig) * 3 + len(elig) * 2      # 3 fake + real_own + real_no_tool
    n_inf_b = n_inf_a
    print("\n" + "=" * 80)
    print("INFERENCE PLAN (3.0 s/inference measured)")
    print(f"  V1 Stage A: {n_inf_a} inferences  ~{n_inf_a*3.0/60:.1f} min")
    print(f"  V2 Stage B: {n_inf_b} inferences  ~{n_inf_b*3.0/60:.1f} min")
    print(f"  total     : {n_inf_a+n_inf_b} inferences  "
          f"~{(n_inf_a+n_inf_b)*3.0/60:.1f} min")

    if not args.freeze:
        print("\n(dry run — pass --freeze to write validation_v4_frozen.json)")
        return

    for e in elig:
        e["wrong_tool_donor_pair_id"] = donors[e["pair_id"]]
    od = os.path.join(out_root, "validation_v4")
    os.makedirs(od, exist_ok=True)
    idxp = os.path.join(od, "tgif_sp_validation_index.json")
    json.dump(recs, open(idxp, "w"), indent=1)
    payload = dict(
        protocol="v4_independent_validation_frozen",
        role=("TRUE independent confirmation: TGIF validation split, coco_ids "
              "disjoint from calibration, pilot, Primary and Secondary."),
        seed=cfg["experiment"]["seed"], readout=f"B_region_{RULE}",
        hypotheses=["H1 StageA correct>no-tool", "H2 StageA correct>wrong",
                    "H3 StageB heatmap effect", "H4 StageB loss of cue matching"],
        conditions=["correct_ela", "no_tool", "wrong_donor_ela",
                    "real_own_ela", "real_no_tool"],
        eligibility_rule=("ela valid AND B_region_p99 non-degenerate AND candidate "
                          "hit GT AND candidate center_in_mask AND gt_cells <= "
                          f"{max_gt}   [identical to pilot/Primary]"),
        donor_rule=("sorted pair_ids rotated by half; coco_id collision -> walk "
                    "forward to first different coco_id"),
        behavioral_subset="B2 = mllm_vs_gt AND mllm_vs_correct_cue (pre-registered)",
        sampling=sample_note,
        frozen_tamper_bins=BNAMES,
        statistics=dict(primary="coco_id-cluster bootstrap 95% CI",
                        iterations=10000, seed=cfg["experiment"]["seed"],
                        supplementary=["Wilson", "McNemar exact", "Cochran Q"]),
        audit=dict(examined=len(recs), eligible=len(elig),
                   exclusion_reasons=dict(reasons),
                   unique_coco_ids=len(ec),
                   coco_id_overlap_with_prior_experiments=0,
                   bands={BNAMES[i]: tb.get(i, 0) for i in range(len(BANDS))},
                   diff_outside_median=float(np.nanmedian(do)),
                   nonbinary_masks=nonbinary),
        hashes=dict(config=sha1_file(args.config), index=sha1_file(idxp),
                    used_coco_ids=sha1_file(os.path.join(out_root, "used_coco_ids.json"))),
        n_samples=len(elig), samples=elig,
    )
    p = os.path.join(od, "validation_v4_frozen.json")
    if os.path.exists(p):
        print(f"\nREFUSING TO OVERWRITE {p}")
        sys.exit(1)
    json.dump(payload, open(p, "w"), indent=1)
    print(f"\nfrozen -> {p}  (n={len(elig)}, {len(ec)} coco_ids)")


if __name__ == "__main__":
    main()
