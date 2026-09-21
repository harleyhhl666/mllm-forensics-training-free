"""Freeze the confirmatory validation sets (Primary + Secondary). Run ONCE.

Eligibility is IDENTICAL to the pilot's, verbatim, and involves no Qwen output:
  ELA valid AND B_region_p99 non-degenerate AND candidate hit GT AND
  candidate center in GT AND GT cells <= max_gt_cells_for_gate3

PRIMARY   : coco_id disjoint from the pilot's coco_ids. Natural tamper_ratio
            distribution accepted as-is; no balancing, no top-up.
SECONDARY : pilot pair_ids excluded, pilot coco_ids ALLOWED. Never independent
            confirmation -- power/subgroup analysis only, cluster-aware stats.

Donor rule (frozen, same family as the pilot): sort the set's pair_ids, rotate by
half the list; if that donor shares the target's coco_id, walk forward to the first
candidate with a different coco_id. Deterministic, result-independent.
"""
import argparse, hashlib, json, os, sys
from collections import Counter, defaultdict
import numpy as np, yaml
from PIL import Image

BANDS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]
BAND_NAMES = ["<1%", "1-3%", "3-10%", ">10%"]
READOUT = "B_region_p99"
RI = 1  # index of p99 in REGION_RULES = (otsu, p99, p995)


def sha1_file(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def band_of(tr):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= tr < hi:
            return i
    return len(BANDS) - 1


def gt_cell_count(maskf, grid, mo):
    h, w = maskf.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    return max(1, sum(1 for i in range(grid) for j in range(grid)
                      if maskf[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() >= mo))


def assign_donors(samples):
    """Frozen donor rule. Returns {pair_id: donor_pair_id}."""
    ids = sorted(s["pair_id"] for s in samples)
    cid = {s["pair_id"]: s["coco_id"] for s in samples}
    n = len(ids)
    half = n // 2
    out = {}
    for k, pid in enumerate(ids):
        for step in range(n):
            cand = ids[(k + half + step) % n]
            if cand != pid and cid[cand] != cid[pid]:
                out[pid] = cand
                break
        else:
            raise RuntimeError(f"no valid donor for {pid}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    seed = cfg["experiment"]["seed"]
    ds = cfg["datasets"]["tgif_sp"]
    root, grid = ds["root"], cfg["localization"]["grid"]
    mo = cfg["localization"]["gt_cell_min_overlap"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    max_gt = cfg["localization"]["max_gt_cells_for_gate3"]

    pilot_path = os.path.join(out_root, "pilot", "pilot_frozen.json")
    pilot = json.load(open(pilot_path))
    pilot_pairs = {s["pair_id"] for s in pilot["samples"]}
    pilot_cocos = {s["coco_id"] for s in pilot["samples"]}
    print(f"pilot: {len(pilot_pairs)} pair_ids over {len(pilot_cocos)} coco_ids")

    idx = {r["pair_id"]: r for r in json.load(open(ds["index"]))}
    wpath = os.path.join(out_root, "readout_tgif", "windows_test.jsonl")

    eligible = []
    reasons = Counter()
    examined = 0
    for line in open(wpath):
        d = json.loads(line)
        if d["tool"] != "ela" or d["label"] != "fake":
            continue
        examined += 1
        if not d.get("valid"):
            reasons["gate0_tool_invalid"] += 1; continue
        b = d.get("blind")
        if not b:
            reasons["no_blind_output"] += 1; continue
        g = b["region"][RI]
        if g["degenerate"] or g["bbox"] is None:
            reasons["candidate_region_degenerate"] += 1; continue
        ev = (d.get("eval") or {}).get("region", [None] * 3)[RI]
        if not ev:
            reasons["no_eval"] += 1; continue
        if not ev["hit"]:
            reasons["candidate_missed_gt"] += 1; continue
        if not ev["center_in_mask"]:
            reasons["candidate_center_outside_gt"] += 1; continue
        rec = idx[d["pair_id"]]
        mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > mbt
        nc = gt_cell_count(mask.astype(np.float32), grid, mo)
        if nc > max_gt:
            reasons["localization_gate_uninformative"] += 1; continue
        eligible.append(dict(
            pair_id=d["pair_id"], coco_id=d["coco_id"], category=d["category"],
            mask_type=d["mask_type"], variant=d["variant"],
            tamper_ratio=d["tamper_ratio"], n_gt_cells=nc,
            candidate_region=g["bbox"], candidate_score=float(g["contrast"]),
            candidate_area_frac=float(g["area_frac"]),
            candidate_gt_hit=True, candidate_center_in_mask=True,
            candidate_iou=float(ev["iou"]), candidate_precision=float(ev["precision"]),
            candidate_recall=float(ev["recall"]),
        ))

    print(f"examined test fakes: {examined}   eligible: {len(eligible)}")
    for k, v in reasons.most_common():
        print(f"  {k:<34}{v:>6}")

    primary = sorted([e for e in eligible if e["coco_id"] not in pilot_cocos],
                     key=lambda e: e["pair_id"])
    secondary = sorted([e for e in eligible if e["pair_id"] not in pilot_pairs],
                       key=lambda e: e["pair_id"])
    assert not ({e["coco_id"] for e in primary} & pilot_cocos), "primary coco leak"
    assert not ({e["pair_id"] for e in secondary} & pilot_pairs), "secondary pair leak"

    def describe(S, name):
        bands = Counter(band_of(e["tamper_ratio"]) for e in S)
        cl = Counter(e["coco_id"] for e in S)
        trs = np.array([e["tamper_ratio"] for e in S], float)
        ious = np.array([e["candidate_iou"] for e in S], float)
        print(f"\n{name}: n={len(S)}  unique coco_ids={len(cl)}  "
              f"max pairs per coco_id={max(cl.values()) if cl else 0}")
        for i, nm in enumerate(BAND_NAMES):
            print(f"    {nm:<7}{bands.get(i,0):>5}")
        print(f"    tamper_ratio: min={trs.min():.4f} med={np.median(trs):.4f} "
              f"max={trs.max():.4f}")
        print(f"    candidate IoU median={np.median(ious):.4f}")
        print(f"    categories: {dict(sorted(Counter(e['category'] for e in S).items()))}")
        print(f"    mask types: {dict(Counter(e['mask_type'] for e in S))}")
        return dict(
            n=len(S), unique_coco_ids=len(cl),
            max_pairs_per_coco_id=int(max(cl.values())) if cl else 0,
            pairs_per_coco_id_distribution=dict(Counter(cl.values())),
            bands={BAND_NAMES[i]: bands.get(i, 0) for i in range(len(BANDS))},
            tamper_ratio=dict(min=float(trs.min()), median=float(np.median(trs)),
                              max=float(trs.max())),
            candidate_iou_median=float(np.median(ious)),
            categories=dict(sorted(Counter(e["category"] for e in S).items())),
            mask_types=dict(Counter(e["mask_type"] for e in S)),
        )

    pri_desc = describe(primary, "PRIMARY (coco-disjoint from pilot)")
    sec_desc = describe(secondary, "SECONDARY (pair-disjoint only)")

    common = dict(
        protocol="v3_frozen_cue_grounding", seed=seed, readout=READOUT,
        eligibility_rule=("ela valid AND B_region_p99 non-degenerate AND candidate "
                          "hit GT AND candidate center_in_mask AND gt_cells <= "
                          f"{max_gt}   [identical to pilot; no Qwen output involved]"),
        donor_rule=("sorted pair_ids rotated by half; on coco_id collision walk "
                    "forward to first different coco_id; deterministic"),
        conditions=["correct_ela", "no_tool", "wrong_donor_ela", "paired_real_own_ela"],
        primary_metrics=["A_gt_localization", "B_cue_adherence",
                         "C_grounding_advantage", "D_joint_correct_grounding",
                         "E_cue_induced_error"],
        frozen_tamper_bins=BAND_NAMES,
        hashes=dict(
            windows_test=sha1_file(wpath),
            index=sha1_file(ds["index"]),
            config=sha1_file(args.config),
            pilot_frozen=sha1_file(pilot_path),
            split_frozen=sha1_file(os.path.join(out_root, "splits_tgif",
                                                "split_frozen.json")),
        ),
        eligibility_audit=dict(examined=examined, eligible=len(eligible),
                               exclusion_reasons=dict(reasons)),
    )

    od = os.path.join(out_root, "validation")
    os.makedirs(od, exist_ok=True)
    for name, S, desc, role in (
        ("validation_primary_frozen.json", primary, pri_desc,
         "PRIMARY independent confirmation: coco_ids disjoint from the discovery "
         "pilot. Core conclusions are judged on this set."),
        ("validation_secondary_frozen.json", secondary, sec_desc,
         "SECONDARY expanded analysis: shares coco_ids with the pilot. NOT "
         "independent confirmation. Power/subgroup only, cluster-aware CIs "
         "clustered on coco_id are mandatory."),
    ):
        donors = assign_donors(S)
        cidmap = {e["pair_id"]: e["coco_id"] for e in S}
        for e in S:
            e["wrong_tool_donor_pair_id"] = donors[e["pair_id"]]
        bad = [p for p, dd in donors.items() if cidmap[dd] == cidmap[p]]
        self_d = [p for p, dd in donors.items() if dd == p]
        assert not bad and not self_d, f"donor sanity failed: {bad[:3]} {self_d[:3]}"
        payload = dict(common, role=role, set_name=name.replace(".json", ""),
                       structure=desc, n_samples=len(S),
                       donor_same_coco_violations=0, donor_self_violations=0,
                       samples=S)
        p = os.path.join(od, name)
        if os.path.exists(p) and not args.force:
            print(f"\nREFUSING TO OVERWRITE {p}")
            sys.exit(1)
        json.dump(payload, open(p, "w"), indent=1)
        print(f"\nfrozen -> {p}  (n={len(S)}, donor checks passed)")

    print("\ninference plan (2.7 s/inference measured on pilot):")
    for nm, S in (("PRIMARY", primary), ("SECONDARY", secondary)):
        for k, lab in ((3, "3 fake conditions"), (4, "+ paired real")):
            n = len(S) * k
            print(f"  {nm:<10}{lab:<22}{n:>6} inferences  ~{n*2.7/60:5.1f} min")


if __name__ == "__main__":
    main()
