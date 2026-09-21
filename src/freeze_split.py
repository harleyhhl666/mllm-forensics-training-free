"""Freeze the TGIF grouped split. Run ONCE; the output is the authority afterwards.

Grouping rule: the unit of splitting is coco_id, NOT the individual image. One
coco_id yields up to 6 fakes (2 mask types x 3 variants) that all derive from the
same source photograph; splitting at image level would put near-duplicates of the
same scene on both sides and leak information across calibration and test.

Stratification: categories are balanced by assigning coco_ids category by
category, so each category contributes ~calibration_frac of its ids to
calibration regardless of how many ids it has.
"""
import json, os, sys, random, hashlib
from collections import defaultdict, Counter
import yaml


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    sp = cfg["split"]
    seed = sp["seed"]
    frac = sp["calibration_frac"]
    ds = cfg["datasets"]["tgif_sp"]
    recs = json.load(open(ds["index"]))
    out_dir = os.path.join(cfg["experiment"]["out_root"], "splits_tgif")
    os.makedirs(out_dir, exist_ok=True)

    # coco_id -> category (verify a coco_id never spans categories)
    cid_cat, conflicts = {}, []
    per_cid = defaultdict(list)
    for r in recs:
        cid = r["coco_id"]
        per_cid[cid].append(r["pair_id"])
        if cid in cid_cat and cid_cat[cid] != r["category"]:
            conflicts.append((cid, cid_cat[cid], r["category"]))
        cid_cat[cid] = r["category"]
    print(f"records={len(recs)}  coco_ids={len(cid_cat)}  categories={len(set(cid_cat.values()))}")
    if conflicts:
        print(f"WARNING: coco_ids spanning categories: {conflicts[:5]}")

    by_cat = defaultdict(list)
    for cid, cat in cid_cat.items():
        by_cat[cat].append(cid)

    rng = random.Random(seed)
    cal_ids, test_ids = [], []
    print(f"\n{'category':<16}{'n_ids':>7}{'cal':>6}{'test':>6}")
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])          # sort first => deterministic given seed
        rng.shuffle(ids)
        k = max(1, round(len(ids) * frac))  # every category contributes >=1
        cal_ids += ids[:k]
        test_ids += ids[k:]
        print(f"{cat:<16}{len(ids):>7}{k:>6}{len(ids)-k:>6}")

    cal_ids, test_ids = sorted(cal_ids), sorted(test_ids)
    assert not (set(cal_ids) & set(test_ids)), "coco_id leak between splits"

    cal_pairs = sorted(p for c in cal_ids for p in per_cid[c])
    test_pairs = sorted(p for c in test_ids for p in per_cid[c])
    assert not (set(cal_pairs) & set(test_pairs)), "pair leak between splits"

    # smoke subset: drawn ONLY from calibration
    smoke_pairs = cal_pairs[:cfg["sampling"]["smoke_n"]]

    def catdist(ids):
        c = Counter(cid_cat[i] for i in ids)
        n = sum(c.values())
        return {k: round(v / n, 4) for k, v in sorted(c.items())}

    cal_dist, test_dist = catdist(cal_ids), catdist(test_ids)
    max_dev = max(abs(cal_dist.get(k, 0) - test_dist.get(k, 0)) for k in
                  set(cal_dist) | set(test_dist))

    out = dict(
        seed=seed, group_key="coco_id", stratify_by="category",
        calibration_frac=frac,
        n_coco_ids=len(cid_cat),
        n_calibration_ids=len(cal_ids), n_test_ids=len(test_ids),
        n_calibration_pairs=len(cal_pairs), n_test_pairs=len(test_pairs),
        calibration_coco_ids=cal_ids, test_coco_ids=test_ids,
        calibration_pair_ids=cal_pairs, test_pair_ids=test_pairs,
        smoke_pair_ids=smoke_pairs,
        category_dist_calibration=cal_dist, category_dist_test=test_dist,
        max_category_proportion_deviation=round(max_dev, 4),
        index_file=ds["index"],
        index_sha1=hashlib.sha1(open(ds["index"], "rb").read()).hexdigest(),
        script="freeze_split.py",
        note=("Split unit is coco_id. All bbox/segm variants of a source image "
              "stay together. Smoke pairs come only from calibration."),
    )
    p = os.path.join(out_dir, "split_frozen.json")
    json.dump(out, open(p, "w"), indent=1)

    print(f"\ncalibration: {len(cal_ids)} coco_ids -> {len(cal_pairs)} fake images")
    print(f"test       : {len(test_ids)} coco_ids -> {len(test_pairs)} fake images")
    print(f"max category proportion deviation: {max_dev:.4f}")
    print(f"smoke pairs (from calibration): {len(smoke_pairs)}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
