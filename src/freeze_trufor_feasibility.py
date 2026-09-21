"""Freeze the TruFor image-level feasibility protocol. No inference here.

Samples 200 unique coco_ids (one fake variant + its paired real each) from the TGIF
sd2-sp data already on disk. Sampling is deterministic and frozen BEFORE any TruFor
score is computed.

Sources are pooled across the testing and validation index files. Prior FEVI stages
used many of these images with Qwen and ELA, which is irrelevant to TruFor: the
checkpoint is fixed and public, TGIF is not its training data, and nothing here
selects a checkpoint or tunes a threshold. That reasoning is recorded in the frozen
document rather than left implicit.
"""
import argparse, hashlib, json, os, sys
from collections import Counter, defaultdict
import numpy as np, yaml

SEED = 20260918


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


METRICS = dict(
    primary="AUROC(score, fake vs real) with source-pair bootstrap 10000x, 95% CI",
    score_field="TruFor 'score' = sigmoid(det) from the confpool detection head",
    score_direction="higher_is_fake",
    controlled_fpr=["TPR@FPR=0.05", "TPR@FPR=0.10"],
    operating_point_note="thresholds reported at controlled FPR are "
                         "EVALUATION-DERIVED OPERATING POINTS, not frozen deployment "
                         "thresholds, and must not be reused as a tuned threshold on "
                         "this same data",
    distribution=["mean", "median", "std", "p10", "p25", "p75", "p90"],
    paired=["mean/median dscore = score_fake - score_real",
            "P(score_fake > score_real)", "paired bootstrap 95% CI"],
    secondary_localization=["pixel AUROC", "IoU", "pixel F1", "3x3 grid hit rate",
                            "stratified by tamper_ratio and by bbox/segm"],
    localization_note="SECONDARY. Never substitutes for the image-level result.",
    mask_preprocessing="mask > 127 (reuses the frozen TGIF binarization choice)",
    confidence_map="DESCRIPTIVE ONLY: mean conf, conf inside GT, conf outside GT, "
                   "real-image mean conf. No pooled image-level reliability score is "
                   "constructed from it.",
    ela_comparison="ELA image-level score computed with the EXISTING frozen "
                   "blind_anomaly_score read-out on the SAME 200 sources, for a "
                   "paired tool comparison. No new ELA read-out is defined.",
)

GO_RULES = dict(
    hard_no_go="AUROC bootstrap CI includes 0.5, or discrimination near chance -> "
               "TruFor is a useful localizer but unsuitable as an image-level "
               "external evidence source in TGIF-sp; stop the TruFor->MLLM line "
               "even if localization is strong",
    yellow="AUROC clearly above chance but TPR at low FPR is poor -> image-level "
           "signal exists but may be too unreliable for gating; pause",
    go="AUROC clearly above chance AND usable TPR at 5/10% FPR AND paired fake score "
        "usually exceeds paired real AND localization still effective",
    no_field_threshold="no numeric cutoff such as AUROC>0.75 is treated as a field "
                       "standard; continuous performance is reported and interpreted",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-sources", type=int, default=200)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]

    idx_t = os.path.join(R, "phase0", "tgif_sp_index.json")
    idx_v = os.path.join(R, "validation_v4", "tgif_sp_validation_index.json")
    pool = json.load(open(idx_t)) + json.load(open(idx_v))
    by = defaultdict(list)
    for r in pool:
        by[r["coco_id"]].append(r)
    print(f"pool: {len(pool)} triples, {len(by)} unique coco_ids")

    # frozen sampling: one variant per coco_id, deterministic, no tamper/category filtering
    rng = np.random.default_rng(SEED)
    srcs = sorted(by)
    rng.shuffle(srcs)
    n = min(args.n_sources, len(srcs))
    srcs = sorted(srcs[:n])
    picked = []
    for c in srcs:
        cand = sorted(by[c], key=lambda r: r["pair_id"])
        rng.shuffle(cand)
        picked.append(cand[0])
    picked.sort(key=lambda r: r["pair_id"])

    miss = [p for p in picked
            if not (os.path.exists(os.path.join(root, p["fake"]))
                    and os.path.exists(os.path.join(root, p["real"]))
                    and os.path.exists(os.path.join(root, p["mask"])))]
    trs = np.array([p["tamper_ratio"] for p in picked], float)
    print(f"sampled {len(picked)} sources; missing files: {len(miss)}")
    print(f"unique coco_ids == n_samples: {len({p['coco_id'] for p in picked}) == len(picked)}")
    print(f"tamper_ratio min={trs.min():.4f} med={np.median(trs):.4f} max={trs.max():.4f}")
    print(f"mask types {dict(Counter(p['mask_type'] for p in picked))}")
    print(f"categories {len(set(p['category'] for p in picked))} distinct")
    assert not miss

    tf = json.load(open("/mnt/disk3/borui/fevi/trufor_smoke/trufor_env_frozen.json"))
    hashes = dict(
        config=sha1f(args.config),
        index_testing=sha1f(idx_t), index_validation=sha1f(idx_v),
        adapter=sha1f("/home/borui/haolin/fevi/src/trufor_adapter.py"),
        forensic_tools=sha1f("/home/borui/haolin/fevi/src/forensic_tools.py"),
        freezer=sha1f(os.path.abspath(__file__)),
        trufor_env_frozen=sha1f("/mnt/disk3/borui/fevi/trufor_smoke/"
                                "trufor_env_frozen.json"))
    print("\nhashes:")
    for k, v in hashes.items():
        print(f"  {k:<22}{v}")
    print(f"\ncheckpoint md5 {tf['checkpoint']['md5']}")
    print(f"repo commit    {tf['repo']['commit']}")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    samples = [dict(sample_id=p["pair_id"], coco_id=p["coco_id"],
                    category=p["category"], mask_type=p["mask_type"],
                    variant=p["variant"], tamper_ratio=p["tamper_ratio"],
                    fake=p["fake"], real=p["real"], mask=p["mask"],
                    width=p.get("width"), height=p.get("height"))
               for p in picked]
    doc = dict(
        stage="TruFor image-level feasibility evaluation",
        question="Can TruFor's whole-image score separate fake from paired real on "
                 "TGIF sd2-sp?",
        seed=SEED, n_sources=len(samples), n_images=2 * len(samples),
        sampling_rule="one fake variant per coco_id, deterministic given the seed, "
                      "pooled over the testing+validation sd2-sp indices; NO "
                      "tamper_ratio or category filtering; frozen before any TruFor "
                      "score was computed",
        pairing="each source contributes exactly 1 fake and its paired real original, "
                "so n_samples == n_unique_coco_id and there is no within-source "
                "variant clustering",
        data_reuse_justification="these images were previously seen by Qwen and by the "
                                 "ELA pipeline, but TruFor was trained on other data "
                                 "(tampCOCO/compRAISE/FantasticReality/CASIA2/IMD), "
                                 "its checkpoint is fixed and public, and this "
                                 "evaluation performs no checkpoint selection and no "
                                 "threshold tuning; therefore no information about "
                                 "TruFor leaked from prior stages",
        frozen_config=dict(repo_commit=tf["repo"]["commit"],
                           checkpoint=tf["checkpoint"]["path"],
                           checkpoint_md5=tf["checkpoint"]["md5"],
                           preprocessing=tf["preprocessing"],
                           environment=tf["environment"]["packages"],
                           python=tf["environment"]["python"],
                           score_semantics=tf["outputs"]["score"],
                           map_semantics=tf["outputs"]["map"],
                           conf_semantics=tf["outputs"]["conf"]),
        forbidden=["change input resize", "change checkpoint", "tune post-processing",
                   "fit any parameter on TGIF", "select a threshold from results",
                   "define a new ELA read-out"],
        metrics=METRICS, go_rules=GO_RULES, hashes=hashes,
        gt_usage="GT masks are never inputs to TruFor; used only for secondary "
                 "localization scoring and confidence-map description after inference",
        samples=samples)
    od = os.path.join(R, "trufor_feasibility")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_feasibility_frozen.json")
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(samples)} sources, {2*len(samples)} images")


if __name__ == "__main__":
    main()
