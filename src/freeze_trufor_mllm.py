"""Phase T0: freeze the TruFor -> MLLM integration replication protocol.

Mirrors the ELA Stage-A/Stage-B structure, replacing the evidence source only.
Sources come from the V4 validation set, which already has ELA Stage-A history for
the same images, so an ELA-vs-TruFor paired comparison is possible later. One
variant per coco_id, so n_samples == n_sources and there is no variant clustering.

Six conditions = {T0 no-tool, T1 correct evidence, T2 donor evidence} x {fake, real}.

An evidence PACKAGE moves as a unit: T2 supplies the donor's localization map AND
the donor's image-level score together, never a donor map with the target's score.

Runs TruFor once per image needed (fake + real of every source) and freezes the
rendering of the localization map. Writes no MLLM inference.
"""
import argparse, hashlib, io, json, os, sys, time
from collections import Counter, defaultdict
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEED = 20260918

# ---------------------------------------------------------------- rendering
# Frozen: the float map in [0,1] is colour-mapped on a FIXED 0-1 scale, never
# min-max stretched, so displayed intensity equals the model's probability and is
# comparable across images and conditions.
RENDER = dict(colormap="inferno", vmin=0.0, vmax=1.0,
              stretch="none (fixed 0-1 scale; NOT min-max normalised)",
              resize="map is already at input resolution; resized to the analysed "
                     "image's size with BILINEAR only if they differ",
              format="RGB PNG")


def render_map(m, size):
    """float [0,1] HxW -> RGB PIL image, fixed scale, frozen colormap."""
    import matplotlib.cm as cm
    from PIL import Image
    a = np.clip(np.asarray(m, np.float32), 0.0, 1.0)
    rgb = (cm.get_cmap(RENDER["colormap"])(a)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    if im.size != tuple(size):
        im = im.resize(tuple(size), Image.BILINEAR)
    return im


# ---------------------------------------------------------------- prompts
# Neutral. No "reliable", "accurate", "state-of-the-art", "trusted"; no benchmark
# numbers; no instruction to prioritise the tool. Score is given as a raw
# continuous value at fixed 3 decimals, with no discretisation and no threshold.
TOOL_DESC = """An automated forensic analysis of the image is provided. The accompanying
localization map indicates where the tool assigns higher manipulation-related
response. A whole-image score in [0,1] is also provided, with higher values
corresponding to stronger manipulation evidence. Treat these outputs as auxiliary
evidence to be interpreted together with the original image, not as proof."""

GRID = """The 3x3 grid names for locations are exactly:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right."""

STAGE_A_TAIL = """
Your task in THIS step is ONLY to report what you observe. Do NOT decide whether the
image is authentic or altered. Do NOT mention authenticity. That decision happens in
a later, separate step.

Report strictly as JSON with these keys:
{
  "evidence_used": true or false,
  "predicted_regions": [list of 3x3 grid cell names for the most suspicious areas],
  "tool_score_interpretation": "high" or "medium" or "low",
  "reason": "one or two sentences on what the tool response corresponds to in the original image"
}
Output JSON only."""

STAGE_A_TOOL = ("You are assisting a digital image forensics analysis.\n\n"
                "You are given an image under examination and the output of an "
                "automated forensic tool applied to it.\n\n" + TOOL_DESC +
                "\n\nTruFor image-level manipulation score: {score}\n\n" + GRID +
                "\n" + STAGE_A_TAIL)

STAGE_A_NOTOOL = ("You are assisting a digital image forensics analysis.\n\n"
                  "You are given only an image under examination. No automated tool "
                  "outputs are provided.\n\n" + GRID + "\n"
                  """
Your task in THIS step is ONLY to report what you observe. Do NOT decide whether the
image is authentic or altered. Do NOT mention authenticity. That decision happens in
a later, separate step.

Report strictly as JSON with these keys:
{
  "evidence_used": false,
  "predicted_regions": [list of 3x3 grid cell names for the most suspicious areas],
  "tool_score_interpretation": "low",
  "reason": "one or two sentences describing what you observe"
}
Output JSON only.""")

STAGE_B_TAIL = """
Now make the final determination for this image.

Report strictly as JSON with these keys:
{
  "final_verdict": "real" or "fake",
  "reason": "one or two sentences justifying the verdict"
}
Output JSON only."""

STAGE_B_TOOL = ("You are completing a digital image forensics analysis.\n\n"
                "You are given an image under examination, the output of an automated "
                "forensic tool applied to it, and the observation summary produced in "
                "the previous analysis step.\n\n" + TOOL_DESC +
                "\n\nTruFor image-level manipulation score: {score}\n\n"
                "OBSERVATION SUMMARY FROM THE PREVIOUS STEP (do not revise it, treat "
                "it as given):\n{stage_a_summary}\n" + STAGE_B_TAIL)

STAGE_B_NOTOOL = ("You are completing a digital image forensics analysis.\n\n"
                  "You are given an image under examination and the observation "
                  "summary produced in the previous analysis step.\n\n"
                  "OBSERVATION SUMMARY FROM THE PREVIOUS STEP (do not revise it, "
                  "treat it as given):\n{stage_a_summary}\n" + STAGE_B_TAIL)

PROMPTS = dict(STAGE_A_TOOL=STAGE_A_TOOL, STAGE_A_NOTOOL=STAGE_A_NOTOOL,
               STAGE_B_TOOL=STAGE_B_TOOL, STAGE_B_NOTOOL=STAGE_B_NOTOOL)

BANNED = ("reliable", "accurate", "state-of-the-art", "trusted", "trustworthy",
          "highly", "auroc", "benchmark", "proven", "prioritize", "prioritise",
          "should trust", "correct", "wrong", "donor", "ground truth")

CONDITIONS = {
    "fake_T0_notool":  dict(label="fake", evidence=None),
    "fake_T1_correct": dict(label="fake", evidence="own"),
    "fake_T2_wrong":   dict(label="fake", evidence="donor"),
    "real_T0_notool":  dict(label="real", evidence=None),
    "real_T1_correct": dict(label="real", evidence="own"),
    "real_T2_wrong":   dict(label="real", evidence="donor"),
}

METRICS = dict(
    stage_a_fake=["GT localization rate (predicted cells intersect GT cells)",
                  "cue-map overlap (predicted cells intersect tool-map cells)",
                  "joint grounding (both GT and cue hit)",
                  "T1 vs T0", "T1 vs T2", "follows donor cue under T2",
                  "still hits target GT under T2"],
    stage_a_real=["descriptive: rate of reported suspicious regions",
                  "whether correct real evidence induces spurious localization"],
    stage_b=["fake recall T0/T1/T2", "real FPR T0/T1/T2", "specificity",
             "Youden J = TPR - FPR"],
    contrasts_fake=["T1-T0", "T1-T2", "T2-T0"],
    contrasts_real=["FPR(T1)-FPR(T0)", "FPR(T2)-FPR(T0)"],
    contrasts_j=["J(T1)-J(T0)", "J(T2)-J(T0)"],
    stats="effect size -> source-level bootstrap 95% CI -> paired McNemar p "
          "(supplementary)",
    confidence_map="retained on disk as secondary analysis output only; NOT shown to "
                   "the MLLM in this round",
)

CASES = {
    "A_reliable_evidence_integrates": "T1 >> T0, T1 >> T2, real FPR rise contained, "
                                      "J(T1) improves -> strong external evidence is "
                                      "usefully integrated",
    "B_context_bias_persists": "T1 and T2 both raise fake verdicts, real FPR rises "
                               "too, T1 ~ T2 -> forensic-evidence framing bias "
                               "persists even with a highly discriminative generator",
    "C_tool_quality_matters": "T1 improves, T2 no longer induces fake, clear "
                              "correct-vs-wrong separation -> the earlier ELA failure "
                              "was substantially an evidence-quality problem",
    "D_mllm_degrades_trufor": "TruFor alone is strong but MLLM final J drops and real "
                              "FPR worsens -> the reasoning layer damages an already "
                              "strong detector (report as-is, do NOT prompt-tune)",
}


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--skip-trufor", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    od = os.path.join(R, "trufor_mllm")
    os.makedirs(od, exist_ok=True)

    print("=" * 88)
    print("1. PROMPT AUDIT (neutral framing)")
    for k, v in PROMPTS.items():
        low = v.lower()
        hits = [w for w in BANNED if w in low]
        print(f"  {k:<16} banned-word hits: {hits or 'none'}")
        assert not hits, f"{k} contains {hits}"
    assert "{score}" in STAGE_A_TOOL and "{score}" in STAGE_B_TOOL
    assert "{score}" not in STAGE_A_NOTOOL and "{score}" not in STAGE_B_NOTOOL
    print("  score placeholder only in tool conditions: True")
    print("  score format: raw continuous, 3 decimals, no %/label/threshold")
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in PROMPTS.items()}
    for k, v in ph.items():
        print(f"  sha1({k}) = {v}")

    print("\n" + "=" * 88)
    print("2. SOURCE SELECTION (V4 validation sources; ELA Stage-A history exists)")
    V = json.load(open(os.path.join(R, "validation_v4", "validation_v4_frozen.json")))
    by = defaultdict(list)
    for s in V["samples"]:
        by[s["coco_id"]].append(s)
    rng = np.random.default_rng(SEED)
    srcs = sorted(by)
    picked = []
    for c in srcs:
        cand = sorted(by[c], key=lambda s: s["pair_id"])
        rng.shuffle(cand)
        picked.append(cand[0])
    picked.sort(key=lambda s: s["pair_id"])
    print(f"  V4 pool: {len(V['samples'])} samples / {len(by)} coco_ids")
    print(f"  selected: {len(picked)} samples, one variant per coco_id")
    print(f"  n_samples == n_coco_id: "
          f"{len({s['coco_id'] for s in picked}) == len(picked)}")
    trs = np.array([s["tamper_ratio"] for s in picked])
    print(f"  tamper_ratio min={trs.min():.4f} med={np.median(trs):.4f} max={trs.max():.4f}")
    print(f"  mask types {dict(Counter(s['mask_type'] for s in picked))}")
    ela_hist = set()
    p = os.path.join(R, "v4_stageA", "stage_a.jsonl")
    if os.path.exists(p):
        for l in open(p):
            ela_hist.add(json.loads(l)["pair_id"])
    n_hist = sum(1 for s in picked if s["pair_id"] in ela_hist)
    print(f"  selected samples with ELA Stage-A history: {n_hist}/{len(picked)}")

    print("\n" + "=" * 88)
    print("3. DONOR MAPPING (frozen; label-matched)")
    ids = [s["pair_id"] for s in picked]
    cid = {s["pair_id"]: s["coco_id"] for s in picked}
    n, half = len(ids), len(ids) // 2
    donors = {}
    for k, pid in enumerate(ids):
        for st in range(n):
            c = ids[(k + half + st) % n]
            if c != pid and cid[c] != cid[pid]:
                donors[pid] = c
                break
    same = [p for p, d in donors.items() if cid[d] == cid[p]]
    selfd = [p for p, d in donors.items() if d == p]
    print(f"  donors assigned {len(donors)}  same-coco {len(same)}  self {len(selfd)}")
    assert not same and not selfd
    print("  rule: fake target takes the donor source's FAKE evidence package;")
    print("        real target takes the donor source's REAL evidence package")
    print("        (label-matched, fixed before inference, never re-matched later)")

    if args.skip_trufor:
        print("\nSKIP-TRUFOR set; stopping before evidence computation.")
        return

    print("\n" + "=" * 88)
    print("4. TRUFOR EVIDENCE (one run per image: fake + real of every source)")
    from trufor_adapter import TruForAdapter
    tf_frozen = json.load(open("/mnt/disk3/borui/fevi/trufor_smoke/"
                               "trufor_env_frozen.json"))
    ad = TruForAdapter(gpu=args.gpu)
    ck_md5 = hashlib.md5(open(ad.ckpt_path, "rb").read()).hexdigest()
    assert ck_md5 == tf_frozen["checkpoint"]["md5"], "checkpoint changed"
    print(f"  checkpoint md5 verified {ck_md5}")
    ev_dir = os.path.join(od, "evidence")
    os.makedirs(ev_dir, exist_ok=True)
    ev = {}
    t0 = time.time()
    for i, s in enumerate(picked):
        for lab in ("fake", "real"):
            key = f"{lab}__{s['pair_id']}"
            npz = os.path.join(ev_dir, key + ".npz")
            png = os.path.join(ev_dir, key + "_map.png")
            if os.path.exists(npz) and os.path.exists(png):
                z = np.load(npz)
                ev[key] = dict(score=float(z["score"]), npz=npz, png=png)
                continue
            r = ad.run(os.path.join(root, s[lab]))
            np.savez_compressed(npz,
                                map=r["localization_map"].astype(np.float16),
                                conf=r["reliability_map"].astype(np.float16),
                                score=np.array(r["integrity_score"]))
            from PIL import Image
            im = Image.open(os.path.join(root, s[lab]))
            render_map(r["localization_map"], im.size).save(png)
            ev[key] = dict(score=float(r["integrity_score"]), npz=npz, png=png)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(picked)} sources  {(time.time()-t0)/60:.1f} min")
    print(f"  evidence for {len(ev)} images in {(time.time()-t0)/60:.1f} min")

    print("\n" + "=" * 88)
    print("5. SCORE DISTRIBUTIONS (reported, NOT used to re-pick donors)")

    def dist(v):
        v = np.asarray(v, float)
        return (f"n={len(v)} mean={v.mean():.4f} median={np.median(v):.4f} "
                f"p10={np.percentile(v,10):.4f} p90={np.percentile(v,90):.4f}")
    fk = [ev[f"fake__{p}"]["score"] for p in ids]
    rl = [ev[f"real__{p}"]["score"] for p in ids]
    dfk = [ev[f"fake__{donors[p]}"]["score"] for p in ids]
    drl = [ev[f"real__{donors[p]}"]["score"] for p in ids]
    print(f"  fake, own (T1)    {dist(fk)}")
    print(f"  fake, donor (T2)  {dist(dfk)}")
    print(f"  real, own (T1)    {dist(rl)}")
    print(f"  real, donor (T2)  {dist(drl)}")
    print(f"\n  imbalance check (same pool permuted, so T1/T2 marginals should match):")
    print(f"    fake own vs donor mean diff {np.mean(fk)-np.mean(dfk):+.4f}")
    print(f"    real own vs donor mean diff {np.mean(rl)-np.mean(drl):+.4f}")
    print("    NOTE: donors are a permutation of the same score pool, so the T1 and")
    print("    T2 score DISTRIBUTIONS are identical by construction; only the")
    print("    image-to-score correspondence is broken. No post-hoc re-matching.")

    if not args.freeze:
        print("\nDRY RUN — protocol not written.")
        return

    samples = []
    for s in picked:
        d = donors[s["pair_id"]]
        samples.append(dict(
            sample_id=s["pair_id"], coco_id=s["coco_id"], category=s["category"],
            mask_type=s["mask_type"], variant=s["variant"],
            tamper_ratio=s["tamper_ratio"], n_gt_cells=s["n_gt_cells"],
            fake=s["fake"], real=s["real"], mask=s["mask"],
            donor_sample_id=d,
            evidence=dict(
                fake_own=dict(score=round(ev[f"fake__{s['pair_id']}"]["score"], 6),
                              npz=ev[f"fake__{s['pair_id']}"]["npz"],
                              map_png=ev[f"fake__{s['pair_id']}"]["png"]),
                real_own=dict(score=round(ev[f"real__{s['pair_id']}"]["score"], 6),
                              npz=ev[f"real__{s['pair_id']}"]["npz"],
                              map_png=ev[f"real__{s['pair_id']}"]["png"]),
                fake_donor=dict(score=round(ev[f"fake__{d}"]["score"], 6),
                                npz=ev[f"fake__{d}"]["npz"],
                                map_png=ev[f"fake__{d}"]["png"]),
                real_donor=dict(score=round(ev[f"real__{d}"]["score"], 6),
                                npz=ev[f"real__{d}"]["npz"],
                                map_png=ev[f"real__{d}"]["png"]))))
    doc = dict(
        stage="TruFor -> MLLM evidence integration replication (Phase T0 freeze)",
        question="Does the forensic-context bias seen with ELA persist when the "
                 "evidence source is highly discriminative (TruFor)?",
        design="ELA Stage-A/Stage-B structure reused; only the evidence source changes",
        seed=SEED, n_sources=len(samples),
        n_samples_equals_n_sources=True,
        source_pool="V4 validation frozen set (ELA Stage-A history exists for the "
                    "same images), one variant per coco_id",
        sampling_rule=f"one variant per coco_id from the V4 frozen set, "
                      f"deterministic under seed {SEED}; NOT selected using TruFor "
                      f"scores and NOT filtered using prior ELA results",
        conditions=CONDITIONS,
        evidence_package_rule="T2 supplies the donor's localization map AND the "
                              "donor's image-level score together as one package; "
                              "a donor map is never combined with the target score",
        donor_rule="label-matched (fake<-fake, real<-real), donor_coco_id != "
                   "target_coco_id, never self, frozen before inference",
        confidence_map_policy="NOT shown to the MLLM this round; stored in the npz "
                              "for secondary analysis only",
        prompts=PROMPTS, prompt_hashes=ph,
        score_presentation="'TruFor image-level manipulation score: X.XXX' — raw "
                           "continuous value, 3 decimals, no percentage, no "
                           "high/medium/low label, no threshold",
        map_rendering=RENDER,
        images_per_condition={k: (1 if v["evidence"] is None else 2)
                              for k, v in CONDITIONS.items()},
        inference_budget=dict(stage_a=len(samples) * 6, stage_b=len(samples) * 6,
                              total=len(samples) * 12),
        metrics=METRICS, prereg_cases=CASES,
        forbidden_this_round=["gate", "verifier", "confidence threshold",
                              "score calibration", "prompt debiasing", "LoRA",
                              "training", "tool ensemble"],
        trufor=dict(repo_commit=tf_frozen["repo"]["commit"],
                    checkpoint_md5=ck_md5,
                    score_semantics=tf_frozen["outputs"]["score"],
                    map_semantics=tf_frozen["outputs"]["map"],
                    feasibility=dict(auroc=0.9845, ci=[0.9700, 0.9951],
                                     tpr_at_fpr05=0.955, tpr_at_fpr10=0.965,
                                     iou=0.7578, pixel_auroc=0.9930)),
        hashes=dict(config=sha1f(args.config),
                    v4_frozen=sha1f(os.path.join(R, "validation_v4",
                                                 "validation_v4_frozen.json")),
                    adapter=sha1f(os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "trufor_adapter.py")),
                    freezer=sha1f(os.path.abspath(__file__)),
                    trufor_env=sha1f("/mnt/disk3/borui/fevi/trufor_smoke/"
                                     "trufor_env_frozen.json")),
        score_distributions=dict(fake_own=dist(fk), fake_donor=dist(dfk),
                                 real_own=dist(rl), real_donor=dist(drl),
                                 note="donor scores are a permutation of the same "
                                      "pool, so T1/T2 marginals match by "
                                      "construction; no post-hoc re-matching"),
        samples=samples)
    fp = os.path.join(od, "trufor_mllm_frozen.json")
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(samples)} sources; Stage A {len(samples)*6}, "
          f"Stage B {len(samples)*6}, total {len(samples)*12} MLLM inferences")


if __name__ == "__main__":
    main()
