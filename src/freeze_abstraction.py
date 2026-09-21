"""Freeze the Evidence Abstraction Layer experiment. No MLLM inference here.

The abstraction layer is deterministic, training-free, uses no ground truth, calls no
second LLM, and never reads the authenticity label. It only converts a TruFor
localization map into structured statistics.

Conditions:
  E0 score only
  E1 score + raw map
  E2 score + structured evidence            <- key condition
  E3 score + structured evidence + raw map
  E4 score + horizontally mirrored structured evidence   (secondary control)

Connected components are labelled with a self-contained numpy/stdlib union-find so
that no package is added to the frozen qwen_vl environment (scipy is absent there).
"""
import argparse, hashlib, json, os, sys
from collections import Counter, deque
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import CELL_NAMES

# ---- pre-registered extraction parameters (frozen before any MLLM inference) ----
BIN_THR = 0.5            # map > 0.5 -> binary mask
MIN_AREA_RATIO = 0.001   # drop components smaller than 0.1% of image area
TOP_K = 3                # keep at most 3 regions
RANK_BY = "integrated_probability"   # sum of map probability inside the component
CONNECTIVITY = 4
ROUND = dict(score=3, area_ratio=4, prob=3, centroid=3)

MIRROR = {"top-left": "top-right", "top-center": "top-center",
          "top-right": "top-left", "center-left": "center-right",
          "center": "center", "center-right": "center-left",
          "bottom-left": "bottom-right", "bottom-center": "bottom-center",
          "bottom-right": "bottom-left"}

NEUTRAL = ("An automated forensic tool provides an image-level manipulation score and "
           "structured localization measurements derived from its localization output. "
           "Treat these measurements as auxiliary evidence and interpret them together "
           "with the original image.")

HEAD = "You are performing a digital image forensics analysis.\n\n"
SCORE_LINE = "TruFor image-level manipulation score: {score}\n\n"
TASK = ("Determine whether this image is an authentic photograph or has been digitally\n"
        "altered.\n\n")
SCHEMA = ('Report strictly as JSON with these keys:\n{\n'
          '  "final_verdict": "real" or "fake",\n'
          '  "referenced_regions": ["grid location(s) your verdict relies on, '
          'from: top-left, top-center, top-right, center-left, center, center-right, '
          'bottom-left, bottom-center, bottom-right; use [] if none"],\n'
          '  "reason": "one or two sentences justifying the verdict"\n'
          '}\nOutput JSON only.')

SCORE_DESC = ("An automated forensic analysis of the image is provided as a whole-image "
              "score in [0,1], where higher values correspond to stronger "
              "manipulation-related evidence. Treat it as auxiliary evidence to be "
              "interpreted together with the original image, not as proof.\n\n")
RAW_DESC = ("An automated forensic analysis of the image is provided. The accompanying "
            "localization map indicates where the tool assigns higher "
            "manipulation-related response. A whole-image score in [0,1] is also "
            "provided, where higher values correspond to stronger manipulation-related "
            "evidence. Treat these outputs as auxiliary evidence to be interpreted "
            "together with the original image, not as proof.\n\n")
STRUCT_DESC = (NEUTRAL + " The measurements are given in the JSON block below. Grid "
               "locations use a 3x3 partition of the image; area_ratio is the fraction "
               "of the image covered by the region; coordinates are normalised to "
               "[0,1] with the origin at the top-left. An empty region list means the "
               "tool localised no region above its internal response level.\n\n")
BOTH_DESC = (NEUTRAL + " Both a localization map image and the corresponding numerical "
             "measurements are provided. Grid locations use a 3x3 partition of the "
             "image; area_ratio is the fraction of the image covered by the region; "
             "coordinates are normalised to [0,1] with the origin at the top-left. An "
             "empty region list means the tool localised no region above its internal "
             "response level.\n\n")

PROMPTS = {
    "E0": HEAD + SCORE_DESC + SCORE_LINE + TASK + SCHEMA,
    "E1": HEAD + RAW_DESC + SCORE_LINE + TASK + SCHEMA,
    "E2": HEAD + STRUCT_DESC + SCORE_LINE + "{structured}\n\n" + TASK + SCHEMA,
    "E3": HEAD + BOTH_DESC + SCORE_LINE + "{structured}\n\n" + TASK + SCHEMA,
}
PROMPTS["E4"] = PROMPTS["E2"]          # byte-identical; only the JSON content differs

CONDITIONS = {
    "E0_score_only":  dict(score=True, raw_map=False, structured=False, shift=False),
    "E1_raw_map":     dict(score=True, raw_map=True,  structured=False, shift=False),
    "E2_structured":  dict(score=True, raw_map=False, structured=True,  shift=False),
    "E3_both":        dict(score=True, raw_map=True,  structured=True,  shift=False),
    "E4_shifted_struct": dict(score=True, raw_map=False, structured=True, shift=True),
}

CONTRASTS = dict(
    structured_benefit="E2 - E0",
    raw_map_cost="E1 - E0",
    structured_vs_raw="E2 - E1   (MOST IMPORTANT)",
    raw_after_abstraction="E3 - E2",
    shift_control="E4 vs E2, primarily on referenced_regions rather than verdict",
    stats="paired effect size -> source-level bootstrap 95% CI -> McNemar supplementary",
)

FAITHFULNESS = dict(
    evidence_citation_rate="P(referenced_regions non-empty | evidence lists >=1 region)",
    spatial_agreement="P(referenced set intersects the evidence grid locations)",
    exact_agreement="P(referenced set == evidence grid location set)",
    unsupported_region_rate="P(model cites a location absent from the evidence)",
    shift_responsiveness="E2->E4: P(referenced region moves consistently with the "
                         "mirrored evidence), evaluated on sources whose mirrored "
                         "location differs from the original",
    hallucinated_grounding="model cites a region while the evidence list is empty",
)

CASES = {
    "A_representation_bottleneck":
        "E2 > E1, E2 at or above E0, high spatial agreement, and E4 explanations follow "
        "the evidence shift -> the limitation lies in how forensic evidence is "
        "REPRESENTED: MLLMs struggle to read spatial evidence from low-level heatmaps "
        "but can use the same evidence once abstracted into structured form",
    "B_spatial_evidence_unused":
        "E2 ~ E0, low spatial agreement, E4 explanations do not respond to the shift "
        "-> the decision layer relies on scalar evidence and does not meaningfully use "
        "spatial evidence even when explicitly represented",
    "C_explanation_only_gain":
        "E2 J ~ E0 but spatial agreement / faithfulness clearly improves -> structured "
        "abstraction improves explanation faithfulness without materially changing "
        "authenticity discrimination",
    "D_structured_harms":
        "E2 < E0 -> report directly, no prompt tuning",
}

SUCCESS_CRITERION = ("preserve most of the scalar discrimination while substantially "
                     "improving spatially faithful explanations (accuracy preservation "
                     "+ explanation faithfulness). Structured evidence is NOT required "
                     "to beat score-only accuracy; the TruFor scalar score is already "
                     "strong.")


# ------------------------------- abstraction layer -------------------------------

def label_components(mask):
    """4-connected labelling, BFS, no scipy. Returns (labels, count)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    cur = 0
    for i in range(H):
        row = mask[i]
        for j in range(W):
            if not row[j] or lab[i, j]:
                continue
            cur += 1
            q = deque([(i, j)])
            lab[i, j] = cur
            while q:
                y, x = q.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        q.append((ny, nx))
    return lab, cur


def grid_cell(cx, cy):
    """Map a normalised centroid to a 3x3 cell name (row-major, as CELL_NAMES)."""
    col = min(2, int(cx * 3))
    row = min(2, int(cy * 3))
    return CELL_NAMES[row * 3 + col]


def abstract(map_arr, score, mirror=False):
    """Deterministic TruFor map -> structured evidence. No GT, no label, no LLM."""
    H, W = map_arr.shape
    area = float(H * W)
    mask = map_arr > BIN_THR
    lab, n = label_components(mask)
    regs = []
    for c in range(1, n + 1):
        sel = lab == c
        a = int(sel.sum())
        if a / area < MIN_AREA_RATIO:
            continue
        vals = map_arr[sel]
        ys, xs = np.nonzero(sel)
        cx, cy = float(xs.mean()) / W, float(ys.mean()) / H
        regs.append(dict(_integrated=float(vals.sum()), area_ratio=a / area,
                         mean_probability=float(vals.mean()),
                         max_probability=float(vals.max()), cx=cx, cy=cy))
    regs.sort(key=lambda r: -r["_integrated"])
    regs = regs[:TOP_K]
    out = []
    for r in regs:
        cx, cy = r["cx"], r["cy"]
        loc = grid_cell(cx, cy)
        if mirror:
            cx = 1.0 - cx
            loc = MIRROR[loc]
        out.append(dict(grid_location=loc,
                        area_ratio=round(r["area_ratio"], ROUND["area_ratio"]),
                        mean_probability=round(r["mean_probability"], ROUND["prob"]),
                        max_probability=round(r["max_probability"], ROUND["prob"]),
                        centroid=[round(cx, ROUND["centroid"]),
                                  round(cy, ROUND["centroid"])]))
    return dict(image_manipulation_score=round(float(score), ROUND["score"]),
                regions=out)


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--probe", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    src = os.path.join(R, "trufor_conflict",
                       "trufor_score_map_conflict_frozen.json")
    prev = json.load(open(src))
    ab = json.load(open(os.path.join(R, "trufor_score_map",
                                     "trufor_score_map_ablation_frozen.json")))
    full = json.load(open(os.path.join(R, "trufor_mllm",
                                       "trufor_mllm_frozen.json")))
    byid = {s["sample_id"]: s for s in full["samples"]}
    S = prev["samples"]

    print("=" * 92)
    print("1. SAMPLES — unchanged, fake + paired real")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    assert len({s["coco_id"] for s in S}) == len(S)
    have_real = sum(1 for s in S if "real_own" in byid[s["sample_id"]]["evidence"])
    print(f"  paired real evidence available: {have_real}/{len(S)}")
    print(f"  inherited from conflict freeze (sha1 {sha1f(src)[:12]})")
    print(f"  primary model: 7B; 32B deferred until 7B results are in")

    print("\n" + "=" * 92)
    print("2. ABSTRACTION LAYER (deterministic, training-free, no GT, no LLM)")
    print(f"  binary threshold         map > {BIN_THR}")
    print(f"  connectivity             {CONNECTIVITY}-connected (own BFS; scipy is "
          f"absent from qwen_vl and is deliberately NOT installed)")
    print(f"  minimum component area   area/image >= {MIN_AREA_RATIO}")
    print(f"  top-K                    {TOP_K}")
    print(f"  ranking                  {RANK_BY} = sum(map probability in component)")
    print(f"  coordinates              normalised [0,1], origin top-left")
    print(f"  grid                     3x3, CELL_NAMES from forensic_tools (reused)")
    print(f"  empty case               regions: []  (no region is ever invented)")
    print(f"  rounding                 {ROUND}")
    print(f"  the layer never reads the authenticity label or the GT mask")

    print("\n" + "=" * 92)
    print("3. EXTRACTION PROBE on the first sources")
    stats = dict(fake=[], real=[])
    for s in S[:args.probe]:
        for lb in ("fake", "real"):
            ev = byid[s["sample_id"]]["evidence"].get(f"{lb}_own")
            if not ev:
                continue
            m = np.load(ev["npz"])["map"].astype(np.float32)
            st = abstract(m, ev["score"])
            stats[lb].append(len(st["regions"]))
            if lb == "fake" and len(stats["fake"]) <= 2:
                print(f"  [{lb}] {s['sample_id']}  tamper {s['tamper_ratio']:.4f}")
                print("   " + json.dumps(st, indent=1).replace("\n", "\n   "))
    print(f"\n  region counts on probe  fake {stats['fake']}  real {stats['real']}")

    print("\n" + "=" * 92)
    print("4. MIRROR CONTROL (E4)")
    ev = byid[S[0]["sample_id"]]["evidence"]["fake_own"]
    m = np.load(ev["npz"])["map"].astype(np.float32)
    a, b = abstract(m, ev["score"]), abstract(m, ev["score"], mirror=True)
    print(f"  original  {[r['grid_location'] for r in a['regions']]}  "
          f"centroids {[r['centroid'] for r in a['regions']]}")
    print(f"  mirrored  {[r['grid_location'] for r in b['regions']]}  "
          f"centroids {[r['centroid'] for r in b['regions']]}")
    same_str = all(x["area_ratio"] == y["area_ratio"]
                   and x["mean_probability"] == y["mean_probability"]
                   and x["max_probability"] == y["max_probability"]
                   for x, y in zip(a["regions"], b["regions"]))
    print(f"  evidence STRENGTH unchanged (area/mean/max): {same_str}")
    print(f"  score unchanged: {a['image_manipulation_score'] == b['image_manipulation_score']}")
    print(f"  mapping: left<->right, center columns unchanged, x' = 1 - x")
    print(f"  the ORIGINAL IMAGE is never mirrored")

    print("\n" + "=" * 92)
    print("5. PROMPTS")
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in PROMPTS.items()}
    for k in ("E0", "E1", "E2", "E3", "E4"):
        print(f"  {k}  sha1 {ph[k]}")
    print(f"  E2 and E4 share a byte-identical prompt: {ph['E2'] == ph['E4']}")
    print(f"  E0 differs from the earlier C10 only by the added referenced_regions key")
    bias = ("reliable", "accurate", "state-of-the-art", "trusted", "auroc",
            "benchmark", "prioritize", "correct", "wrong", "donor", "ground truth",
            "threshold", "mirror", "shift")
    hits = {k: [b for b in bias if b in v.lower()] for k, v in PROMPTS.items()}
    print(f"  bias/leak word hits: { {k: v for k, v in hits.items() if v} or 'none'}")
    assert not any(hits.values())
    print(f"  no Stage-A summary in any prompt: "
          f"{all('{stage_a_summary}' not in v for v in PROMPTS.values())}")
    print("\n  --- E2 prompt ---")
    print("  " + PROMPTS["E2"].replace("\n", "\n  "))

    print("\n" + "=" * 92)
    print("6. CONTRASTS, FAITHFULNESS METRICS, CASES")
    for k, v in CONTRASTS.items():
        print(f"  {k:<24} {v}")
    print()
    for k, v in FAITHFULNESS.items():
        print(f"  {k:<26} {v}")
    print()
    for k in CASES:
        print(f"  case {k}")
    print(f"\n  success criterion: {SUCCESS_CRITERION}")

    print("\n" + "=" * 92)
    print("7. FULL-CORPUS EXTRACTION STATS (computed before freezing)")
    corpus, elig_ids = {}, {}
    for lb in ("fake", "real"):
        nreg, locs, ar, eids = [], Counter(), [], []
        for s in S:
            ev = byid[s["sample_id"]]["evidence"][f"{lb}_own"]
            m = np.load(ev["npz"])["map"].astype(np.float32)
            st = abstract(m, ev["score"])
            nreg.append(len(st["regions"]))
            for r in st["regions"]:
                locs[r["grid_location"]] += 1
                ar.append(r["area_ratio"])
            if st["regions"] and MIRROR[st["regions"][0]["grid_location"]] != \
                    st["regions"][0]["grid_location"]:
                eids.append(s["sample_id"])
        corpus[lb] = dict(region_count_dist={str(k): v for k, v in
                                            sorted(Counter(nreg).items())},
                          empty_lists=int(sum(1 for x in nreg if x == 0)),
                          mean_regions=round(float(np.mean(nreg)), 3),
                          area_ratio_median=round(float(np.median(ar)), 5) if ar else None,
                          top_locations=locs.most_common(),
                          e4_eligible=len(eids))
        elig_ids[lb] = eids
        c = corpus[lb]
        print(f"  [{lb}] regions {c['region_count_dist']}  empty {c['empty_lists']}"
              f"/{len(S)}  mean {c['mean_regions']}  area median "
              f"{c['area_ratio_median']}")
        print(f"        top locations {c['top_locations'][:4]}")
        print(f"        E4-eligible (primary location moves under mirror): "
              f"{c['e4_eligible']}/{len(S)}")
    print("\n  DISCLOSED LIMITATION: the 3x3 mirror leaves the centre column and")
    print("  'center' invariant, so shift-responsiveness can only be measured on the")
    print(f"  eligible subset (fake {corpus['fake']['e4_eligible']}, real "
          f"{corpus['real']['e4_eligible']}). This eligibility rule is frozen here,")
    print("  BEFORE any MLLM inference, and is not chosen from results.")
    print("  NOTE: fake maps yield a non-empty region list in "
          f"{len(S)-corpus['fake']['empty_lists']}/{len(S)} cases vs "
          f"{len(S)-corpus['real']['empty_lists']}/{len(S)} for real, so the mere")
    print("  presence of regions carries some label information; real images average")
    print(f"  {corpus['real']['mean_regions']} regions vs {corpus['fake']['mean_regions']}"
          " for fake, i.e. MORE but much smaller regions.")

    print("\n" + "=" * 92)
    print("8. BUDGET")
    n = len(CONDITIONS) * 2 * len(S)
    print(f"  {len(CONDITIONS)} conditions x 2 labels x {len(S)} sources = {n}")
    print(f"  7B at ~2.7 s/inf -> ~{n*2.7/60:.0f} min")
    print(f"  E0/E2/E4 single image, E1/E3 two images")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="Evidence Abstraction Layer experiment",
        question="Does converting the low-level forensic map into explicit structured "
                 "spatial evidence let the MLLM use spatial evidence more effectively "
                 "and more faithfully?",
        abstraction_layer=dict(
            properties=["deterministic", "training-free", "no ground truth",
                        "no second LLM", "never reads the authenticity label"],
            binary_threshold=BIN_THR, connectivity=CONNECTIVITY,
            min_area_ratio=MIN_AREA_RATIO, top_k=TOP_K, rank_by=RANK_BY,
            coordinates="normalised [0,1], origin top-left",
            grid="3x3 CELL_NAMES reused from forensic_tools.py",
            grid_rule="component centroid determines the primary location even when "
                      "the component spans several cells",
            empty_case="regions: [] — regions are never invented",
            rounding=ROUND,
            implementation_note="4-connected BFS implemented in numpy/stdlib because "
                                "scipy is absent from the frozen qwen_vl environment "
                                "and must not be installed"),
        schema_example=dict(image_manipulation_score=0.997, regions=[dict(
            grid_location="bottom-right", area_ratio=0.031, mean_probability=0.84,
            max_probability=0.998, centroid=[0.76, 0.71])]),
        conditions=CONDITIONS, prompts=PROMPTS, prompt_hashes=ph,
        output_schema=dict(final_verdict="real|fake",
                           referenced_regions="list of 3x3 grid locations, [] if none",
                           reason="one or two sentences"),
        mirror_transform=dict(grid=MIRROR, centroid="x' = 1 - x",
                              unchanged=["area_ratio", "mean_probability",
                                         "max_probability", "image_manipulation_score"],
                              note="only WHERE the evidence says the anomaly is "
                                   "changes; the original image is never mirrored"),
        contrasts=CONTRASTS, faithfulness_metrics=FAITHFULNESS,
        interpretation_cases=CASES, success_criterion=SUCCESS_CRITERION,
        model="7B primary; 32B deferred pending 7B results",
        stage_a_policy="no Stage A, no Stage-A summary",
        n_sources=len(S), inference_budget=dict(total=n, per_cell=len(S)),
        samples_inherited_from="trufor_score_map_conflict_frozen.json",
        score_presentation=prev["score_presentation"],
        map_rendering=prev["map_rendering"],
        decoding=cfg["model"]["generation"],
        min_pixels=cfg["model"]["min_pixels"], max_pixels=cfg["model"]["max_pixels"],
        prior_results=dict(
            seven_b=dict(C10=0.755, F1=0.522, F4=0.495,
                         spatial_correspondence=-0.027),
            thirty_two_b=dict(B0=0.978, B1=0.880, B3=0.853,
                              spatial_correspondence=-0.027),
            delta_spatial_correspondence="+0.000 [-0.065,+0.065]"),
        corpus_stats=corpus,
        e4_eligible_ids=elig_ids,
        e4_eligibility_rule="shift responsiveness is evaluated only where the primary "
                            "region's grid location changes under the mirror; the 3x3 "
                            "mirror leaves the centre column invariant. Frozen before "
                            "inference, not selected from results.",
        forbidden=["prompt tuning", "GT in the abstraction layer", "second LLM",
                   "training", "threshold tuning on MLLM results", "confidence map"],
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    conflict_protocol=sha1f(src)),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      evidence=byid[s["sample_id"]]["evidence"]) for s in S])
    od = os.path.join(R, "trufor_abstraction")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_evidence_abstraction_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} sources, {n} inferences")


if __name__ == "__main__":
    main()
