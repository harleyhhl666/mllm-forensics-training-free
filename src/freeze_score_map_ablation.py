"""Freeze the TruFor score/map causal decomposition (2x2 factorial). No inference.

Same 184 sources, same pairing, same TruFor outputs, same decoding as the previous
round. The ONLY protocol change is that Stage B no longer receives any Stage-A
summary: that text is derived from the map and would leak spatial evidence into the
map-absent conditions.

Because of that change the previous round's C00/C11 results are NOT reusable -- its
Stage-B prompts both embedded {stage_a_summary} (verified). All four conditions are
therefore rerun: 4 x 368 = 1472 inferences.

Factors:  S = image-level score absent/present,  M = localization map absent/present
  C00 image only          C10 score only
  C01 map only            C11 score + map
"""
import argparse, hashlib, json, os, sys
import numpy as np, yaml

SEED = 20260918

# ---------------------------------------------------------------- prompt parts
# Shared spine so the four conditions differ ONLY in which evidence modality is
# described and supplied. Neutral wording; no reliability claims, no benchmark
# numbers, no threshold, no discretisation of the score.
HEAD = "You are performing a digital image forensics analysis.\n\n"

DESC_SCORE = ("An automated forensic analysis of the image is provided as a "
              "whole-image score in [0,1], where higher values correspond to "
              "stronger manipulation-related evidence. Treat it as auxiliary "
              "evidence to be interpreted together with the original image, not as "
              "proof.")

DESC_MAP = ("An automated forensic analysis of the image is provided as a "
            "localization map, which indicates where the tool assigns higher "
            "manipulation-related response. Treat it as auxiliary evidence to be "
            "interpreted together with the original image, not as proof.")

DESC_BOTH = ("An automated forensic analysis of the image is provided. The "
             "accompanying localization map indicates where the tool assigns higher "
             "manipulation-related response. A whole-image score in [0,1] is also "
             "provided, where higher values correspond to stronger "
             "manipulation-related evidence. Treat these outputs as auxiliary "
             "evidence to be interpreted together with the original image, not as "
             "proof.")

SCORE_LINE = "TruFor image-level manipulation score: {score}"

TAIL = """Determine whether this image is an authentic photograph or has been digitally
altered.

Report strictly as JSON with these keys:
{
  "final_verdict": "real" or "fake",
  "reason": "one or two sentences justifying the verdict"
}
Output JSON only."""

C00 = HEAD + "You are given an image under examination.\n\n" + TAIL
C10 = HEAD + DESC_SCORE + "\n\n" + SCORE_LINE + "\n\n" + TAIL
C01 = HEAD + DESC_MAP + "\n\n" + TAIL
C11 = HEAD + DESC_BOTH + "\n\n" + SCORE_LINE + "\n\n" + TAIL

PROMPTS = dict(C00=C00, C10=C10, C01=C01, C11=C11)

CONDITIONS = {
    "C00": dict(score=False, map=False, n_images=1, prompt="C00"),
    "C10": dict(score=True,  map=False, n_images=1, prompt="C10"),
    "C01": dict(score=False, map=True,  n_images=2, prompt="C01"),
    "C11": dict(score=True,  map=True,  n_images=2, prompt="C11"),
}

BANNED = ("reliable", "accurate", "state-of-the-art", "trusted", "trustworthy",
          "auroc", "benchmark", "proven", "prioritize", "prioritise",
          "should trust", "donor", "ground truth", "probability of being fake",
          "high confidence", "threshold")

CONTRASTS = {
    "score_effect_without_map": "C10 - C00",
    "score_effect_with_map": "C11 - C01",
    "map_effect_without_score": "C01 - C00",
    "map_effect_with_score": "C11 - C10",
    "interaction_a": "(C11 - C01) - (C10 - C00)",
    "interaction_b": "(C11 - C10) - (C01 - C00)",
    "applied_to": ["fake recall", "real FPR", "Youden J"],
    "stats": "paired effect size -> source-level bootstrap 95% CI -> McNemar p "
             "(supplementary)",
}

CASES = {
    "1_score_dominated": "C10 ~ C11, C10 >> C00, C01 ~ C00, small map main effect -> "
                         "improvement is primarily driven by the image-level score; "
                         "spatial evidence contributes little to the verdict",
    "2_independent_spatial": "C01 > C00 and C11 > C10 with map-effect CI excluding "
                             "zero -> spatial localization contributes independently "
                             "of the scalar score",
    "3_synergy": "both single-modality conditions improve, C11 clearly exceeds both, "
                 "interaction > 0 -> scalar and spatial evidence are jointly "
                 "integrated",
    "4_map_harms": "C10 > C11, or adding the map raises real FPR -> spatial "
                   "visualization interferes with an otherwise useful scalar signal",
}

EXPLANATION_AUDIT = dict(
    method="deterministic keyword rules over the saved reason strings; NO LLM is "
           "asked to score explanations subjectively",
    categories=dict(
        mentions_score=["score", "0.", "value", "scalar"],
        mentions_region=["region", "area", "top-", "bottom-", "center", "left",
                         "right", "corner"],
        mentions_map=["map", "heatmap", "visualization", "highlighted", "response"],
        mentions_semantic_content=["object", "person", "face", "sky", "texture",
                                   "edge", "background", "building", "animal",
                                   "shadow", "lighting"],
        mentions_consistency=["consistent", "corresponds", "matches", "aligns",
                              "agree", "in line with"]),
    key_comparison="whether C11 reasons couple a region to image content "
                   "('map indicates X and this corresponds to Y') rather than only "
                   "restating the score",
    status="SECONDARY; never used to change the primary conclusion",
)


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    prev = json.load(open(os.path.join(R, "trufor_mllm", "trufor_mllm_frozen.json")))

    print("=" * 88)
    print("1. REUSE CHECK — can the previous C00/C11 be reused?")
    for k in ("STAGE_B_NOTOOL", "STAGE_B_TOOL"):
        has = "{stage_a_summary}" in prev["prompts"][k]
        print(f"  previous {k:<16} embeds Stage-A summary: {has}")
    print("  This round forbids any Stage-A summary in Stage B (protocol section 9),")
    print("  because that text is map-derived and would leak spatial evidence into")
    print("  the map-absent conditions.")
    print("  => previous C00/C11 are NOT prompt-compatible; ALL FOUR conditions rerun.")

    print("\n" + "=" * 88)
    print("2. PROMPT AUDIT")
    for k, v in PROMPTS.items():
        low = v.lower()
        hits = [w for w in BANNED if w in low]
        print(f"  {k}: banned hits {hits or 'none'}")
        assert not hits, f"{k}: {hits}"
    assert "{score}" in C10 and "{score}" in C11
    assert "{score}" not in C00 and "{score}" not in C01
    print("  score line present only in C10/C11: True")
    assert "localization map" in C01 and "localization map" in C11
    assert "localization map" not in C00 and "localization map" not in C10
    print("  map described only in C01/C11      : True")
    for k, v in PROMPTS.items():
        assert v.endswith(TAIL), k
    print("  identical task tail in all four    : True")
    assert "{stage_a_summary}" not in "".join(PROMPTS.values())
    print("  no Stage-A summary anywhere        : True")
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in PROMPTS.items()}
    for k, v in ph.items():
        print(f"  sha1({k}) = {v}")

    print("\n" + "=" * 88)
    print("3. SAMPLES (identical to the previous round)")
    S = prev["samples"]
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    print(f"  n_samples == n_sources: {len({s['coco_id'] for s in S}) == len(S)}")
    print(f"  images: {len(S)} fake + {len(S)} paired real = {2*len(S)}")
    print(f"  sampling rule (inherited): {prev['sampling_rule']}")
    # evidence reuse: correct (own) packages only; donor is not part of this design
    miss = [s["sample_id"] for s in S
            if not (os.path.exists(s["evidence"]["fake_own"]["npz"])
                    and os.path.exists(s["evidence"]["fake_own"]["map_png"])
                    and os.path.exists(s["evidence"]["real_own"]["npz"])
                    and os.path.exists(s["evidence"]["real_own"]["map_png"]))]
    print(f"  own-evidence files present for all sources: {not miss}")
    assert not miss
    fk = np.array([s["evidence"]["fake_own"]["score"] for s in S])
    rl = np.array([s["evidence"]["real_own"]["score"] for s in S])
    print(f"  fake own score: mean {fk.mean():.4f} median {np.median(fk):.4f}")
    print(f"  real own score: mean {rl.mean():.4f} median {np.median(rl):.4f}")
    print("  NOTE: only the CORRECT (own) evidence package is used this round; the")
    print("  donor package belongs to the previous design and is not part of the 2x2.")

    print("\n" + "=" * 88)
    print("4. BUDGET")
    n = 4 * 2 * len(S)
    print(f"  4 conditions x (fake + real) x {len(S)} sources = {n} inferences")
    print(f"  at ~2.7 s each: ~{n*2.7/60:.0f} min")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="TruFor score/map causal decomposition (2x2 factorial, Stage-B only)",
        question="Is the MLLM verdict improvement driven by the image-level scalar "
                 "score or by the spatial localization map?",
        hypothesis_under_test="improvement is primarily score-driven; spatial "
                              "localization contributes substantially less",
        seed=SEED, n_sources=len(S), n_images=2 * len(S),
        samples_identical_to="runs/trufor_mllm/trufor_mllm_frozen.json",
        previous_protocol_sha1=sha1f(os.path.join(R, "trufor_mllm",
                                                  "trufor_mllm_frozen.json")),
        reuse_decision=dict(
            reusable=False,
            reason="both previous Stage-B prompts embed {stage_a_summary}; this round "
                   "forbids it, so prompt semantics differ and old results must not "
                   "be mixed in",
            consequence="all four conditions rerun, 1472 inferences"),
        stage_a_summary_policy="NOT provided in any condition — it is map-derived and "
                               "would contaminate the map-absent conditions",
        conditions=CONDITIONS, prompts=PROMPTS, prompt_hashes=ph,
        score_presentation="'TruFor image-level manipulation score: X.XXX' — raw "
                           "continuous value, 3 decimals; no high/low label, no "
                           "threshold, no percentage, no fake-probability wording",
        map_rendering=prev["map_rendering"],
        evidence_used="own (correct) package only; donor packages are not part of "
                      "this factorial design",
        decoding=cfg["model"]["generation"],
        min_pixels=cfg["model"]["min_pixels"], max_pixels=cfg["model"]["max_pixels"],
        metrics=["fake recall/TPR", "real FPR", "specificity", "Youden J = TPR-FPR"],
        contrasts=CONTRASTS, prereg_cases=CASES,
        explanation_audit=EXPLANATION_AUDIT,
        forbidden_this_round=["mitigation design", "prompt tuning", "confidence map",
                              "TruFor modification", "threshold calibration",
                              "new multi-step reasoning", "Stage-A reuse"],
        prior_round_reference=dict(
            note="previous round included a Stage-A summary, so it is a different "
                 "protocol; consistency is checked descriptively only",
            T0=dict(recall=0.201, fpr=0.125, j=0.076),
            T1_correct=dict(recall=0.804, fpr=0.011, j=0.793),
            T2_donor=dict(recall=0.717, fpr=0.011, j=0.707)),
        trufor_standalone=dict(auroc=0.9845, tpr_at_fpr05=0.955, tpr_at_fpr10=0.965),
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    prev_protocol=sha1f(os.path.join(R, "trufor_mllm",
                                                     "trufor_mllm_frozen.json"))),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      mask=s["mask"],
                      evidence=dict(fake_own=s["evidence"]["fake_own"],
                                    real_own=s["evidence"]["real_own"]))
                 for s in S])
    od = os.path.join(R, "trufor_score_map")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_score_map_ablation_frozen.json")
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} sources, {n} inferences")


if __name__ == "__main__":
    main()
