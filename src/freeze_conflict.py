"""Freeze the score-map conflict intervention protocol. No MLLM inference here.

Same 184 sources. Stage-B only, no Stage-A summary. The image-level score is ALWAYS
the target's own score; only the accompanying visualization is manipulated, so every
fake contrast isolates a property of the map while the scalar evidence is held fixed.

Fake conditions (6):
  F1 own map                     baseline (= previous C11)
  F2 blank map                   all-zero values, identical rendering pipeline
  F3 donor map                   different coco_id, same label; OWN score retained
  F4 shifted own map             cyclic shift by 50% width; values/statistics intact
  F5 amplified own map           display = clip(raw*2, 0, 1); visualization only
  F6 binary own map              display = (raw > 0.5); visualization only

Real controls (3): R1 own map, R2 blank map, R3 donor real map.

F5/F6 alter only how the map is DISPLAYED. They do not represent new TruFor output
and are labelled visualization manipulations throughout.
"""
import argparse, hashlib, json, os, sys
import numpy as np, yaml

SEED = 20260918
SHIFT_FRAC = 0.50          # cyclic horizontal shift, fraction of width
AMPLIFY_K = 2.0            # F5 multiplier, pre-registered
BINARY_THR = 0.5           # F6 display threshold, pre-registered

TRANSFORMS = dict(
    own=dict(op="identity", note="raw TruFor map as rendered in earlier rounds"),
    blank=dict(op="zeros", note="all-zero map; same shape, same colormap, same "
                                "fixed 0-1 scale; the prompt never says 'blank'"),
    donor=dict(op="substitute", note="localization map of a different coco_id, same "
                                     "label; the TARGET's own score is retained"),
    shifted=dict(op=f"np.roll(map, int({SHIFT_FRAC}*W), axis=1)",
                 note="cyclic horizontal shift by 50% of width; value distribution, "
                      "global intensity and texture statistics are preserved exactly, "
                      "only spatial correspondence with the image is destroyed; NO "
                      "renormalisation"),
    amplified=dict(op=f"clip(map * {AMPLIFY_K}, 0, 1)",
                   note="VISUALIZATION MANIPULATION ONLY; not a new TruFor output"),
    binary=dict(op=f"(map > {BINARY_THR}).astype(float)",
                note="VISUALIZATION MANIPULATION ONLY; threshold is for visual "
                     "representation, never for performance tuning"),
)

CONDITIONS = {
    "F1_own":       dict(label="fake", map="own",       score="own"),
    "F2_blank":     dict(label="fake", map="blank",     score="own"),
    "F3_donor":     dict(label="fake", map="donor",     score="own"),
    "F4_shifted":   dict(label="fake", map="shifted",   score="own"),
    "F5_amplified": dict(label="fake", map="amplified", score="own"),
    "F6_binary":    dict(label="fake", map="binary",    score="own"),
    "R1_own":       dict(label="real", map="own",       score="own"),
    "R2_blank":     dict(label="real", map="blank",     score="own"),
    "R3_donor":     dict(label="real", map="donor",     score="own"),
}

CONTRASTS = {
    "blank_effect": "F2 - F1  (is the real map itself suppressing the score, or is "
                    "any second image enough?)",
    "mismatch_effect": "F3 - F1  (does a wrong-image map hurt further?)",
    "shift_effect": "F4 - F1  (cleanest correspondence test: same map statistics, "
                    "broken alignment)",
    "saliency_effect_amplified": "F5 - F1",
    "saliency_effect_binary": "F6 - F1",
    "real_controls": ["R2 - R1", "R3 - R1"],
    "stats": "paired effect size -> source-level bootstrap 95% CI -> McNemar p "
             "(supplementary)",
}

MECHANISMS = {
    "A_weak_visual_veto": "blank > own AND amplified/binary > own, while "
                          "shifted/donor need not be clearly worse -> the model reads "
                          "visually weak or small localized maps as counter-evidence "
                          "against a high scalar score",
    "B_spatial_mismatch": "own > donor AND own > shifted, while blank need not be "
                          "better -> the model is sensitive to map-image spatial "
                          "correspondence",
    "C_generic_interference": "own ~ donor ~ shifted ~ blank, all below score-only -> "
                              "a second forensic visualization causes generic "
                              "multimodal interference independent of map content",
    "D_mixed": "several mechanisms may hold at once; do not force a single label",
}

EXPLANATION_RULES = dict(
    method="deterministic keyword rules over saved reason strings; no LLM scoring",
    patterns=dict(
        score_high=["score of 0.9", "score of 1.0", "high score", "score indicates",
                    "high likelihood", "score suggests"],
        map_shows_little=["does not indicate", "no significant", "little", "minimal",
                          "does not show", "no clear", "lacks", "absence of",
                          "no obvious", "uniform"],
        map_inconsistent=["inconsistent", "contradict", "does not correspond",
                          "does not align", "conflict", "mismatch", "disagree"],
        region_small=["small", "localized", "localised", "limited", "confined",
                      "isolated", "tiny", "narrow"],
        map_supports=["supports", "consistent with", "corroborat", "agrees",
                      "confirms", "aligns with", "in line with"]),
    key_question="among fake samples judged Real under F1, how often does the reason "
                 "say the map is weak/insignificant, and does that fall under "
                 "F2/F5/F6?",
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
    prev = json.load(open(os.path.join(R, "trufor_score_map",
                                       "trufor_score_map_ablation_frozen.json")))
    S = prev["samples"]

    print("=" * 90)
    print("1. SAMPLES (unchanged)")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    print(f"  n_samples == n_sources: {len({s['coco_id'] for s in S}) == len(S)}")
    print(f"  inherited from: trufor_score_map_ablation_frozen.json "
          f"(sha1 {sha1f(os.path.join(R,'trufor_score_map','trufor_score_map_ablation_frozen.json'))[:12]})")

    print("\n" + "=" * 90)
    print("2. PROMPTS — reused verbatim from the 2x2 freeze (C11 for map conditions)")
    P = dict(MAP_AND_SCORE=prev["prompts"]["C11"])
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in P.items()}
    print(f"  every condition here supplies BOTH a score and a map image, so all use")
    print(f"  the frozen C11 prompt unchanged: sha1 {ph['MAP_AND_SCORE']}")
    assert ph["MAP_AND_SCORE"] == prev["prompt_hashes"]["C11"]
    assert "{stage_a_summary}" not in P["MAP_AND_SCORE"]
    print(f"  matches the 2x2 C11 hash        : True")
    print(f"  contains no Stage-A summary     : True")
    print(f"  F1 is therefore an exact protocol replicate of the 2x2 C11 cell")
    print(f"  NOTE: the prompt never mentions blank/shifted/amplified/binary/donor;")
    print(f"  the model is told only that a localization map is provided.")

    print("\n" + "=" * 90)
    print("3. DONOR MAPPING (frozen, label-matched)")
    ids = [s["sample_id"] for s in S]
    cid = {s["sample_id"]: s["coco_id"] for s in S}
    n, half = len(ids), len(ids) // 2
    donors = {}
    for k, pid in enumerate(ids):
        for st in range(n):
            c = ids[(k + half + st) % n]
            if c != pid and cid[c] != cid[pid]:
                donors[pid] = c
                break
    bad = [p for p, d in donors.items() if cid[d] == cid[p] or d == p]
    print(f"  assigned {len(donors)}  same-coco/self {len(bad)}")
    assert not bad
    prev_d = {s["sample_id"]: s.get("donor_sample_id") for s in
              json.load(open(os.path.join(R, "trufor_mllm",
                                          "trufor_mllm_frozen.json")))["samples"]}
    same_as_prev = sum(1 for p in ids if prev_d.get(p) == donors[p])
    print(f"  identical to the earlier round's donor mapping: {same_as_prev}/{len(ids)}")

    print("\n" + "=" * 90)
    print("4. TRANSFORMS (pre-registered, applied identically to every sample)")
    for k, v in TRANSFORMS.items():
        print(f"  {k:<10} {v['op']}")
        print(f"             {v['note']}")

    print("\n" + "=" * 90)
    print("5. TRANSFORM VERIFICATION on a real map")
    z = np.load(S[0]["evidence"]["fake_own"]["npz"])
    m = z["map"].astype(np.float32)
    H, W = m.shape
    sh = np.roll(m, int(SHIFT_FRAC * W), axis=1)
    am = np.clip(m * AMPLIFY_K, 0, 1)
    bi = (m > BINARY_THR).astype(np.float32)
    bl = np.zeros_like(m)
    print(f"  sample {S[0]['sample_id']}  shape {m.shape}")
    print(f"  {'variant':<12}{'min':>8}{'max':>8}{'mean':>9}{'>0.5 frac':>11}")
    for nm, a in (("own", m), ("blank", bl), ("shifted", sh), ("amplified", am),
                  ("binary", bi)):
        print(f"  {nm:<12}{a.min():>8.4f}{a.max():>8.4f}{a.mean():>9.5f}"
              f"{(a>0.5).mean():>11.5f}")
    print(f"\n  shifted preserves the value multiset exactly: "
          f"{np.array_equal(np.sort(m.ravel()), np.sort(sh.ravel()))}")
    print(f"  shifted differs spatially from own          : {not np.array_equal(m, sh)}")
    print(f"  amplified/binary change appearance only, raw map untouched on disk")

    print("\n" + "=" * 90)
    print("6. SUPPRESSION SUBGROUP from the 2x2 round")
    sup = json.load(open(os.path.join(R, "trufor_score_map",
                                      "suppression_ids.json")))
    print(f"  cases where score-only said Fake but score+map said Real: {len(sup)}")
    print(f"  (the earlier report's '42' was the NET recall difference 139-97;")
    print(f"   the actual flip count is {len(sup)} suppressed with 0 gained)")
    trs = np.array([next(s for s in S if s["sample_id"] == p)["tamper_ratio"]
                    for p in sup])
    allt = np.array([s["tamper_ratio"] for s in S])
    print(f"  tamper_ratio  suppressed: median {np.median(trs):.4f}  "
          f"all: median {np.median(allt):.4f}")

    print("\n" + "=" * 90)
    print("7. BUDGET")
    nf = 6 * len(S)
    nr = 3 * len(S)
    print(f"  fake conditions 6 x {len(S)} = {nf}")
    print(f"  real controls   3 x {len(S)} = {nr}")
    print(f"  TOTAL {nf+nr} inferences  (~{(nf+nr)*2.2/60:.0f} min at 2.2 s)")
    print(f"  note: F1 repeats the 2x2 C11 cell under an identical prompt, giving a")
    print(f"  built-in replication check rather than a reused number.")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="TruFor score-map conflict causal intervention (Stage-B only)",
        question="When the scalar score and the spatial map disagree, which governs "
                 "the MLLM's final verdict?",
        design="score is ALWAYS the target's own; only the visualization varies",
        seed=SEED, n_sources=len(S),
        samples_inherited_from="trufor_score_map_ablation_frozen.json",
        stage_a_summary_policy="NOT used in any condition",
        conditions=CONDITIONS, transforms=TRANSFORMS,
        transform_params=dict(shift_fraction=SHIFT_FRAC, amplify_k=AMPLIFY_K,
                              binary_threshold=BINARY_THR),
        visualization_manipulation_disclaimer=(
            "F5 amplified and F6 binary change ONLY how the map is displayed. They do "
            "not represent new TruFor outputs and must never be reported as improved "
            "tool performance."),
        prompts=P, prompt_hashes=ph,
        prompt_note="all conditions supply a score and a map image, so the frozen C11 "
                    "prompt is reused verbatim; the prompt never describes the "
                    "transform",
        score_presentation=prev["score_presentation"],
        map_rendering=prev["map_rendering"],
        donor_mapping={p: donors[p] for p in ids},
        donor_rule="label-matched, donor_coco_id != target_coco_id, never self, "
                   "frozen before inference",
        contrasts=CONTRASTS, mechanisms=MECHANISMS,
        explanation_audit=EXPLANATION_RULES,
        suppression_subgroup=dict(
            definition="2x2 round: C10 (score only) = fake AND C11 (score+map) = real",
            n=len(sup), ids=sup,
            correction="an earlier report said 42; that was the net recall difference "
                       f"(139-97). The true flip count is {len(sup)}, with 0 reverse "
                       "flips.",
            planned_analysis=["tamper_ratio", "map active area", "max/mean map prob",
                              "score", "reason text",
                              "recovery under F2 blank / F5 amplified / F6 binary"],
            status="SECONDARY mechanistic analysis; the primary analysis stays on all "
                   f"{len(S)} sources"),
        prior_2x2=dict(C00=dict(recall=0.038, fpr=0.016, j=0.022),
                       C10=dict(recall=0.755, fpr=0.011, j=0.745),
                       C01=dict(recall=0.098, fpr=0.033, j=0.065),
                       C11=dict(recall=0.527, fpr=0.011, j=0.516),
                       map_effect_with_score_J=-0.228, interaction_J=-0.272),
        decoding=cfg["model"]["generation"],
        min_pixels=cfg["model"]["min_pixels"], max_pixels=cfg["model"]["max_pixels"],
        forbidden_this_round=["prompt tuning", "threshold optimization", "retraining",
                              "gate", "verifier", "LoRA", "confidence map",
                              "new detector", "new dataset"],
        inference_budget=dict(fake=nf, real=nr, total=nf + nr),
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    prev_2x2=sha1f(os.path.join(
                        R, "trufor_score_map",
                        "trufor_score_map_ablation_frozen.json"))),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      mask=s["mask"], donor_sample_id=donors[s["sample_id"]],
                      evidence=s["evidence"]) for s in S])
    od = os.path.join(R, "trufor_conflict")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_score_map_conflict_frozen.json")
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} sources, {nf+nr} inferences")


if __name__ == "__main__":
    main()
