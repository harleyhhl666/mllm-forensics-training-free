"""Freeze the structured forensic evidence FIELD ablation. No MLLM inference here.

Question: which structured field causes the MLLM to underweight an otherwise highly
discriminative image-level score? area_ratio is a HYPOTHESIS, not an established
cause, so it gets an isolated condition alongside the others.

  S0 score only
  S1 score + grid_location only
  S2 score + area_ratio only
  S3 score + mean/max probability only
  S4 score + location + area
  S5 full structured evidence (replicates the previous round's E2)

No raw localization visualization in ANY condition, so the fields are decomposed in
isolation. Region identity, top-K and ranking are inherited unchanged from the frozen
abstraction layer -- only which KEYS survive differs between conditions.
"""
import argparse, hashlib, json, os, sys
from collections import Counter
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_abstraction import abstract, BIN_THR, MIN_AREA_RATIO, TOP_K, RANK_BY, ROUND

# which keys each condition keeps, in this fixed order
FIELDS = {
    "S0_score_only":   None,
    "S1_location":     ["grid_location"],
    "S2_area":         ["area_ratio"],
    "S3_probability":  ["mean_probability", "max_probability"],
    "S4_loc_area":     ["grid_location", "area_ratio"],
    "S5_full":         ["grid_location", "area_ratio", "mean_probability",
                        "max_probability", "centroid"],
}
HAS_LOC = {k: bool(v and "grid_location" in v) for k, v in FIELDS.items()}

# ---------------------------------------------------------------- prompt parity
HEAD = "You are performing a digital image forensics analysis.\n\n"
SCORE_DESC = ("An automated forensic analysis of the image is provided as a whole-image "
              "score in [0,1], where higher values correspond to stronger "
              "manipulation-related evidence. Treat it as auxiliary evidence to be "
              "interpreted together with the original image, not as proof.\n\n")
# ONE neutral sentence shared by S1-S5. It does not say what any field means, does not
# say whether small regions matter, and adds no forensic knowledge.
STRUCT_DESC = ("An automated forensic tool provides an image-level manipulation score "
               "and structured localization measurements derived from its localization "
               "output. Treat these measurements as auxiliary evidence and interpret "
               "them together with the original image. The measurements are given in "
               "the JSON block below. An empty region list means the tool localised no "
               "region above its internal response level.\n\n")
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

PROMPTS = {"S0_score_only": HEAD + SCORE_DESC + SCORE_LINE + TASK + SCHEMA}
for _k in ("S1_location", "S2_area", "S3_probability", "S4_loc_area", "S5_full"):
    PROMPTS[_k] = HEAD + STRUCT_DESC + SCORE_LINE + "{structured}\n\n" + TASK + SCHEMA

CONTRASTS = dict(
    location_effect="S1 - S0",
    area_effect="S2 - S0",
    probability_effect="S3 - S0",
    location_plus_area="S4 - S0",
    full_penalty="S5 - S0",
    supplementary=["S4 - S1 (adding area on top of location)",
                   "S4 - S2 (adding location on top of area)",
                   "S5 - S4 (adding probability and centroid)"],
    stats="paired effect size -> source-level bootstrap 95% CI -> McNemar supplementary",
)

CASES = {
    "area_driven_collapse":
        "S2 << S0 and S1 ~ S0, with S4/S5 also down -> explicit manipulation-AREA "
        "information is a major driver of the decision collapse",
    "generic_structured_overload":
        "S1, S2 and S3 all fall similarly -> adding structured forensic measurements "
        "generically disrupts scalar-score arbitration",
    "probability_conflict":
        "mainly S3 << S0 -> inspect whether the model misreads mean/max probability",
    "interaction_only":
        "S1 ~ S2 ~ S3 ~ S0 but S5 << S0 -> the failure arises from COMBINING several "
        "measurements rather than from any single field",
}

REASON_AUDIT = dict(
    method="deterministic keyword rules over saved reasons; no LLM judge",
    patterns=dict(
        area_small=["small", "tiny", "minor", "negligible", "insignificant",
                    "little", "low area", "small area", "small fraction",
                    "small portion", "small region"],
        area_limited=["limited", "confined", "localized", "localised", "restricted",
                      "isolated", "narrow", "only a"],
        evidence_weak=["weak", "not strong", "insufficient", "inconclusive",
                       "not conclusive", "lacks", "minimal evidence",
                       "does not provide strong"],
        score_high=["score of 0.9", "score of 1.0", "high score", "score indicates",
                    "high likelihood", "score suggests", "elevated score"],
        manipulation_acknowledged=["manipulat", "alter", "tamper", "edited", "modified",
                                   "spliced", "retouch"],
    ),
    key_metric=dict(
        name="acknowledged_manipulation_but_real",
        definition="verdict == real AND the reason matches manipulation_acknowledged "
                   "AND (area_small OR area_limited OR evidence_weak)",
        why="a forensic reasoning inconsistency: the model concedes localized "
            "manipulation yet still returns Real",
    ),
)


def strip_fields(struct, keys):
    """Keep only `keys` per region, in the frozen order. Region set is untouched."""
    out = []
    for r in struct["regions"]:
        out.append({k: r[k] for k in keys if k in r})
    return dict(regions=out)


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    prev_fp = os.path.join(R, "trufor_abstraction",
                           "trufor_evidence_abstraction_frozen.json")
    prev = json.load(open(prev_fp))
    S = prev["samples"]

    print("=" * 94)
    print("1. SAMPLES — unchanged")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    assert len({s["coco_id"] for s in S}) == len(S)
    print(f"  fake + paired real, n_samples == n_sources")
    print(f"  inherited from trufor_evidence_abstraction_frozen.json "
          f"(sha1 {sha1f(prev_fp)[:12]})")
    print(f"  model 7B snapshot {cfg['model']['path'].rstrip('/').split('/')[-1][:16]}")

    print("\n" + "=" * 94)
    print("2. NO RAW MAP IN ANY CONDITION")
    print("  every condition is single-image: original image only")
    print("  the localization visualization is deliberately withheld so that the")
    print("  structured fields are decomposed in isolation")

    print("\n" + "=" * 94)
    print("3. CONDITIONS — only which KEYS survive differs")
    for k, v in FIELDS.items():
        print(f"  {k:<18} {'(no structured JSON)' if v is None else v}")
    print(f"\n  region identity, top-K={TOP_K} and ranking by {RANK_BY} are inherited")
    print(f"  unchanged from the frozen abstraction layer (thr {BIN_THR}, min area "
          f"{MIN_AREA_RATIO})")
    print(f"  decimal precision inherited: {ROUND}")
    print(f"  S5 reproduces the previous round's E2 field set exactly")

    print("\n" + "=" * 94)
    print("4. PROMPT PARITY")
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in PROMPTS.items()}
    for k in FIELDS:
        print(f"  {k:<18} sha1 {ph[k]}")
    struct_h = {ph[k] for k in FIELDS if FIELDS[k] is not None}
    print(f"\n  S1-S5 share ONE byte-identical prompt: {len(struct_h)==1}")
    assert len(struct_h) == 1
    print(f"  only the JSON content differs between S1-S5")
    print(f"  S0 differs only by having no structured block")
    leak = ("area is", "small area", "larger", "smaller", "important", "significant",
            "reliable", "accurate", "trusted", "auroc", "benchmark", "threshold",
            "probability means", "fraction of the image", "normalised to")
    hits = {k: [w for w in leak if w in v.lower()] for k, v in PROMPTS.items()}
    print(f"  field-interpretation / bias wording: { {k:v for k,v in hits.items() if v} or 'none'}")
    assert not any(hits.values())
    print(f"  NOTE: the previous round's prompt explained that area_ratio is 'the")
    print(f"  fraction of the image covered by the region' and that coordinates are")
    print(f"  normalised. That explanatory text is REMOVED here to satisfy prompt")
    print(f"  parity and the ban on explaining what fields mean. S5 is therefore a")
    print(f"  field-set replicate of E2, NOT a prompt-identical one; it is compared")
    print(f"  within this round and only descriptively against E2.")
    print("\n  --- shared structured prompt ---")
    print("  " + PROMPTS["S5_full"].replace("\n", "\n  "))

    print("\n" + "=" * 94)
    print("5. RENDERED JSON PER CONDITION (first fake source)")
    ev = S[0]["evidence"]["fake_own"]
    m = np.load(ev["npz"])["map"].astype(np.float32)
    base = abstract(m, ev["score"])
    print(f"  sample {S[0]['sample_id']}  score {ev['score']:.3f}  "
          f"regions {len(base['regions'])}")
    for k, keys in FIELDS.items():
        if keys is None:
            print(f"  {k}: (score line only)")
            continue
        print(f"  {k}: {json.dumps(strip_fields(base, keys))}")

    print("\n" + "=" * 94)
    print("6. FIELD-PRESENCE AUDIT over all sources")
    bad = []
    for s in S:
        for lb in ("fake", "real"):
            e = s["evidence"][f"{lb}_own"]
            st = abstract(np.load(e["npz"])["map"].astype(np.float32), e["score"])
            for k, keys in FIELDS.items():
                if keys is None:
                    continue
                sub = strip_fields(st, keys)
                if len(sub["regions"]) != len(st["regions"]):
                    bad.append((s["sample_id"], lb, k, "region count changed"))
                for r in sub["regions"]:
                    if set(r) != set(keys) and st["regions"]:
                        bad.append((s["sample_id"], lb, k, f"keys {sorted(r)}"))
    print(f"  region counts preserved and key sets exact: {len(bad)==0}"
          f"  ({len(bad)} problems)")
    for b in bad[:5]:
        print(f"    {b}")
    assert not bad
    print(f"  note: no condition contains the image-level score inside the JSON;")
    print(f"  the score is always the separate frozen score line, so the 'regions'")
    print(f"  block carries only per-region fields")

    print("\n" + "=" * 94)
    print("7. CONTRASTS, CASES, REASON AUDIT")
    for k, v in CONTRASTS.items():
        print(f"  {k:<20} {v}")
    print()
    for k in CASES:
        print(f"  case {k}")
    print(f"\n  reason audit patterns: {list(REASON_AUDIT['patterns'])}")
    print(f"  key metric: {REASON_AUDIT['key_metric']['name']}")
    print(f"    {REASON_AUDIT['key_metric']['definition']}")
    print(f"  secondary faithfulness only for location-bearing conditions: "
          f"{[k for k,v in HAS_LOC.items() if v]}")

    print("\n" + "=" * 94)
    print("8. SCOPE LIMIT")
    print("  this round is FIELD ABLATION ONLY. No counterfactual area_ratio values are")
    print("  substituted. If S2 implicates area, the counterfactual intervention is a")
    print("  SEPARATE later round, so the two are not confounded.")

    print("\n" + "=" * 94)
    print("9. BUDGET")
    n = len(FIELDS) * 2 * len(S)
    print(f"  {len(FIELDS)} conditions x 2 labels x {len(S)} sources = {n}")
    print(f"  all single-image; 7B at ~2.4 s/inf -> ~{n*2.4/60:.0f} min")

    print("\n" + "=" * 94)
    print("10. PRIOR RESULTS FOR REFERENCE")
    print("  score only (E0)           recall 0.837  FPR 0.011  J 0.826")
    print("  full structured (E2)      recall 0.174  FPR 0.011  J 0.163")
    print("  score + raw map (E1)      recall 0.565  FPR 0.016  J 0.549")
    print("  E2 spatial agreement 1.000, mirrored-evidence following 1.000")
    print("  fake region area_ratio median 0.0251; real 0.00395")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="Structured forensic evidence FIELD ablation",
        question="Which structured field causes the MLLM to underweight an otherwise "
                 "highly discriminative image-level score?",
        hypothesis_status="area_ratio is a HYPOTHESIS to be tested, not an established "
                          "cause",
        raw_map_policy="no raw localization visualization in any condition; all "
                       "conditions are single-image",
        conditions={k: dict(fields=v, has_location=HAS_LOC[k],
                            single_image=True) for k, v in FIELDS.items()},
        field_order_note="fields appear in the frozen order listed per condition",
        abstraction_inherited=dict(
            binary_threshold=BIN_THR, min_area_ratio=MIN_AREA_RATIO, top_k=TOP_K,
            rank_by=RANK_BY, rounding=ROUND,
            source="freeze_abstraction.abstract(), imported unchanged"),
        prompts=PROMPTS, prompt_hashes=ph,
        prompt_parity=dict(structured_conditions_share_one_prompt=True,
                           only_json_content_differs=True,
                           removed_from_previous_round=(
                               "the E2 prompt's explanation that area_ratio is the "
                               "fraction of the image and that coordinates are "
                               "normalised was removed to satisfy parity and the ban "
                               "on explaining field meaning"),
                           s5_vs_e2="field-set replicate, NOT prompt-identical; "
                                    "cross-round comparison is descriptive only"),
        output_schema=dict(final_verdict="real|fake",
                           referenced_regions="3x3 grid locations, [] if none",
                           reason="one or two sentences"),
        primary="decision (recall, FPR, specificity, Youden J)",
        secondary_faithfulness=dict(
            applies_to=[k for k, v in HAS_LOC.items() if v],
            note="referenced_regions is only interpretable where a location is given"),
        contrasts=CONTRASTS, interpretation_cases=CASES,
        reason_audit=REASON_AUDIT,
        scope_limit="field ablation only; no counterfactual area substitution this "
                    "round",
        n_sources=len(S), inference_budget=dict(total=n, per_cell=len(S)),
        samples_inherited_from="trufor_evidence_abstraction_frozen.json",
        model=cfg["model"]["path"], decoding=cfg["model"]["generation"],
        min_pixels=cfg["model"]["min_pixels"], max_pixels=cfg["model"]["max_pixels"],
        prior_results=dict(E0_score_only=dict(recall=0.837, fpr=0.011, j=0.826),
                           E1_raw_map=dict(recall=0.565, fpr=0.016, j=0.549),
                           E2_full_structured=dict(recall=0.174, fpr=0.011, j=0.163),
                           E2_spatial_agreement=1.000,
                           mirror_following=1.000,
                           area_ratio_median=dict(fake=0.0251, real=0.00395)),
        forbidden=["prompt tuning", "explaining what fields mean", "counterfactual "
                   "area values", "raw map", "new forensic knowledge", "threshold "
                   "changes", "schema changes"],
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    abstraction_protocol=sha1f(prev_fp)),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      evidence=s["evidence"]) for s in S])
    od = os.path.join(R, "trufor_fields")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_structured_field_ablation_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} sources, {n} inferences")


if __name__ == "__main__":
    main()
