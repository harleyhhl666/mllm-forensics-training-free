"""Freeze the structured-evidence framing / region-presence control. No inference.

Purpose: the previous round confounded three things, because S0 used a score-only
prompt while S1-S5 used a structured-evidence prompt. This round separates

  (1) structured prompt framing,
  (2) presence of a region-level evidence block,
  (3) actual region field values.

  P0 original score-only baseline (replicates S0, old prompt)
  P1 structured prompt, NO structured block
  P2 structured prompt + {"regions": []}
  P3 structured prompt + region placeholder (region_id only)
  P4 structured prompt + true grid_location   (replicates S1)
  P5 structured prompt + neutral scalar metadata (region_id + order)

P1-P5 share one byte-identical instruction template; only the JSON payload differs.
"""
import argparse, hashlib, json, os, sys
from collections import Counter
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_abstraction import abstract, BIN_THR, MIN_AREA_RATIO, TOP_K, RANK_BY
from freeze_fields import PROMPTS as FIELD_PROMPTS, strip_fields

# payload builders, all deterministic; region COUNT is inherited from the frozen
# abstraction layer so the container size is held fixed across P3/P4/P5
PAYLOAD = {
    "P0_score_only":    dict(kind="none",        prompt="score_only"),
    "P1_framing_only":  dict(kind="none",        prompt="structured"),
    "P2_empty_regions": dict(kind="empty",       prompt="structured"),
    "P3_placeholder":   dict(kind="placeholder", prompt="structured"),
    "P4_true_location": dict(kind="location",    prompt="structured"),
    "P5_neutral_meta":  dict(kind="neutral",     prompt="structured"),
}

CONTRASTS = dict(
    prompt_framing_effect="P1 - P0",
    empty_container_effect="P2 - P1",
    region_object_effect="P3 - P2",
    actual_spatial_information_effect="P4 - P3",
    generic_metadata_effect="P5 - P3",
    supplementary=["P4 - P0 (should reproduce last round's S1 - S0 = -0.696)",
                   "P5 - P4 (neutral metadata vs real location)"],
    stats="paired effect size -> source-level bootstrap 95% CI -> McNemar supplementary",
)

CASES = {
    "F1_prompt_framing_drives_collapse":
        "P1 << P0 while P2/P3/P4 differ little from P1 -> the collapse is induced "
        "primarily by structured-forensic FRAMING rather than by the actual region "
        "measurements",
    "F2_region_presence_drives_collapse":
        "P1 ~ P0 but P2 or P3 << P1 -> explicit region-level evidence representation "
        "changes decision policy even without meaningful forensic values",
    "F3_actual_forensic_fields_drive_collapse":
        "P1 ~ P2 ~ P3 ~ P0 but P4 << P3 -> concrete region information, not framing, "
        "drives the failure",
    "F4_generic_structured_token_overload":
        "P3 and P5 both fall by a similar amount -> generic structured information / "
        "extra metadata interferes with scalar evidence arbitration",
}

REASON_AUDIT = dict(
    method="deterministic keyword rules; no LLM judge; inherited from the field round",
    patterns=dict(
        score_high=["score of 0.9", "score of 1.0", "high score", "score indicates",
                    "high likelihood", "score suggests", "elevated score"],
        manipulation_acknowledged=["manipulat", "alter", "tamper", "edited", "modified",
                                   "spliced", "retouch"],
        locality=["localized", "localised", "confined", "limited", "isolated",
                  "restricted", "specific region", "specific area", "only a",
                  "narrow", "small region", "small area", "particular region"],
        evidence_weak=["weak", "not strong", "insufficient", "inconclusive",
                       "not conclusive", "lacks", "minimal evidence",
                       "does not provide strong"],
    ),
    key_metrics=dict(
        acknowledged_manipulation_but_real=(
            "verdict == real AND manipulation_acknowledged AND (locality OR "
            "evidence_weak)"),
        locality_without_locality_data=(
            "locality language in P1/P2/P3/P5, which carry NO location, area or "
            "probability value. If present, the locality narrative is activated by "
            "framing/schema rather than derived from evidence values."),
    ),
)


def build_payload(kind, full):
    if kind == "none":
        return None
    if kind == "empty":
        return dict(regions=[])
    n = len(full["regions"])
    if kind == "placeholder":
        return dict(regions=[dict(region_id=f"region_{i+1}") for i in range(n)])
    if kind == "location":
        return strip_fields(full, ["grid_location"])
    if kind == "neutral":
        return dict(regions=[dict(region_id=i + 1, order=i + 1) for i in range(n)])
    raise ValueError(kind)


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    src = os.path.join(R, "trufor_fields",
                       "trufor_structured_field_ablation_frozen.json")
    prev = json.load(open(src))
    S = prev["samples"]

    print("=" * 94)
    print("1. SAMPLES — unchanged")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    assert len({s["coco_id"] for s in S}) == len(S)
    print(f"  fake + paired real; single-image input in every condition")
    print(f"  inherited from the field-ablation freeze (sha1 {sha1f(src)[:12]})")
    print(f"  model 7B {cfg['model']['path'].rstrip('/').split('/')[-1][:16]}")

    print("\n" + "=" * 94)
    print("2. PROMPTS")
    P_SCORE = FIELD_PROMPTS["S0_score_only"]
    P_STRUCT = FIELD_PROMPTS["S5_full"]
    assert P_SCORE == prev["prompts"]["S0_score_only"]
    assert P_STRUCT == prev["prompts"]["S5_full"]
    hs = hashlib.sha1(P_SCORE.encode()).hexdigest()
    ht = hashlib.sha1(P_STRUCT.encode()).hexdigest()
    print(f"  P0 score-only template   sha1 {hs}  (identical to last round's S0)")
    print(f"  P1-P5 structured template sha1 {ht}  (identical to last round's S1-S5)")
    assert hs == prev["prompt_hashes"]["S0_score_only"]
    assert ht == prev["prompt_hashes"]["S5_full"]
    print(f"  both templates are reused BYTE-IDENTICALLY, so P0 reproduces S0 and")
    print(f"  P4 reproduces S1 exactly")
    print(f"\n  IMPORTANT rendering detail for P1:")
    print(f"  the structured template contains a payload placeholder. For P1 the")
    print(f"  placeholder is replaced by the EMPTY STRING, so every instruction word")
    print(f"  is byte-identical to P2-P5 and only the payload area is blank. This is")
    print(f"  the closest possible realisation of 'structured prompt, no structured")
    print(f"  block'; the rendered prompts therefore differ only in the payload.")
    ren = {}
    ev0 = S[0]["evidence"]["fake_own"]
    m0 = np.load(ev0["npz"])["map"].astype(np.float32)
    full0 = abstract(m0, ev0["score"])
    for k, spec in PAYLOAD.items():
        tpl = P_SCORE if spec["prompt"] == "score_only" else P_STRUCT
        pl = build_payload(spec["kind"], full0)
        t = tpl.replace("{score}", f"{ev0['score']:.3f}")
        if "{structured}" in t:
            t = t.replace("{structured}", "" if pl is None else json.dumps(pl))
        ren[k] = t
    print(f"\n  rendered-prompt hashes (first fake source):")
    for k in PAYLOAD:
        print(f"    {k:<18} {hashlib.sha1(ren[k].encode()).hexdigest()[:16]}")
    instr = {k: " ".join(v.split()) for k, v in ren.items()}
    base = instr["P2_empty_regions"].replace('{"regions": []}', "")
    same = all(" ".join(
        instr[k].replace(json.dumps(build_payload(PAYLOAD[k]["kind"], full0)), "")
        .split()) == " ".join(base.split())
        for k in ("P3_placeholder", "P4_true_location", "P5_neutral_meta"))
    print(f"  P2-P5 instruction text identical once the payload is removed: {same}")
    assert same

    print("\n" + "=" * 94)
    print("3. PAYLOADS (first fake source, 2 real regions)")
    print(f"  sample {S[0]['sample_id']}  score {ev0['score']:.3f}  "
          f"regions {len(full0['regions'])}")
    for k, spec in PAYLOAD.items():
        pl = build_payload(spec["kind"], full0)
        print(f"  {k:<18} {'(no payload)' if pl is None else json.dumps(pl)}")
    print(f"\n  region COUNT in P3/P4/P5 is inherited from the frozen abstraction")
    print(f"  layer (thr {BIN_THR}, min area {MIN_AREA_RATIO}, top-K {TOP_K}, ranked")
    print(f"  by {RANK_BY}), so container size is held fixed across those three.")
    print(f"  P3 region_id is a STRING entry label; P5 uses two NUMERIC fields to")
    print(f"  control numeric-token presence and JSON length. Neither encodes")
    print(f"  location, size, probability or confidence, and the prompt never says")
    print(f"  they have forensic meaning. No random numbers are used.")

    print("\n" + "=" * 94)
    print("4. PAYLOAD AUDIT over all sources")
    lens, bad = {k: [] for k in PAYLOAD}, []
    nreg = []
    for s in S:
        for lb in ("fake", "real"):
            e = s["evidence"][f"{lb}_own"]
            fl = abstract(np.load(e["npz"])["map"].astype(np.float32), e["score"])
            nreg.append(len(fl["regions"]))
            for k, spec in PAYLOAD.items():
                pl = build_payload(spec["kind"], fl)
                lens[k].append(0 if pl is None else len(json.dumps(pl)))
                if pl is None or spec["kind"] == "empty":
                    continue
                if len(pl["regions"]) != len(fl["regions"]):
                    bad.append((s["sample_id"], lb, k, "count mismatch"))
                for r in pl["regions"]:
                    keys = set(r)
                    exp = ({"region_id"} if spec["kind"] == "placeholder"
                           else {"grid_location"} if spec["kind"] == "location"
                           else {"region_id", "order"})
                    if keys != exp:
                        bad.append((s["sample_id"], lb, k, sorted(keys)))
    print(f"  problems: {len(bad)}")
    assert not bad
    print(f"  region count distribution {dict(sorted(Counter(nreg).items()))}")
    print(f"  {'condition':<18}{'mean JSON chars':>17}")
    for k in PAYLOAD:
        print(f"  {k:<18}{np.mean(lens[k]):>17.1f}")
    print(f"  P5 is longer than P3 by construction (two fields vs one), which is the")
    print(f"  intended control on JSON length and numeric tokens; P4 length sits")
    print(f"  between them. Exact length matching is NOT claimed.")

    print("\n" + "=" * 94)
    print("5. CONTRASTS AND CASES")
    for k, v in CONTRASTS.items():
        print(f"  {k:<36} {v}")
    print()
    for k in CASES:
        print(f"  case {k}")

    print("\n" + "=" * 94)
    print("6. REASON AUDIT")
    for k, v in REASON_AUDIT["patterns"].items():
        print(f"  {k:<26} {len(v)} patterns")
    for k, v in REASON_AUDIT["key_metrics"].items():
        print(f"  KEY {k}:")
        print(f"      {v}")

    print("\n" + "=" * 94)
    print("7. SCHEMA ARTIFACT POLICY")
    print("  the output schema is UNCHANGED from the previous rounds to preserve")
    print("  comparability, so referenced_regions still enumerates the nine cells.")
    print("  the all-9 echo artifact is recorded separately and is NOT used for any")
    print("  mechanism judgement this round; primary is decision, not faithfulness.")

    print("\n" + "=" * 94)
    print("8. SCOPE LIMIT")
    print("  no numeric counterfactual this round: area_ratio, mean_probability and")
    print("  max_probability are never altered. A numeric counterfactual is only")
    print("  warranted if P1/P2/P3 turn out near-harmless and P4 alone collapses.")

    print("\n" + "=" * 94)
    print("9. BUDGET AND REFERENCE")
    n = len(PAYLOAD) * 2 * len(S)
    print(f"  {len(PAYLOAD)} conditions x 2 labels x {len(S)} = {n} inferences")
    print(f"  all single-image; 7B at ~2.7 s -> ~{n*2.7/60:.0f} min")
    print(f"  last round: S0 0.837  S1 0.141  S2 0.196  S3 0.103  S4 0.212  S5 0.114")
    print(f"  P0 must reproduce 0.837 and P4 must reproduce 0.141 within sampling")
    print(f"  noise; both are exact prompt+payload replicates, so a large deviation")
    print(f"  would indicate a pipeline problem rather than a finding.")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="Structured evidence framing / region-presence control",
        question="Does structured-evidence FRAMING or the mere presence of a "
                 "region-level block cause the decision collapse, rather than the "
                 "region field values?",
        confound_addressed="previously S0 used a score-only prompt while S1-S5 used a "
                           "structured prompt, so framing, region-block presence and "
                           "field values were confounded",
        out_of_scope=["area magnitude", "probability magnitude", "spatial correctness",
                      "raw map", "32B"],
        conditions={k: dict(payload_kind=v["kind"], prompt=v["prompt"],
                            single_image=True) for k, v in PAYLOAD.items()},
        prompts=dict(score_only=P_SCORE, structured=P_STRUCT),
        prompt_hashes=dict(score_only=hs, structured=ht),
        prompt_provenance="both templates reused byte-identically from the field "
                          "ablation freeze; P0 replicates S0 and P4 replicates S1",
        p1_rendering_note="for P1 the payload placeholder is replaced by the empty "
                          "string, so every instruction word is byte-identical to "
                          "P2-P5 and only the payload area is blank",
        payload_rules=dict(
            region_count_source="inherited from freeze_abstraction.abstract(); P3/P4/P5 "
                                "all carry the same number of region objects",
            p3="region_id string entry label only; no location, size, probability or "
               "confidence; the prompt never ascribes forensic meaning to it",
            p5="region_id and order, two numeric fields, to control numeric-token "
               "presence and JSON length; no random numbers",
            p2="empty region list",
            length_note="exact JSON length matching is not claimed; P5 > P4 > P3 by "
                        "construction"),
        abstraction_inherited=dict(binary_threshold=BIN_THR,
                                   min_area_ratio=MIN_AREA_RATIO, top_k=TOP_K,
                                   rank_by=RANK_BY),
        output_schema_policy="unchanged to preserve comparability; the all-9 echo "
                             "artifact is recorded but never used for mechanism "
                             "judgement this round",
        contrasts=CONTRASTS, interpretation_cases=CASES, reason_audit=REASON_AUDIT,
        primary="decision (recall, FPR, specificity, Youden J)",
        scope_limit="no numeric counterfactual this round",
        n_sources=len(S), inference_budget=dict(total=n, per_cell=len(S)),
        samples_inherited_from="trufor_structured_field_ablation_frozen.json",
        model=cfg["model"]["path"], decoding=cfg["model"]["generation"],
        min_pixels=cfg["model"]["min_pixels"], max_pixels=cfg["model"]["max_pixels"],
        prior_results=dict(S0=0.837, S1=0.141, S2=0.196, S3=0.103, S4=0.212, S5=0.114,
                           all_structured_fpr=0.0,
                           spatial_agreement=1.000, shift_responsiveness=1.000,
                           area_hypothesis="refuted; S4-S1 was +0.071"),
        replication_checks=dict(P0_expect=0.837, P4_expect=0.141),
        forbidden=["prompt tuning", "numeric counterfactual", "schema change",
                   "raw map", "32B", "random placeholder values"],
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    field_protocol=sha1f(src)),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      evidence=s["evidence"]) for s in S])
    od = os.path.join(R, "trufor_framing")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_structured_framing_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} sources, {n} inferences")


if __name__ == "__main__":
    main()
