"""Freeze the 32B capacity replication of the FINAL TruFor integration findings.

No inference here. The question is NOT "is 32B more accurate" but whether three
mechanism conclusions established on 7B survive a 4.5x capacity increase:

  1. binary task semantics matter for decision quality
  2. structured forensic evidence badly harms decisions under wrong framing
  3. under correct semantics, structured evidence mainly buys spatially faithful
     explanation, not extra classification gain

  B0 score only, no rule          <- 7B A0
  B1 score only + rule            <- 7B A1
  B2 score + structured, no rule  <- 7B A2
  B3 score + structured + rule    <- 7B A3
  B4 B3 with MIRRORED structured locations, frozen eligible subset only (secondary)

All prompts are reused byte-identically from the 7B 2x2 freeze, so the only
intended variable is the model.
"""
import argparse, hashlib, json, os, sys
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_abstraction import (abstract, BIN_THR, MIN_AREA_RATIO, TOP_K, RANK_BY,
                                MIRROR, CONNECTIVITY)
from freeze_fields import strip_fields

CELLS = {
    "B0_score_only":      dict(structured=False, rule=False, mirrored=False,
                               maps_to="A0"),
    "B1_score_rule":      dict(structured=False, rule=True, mirrored=False,
                               maps_to="A1"),
    "B2_structured":      dict(structured=True, rule=False, mirrored=False,
                               maps_to="A2"),
    "B3_structured_rule": dict(structured=True, rule=True, mirrored=False,
                               maps_to="A3"),
    "B4_mirrored":        dict(structured=True, rule=True, mirrored=True,
                               maps_to="7B E4 analogue", subset="eligible only"),
}
EFFECTS = [("rule_without_structured", "B1_score_rule", "B0_score_only",
            "A1", "A0"),
           ("rule_with_structured", "B3_structured_rule", "B2_structured",
            "A3", "A2"),
           ("structured_without_rule", "B2_structured", "B0_score_only",
            "A2", "A0"),
           ("structured_with_rule", "B3_structured_rule", "B1_score_rule",
            "A3", "A1")]

SEVEN_B = dict(
    cells=dict(A0=dict(recall=0.837, fpr=0.011, j=0.826),
               A1=dict(recall=0.978, fpr=0.082, j=0.897),
               A2=dict(recall=0.114, fpr=0.000, j=0.114),
               A3=dict(recall=0.967, fpr=0.065, j=0.902)),
    effects_recall=dict(rule_without_structured=+0.141, rule_with_structured=+0.853,
                        structured_without_rule=-0.723, structured_with_rule=-0.011),
    effects_j=dict(rule_without_structured=+0.071, rule_with_structured=+0.788,
                   structured_without_rule=-0.712, structured_with_rule=+0.005),
    interaction=dict(recall=+0.712, fpr=-0.005, j=+0.717),
    faithfulness=dict(spatial_agreement=1.000, mirror_shift_responsiveness=1.000),
    conclusion="scalar score supports decision; structured spatial evidence supports "
               "faithful explanation",
)

HYPOTHESES = {
    "C1_capacity_invariant_task_semantics":
        "B1 > B0 and B3 > B2 with the rule effect still clearly present at 32B, and "
        "delta rule-effect not significantly changed -> binary task semantics remain "
        "important at larger capacity",
    "C2_32B_reduces_framing_collapse":
        "B2 - B0 is clearly less negative than the 7B A2 - A0 = -0.723 and the delta "
        "CI excludes zero -> larger capacity reduces structured-framing-induced "
        "decision collapse",
    "C3_32B_uses_structured_for_decision":
        "B3 - B1 > 0 with CI excluding zero, and clearly above the 7B A3 - A1 ~ 0 -> "
        "larger capacity lets structured spatial evidence add decision utility under "
        "correct semantics",
    "C4_role_separation_capacity_invariant":
        "B3 ~ B1 while structured-condition explanation faithfulness stays high -> even "
        "at 32B structured evidence mainly improves faithful explanation rather than "
        "discrimination",
}

REASON_AUDIT = dict(
    method="deterministic keyword rules, inherited unchanged; no LLM judge",
    patterns=dict(
        score_high=["score of 0.9", "score of 1.0", "high score", "score indicates",
                    "high likelihood", "score suggests", "elevated score"],
        locality=["localized", "localised", "confined", "limited", "isolated",
                  "restricted", "specific region", "specific area", "only a",
                  "narrow", "small region", "small area", "particular region"],
        manipulation_acknowledged=["manipulat", "alter", "tamper", "edited",
                                   "modified", "spliced", "retouch"],
        rule_consistent=["any genuine manipulation", "even though the manipulation is",
                         "regardless of how small", "regardless of size",
                         "even if localized", "even if localised", "binary",
                         "any part of the image has been", "however small",
                         "no matter how small"],
    ),
    focus="B0 vs B1 and B2 vs B3: does the binary rule raise the weight of the scalar "
          "score in the stated reasoning, as it did on 7B (score_high 0.668 -> 0.946)?",
)


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_7b")
    ap.add_argument("config_32b")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    c7 = yaml.safe_load(open(args.config_7b))
    c32 = yaml.safe_load(open(args.config_32b))
    R = c7["experiment"]["out_root"]
    tw_fp = os.path.join(R, "trufor_2x2", "trufor_score_semantics_2x2_frozen.json")
    ab_fp = os.path.join(R, "trufor_abstraction",
                         "trufor_evidence_abstraction_frozen.json")
    tw, ab = json.load(open(tw_fp)), json.load(open(ab_fp))
    S = tw["samples"]

    print("=" * 94)
    print("1. MODEL AND PARITY")
    p32 = c32["model"]["path"]
    snap = p32.rstrip("/").split("/")[-1]
    print(f"  path     {p32}")
    print(f"  snapshot {snap}")
    assert snap == "7cfb30d71a1f4f49a57592323337a4a4727301da"
    print(f"  snapshot matches the required pin: True")
    env = os.path.join(R, "qwen32b", "qwen32b_env_frozen.json")
    print(f"  env qwen_vl_32b; frozen record sha1 {sha1f(env)[:12]}")
    print(f"  dtype {c32['model']['dtype']}  device_map {c32['model']['device_map']}  "
          f"quantization none  CPU offload none")
    assert c32["model"]["dtype"] == "bfloat16"
    assert c32["model"]["device_map"] == "auto"
    checks = dict(min_pixels=200704, max_pixels=802816)
    for k, v in checks.items():
        print(f"  {k:<14} 7B {c7['model'][k]}  32B {c32['model'][k]}  "
              f"required {v}  OK {c7['model'][k]==c32['model'][k]==v}")
        assert c7["model"][k] == c32["model"][k] == v
    g7, g32 = c7["model"]["generation"], c32["model"]["generation"]
    req = dict(do_sample=False, temperature=0.0, max_new_tokens=320,
               repetition_penalty=1.0)
    for k, v in req.items():
        print(f"  {k:<18} 7B {g7[k]}  32B {g32[k]}  required {v}  "
              f"OK {g7[k]==g32[k]==v}")
        assert g7[k] == g32[k] == v
    print(f"  MultiGPU inference path: mllm_probe_mgpu.QwenVLProbeMultiGPU (verified "
          f"in the earlier 32B round)")
    print(f"  visual token budget unchanged: True")

    print("\n" + "=" * 94)
    print("2. SAMPLES")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    assert len({s["coco_id"] for s in S}) == len(S)
    print(f"  184 fake + 184 paired real = {2*len(S)} images")
    print(f"  identical source set to 7B A0-A3, inherited from the 2x2 freeze "
          f"(sha1 {sha1f(tw_fp)[:12]})")
    print(f"  NOT reselected and NOT filtered on any 7B outcome")

    print("\n" + "=" * 94)
    print("3. EVIDENCE — frozen, not recomputed")
    fields = ab["abstraction_layer"]
    print(f"  threshold {BIN_THR}  connectivity {CONNECTIVITY}-connected  "
          f"min area {MIN_AREA_RATIO}  top-K {TOP_K}")
    print(f"  sort by {RANK_BY}")
    assert BIN_THR == 0.5 and MIN_AREA_RATIO == 0.001 and TOP_K == 3
    FLD = ["grid_location", "area_ratio", "mean_probability", "max_probability",
           "centroid"]
    print(f"  structured fields {FLD}")
    ev = S[0]["evidence"]["fake_own"]
    full = abstract(np.load(ev["npz"])["map"].astype(np.float32), ev["score"])
    pl = strip_fields(full, FLD)
    print(f"  example payload {json.dumps(pl)[:140]}...")
    # payload hash over the whole corpus, so drift is detectable
    h = hashlib.sha1()
    for s in S:
        for lb in ("fake", "real"):
            e = s["evidence"][f"{lb}_own"]
            f2 = abstract(np.load(e["npz"])["map"].astype(np.float32), e["score"])
            h.update(json.dumps(strip_fields(f2, FLD), sort_keys=True).encode())
    PAYLOAD_HASH = h.hexdigest()
    print(f"  corpus-wide structured payload sha1 {PAYLOAD_HASH}")
    print(f"  the runner must reproduce this hash or refuse to start")

    print("\n" + "=" * 94)
    print("4. PROMPTS — reused BYTE-IDENTICALLY from the 7B 2x2 freeze")
    P = {"B0_score_only": tw["prompts"]["A0_score_only"],
         "B1_score_rule": tw["prompts"]["A1_score_rule"],
         "B2_structured": tw["prompts"]["A2_structured"],
         "B3_structured_rule": tw["prompts"]["A3_structured_rule"]}
    P["B4_mirrored"] = P["B3_structured_rule"]
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in P.items()}
    for k, v in CELLS.items():
        print(f"  {k:<20} <- 7B {v['maps_to']:<16} sha1 {ph[k]}")
    for b, a in (("B0_score_only", "A0_score_only"),
                 ("B1_score_rule", "A1_score_rule"),
                 ("B2_structured", "A2_structured"),
                 ("B3_structured_rule", "A3_structured_rule")):
        assert ph[b] == tw["prompt_hashes"][a], b
    print(f"  all four hashes match their 7B originals: True")
    print(f"  B4 shares B3's prompt byte-for-byte: "
          f"{ph['B4_mirrored'] == ph['B3_structured_rule']}")
    rule = tw["prompt_construction"]["rule_text"]
    assert rule in P["B1_score_rule"] and rule in P["B3_structured_rule"]
    assert rule not in P["B0_score_only"] and rule not in P["B2_structured"]
    print(f"  binary rule present in B1/B3 only: True")
    print(f"  rule text (verbatim, unchanged):")
    print("    " + rule.strip())

    print("\n" + "=" * 94)
    print("5. OUTPUT SCHEMA — identical for all cells")
    print(f"  final_verdict, referenced_regions, reason")
    print(f"  NO manipulation_present, NO manipulation_extent: 7B showed explicit")
    print(f"  presence decomposition causes over-detection (presence FPR 0.951), so it")
    print(f"  is deliberately excluded from this final mechanism replication")
    for k, v in P.items():
        assert "manipulation_present" not in v and "manipulation_extent" not in v

    print("\n" + "=" * 94)
    print("6. B4 MIRROR CONTROL (secondary)")
    elig = ab["e4_eligible_ids"]
    print(f"  frozen eligible subset: fake {len(elig['fake'])}  real {len(elig['real'])}")
    print(f"  total B4 inferences {len(elig['fake'])+len(elig['real'])}")
    print(f"  transform: grid left<->right, centre column unchanged, centroid x' = 1-x")
    print(f"  unchanged: area_ratio, mean_probability, max_probability, score")
    print(f"  the ORIGINAL IMAGE is never mirrored; prompt identical to B3")
    print(f"  eligibility was frozen in the abstraction round BEFORE any 7B mirror")
    print(f"  result and is reused here unchanged — not reselected for 32B")
    m = abstract(np.load(ev["npz"])["map"].astype(np.float32), ev["score"], mirror=True)
    a, b = strip_fields(full, FLD), strip_fields(m, FLD)
    print(f"  example: {[r['grid_location'] for r in a['regions']]} -> "
          f"{[r['grid_location'] for r in b['regions']]}")
    same = all(x["area_ratio"] == y["area_ratio"]
               and x["mean_probability"] == y["mean_probability"]
               and x["max_probability"] == y["max_probability"]
               for x, y in zip(a["regions"], b["regions"]))
    print(f"  evidence STRENGTH unchanged under mirroring: {same}")
    assert same

    print("\n" + "=" * 94)
    print("7. EFFECTS AND THE 7B REFERENCE")
    print(f"  {'effect':<26}{'32B':<26}{'7B':<14}{'7B recall':>11}{'7B J':>9}")
    for nm, hi, lo, a7, b7 in EFFECTS:
        print(f"  {nm:<26}{hi.split('_')[0]+' - '+lo.split('_')[0]:<26}"
              f"{a7+' - '+b7:<14}{SEVEN_B['effects_recall'][nm]:>+11.3f}"
              f"{SEVEN_B['effects_j'][nm]:>+9.3f}")
    print(f"\n  interaction  (B3-B2) - (B1-B0)   7B: recall "
          f"{SEVEN_B['interaction']['recall']:+.3f}  J "
          f"{SEVEN_B['interaction']['j']:+.3f}")
    print(f"  PRIMARY capacity analysis: delta_effect = effect_32B - effect_7B with")
    print(f"  bootstrap 95% CI, computed PER SOURCE since both models ran the same")
    print(f"  184 sources. Significance-vs-significance comparison is NOT used.")
    print(f"\n  7B cells for reference:")
    for k, v in SEVEN_B["cells"].items():
        print(f"    {k}  recall {v['recall']:.3f}  FPR {v['fpr']:.3f}  J {v['j']:+.3f}")

    print("\n" + "=" * 94)
    print("8. CAPACITY HYPOTHESES")
    for k, v in HYPOTHESES.items():
        print(f"  {k}")
        print(f"    {v}")

    print("\n" + "=" * 94)
    print("9. FAITHFULNESS AND REASON AUDIT")
    print(f"  B2/B3: citation rate, spatial agreement, exact agreement,")
    print(f"  unsupported-region rate; compared against 7B A2/A3")
    print(f"  7B reference: spatial agreement {SEVEN_B['faithfulness']['spatial_agreement']:.3f}, "
          f"mirror shift responsiveness "
          f"{SEVEN_B['faithfulness']['mirror_shift_responsiveness']:.3f}")
    print(f"  reason audit focus: {REASON_AUDIT['focus']}")

    print("\n" + "=" * 94)
    print("10. BUDGET AND RUNTIME")
    main_n = 4 * 2 * len(S)
    b4_n = len(elig["fake"]) + len(elig["real"])
    tot = main_n + b4_n
    prev = json.load(open(os.path.join(R, "trufor_32b", "run_meta.json")))
    rate = prev.get("mean_seconds", 8.6)
    print(f"  main 4 x 2 x {len(S)} = {main_n}")
    print(f"  B4 eligible subset      = {b4_n}")
    print(f"  TOTAL {tot}")
    print(f"  measured 32B rate {rate:.2f} s/inf (single-image, previous 32B round)")
    print(f"  estimate {tot*rate/3600:.1f} h; allow up to {tot*11.4/3600:.1f} h at the")
    print(f"  slower historical rate")
    print(f"  GPUs 0-3, BF16, tmux; historical peaks 16.3/18.2/18.2/16.8 GB")

    print("\n" + "=" * 94)
    print("11. FORBIDDEN")
    for x in ("raw map", "confidence map", "presence decomposition",
              "extent decomposition", "prompts outside B0-B4", "prompt tuning",
              "threshold tuning", "LoRA", "training", "new detector", "new dataset"):
        print(f"  - {x}")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="32B capacity replication of the final TruFor integration findings",
        question="Does increasing capacity from 7B to 32B change three mechanism "
                 "conclusions: (1) binary task semantics matter, (2) structured "
                 "evidence harms decisions under wrong framing, (3) under correct "
                 "semantics structured evidence buys faithful explanation rather than "
                 "classification gain?",
        not_the_question="whether 32B is more accurate than 7B",
        model=dict(path=p32, snapshot=snap, env="qwen_vl_32b", dtype="bfloat16",
                   device_map="auto", quantization=None, cpu_offload=False,
                   gpus="4x RTX 3090 (0-3)",
                   inference_path="mllm_probe_mgpu.QwenVLProbeMultiGPU",
                   env_record=env),
        parity=dict(min_pixels=200704, max_pixels=802816, do_sample=False,
                    temperature=0.0, max_new_tokens=320, repetition_penalty=1.0,
                    identical_to_7B=True, asserted_at_freeze=True),
        cells=CELLS, prompts=P, prompt_hashes=ph,
        prompt_provenance="byte-identical to the 7B 2x2 freeze; B0<-A0, B1<-A1, "
                          "B2<-A2, B3<-A3; B4 reuses B3's prompt",
        binary_rule=rule,
        output_schema=dict(final_verdict="real|fake",
                           referenced_regions="3x3 grid locations, [] if none",
                           reason="one or two sentences"),
        excluded_fields=dict(
            manipulation_present="excluded", manipulation_extent="excluded",
            why="7B showed explicit presence decomposition causes over-detection "
                "(presence FPR 0.951, verdict J 0.049)"),
        evidence=dict(structured_fields=FLD,
                      extraction=dict(binary_threshold=BIN_THR,
                                      connectivity=CONNECTIVITY,
                                      min_area_ratio=MIN_AREA_RATIO, top_k=TOP_K,
                                      rank_by=RANK_BY,
                                      source="freeze_abstraction.abstract() + "
                                             "freeze_fields.strip_fields(), imported "
                                             "unchanged"),
                      corpus_payload_sha1=PAYLOAD_HASH,
                      runner_must_verify=True, raw_map=False),
        b4=dict(eligible_ids=elig,
                n=b4_n,
                rule="grid left<->right, centre column unchanged, centroid x'=1-x; "
                     "area_ratio / mean_probability / max_probability / score "
                     "unchanged; original image never mirrored",
                eligibility_provenance="frozen in the abstraction round before any 7B "
                                       "mirror result; reused unchanged, not "
                                       "reselected for 32B",
                purpose="shift responsiveness — does 32B actually consume structured "
                        "spatial evidence?",
                status="secondary"),
        effects=[dict(name=nm, cell_32b=f"{hi} - {lo}", cell_7b=f"{a7} - {b7}")
                 for nm, hi, lo, a7, b7 in EFFECTS],
        interaction="(B3 - B2) - (B1 - B0), on recall / FPR / J",
        primary_capacity_analysis="delta_effect = effect_32B - effect_7B, paired per "
                                  "source, bootstrap 95% CI. Comparing significance "
                                  "between models is explicitly NOT used.",
        seven_b_reference=SEVEN_B,
        capacity_hypotheses=HYPOTHESES,
        faithfulness=dict(metrics=["citation rate", "spatial agreement",
                                   "exact agreement", "unsupported-region rate"],
                          applies_to=["B2_structured", "B3_structured_rule",
                                      "B4_mirrored"],
                          compare_against="7B A2/A3"),
        reason_audit=REASON_AUDIT,
        n_sources=len(S),
        inference_budget=dict(main=main_n, b4=b4_n, total=tot),
        runtime_estimate_hours=round(tot * rate / 3600, 1),
        samples_inherited_from="trufor_score_semantics_2x2_frozen.json",
        forbidden=["raw map", "confidence map", "presence decomposition",
                   "extent decomposition", "prompts outside B0-B4", "prompt tuning",
                   "threshold tuning", "LoRA", "training", "new detector",
                   "new dataset"],
        promotion_criterion="if all three conclusions hold at 32B, the finding moves "
                            "from '7B-specific phenomenon' to 'stable across "
                            "Qwen2.5-VL 7B and 32B within the same model family'",
        hashes=dict(config_7b=sha1f(args.config_7b), config_32b=sha1f(args.config_32b),
                    freezer=sha1f(os.path.abspath(__file__)),
                    twoxtwo_protocol=sha1f(tw_fp), abstraction_protocol=sha1f(ab_fp)),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      evidence=s["evidence"]) for s in S])
    od = os.path.join(R, "trufor_32b_final")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_32b_final_capacity_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} sources; {main_n} main + {b4_n} B4 = {tot} inferences")


if __name__ == "__main__":
    main()
