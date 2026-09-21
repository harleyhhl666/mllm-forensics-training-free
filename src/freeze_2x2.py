"""Freeze the Score x Task-Semantics 2x2 causal control. No inference here.

Three cells already exist and are reused verbatim, not recomputed:
  A0 score-only  + original semantics  = frozen S0/P0 cell (byte-identical pair)
  A2 structured  + original semantics  = D0
  A3 structured  + binary rule         = D1

Only the missing cell is run:
  A1 score-only  + binary rule         = score-only template + the D1 rule text

Question: is D1's success generic binary task clarification, or does clarification
specifically repair the policy shift induced by structured-evidence framing? That is
answered by the factorial interaction (A3-A2) - (A1-A0).
"""
import argparse, hashlib, json, os, sys
import numpy as np, yaml

RULE_ANCHOR = "Report strictly as JSON"


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    fr_fp = os.path.join(R, "trufor_framing",
                         "trufor_structured_framing_frozen.json")
    se_fp = os.path.join(R, "trufor_semantics", "trufor_task_semantics_frozen.json")
    fr, se = json.load(open(fr_fp)), json.load(open(se_fp))
    S = se["samples"]

    print("=" * 94)
    print("1. SAMPLES — unchanged")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    assert len({s["coco_id"] for s in S}) == len(S)
    print(f"  184 fake + 184 paired real; single-image input")
    print(f"  model 7B {cfg['model']['path'].rstrip('/').split('/')[-1][:16]}")
    print(f"  inherited from the task-semantics freeze (sha1 {sha1f(se_fp)[:12]})")

    print("\n" + "=" * 94)
    print("2. THE 2x2")
    print(f"  {'cell':<6}{'score':<10}{'structured':<13}{'rule':<7}{'source'}")
    print(f"  {'A0':<6}{'yes':<10}{'no':<13}{'no':<7}reuse frozen S0 / P0 cell")
    print(f"  {'A1':<6}{'yes':<10}{'no':<13}{'YES':<7}NEW — 368 inferences this round")
    print(f"  {'A2':<6}{'yes':<10}{'YES':<13}{'no':<7}reuse D0")
    print(f"  {'A3':<6}{'yes':<10}{'YES':<13}{'YES':<7}reuse D1")
    print(f"\n  the TruFor score line is present in ALL four cells; the two factors are")
    print(f"  (a) structured evidence JSON present/absent, (b) binary rule present/absent")

    print("\n" + "=" * 94)
    print("3. A0 PROVENANCE — two rounds produced this cell; they are identical")
    def cells(path, cond):
        out = {}
        for l in open(path):
            d = json.loads(l)
            if d["condition"] == cond:
                out[(d["sample_id"], d["label"])] = d
        return out
    P0 = cells(os.path.join(R, "trufor_framing", "verdicts.jsonl"), "P0_score_only")
    S0 = cells(os.path.join(R, "trufor_fields", "verdicts.jsonl"), "S0_score_only")
    k = sorted(set(P0) & set(S0))
    same = sum(1 for x in k if P0[x]["prompt_sha1"] == S0[x]["prompt_sha1"])
    agree = np.mean([P0[x]["final_verdict"] == S0[x]["final_verdict"] for x in k])
    print(f"  shared cells {len(k)}  rendered prompt identical {same}/{len(k)}  "
          f"verdict agreement {agree:.3f}")
    assert same == len(k) and agree == 1.0
    print(f"  A0 is therefore unambiguous; this freeze designates the FIELD-round")
    print(f"  S0_score_only cell as the canonical A0 source")

    print("\n" + "=" * 94)
    print("4. PROMPTS")
    so = fr["prompts"]["score_only"]
    d0, d1 = se["prompts"]["D0_baseline"], se["prompts"]["D1_binary_rule"]
    rule = se["inserted_blocks"]["binary_rule"]
    assert d1 == d0.replace(RULE_ANCHOR, rule + RULE_ANCHOR)
    print(f"  verified: D1 == D0 with the rule inserted immediately before "
          f"'{RULE_ANCHOR}'")
    a1 = so.replace(RULE_ANCHOR, rule + RULE_ANCHOR)
    assert a1 != so and rule in a1
    P = {"A0_score_only": so, "A1_score_rule": a1, "A2_structured": d0,
         "A3_structured_rule": d1}
    ph = {k2: hashlib.sha1(v.encode()).hexdigest() for k2, v in P.items()}
    for k2 in ("A0_score_only", "A1_score_rule", "A2_structured",
               "A3_structured_rule"):
        print(f"  {k2:<20} sha1 {ph[k2]}")
    assert ph["A0_score_only"] == fr["prompt_hashes"]["score_only"]
    assert ph["A2_structured"] == se["prompt_hashes"]["D0_baseline"]
    assert ph["A3_structured_rule"] == se["prompt_hashes"]["D1_binary_rule"]
    print(f"\n  A0/A2/A3 hashes match their frozen originals: True")
    print(f"  A1 is built by the SAME transformation that produced A3 from A2:")
    print(f"    A1 = score_only  + rule at the identical anchor position")
    print(f"    A3 = structured  + rule at the identical anchor position")
    print(f"  so the rule text and its placement are constant across both rule cells")
    print(f"  rule text is byte-identical in A1 and A3: "
          f"{rule in a1 and rule in d1}")
    d_a1 = a1.replace(rule, "")
    d_a3 = d1.replace(rule, "")
    print(f"  removing the rule returns A1->A0 and A3->A2 exactly: "
          f"{d_a1 == so and d_a3 == d0}")
    assert d_a1 == so and d_a3 == d0
    print(f"\n  --- A1 prompt ---")
    print("  " + a1.replace("\n", "\n  "))

    print("\n" + "=" * 94)
    print("5. CONTRASTS")
    C = dict(
        rule_effect_without_structured="A1 - A0",
        rule_effect_with_structured="A3 - A2",
        structured_effect_without_rule="A2 - A0",
        structured_effect_with_rule="A3 - A1",
        interaction="(A3 - A2) - (A1 - A0)",
        equivalent_form="(A3 - A1) - (A2 - A0)  — algebraically identical",
        stats="paired effect size -> source-level bootstrap 95% CI; McNemar "
              "supplementary",
        metrics=["fake recall", "real FPR", "specificity", "Youden J"],
    )
    for k2, v in C.items():
        print(f"  {k2:<32} {v}")

    print("\n" + "=" * 94)
    print("6. WHAT THE INTERACTION DECIDES")
    print("  interaction ~ 0  -> the binary rule helps by roughly the same amount")
    print("     regardless of structured evidence: D1's gain is GENERIC task")
    print("     clarification, not a repair of structured-framing policy shift")
    print("  interaction >> 0 -> the rule helps far more when structured evidence is")
    print("     present: clarification SPECIFICALLY repairs the framing-induced shift")
    print("  interaction << 0 -> the rule helps mainly WITHOUT structured evidence")
    print("  note the ceiling: A0 already sits at recall 0.837, so the rule has little")
    print("  head-room in the score-only arm. A near-zero A1-A0 could reflect that")
    print("  ceiling rather than an absent effect; J and FPR are reported alongside")
    print("  recall so a ceiling artefact stays visible.")

    print("\n" + "=" * 94)
    print("7. KNOWN CELL VALUES (three of four already measured)")
    known = dict(A0=dict(recall=0.837, fpr=0.011, j=0.826),
                 A2=dict(recall=0.114, fpr=0.000, j=0.114),
                 A3=dict(recall=0.967, fpr=0.065, j=0.902))
    for k2, v in known.items():
        print(f"  {k2}  recall {v['recall']:.3f}  FPR {v['fpr']:.3f}  J {v['j']:+.3f}")
    print(f"  A1  to be measured ({2*len(S)} inferences)")
    print(f"\n  implied, IF A1 were to land at A0's level (recall 0.837):")
    print(f"    A1-A0 = +0.000, A3-A2 = +0.853, interaction = +0.853")
    print(f"  implied, IF A1 were to reach A3's level (recall 0.967):")
    print(f"    A1-A0 = +0.130, A3-A2 = +0.853, interaction = +0.723")
    print(f"  both illustrative only; not predictions")

    print("\n" + "=" * 94)
    print("8. BUDGET")
    n = 2 * len(S)
    print(f"  A1 only: {len(S)} fake + {len(S)} real = {n} inferences")
    print(f"  single-image; 7B at ~2.4 s/inf -> ~{n*2.4/60:.0f} min")
    print(f"  A0/A2/A3 are NOT re-run; their frozen verdicts are read from disk")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="Score x Task-Semantics 2x2 causal control",
        question="Is D1's success generic binary task clarification, or does "
                 "clarification specifically repair the structured-evidence "
                 "framing-induced policy shift?",
        factors=dict(structured_evidence=["absent", "present"],
                     binary_rule=["absent", "present"]),
        note="the TruFor score line is present in all four cells",
        cells=dict(
            A0_score_only=dict(structured=False, rule=False, source="reuse",
                               origin="trufor_fields/verdicts.jsonl S0_score_only",
                               also_identical_to="trufor_framing P0_score_only"),
            A1_score_rule=dict(structured=False, rule=True, source="NEW",
                               n_inferences=n),
            A2_structured=dict(structured=True, rule=False, source="reuse",
                               origin="trufor_semantics/verdicts.jsonl D0_baseline"),
            A3_structured_rule=dict(structured=True, rule=True, source="reuse",
                                    origin="trufor_semantics/verdicts.jsonl "
                                           "D1_binary_rule")),
        a0_provenance=dict(rendered_prompt_identical=f"{same}/{len(k)}",
                           verdict_agreement=float(agree),
                           canonical="field-round S0_score_only"),
        prompts=P, prompt_hashes=ph,
        prompt_construction=dict(
            rule_text=rule, anchor=RULE_ANCHOR,
            transformation="A1 = A0 with rule inserted immediately before the anchor; "
                           "A3 = A2 with the same insertion. Removing the rule returns "
                           "A1->A0 and A3->A2 byte-exactly.",
            verified=True),
        evidence_package=dict(
            structured_cells=se["evidence_package"]["fields"],
            score_only_cells="frozen TruFor score line only, no JSON",
            extraction="freeze_abstraction.abstract() then strip_fields(), imported "
                       "unchanged",
            raw_map=False),
        output_schema=dict(final_verdict="real|fake",
                           referenced_regions="3x3 grid locations, [] if none",
                           reason="one or two sentences"),
        contrasts=C,
        interaction_reading=dict(
            near_zero="generic task clarification",
            strongly_positive="clarification specifically repairs the "
                              "structured-framing policy shift",
            strongly_negative="the rule helps mainly without structured evidence",
            ceiling_caveat="A0 already sits at recall 0.837, so the score-only arm has "
                           "limited head-room; a small A1-A0 may reflect that ceiling "
                           "rather than an absent rule effect. Report J and FPR "
                           "alongside recall."),
        known_cells=known,
        n_sources=len(S), inference_budget=dict(total=n, new_cell="A1_score_rule"),
        samples_inherited_from="trufor_task_semantics_frozen.json",
        model=cfg["model"]["path"], decoding=cfg["model"]["generation"],
        min_pixels=cfg["model"]["min_pixels"], max_pixels=cfg["model"]["max_pixels"],
        forbidden=["prompt tuning", "re-running A0/A2/A3", "raw map", "32B",
                   "schema change", "new samples"],
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    framing_protocol=sha1f(fr_fp), semantics_protocol=sha1f(se_fp)),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      evidence=s["evidence"]) for s in S])
    od = os.path.join(R, "trufor_2x2")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_score_semantics_2x2_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  new inferences: {n} (cell A1 only)")


if __name__ == "__main__":
    main()
