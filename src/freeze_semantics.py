"""Freeze the task-semantics / verdict-decomposition experiment. No inference here.

The evidence payload is IDENTICAL across all four conditions (full structured
evidence). Only the decision instruction changes:

  D0 original structured framing baseline  (byte-identical to the frozen S5 prompt)
  D1 + explicit binary authenticity rule
  D2 + presence/extent decomposition fields
  D3 + presence-first reasoning order

Mechanism question: why does structured forensic framing make the model concede
manipulation yet still output Real? This round separates
(1) binary-rule misunderstanding, (2) presence-judgement failure,
(3) extent/existence confusion, (4) final arbitration failure.
"""
import argparse, hashlib, json, os, sys
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_abstraction import abstract, BIN_THR, MIN_AREA_RATIO, TOP_K, RANK_BY

GRID = ("grid location(s) your verdict relies on, from: top-left, top-center, "
        "top-right, center-left, center, center-right, bottom-left, bottom-center, "
        "bottom-right; use [] if none")

# ---- shared blocks, byte-identical across conditions -------------------------
HEAD = "You are performing a digital image forensics analysis.\n\n"
EVID = ("An automated forensic tool provides an image-level manipulation score and "
        "structured localization measurements derived from its localization output. "
        "Treat these measurements as auxiliary evidence and interpret them together "
        "with the original image. The measurements are given in the JSON block below. "
        "An empty region list means the tool localised no region above its internal "
        "response level.\n\n")
SCORE = "TruFor image-level manipulation score: {score}\n\n"
PAYLOAD = "{structured}\n\n"
TASK = ("Determine whether this image is an authentic photograph or has been digitally\n"
        "altered.\n\n")

# ---- the only things that differ --------------------------------------------
RULE = ("For this task, the authenticity label is binary. If there is credible "
        "evidence that any part of the image has been genuinely manipulated, the "
        "image should be classified as fake, regardless of how small or localized "
        "the manipulated region is.\n\n")
DECOMP = ("Report manipulation existence and manipulation extent as separate "
          "judgements. The field manipulation_present states whether any genuine "
          "manipulation exists. The field manipulation_extent only describes how far "
          "it spreads. Extent does not substitute for existence: a small extent is "
          "not a reason to set manipulation_present to false. If "
          "manipulation_present is true, then under the binary authenticity rule "
          "final_verdict is fake.\n\n")
ORDERED = ("Decide in this order. Step 1: judge whether any genuine manipulation "
           "exists and set manipulation_present. Step 2: only if it exists, judge "
           "how far it spreads and set manipulation_extent. Step 3: apply the binary "
           "authenticity rule to set final_verdict. Do not let the extent judgement "
           "from Step 2 revise the existence judgement from Step 1.\n\n")

SCHEMA_BASE = ('Report strictly as JSON with these keys:\n{\n'
               '  "final_verdict": "real" or "fake",\n'
               f'  "referenced_regions": ["{GRID}"],\n'
               '  "reason": "one or two sentences justifying the verdict"\n'
               '}\nOutput JSON only.')
SCHEMA_DEC = ('Report strictly as JSON with these keys:\n{\n'
              '  "manipulation_present": true or false,\n'
              '  "manipulation_extent": "none" or "localized" or "widespread",\n'
              '  "final_verdict": "real" or "fake",\n'
              f'  "referenced_regions": ["{GRID}"],\n'
              '  "reason": "one or two sentences justifying the verdict"\n'
              '}\nOutput JSON only.')

PROMPTS = {
    "D0_baseline":      HEAD + EVID + SCORE + PAYLOAD + TASK + SCHEMA_BASE,
    "D1_binary_rule":   HEAD + EVID + SCORE + PAYLOAD + TASK + RULE + SCHEMA_BASE,
    "D2_decomposition": HEAD + EVID + SCORE + PAYLOAD + TASK + RULE + DECOMP
                        + SCHEMA_DEC,
    "D3_presence_first": HEAD + EVID + SCORE + PAYLOAD + TASK + RULE + DECOMP
                         + ORDERED + SCHEMA_DEC,
}
CONDITIONS = {
    "D0_baseline": dict(rule=False, decomposition=False, ordered=False,
                        schema="base"),
    "D1_binary_rule": dict(rule=True, decomposition=False, ordered=False,
                           schema="base"),
    "D2_decomposition": dict(rule=True, decomposition=True, ordered=False,
                             schema="decomposed"),
    "D3_presence_first": dict(rule=True, decomposition=True, ordered=True,
                              schema="decomposed"),
}

CONTRASTS = dict(
    rule_clarification_effect="D1 - D0",
    decomposition_effect="D2 - D0",
    presence_first_effect="D3 - D0",
    ordered_arbitration_benefit="D3 - D2",
    stats="paired effect size -> source-level bootstrap 95% CI -> McNemar "
          "supplementary",
)

PRESENCE_METRICS = dict(
    applies_to=["D2_decomposition", "D3_presence_first"],
    presence_tpr="P(manipulation_present = true | fake)",
    presence_fpr="P(manipulation_present = true | real)",
    presence_j="presence_tpr - presence_fpr",
    logic="if presence is accurate but final_verdict is poor, the failure sits in "
          "final arbitration; if presence is also poor, the problem remains in "
          "evidence interpretation / presence judgement",
)

INCONSISTENCY = dict(
    A_primary="manipulation_present = true AND final_verdict = real  "
              "(the structured-field version of last round's "
              "acknowledged_manipulation_but_real, no regex needed)",
    B="manipulation_present = false AND final_verdict = fake",
    reported_for=["fake", "real"],
)

EXTENT = dict(
    distribution="share of none / localized / widespread, per label",
    key_quantity="P(final_verdict = real | manipulation_present = true, "
                 "manipulation_extent = localized)",
    reading="if high in D0/D2 but clearly lower in D1/D3, the model was treating "
            "'localized' as an authenticity exemption",
)

CASES = {
    "T1_task_semantics_misunderstanding":
        "D1 >> D0 with recall largely restored, FPR not clearly up, and "
        "present-true-but-real sharply down -> the collapse is largely a "
        "misunderstanding of the binary authenticity rule",
    "T2_arbitration_layer_failure":
        "presence is accurate in D2/D3 but final verdict stays poor and "
        "present-true-but-real persists -> the model identifies manipulation "
        "presence correctly and fails during final evidence arbitration",
    "T3_decomposition_fixes_arbitration":
        "D3 > D2 with presence accuracy preserved and verdict consistency clearly "
        "improved -> separating existence from extent and enforcing presence-first "
        "order improves forensic arbitration",
    "T4_framing_failure_persists":
        "D1/D2/D3 all stay at low recall -> the framing-induced policy shift is "
        "deeper than a task-definition misunderstanding",
}

REASON_AUDIT = dict(
    method="deterministic keyword rules; no LLM judge",
    patterns=dict(
        locality=["localized", "localised", "confined", "limited", "isolated",
                  "restricted", "specific region", "specific area", "only a",
                  "narrow", "small region", "small area", "particular region"],
        score_high=["score of 0.9", "score of 1.0", "high score", "score indicates",
                    "high likelihood", "score suggests", "elevated score"],
        manipulation_acknowledged=["manipulat", "alter", "tamper", "edited",
                                   "modified", "spliced", "retouch"],
        evidence_weak=["weak", "not strong", "insufficient", "inconclusive",
                       "not conclusive", "lacks", "minimal evidence",
                       "does not provide strong"],
        local_exemption=["does not necessarily", "not necessarily make",
                         "may still be considered real", "still be considered "
                         "authentic", "does not make the whole", "overall authentic",
                         "largely authentic", "mostly authentic", "still authentic",
                         "not enough to classify", "insufficient to classify"],
        rule_consistent=["any genuine manipulation", "even though the manipulation is",
                         "regardless of how small", "regardless of size",
                         "even if localized", "even if localised",
                         "binary", "any part of the image has been",
                         "however small", "no matter how small"],
    ),
    new_this_round="local_exemption captures 'localized manipulation does not make "
                   "the whole image fake'; rule_consistent captures explicit "
                   "adherence to the binary rule and is expected mainly in D1/D3",
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
    fsrc = os.path.join(R, "trufor_fields",
                        "trufor_structured_field_ablation_frozen.json")
    prev = json.load(open(fsrc))
    S = prev["samples"]

    print("=" * 94)
    print("1. SAMPLES — unchanged")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    assert len({s["coco_id"] for s in S}) == len(S)
    print(f"  184 fake + 184 paired real = {2*len(S)} images; single-image input")
    print(f"  inherited from the field-ablation freeze (sha1 {sha1f(fsrc)[:12]})")
    print(f"  model 7B {cfg['model']['path'].rstrip('/').split('/')[-1][:16]}")
    print(f"  no raw map in any condition")

    print("\n" + "=" * 94)
    print("2. EVIDENCE PACKAGE — identical in all four conditions")
    print(f"  full structured payload: grid_location, area_ratio, mean_probability,")
    print(f"  max_probability, centroid")
    print(f"  extraction inherited unchanged: thr {BIN_THR}, min area "
          f"{MIN_AREA_RATIO}, top-K {TOP_K}, ranked by {RANK_BY}")
    ev = S[0]["evidence"]["fake_own"]
    full = abstract(np.load(ev["npz"])["map"].astype(np.float32), ev["score"])
    print(f"  example: {json.dumps(full)[:150]}...")
    print(f"  NO field ablation this round; the payload is a constant across D0-D3")

    print("\n" + "=" * 94)
    print("3. PROMPT DIFFERENCES (payload and evidence text held fixed)")
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in PROMPTS.items()}
    for k, v in CONDITIONS.items():
        bits = []
        if v["rule"]:
            bits.append("binary rule")
        if v["decomposition"]:
            bits.append("presence/extent fields")
        if v["ordered"]:
            bits.append("presence-first order")
        print(f"  {k:<20} {'+ ' + ', '.join(bits) if bits else 'baseline':<46} "
              f"schema={v['schema']}")
        print(f"  {'':<20} sha1 {ph[k]}")
    print(f"\n  D0 is BYTE-IDENTICAL to the frozen S5_full prompt: "
          f"{PROMPTS['D0_baseline'] == prev['prompts']['S5_full']}")
    assert PROMPTS["D0_baseline"] == prev["prompts"]["S5_full"]
    assert ph["D0_baseline"] == prev["prompt_hashes"]["S5_full"]
    print(f"  so the replication check against last round is exact, not approximate")
    shared = HEAD + EVID + SCORE + PAYLOAD + TASK
    print(f"  all four share the same opening {len(shared)} chars: "
          f"{all(v.startswith(shared) for v in PROMPTS.values())}")
    assert all(v.startswith(shared) for v in PROMPTS.values())
    print(f"  D2 and D3 differ ONLY by the ordered-reasoning block: "
          f"{PROMPTS['D3_presence_first'].replace(ORDERED,'') == PROMPTS['D2_decomposition']}")
    assert PROMPTS["D3_presence_first"].replace(ORDERED, "") == \
        PROMPTS["D2_decomposition"]
    bias = ("accurate", "reliable", "trusted", "trust the tool", "auroc", "benchmark",
            "threshold", "ground truth", "donor", "correct answer", "state-of-the-art")
    hits = {k: [b for b in bias if b in v.lower()] for k, v in PROMPTS.items()}
    print(f"  bias/leak wording: { {k:v for k,v in hits.items() if v} or 'none'}")
    assert not any(hits.values())
    print(f"  the rule states the TASK DEFINITION only; it never says the tool is")
    print(f"  reliable, never says this evidence is genuine, never reveals an answer")

    print("\n" + "=" * 94)
    print("4. THE THREE INSERTED BLOCKS, VERBATIM")
    for nm, blk in (("binary rule (D1,D2,D3)", RULE),
                    ("decomposition (D2,D3)", DECOMP),
                    ("ordered arbitration (D3)", ORDERED)):
        print(f"\n  [{nm}]")
        print("  " + blk.strip().replace("\n", "\n  "))

    print("\n" + "=" * 94)
    print("5. OUTPUT SCHEMAS")
    print("  D0/D1  final_verdict, referenced_regions, reason")
    print("  D2/D3  + manipulation_present, manipulation_extent")
    print("  the final verdict is ALWAYS produced by the model; no external program")
    print("  ever overrides it from manipulation_present")

    print("\n" + "=" * 94)
    print("6. METRICS")
    for k, v in CONTRASTS.items():
        print(f"  {k:<32} {v}")
    print()
    for k, v in PRESENCE_METRICS.items():
        print(f"  presence.{k:<14} {v}")
    print()
    for k, v in INCONSISTENCY.items():
        print(f"  inconsistency.{k:<10} {v}")
    print()
    for k, v in EXTENT.items():
        print(f"  extent.{k:<16} {v}")
    print(f"\n  faithfulness (SECONDARY): citation rate, spatial agreement, "
          f"unsupported-region rate")
    print(f"  goal is recall recovery WITHOUT losing the spatial faithfulness already")
    print(f"  achieved (agreement was 1.000 in the abstraction round)")

    print("\n" + "=" * 94)
    print("7. REASON AUDIT")
    for k, v in REASON_AUDIT["patterns"].items():
        print(f"  {k:<26} {len(v)} patterns")
    print(f"  new: {REASON_AUDIT['new_this_round']}")

    print("\n" + "=" * 94)
    print("8. CASES")
    for k, v in CASES.items():
        print(f"  {k}")
        print(f"    {v}")

    print("\n" + "=" * 94)
    print("9. REPLICATION CHECK AND BUDGET")
    n = len(PROMPTS) * 2 * len(S)
    print(f"  D0 must land near the frozen full-structured baseline recall 0.114")
    print(f"  (field round S5). D0 uses that exact prompt, so a large deviation means")
    print(f"  pipeline drift, NOT a finding.")
    print(f"  {len(PROMPTS)} conditions x 2 labels x {len(S)} = {n} inferences")
    print(f"  all single-image; 7B at ~2.7 s/inf -> ~{n*2.7/60:.0f} min")
    print(f"  D2/D3 emit more fields, so expect somewhat longer outputs")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="Task-semantics intervention / verdict decomposition",
        question="Why does structured forensic framing make the model concede "
                 "manipulation yet still output Real?",
        separates=["binary authenticity rule misunderstanding",
                   "manipulation presence judgement failure",
                   "extent / existence confusion",
                   "final arbitration failure"],
        out_of_scope=["raw map", "confidence map", "new detector",
                      "numeric counterfactual", "threshold tuning", "LoRA",
                      "training", "gate", "verifier", "32B", "new dataset",
                      "field ablation"],
        evidence_package=dict(
            kind="full structured evidence, CONSTANT across D0-D3",
            fields=["grid_location", "area_ratio", "mean_probability",
                    "max_probability", "centroid"],
            extraction=dict(binary_threshold=BIN_THR,
                            min_area_ratio=MIN_AREA_RATIO, top_k=TOP_K,
                            rank_by=RANK_BY,
                            source="freeze_abstraction.abstract(), imported unchanged"),
            raw_map=False),
        conditions=CONDITIONS, prompts=PROMPTS, prompt_hashes=ph,
        inserted_blocks=dict(binary_rule=RULE, decomposition=DECOMP,
                             ordered_arbitration=ORDERED),
        prompt_parity=dict(
            d0_is_byte_identical_to_S5=True,
            shared_opening_chars=len(shared),
            d3_minus_ordered_equals_d2=True,
            note="evidence description, score formatting and payload are identical; "
                 "only the decision instruction and output schema change"),
        output_schemas=dict(base=SCHEMA_BASE, decomposed=SCHEMA_DEC),
        verdict_policy="the model always produces final_verdict itself; no external "
                       "program derives or overrides it from manipulation_present",
        contrasts=CONTRASTS, presence_metrics=PRESENCE_METRICS,
        inconsistency=INCONSISTENCY, extent_analysis=EXTENT,
        faithfulness="secondary: citation rate, spatial agreement, unsupported-region "
                     "rate; must not regress while recall recovers",
        reason_audit=REASON_AUDIT, interpretation_cases=CASES,
        replication_check=dict(D0_expect_recall=0.114,
                               source="field round S5_full",
                               policy="a large deviation indicates pipeline drift, "
                                      "not a finding"),
        prior_results=dict(
            score_only=dict(recall=0.837, fpr=0.011, j=0.826),
            structured_framing_no_payload=dict(recall=0.125, fpr=0.005, j=0.120),
            full_structured_S5=dict(recall=0.114),
            framing_effect_recall=-0.712,
            locality_without_locality_data=dict(P1=0.489, P2=0.989),
            acknowledged_manipulation_but_real=dict(P0=0.000, P2=0.957)),
        n_sources=len(S), inference_budget=dict(total=n, per_cell=len(S)),
        samples_inherited_from="trufor_structured_field_ablation_frozen.json",
        model=cfg["model"]["path"], decoding=cfg["model"]["generation"],
        min_pixels=cfg["model"]["min_pixels"], max_pixels=cfg["model"]["max_pixels"],
        outputs=["verdicts.jsonl", "run_meta.json", "analysis.txt",
                 "presence_extent_analysis.json", "inconsistency_analysis.json"],
        goal="mechanism verification, NOT prompt search. Only if D1/D3 restore "
             "decision utility while preserving spatial faithfulness is a formal "
             "Evidence Arbitration Layer justified.",
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    field_protocol=sha1f(fsrc)),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"], real=s["real"],
                      evidence=s["evidence"]) for s in S])
    od = os.path.join(R, "trufor_semantics")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_task_semantics_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} sources, {n} inferences")


if __name__ == "__main__":
    main()
