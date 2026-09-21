"""Freeze the M1 mitigation protocol: samples, donors, prompts, mapping, contrasts.

Seven conditions. The sham pair (S) exists to split the two things that M changes
at once relative to B: running an extra Stage-V pass, and injecting the Stage-V
state into Stage D. S runs the same Stage V but Stage D never sees its result, so
S-vs-B isolates deliberation and M-vs-S isolates verification-state gating.

Stage-V results are computed ONCE per (sample, label, cue) and reused by both the
M and S conditions, so the sham costs only the extra Stage-D calls.

DRY unless --freeze is passed.
"""
import argparse, hashlib, io, json, os, sys
from collections import Counter, defaultdict
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools

SEED = 20260918

# ------------------------------------------------------------------ STAGE V
STAGE_V = """You are performing a verification step on an automated image analysis output.

You are given:
1. An image under examination.
2. A colour-mapped visualization produced by an automated analysis procedure.

The procedure re-compresses an image and amplifies per-pixel differences. Strong
responses can arise from ordinary image content such as texture, edges, and fine
detail, or from local differences in how a region was encoded.

The 3x3 grid names for locations are exactly:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right.

Answer TWO independent questions. Do NOT decide whether the image is authentic or
altered. Do NOT mention authenticity. That decision happens in a later, separate
step.

QUESTION 1 - image_consistency:
Look at where the visualization's strongest responses fall. Does what occupies
those locations in the original image plausibly account for a strong response
there? Answer "matched" if the responses correspond to the image's actual content
and structure at those locations, or "mismatched" if the pattern of responses does
not correspond to what is present at those locations in this image.

QUESTION 2 - forensic_support:
Assuming the responses do correspond to this image, do they carry information
beyond what ordinary texture, edges, high-frequency detail, or overall image
structure would already produce? Answer "supported" only if the responses show
something that ordinary image content does not readily explain. Answer
"insufficient" if the responses are adequately explained by ordinary texture,
edges, detail, or structure.

Report strictly as JSON with these keys:
{
  "image_consistency": "matched" or "mismatched",
  "forensic_support": "supported" or "insufficient",
  "strongest_region": one 3x3 grid cell name,
  "reason": "one or two sentences"
}
Output JSON only."""

# ------------------------------------------------------------------ STAGE D
_VERDICT_TAIL = """
Determine whether this image is an authentic photograph or has been digitally
altered.

Report strictly as JSON with these keys:
{
  "final_verdict": "real" or "fake",
  "reason": "brief explanation"
}
Output JSON only."""

_D_HEAD = """You are examining a digital image.

A colour-mapped visualization produced by an automated analysis procedure is
provided alongside the image.

"""

STATES = {
    "A": ("An automated analysis produced a localized response that was judged "
          "consistent with the image and potentially informative for local "
          "editing. Treat it as supporting evidence, not proof."),
    "B": ("The visualization is consistent with image content, but its responses "
          "can be explained by ordinary texture, edges, or image structure. It "
          "should not be treated as evidence of editing."),
    "C": ("The visualization was determined not to correspond reliably to this "
          "image and should not be used as evidence."),
}
STAGE_D = {k: _D_HEAD + v + "\n" + _VERDICT_TAIL for k, v in STATES.items()}

# ------------------------------------------- baselines / sham Stage-D prompt
ELA_SHARED = """An automated Error Level Analysis (ELA) has been applied to this image.
ELA re-compresses an image and amplifies per-pixel differences. Strong responses
can arise from genuine image content such as texture and edges or from local
re-encoding differences, so ELA should be treated as one indicator rather than
proof."""

B_NONE = "You are examining a digital image.\n" + _VERDICT_TAIL
# B-correct, B-wrong, S-correct and S-wrong all use this identical prompt. For the
# S conditions no Stage-V output is placed in it, which is exactly what makes
# S-vs-B a pure deliberation contrast.
B_CUE = ("You are examining a digital image.\n\n" + ELA_SHARED +
         "\nThe visualization output is provided alongside the image.\n" + _VERDICT_TAIL)
M_NONE = B_NONE

PROMPTS = {"STAGE_V": STAGE_V, "STAGE_D_A": STAGE_D["A"], "STAGE_D_B": STAGE_D["B"],
           "STAGE_D_C": STAGE_D["C"], "B_NONE": B_NONE, "B_CUE": B_CUE,
           "M_NONE": M_NONE}

STATE_MAPPING = {
    "mismatched -> C": "any image_consistency=mismatched, regardless of support",
    "matched + supported -> A": "cue retains evidence status",
    "matched + insufficient -> B": "evidence status revoked as nondiagnostic",
    "unparseable Stage V -> C": "conservative; counted separately; a failed "
                                "verification is never treated as a pass",
}

CONDITIONS = {
    "B-none":    dict(cue=None,      stage_v=False, inject=False, prompt="B_NONE"),
    "B-correct": dict(cue="correct", stage_v=False, inject=False, prompt="B_CUE"),
    "B-wrong":   dict(cue="donor",   stage_v=False, inject=False, prompt="B_CUE"),
    "S-correct": dict(cue="correct", stage_v=True,  inject=False, prompt="B_CUE"),
    "S-wrong":   dict(cue="donor",   stage_v=True,  inject=False, prompt="B_CUE"),
    "M-correct": dict(cue="correct", stage_v=True,  inject=True,  prompt="STAGE_D_*"),
    "M-wrong":   dict(cue="donor",   stage_v=True,  inject=True,  prompt="STAGE_D_*"),
    "M-none":    dict(cue=None,      stage_v=False, inject=False, prompt="M_NONE"),
}

CONTRASTS = {
    "goal1_wrong_cue_bias_suppressed": "FPR(M-wrong) - FPR(B-wrong)  target < 0",
    "goal2_core_mitigation":           "J(M-correct) - J(B-correct)  target > 0",
    "goal3_structure_no_bias_fpr":     "FPR(M-none) - FPR(B-none)    target ~ 0",
    "goal3_structure_no_bias_tpr":     "TPR(M-none) - TPR(B-none)    target ~ 0",
    "sec_correct_cue_benefit_kept":    "TPR(M-correct) - TPR(B-none)",
    "sec_wrong_cue_harmless":          "J(M-wrong) - J(B-none)       target ~ 0",
    "attr_deliberation_wrong":         "FPR(S-wrong) - FPR(B-wrong)",
    "attr_deliberation_correct":       "J(S-correct) - J(B-correct)",
    "attr_gating_wrong":               "FPR(M-wrong) - FPR(S-wrong)  target < 0",
    "attr_gating_correct":             "J(M-correct) - J(S-correct)  target >= 0",
    "dJ_correct":                      "J(M-correct) - J(B-correct)",
    "dJ_wrong":                        "J(M-wrong) - J(B-wrong)",
}

ATTRIBUTION_RULES = {
    "S~B and M>S": "verification-state gating produces the mitigation benefit",
    "S>B and M~S": "benefit is mainly extra deliberation; do NOT attribute to gating",
    "S>B and M>S": "both deliberation and explicit gating contribute",
    "no improvement anywhere": "M1 mitigation fails",
}

NOTES = {
    "stage_v_character": "conservative evidence-verification criterion; the "
        "asymmetric 'supported only if' wording is prompt-induced conservatism, "
        "NOT naturally emergent model calibration",
    "m_none": "M_NONE is byte-identical to B_NONE; M-none is an execution "
        "stability check only and carries no causal interpretation",
    "practical_threshold": "project-internal engineering standard only, not a "
        "field standard: residual FPR(M-wrong) <= 0.10 counts as a large drop "
        "from the ~0.29-0.44 range observed earlier",
    "equivalence": "report differences with cluster bootstrap CIs; CI overlap is "
        "never used as evidence of equivalence",
    "stage_v_reuse": "one Stage-V result per (sample,label,cue) is shared by the "
        "M and S conditions; S differs only in that Stage D never sees it",
}


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-sources", type=int, default=100)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    ds = cfg["datasets"]["tgif_sp"]
    root, grid, tool_cfg = ds["root"], cfg["localization"]["grid"], cfg["forensic_tools"]
    R = os.path.join(out_root)

    idx = json.load(open(os.path.join(R, "phase0", "tgif_sp_index.json")))
    used = set(json.load(open(os.path.join(R, "used_coco_ids.json"))))
    V4 = {s["coco_id"] for s in json.load(open(os.path.join(
        R, "validation_v4", "validation_v4_frozen.json")))["samples"]}
    MECH = {s["coco_id"] for s in json.load(open(os.path.join(
        R, "mechanism", "mechanism_frozen.json")))["samples"]}

    print("=" * 84)
    print("DATA INDEPENDENCE")
    tc = {r["coco_id"] for r in idx}
    un = tc - used - V4 - MECH
    print(f"  testing-split coco_ids     : {len(tc)}")
    print(f"  used by calib/pilot/primary/secondary : {len(used & tc)}")
    print(f"  overlap with V4 / mechanism (validation split): "
          f"{len(un & V4)} / {len(un & MECH)}")
    print(f"  UNTOUCHED and available    : {len(un)}")
    assert not (un & V4) and not (un & MECH)
    pool = [r for r in idx if r["coco_id"] in un]
    havereal = sum(1 for r in pool if os.path.exists(os.path.join(root, r["real"])))
    print(f"  triples in pool            : {len(pool)}")
    print(f"  paired originals present   : {havereal}/{len(pool)}")

    n_src = min(args.n_sources, len(un))
    rng = np.random.default_rng(SEED)
    srcs = sorted(un)
    rng.shuffle(srcs)
    srcs = sorted(srcs[:n_src])
    by = defaultdict(list)
    for r in pool:
        by[r["coco_id"]].append(r)
    picked = []
    for c in srcs:
        cand = sorted(by[c], key=lambda r: r["pair_id"])
        rng.shuffle(cand)
        picked.append(cand[0])
    picked.sort(key=lambda r: r["pair_id"])
    trs = np.array([p["tamper_ratio"] for p in picked], float)
    print(f"\n  SAMPLED {len(picked)} fakes from "
          f"{len({p['coco_id'] for p in picked})} sources (1 variant each)")
    print(f"  tamper_ratio min={trs.min():.4f} med={np.median(trs):.4f} "
          f"max={trs.max():.4f}")
    print(f"  categories {dict(sorted(Counter(p['category'] for p in picked).items()))}")
    print(f"  mask types {dict(Counter(p['mask_type'] for p in picked))}")

    ids = [p["pair_id"] for p in picked]
    cidm = {p["pair_id"]: p["coco_id"] for p in picked}
    n, half = len(ids), len(ids) // 2
    donors = {}
    for k, pid in enumerate(ids):
        for st in range(n):
            c = ids[(k + half + st) % n]
            if c != pid and cidm[c] != cidm[pid]:
                donors[pid] = c
                break
    same = [p for p, d in donors.items() if cidm[d] == cidm[p]]
    selfd = [p for p, d in donors.items() if d == p]
    print(f"  donors assigned {len(donors)}  same-coco {len(same)}  self {len(selfd)}")
    assert not same and not selfd

    print("\n" + "=" * 84)
    print("PROMPT HASHES")
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in PROMPTS.items()}
    for k, v in ph.items():
        print(f"  {k:<12} {v}")
    assert ph["M_NONE"] == ph["B_NONE"], "M_NONE must equal B_NONE"
    print(f"  M_NONE == B_NONE : True (execution-stability check only)")

    print("\n" + "=" * 84)
    print("SHAM ISOLATION CHECK")
    for c in ("B-correct", "B-wrong", "S-correct", "S-wrong"):
        assert CONDITIONS[c]["prompt"] == "B_CUE"
    print("  B-correct / B-wrong / S-correct / S-wrong all use the identical")
    print("  B_CUE Stage-D prompt: True")
    for c in ("S-correct", "S-wrong"):
        assert CONDITIONS[c]["stage_v"] and not CONDITIONS[c]["inject"]
    print("  S conditions run Stage V but never inject its result: True")
    for tok in ("matched", "mismatched", "supported", "insufficient",
                "strongest_region"):
        assert tok not in B_CUE.lower(), tok
    print("  B_CUE leaks no Stage-V vocabulary (matched/supported/region): True")

    print("\n" + "=" * 84)
    print("INFERENCE BUDGET")
    # Stage V is run once per (sample,label,cue) and shared by M and S.
    nb = len(picked) * 2
    stage_v = nb * 2                 # correct + donor cue
    stage_d = nb * (1 + 2 + 2 + 2 + 1)   # B-none,B-cue x2,S x2,M x2,M-none
    print(f"  samples {len(picked)} x 2 labels = {nb} image instances")
    print(f"  Stage V (shared by M and S)        : {stage_v}")
    print(f"  Stage D / single-stage conditions  : {stage_d}")
    print(f"    B-none {nb}  B-correct {nb}  B-wrong {nb}")
    print(f"    S-correct {nb}  S-wrong {nb}  M-correct {nb}  M-wrong {nb}  "
          f"M-none {nb}")
    total = stage_v + stage_d
    print(f"  TOTAL                              : {total}  "
          f"(~{total*2.9/60:.0f} min at 2.9 s each)")

    print("\n" + "=" * 84)
    print("VISUALIZATION SANITY (first 3 samples)")
    for p in picked[:3]:
        own, _ = run_tools(os.path.join(root, p["fake"]), tool_cfg, grid=grid)
        d = next(r for r in pool if r["pair_id"] == donors[p["pair_id"]])
        dn, _ = run_tools(os.path.join(root, d["fake"]), tool_cfg, grid=grid)
        ov = Image.fromarray(cv2.cvtColor(own["ela"]["vis_bgr"], cv2.COLOR_BGR2RGB))
        dv = Image.fromarray(cv2.cvtColor(dn["ela"]["vis_bgr"], cv2.COLOR_BGR2RGB))
        print(f"  {p['pair_id'][:30]:<30} own={hashlib.sha1(png_bytes(ov)).hexdigest()[:12]} "
              f"donor={hashlib.sha1(png_bytes(dv)).hexdigest()[:12]} "
              f"differ={png_bytes(ov) != png_bytes(dv)}")

    code = {}
    for f in ("mitigation_freeze.py", "forensic_tools.py", "mllm_probe.py"):
        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
        if os.path.exists(pth):
            code[f] = hashlib.sha1(open(pth, "rb").read()).hexdigest()
    cfg_h = hashlib.sha1(open(args.config, "rb").read()).hexdigest()
    idx_h = hashlib.sha1(open(os.path.join(
        R, "phase0", "tgif_sp_index.json"), "rb").read()).hexdigest()
    print("\n" + "=" * 84)
    print("HASHES")
    print(f"  config      {cfg_h}")
    print(f"  index       {idx_h}")
    for k, v in code.items():
        print(f"  {k:<22}{v}")

    if not args.freeze:
        print("\nDRY RUN — nothing written. Pass --freeze to write the protocol.")
        return

    samples = [dict(pair_id=p["pair_id"], coco_id=p["coco_id"],
                    category=p["category"], mask_type=p["mask_type"],
                    variant=p["variant"], tamper_ratio=p["tamper_ratio"],
                    fake=p["fake"], real=p["real"], mask=p["mask"],
                    donor_pair_id=donors[p["pair_id"]]) for p in picked]
    doc = dict(
        protocol="M1 verification-gated mitigation, with sham two-stage attribution",
        seed=SEED, n_samples=len(samples), n_clusters=len({s["coco_id"] for s in samples}),
        sampling_rule=("1 variant per coco_id, 100 untouched TESTING-split coco_ids, "
                       "natural tamper distribution, no band balancing, seed 20260918"),
        split="testing", conditions=CONDITIONS, prompts=PROMPTS, prompt_hashes=ph,
        state_mapping=STATE_MAPPING, contrasts=CONTRASTS,
        attribution_rules=ATTRIBUTION_RULES, notes=NOTES,
        stage_v_metrics=["P(matched|fake,correct)", "P(supported|fake,correct)",
                         "P(mismatched|fake,donor)", "P(matched|real,own)",
                         "P(supported|real,own)", "P(insufficient|real,own)",
                         "state distribution A/B/C per cue x label",
                         "P(matched|correct) - P(matched|donor)"],
        inference_budget=dict(stage_v=stage_v, stage_d=stage_d, total=total),
        hashes=dict(config=cfg_h, index=idx_h, code=code), samples=samples)
    od = os.path.join(R, "mitigation")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "mitigation_frozen.json")
    json.dump(doc, open(fp, "w"), indent=1)
    fh = hashlib.sha1(open(fp, "rb").read()).hexdigest()
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {fh}")
    print(f"  samples {len(samples)}  clusters {doc['n_clusters']}  "
          f"total inferences {total}")


if __name__ == "__main__":
    main()
