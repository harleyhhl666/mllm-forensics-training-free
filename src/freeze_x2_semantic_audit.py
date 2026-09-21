"""Freeze the semantic explanation audit protocol (Phase X2). No inference, no
annotation. Only sample selection, the annotator prompt, the schema and the metrics.

Canonical conditions: 7B D0_baseline and 32B B2_structured. These two prompts are
BYTE-IDENTICAL (sha1 22b9ad9a944812ef...), so the only difference between the two
explanation sets is the model. The abstraction-round E2 prompt is NOT used: it carries
an extra sentence explaining area_ratio and coordinate normalisation, which would make
the 7B text non-comparable to 32B.

Neither canonical condition contains the binary rule, so rule wording cannot colour
the reason text.
"""
import argparse, hashlib, json, os, random, sys
from collections import Counter, defaultdict
import numpy as np, yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import CELL_NAMES

SEED = 20260918
N_FAKE, N_REAL = 100, 50
N_REPEAT_FAKE, N_REPEAT_REAL = 20, 10

CLAIM_TYPES = ["location", "object", "texture", "lighting", "boundary", "geometry",
               "physics", "manipulation", "other"]
SUPPORT = ["supported", "partially_supported", "unsupported", "uncertain",
           "not_applicable"]

# ----------------------------------------------------------------- annotator prompt
PROMPT_FAKE = """You are auditing a written analysis of a photograph against supplied visual evidence.

You are given:
1. IMAGE A - the photograph the analysis refers to.
2. IMAGE B - a source photograph of the same scene, provided for comparison.
3. IMAGE C - a binary annotation mask marking a region of interest in IMAGE A.
4. IMAGE D - IMAGE A with that region outlined.
5. A written EXPLANATION about IMAGE A.
6. REFERENCED_REGIONS: the 3x3 grid cells the explanation names.

The 3x3 grid divides the image into nine equal cells, named:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right.

Your task is to evaluate whether each statement in the EXPLANATION is supported by the
supplied visual evidence and annotations. You are not asked to decide whether the
photograph is authentic, and you are not asked to produce your own analysis.

Procedure:

STEP 1 - Decompose the EXPLANATION into atomic claims. A sentence such as
"The bottom-right chair has unnatural texture and inconsistent lighting, suggesting
local editing" contains several separable claims: that the relevant region is
bottom-right; that a chair is present there; that its texture is unnatural; that
lighting is inconsistent; and that this indicates local editing. List every claim
separately. Do not merge claims and do not add claims the explanation never made.

STEP 2 - Assign each claim a type from exactly this list:
location, object, texture, lighting, boundary, geometry, physics, manipulation, other.
  location   - a position in the image, e.g. "top-left", "the left side"
  object     - the presence of a thing, e.g. "a chair", "a person", "a wall"
  texture    - surface appearance, e.g. unnatural texture, smoothing, repeated pattern
  lighting   - illumination or shadow, e.g. brightness mismatch, shadow mismatch
  boundary   - edges, e.g. seam, blending artifact, edge discontinuity, halo
  geometry   - shape or perspective, e.g. distortion, malformed structure
  physics    - physical plausibility, e.g. impossible shadow, implausible contact
  manipulation - a claim that content was inserted, replaced, edited or inpainted
  other      - anything that fits none of the above

STEP 3 - Judge each claim:
  supported            - the imagery or the annotation clearly supports it
  partially_supported  - partly true but overstated, e.g. "strong lighting
                         inconsistency" where only a slight brightness change is visible
  unsupported          - no observable evidence supports it
  uncertain            - resolution, the annotation, or the visible information is
                         insufficient to judge reliably
  not_applicable       - the claim does not assert anything checkable

Rules you must follow:

- Judge only whether a claim is supported by what you can observe. Do not reason
  backwards from any assumption about the photograph's status.
- IMAGE C marks a region. Use it to check whether a stated location falls in that
  region, and use IMAGE A versus IMAGE B to check whether content there differs. The
  mask alone never establishes a texture, lighting, geometry or physics anomaly - those
  must be visible to you.
- A claim about a specific mechanism, for example that content was copy-moved from
  elsewhere in the same image, requires evidence for that specific mechanism. Do not
  accept it merely because the region differs from IMAGE B.
- Use uncertain freely. Texture, lighting and subtle boundary effects are often not
  reliably judgeable. Hedged wording such as "there may be a slight texture
  inconsistency" describing something genuinely ambiguous should be uncertain, not
  unsupported.
- Do not guess. uncertain is always preferable to an unsupported guess.

Return STRICT JSON only, with this shape:

{
  "claims": [
    {"claim_id": "C1", "claim_text": "...", "claim_type": "location",
     "support": "supported",
     "evidence_basis": "gt_mask|paired_real|visible_fake_image|visible_real_image|none",
     "brief_justification": "..."}
  ],
  "region_correct": "yes|no|uncertain|not_applicable",
  "object_correct": "yes|no|partial|uncertain|not_applicable",
  "manipulation_claim_supported": "yes|no|partial|uncertain|not_applicable",
  "unsupported_claim_present": true,
  "contradictory_claim_present": false,
  "hallucinated_object_present": false,
  "overall_semantic_support": "supported|partially_supported|unsupported|uncertain",
  "confidence": "high|medium|low",
  "notes": "..."
}

EXPLANATION:
{explanation}

REFERENCED_REGIONS: {referenced_regions}
"""

PROMPT_REAL = """You are auditing a written analysis of a photograph against the photograph itself.

You are given:
1. IMAGE A - the photograph the analysis refers to.
2. A written EXPLANATION about IMAGE A.
3. REFERENCED_REGIONS: the 3x3 grid cells the explanation names.

The 3x3 grid divides the image into nine equal cells, named:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right.

Your task is to evaluate whether each statement in the EXPLANATION is supported by what
is visible in IMAGE A. You are not asked to decide whether the photograph is authentic,
and you are not asked to produce your own analysis. No annotation mask is available for
this image, so every judgement rests on what you can see.

Procedure:

STEP 1 - Decompose the EXPLANATION into atomic claims, listing each separately. Do not
merge claims and do not add claims the explanation never made.

STEP 2 - Assign each claim a type from exactly this list:
location, object, texture, lighting, boundary, geometry, physics, manipulation, other.
  location   - a position in the image, e.g. "top-left", "the left side"
  object     - the presence of a thing, e.g. "a chair", "a person", "a wall"
  texture    - surface appearance, e.g. unnatural texture, smoothing, repeated pattern
  lighting   - illumination or shadow, e.g. brightness mismatch, shadow mismatch
  boundary   - edges, e.g. seam, blending artifact, edge discontinuity, halo
  geometry   - shape or perspective, e.g. distortion, malformed structure
  physics    - physical plausibility, e.g. impossible shadow, implausible contact
  manipulation - a claim that content was inserted, replaced, edited or inpainted
  other      - anything that fits none of the above

STEP 3 - Judge each claim:
  supported            - what is visible clearly supports it
  partially_supported  - partly true but overstated
  unsupported          - no observable evidence supports it
  uncertain            - the visible information is insufficient to judge reliably
  not_applicable       - the claim does not assert anything checkable

Rules you must follow:

- Judge only whether a claim is supported by what you can observe. Do not reason
  backwards from any assumption about the photograph's status. A claim of an anomaly is
  not automatically wrong, and an absence of stated anomalies is not automatically
  right.
- If the explanation names an object, check that the object exists and sits where the
  explanation places it.
- Use uncertain freely; texture, lighting and subtle boundary effects are often not
  reliably judgeable. Hedged wording about something genuinely ambiguous should be
  uncertain, not unsupported.
- Do not guess. uncertain is always preferable to an unsupported guess.

Return STRICT JSON only, with this shape:

{
  "claims": [
    {"claim_id": "C1", "claim_text": "...", "claim_type": "location",
     "support": "supported",
     "evidence_basis": "visible_real_image|none",
     "brief_justification": "..."}
  ],
  "region_correct": "yes|no|uncertain|not_applicable",
  "object_correct": "yes|no|partial|uncertain|not_applicable",
  "manipulation_claim_supported": "yes|no|partial|uncertain|not_applicable",
  "unsupported_claim_present": true,
  "contradictory_claim_present": false,
  "hallucinated_object_present": false,
  "overall_semantic_support": "supported|partially_supported|unsupported|uncertain",
  "confidence": "high|medium|low",
  "notes": "..."
}

EXPLANATION:
{explanation}

REFERENCED_REGIONS: {referenced_regions}
"""

SCHEMA = dict(
    claims=[dict(claim_id="str", claim_text="str", claim_type=CLAIM_TYPES,
                 support=SUPPORT,
                 evidence_basis=["gt_mask", "paired_real", "visible_fake_image",
                                 "visible_real_image", "none"],
                 brief_justification="str")],
    region_correct=["yes", "no", "uncertain", "not_applicable"],
    object_correct=["yes", "no", "partial", "uncertain", "not_applicable"],
    manipulation_claim_supported=["yes", "no", "partial", "uncertain",
                                 "not_applicable"],
    unsupported_claim_present="bool", contradictory_claim_present="bool",
    hallucinated_object_present="bool",
    overall_semantic_support=["supported", "partially_supported", "unsupported",
                             "uncertain"],
    confidence=["high", "medium", "low"], notes="str")

METRICS = dict(
    claim_level_support="discrete proportions of supported / partially_supported / "
                        "unsupported / uncertain, per claim_type. partial is NOT "
                        "collapsed to 0.5 in the primary report.",
    denominator="EVALUABLE claims only: not_applicable is excluded and never counted "
                "as correct. Each claim_type rate uses only explanations that actually "
                "made a claim of that type.",
    unsupported_claim_rate="UCR = unsupported / evaluable, reported for all claims and "
                           "per type (texture, lighting, boundary, geometry, physics, "
                           "manipulation)",
    explanation_level="proportions of fully supported / partially supported / "
                      "unsupported / uncertain",
    hallucinated_object_rate="HOR = explanations containing a nonexistent object / all "
                             "explanations",
    false_forensic_explanation_rate="FFER, REAL ONLY = real explanations asserting a "
                                    "forensic anomaly judged unsupported / all real "
                                    "explanations",
    paired_7b_32b="per source: unsupported claim count, hallucinated object, overall "
                  "support, and manipulation / texture / lighting / boundary support. "
                  "Report means AND paired counts of 7B-better / tie / 32B-better.",
    citation_split="cited vs no-citation reported separately; a no-citation explanation "
                   "is NOT scored as an explanation error",
    stratified_by_spatial_correctness="split fake by Phase X1 Explanation->GT hit, to "
                                      "test whether getting the location right goes "
                                      "with getting the semantics right",
    stratified_by_gt_extent="small / medium / large GT area, since small edits may be "
                            "harder to verify visually",
)

UNCERTAINTY_POLICY = dict(
    uncertain_is_first_class="uncertain is never merged into unsupported and never "
                             "counted as an error",
    hedged_language="hedged wording about something genuinely ambiguous is uncertain",
    annotator_status="Claude annotations are an ASSISTIVE measurement, NOT ground "
                     "truth. Reliability is quantified by the repeat subset and any "
                     "reported metric must carry that caveat.",
    disagreement="repeat-subset disagreements are reported per field, with Cohen's "
                 "kappa for the binary fields; disagreeing items are listed for "
                 "optional human spot-check and are NOT silently resolved",
    exclusion="an explanation is excluded only if the annotator returns unparseable "
              "JSON twice; exclusions are counted and reported, never backfilled",
)


def sha1s(s):
    return hashlib.sha1(s.encode()).hexdigest()


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def gt_extent(mask_path, thr=127):
    im = np.array(Image.open(mask_path).convert("L"))
    m = im > thr
    H, W = m.shape
    tot = m.sum()
    if tot == 0:
        return 0.0, 0
    n = 0
    for r in range(3):
        for c in range(3):
            sub = m[r * H // 3:(r + 1) * H // 3, c * W // 3:(c + 1) * W // 3]
            if sub.sum() / tot >= 0.05:
                n += 1
    return float(tot / (H * W)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    thr = cfg["datasets"].get("mask_binarize_threshold", 127)

    conf = json.load(open(os.path.join(
        R, "trufor_conflict", "trufor_score_map_conflict_frozen.json")))
    SM = {s["sample_id"]: s for s in conf["samples"]}
    se = json.load(open(os.path.join(
        R, "trufor_semantics", "trufor_task_semantics_frozen.json")))
    f32 = json.load(open(os.path.join(
        R, "trufor_32b_final", "trufor_32b_final_capacity_frozen.json")))

    print("=" * 96)
    print("1. CANONICAL CONDITIONS")
    d0, b2 = se["prompts"]["D0_baseline"], f32["prompts"]["B2_structured"]
    ab = json.load(open(os.path.join(
        R, "trufor_abstraction", "trufor_evidence_abstraction_frozen.json")))
    e2 = ab["prompts"]["E2"]
    print(f"  7B  D0_baseline   sha1 {sha1s(d0)[:16]}")
    print(f"  32B B2_structured sha1 {sha1s(b2)[:16]}")
    print(f"  D0 == B2 byte-identical: {d0 == b2}")
    assert d0 == b2
    print(f"  7B  E2 (abstraction round) sha1 {sha1s(e2)[:16]}  == B2: {e2 == b2}")
    print(f"  CHOICE: canonical = D0 (7B) / B2 (32B).")
    print(f"  REASON: they are byte-identical, so the 7B-vs-32B explanation comparison")
    print(f"  has exactly one variable, the model. E2 is rejected because it adds a")
    print(f"  sentence explaining area_ratio and coordinate normalisation, which would")
    print(f"  make the 7B text non-comparable.")
    print(f"  Neither condition contains the binary rule: "
          f"{'binary' not in d0.lower()}")
    assert "authenticity label is binary" not in d0

    V7 = {}
    for l in open(os.path.join(R, "trufor_semantics", "verdicts.jsonl")):
        d = json.loads(l)
        if d["condition"] == "D0_baseline":
            V7[(d["sample_id"], d["label"])] = d
    V32 = {}
    for l in open(os.path.join(R, "trufor_32b_final", "verdicts.jsonl")):
        d = json.loads(l)
        if d["condition"] == "B2_structured":
            V32[(d["sample_id"], d["label"])] = d
    print(f"\n  7B D0 cells {len(V7)}   32B B2 cells {len(V32)}")

    print("\n" + "=" * 96)
    print("2. FAKE SAMPLING — stratified, seed fixed BEFORE any annotation")
    fakes = sorted({p for (p, l) in V7 if l == "fake"} & {p for (p, l) in V32
                                                          if l == "fake"}
                   & set(SM))
    print(f"  eligible fake sources (present in BOTH models) {len(fakes)}")
    meta = {}
    for s in fakes:
        ar, nc = gt_extent(os.path.join(root, SM[s]["mask"]), thr)
        meta[s] = dict(area_ratio=ar, n_cells=nc, mask_type=SM[s]["mask_type"],
                       category=SM[s]["category"])
    ars = np.array([meta[s]["area_ratio"] for s in fakes])
    qs = np.percentile(ars, [25, 50, 75])
    for s in fakes:
        meta[s]["area_q"] = int(np.digitize(meta[s]["area_ratio"], qs)) + 1
        n = meta[s]["n_cells"]
        meta[s]["spread"] = "1cell" if n <= 1 else ("2-3cells" if n <= 3 else ">=4cells")
    print(f"  GT area_ratio quartile cuts {np.round(qs, 4)}")
    rng = random.Random(SEED)
    # proportional allocation across area quartile x spread, then top-up
    strata = defaultdict(list)
    for s in fakes:
        strata[(meta[s]["area_q"], meta[s]["spread"])].append(s)
    for k in strata:
        strata[k].sort()
        rng.shuffle(strata[k])
    sel_f, keys = [], sorted(strata)
    target = {k: max(1, round(N_FAKE * len(strata[k]) / len(fakes))) for k in keys}
    for k in keys:
        sel_f += strata[k][:target[k]]
    pool = [s for s in fakes if s not in set(sel_f)]
    rng.shuffle(pool)
    while len(sel_f) < N_FAKE:
        sel_f.append(pool.pop())
    if len(sel_f) > N_FAKE:
        rng.shuffle(sel_f)
        sel_f = sel_f[:N_FAKE]
    sel_f = sorted(sel_f)
    print(f"  selected {len(sel_f)} fake")
    print(f"  {'stratum':<22}{'population':>12}{'selected':>10}")
    for k in keys:
        n_sel = sum(1 for s in sel_f if (meta[s]["area_q"], meta[s]["spread"]) == k)
        print(f"  Q{k[0]} {k[1]:<18}{len(strata[k]):>12}{n_sel:>10}")
    print(f"\n  area quartile: "
          f"{dict(sorted(Counter(meta[s]['area_q'] for s in sel_f).items()))}")
    print(f"  grid spread  : {dict(Counter(meta[s]['spread'] for s in sel_f))}")
    print(f"  mask_type    : {dict(Counter(meta[s]['mask_type'] for s in sel_f))}"
          f"   population {dict(Counter(meta[s]['mask_type'] for s in fakes))}")
    print(f"  categories   : {len({meta[s]['category'] for s in sel_f})} distinct")
    vf7 = np.mean([V7[(s, 'fake')]['final_verdict'] == 'fake' for s in sel_f])
    vf32 = np.mean([V32[(s, 'fake')]['final_verdict'] == 'fake' for s in sel_f])
    print(f"  NOT selected on verdict; resulting 7B fake-verdict rate {vf7:.3f}, "
          f"32B {vf32:.3f}")

    print("\n" + "=" * 96)
    print("3. REAL SAMPLING — one shared set, audited for BOTH models")
    reals_all = sorted({p for (p, l) in V7 if l == "real"} & {p for (p, l) in V32
                                                             if l == "real"})
    # A fake unit shows the annotator the paired real image as "IMAGE B, a source
    # photograph". If that same image later arrived as a real unit, the annotator
    # would already have seen it framed as a source. So the real set is drawn ONLY
    # from sources not selected for the fake set.
    reals = [s for s in reals_all if s not in set(sel_f)]
    print(f"  all real sources {len(reals_all)}; excluding the {len(sel_f)} used as")
    print(f"  fake units (their real image is shown there as IMAGE B) leaves "
          f"{len(reals)}")
    cite7 = {s: bool(V7[(s, "real")]["referenced_regions"]) and
             len(V7[(s, "real")]["referenced_regions"]) != 9 for s in reals}
    cite32 = {s: bool(V32[(s, "real")]["referenced_regions"]) and
              len(V32[(s, "real")]["referenced_regions"]) != 9 for s in reals}
    grp = defaultdict(list)
    for s in reals:
        grp[(cite7[s], cite32[s])].append(s)
    print(f"  eligible real sources {len(reals)}")
    print(f"  citation pattern (7B cited, 32B cited):")
    for k in sorted(grp, key=lambda x: (-len(grp[x]), str(x))):
        print(f"    7B={'cite' if k[0] else 'none':<5} 32B="
              f"{'cite' if k[1] else 'none':<5}  {len(grp[k])}")
    rng2 = random.Random(SEED + 1)
    sel_r = []
    # guarantee presence of both 32B-cited and 32B-uncited, proportionally otherwise
    for k in sorted(grp):
        grp[k].sort()
        rng2.shuffle(grp[k])
    both = [k for k in grp if grp[k]]
    alloc = {k: max(1, round(N_REAL * len(grp[k]) / len(reals))) for k in both}
    for k in sorted(alloc):
        sel_r += grp[k][:alloc[k]]
    poolr = [s for s in reals if s not in set(sel_r)]
    rng2.shuffle(poolr)
    while len(sel_r) < N_REAL:
        sel_r.append(poolr.pop())
    if len(sel_r) > N_REAL:
        rng2.shuffle(sel_r)
        sel_r = sel_r[:N_REAL]
    sel_r = sorted(sel_r)
    print(f"  selected {len(sel_r)} real")
    print(f"  7B cited {sum(cite7[s] for s in sel_r)}/{len(sel_r)}   "
          f"32B cited {sum(cite32[s] for s in sel_r)}/{len(sel_r)}")
    print(f"  both classes present for 32B: "
          f"{0 < sum(cite32[s] for s in sel_r) < len(sel_r)}")
    assert 0 < sum(cite32[s] for s in sel_r) < len(sel_r)

    print("\n" + "=" * 96)
    print("4. ANNOTATION UNITS — same source audited for both models")
    units = []
    for s in sel_f:
        for mdl, V in (("M1", V7), ("M2", V32)):
            units.append(dict(source=s, label="fake", model_code=mdl))
    for s in sel_r:
        for mdl, V in (("M1", V7), ("M2", V32)):
            units.append(dict(source=s, label="real", model_code=mdl))
    print(f"  fake units {2*len(sel_f)}   real units {2*len(sel_r)}   "
          f"total {len(units)}")
    print(f"  every selected source contributes one 7B and one 32B explanation")

    print("\n" + "=" * 96)
    print("5. ANONYMISATION")
    rng3 = random.Random(SEED + 2)
    order = list(range(len(units)))
    rng3.shuffle(order)
    for i, idx in enumerate(order):
        units[idx]["anon_id"] = f"EXPL-{i+1:04d}"
    MAP = {}
    for u in units:
        V = V7 if u["model_code"] == "M1" else V32
        r = V[(u["source"], u["label"])]
        u["explanation"] = r["reason"]
        u["referenced_regions"] = r["referenced_regions"]
        MAP[u["anon_id"]] = dict(source=u["source"], label=u["label"],
                                 model_code=u["model_code"])
    print(f"  anon IDs EXPL-0001..EXPL-{len(units):04d}, order shuffled seed {SEED+2}")
    print(f"  model_code M1/M2 is stored ONLY in the mapping file, never shown")
    print(f"  each explanation is annotated INDEPENDENTLY — the annotator never sees")
    print(f"  two explanations of the same image together, so no relative comparison")
    leak = ("7b", "32b", "qwen", "trufor", "d0_", "b2_", "baseline", "condition",
            "verdict", "score", "structured evidence", "larger model")
    for nm, p in (("fake", PROMPT_FAKE), ("real", PROMPT_REAL)):
        hits = [w for w in leak if w in p.lower()]
        print(f"  {nm} prompt leak scan: {hits or 'none'}")
        assert not hits, hits
    banned = ("verify whether the model is correct", "this is a fake",
              "models often hallucinate", "structured evidence is accurate")
    for p in (PROMPT_FAKE, PROMPT_REAL):
        for b in banned:
            assert b not in p.lower()
    print(f"  banned leading phrasings absent: True")
    print(f"  prompts never mention verdict, TruFor score, condition or model identity")

    print("\n" + "=" * 96)
    print("6. WHAT THE ANNOTATOR SEES")
    print(f"  FAKE: image A (the photograph), image B (paired source), image C (mask),")
    print(f"        image D (outlined overlay), explanation text, referenced_regions,")
    print(f"        grid definition")
    print(f"  REAL: image A, explanation text, referenced_regions, grid definition")
    print(f"        NO paired counterpart — a comparison image could induce suspicion")
    print(f"  WITHHELD in both: TruFor score, structured evidence JSON, final verdict,")
    print(f"                    model identity, condition name, any prior metric")

    print("\n" + "=" * 96)
    print("7. REPEAT SUBSET")
    rng4 = random.Random(SEED + 3)
    rf = sorted(rng4.sample(sel_f, N_REPEAT_FAKE))
    rr = sorted(rng4.sample(sel_r, N_REPEAT_REAL))
    rep = [u["anon_id"] for u in units
           if (u["label"] == "fake" and u["source"] in rf)
           or (u["label"] == "real" and u["source"] in rr)]
    print(f"  repeat sources: {N_REPEAT_FAKE} fake + {N_REPEAT_REAL} real = "
          f"{len(rf)+len(rr)}")
    print(f"  repeat annotation units {len(rep)} (both models per source)")
    print(f"  re-annotated in a fresh independent context, identical rubric, first")
    print(f"  annotation never shown, identical image preprocessing")
    print(f"  same-model repeat consistency and any cross-model annotator agreement")
    print(f"  are reported SEPARATELY, never merged")

    print("\n" + "=" * 96)
    print("8. HASHES AND BUDGET")
    hf, hr = sha1s(PROMPT_FAKE), sha1s(PROMPT_REAL)
    hs = sha1s(json.dumps(SCHEMA, sort_keys=True))
    print(f"  fake prompt sha1   {hf}")
    print(f"  real prompt sha1   {hr}")
    print(f"  schema sha1        {hs}")
    print(f"  primary annotations {len(units)}")
    print(f"  repeat annotations  {len(rep)}")
    print(f"  TOTAL annotation calls {len(units)+len(rep)}")
    nimg = 4 * 2 * len(sel_f) + 1 * 2 * len(sel_r)
    print(f"  images passed to the annotator: {nimg} "
          f"(4 per fake unit, 1 per real unit)")
    el = [len(u["explanation"]) for u in units]
    print(f"  explanation length chars: mean {np.mean(el):.0f}  max {max(el)}")
    print(f"  rough token estimate: prompt ~{len(PROMPT_FAKE)//4} tok + images; the")
    print(f"  4-image fake units dominate cost. Expect the fake half to be several")
    print(f"  times the real half per unit.")

    print("\n" + "=" * 96)
    print("9. FREEZE AUDIT")
    ck = []
    ck.append(("100 fake / 50 real", len(sel_f) == N_FAKE and len(sel_r) == N_REAL))
    ck.append(("unique sources, fake and real sets disjoint",
               len(set(sel_f)) == N_FAKE and len(set(sel_r)) == N_REAL
               and not (set(sel_f) & set(sel_r))))
    ck.append(("7B and 32B both matched for every unit",
               all((u["source"], u["label"]) in (V7 if u["model_code"] == "M1"
                                                 else V32) for u in units)))
    ck.append(("fake stratification covers all 4 area quartiles",
               len({meta[s]["area_q"] for s in sel_f}) == 4))
    ck.append(("fake stratification covers all 3 spread classes",
               len({meta[s]["spread"] for s in sel_f}) == 3))
    ck.append(("both mask_types present",
               len({meta[s]["mask_type"] for s in sel_f}) == 2))
    ck.append((f"repeat subset fixed at {N_REPEAT_FAKE}+{N_REPEAT_REAL}",
               len(rf) == N_REPEAT_FAKE and len(rr) == N_REPEAT_REAL))
    ck.append(("anonymisation hides model identity",
               all("model_code" not in u.get("anon_id", "") for u in units)))
    ck.append(("annotator sees no verdict / score / condition",
               not any(w in (PROMPT_FAKE + PROMPT_REAL).lower()
                       for w in ("verdict", "trufor", "condition"))))
    ck.append(("schema fixed", bool(hs)))
    ck.append(("uncertain and not_applicable defined and non-punitive",
               "uncertain" in SUPPORT and "not_applicable" in SUPPORT))
    ck.append(("real set shared between models, not per-model",
               True))
    for nm, ok in ck:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}")
    if not all(ok for _, ok in ck):
        print("\n  AUDIT FAILED — not proceeding to freeze")
        sys.exit(1)

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="Phase X2 — semantic explanation audit protocol freeze",
        question="when the model says a region has some anomaly, is that natural-"
                 "language claim actually supported by the image?",
        not_the_question=["classification accuracy", "final verdict correctness"],
        canonical_conditions=dict(
            seven_b="D0_baseline", thirty_two_b="B2_structured",
            prompt_sha1=sha1s(d0), byte_identical=True,
            rejected="abstraction-round E2, which adds a sentence explaining "
                     "area_ratio and coordinate normalisation and would make the 7B "
                     "text non-comparable to 32B",
            no_binary_rule=True,
            reason="binary rule changes decision policy and its wording could colour "
                   "the reason text; this phase studies the explanation only"),
        sampling=dict(seed=SEED, n_fake=N_FAKE, n_real=N_REAL,
                      fake_strata="GT area quartile x grid spread, proportional",
                      area_quartile_cuts=[float(x) for x in qs],
                      not_selected_on=["verdict", "explanation quality",
                                       "spatial correctness"],
                      fake_ids=sel_f, real_ids=sel_r,
                      real_pool_disjoint_from_fake="the real set excludes every source "
                          "used as a fake unit, because a fake unit already shows that "
                          "source's real image to the annotator as IMAGE B",
                      fake_metadata={s: meta[s] for s in sel_f},
                      real_citation_status={s: dict(m1_cited=cite7[s],
                                                    m2_cited=cite32[s])
                                            for s in sel_r}),
        annotation_units=units, anonymization_map=MAP,
        anonymization=dict(scheme="EXPL-0001..", shuffle_seed=SEED + 2,
                           independent="each explanation annotated alone; the annotator "
                                       "never sees two explanations of one image, so it "
                                       "cannot make relative comparisons",
                           model_codes="M1 / M2 stored only in the mapping file"),
        prompts=dict(fake=PROMPT_FAKE, real=PROMPT_REAL),
        prompt_hashes=dict(fake=hf, real=hr),
        schema=SCHEMA, schema_sha1=hs,
        claim_types=CLAIM_TYPES, support_levels=SUPPORT,
        annotator_inputs=dict(
            fake=["image A: photograph", "image B: paired source",
                  "image C: binary mask", "image D: outlined overlay",
                  "explanation", "referenced_regions", "grid definition"],
            real=["image A: photograph", "explanation", "referenced_regions",
                  "grid definition"],
            withheld=["TruFor score", "structured evidence JSON", "final verdict",
                      "model identity", "condition name", "prior metrics"],
            real_no_counterpart="a paired counterpart is deliberately withheld for "
                                "real images; a comparison image could induce suspicion"),
        gt_usage=dict(allowed=["checking a stated location", "checking whether content "
                               "differs from the paired source"],
                      forbidden=["treating the mask as proof of a texture, lighting, "
                                 "geometry or physics anomaly"],
                      mechanism_claims="a specific mechanism such as copy-move needs "
                                       "evidence for that mechanism, not merely a "
                                       "difference from the source"),
        repeat_subset=dict(fake_sources=rf, real_sources=rr, unit_ids=rep,
                           n_units=len(rep), seed=SEED + 3,
                           procedure="fresh independent context, identical rubric and "
                                     "preprocessing, first annotation never shown",
                           optional_second_annotator="if a second annotator "
                                                     "configuration is used, same-model "
                                                     "repeat consistency and cross-model "
                                                     "agreement are reported separately"),
        metrics=METRICS, uncertainty_policy=UNCERTAINTY_POLICY,
        budget=dict(primary=len(units), repeat=len(rep),
                    total_calls=len(units) + len(rep), images=nimg,
                    explanation_chars_mean=float(np.mean(el))),
        hashes=dict(config=sha1f(args.config), freezer=sha1f(os.path.abspath(__file__)),
                    semantics_protocol=sha1f(os.path.join(
                        R, "trufor_semantics", "trufor_task_semantics_frozen.json")),
                    final32_protocol=sha1f(os.path.join(
                        R, "trufor_32b_final",
                        "trufor_32b_final_capacity_frozen.json"))),
        stop_condition="Phase X2 ends at freeze; no annotation is started",
    )
    od = os.path.join(R, "explainability")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "x2_semantic_audit_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(units)} primary + {len(rep)} repeat = {len(units)+len(rep)} calls")


if __name__ == "__main__":
    main()
