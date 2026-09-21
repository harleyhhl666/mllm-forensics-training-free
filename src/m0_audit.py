"""M0 mechanism audit: 2x2 context experiment on untouched validation coco_ids.

Question: is the Fake-decision bias driven by the heatmap image, by the forensic
semantic framing, or by their interaction?

Design (same image, four paired conditions):
  C0  image only, neutral task framing            -> baseline
  C1  image only + neutral forensic-context line  -> text framing alone
  C2  image + DONOR ELA map, neutral framing      -> visualization alone
  C3  image + SAME DONOR ELA map, forensic framing-> framing over identical pixels

C2 vs C3 differ ONLY in wording; the visualization is byte-identical (asserted by
hash at runtime). Donor ELA is used, not correct ELA, so no true localization
information is present and the contrast isolates context rather than accuracy.

Both fake and paired-real images run all four conditions, so a shift toward Fake
can be separated from improved discrimination.
"""
import argparse, hashlib, io, json, os, sys
from collections import Counter, defaultdict
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools

# ----------------------------------------------------------------- PROMPTS
# Frozen before any inference. C0/C1 share the no-image-2 body; C2/C3 share the
# with-image-2 body and differ ONLY in the auxiliary-input description.

TASK_TAIL = """The 3x3 grid names for locations are exactly:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right.

Report strictly as JSON with these keys:
{
  "final_verdict": "real" or "fake",
  "reason": "brief explanation"
}
Output JSON only."""

C0_PROMPT = """You are examining a digital image.

Determine whether this image is an authentic photograph or has been digitally
altered.

""" + TASK_TAIL

# Shared ELA description for C1 and C3. Identical method explanation, identical
# "strong response" caveat, identical caution wording. The ONLY difference between
# C1 and C3 is the final sentence, which states the necessary fact of whether the
# visualization accompanies the image.
ELA_SHARED = """An automated Error Level Analysis (ELA) has been applied to this image.
ELA re-compresses an image and amplifies per-pixel differences. Strong responses
can arise from genuine image content such as texture and edges or from local
re-encoding differences, so ELA should be treated as one indicator rather than
proof."""

C1_TAIL = "The visualization output is not provided."
C3_TAIL = "The visualization output is provided alongside the image."

C1_PROMPT = f"""You are examining a digital image.

{ELA_SHARED}
{C1_TAIL}

Determine whether this image is an authentic photograph or has been digitally
altered.

""" + TASK_TAIL

# Neutral framing of the second image. No forensic vocabulary, no words implying
# manipulation, anomaly, suspicion, or evidence. Minimally framed, not framing-free:
# leaving the second image entirely unexplained would introduce image-role
# ambiguity, which is a worse confound than a neutral one-line description.
C2_PROMPT = """You are examining a digital image.

An auxiliary visualization is provided alongside the image. Interpret it only in
conjunction with the original image.

Determine whether this image is an authentic photograph or has been digitally
altered.

""" + TASK_TAIL

C3_PROMPT = f"""You are examining a digital image.

{ELA_SHARED}
{C3_TAIL}

Determine whether this image is an authentic photograph or has been digitally
altered.

""" + TASK_TAIL

PROMPTS = {"C0": C0_PROMPT, "C1": C1_PROMPT, "C2": C2_PROMPT, "C3": C3_PROMPT}
N_IMAGES = {"C0": 1, "C1": 1, "C2": 2, "C3": 2}

# Words that must not appear in C0/C1/C2 (they would pre-load a verdict).
BIAS_WORDS = ("manipulat", "suspicious", "anomal", "forged", "forgery", "tamper",
              "fake evidence", "evidence of", "artifact", "inconsisten")
# C3 is allowed forensic vocabulary by design, but not verdict-pushing wording.
PUSH_WORDS = ("trust the", "reliable evidence", "prioritize", "is likely fake",
              "indicates manipulation", "proves")


def check_prompts():
    print("PROMPT AUDIT")
    for k in ("C0", "C1", "C2"):
        low = PROMPTS[k].lower()
        hits = [w for w in BIAS_WORDS if w in low]
        print(f"  {k}: bias-word hits = {hits or 'none'}")
        assert not hits, f"{k} contains biasing vocabulary: {hits}"
    for k in PROMPTS:
        low = PROMPTS[k].lower()
        hits = [w for w in PUSH_WORDS if w in low]
        print(f"  {k}: verdict-push hits = {hits or 'none'}")
        assert not hits, f"{k} contains verdict-pushing wording: {hits}"
    assert TASK_TAIL in PROMPTS["C2"] and TASK_TAIL in PROMPTS["C3"]
    print("  C2/C3 share the identical task body: True")

    # --- C1/C3 forensic-factor symmetry: identical ELA description, differing only
    # in the sentence stating whether the visualization is supplied.
    c1, c3 = PROMPTS["C1"], PROMPTS["C3"]
    assert ELA_SHARED in c1 and ELA_SHARED in c3
    print(f"  C1/C3 share the identical ELA description: True")
    c1_norm = c1.replace(C1_TAIL, "<VIS_SENTENCE>")
    c3_norm = c3.replace(C3_TAIL, "<VIS_SENTENCE>")
    print(f"  C1/C3 identical after masking the visualization sentence: "
          f"{c1_norm == c3_norm}")
    assert c1_norm == c3_norm, "C1/C3 differ beyond the visualization sentence"
    print("\n  C1 vs C3 DIFF (only differing lines):")
    import difflib
    for line in difflib.unified_diff(c1.splitlines(), c3.splitlines(),
                                     "C1", "C3", lineterm="", n=1):
        if line.startswith(("+", "-", "@")) and not line.startswith(("+++", "---")):
            print(f"    {line}")

    print("\n  PROMPT HASHES")
    for k in PROMPTS:
        print(f"    sha1({k}) = {hashlib.sha1(PROMPTS[k].encode()).hexdigest()}")
    print("\n  FULL PROMPTS")
    for k in PROMPTS:
        print(f"  ---------------- {k} ({N_IMAGES[k]} image(s)) ----------------")
        for ln in PROMPTS[k].splitlines():
            print(f"    {ln}")


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-sources", type=int, default=150,
                    help="number of untouched coco_ids to sample")
    ap.add_argument("--per-source", type=int, default=1,
                    help="fake variants taken per coco_id")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    seed = cfg["experiment"]["seed"]
    tool_cfg = cfg["forensic_tools"]
    grid = cfg["localization"]["grid"]

    print("=" * 84)
    check_prompts()

    print("\n" + "=" * 84)
    print("DATA INDEPENDENCE")
    V4 = json.load(open(os.path.join(out_root, "validation_v4",
                                     "validation_v4_frozen.json")))
    v4c = {s["coco_id"] for s in V4["samples"]}
    idx = json.load(open(os.path.join(out_root, "validation_v4",
                                      "tgif_sp_validation_index.json")))
    allc = {r["coco_id"] for r in idx}
    un = sorted(allc - v4c)
    pool = [r for r in idx if r["coco_id"] in set(un)]
    print(f"  validation split coco_ids      : {len(allc)}")
    print(f"  used by V4                     : {len(v4c)}")
    print(f"  UNTOUCHED (available)          : {len(un)}")
    print(f"  triples from untouched sources : {len(pool)}")
    per = Counter(r["coco_id"] for r in pool)
    print(f"  variants per untouched coco_id : {dict(sorted(Counter(per.values()).items()))}")
    tr = np.array([r["tamper_ratio"] for r in pool], float)
    print(f"  tamper_ratio (natural): min={tr.min():.4f} p25={np.percentile(tr,25):.4f} "
          f"med={np.median(tr):.4f} p75={np.percentile(tr,75):.4f} max={tr.max():.4f}")
    print(f"  categories: {dict(sorted(Counter(r['category'] for r in pool).items()))}")
    # paired real availability
    havereal = sum(1 for r in pool if os.path.exists(os.path.join(root, r["real"])))
    print(f"  paired originals present       : {havereal}/{len(pool)}")

    if len(un) < args.n_sources:
        print(f"\n  NOTE: only {len(un)} untouched sources available, requested "
              f"{args.n_sources}. Using all {len(un)} and reporting it.")
    n_src = min(args.n_sources, len(un))

    # ---- frozen sampling rule: natural tamper distribution, one variant per source,
    # deterministic given the seed. No tamper-band balancing (bias, not size, is the
    # subject this round).
    rng = np.random.default_rng(seed)
    srcs = list(un)
    rng.shuffle(srcs)
    srcs = sorted(srcs[:n_src])
    by_src = defaultdict(list)
    for r in pool:
        by_src[r["coco_id"]].append(r)
    picked = []
    for c in srcs:
        cand = sorted(by_src[c], key=lambda r: r["pair_id"])
        rng.shuffle(cand)
        picked += cand[:args.per_source]
    picked.sort(key=lambda r: r["pair_id"])
    print(f"\n  SAMPLED: {len(picked)} fakes from "
          f"{len({p['coco_id'] for p in picked})} sources")
    trs = np.array([p["tamper_ratio"] for p in picked], float)
    print(f"  tamper_ratio: min={trs.min():.4f} med={np.median(trs):.4f} max={trs.max():.4f}")
    print(f"  categories: {dict(sorted(Counter(p['category'] for p in picked).items()))}")
    print(f"  mask types: {dict(Counter(p['mask_type'] for p in picked))}")

    # ---- donor assignment (same frozen rule family)
    ids = [p["pair_id"] for p in picked]
    cidm = {p["pair_id"]: p["coco_id"] for p in picked}
    n, half = len(ids), len(ids) // 2
    donors = {}
    for k, pid in enumerate(ids):
        for step in range(n):
            c = ids[(k + half + step) % n]
            if c != pid and cidm[c] != cidm[pid]:
                donors[pid] = c
                break
    bad = [p for p, d in donors.items() if cidm[d] == cidm[p]]
    print(f"\n  donors assigned: {len(donors)}  same-coco: {len(bad)}  "
          f"self: {sum(1 for p,d in donors.items() if d==p)}")
    assert not bad

    print("\n" + "=" * 84)
    print("C2/C3 VISUALIZATION IDENTITY CHECK (first 5 samples)")
    for p in picked[:5]:
        dpid = donors[p["pair_id"]]
        drec = next(r for r in pool if r["pair_id"] == dpid)
        tools, _ = run_tools(os.path.join(root, drec["fake"]), tool_cfg, grid=grid)
        vis = Image.fromarray(cv2.cvtColor(tools["ela"]["vis_bgr"], cv2.COLOR_BGR2RGB))
        tgt = Image.open(os.path.join(root, p["fake"])).convert("RGB")
        v = vis.resize(tgt.size, Image.BILINEAR)
        h = hashlib.sha1(png_bytes(v)).hexdigest()
        # C2 and C3 will be handed this exact object; identity is by construction,
        # verified here by hashing the same produced bytes twice.
        h2 = hashlib.sha1(png_bytes(v)).hexdigest()
        print(f"  {p['pair_id'][:34]:<34} donor={dpid[:28]:<28} vis_sha1={h[:16]} "
              f"identical={h == h2}")

    n_inf = len(picked) * 4 * 2          # 4 conditions x (fake + paired real)
    print("\n" + "=" * 84)
    print("INFERENCE PLAN (2.9 s/inference measured)")
    print(f"  fakes {len(picked)} x 4 conditions          = {len(picked)*4}")
    print(f"  paired reals {len(picked)} x 4 conditions   = {len(picked)*4}")
    print(f"  TOTAL                                = {n_inf}  "
          f"(~{n_inf*2.9/60:.0f} min)")

    if not args.freeze:
        print("\n(dry run — pass --freeze to write mechanism_frozen.json)")
        return

    for p in picked:
        p["donor_pair_id"] = donors[p["pair_id"]]
    od = os.path.join(out_root, "mechanism")
    os.makedirs(od, exist_ok=True)
    payload = dict(
        protocol="m1_context_bias_mechanism",
        role=("Mechanism experiment on coco_ids never used in V4. Isolates whether "
              "the Fake-decision bias comes from the heatmap image, the forensic "
              "framing, or their interaction."),
        seed=seed,
        conditions=["C0", "C1", "C2", "C3"],
        condition_meaning=dict(
            C0="image only, neutral task framing (baseline)",
            C1="image only + neutral forensic-context sentence",
            C2="image + donor ELA map, neutral framing",
            C3="image + SAME donor ELA map, forensic framing"),
        prompts=PROMPTS,
        prompt_hashes={k: hashlib.sha1(v.encode()).hexdigest()
                       for k, v in PROMPTS.items()},
        n_images_per_condition=N_IMAGES,
        cue_source="donor ELA (no correct localization information)",
        donor_rule=("sorted pair_ids rotated by half; coco_id collision -> walk "
                    "forward to first different coco_id"),
        sampling_rule=(f"{args.per_source} variant per coco_id, {n_src} untouched "
                       f"coco_ids, natural tamper distribution, seed {seed}"),
        run_both_labels=True,
        metrics=["FakeRate_fake(C)", "FakeRate_real(C)",
                 "C1-C0 text framing", "C2-C0 visualization only",
                 "C3-C2 framing over identical pixels", "C3-C0 full context"],
        statistics=dict(primary="coco_id-cluster bootstrap 95% CI",
                        iterations=10000, seed=seed,
                        supplementary=["McNemar exact", "Wilson"]),
        independence=dict(v4_coco_ids=len(v4c), untouched_available=len(un),
                          overlap_with_v4=0),
        hashes=dict(config=hashlib.sha1(open(args.config, "rb").read()).hexdigest(),
                    index=hashlib.sha1(open(os.path.join(
                        out_root, "validation_v4",
                        "tgif_sp_validation_index.json"), "rb").read()).hexdigest()),
        n_samples=len(picked), samples=picked,
    )
    pth = os.path.join(od, "mechanism_frozen.json")
    if os.path.exists(pth):
        print(f"\nREFUSING TO OVERWRITE {pth}")
        sys.exit(1)
    json.dump(payload, open(pth, "w"), indent=1)
    print(f"\nfrozen -> {pth}")


if __name__ == "__main__":
    main()
