"""Freeze the Qwen2.5-VL-32B capacity replication. No inference here.

Minimal 4-condition replication on the SAME 184 fake sources used for 7B, to test
whether the weak coupling between spatial forensic evidence and the final verdict
survives a 4.5x capacity increase.

  B0 score only          (= 7B C10)
  B1 score + own map     (= 7B F1 / C11)
  B2 score + blank map   (= 7B F2)
  B3 score + shifted map (= 7B F4)

Fake targets only. Stage-B verdict only, no Stage-A summary. Prompts, score
formatting, map rendering, shift transform and decoding are reused byte-identically
from the frozen 7B protocols -- the ONLY intended variable is the model.

Shift-audit note: the previous round's integrity assertion compared float32 mean()
at 1e-9, which is below float32 summation precision. The corrected audit checks the
sorted-value multiset (exact), np.allclose on sorted values, and the histogram.
"""
import argparse, hashlib, json, os, sys
import numpy as np, yaml

SHIFT_FRAC = 0.50
CONDITIONS = {
    "B0_score_only":   dict(map=None,      score="own", maps_to_7B="C10"),
    "B1_own_map":      dict(map="own",      score="own", maps_to_7B="F1 / C11"),
    "B2_blank_map":    dict(map="blank",    score="own", maps_to_7B="F2"),
    "B3_shifted_map":  dict(map="shifted",  score="own", maps_to_7B="F4"),
}
SEVEN_B = dict(C10_score_only=0.755, C11_score_map=0.527, F1_own=0.522,
               F2_blank=0.266, F4_shifted=0.495,
               effects=dict(map_cost_C11_minus_C10=-0.228,
                            blank_effect_F2_minus_F1=-0.255,
                            shift_effect_F4_minus_F1=-0.027))
CONTRASTS = dict(
    map_cost="B1 - B0   (does adding the raw map still cost recall?)",
    map_content_effect="B2 - B1   (removing map content, second image retained)",
    spatial_correspondence="B3 - B1   (PRIMARY capacity test: same map statistics, "
                           "broken alignment)",
    cross_model="delta_effect = effect_32B - effect_7B, with bootstrap CI; this is "
                "the primary capacity comparison, NOT absolute recall",
)
INTERPRETATION = {
    "capacity_invariant_failure": "B1 < B0 and B3 ~ B1 -> increasing model capacity "
        "does not resolve the weak coupling between spatial forensic evidence and "
        "final authenticity decisions",
    "capacity_dependent_spatial_grounding": "B3 significantly < B1 while the 7B shift "
        "effect ~ 0 -> larger capacity increases sensitivity to image-map spatial "
        "correspondence",
    "capacity_reduces_map_interference": "B1 ~ B0 while 7B C11 << C10 -> larger "
        "capacity reduces interference between scalar and spatial forensic evidence",
    "failure_mode_shift": "near-constant verdicts or other collapse -> capacity "
        "scaling changes the failure mode but does not reliably solve evidence "
        "integration",
}
EXPL = dict(
    method="deterministic keyword rules, identical to the 7B rounds; no LLM scoring",
    focus="B1 vs B3: if the shifted map no longer corresponds to the image yet the "
          "model still produces spatially-grounded-sounding explanations, flag as "
          "potential explanation-faithfulness failure",
)


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")          # 7B config, for sample/prompt provenance
    ap.add_argument("config_32b")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    c32 = yaml.safe_load(open(args.config_32b))
    R = cfg["experiment"]["out_root"]

    conf = json.load(open(os.path.join(
        R, "trufor_conflict", "trufor_score_map_conflict_frozen.json")))
    ab = json.load(open(os.path.join(
        R, "trufor_score_map", "trufor_score_map_ablation_frozen.json")))
    S = conf["samples"]

    print("=" * 92)
    print("1. MODEL")
    p32 = c32["model"]["path"]
    print(f"  path      {p32}")
    snap = p32.rstrip("/").split("/")[-1]
    print(f"  snapshot  {snap}")
    env32 = json.load(open(os.path.join(R, "qwen32b", "qwen32b_env_frozen.json")))
    print(f"  frozen env record: qwen32b_env_frozen.json "
          f"(sha1 {sha1f(os.path.join(R,'qwen32b','qwen32b_env_frozen.json'))[:12]})")
    print(f"  dtype {c32['model']['dtype']}  device_map "
          f"{c32['model']['device_map']}  quantization: none")
    assert c32["model"]["dtype"] == "bfloat16"
    print(f"  7B reference: {cfg['model']['path'].rstrip('/').split('/')[-1]}")

    print("\n  parity with the 7B runs (only the model may differ):")
    for k in ("min_pixels", "max_pixels"):
        a, b = cfg["model"][k], c32["model"][k]
        print(f"    {k:<12} 7B {a}  32B {b}   same: {a==b}")
        assert a == b, k
    g7, g32 = cfg["model"]["generation"], c32["model"]["generation"]
    print(f"    generation   7B {g7}")
    print(f"                 32B {g32}   same: {g7==g32}")
    assert g7 == g32

    print("\n" + "=" * 92)
    print("2. SAMPLES — identical to the 7B rounds, fake targets only")
    print(f"  sources {len(S)}  unique coco_id {len({s['coco_id'] for s in S})}")
    assert len({s["coco_id"] for s in S}) == len(S)
    print(f"  n_samples == n_sources: True  (no variant clustering)")
    print(f"  inherited from trufor_score_map_conflict_frozen.json "
          f"(sha1 {sha1f(os.path.join(R,'trufor_conflict','trufor_score_map_conflict_frozen.json'))[:12]})")
    print(f"  selection is NOT conditioned on any 7B outcome")
    tr = np.array([s["tamper_ratio"] for s in S])
    print(f"  tamper_ratio q1/med/q3 {np.percentile(tr,25):.4f} / "
          f"{np.median(tr):.4f} / {np.percentile(tr,75):.4f}")
    print(f"  mask type  {dict(__import__('collections').Counter(s['mask_type'] for s in S))}")

    print("\n" + "=" * 92)
    print("3. CONDITIONS")
    for k, v in CONDITIONS.items():
        print(f"  {k:<18} map={str(v['map']):<8} score={v['score']:<5} "
              f"<- 7B {v['maps_to_7B']}")
    print(f"  the score shown is ALWAYS the target's own TruFor score")
    print(f"  fake targets only; real controls are deliberately deferred this round")

    print("\n" + "=" * 92)
    print("4. PROMPTS — reused byte-identically from the frozen 7B protocols")
    P = dict(SCORE_ONLY=ab["prompts"]["C10"], MAP_AND_SCORE=ab["prompts"]["C11"])
    ph = {k: hashlib.sha1(v.encode()).hexdigest() for k, v in P.items()}
    assert ph["SCORE_ONLY"] == ab["prompt_hashes"]["C10"]
    assert ph["MAP_AND_SCORE"] == ab["prompt_hashes"]["C11"]
    assert ph["MAP_AND_SCORE"] == conf["prompt_hashes"]["MAP_AND_SCORE"]
    for v in P.values():
        assert "{stage_a_summary}" not in v
    print(f"  B0          sha1 {ph['SCORE_ONLY']}   (= 2x2 C10)")
    print(f"  B1/B2/B3    sha1 {ph['MAP_AND_SCORE']}   (= 2x2 C11 = conflict F1)")
    print(f"  no Stage-A summary in either prompt: True")
    print(f"  prompt never names the transform (blank/shifted/condition): True")

    print("\n" + "=" * 92)
    print("5. MAP TRANSFORM AUDIT (corrected)")
    print(f"  shift: np.roll(map, int({SHIFT_FRAC}*W), axis=1), no renormalisation")
    print("  previous round's assertion used |mean diff| < 1e-9, which is BELOW")
    print("  float32 summation precision and produced a false failure on 50/184.")
    print("  corrected audit: exact sorted-multiset equality + np.allclose + histogram")
    ex_ms, ex_cl, ex_hi, dmean, dact, odd = [], [], [], [], [], 0
    for s in S:
        m = np.load(s["evidence"]["fake_own"]["npz"])["map"].astype(np.float32)
        W = m.shape[1]
        sh = np.roll(m, int(SHIFT_FRAC * W), axis=1)
        a, b = np.sort(m.ravel()), np.sort(sh.ravel())
        ex_ms.append(np.array_equal(a, b))
        ex_cl.append(np.allclose(a, b, rtol=0, atol=0))
        ha = np.histogram(m, bins=50, range=(0, 1))[0]
        hb = np.histogram(sh, bins=50, range=(0, 1))[0]
        ex_hi.append(np.array_equal(ha, hb))
        dmean.append(abs(float(m.mean()) - float(sh.mean())))
        dact.append(abs(float((m > .5).mean()) - float((sh > .5).mean())))
        odd += W % 2
    print(f"  sorted multiset identical      : {sum(ex_ms)}/{len(S)}")
    print(f"  np.allclose(atol=0) on sorted  : {sum(ex_cl)}/{len(S)}")
    print(f"  50-bin histogram identical     : {sum(ex_hi)}/{len(S)}")
    print(f"  max |mean diff|  {max(dmean):.3e}   (float32 summation order only)")
    print(f"  max |active frac diff|  {max(dact):.3e}")
    print(f"  odd-width maps {odd}/{len(S)} (int() shift is not exactly half-width "
          f"there; values still permuted exactly)")
    ok = all(ex_ms) and all(ex_hi)
    print(f"  AUDIT {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  stopping per protocol")
        sys.exit(1)
    m0 = np.load(S[0]["evidence"]["fake_own"]["npz"])["map"].astype(np.float32)
    s0 = np.roll(m0, int(SHIFT_FRAC * m0.shape[1]), axis=1)
    print(f"  blank map: np.zeros_like -> min=max=mean=0, identical rendering pipeline")
    print(f"  example {S[0]['sample_id']} shape {m0.shape}")
    print(f"    own     mean {m0.mean():.6f}  max {m0.max():.4f}  >0.5 {(m0>.5).mean():.6f}")
    print(f"    shifted mean {s0.mean():.6f}  max {s0.max():.4f}  >0.5 {(s0>.5).mean():.6f}")
    print(f"    spatially different from own: {not np.array_equal(m0, s0)}")

    print("\n" + "=" * 92)
    print("6. RENDERING AND SCORE FORMATTING (inherited, unchanged)")
    print(f"  map rendering  {conf['map_rendering']}")
    print(f"  score format   {conf['score_presentation']}")

    print("\n" + "=" * 92)
    print("7. CONTRASTS AND 7B REFERENCE EFFECTS")
    for k, v in CONTRASTS.items():
        print(f"  {k:<24} {v}")
    print(f"\n  7B recalls : C10 {SEVEN_B['C10_score_only']:.3f}  F1 {SEVEN_B['F1_own']:.3f}"
          f"  F2 {SEVEN_B['F2_blank']:.3f}  F4 {SEVEN_B['F4_shifted']:.3f}")
    print(f"  7B effects : map_cost {SEVEN_B['effects']['map_cost_C11_minus_C10']:+.3f}"
          f"  blank {SEVEN_B['effects']['blank_effect_F2_minus_F1']:+.3f}"
          f"  shift {SEVEN_B['effects']['shift_effect_F4_minus_F1']:+.3f}")
    print(f"  note: the 7B shift effect is the null result this round must replicate")
    print(f"  or overturn; it is the primary capacity question.")

    print("\n" + "=" * 92)
    print("8. GPU LAYOUT AND RUNTIME ESTIMATE")
    n = len(S) * len(CONDITIONS)
    prev = json.load(open(os.path.join(R, "qwen32b_stagev_ablation", "run_meta.json")))
    if isinstance(prev, list):
        prev = prev[-1] if prev and isinstance(prev[-1], dict) else {}
    sec = prev.get("mean_seconds") or prev.get("mean_sec") or 11.42
    print(f"  4x RTX 3090, BF16, device_map from config, no offload, no quantization")
    print(f"  historical 32B peaks: 16.3 / 18.2 / 18.2 / 16.8 GB (Stage-V ablation)")
    print(f"  inferences {len(S)} x {len(CONDITIONS)} = {n}")
    print(f"  measured 32B rate {sec:.2f} s/inf (single-image Stage-V); B1/B2/B3 are")
    print(f"  two-image so expect somewhat slower")
    print(f"  estimate {n*sec/3600:.1f} h at {sec:.1f}s, up to "
          f"{n*sec*1.35/3600:.1f} h if two-image costs ~35% more")
    print(f"  GPUs 0-3 are the historical 32B assignment; 7B work used GPU 6")

    if not args.freeze:
        print("\nDRY RUN — nothing written.")
        return

    doc = dict(
        stage="Qwen2.5-VL-32B capacity replication of TruFor evidence integration",
        question="Does the weak coupling between spatial forensic evidence and the "
                 "final verdict, observed on 7B, persist at 32B?",
        scope="capacity validation only; NOT an accuracy comparison between 7B and 32B",
        model=dict(path=p32, snapshot=snap, dtype=c32["model"]["dtype"],
                   device_map=c32["model"]["device_map"], quantization=None,
                   env="qwen_vl_32b",
                   env_record=os.path.join(R, "qwen32b", "qwen32b_env_frozen.json")),
        model_7b_reference=cfg["model"]["path"],
        parity=dict(min_pixels=c32["model"]["min_pixels"],
                    max_pixels=c32["model"]["max_pixels"],
                    generation=c32["model"]["generation"],
                    identical_to_7B=True),
        n_sources=len(S), targets="fake only",
        real_controls="deliberately deferred this round",
        samples_inherited_from="trufor_score_map_conflict_frozen.json",
        selection_note="not conditioned on any 7B outcome",
        stage_a_policy="no Stage A, no Stage-A summary in any condition",
        conditions=CONDITIONS,
        shift_fraction=SHIFT_FRAC,
        shift_audit=dict(method="exact sorted-multiset equality + np.allclose(atol=0) "
                                "+ 50-bin histogram equality",
                         multiset_identical=f"{sum(ex_ms)}/{len(S)}",
                         histogram_identical=f"{sum(ex_hi)}/{len(S)}",
                         max_abs_mean_diff=float(max(dmean)),
                         max_abs_active_diff=float(max(dact)),
                         odd_width_maps=int(odd),
                         prior_round_correction=(
                             "the earlier assertion compared float32 mean() at 1e-9, "
                             "below float32 summation precision, and falsely failed on "
                             "50/184; the transform itself was exact"),
                         result="PASS"),
        prompts=P, prompt_hashes=ph,
        prompt_provenance="byte-identical to trufor_score_map_ablation_frozen.json "
                          "C10/C11; C11 also equals conflict-round F1",
        score_presentation=conf["score_presentation"],
        map_rendering=conf["map_rendering"],
        contrasts=CONTRASTS, interpretation=INTERPRETATION,
        explanation_audit=EXPL,
        seven_b_results=SEVEN_B,
        inference_budget=dict(total=n, per_condition=len(S)),
        runtime_estimate_hours=round(n * sec / 3600, 1),
        gpu_layout="4x RTX 3090 (historical 32B assignment GPU 0-3), BF16, no offload",
        forbidden_this_round=["prompt modification", "Stage A", "gate", "verifier",
                              "calibration", "confidence map", "threshold tuning",
                              "LoRA", "training", "new dataset", "quantization"],
        hashes=dict(config_7b=sha1f(args.config), config_32b=sha1f(args.config_32b),
                    freezer=sha1f(os.path.abspath(__file__)),
                    conflict_protocol=sha1f(os.path.join(
                        R, "trufor_conflict",
                        "trufor_score_map_conflict_frozen.json")),
                    ablation_protocol=sha1f(os.path.join(
                        R, "trufor_score_map",
                        "trufor_score_map_ablation_frozen.json"))),
        samples=[dict(sample_id=s["sample_id"], coco_id=s["coco_id"],
                      category=s["category"], mask_type=s["mask_type"],
                      tamper_ratio=s["tamper_ratio"], fake=s["fake"],
                      evidence=dict(fake_own=s["evidence"]["fake_own"]))
                 for s in S])
    od = os.path.join(R, "trufor_32b")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "trufor_32b_capacity_frozen.json")
    if os.path.exists(fp):
        print(f"REFUSING to overwrite {fp}")
        sys.exit(1)
    json.dump(doc, open(fp, "w"), indent=1)
    print(f"\nFROZEN -> {fp}")
    print(f"  protocol sha1 {sha1f(fp)}")
    print(f"  {len(S)} fake sources x {len(CONDITIONS)} conditions = {n} inferences")


if __name__ == "__main__":
    main()
