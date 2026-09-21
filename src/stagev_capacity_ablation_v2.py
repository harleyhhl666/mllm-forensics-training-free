"""Stage-V capacity ablation v2: four conditions, label-aware visualization cache.

Supersedes the Stage-V portion of mitigation_run.py, which is left untouched as the
execution record of the frozen 7B M1 run. That script cached ELA visualizations
under the cue name alone, so the real-image condition silently received the ELA of
the FAKE counterpart. Here the cache key is (pair_id, label, cue, source) and the
source image for each condition is stated explicitly.

Conditions (100 sources each):
  A  fake + correct ELA                 ELA(fake)   -> paired with 7B history
  B  fake + donor ELA                   ELA(donor)  -> paired with 7B history
  C_legacy  real + fake-counterpart ELA ELA(fake)   -> reproduces the historical input
  C_own     real + its own ELA          ELA(real)   -> semantically correct control

Runs on either model; the model path comes from the config, nothing else differs.
"""
import argparse, hashlib, io, json, os, sys, time
import numpy as np, yaml, cv2, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools
from mitigation_run import parse_stage_v, assert_clean

# condition -> (image to analyse, image whose ELA is shown)
CONDITIONS = {
    "A_fake_correct":  ("fake", "fake"),
    "B_fake_donor":    ("fake", "donor"),
    "C_legacy_real_fakecounterpart": ("real", "fake"),
    "C_own_real_own":  ("real", "real"),
}


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def sha_img(img):
    return hashlib.sha1(png_bytes(img)).hexdigest()


class ElaCache:
    """Cache keyed by the SOURCE FILE of the ELA, never by cue name alone."""

    def __init__(self, root, tool_cfg, grid):
        self.root, self.tc, self.grid = root, tool_cfg, grid
        self._c = {}

    def raw(self, relpath):
        if relpath not in self._c:
            t, _ = run_tools(os.path.join(self.root, relpath), self.tc, grid=self.grid)
            self._c[relpath] = Image.fromarray(
                cv2.cvtColor(t["ela"]["vis_bgr"], cv2.COLOR_BGR2RGB))
        return self._c[relpath]

    def sized(self, relpath, size):
        return self.raw(relpath).resize(size, Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conditions", default="all",
                    help="comma list of condition keys, or 'all'")
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    ds = cfg["datasets"]["tgif_sp"]
    root, grid, tool_cfg = ds["root"], cfg["localization"]["grid"], cfg["forensic_tools"]
    conds = (list(CONDITIONS) if args.conditions == "all"
             else [c.strip() for c in args.conditions.split(",")])
    for c in conds:
        assert c in CONDITIONS, f"unknown condition {c}"

    F = json.load(open(os.path.join(out_root, "mitigation", "mitigation_frozen.json")))
    STAGE_V = F["prompts"]["STAGE_V"]
    assert_clean(STAGE_V)
    ph = hashlib.sha1(STAGE_V.encode()).hexdigest()
    assert ph == F["prompt_hashes"]["STAGE_V"], "Stage-V prompt drifted"
    S = F["samples"][:args.n] if args.n else F["samples"]
    byid = {s["pair_id"]: s for s in F["samples"]}
    cidm = {s["pair_id"]: s["coco_id"] for s in F["samples"]}
    assert all(cidm[s["donor_pair_id"]] != s["coco_id"] for s in F["samples"])

    print(f"model_label      : {args.model_label}")
    print(f"model_path       : {cfg['model']['path']}")
    print(f"Stage-V sha1     : {ph} (matches frozen protocol)")
    print(f"conditions       : {conds}")
    print(f"samples          : {len(S)}  coco_ids {len({s['coco_id'] for s in S})}")
    print(f"decoding         : {cfg['model']['generation']}")
    print(f"pixels           : {cfg['model']['min_pixels']}/{cfg['model']['max_pixels']}")

    ela = ElaCache(root, tool_cfg, grid)

    # ---------------- input hash audit, BEFORE any inference
    hist = {}
    hp = os.path.join(out_root, "mitigation_run", "stage_v.jsonl")
    if os.path.exists(hp):
        for l in open(hp):
            r = json.loads(l)
            hist[(r["pair_id"], r["label"], r["cue"])] = r["visualization_sha1"]
    print("\n" + "=" * 86)
    print("INPUT HASH AUDIT (first 5 samples)")
    ok_legacy = ok_distinct = n_chk = 0
    for s in S[:5]:
        pid = s["pair_id"]
        dn = byid[s["donor_pair_id"]]
        real = Image.open(os.path.join(root, s["real"])).convert("RGB")
        fake = Image.open(os.path.join(root, s["fake"])).convert("RGB")
        h_fake_on_real = sha_img(ela.sized(s["fake"], real.size))
        h_real_on_real = sha_img(ela.sized(s["real"], real.size))
        h_fake_on_fake = sha_img(ela.sized(s["fake"], fake.size))
        h_donor_on_fake = sha_img(ela.sized(dn["fake"], fake.size))
        hl = hist.get((pid, "real", "correct"))
        hA = hist.get((pid, "fake", "correct"))
        hB = hist.get((pid, "fake", "donor"))
        print(f"  {pid[:26]:<26}")
        print(f"    A  ELA(fake)  on fake : {h_fake_on_fake[:14]}  7B={str(hA)[:14]}"
              f"  MATCH={h_fake_on_fake == hA}")
        print(f"    B  ELA(donor) on fake : {h_donor_on_fake[:14]}  7B={str(hB)[:14]}"
              f"  MATCH={h_donor_on_fake == hB}")
        print(f"    C_legacy ELA(fake) on real : {h_fake_on_real[:14]}  "
              f"7B={str(hl)[:14]}  MATCH={h_fake_on_real == hl}")
        print(f"    C_own    ELA(real) on real : {h_real_on_real[:14]}  "
              f"DISTINCT from legacy={h_real_on_real != h_fake_on_real}")
        n_chk += 1
        ok_legacy += int(h_fake_on_real == hl)
        ok_distinct += int(h_real_on_real != h_fake_on_real)
    print(f"\n  legacy reproduces 7B input : {ok_legacy}/{n_chk}")
    print(f"  C_own distinct from legacy : {ok_distinct}/{n_chk}")
    assert ok_legacy == n_chk, "C_legacy must reproduce the historical visualization"
    assert ok_distinct == n_chk, "C_own must differ from the fake-counterpart ELA"
    if args.audit_only:
        print("\nAUDIT ONLY — no inference run.")
        return

    out_dir = os.path.join(out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage_v.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                d = json.loads(l)
                done.add((d["pair_id"], d["condition"]))
            except Exception:
                pass
    print(f"\nresume: {len(done)} rows on disk")

    if args.multi_gpu:
        from mllm_probe_mgpu import QwenVLProbeMultiGPU, gpu_mem, nvidia_used
        probe = QwenVLProbeMultiGPU(cfg)
        pr = probe.placement_report()
        assert not pr["has_offload"], "offload not acceptable"
        print(f"loaded {probe.load_time_s:.1f}s devices={pr['devices']} offload=NONE")
    else:
        from mllm_probe import QwenVLProbe
        probe = QwenVLProbe(cfg)
        from mllm_probe_mgpu import gpu_mem, nvidia_used
        pr = dict(single_gpu=cfg["model"]["device_map"])
        print(f"loaded single-device: {cfg['model']['device_map']}")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.set_device(0)

    fh = open(path, "a", buffering=1)
    t0, n, times = time.time(), 0, []
    for k, s in enumerate(S):
        pid = s["pair_id"]
        dn = byid[s["donor_pair_id"]]
        for cond in conds:
            if (pid, cond) in done:
                continue
            img_key, ela_key = CONDITIONS[cond]
            base = Image.open(os.path.join(root, s[img_key])).convert("RGB")
            src = dn["fake"] if ela_key == "donor" else s[ela_key]
            vis = ela.sized(src, base.size)
            ts = time.time()
            raw, info = probe.ask(STAGE_V, [base, vis])
            dt = time.time() - ts
            times.append(dt)
            n += 1
            st, fields, notes = parse_stage_v(raw)
            fh.write(json.dumps(dict(
                pair_id=pid, coco_id=s["coco_id"], category=s["category"],
                mask_type=s["mask_type"], tamper_ratio=s["tamper_ratio"],
                condition=cond, analysed_image=img_key, ela_source=ela_key,
                ela_source_path=src, model=args.model_label,
                state=st, **fields, notes=notes, raw=raw,
                visualization_sha1=sha_img(vis), prompt_sha1=ph,
                donor_pair_id=s["donor_pair_id"], seconds=round(dt, 2),
                info=info), ensure_ascii=False) + "\n")
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(S)}  {n} inf  {(time.time()-t0)/60:.1f} min  "
                  f"mean {np.mean(times):.1f}s", flush=True)
    fh.close()
    peak = {i: gpu_mem()[i]["peak_GB"] for i in range(torch.cuda.device_count())}
    meta = dict(model_label=args.model_label, model_path=cfg["model"]["path"],
                conditions=conds, n_inferences=n,
                total_minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB_per_gpu=peak, nvidia_used_MiB=nvidia_used(),
                placement=pr, prompt_sha1=ph,
                decoding=cfg["model"]["generation"],
                min_pixels=cfg["model"]["min_pixels"],
                max_pixels=cfg["model"]["max_pixels"],
                runner_sha1=hashlib.sha1(
                    open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                config_sha1=hashlib.sha1(open(args.config, "rb").read()).hexdigest())
    mp = os.path.join(out_dir, "run_meta.json")
    old = json.load(open(mp)) if os.path.exists(mp) else []
    old = old if isinstance(old, list) else [old]
    old.append(meta)
    json.dump(old, open(mp, "w"), indent=1)
    print(f"done: {n} inferences in {meta['total_minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {peak}")


if __name__ == "__main__":
    main()
