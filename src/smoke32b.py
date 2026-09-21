"""Three-level smoke test for the 32B capacity ablation. Environment check only.

Level 1  text-only        -> load, tokenizer, generate, no OOM
Level 2  single image     -> vision encoder across shards, no device/dtype error
Level 3  one Stage-V pair -> frozen prompt, schema, timing, peak memory

Level 3 uses a DEVELOPMENT sample (already consumed by the 7B M1 run), never an
untouched source, and the Stage-V prompt is read from the frozen mitigation
protocol so it cannot drift from the 7B version.

This script does NOT run the 300-inference ablation and does not modify any
7B artifact.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml, cv2, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe_mgpu import QwenVLProbeMultiGPU, gpu_mem, nvidia_used
from mllm_probe import extract_json
from forensic_tools import run_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc = cfg["model"]
    rep = dict(config_path=os.path.abspath(args.config))

    print("=" * 86)
    print("0. PRE-LOAD STATE")
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"devices {torch.cuda.device_count()}")
    import transformers, accelerate
    print(f"  transformers {transformers.__version__}  accelerate {accelerate.__version__}")
    print(f"  nvidia used (MiB): {nvidia_used()}")
    print(f"  model path : {mc['path']}")
    print(f"  dtype      : {mc['dtype']}")
    print(f"  device_map : {mc['device_map']}")
    print(f"  max_memory : {mc.get('max_memory')}")
    print(f"  min_pixels {mc['min_pixels']}  max_pixels {mc['max_pixels']}")
    print(f"  generation : {mc['generation']}")
    rep["env"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                      transformers=transformers.__version__,
                      accelerate=accelerate.__version__,
                      n_gpus=torch.cuda.device_count(),
                      python=sys.version.split()[0])
    rep["model_cfg"] = {k: mc[k] for k in
                        ("path", "dtype", "device_map", "min_pixels",
                         "max_pixels", "generation")}
    rep["model_cfg"]["max_memory"] = mc.get("max_memory")

    print("\n" + "=" * 86)
    print("1. LOAD")
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            torch.cuda.reset_peak_memory_stats(i)
        except Exception as e:
            print(f"  (peak-stat reset skipped on cuda:{i}: {e})")
    torch.cuda.set_device(0)
    t0 = time.time()
    probe = QwenVLProbeMultiGPU(cfg)
    print(f"  load time: {probe.load_time_s:.1f}s")
    pr = probe.placement_report()
    print(f"  input device        : {pr['input_device']}")
    print(f"  devices used        : {pr['devices']}")
    print(f"  modules per device  : {pr['n_modules_per_device']}")
    print(f"  CPU/disk offload    : "
          f"{pr['cpu_or_disk_offload'] or 'NONE (good — pure GPU BF16)'}")
    print(f"  torch per-GPU after load: {gpu_mem()}")
    print(f"  nvidia-smi (MiB)        : {nvidia_used()}")
    rep["load"] = dict(load_time_s=round(probe.load_time_s, 1), placement=pr,
                       torch_mem=gpu_mem(), nvidia_used_MiB=nvidia_used())
    if pr["has_offload"]:
        print("  WARNING: offload detected — NOT acceptable as the final ablation "
              "config; report before proceeding.")

    # ---------------------------------------------------------------- Smoke 1
    print("\n" + "=" * 86)
    print("2. SMOKE 1 — text only ('hello')")
    t0 = time.time()
    txt, info = probe.ask("hello", [])
    dt = time.time() - t0
    print(f"  time {dt:.1f}s   prompt_tokens={info['n_prompt_tokens']}")
    print(f"  output: {txt[:300]!r}")
    print(f"  peak per-GPU: {gpu_mem()}")
    rep["smoke1"] = dict(seconds=round(dt, 1), output=txt[:500], info=info,
                         mem=gpu_mem())

    # ---------------------------------------------------------------- Smoke 2
    print("\n" + "=" * 86)
    print("3. SMOKE 2 — single image (synthetic, NOT an experiment sample)")
    rng = np.random.default_rng(0)
    arr = np.full((448, 640, 3), 40, np.uint8)
    arr[120:300, 180:420] = (200, 80, 60)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + rng.integers(0, 25, arr.shape[:2]), 0, 255)
    img = Image.fromarray(arr)
    t0 = time.time()
    txt2, info2 = probe.ask("Describe this image in one sentence.", [img])
    dt2 = time.time() - t0
    print(f"  time {dt2:.1f}s  visual_tokens={info2['n_visual_tokens']}  "
          f"prompt_tokens={info2['n_prompt_tokens']}")
    print(f"  output: {txt2[:300]!r}")
    print(f"  peak per-GPU: {gpu_mem()}")
    rep["smoke2"] = dict(seconds=round(dt2, 1), output=txt2[:500], info=info2,
                         mem=gpu_mem())

    # ---------------------------------------------------------------- Smoke 3
    print("\n" + "=" * 86)
    print("4. SMOKE 3 — one Stage-V dry sample (development sample, frozen prompt)")
    out_root = cfg["experiment"]["out_root"]
    F = json.load(open(os.path.join(out_root, "mitigation",
                                    "mitigation_frozen.json")))
    STAGE_V = F["prompts"]["STAGE_V"]
    h = hashlib.sha1(STAGE_V.encode()).hexdigest()
    same = h == F["prompt_hashes"]["STAGE_V"]
    print(f"  Stage-V prompt sha1 {h}")
    print(f"  matches frozen 7B prompt: {same}")
    assert same, "Stage-V prompt drifted from the frozen 7B version"
    s = F["samples"][0]          # already used by the 7B M1 run -> development
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    print(f"  development sample: {s['pair_id']} (coco {s['coco_id']}) "
          f"— already consumed by 7B M1, not an untouched source")
    base = Image.open(os.path.join(root, s["fake"])).convert("RGB")
    tools, _ = run_tools(os.path.join(root, s["fake"]),
                         cfg["forensic_tools"], grid=cfg["localization"]["grid"])
    vis = Image.fromarray(cv2.cvtColor(tools["ela"]["vis_bgr"], cv2.COLOR_BGR2RGB))
    vis = vis.resize(base.size, Image.BILINEAR)
    t0 = time.time()
    raw, info3 = probe.ask(STAGE_V, [base, vis])
    dt3 = time.time() - t0
    obj, _, ok = extract_json(raw)
    print(f"  time {dt3:.1f}s  visual_tokens={info3['n_visual_tokens']}  "
          f"prompt_tokens={info3['n_prompt_tokens']}")
    print(f"  JSON parsed: {ok}")
    print(f"  raw output:\n    {raw[:700]}")
    if ok and isinstance(obj, dict):
        for k in ("image_consistency", "forensic_support", "strongest_region",
                  "reason"):
            print(f"    {k:<20} {str(obj.get(k))[:90]!r}")
        missing = [k for k in ("image_consistency", "forensic_support",
                               "strongest_region", "reason") if k not in obj]
        print(f"  schema fields missing: {missing or 'none'}")
    print(f"  peak per-GPU: {gpu_mem()}")
    print(f"  nvidia-smi (MiB): {nvidia_used()}")
    rep["smoke3"] = dict(seconds=round(dt3, 1), stage_v_sha1=h,
                         matches_frozen_prompt=same, sample=s["pair_id"],
                         json_ok=ok, parsed=obj if ok else None, raw=raw[:2000],
                         info=info3, mem=gpu_mem(),
                         nvidia_used_MiB=nvidia_used())

    print("\n" + "=" * 86)
    print("5. VISUAL-TOKEN BUDGET PARITY vs 7B")
    print(f"  Stage-V visual tokens on 32B: {info3['n_visual_tokens']}")
    print(f"  min_pixels/max_pixels identical to the 7B config: "
          f"{mc['min_pixels']}/{mc['max_pixels']}")
    rep["visual_token_parity"] = dict(
        stage_v_visual_tokens=info3["n_visual_tokens"],
        min_pixels=mc["min_pixels"], max_pixels=mc["max_pixels"])

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1, default=str)
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
