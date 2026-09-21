"""Qwen2.5-VL-32B capacity replication: 736 Stage-B verdicts on 184 fake sources.

Four conditions (B0 score-only, B1 own map, B2 blank map, B3 shifted map). The only
intended difference from the 7B runs is the model. Prompts, score formatting, map
rendering, shift transform, visual token budget and decoding are read from the frozen
protocol and never modified here.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json, normalize_stage_b
from mllm_probe_mgpu import QwenVLProbeMultiGPU, gpu_mem, nvidia_used

ORDER = ("B0_score_only", "B1_own_map", "B2_blank_map", "B3_shifted_map")
LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox", "blank",
        "shifted", "amplified", "binary", "condition", "control", "b0", "b1", "b2",
        "b3")


def assert_clean(p):
    rx = re.compile("|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
                             for t in LEAK), re.I)
    m = rx.search(p)
    if m:
        raise AssertionError(f"prompt leak: {m.group(0)!r}")


def render(m, size, rnd):
    import matplotlib.cm as cm
    a = np.clip(np.asarray(m, np.float32), rnd["vmin"], rnd["vmax"])
    im = Image.fromarray((cm.get_cmap(rnd["colormap"])(a)[..., :3] * 255).astype(np.uint8))
    return im if im.size == tuple(size) else im.resize(tuple(size), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_32b")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config_32b))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    od = os.path.join(R, "trufor_32b")
    F = json.load(open(os.path.join(od, "trufor_32b_capacity_frozen.json")))

    P = F["prompts"]
    for k, v in P.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == F["prompt_hashes"][k]
        assert "{stage_a_summary}" not in v
    print(f"prompts verified {F['prompt_hashes']}")
    assert cfg["model"]["path"] == F["model"]["path"]
    assert cfg["model"]["dtype"] == F["model"]["dtype"] == "bfloat16"
    assert cfg["model"]["min_pixels"] == F["parity"]["min_pixels"]
    assert cfg["model"]["max_pixels"] == F["parity"]["max_pixels"]
    assert cfg["model"]["generation"] == F["parity"]["generation"]
    print(f"model/parity assertions passed: snapshot {F['model']['snapshot'][:16]} "
          f"{cfg['model']['dtype']} device_map {cfg['model']['device_map']}")
    SH, RND = F["shift_fraction"], F["map_rendering"]
    C = F["conditions"]
    S = F["samples"][:args.n] if args.n else F["samples"]
    print(f"sources {len(S)}  planned {len(S)*len(ORDER)} inferences")

    path = os.path.join(od, "verdicts.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                d = json.loads(l)
                done.add((d["sample_id"], d["condition"]))
            except Exception:
                pass
    print(f"resume: {len(done)} rows on disk")

    probe = QwenVLProbeMultiGPU(cfg)
    print(f"device map: {probe.device_map_resolved}")
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    fh = open(path, "a", buffering=1)
    t0, n, times = time.time(), 0, []
    for k, s in enumerate(S):
        base = None
        raw = None
        for cond in ORDER:
            if (s["sample_id"], cond) in done:
                continue
            spec = C[cond]
            if base is None:
                base = Image.open(os.path.join(root, s["fake"])).convert("RGB")
            own = s["evidence"]["fake_own"]
            score = own["score"]
            prompt = P["SCORE_ONLY"] if spec["map"] is None else P["MAP_AND_SCORE"]
            prompt = prompt.replace("{score}", f"{score:.3f}")
            assert "{score}" not in prompt
            if spec["map"] is None:
                imgs, disp = [base], None
            else:
                if raw is None:
                    raw = np.load(own["npz"])["map"].astype(np.float32)
                if spec["map"] == "own":
                    disp = raw
                elif spec["map"] == "blank":
                    disp = np.zeros_like(raw)
                elif spec["map"] == "shifted":
                    disp = np.roll(raw, int(SH * raw.shape[1]), axis=1)
                else:
                    raise ValueError(spec["map"])
                imgs = [base, render(disp, base.size, RND)]
            ts = time.time()
            out, info = probe.ask(prompt, imgs)
            dt = time.time() - ts
            times.append(dt)
            n += 1
            obj, _, ok = extract_json(out)
            sb, notes = normalize_stage_b(obj) if ok else (None, ["json_parse_failed"])
            fh.write(json.dumps(dict(
                sample_id=s["sample_id"], coco_id=s["coco_id"],
                category=s["category"], mask_type=s["mask_type"],
                tamper_ratio=s["tamper_ratio"], condition=cond, label="fake",
                ground_truth="fake", map_kind=spec["map"], score_shown=round(score, 6),
                map_mean=(None if disp is None else float(disp.mean())),
                map_active_frac=(None if disp is None else float((disp > .5).mean())),
                final_verdict=(sb["final_verdict"] if sb else None),
                reason=(sb["reason"] if sb else ""), raw=out, notes=notes,
                prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                n_images=len(imgs), seconds=round(dt, 2), info=info),
                ensure_ascii=False) + "\n")
        if (k + 1) % 10 == 0:
            el = (time.time() - t0) / 60
            print(f"  {k+1}/{len(S)} sources  {n} inf  {el:.1f} min  "
                  f"mean {np.mean(times):.2f}s  eta "
                  f"{el/max(n,1)*(len(S)*len(ORDER)-n):.0f} min", flush=True)
    fh.close()
    mem = gpu_mem()
    meta = dict(phase="trufor_32b_capacity", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                gpu_mem=mem, nvidia_used_MiB=nvidia_used(),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_32b_capacity_frozen.json"), "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], dtype=cfg["model"]["dtype"],
                device_map_resolved=str(probe.device_map_resolved),
                decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s)")
    print(f"peaks: {mem}")


if __name__ == "__main__":
    main()
