"""Run the frozen score-map conflict intervention: 1656 Stage-B verdicts.

Nine conditions. The image-level score is always the target's own; only the
visualization varies. Transforms are applied to the raw float map at render time
using the frozen parameters -- the stored .npz maps are never modified.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json, normalize_stage_b

LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox",
        "blank", "shifted", "amplified", "binary", "condition", "control",
        "f1", "f2", "f3", "f4", "f5", "f6")
ORDER = ("F1_own", "F2_blank", "F3_donor", "F4_shifted", "F5_amplified",
         "F6_binary", "R1_own", "R2_blank", "R3_donor")


def assert_clean(p):
    rx = re.compile("|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
                             for t in LEAK), re.I)
    m = rx.search(p)
    if m:
        raise AssertionError(f"prompt leak: {m.group(0)!r}")


def render(m, size, colormap, vmin, vmax):
    """Frozen rendering: fixed 0-1 scale, no min-max stretch."""
    import matplotlib.cm as cm
    a = np.clip(np.asarray(m, np.float32), vmin, vmax)
    rgb = (cm.get_cmap(colormap)(a)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    if im.size != tuple(size):
        im = im.resize(tuple(size), Image.BILINEAR)
    return im


def transform(raw, kind, params):
    if kind == "own":
        return raw
    if kind == "blank":
        return np.zeros_like(raw)
    if kind == "shifted":
        return np.roll(raw, int(params["shift_fraction"] * raw.shape[1]), axis=1)
    if kind == "amplified":
        return np.clip(raw * params["amplify_k"], 0.0, 1.0)
    if kind == "binary":
        return (raw > params["binary_threshold"]).astype(np.float32)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]

    od = os.path.join(R, "trufor_conflict")
    F = json.load(open(os.path.join(od, "trufor_score_map_conflict_frozen.json")))
    PROMPT = F["prompts"]["MAP_AND_SCORE"]
    assert_clean(PROMPT)
    assert hashlib.sha1(PROMPT.encode()).hexdigest() == F["prompt_hashes"]["MAP_AND_SCORE"]
    assert "{stage_a_summary}" not in PROMPT
    print(f"prompt verified: sha1 {F['prompt_hashes']['MAP_AND_SCORE']}, no summary")
    C, TP = F["conditions"], F["transform_params"]
    RND = F["map_rendering"]
    print(f"transform params {TP}")
    print(f"rendering {RND['colormap']} fixed [{RND['vmin']},{RND['vmax']}]")
    S = F["samples"][:args.n] if args.n else F["samples"]
    byid = {s["sample_id"]: s for s in F["samples"]}
    dmap = F["donor_mapping"]
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

    from mllm_probe import QwenVLProbe
    probe = QwenVLProbe(cfg)
    from mllm_probe_mgpu import gpu_mem, nvidia_used
    torch.cuda.reset_peak_memory_stats(0)

    fh = open(path, "a", buffering=1)
    t0, n, times = time.time(), 0, []
    for k, s in enumerate(S):
        base_cache, raw_cache = {}, {}
        for cond in ORDER:
            if (s["sample_id"], cond) in done:
                continue
            spec = C[cond]
            label, mkind = spec["label"], spec["map"]
            if label not in base_cache:
                base_cache[label] = Image.open(
                    os.path.join(root, s[label])).convert("RGB")
            base = base_cache[label]
            own = s["evidence"][f"{label}_own"]
            score = own["score"]              # ALWAYS the target's own score
            if mkind == "donor":
                dn = byid[dmap[s["sample_id"]]]
                src_npz = dn["evidence"][f"{label}_own"]["npz"]
            else:
                src_npz = own["npz"]
            if src_npz not in raw_cache:
                raw_cache[src_npz] = np.load(src_npz)["map"].astype(np.float32)
            raw = raw_cache[src_npz]
            disp = raw if mkind == "donor" else transform(raw, mkind, TP)
            vis = render(disp, base.size, RND["colormap"], RND["vmin"], RND["vmax"])
            prompt = PROMPT.replace("{score}", f"{score:.3f}")
            assert "{score}" not in prompt
            ts = time.time()
            raw_out, info = probe.ask(prompt, [base, vis])
            dt = time.time() - ts
            times.append(dt)
            n += 1
            obj, _, ok = extract_json(raw_out)
            sb, notes = normalize_stage_b(obj) if ok else (None, ["json_parse_failed"])
            fh.write(json.dumps(dict(
                sample_id=s["sample_id"], coco_id=s["coco_id"],
                category=s["category"], mask_type=s["mask_type"],
                tamper_ratio=s["tamper_ratio"], condition=cond, label=label,
                ground_truth=label, map_kind=mkind, score_shown=round(score, 6),
                donor_sample_id=(dmap[s["sample_id"]] if mkind == "donor" else None),
                map_mean=float(disp.mean()), map_max=float(disp.max()),
                map_active_frac=float((disp > 0.5).mean()),
                final_verdict=(sb["final_verdict"] if sb else None),
                reason=(sb["reason"] if sb else ""), raw=raw_out, notes=notes,
                prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                n_images=2, seconds=round(dt, 2), info=info),
                ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(S)} sources  {n} inf  "
                  f"{(time.time()-t0)/60:.1f} min  mean {np.mean(times):.2f}s",
                  flush=True)
    fh.close()
    meta = dict(phase="score_map_conflict", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=gpu_mem()[0]["peak_GB"], nvidia_used_MiB=nvidia_used(),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_score_map_conflict_frozen.json"),
                    "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {meta['peak_GB']} GB")


if __name__ == "__main__":
    main()
