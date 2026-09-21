"""Run the frozen 2x2 score/map decomposition: 1472 Stage-B verdicts.

Four conditions x {fake, real} x 184 sources. No Stage-A summary is used anywhere,
so the map-absent conditions carry no spatial information. Only the own (correct)
evidence package is used.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json, normalize_stage_b

LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox",
        "c00", "c10", "c01", "c11", "condition", "observation summary")
CONDS = ("C00", "C10", "C01", "C11")


def assert_clean(p):
    rx = re.compile("|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
                             for t in LEAK), re.I)
    m = rx.search(p)
    if m:
        raise AssertionError(f"prompt leak: {m.group(0)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", default="trufor_score_map")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]

    od = os.path.join(R, "trufor_score_map")
    F = json.load(open(os.path.join(od, "trufor_score_map_ablation_frozen.json")))
    P, C = F["prompts"], F["conditions"]
    for k, v in P.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == F["prompt_hashes"][k], k
        assert "{stage_a_summary}" not in v
    print(f"prompt hashes verified: {len(P)}; no Stage-A summary in any prompt")
    S = F["samples"][:args.n] if args.n else F["samples"]
    print(f"sources {len(S)}  planned inferences {len(S)*8}")

    path = os.path.join(od, "verdicts.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                d = json.loads(l)
                done.add((d["sample_id"], d["label"], d["condition"]))
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
        for label in ("fake", "real"):
            base = None
            ev = s["evidence"][f"{label}_own"]
            for cond in CONDS:
                if (s["sample_id"], label, cond) in done:
                    continue
                spec = C[cond]
                if base is None:
                    base = Image.open(os.path.join(root, s[label])).convert("RGB")
                prompt = P[cond]
                if spec["score"]:
                    prompt = prompt.replace("{score}", f"{ev['score']:.3f}")
                assert "{score}" not in prompt
                images = [base]
                if spec["map"]:
                    vis = Image.open(ev["map_png"]).convert("RGB")
                    if vis.size != base.size:
                        vis = vis.resize(base.size, Image.BILINEAR)
                    images.append(vis)
                assert len(images) == spec["n_images"]
                ts = time.time()
                raw, info = probe.ask(prompt, images)
                dt = time.time() - ts
                times.append(dt)
                n += 1
                obj, _, ok = extract_json(raw)
                sb, notes = normalize_stage_b(obj) if ok else (None, ["json_parse_failed"])
                fh.write(json.dumps(dict(
                    sample_id=s["sample_id"], coco_id=s["coco_id"],
                    category=s["category"], mask_type=s["mask_type"],
                    tamper_ratio=s["tamper_ratio"], label=label, ground_truth=label,
                    condition=cond, score_given=spec["score"], map_given=spec["map"],
                    evidence_score=(round(ev["score"], 6) if spec["score"] else None),
                    actual_score=round(ev["score"], 6),
                    final_verdict=(sb["final_verdict"] if sb else None),
                    reason=(sb["reason"] if sb else ""), raw=raw, notes=notes,
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    n_images=len(images), seconds=round(dt, 2), info=info),
                    ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(S)} sources  {n} inf  "
                  f"{(time.time()-t0)/60:.1f} min  mean {np.mean(times):.2f}s",
                  flush=True)
    fh.close()
    meta = dict(phase="score_map_2x2", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=gpu_mem()[0]["peak_GB"], nvidia_used_MiB=nvidia_used(),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_score_map_ablation_frozen.json"),
                    "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {meta['peak_GB']} GB")


if __name__ == "__main__":
    main()
