"""Run TruFor over the frozen feasibility set, plus the existing frozen ELA read-out.

400 TruFor inferences (200 fake + 200 paired real). For each image the whole-image
score, localization map and confidence map are stored. GT masks are NOT inputs; they
are only recorded by path for later scoring.

The ELA column uses the already-frozen blind_anomaly_score from forensic_tools --
no new read-out is defined -- so a paired tool comparison is possible on identical
sources later.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trufor_adapter import TruForAdapter
from forensic_tools import run_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--no-ela", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    grid = cfg["localization"]["grid"]
    tool_cfg = cfg["forensic_tools"]

    od = os.path.join(R, "trufor_feasibility")
    F = json.load(open(os.path.join(od, "trufor_feasibility_frozen.json")))
    S = F["samples"][:args.n] if args.n else F["samples"]
    maps_dir = os.path.join(od, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    out_path = os.path.join(od, "scores.jsonl")

    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                d = json.loads(l)
                done.add((d["sample_id"], d["label"]))
            except Exception:
                pass
    print(f"frozen protocol: {F['n_sources']} sources, {F['n_images']} images")
    print(f"resume: {len(done)} rows on disk")

    ad = TruForAdapter(gpu=args.gpu)
    ck_md5 = hashlib.md5(open(ad.ckpt_path, "rb").read()).hexdigest()
    assert ck_md5 == F["frozen_config"]["checkpoint_md5"], "checkpoint changed!"
    print(f"checkpoint md5 verified: {ck_md5}")
    torch.cuda.reset_peak_memory_stats(0)

    fh = open(out_path, "a", buffering=1)
    t0, n, times = time.time(), 0, []
    for k, s in enumerate(S):
        for label in ("fake", "real"):
            if (s["sample_id"], label) in done:
                continue
            img = os.path.join(root, s[label])
            r = ad.run(img)
            n += 1
            times.append(r["inference_time"])
            npz = os.path.join(maps_dir, f"{label}__{s['sample_id']}.npz")
            np.savez_compressed(npz, map=r["localization_map"].astype(np.float16),
                                conf=r["reliability_map"].astype(np.float16))
            rec = dict(coco_id=s["coco_id"], sample_id=s["sample_id"], label=label,
                       score=r["integrity_score"],
                       inference_time=round(r["inference_time"], 3),
                       localization_map_path=npz + "::map",
                       confidence_map_path=npz + "::conf",
                       input_size=r["input_size"], category=s["category"])
            if label == "fake":
                rec.update(gt_mask_path=os.path.join(root, s["mask"]),
                           tamper_ratio=s["tamper_ratio"], mask_type=s["mask_type"])
            if not args.no_ela:
                tools, _ = run_tools(img, tool_cfg, grid=grid)
                rec["ela_anomaly_score"] = float(tools["ela"]["anomaly_score"])
                rec["ela_valid"] = bool(tools["ela"]["valid"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(S)} sources  {n} inferences  "
                  f"{(time.time()-t0)/60:.1f} min  mean {np.mean(times):.2f}s",
                  flush=True)
    fh.close()
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    meta = dict(n_inferences=n, minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 3) if times else None,
                peak_gpu_GB=round(peak, 2), checkpoint_md5=ck_md5,
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_feasibility_frozen.json"), "rb").read()).hexdigest(),
                gpu=torch.cuda.get_device_name(0))
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences, {meta['minutes']} min, mean "
          f"{meta['mean_seconds']}s, peak {peak:.2f} GB")


if __name__ == "__main__":
    main()
