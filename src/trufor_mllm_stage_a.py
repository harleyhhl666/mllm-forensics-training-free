"""Phase T1: TruFor -> MLLM Stage-A grounding, 1104 inferences, protocol frozen.

Six conditions per source, from trufor_mllm_frozen.json:
  fake/real x {T0 no-tool, T1 own evidence, T2 donor evidence}

Stage A only. No verdict is requested and none is used. The evidence package
(localization map PNG + image-level score) is taken verbatim from the frozen
protocol, so the donor package moves as a unit and nothing is recomputed here.

Grid cell names are imported from forensic_tools (CELL_NAMES), not redefined.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import CELL_NAMES
from mllm_probe import extract_json

CELLS = set(CELL_NAMES)
LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox",
        "t0", "t1", "t2", "correct", "wrong", "condition")
ORDER = ("fake_T0_notool", "fake_T1_correct", "fake_T2_wrong",
         "real_T0_notool", "real_T1_correct", "real_T2_wrong")
EV_KEY = {("fake", "own"): "fake_own", ("fake", "donor"): "fake_donor",
          ("real", "own"): "real_own", ("real", "donor"): "real_donor"}


def assert_clean(p):
    rx = re.compile("|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
                             for t in LEAK), re.I)
    m = rx.search(p)
    if m:
        raise AssertionError(f"prompt leak: {m.group(0)!r}")


def cells_from_mask(m01, grid, min_overlap):
    H, W = m01.shape
    out = set()
    for i in range(grid):
        for j in range(grid):
            sub = m01[i * H // grid:(i + 1) * H // grid,
                      j * W // grid:(j + 1) * W // grid]
            if sub.size and sub.mean() >= min_overlap:
                out.add(CELL_NAMES[i * grid + j])
    return out


def parse_stage_a(raw):
    obj, _, ok = extract_json(raw)
    notes = []
    if not ok or not isinstance(obj, dict):
        return dict(evidence_used=None, predicted_regions=[],
                    tool_score_interpretation=None, reason=""), ["parse_failed"]
    low = {str(k).lower(): v for k, v in obj.items()}
    eu = low.get("evidence_used")
    if isinstance(eu, str):
        eu = eu.strip().lower() in ("true", "yes")
    if not isinstance(eu, bool):
        eu = None
        notes.append("bad_evidence_used")
    regs = low.get("predicted_regions") or []
    if isinstance(regs, str):
        regs = [regs]
    clean = []
    for r in regs:
        if not isinstance(r, str):
            continue
        t = r.strip().lower().replace("_", "-")
        if t in CELLS:
            clean.append(t)
        else:
            hit = [c for c in CELL_NAMES if c in t]
            if hit:
                clean.append(hit[0])
            else:
                notes.append("bad_region")
    si = str(low.get("tool_score_interpretation", "")).strip().lower()
    if si not in ("high", "medium", "low"):
        si = None
        notes.append("bad_score_interpretation")
    return dict(evidence_used=eu, predicted_regions=sorted(set(clean)),
                tool_score_interpretation=si,
                reason=str(low.get("reason", ""))[:700]), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", default="trufor_mllm_stageA")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--multi-gpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    grid = cfg["localization"]["grid"]
    mino = cfg["localization"]["gt_cell_min_overlap"]
    mthr = cfg["localization"]["mask_binarize_threshold"]

    od = os.path.join(R, "trufor_mllm")
    F = json.load(open(os.path.join(od, "trufor_mllm_frozen.json")))
    P, C = F["prompts"], F["conditions"]
    for k, v in P.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == F["prompt_hashes"][k], k
    print(f"prompt hashes verified: {len(P)} match the frozen protocol")
    S = F["samples"][:args.n] if args.n else F["samples"]
    print(f"sources {len(S)}  conditions {len(ORDER)}  "
          f"planned inferences {len(S)*len(ORDER)}")

    out_dir = os.path.join(R, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage_a.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                d = json.loads(l)
                done.add((d["sample_id"], d["condition"]))
            except Exception:
                pass
    print(f"resume: {len(done)} rows on disk")

    if args.multi_gpu:
        from mllm_probe_mgpu import QwenVLProbeMultiGPU as Probe
        probe = Probe(cfg)
    else:
        from mllm_probe import QwenVLProbe as Probe
        probe = Probe(cfg)
    from mllm_probe_mgpu import gpu_mem, nvidia_used
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.set_device(0)

    fh = open(path, "a", buffering=1)
    t0, n, times = time.time(), 0, []
    for k, s in enumerate(S):
        # GT cells from the mask, for fake targets (never shown to the model)
        gm = np.array(Image.open(os.path.join(root, s["mask"])).convert("L"))
        gtb = (gm > mthr).astype(np.uint8)
        gt_cells = sorted(cells_from_mask(gtb, grid, mino))

        for cond in ORDER:
            if (s["sample_id"], cond) in done:
                continue
            spec = C[cond]
            label, evk = spec["label"], spec["evidence"]
            base = Image.open(os.path.join(root, s[label])).convert("RGB")
            if evk is None:
                prompt = P["STAGE_A_NOTOOL"]
                images = [base]
                ev = None
                cue_cells = []
            else:
                ev = s["evidence"][EV_KEY[(label, evk)]]
                # the prompt body contains literal JSON braces, so str.format
                # cannot be used; replace only the score placeholder
                prompt = P["STAGE_A_TOOL"].replace("{score}", f"{ev['score']:.3f}")
                assert "{score}" not in prompt
                vis = Image.open(ev["map_png"]).convert("RGB")
                if vis.size != base.size:
                    vis = vis.resize(base.size, Image.BILINEAR)
                images = [base, vis]
                z = np.load(ev["npz"])
                cue_cells = sorted(cells_from_mask(
                    (z["map"].astype(np.float32) > 0.5).astype(np.uint8),
                    grid, mino))
            ts = time.time()
            raw, info = probe.ask(prompt, images)
            dt = time.time() - ts
            times.append(dt)
            n += 1
            sa, notes = parse_stage_a(raw)
            pred = set(sa["predicted_regions"])
            rec = dict(
                sample_id=s["sample_id"], coco_id=s["coco_id"],
                category=s["category"], mask_type=s["mask_type"],
                tamper_ratio=s["tamper_ratio"], n_gt_cells=s["n_gt_cells"],
                condition=cond, label=label, evidence=evk,
                donor_sample_id=s["donor_sample_id"],
                evidence_score=(round(ev["score"], 6) if ev else None),
                gt_cells=gt_cells, cue_cells=cue_cells,
                predicted_cells=sorted(pred),
                gt_hit=bool(pred & set(gt_cells)) if gt_cells else None,
                cue_hit=bool(pred & set(cue_cells)) if cue_cells else None,
                joint_hit=(bool(pred & set(gt_cells)) and
                           bool(pred & set(cue_cells)))
                if (gt_cells and cue_cells) else None,
                n_pred=len(pred),
                stage_a=sa, notes=notes, raw=raw,
                prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                n_images=len(images), seconds=round(dt, 2), info=info)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(S)} sources  {n} inf  "
                  f"{(time.time()-t0)/60:.1f} min  mean {np.mean(times):.2f}s",
                  flush=True)
    fh.close()
    peak = {i: gpu_mem()[i]["peak_GB"] for i in range(torch.cuda.device_count())}
    meta = dict(phase="T1_stage_a", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=peak, nvidia_used_MiB=nvidia_used(),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_mllm_frozen.json"), "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"],
                decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {peak}")


if __name__ == "__main__":
    main()
