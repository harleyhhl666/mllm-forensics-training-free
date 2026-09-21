"""Phase T2: TruFor -> MLLM Stage-B final verdict, 1104 inferences.

Each condition receives its OWN Stage-A summary, read from the frozen Stage-A
output on disk. Stage A is never re-run and never mutated; this script only reads
it. The evidence package (map PNG + score) is taken verbatim from the frozen
protocol, so a donor package stays a donor package.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json, normalize_stage_b

LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox",
        "t0", "t1", "t2", "condition")
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


def summarize(sa):
    """Verbatim restatement of this condition's own Stage-A JSON output."""
    return json.dumps(dict(
        evidence_used=sa.get("evidence_used"),
        predicted_regions=sa.get("predicted_regions", []),
        tool_score_interpretation=sa.get("tool_score_interpretation"),
        reason=sa.get("reason", "")), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", default="trufor_mllm_stageB")
    ap.add_argument("--stage-a-tag", default="trufor_mllm_stageA")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--multi-gpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]

    od = os.path.join(R, "trufor_mllm")
    F = json.load(open(os.path.join(od, "trufor_mllm_frozen.json")))
    P, C = F["prompts"], F["conditions"]
    for k, v in P.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == F["prompt_hashes"][k], k
    print(f"prompt hashes verified: {len(P)} match the frozen protocol")

    # frozen Stage-A outputs, read-only
    A = {}
    ap_path = os.path.join(R, args.stage_a_tag, "stage_a.jsonl")
    for l in open(ap_path):
        r = json.loads(l)
        A[(r["sample_id"], r["condition"])] = r
    print(f"Stage-A rows loaded (read-only): {len(A)} from {ap_path}")
    S = F["samples"][:args.n] if args.n else F["samples"]
    need = [(s["sample_id"], c) for s in S for c in ORDER]
    missing = [k for k in need if k not in A]
    assert not missing, f"missing Stage-A for {len(missing)} cells"
    print(f"every Stage-B cell has its own Stage-A summary: True ({len(need)})")

    out_dir = os.path.join(R, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage_b.jsonl")
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
        for cond in ORDER:
            if (s["sample_id"], cond) in done:
                continue
            spec = C[cond]
            label, evk = spec["label"], spec["evidence"]
            sa = A[(s["sample_id"], cond)]["stage_a"]
            summ = summarize(sa)
            base = Image.open(os.path.join(root, s[label])).convert("RGB")
            if evk is None:
                prompt = P["STAGE_B_NOTOOL"].replace("{stage_a_summary}", summ)
                images = [base]
                ev = None
            else:
                ev = s["evidence"][EV_KEY[(label, evk)]]
                prompt = (P["STAGE_B_TOOL"]
                          .replace("{score}", f"{ev['score']:.3f}")
                          .replace("{stage_a_summary}", summ))
                vis = Image.open(ev["map_png"]).convert("RGB")
                if vis.size != base.size:
                    vis = vis.resize(base.size, Image.BILINEAR)
                images = [base, vis]
            assert "{score}" not in prompt and "{stage_a_summary}" not in prompt
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
                tamper_ratio=s["tamper_ratio"], condition=cond,
                label=label, ground_truth=label, evidence=evk,
                donor_sample_id=s["donor_sample_id"],
                evidence_score=(round(ev["score"], 6) if ev else None),
                stage_a_gt_hit=A[(s["sample_id"], cond)]["gt_hit"],
                stage_a_cue_hit=A[(s["sample_id"], cond)]["cue_hit"],
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
    peak = {i: gpu_mem()[i]["peak_GB"] for i in range(torch.cuda.device_count())}
    meta = dict(phase="T2_stage_b", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=peak, nvidia_used_MiB=nvidia_used(),
                stage_a_source=ap_path,
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_mllm_frozen.json"), "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"],
                decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {peak}")


if __name__ == "__main__":
    main()
