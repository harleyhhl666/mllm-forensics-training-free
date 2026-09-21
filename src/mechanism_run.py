"""Mechanism experiment: 2x2 context-bias design, frozen prompts, both labels.

Conditions come from mechanism_frozen.json. For a given sample the donor ELA
visualization is produced ONCE and handed to both C2 and C3, so their pixels are
identical by construction; the hash is recorded per row for verification.

Each sample runs all four conditions twice: once on the fake image, once on the
paired original. Nothing about ground truth, donor identity, or the experiment
design enters any prompt.
"""
import argparse, hashlib, io, json, os, sys, time
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools
from mllm_probe import QwenVLProbe, extract_json, normalize_stage_b

LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "eligib", "bbox",
        "iou", "manipulated region", "c0", "c1", "c2", "c3", "condition")


def assert_clean(p):
    import re
    rx = re.compile("|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
                            for t in LEAK), re.I)
    m = rx.search(p)
    if m:
        raise AssertionError(f"mechanism prompt leak: {m.group(0)!r}")


def png_sha1(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return hashlib.sha1(b.getvalue()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", default="mechanism_run")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    ds = cfg["datasets"]["tgif_sp"]
    root, grid = ds["root"], cfg["localization"]["grid"]
    tool_cfg = cfg["forensic_tools"]

    F = json.load(open(os.path.join(out_root, "mechanism", "mechanism_frozen.json")))
    PROMPTS = F["prompts"]
    NIMG = F["n_images_per_condition"]
    S = F["samples"][:args.n] if args.n else F["samples"]
    idx = {r["pair_id"]: r for r in json.load(open(os.path.join(
        out_root, "validation_v4", "tgif_sp_validation_index.json")))}

    for k, v in PROMPTS.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == F["prompt_hashes"][k], \
            f"prompt {k} does not match the frozen hash"
    print("prompt hash verification: all 4 match the frozen values")
    cidm = {s["pair_id"]: s["coco_id"] for s in F["samples"]}
    assert all(cidm[s["donor_pair_id"]] != s["coco_id"] for s in F["samples"])
    print(f"donor sanity: 0 same-coco, 0 self   samples={len(S)}")

    out_dir = os.path.join(out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mechanism.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
                done.add((d["pair_id"], d["label"], d["condition"]))
            except Exception:
                pass
    print(f"resume: {len(done)} rows already recorded")

    probe = QwenVLProbe(cfg)
    fh = open(path, "a", buffering=1)
    t0, n = time.time(), 0

    for k, s in enumerate(S):
        pid = s["pair_id"]
        rec = idx[pid]
        drec = idx[s["donor_pair_id"]]
        imgs_cache, donor_vis, donor_sha = {}, None, None

        for label, key in (("fake", "fake"), ("real", "real")):
            todo = [c for c in ("C0", "C1", "C2", "C3")
                    if (pid, label, c) not in done]
            if not todo:
                continue
            if label not in imgs_cache:
                imgs_cache[label] = Image.open(
                    os.path.join(root, rec[key])).convert("RGB")
            base = imgs_cache[label]
            if any(NIMG[c] == 2 for c in todo) and donor_vis is None:
                dt, _ = run_tools(os.path.join(root, drec["fake"]), tool_cfg, grid=grid)
                dv = Image.fromarray(cv2.cvtColor(dt["ela"]["vis_bgr"], cv2.COLOR_BGR2RGB))
                donor_vis = dv
                donor_sha = None            # resized per target below

            for cond in todo:
                prompt = PROMPTS[cond]
                if NIMG[cond] == 2:
                    vis = donor_vis.resize(base.size, Image.BILINEAR)
                    vsha = png_sha1(vis)     # same object -> same bytes for C2 & C3
                    images = [base, vis]
                else:
                    vsha = None
                    images = [base]
                raw, info = probe.ask(prompt, images)
                n += 1
                obj, _, ok = extract_json(raw)
                sb, notes = normalize_stage_b(obj) if ok else (None, ["json_parse_failed"])
                fh.write(json.dumps(dict(
                    pair_id=pid, coco_id=s["coco_id"], category=s["category"],
                    mask_type=s["mask_type"], tamper_ratio=s["tamper_ratio"],
                    label=label, condition=cond,
                    donor_pair_id=s["donor_pair_id"],
                    visualization_sha1=vsha, n_images=len(images),
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    final_verdict=(sb["final_verdict"] if sb else None),
                    reason=(sb["reason"] if sb else ""),
                    raw=raw, notes=notes, info=info,
                ), ensure_ascii=False) + "\n")
        if (k + 1) % 15 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(S)} samples  {n} inferences  {el/60:.1f} min",
                  flush=True)
    fh.close()
    print(f"done: {n} inferences in {(time.time()-t0)/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
