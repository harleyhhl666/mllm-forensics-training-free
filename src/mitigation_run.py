"""Run the frozen M1 mitigation experiment: 7 conditions + M-none, both labels.

Per (sample, label):
  Stage V is run ONCE per cue (correct, donor) and the parsed state is reused by
  the matching M and S conditions. S conditions get the plain B_CUE prompt, so the
  Stage-V pass happened but its content never reaches Stage D.

Nothing about ground truth, donor identity, tamper ratio or the condition name
enters any prompt.
"""
import argparse, hashlib, io, json, os, re, sys, time
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools
from mllm_probe import QwenVLProbe, extract_json, normalize_stage_b

CELLS = ("top-left", "top-center", "top-right", "center-left", "center",
         "center-right", "bottom-left", "bottom-center", "bottom-right")
LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox",
        "b-correct", "b-wrong", "m-correct", "m-wrong", "s-correct", "s-wrong")


def assert_clean(p):
    rx = re.compile("|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
                             for t in LEAK), re.I)
    m = rx.search(p)
    if m:
        raise AssertionError(f"prompt leak: {m.group(0)!r}")


def png_sha1(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return hashlib.sha1(b.getvalue()).hexdigest()


def parse_stage_v(raw):
    """-> (state, fields, notes). Unparseable or invalid -> State C, flagged."""
    obj, _, ok = extract_json(raw)
    notes = []
    if not ok or not isinstance(obj, dict):
        return "C", dict(image_consistency=None, forensic_support=None,
                         strongest_region=None, reason=""), ["stage_v_parse_failed"]
    low = {str(k).lower(): v for k, v in obj.items()}
    ic = str(low.get("image_consistency", "")).strip().lower()
    fs = str(low.get("forensic_support", "")).strip().lower()
    reg = str(low.get("strongest_region", "")).strip().lower().replace("_", "-")
    if ic not in ("matched", "mismatched"):
        ic = None
        notes.append("bad_image_consistency")
    if fs not in ("supported", "insufficient"):
        fs = None
        notes.append("bad_forensic_support")
    if reg not in CELLS:
        hit = [c for c in CELLS if c in reg]
        reg = hit[0] if hit else None
        if reg is None:
            notes.append("bad_region")
    if ic is None or fs is None:
        state = "C"
        notes.append("invalid_stage_v_to_state_C")
    elif ic == "mismatched":
        state = "C"
    else:
        state = "A" if fs == "supported" else "B"
    return state, dict(image_consistency=ic, forensic_support=fs,
                       strongest_region=reg,
                       reason=str(low.get("reason", ""))[:600]), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", default="mitigation_run")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    ds = cfg["datasets"]["tgif_sp"]
    root, grid, tool_cfg = ds["root"], cfg["localization"]["grid"], cfg["forensic_tools"]

    F = json.load(open(os.path.join(out_root, "mitigation", "mitigation_frozen.json")))
    P, COND = F["prompts"], F["conditions"]
    for k, v in P.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == F["prompt_hashes"][k], k
    print(f"prompt hash verification: all {len(P)} match the frozen protocol")
    S = F["samples"][:args.n] if args.n else F["samples"]
    cidm = {s["pair_id"]: s["coco_id"] for s in F["samples"]}
    assert all(cidm[s["donor_pair_id"]] != s["coco_id"] for s in F["samples"])
    print(f"donor sanity: 0 same-coco, 0 self   samples={len(S)}")
    byid = {s["pair_id"]: s for s in F["samples"]}

    out_dir = os.path.join(out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    dpath = os.path.join(out_dir, "decisions.jsonl")
    vpath = os.path.join(out_dir, "stage_v.jsonl")
    done, vdone = set(), {}
    if os.path.exists(dpath):
        for l in open(dpath):
            try:
                d = json.loads(l)
                done.add((d["pair_id"], d["label"], d["condition"]))
            except Exception:
                pass
    if os.path.exists(vpath):
        for l in open(vpath):
            try:
                d = json.loads(l)
                vdone[(d["pair_id"], d["label"], d["cue"])] = d
            except Exception:
                pass
    print(f"resume: {len(done)} decisions, {len(vdone)} Stage-V results on disk")

    probe = QwenVLProbe(cfg)
    dfh, vfh = open(dpath, "a", buffering=1), open(vpath, "a", buffering=1)
    t0, n = time.time(), 0
    ORDER = ("B-none", "B-correct", "B-wrong", "S-correct", "S-wrong",
             "M-correct", "M-wrong", "M-none")

    for si, s in enumerate(S):
        pid = s["pair_id"]
        dn = byid[s["donor_pair_id"]]
        vis_cache, base_cache = {}, {}

        def cue_vis(label, which):
            """ELA visualization, resized to the target image, cached."""
            if which in vis_cache:
                return vis_cache[which]
            src = os.path.join(root, s["fake"] if which == "correct" else dn["fake"])
            t, _ = run_tools(src, tool_cfg, grid=grid)
            v = Image.fromarray(cv2.cvtColor(t["ela"]["vis_bgr"], cv2.COLOR_BGR2RGB))
            vis_cache[which] = v
            return v

        for label, key in (("fake", "fake"), ("real", "real")):
            if label not in base_cache:
                base_cache[label] = Image.open(
                    os.path.join(root, s[key])).convert("RGB")
            base = base_cache[label]
            vis_cache.clear()

            # ---- Stage V once per cue, shared by the M and S conditions
            states = {}
            for which in ("correct", "donor"):
                k = (pid, label, which)
                if k in vdone:
                    states[which] = vdone[k]
                    continue
                v = cue_vis(label, which).resize(base.size, Image.BILINEAR)
                raw, info = probe.ask(P["STAGE_V"], [base, v])
                n += 1
                st, fields, notes = parse_stage_v(raw)
                rec = dict(pair_id=pid, coco_id=s["coco_id"], label=label, cue=which,
                           state=st, **fields, notes=notes, raw=raw,
                           visualization_sha1=png_sha1(v), info=info,
                           donor_pair_id=s["donor_pair_id"],
                           tamper_ratio=s["tamper_ratio"], mask_type=s["mask_type"])
                vfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                vdone[k] = rec
                states[which] = rec

            # ---- decision conditions
            for cond in ORDER:
                if (pid, label, cond) in done:
                    continue
                c = COND[cond]
                which = {"correct": "correct", "donor": "donor"}.get(c["cue"])
                if c["cue"] is None:
                    images, prompt, st = [base], P[c["prompt"]], None
                else:
                    v = cue_vis(label, which).resize(base.size, Image.BILINEAR)
                    images = [base, v]
                    if c["inject"]:
                        st = states[which]["state"]
                        prompt = P[f"STAGE_D_{st}"]
                    else:
                        st = None                      # sham: state withheld
                        prompt = P["B_CUE"]
                raw, info = probe.ask(prompt, images)
                n += 1
                obj, _, ok = extract_json(raw)
                sb, notes = normalize_stage_b(obj) if ok else (None, ["json_parse_failed"])
                dfh.write(json.dumps(dict(
                    pair_id=pid, coco_id=s["coco_id"], category=s["category"],
                    mask_type=s["mask_type"], tamper_ratio=s["tamper_ratio"],
                    label=label, condition=cond, cue=c["cue"],
                    stage_v_used=bool(c["stage_v"]), state_injected=st,
                    stage_v_state=(states[which]["state"] if c["cue"] else None),
                    donor_pair_id=s["donor_pair_id"], n_images=len(images),
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    final_verdict=(sb["final_verdict"] if sb else None),
                    reason=(sb["reason"] if sb else ""), raw=raw, notes=notes,
                    info=info), ensure_ascii=False) + "\n")
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(S)} samples  {n} inferences  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    dfh.close()
    vfh.close()
    print(f"done: {n} inferences in {(time.time()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
