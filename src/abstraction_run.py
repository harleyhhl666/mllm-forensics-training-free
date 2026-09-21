"""Run the frozen Evidence Abstraction Layer experiment: 1840 verdicts.

Five conditions x {fake, real} x 184 sources. Structured evidence is produced by
importing the frozen `abstract()` from freeze_abstraction.py, so the runner cannot
drift from the audited extraction rules.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json
from freeze_abstraction import abstract, MIRROR, BIN_THR, MIN_AREA_RATIO, TOP_K
from forensic_tools import CELL_NAMES

ORDER = ("E0_score_only", "E1_raw_map", "E2_structured", "E3_both",
         "E4_shifted_struct")
LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox", "mirror",
        "shifted", "condition", "control", "e0", "e1", "e2", "e3", "e4")
CELLSET = set(CELL_NAMES)


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


def parse_regions(obj):
    """Normalise referenced_regions to a set of valid 3x3 names + record junk."""
    v = obj.get("referenced_regions", []) if isinstance(obj, dict) else []
    if isinstance(v, str):
        v = [v]
    out, junk = set(), []
    for x in (v or []):
        if not isinstance(x, str):
            junk.append(str(x))
            continue
        s = x.strip().lower().replace("_", "-").replace(" ", "-")
        s = s.replace("centre", "center").replace("middle", "center")
        if s in CELLSET:
            out.add(s)
        elif s in ("none", "", "[]", "n/a"):
            pass
        else:
            hit = [c for c in CELL_NAMES if c in s]
            if len(hit) == 1:
                out.add(hit[0])
            else:
                junk.append(x)
    return sorted(out), junk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    od = os.path.join(R, "trufor_abstraction")
    F = json.load(open(os.path.join(od, "trufor_evidence_abstraction_frozen.json")))

    P, PH, C = F["prompts"], F["prompt_hashes"], F["conditions"]
    for k, v in P.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == PH[k]
        assert "{stage_a_summary}" not in v
    assert PH["E2"] == PH["E4"]
    print(f"prompts verified; E2==E4 {PH['E2']==PH['E4']}")
    assert (BIN_THR == F["abstraction_layer"]["binary_threshold"]
            and MIN_AREA_RATIO == F["abstraction_layer"]["min_area_ratio"]
            and TOP_K == F["abstraction_layer"]["top_k"])
    print(f"abstraction params match freeze: thr {BIN_THR} min_area "
          f"{MIN_AREA_RATIO} K {TOP_K}")
    RND = F["map_rendering"]
    S = F["samples"][:args.n] if args.n else F["samples"]
    ELIG = {k: set(v) for k, v in F["e4_eligible_ids"].items()}
    print(f"sources {len(S)}  planned {len(S)*len(ORDER)*2} inferences")
    print(f"E4 eligible: fake {len(ELIG['fake'])} real {len(ELIG['real'])}")

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
    torch.cuda.reset_peak_memory_stats(0)

    fh = open(path, "a", buffering=1)
    t0, n, times = time.time(), 0, []
    for k, s in enumerate(S):
        for label in ("fake", "real"):
            ev = s["evidence"][f"{label}_own"]
            score = ev["score"]
            base, raw, struct, mstruct = None, None, None, None
            for cond in ORDER:
                if (s["sample_id"], label, cond) in done:
                    continue
                spec = C[cond]
                if base is None:
                    base = Image.open(os.path.join(root, s[label])).convert("RGB")
                if raw is None and (spec["raw_map"] or spec["structured"]):
                    raw = np.load(ev["npz"])["map"].astype(np.float32)
                if spec["structured"]:
                    if spec["shift"]:
                        if mstruct is None:
                            mstruct = abstract(raw, score, mirror=True)
                        st = mstruct
                    else:
                        if struct is None:
                            struct = abstract(raw, score, mirror=False)
                        st = struct
                else:
                    st = None
                prompt = P[cond.split("_")[0]].replace("{score}", f"{score:.3f}")
                if st is not None:
                    prompt = prompt.replace("{structured}", json.dumps(st, indent=1))
                assert "{score}" not in prompt and "{structured}" not in prompt
                imgs = [base] + ([render(raw, base.size, RND)] if spec["raw_map"] else [])
                ts = time.time()
                out, info = probe.ask(prompt, imgs)
                dt = time.time() - ts
                times.append(dt)
                n += 1
                obj, _, ok = extract_json(out)
                verdict, notes = None, []
                if ok and isinstance(obj, dict):
                    fv = str(obj.get("final_verdict", "")).strip().lower()
                    verdict = fv if fv in ("real", "fake") else None
                    if verdict is None:
                        notes.append("bad_verdict")
                else:
                    notes.append("json_parse_failed")
                regs, junk = parse_regions(obj if ok else {})
                if junk:
                    notes.append("unparsed_region_tokens")
                ev_locs = ([r["grid_location"] for r in st["regions"]]
                           if st is not None else None)
                fh.write(json.dumps(dict(
                    sample_id=s["sample_id"], coco_id=s["coco_id"],
                    category=s["category"], mask_type=s["mask_type"],
                    tamper_ratio=s["tamper_ratio"], condition=cond, label=label,
                    ground_truth=label, score_shown=round(score, 6),
                    n_images=len(imgs), has_structured=bool(st),
                    evidence_locations=ev_locs,
                    evidence_n_regions=(None if st is None else len(st["regions"])),
                    structured=st,
                    e4_eligible=(s["sample_id"] in ELIG[label]),
                    final_verdict=verdict, referenced_regions=regs,
                    junk_regions=junk, reason=(obj.get("reason", "") if ok
                                               and isinstance(obj, dict) else ""),
                    raw=out, notes=notes,
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    seconds=round(dt, 2), info=info), ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            el = (time.time() - t0) / 60
            print(f"  {k+1}/{len(S)} sources  {n} inf  {el:.1f} min  "
                  f"mean {np.mean(times):.2f}s  eta "
                  f"{el/max(n,1)*(len(S)*len(ORDER)*2-n):.0f} min", flush=True)
    fh.close()
    meta = dict(phase="evidence_abstraction", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_evidence_abstraction_frozen.json"),
                    "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {meta['peak_GB']} GB")


if __name__ == "__main__":
    main()
