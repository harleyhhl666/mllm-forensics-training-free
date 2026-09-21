"""Run the frozen structured-field ablation: 2208 verdicts.

Six conditions x {fake, real} x 184 sources, all single-image. Structured JSON is
built by importing the frozen abstract() and strip_fields() so the field sets cannot
drift from the audited protocol.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json
from freeze_abstraction import abstract
from freeze_fields import strip_fields, FIELDS
from forensic_tools import CELL_NAMES

ORDER = ("S0_score_only", "S1_location", "S2_area", "S3_probability", "S4_loc_area",
         "S5_full")
LEAK = ("mask", "ground truth", "tamper", "donor", "coco", "iou", "bbox", "condition",
        "control", "s0", "s1", "s2", "s3", "s4", "s5")
CELLSET = set(CELL_NAMES)


def assert_clean(p):
    rx = re.compile("|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b"
                             for t in LEAK), re.I)
    m = rx.search(p)
    if m:
        raise AssertionError(f"prompt leak: {m.group(0)!r}")


def parse_regions(obj):
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
    od = os.path.join(R, "trufor_fields")
    F = json.load(open(os.path.join(
        od, "trufor_structured_field_ablation_frozen.json")))

    P, PH = F["prompts"], F["prompt_hashes"]
    for k, v in P.items():
        assert_clean(v)
        assert hashlib.sha1(v.encode()).hexdigest() == PH[k]
        assert "{stage_a_summary}" not in v
    assert len({PH[k] for k in ORDER if k != "S0_score_only"}) == 1
    print(f"prompts verified; S1-S5 share one hash "
          f"{PH['S5_full'][:16]}; S0 {PH['S0_score_only'][:16]}")
    for k in ORDER:
        assert FIELDS[k] == F["conditions"][k]["fields"], k
    print(f"field sets match freeze: "
          f"{ {k: F['conditions'][k]['fields'] for k in ORDER} }")
    S = F["samples"][:args.n] if args.n else F["samples"]
    print(f"sources {len(S)}  planned {len(S)*len(ORDER)*2} inferences (all 1-image)")

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
            base, full = None, None
            for cond in ORDER:
                if (s["sample_id"], label, cond) in done:
                    continue
                keys = FIELDS[cond]
                if base is None:
                    base = Image.open(os.path.join(root, s[label])).convert("RGB")
                prompt = P[cond].replace("{score}", f"{score:.3f}")
                st = None
                if keys is not None:
                    if full is None:
                        full = abstract(
                            np.load(ev["npz"])["map"].astype(np.float32), score)
                    st = strip_fields(full, keys)
                    prompt = prompt.replace("{structured}", json.dumps(st))
                assert "{score}" not in prompt and "{structured}" not in prompt
                ts = time.time()
                out, info = probe.ask(prompt, [base])
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
                fh.write(json.dumps(dict(
                    sample_id=s["sample_id"], coco_id=s["coco_id"],
                    category=s["category"], mask_type=s["mask_type"],
                    tamper_ratio=s["tamper_ratio"], condition=cond, label=label,
                    ground_truth=label, score_shown=round(score, 6), n_images=1,
                    fields=keys, has_location=F["conditions"][cond]["has_location"],
                    structured=st,
                    evidence_locations=([r["grid_location"] for r in full["regions"]]
                                        if full is not None else None),
                    evidence_n_regions=(None if full is None
                                        else len(full["regions"])),
                    evidence_areas=(None if full is None else
                                    [r["area_ratio"] for r in full["regions"]]),
                    final_verdict=verdict, referenced_regions=regs,
                    junk_regions=junk,
                    reason=(obj.get("reason", "") if ok and isinstance(obj, dict)
                            else ""),
                    raw=out, notes=notes,
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    seconds=round(dt, 2), info=info), ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            el = (time.time() - t0) / 60
            print(f"  {k+1}/{len(S)} sources  {n} inf  {el:.1f} min  "
                  f"mean {np.mean(times):.2f}s  eta "
                  f"{el/max(n,1)*(len(S)*len(ORDER)*2-n):.0f} min", flush=True)
    fh.close()
    meta = dict(phase="structured_field_ablation", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_structured_field_ablation_frozen.json"),
                    "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {meta['peak_GB']} GB")


if __name__ == "__main__":
    main()
