"""Run the frozen 32B final capacity replication: 1622 verdicts.

B0-B3 across 184 fake + 184 paired real, plus B4 on the frozen mirror-eligible
subset. Prompts and payloads are reused byte-identically from the 7B protocols; the
corpus payload hash is verified at startup, which is the guard that was missing when
a payload slip invalidated an earlier round.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json
from mllm_probe_mgpu import QwenVLProbeMultiGPU, gpu_mem, nvidia_used
from freeze_abstraction import abstract
from freeze_fields import strip_fields
from forensic_tools import CELL_NAMES

MAIN = ("B0_score_only", "B1_score_rule", "B2_structured", "B3_structured_rule")
CELLSET = set(CELL_NAMES)


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
    ap.add_argument("config_32b")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config_32b))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    od = os.path.join(R, "trufor_32b_final")
    F = json.load(open(os.path.join(od, "trufor_32b_final_capacity_frozen.json")))

    P, PH, C = F["prompts"], F["prompt_hashes"], F["cells"]
    for k, v in P.items():
        assert hashlib.sha1(v.encode()).hexdigest() == PH[k], k
        assert "manipulation_present" not in v and "manipulation_extent" not in v
    print(f"prompts verified {({k: PH[k][:12] for k in P})}")
    assert cfg["model"]["path"] == F["model"]["path"]
    assert cfg["model"]["dtype"] == "bfloat16"
    for k in ("min_pixels", "max_pixels"):
        assert cfg["model"][k] == F["parity"][k], k
    for k in ("do_sample", "temperature", "max_new_tokens", "repetition_penalty"):
        assert cfg["model"]["generation"][k] == F["parity"][k], k
    print(f"parity OK: snapshot {F['model']['snapshot'][:16]} {cfg['model']['dtype']} "
          f"device_map {cfg['model']['device_map']}")

    FLD = F["evidence"]["structured_fields"]
    S = F["samples"][:args.n] if args.n else F["samples"]
    ELIG = {k: set(v) for k, v in F["b4"]["eligible_ids"].items()}

    # --- payload guard: reproduce the frozen corpus hash or refuse to start ---
    h = hashlib.sha1()
    PAY, MPAY = {}, {}
    for s in F["samples"]:
        for lb in ("fake", "real"):
            e = s["evidence"][f"{lb}_own"]
            m = np.load(e["npz"])["map"].astype(np.float32)
            full = abstract(m, e["score"])
            sub = strip_fields(full, FLD)
            h.update(json.dumps(sub, sort_keys=True).encode())
            PAY[(s["sample_id"], lb)] = json.dumps(sub)
            if s["sample_id"] in ELIG[lb]:
                MPAY[(s["sample_id"], lb)] = json.dumps(
                    strip_fields(abstract(m, e["score"], mirror=True), FLD))
    got = h.hexdigest()
    want = F["evidence"]["corpus_payload_sha1"]
    print(f"corpus payload sha1 {got[:16]} vs frozen {want[:16]}")
    assert got == want, "payload drift — refusing to run"
    print("payload guard OK")
    print(f"B4 eligible payloads prepared: {len(MPAY)}")

    plan = [(s, lb, c) for s in S for lb in ("fake", "real") for c in MAIN]
    plan += [(s, lb, "B4_mirrored") for s in S for lb in ("fake", "real")
             if s["sample_id"] in ELIG[lb]]
    print(f"sources {len(S)}  planned {len(plan)} inferences (single-image)")

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

    probe = QwenVLProbeMultiGPU(cfg)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    fh = open(path, "a", buffering=1)
    t0, n, times = time.time(), 0, []
    total = len(plan) - len(done)
    print(f"start {time.strftime('%Y-%m-%d %H:%M:%S %Z')}  to run {total}", flush=True)
    imgcache = {}
    for i, (s, label, cond) in enumerate(plan):
        key = (s["sample_id"], label, cond)
        if key in done:
            continue
        ev = s["evidence"][f"{label}_own"]
        score = ev["score"]
        ik = (s["sample_id"], label)
        if ik not in imgcache:
            imgcache.clear()
            imgcache[ik] = Image.open(os.path.join(root, s[label])).convert("RGB")
        base = imgcache[ik]
        spec = C[cond]
        prompt = P[cond].replace("{score}", f"{score:.3f}")
        if spec["structured"]:
            pl = MPAY[ik] if spec["mirrored"] else PAY[ik]
            prompt = prompt.replace("{structured}", pl)
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
        evloc = [r["grid_location"] for r in json.loads(
            MPAY[ik] if spec["mirrored"] else PAY[ik])["regions"]] \
            if spec["structured"] else None
        fh.write(json.dumps(dict(
            sample_id=s["sample_id"], coco_id=s["coco_id"], category=s["category"],
            mask_type=s["mask_type"], tamper_ratio=s["tamper_ratio"],
            condition=cond, label=label, ground_truth=label,
            score_shown=round(score, 6), n_images=1,
            structured=spec["structured"], rule=spec["rule"],
            mirrored=spec["mirrored"], evidence_locations=evloc,
            final_verdict=verdict, referenced_regions=regs,
            all9_echo=int(len(regs) == 9), junk_regions=junk,
            reason=(obj.get("reason", "") if ok and isinstance(obj, dict) else ""),
            raw=out, notes=notes,
            prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
            seconds=round(dt, 2), info=info), ensure_ascii=False) + "\n")
        if n % 80 == 0:
            el = time.time() - t0
            rem = el / n * (total - n)
            print(f"  {n}/{total}  {el/60:.1f} min  mean {np.mean(times):.2f}s  "
                  f"now {time.strftime('%H:%M:%S')}  eta "
                  f"{time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()+rem))}",
                  flush=True)
    fh.close()
    meta = dict(phase="trufor_32b_final_capacity", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                gpu_mem=gpu_mem(), nvidia_used_MiB=nvidia_used(),
                started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                payload_sha1=got,
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_32b_final_capacity_frozen.json"),
                    "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], dtype=cfg["model"]["dtype"],
                device_map_resolved=str(probe.device_map_resolved)[:200],
                decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) at {meta['finished']}")


if __name__ == "__main__":
    main()
