"""Run the frozen framing / region-presence control: 2208 verdicts.

Six conditions x {fake, real} x 184 sources, all single-image. Payload builders are
imported from the frozen freezer so they cannot drift.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json
from freeze_abstraction import abstract
from freeze_framing import build_payload, PAYLOAD
from forensic_tools import CELL_NAMES

ORDER = ("P0_score_only", "P1_framing_only", "P2_empty_regions", "P3_placeholder",
         "P4_true_location", "P5_neutral_meta")
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
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    od = os.path.join(R, "trufor_framing")
    F = json.load(open(os.path.join(od, "trufor_structured_framing_frozen.json")))

    TPL = F["prompts"]
    assert hashlib.sha1(TPL["score_only"].encode()).hexdigest() == \
        F["prompt_hashes"]["score_only"]
    assert hashlib.sha1(TPL["structured"].encode()).hexdigest() == \
        F["prompt_hashes"]["structured"]
    print(f"prompt templates verified: score_only "
          f"{F['prompt_hashes']['score_only'][:16]}  structured "
          f"{F['prompt_hashes']['structured'][:16]}")
    for k in ORDER:
        assert PAYLOAD[k]["kind"] == F["conditions"][k]["payload_kind"]
        assert PAYLOAD[k]["prompt"] == F["conditions"][k]["prompt"]
    print(f"payload kinds match freeze: "
          f"{ {k: PAYLOAD[k]['kind'] for k in ORDER} }")
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
    print(f"start {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
    for k, s in enumerate(S):
        for label in ("fake", "real"):
            ev = s["evidence"][f"{label}_own"]
            score = ev["score"]
            base, full = None, None
            for cond in ORDER:
                if (s["sample_id"], label, cond) in done:
                    continue
                spec = PAYLOAD[cond]
                if base is None:
                    base = Image.open(os.path.join(root, s[label])).convert("RGB")
                if full is None:
                    full = abstract(
                        np.load(ev["npz"])["map"].astype(np.float32), score)
                pl = build_payload(spec["kind"], full)
                tpl = TPL["score_only"] if spec["prompt"] == "score_only" \
                    else TPL["structured"]
                prompt = tpl.replace("{score}", f"{score:.3f}")
                if "{structured}" in prompt:
                    prompt = prompt.replace(
                        "{structured}", "" if pl is None else json.dumps(pl))
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
                    payload_kind=spec["kind"], prompt_template=spec["prompt"],
                    payload=pl,
                    payload_chars=(0 if pl is None else len(json.dumps(pl))),
                    evidence_n_regions=len(full["regions"]),
                    final_verdict=verdict, referenced_regions=regs,
                    all9_echo=int(len(regs) == 9), junk_regions=junk,
                    reason=(obj.get("reason", "") if ok and isinstance(obj, dict)
                            else ""),
                    raw=out, notes=notes,
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    seconds=round(dt, 2), info=info), ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            rem = el / max(n, 1) * (len(S) * len(ORDER) * 2 - n)
            print(f"  {k+1}/{len(S)} sources  {n} inf  {el/60:.1f} min  "
                  f"mean {np.mean(times):.2f}s  now "
                  f"{time.strftime('%H:%M:%S')}  eta "
                  f"{time.strftime('%H:%M:%S', time.localtime(time.time()+rem))}",
                  flush=True)
    fh.close()
    meta = dict(phase="structured_framing_control", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                started=time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(t0)),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_structured_framing_frozen.json"),
                    "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {meta['peak_GB']} GB "
          f"at {meta['finished']}")


if __name__ == "__main__":
    main()
