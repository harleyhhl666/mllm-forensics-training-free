"""Run cell A1 of the Score x Task-Semantics 2x2: score-only + binary rule, 368 runs.

A0/A2/A3 are never re-run. A startup assertion re-derives A1 from the frozen A0
template and checks the rule-stripped prompt reproduces A0 byte-exactly, so the
rule insertion cannot drift from the audited construction.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json
from forensic_tools import CELL_NAMES

CELLSET = set(CELL_NAMES)
COND = "A1_score_rule"


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
    od = os.path.join(R, "trufor_2x2")
    F = json.load(open(os.path.join(od, "trufor_score_semantics_2x2_frozen.json")))

    TPL = F["prompts"][COND]
    rule = F["prompt_construction"]["rule_text"]
    assert hashlib.sha1(TPL.encode()).hexdigest() == F["prompt_hashes"][COND]
    # A1 stripped of the rule must equal A0 exactly
    assert TPL.replace(rule, "") == F["prompts"]["A0_score_only"]
    assert "{structured}" not in TPL, "A1 must carry no structured payload"
    print(f"A1 template verified sha1 {F['prompt_hashes'][COND][:16]}")
    print(f"  rule-stripped A1 == A0 byte-exactly: True")
    print(f"  A1 carries no structured payload: True")
    S = F["samples"][:args.n] if args.n else F["samples"]
    print(f"sources {len(S)}  planned {len(S)*2} inferences (single-image)")

    path = os.path.join(od, "verdicts_A1.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                d = json.loads(l)
                done.add((d["sample_id"], d["label"]))
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
            if (s["sample_id"], label) in done:
                continue
            ev = s["evidence"][f"{label}_own"]
            score = ev["score"]
            base = Image.open(os.path.join(root, s[label])).convert("RGB")
            prompt = TPL.replace("{score}", f"{score:.3f}")
            assert "{score}" not in prompt
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
                tamper_ratio=s["tamper_ratio"], condition=COND, label=label,
                ground_truth=label, score_shown=round(score, 6), n_images=1,
                structured=False, rule=True,
                final_verdict=verdict, referenced_regions=regs,
                all9_echo=int(len(regs) == 9), junk_regions=junk,
                reason=(obj.get("reason", "") if ok and isinstance(obj, dict) else ""),
                raw=out, notes=notes,
                prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                seconds=round(dt, 2), info=info), ensure_ascii=False) + "\n")
        if (k + 1) % 40 == 0:
            el = time.time() - t0
            rem = el / max(n, 1) * (len(S) * 2 - n)
            print(f"  {k+1}/{len(S)} sources  {n} inf  {el/60:.1f} min  "
                  f"mean {np.mean(times):.2f}s  now {time.strftime('%H:%M:%S')}  "
                  f"eta {time.strftime('%H:%M:%S', time.localtime(time.time()+rem))}",
                  flush=True)
    fh.close()
    meta = dict(phase="score_semantics_2x2_A1", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_score_semantics_2x2_frozen.json"),
                    "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta_A1.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) at {meta['finished']}")


if __name__ == "__main__":
    main()
