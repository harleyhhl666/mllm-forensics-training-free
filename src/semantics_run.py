"""Run the frozen task-semantics experiment: 1472 verdicts.

Four conditions x {fake, real} x 184 sources, all single-image. The structured
evidence payload is a constant; only the decision instruction differs. The model's
own final_verdict is recorded verbatim -- manipulation_present never overrides it.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, yaml, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import extract_json
from freeze_abstraction import abstract
from freeze_fields import strip_fields
from forensic_tools import CELL_NAMES

ORDER = ("D0_baseline", "D1_binary_rule", "D2_decomposition", "D3_presence_first")
CELLSET = set(CELL_NAMES)
EXTENTS = ("none", "localized", "widespread")


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


def parse_present(obj):
    """Strict-ish bool parse; returns (value_or_None, note)."""
    if not isinstance(obj, dict) or "manipulation_present" not in obj:
        return None, "present_missing"
    v = obj["manipulation_present"]
    if isinstance(v, bool):
        return v, None
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return True, "present_coerced"
    if s in ("false", "no", "0"):
        return False, "present_coerced"
    return None, "present_unparsed"


def parse_extent(obj):
    if not isinstance(obj, dict) or "manipulation_extent" not in obj:
        return None, "extent_missing"
    s = str(obj["manipulation_extent"]).strip().lower()
    if s in EXTENTS:
        return s, None
    for e in EXTENTS:
        if e in s:
            return e, "extent_coerced"
    if s in ("localised",):
        return "localized", "extent_coerced"
    return None, "extent_unparsed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated condition ids")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    od = os.path.join(R, "trufor_semantics")
    F = json.load(open(os.path.join(od, "trufor_task_semantics_frozen.json")))

    P, PH, C = F["prompts"], F["prompt_hashes"], F["conditions"]
    for k, v in P.items():
        assert hashlib.sha1(v.encode()).hexdigest() == PH[k]
    print(f"prompts verified: { {k: PH[k][:12] for k in ORDER} }")
    print(f"D0 byte-identical to frozen S5: "
          f"{PH['D0_baseline'] == '22b9ad9a944812ef48723581a35c01bc3eaba7f0'}")
    FIELDS = F["evidence_package"]["fields"]
    print(f"payload fields from frozen doc: {FIELDS}")
    conds = tuple(x for x in ORDER if x in
                  (args.only.split(",") if args.only else ORDER))
    S = F["samples"][:args.n] if args.n else F["samples"]
    print(f"conditions {conds}")
    print(f"sources {len(S)}  planned {len(S)*len(conds)*2} inferences (all 1-image)")

    # HARD GUARD: the rendered D0 prompt must byte-match the field round's S5 cell.
    # The previous run silently included image_manipulation_score in the payload and
    # this check is the only thing that catches such payload drift.
    prev = {}
    for l in open(os.path.join(R, "trufor_fields", "verdicts.jsonl")):
        d = json.loads(l)
        if d["condition"] == "S5_full":
            prev[(d["sample_id"], d["label"])] = d["prompt_sha1"]
    s0 = S[0]
    e0 = s0["evidence"]["fake_own"]
    f0 = abstract(np.load(e0["npz"])["map"].astype(np.float32), e0["score"])
    sc0 = e0["score"]
    chk = P["D0_baseline"].replace("{score}", f"{sc0:.3f}").replace(
        "{structured}", json.dumps(strip_fields(f0, FIELDS)))
    h = hashlib.sha1(chk.encode()).hexdigest()
    want = prev.get((s0["sample_id"], "fake"))
    print(f"D0 rendered-prompt cross-round check: {h[:16]} vs {str(want)[:16]}")
    assert h == want, ("rendered D0 prompt does not match the field round S5 cell; "
                       "payload drift - refusing to run")
    print("cross-round prompt parity OK")

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
            base = Image.open(os.path.join(root, s[label])).convert("RGB")
            full = abstract(np.load(ev["npz"])["map"].astype(np.float32), score)
            payload = json.dumps(strip_fields(full, FIELDS))
            for cond in conds:
                if (s["sample_id"], label, cond) in done:
                    continue
                prompt = P[cond].replace("{score}", f"{score:.3f}") \
                                .replace("{structured}", payload)
                assert "{score}" not in prompt and "{structured}" not in prompt
                ts = time.time()
                out, info = probe.ask(prompt, [base])
                dt = time.time() - ts
                times.append(dt)
                n += 1
                obj, _, ok = extract_json(out)
                notes, verdict = [], None
                if ok and isinstance(obj, dict):
                    fv = str(obj.get("final_verdict", "")).strip().lower()
                    verdict = fv if fv in ("real", "fake") else None
                    if verdict is None:
                        notes.append("bad_verdict")
                else:
                    notes.append("json_parse_failed")
                dec = C[cond]["schema"] == "decomposed"
                present = extent = None
                if dec and ok:
                    present, n1 = parse_present(obj)
                    extent, n2 = parse_extent(obj)
                    notes += [x for x in (n1, n2) if x]
                regs, junk = parse_regions(obj if ok else {})
                if junk:
                    notes.append("unparsed_region_tokens")
                fh.write(json.dumps(dict(
                    sample_id=s["sample_id"], coco_id=s["coco_id"],
                    category=s["category"], mask_type=s["mask_type"],
                    tamper_ratio=s["tamper_ratio"], condition=cond, label=label,
                    ground_truth=label, score_shown=round(score, 6), n_images=1,
                    schema=C[cond]["schema"], decomposed=dec,
                    evidence_locations=[r["grid_location"] for r in full["regions"]],
                    evidence_n_regions=len(full["regions"]),
                    final_verdict=verdict, manipulation_present=present,
                    manipulation_extent=extent, referenced_regions=regs,
                    all9_echo=int(len(regs) == 9), junk_regions=junk,
                    reason=(obj.get("reason", "") if ok and isinstance(obj, dict)
                            else ""),
                    raw=out, notes=notes,
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    seconds=round(dt, 2), info=info), ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            rem = el / max(n, 1) * (len(S) * len(conds) * 2 - n)
            print(f"  {k+1}/{len(S)} sources  {n} inf  {el/60:.1f} min  "
                  f"mean {np.mean(times):.2f}s  now {time.strftime('%H:%M:%S')}  "
                  f"eta {time.strftime('%H:%M:%S', time.localtime(time.time()+rem))}",
                  flush=True)
    fh.close()
    meta = dict(phase="task_semantics", n_inferences=n,
                minutes=round((time.time() - t0) / 60, 1),
                mean_seconds=round(float(np.mean(times)), 2) if times else None,
                peak_GB=round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                protocol_sha1=hashlib.sha1(open(os.path.join(
                    od, "trufor_task_semantics_frozen.json"), "rb").read()).hexdigest(),
                model_path=cfg["model"]["path"], decoding=cfg["model"]["generation"])
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(f"done: {n} inferences in {meta['minutes']} min "
          f"(mean {meta['mean_seconds']}s) peak {meta['peak_GB']} GB "
          f"at {meta['finished']}")


if __name__ == "__main__":
    main()
