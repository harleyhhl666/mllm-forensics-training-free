"""Exploratory Stage B on the Primary set. Reads frozen Stage-A records; never
recomputes or edits them.

Per condition the model sees only: that condition's image, that condition's forensic
visualization (if any), and that condition's OWN frozen Stage-A summary. It never
sees GT, cue correctness, B2/B3 membership, localization outcomes, or any evaluator
field. correct_ela and wrong_donor_ela share a byte-identical prompt (asserted).

The prompt contains no language urging the model to trust the tool.

Decoding is the frozen deterministic config. A separate --stochastic mode exists for
the stability probe and is only used on reversal candidates, with fixed seeds.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools, CELL_NAMES
from mllm_probe import QwenVLProbe, extract_json, normalize_stage_b

# Neutral tool description, identical to Stage A's, no trust-inducing wording.
ELA_DESC = ("- ela: Error Level Analysis. The image is re-compressed and the "
            "per-pixel difference is amplified and colour-mapped. Brighter areas "
            "indicate a stronger response, which can arise from genuine image "
            "content such as texture and edges, or from local re-encoding "
            "differences. Treat it as one indicator, not a proof.")

STAGE_B_TOOL = """You are completing a digital image forensics analysis.

You are given:
1. The image under examination.
2. The output of an automated forensic tool applied to that image.
3. The observation summary produced in the previous analysis step.

TOOL METADATA:
{tool_meta}

OBSERVATION SUMMARY FROM THE PREVIOUS STEP (treat it as given; do not revise it):
{stage_a_summary}

Now make the final determination for this image.

Report strictly as JSON with these keys:
{{
  "final_verdict": "real" or "fake",
  "reason": "brief explanation"
}}
Output JSON only."""

STAGE_B_NOTOOL = """You are completing a digital image forensics analysis.

You are given:
1. The image under examination.
2. The observation summary produced in the previous analysis step.

OBSERVATION SUMMARY FROM THE PREVIOUS STEP (treat it as given; do not revise it):
{stage_a_summary}

Now make the final determination for this image.

Report strictly as JSON with these keys:
{{
  "final_verdict": "real" or "fake",
  "reason": "brief explanation"
}}
Output JSON only."""

import re

# Substring matching would false-positive ("iou" inside "previous"), so tokens are
# matched on word boundaries. Multi-word phrases are matched as phrases.
LEAK_TOKENS = ("mask", "ground truth", "ground-truth", "tamper", "tampered",
               "donor", "wrong", "candidate", "eligible", "eligibility", "bbox",
               "iou", "coco", "manipulated region", "b2", "b3", "grounded",
               "shuffle", "shuffled", "correct cue", "mismatch", "mismatched",
               "trust the", "reliable evidence", "prioritize")
_LEAK_RE = re.compile(
    "|".join(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b" for t in LEAK_TOKENS),
    re.I)


def assert_no_leak(p):
    m = _LEAK_RE.search(p)
    if m:
        raise AssertionError(f"Stage-B prompt leak: {m.group(0)!r}")


def summary_of(sa):
    """Verbatim Stage-A summary, restricted to the four reported fields."""
    if not sa:
        return "(the previous step produced no parseable summary)"
    return json.dumps({k: sa[k] for k in ("evidence_detected", "evidence_sources",
                                          "predicted_regions", "evidence_description")},
                      ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stage-a-tag", default="validation_primary_stageA")
    ap.add_argument("--set", default="validation/validation_primary_frozen.json")
    ap.add_argument("--tag", default="validation_primary_stageB")
    ap.add_argument("--only-pairs", default=None,
                    help="path to json list of pair_ids (stability probe)")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seeds", default="101,202,303,404,505")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    ds = cfg["datasets"]["tgif_sp"]
    root, grid = ds["root"], cfg["localization"]["grid"]
    tool_cfg = cfg["forensic_tools"]

    frozen = json.load(open(os.path.join(out_root, args.set)))
    idx_path = ds["index"]
    cand = os.path.join(os.path.dirname(os.path.join(out_root, args.set)),
                        "tgif_sp_validation_index.json")
    if os.path.exists(cand):
        idx_path = cand
    print(f"index: {idx_path}")
    idx = {r["pair_id"]: r for r in json.load(open(idx_path))}
    A = {}
    for line in open(os.path.join(out_root, args.stage_a_tag, "stage_a.jsonl")):
        d = json.loads(line)
        A[(d["pair_id"], d["condition"])] = d
    S = {s["pair_id"]: s for s in frozen["samples"]}

    pairs = sorted(S)
    if args.only_pairs:
        want = set(json.load(open(args.only_pairs)))
        pairs = [p for p in pairs if p in want]

    CONDS = ("fake_correct", "fake_notool", "fake_wrong", "real_own", "real_notool")
    p_tool = STAGE_B_TOOL.format(tool_meta=ELA_DESC, stage_a_summary="X")
    assert_no_leak(p_tool)
    assert_no_leak(STAGE_B_NOTOOL.format(stage_a_summary="X"))
    tmpl_sha = hashlib.sha1(STAGE_B_TOOL.encode()).hexdigest()
    print(f"Stage-B template sha1={tmpl_sha[:12]}  samples={len(pairs)}  "
          f"conditions={len(CONDS)}")
    print(f"decoding: {'STOCHASTIC probe' if args.stochastic else 'frozen deterministic'}")

    seeds = [int(x) for x in args.seeds.split(",")] if args.stochastic else [None]
    out_dir = os.path.join(out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage_b.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
                done.add((d["pair_id"], d["condition"], d.get("seed")))
            except Exception:
                pass
    print(f"already recorded: {len(done)}")

    probe = QwenVLProbe(cfg)
    stab = cfg["stability"]
    gen_ov = None
    if args.stochastic:
        gen_ov = dict(do_sample=True, temperature=stab["temperature"],
                      top_p=stab["top_p"])

    fh = open(path, "a", buffering=1)
    t0, n = time.time(), 0
    for k, pid in enumerate(pairs):
        s, rec = S[pid], idx[pid]
        f_img = None
        for cond in CONDS:
            a = A.get((pid, cond))
            if a is None:
                continue
            for sd in seeds:
                if (pid, cond, sd) in done:
                    continue
                if cond in ("real_own", "real_notool"):
                    tools, rgb = run_tools(os.path.join(root, rec["real"]),
                                           tool_cfg, grid=grid)
                    r_img = Image.fromarray(rgb)
                    if cond == "real_own":
                        r_vis = Image.fromarray(cv2.cvtColor(tools["ela"]["vis_bgr"],
                                                             cv2.COLOR_BGR2RGB))
                        imgs = [r_img, r_vis]
                        prompt = STAGE_B_TOOL.format(
                            tool_meta=ELA_DESC,
                            stage_a_summary=summary_of(a["stage_a"]))
                    else:
                        imgs = [r_img]
                        prompt = STAGE_B_NOTOOL.format(
                            stage_a_summary=summary_of(a["stage_a"]))
                else:
                    if f_img is None:
                        tools, rgb = run_tools(os.path.join(root, rec["fake"]),
                                               tool_cfg, grid=grid)
                        f_img = Image.fromarray(rgb)
                        f_vis = Image.fromarray(cv2.cvtColor(tools["ela"]["vis_bgr"],
                                                            cv2.COLOR_BGR2RGB))
                    if cond == "fake_notool":
                        imgs = [f_img]
                        prompt = STAGE_B_NOTOOL.format(
                            stage_a_summary=summary_of(a["stage_a"]))
                    elif cond == "fake_correct":
                        imgs = [f_img, f_vis]
                        prompt = STAGE_B_TOOL.format(
                            tool_meta=ELA_DESC,
                            stage_a_summary=summary_of(a["stage_a"]))
                    else:  # fake_wrong -- donor visualization, identical prompt text
                        dpid = s["wrong_tool_donor_pair_id"]
                        dt, _ = run_tools(os.path.join(root, idx[dpid]["fake"]),
                                          tool_cfg, grid=grid)
                        dv = Image.fromarray(cv2.cvtColor(dt["ela"]["vis_bgr"],
                                                          cv2.COLOR_BGR2RGB))
                        imgs = [f_img, dv.resize(f_img.size, Image.BILINEAR)]
                        prompt = STAGE_B_TOOL.format(
                            tool_meta=ELA_DESC,
                            stage_a_summary=summary_of(a["stage_a"]))

                assert_no_leak(prompt)
                if sd is not None:
                    import torch
                    torch.manual_seed(sd)
                raw, info = probe.ask(prompt, imgs, gen_override=gen_ov)
                n += 1
                obj, _, ok = extract_json(raw)
                sb, notes = normalize_stage_b(obj) if ok else (None, ["json_parse_failed"])
                fh.write(json.dumps(dict(
                    pair_id=pid, coco_id=s["coco_id"], category=s["category"],
                    condition=cond,
                    ground_truth=("real" if cond in ("real_own", "real_notool")
                                  else "fake"),
                    tamper_ratio=s["tamper_ratio"], seed=sd,
                    stochastic=bool(args.stochastic),
                    final_verdict=(sb["final_verdict"] if sb else None),
                    reason=(sb["reason"] if sb else ""),
                    stage_b_raw=raw, stage_b_notes=notes,
                    stage_a_summary_given=summary_of(a["stage_a"]),
                    stage_a_mllm_vs_gt=a["mllm_vs_gt"],
                    stage_a_mllm_vs_cue=a["mllm_vs_cue"],
                    stage_a_joint_correct=a["joint_correct"],
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    n_images=len(imgs), info=info,
                ), ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(pairs)} samples  {n} inferences  {el/60:.1f} min",
                  flush=True)
    fh.close()
    print(f"done: {n} inferences in {(time.time()-t0)/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
