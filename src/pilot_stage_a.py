"""Stage A diagnostic pilot: 4 conditions, NO Stage B, NO verdicts.

Conditions (720 inferences total):
  fake_correct : fake image + its OWN ela visualization      (180)
  fake_notool  : fake image only                             (180)
  fake_wrong   : fake image + DONOR image's ela visualization(180)
  real_own     : paired original + its OWN ela visualization (180)

BLINDNESS GUARANTEES (asserted at runtime, not merely intended):
  * fake_correct and fake_wrong use the byte-identical prompt string. The only
    difference is the pixels of the second image. Therefore the tool description
    must contain NO map-derived statistics -- a fixed generic text is used.
  * No GT mask, no candidate bbox, no eligibility fact, no tamper_ratio, and no
    condition name ever enters the prompt. assert_no_leak() re-checks every prompt
    against a blacklist before generation.
  * Stage A prompts forbid a real/fake verdict (unchanged from v1).

Recorded per (sample, condition):
  A evidence_detected
  B mllm_vs_gt        -- predicted cells vs GT cells
  C mllm_vs_cue       -- predicted cells vs THAT CONDITION's cue cells
  D joint_correct     -- B and C together (correct condition only)
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools, CELL_NAMES
from mllm_probe import (QwenVLProbe, extract_json, normalize_stage_a,
                        STAGE_A_NO_TOOLS)

# Fixed tool description: identical for correct and wrong conditions, and free of
# any per-image measurement, so the two prompts cannot be distinguished by text.
ELA_DESC = ("- ela: Error Level Analysis. The image is re-compressed and the "
            "per-pixel difference is amplified and colour-mapped. Brighter areas "
            "indicate a stronger response, which can arise from genuine image "
            "content such as texture and edges, or from local re-encoding "
            "differences. Treat it as one indicator, not a proof.")

STAGE_A_TOOL = """You are assisting a digital image forensics analysis.

You are given:
1. An image under examination.
2. The output of an automated forensic tool applied to that image.

The 3x3 grid names for locations are exactly:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right.

TOOL METADATA:
{tool_meta}

Your task in THIS step is ONLY to report what the forensic tool output shows.
Do NOT decide whether the image is real or fake. Do NOT mention authenticity.
That decision happens in a later, separate step.

Report strictly as JSON with these keys:
{{
  "evidence_detected": true or false,
  "evidence_sources": ["ela"],
  "predicted_regions": [list of 3x3 grid cell names where the anomaly is strongest],
  "evidence_description": "one or two sentences describing the anomaly"
}}
Output JSON only."""

# Tokens that must never appear in a prompt.
LEAK_TOKENS = ("mask", "ground truth", "ground-truth", "tamper", "donor", "wrong",
               "correct cue", "candidate", "eligib", "bbox", "iou", "coco",
               "manipulated region", "fake_", "real_", "shuffle")


def assert_no_leak(prompt):
    low = prompt.lower()
    for t in LEAK_TOKENS:
        if t in low:
            raise AssertionError(f"prompt leak: {t!r} present in Stage A prompt")


def bbox_to_cells(bbox, h, w, grid=3, min_overlap=0.05):
    """Convert a pixel bbox to 3x3 cell indices by area overlap."""
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = []
    for i in range(grid):
        for j in range(grid):
            cy1, cy2, cx1, cx2 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            iw = max(0, min(x2, cx2) - max(x1, cx1))
            ih = max(0, min(y2, cy2) - max(y1, cy1))
            if iw * ih / max((cy2 - cy1) * (cx2 - cx1), 1) >= min_overlap:
                out.append(i * grid + j)
    if not out:                      # fall back to the cell holding the centre
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        i = min(grid - 1, max(0, int(cy / max(h, 1) * grid)))
        j = min(grid - 1, max(0, int(cx / max(w, 1) * grid)))
        out = [i * grid + j]
    return out


def mask_to_cells(maskf, grid, mo):
    h, w = maskf.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = [i * grid + j for i in range(grid) for j in range(grid)
           if maskf[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() >= mo]
    if not out:
        cov = [maskf[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
               for i in range(grid) for j in range(grid)]
        out = [int(np.argmax(cov))]
    return out


def ela_vis(path, tool_cfg, grid):
    tools, rgb = run_tools(path, tool_cfg, grid=grid)
    r = tools["ela"]
    vis = Image.fromarray(cv2.cvtColor(r["vis_bgr"], cv2.COLOR_BGR2RGB))
    return Image.fromarray(rgb), vis, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=0, help="limit samples (smoke)")
    ap.add_argument("--tag", default="pilot_stageA")
    ap.add_argument("--set", default="pilot/pilot_frozen.json",
                    help="frozen set file, relative to out_root")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    grid = cfg["localization"]["grid"]
    mo = cfg["localization"]["gt_cell_min_overlap"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    tool_cfg = cfg["forensic_tools"]

    pilot = json.load(open(os.path.join(out_root, args.set)))
    S = pilot["samples"]
    # The frozen set names its own index when it is not the testing-split one, so a
    # validation-split run cannot silently read testing-split paths.
    idx_path = ds["index"]
    cand = os.path.join(os.path.dirname(os.path.join(out_root, args.set)),
                        "tgif_sp_validation_index.json")
    if os.path.exists(cand):
        idx_path = cand
    print(f"index: {idx_path}")
    idx = {r["pair_id"]: r for r in json.load(open(idx_path))}
    missing = [s["pair_id"] for s in S if s["pair_id"] not in idx]
    if missing:
        sys.exit(f"ABORT: {len(missing)} frozen pair_ids absent from the index "
                 f"(e.g. {missing[:3]}) -- wrong index file for this set")

    # ---- donor integrity, asserted before any inference
    cid = {s["pair_id"]: s["coco_id"] for s in S}
    same = [s["pair_id"] for s in S if cid[s["wrong_tool_donor_pair_id"]] == s["coco_id"]]
    print(f"donor check: {len(S)} samples, same-coco_id donors = {len(same)}")
    if same:
        sys.exit(f"ABORT: {len(same)} donors share coco_id with target: {same[:5]}")
    assert all(s["wrong_tool_donor_pair_id"] != s["pair_id"] for s in S)

    # prompt identity guarantee
    p_tool = STAGE_A_TOOL.format(tool_meta=ELA_DESC)
    assert_no_leak(p_tool)
    assert_no_leak(STAGE_A_NO_TOOLS)
    print(f"prompt leak check: passed  (tool prompt sha1="
          f"{hashlib.sha1(p_tool.encode()).hexdigest()[:12]})")

    if args.n:
        S = S[:args.n]
    out_dir = os.path.join(out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage_a.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line); done.add((d["pair_id"], d["condition"]))
            except Exception:
                pass
    print(f"samples={len(S)}  conditions=4  planned inferences={len(S)*4}  "
          f"already done={len(done)}")

    probe = QwenVLProbe(cfg)
    fh = open(path, "a", buffering=1)
    t0, n_inf = time.time(), 0

    for k, s in enumerate(S):
        pid = s["pair_id"]
        rec = idx[pid]
        need = [c for c in ("fake_correct", "fake_notool", "fake_wrong",
                            "real_own", "real_notool")
                if (pid, c) not in done]
        if not need:
            continue

        fake_path = os.path.join(root, rec["fake"])
        real_path = os.path.join(root, rec["real"])
        mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > mbt
        gt_cells = mask_to_cells(mask.astype(np.float32), grid, mo)
        H, W = mask.shape

        f_img, f_vis, f_r = ela_vis(fake_path, tool_cfg, grid)
        correct_cue_cells = bbox_to_cells(s["candidate_region"], H, W, grid)

        for cond in need:
            if cond == "fake_correct":
                imgs, prompt = [f_img, f_vis], p_tool
                cue_cells = correct_cue_cells
                cue_bbox = s["candidate_region"]
            elif cond == "fake_notool":
                imgs, prompt = [f_img], STAGE_A_NO_TOOLS
                cue_cells, cue_bbox = [], None
            elif cond == "fake_wrong":
                dpid = s["wrong_tool_donor_pair_id"]
                d_rec = idx[dpid]
                _, d_vis, _ = ela_vis(os.path.join(root, d_rec["fake"]), tool_cfg, grid)
                d_vis = d_vis.resize(f_img.size, Image.BILINEAR)
                imgs, prompt = [f_img, d_vis], p_tool      # identical prompt text
                dsamp = next(x for x in pilot["samples"] if x["pair_id"] == dpid)
                dh, dw = Image.open(os.path.join(root, d_rec["mask"])).size[::-1]
                # donor cue mapped into the donor's own frame, then rescaled
                db = dsamp["candidate_region"]
                sx, sy = W / dw, H / dh
                cue_bbox = [int(db[0]*sx), int(db[1]*sy), int(db[2]*sx), int(db[3]*sy)]
                cue_cells = bbox_to_cells(cue_bbox, H, W, grid)
            else:  # real_own / real_notool
                r_img, r_vis, r_r = ela_vis(real_path, tool_cfg, grid)
                if cond == "real_own":
                    imgs, prompt = [r_img, r_vis], p_tool
                else:                                  # real_notool
                    imgs, prompt = [r_img], STAGE_A_NO_TOOLS
                cue_cells, cue_bbox = [], None

            assert_no_leak(prompt)
            raw, info = probe.ask(prompt, imgs)
            n_inf += 1
            obj, _, ok = extract_json(raw)
            valid_src = (["visual_inspection"]
                         if cond in ("fake_notool", "real_notool") else ["ela"])
            sa, notes = normalize_stage_a(obj, valid_src) if ok else (None, ["json_parse_failed"])

            pred = [CELL_NAMES.index(x) for x in (sa["predicted_regions"] if sa else [])]
            ack = bool(sa and sa["evidence_detected"] is True)
            is_fake = cond not in ("real_own", "real_notool")
            vs_gt = bool(is_fake and pred and set(pred) & set(gt_cells))
            vs_cue = bool(cue_cells and pred and set(pred) & set(cue_cells))

            row = dict(
                pair_id=pid, coco_id=s["coco_id"], category=s["category"],
                condition=cond, ground_truth=("fake" if is_fake else "real"),
                tamper_ratio=s["tamper_ratio"], n_gt_cells=len(gt_cells),
                gt_cells=[CELL_NAMES[c] for c in gt_cells] if is_fake else [],
                cue_bbox=cue_bbox,
                cue_cells=[CELL_NAMES[c] for c in cue_cells],
                predicted_cells=[CELL_NAMES[c] for c in pred],
                evidence_detected=ack,                       # A
                mllm_vs_gt=vs_gt,                            # B
                mllm_vs_cue=vs_cue,                          # C
                joint_correct=bool(vs_gt and vs_cue),        # D
                stage_a=sa, stage_a_raw=raw, stage_a_notes=notes, info=info,
                donor_pair_id=(s["wrong_tool_donor_pair_id"]
                               if cond == "fake_wrong" else None),
                prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                n_images=len(imgs),
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(S)} samples  {n_inf} inferences  {el/60:.1f} min "
                  f"({el/max(n_inf,1):.1f} s/inference)", flush=True)
    fh.close()
    el = time.time() - t0
    print(f"done: {n_inf} inferences in {el/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
