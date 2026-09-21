"""Phase-1 main run: Gates 0-4 over a dataset, with Controls 1-3.

Per-sample record is written to a JSONL file IMMEDIATELY after that sample
finishes, so a crash or timeout never loses completed work and Stage A can never
be retroactively edited by Stage B.

Order of operations per sample (strict, so failure stages stay separable):
  1. run forensic tools           -> Gate 0 (validity)
  2. blind anomaly score + frozen threshold -> Gate 1 (objective evidence)
  3. Stage A with tools           -> Gate 2 (acknowledgment), Gate 3 (localization)
  4. Stage A without tools        -> Control 1 (no-tool localization)
  5. Stage B, given Stage A       -> Gate 4 (verdict reversal)
  6. if Gates 0-3 pass: repeat Stage B n times under sampling -> Control 3
"""
import argparse, json, os, sys, random, time
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools, CELL_NAMES
from tool_diagnostic import gt_cells
from mllm_probe import (QwenVLProbe, extract_json, normalize_stage_a, normalize_stage_b,
                        STAGE_A_WITH_TOOLS, STAGE_A_NO_TOOLS, STAGE_B)


def tool_meta_text(tools, frozen):
    """Human-readable tool metadata shown to the model. Includes validity so the
    model is never asked to trust a tool we ourselves marked invalid."""
    lines = []
    for name, r in tools.items():
        s = r["stats"]
        if not r["valid"]:
            lines.append(f"- {name}: OUTPUT NOT USABLE for this image "
                         f"(reason: {', '.join(r['invalid_reasons'])}). Ignore it.")
            continue
        lines.append(
            f"- {name}: valid. output variance={s['variance']:.4f}, "
            f"dynamic range={s['dynamic_range']:.3f}, "
            f"near-constant ratio={s['near_constant_ratio']:.3f}. "
            f"Brighter areas in this tool's visualization indicate stronger response.")
    return "\n".join(lines) if lines else "- no tool output available"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dataset", default="cocoglide")
    ap.add_argument("--split", default="test", choices=["test", "calibration", "smoke"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = cfg["datasets"][args.dataset]
    root = ds["root"]
    grid = cfg["localization"]["grid"]
    mo = cfg["localization"]["gt_cell_min_overlap"]
    tool_cfg = cfg["forensic_tools"]

    # ---- frozen Gate-1 decision is MANDATORY: refuse to invent a threshold
    frozen_path = os.path.join(cfg["experiment"]["out_root"], "calibration", "frozen_gate1.json")
    if not os.path.exists(frozen_path):
        sys.exit(f"REFUSING TO RUN: no frozen Gate-1 decision at {frozen_path}.\n"
                 f"Run calibrate.py, inspect the comparison, and write the frozen file "
                 f"explicitly. Gate 1 must not be tuned on the test split.")
    frozen = json.load(open(frozen_path))
    print(f"frozen Gate-1: tool={frozen['tool']} scorer={frozen['scorer']} "
          f"threshold={frozen['threshold']:.4f} (source: {frozen['derived_from']})")

    index_path = os.path.join(cfg["experiment"]["out_root"], "phase0",
                              f"{args.dataset}_index.json")
    idx = {r["pair_id"]: r for r in json.load(open(index_path))}
    splits = json.load(open(os.path.join(cfg["experiment"]["out_root"],
                                         "calibration", "splits.json")))
    if args.split == "smoke":
        ids = splits["calibration"][:cfg["sampling"]["smoke_n"]]
    else:
        ids = splits[args.split]
    if args.n:
        ids = ids[:args.n]
    print(f"split={args.split}  n_pairs={len(ids)}")

    out_dir = os.path.join(cfg["experiment"]["out_root"], args.tag)
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "samples.jsonl")
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try: done.add(json.loads(line)["image_id"])
            except Exception: pass
        print(f"resuming: {len(done)} samples already recorded")
    json.dump(cfg, open(os.path.join(out_dir, "config_snapshot.json"), "w"), indent=1)

    probe = QwenVLProbe(cfg)
    stab = cfg["stability"]
    fh = open(jsonl, "a", buffering=1)
    t0 = time.time()

    for n_done, pid in enumerate(ids):
        rec = idx[pid]
        for gt_label, key in (("fake", "fake"), ("real", "real")):
            image_id = f"{pid}::{gt_label}"
            if image_id in done:
                continue
            img_path = os.path.join(root, rec[key])
            tools, rgb = run_tools(img_path, tool_cfg, grid=grid)
            pil = Image.fromarray(rgb)

            # ---- Gate 0
            tool_validity = {k: bool(v["valid"]) for k, v in tools.items()}
            gate0 = any(tool_validity.values())

            # ---- Gate 1 (frozen scorer + frozen threshold, blind)
            ft = frozen["tool"]
            g1_score = None
            gate1 = False
            if tools.get(ft, {}).get("valid"):
                from calibrate import SCORERS
                g1_score, peak = SCORERS[frozen["scorer"]](tools[ft]["map01"], grid)
                gate1 = bool(g1_score > frozen["threshold"])
            strength = None
            if g1_score is not None and "strength_cuts" in frozen:
                c = frozen["strength_cuts"]
                strength = ("weak" if g1_score < c["weak_medium"]
                            else "medium" if g1_score < c["medium_strong"] else "strong")

            # ---- GT regions (evaluation only)
            if gt_label == "fake":
                mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > 127
                gcells = gt_cells(mask.astype(np.float32), grid, mo)
                tamper_ratio = rec["tamper_ratio"]
            else:
                gcells, tamper_ratio = [], 0.0
            gt_region_names = [CELL_NAMES[c] for c in gcells]

            valid_names = [k for k, v in tool_validity.items() if v]
            meta = tool_meta_text(tools, frozen)
            vis_imgs = [pil] + [Image.fromarray(cv2.cvtColor(tools[k]["vis_bgr"],
                                                             cv2.COLOR_BGR2RGB))
                                for k in valid_names]

            # ---- Stage A, with tools
            pa = STAGE_A_WITH_TOOLS.format(tool_meta=meta,
                                           tool_names=", ".join(valid_names) or "none")
            raw_a, info_a = probe.ask(pa, vis_imgs)
            obj_a, _, ok_a = extract_json(raw_a)
            sa, notes_a = normalize_stage_a(obj_a, valid_names) if ok_a else (None, ["json_parse_failed"])

            # ---- Control 1: Stage A, no tools
            raw_a0, info_a0 = probe.ask(STAGE_A_NO_TOOLS, [pil])
            obj_a0, _, ok_a0 = extract_json(raw_a0)
            sa0, notes_a0 = normalize_stage_a(obj_a0, ["visual_inspection"]) if ok_a0 \
                else (None, ["json_parse_failed"])

            # ---- Gate 2 / Gate 3
            gate2 = bool(sa and sa["evidence_detected"] is True)
            pred_cells = [CELL_NAMES.index(r) for r in (sa["predicted_regions"] if sa else [])]
            # Gate 3 is only meaningful when the GT mask does not blanket the grid
            max_gt = cfg["localization"]["max_gt_cells_for_gate3"]
            loc_uninformative = bool(gt_label == "fake" and len(gcells) > max_gt)
            gate3 = bool(gate2 and gcells and not loc_uninformative
                         and set(pred_cells) & set(gcells))
            pred_cells0 = [CELL_NAMES.index(r) for r in (sa0["predicted_regions"] if sa0 else [])]
            loc_correct_no_tool = bool(gcells and not loc_uninformative
                                       and set(pred_cells0) & set(gcells))

            # ---- Stage B (Stage A record is already fixed above)
            sa_summary = json.dumps({k: sa[k] for k in
                                     ("evidence_detected", "evidence_sources",
                                      "predicted_regions", "evidence_description")},
                                    ensure_ascii=False) if sa else "(previous step produced no parseable summary)"
            pb = STAGE_B.format(tool_meta=meta, stage_a_summary=sa_summary)
            raw_b, info_b = probe.ask(pb, vis_imgs)
            obj_b, _, ok_b = extract_json(raw_b)
            sb, notes_b = normalize_stage_b(obj_b) if ok_b else (None, ["json_parse_failed"])

            gates_0_3 = bool(gate0 and gate1 and gate2 and gate3)
            rejection = bool(gates_0_3 and gt_label == "fake"
                             and sb and sb["final_verdict"] == "real")

            # ---- Control 3: repeated Stage B on key samples only
            repeats = []
            if gates_0_3:
                for _ in range(stab["n_repeats"]):
                    rb, _ = probe.ask(pb, vis_imgs,
                                      gen_override=dict(do_sample=stab["do_sample"],
                                                        temperature=stab["temperature"],
                                                        top_p=stab["top_p"]))
                    ob, _, okb = extract_json(rb)
                    nb, _ = normalize_stage_b(ob) if okb else (None, None)
                    repeats.append(nb["final_verdict"] if nb else None)

            out = dict(
                image_id=image_id, dataset=args.dataset, ground_truth=gt_label,
                image_path=rec[key], tamper_ratio=tamper_ratio,
                tool_validity=tool_validity,
                tool_stats={k: v["stats"] for k, v in tools.items()},
                tool_invalid_reasons={k: v["invalid_reasons"] for k, v in tools.items()},
                gate1_tool=ft, gate1_score=g1_score,
                gate1_threshold=frozen["threshold"], evidence_strength=strength,
                stage_a=sa, stage_a_raw=raw_a, stage_a_notes=notes_a, stage_a_info=info_a,
                stage_a_no_tool=sa0, stage_a_no_tool_raw=raw_a0, stage_a_no_tool_notes=notes_a0,
                gt_regions=gt_region_names, n_gt_cells=len(gcells),
                localization_gate_uninformative=loc_uninformative,
                localization_correct=gate3, localization_correct_no_tool=loc_correct_no_tool,
                stage_b=sb, stage_b_raw=raw_b, stage_b_notes=notes_b,
                gate0=gate0, gate1=gate1, gate2=gate2, gate3=gate3,
                gates_0_3_pass=gates_0_3,
                evidence_rejection_case=rejection,
                stage_b_repeats=repeats,
            )
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        if (n_done + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  {n_done+1}/{len(ids)} pairs  {el/60:.1f} min  "
                  f"({el/max(n_done+1,1):.1f} s/pair)", flush=True)
    fh.close()
    print(f"done -> {jsonl}")


if __name__ == "__main__":
    main()
