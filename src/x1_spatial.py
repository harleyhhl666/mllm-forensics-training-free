"""Phase X1: retrospective explainability analysis. NO new MLLM inference.

Builds three spatial objects per fake sample -- GT regions, evidence regions,
explanation regions -- and measures the three directional relations:

  Evidence -> GT        does TruFor's structured evidence point at real tampering?
  Explanation -> Evidence   does the model follow the evidence it was given?
  Explanation -> GT      NEW primary metric: is the stated location actually right?

Then repeats for the mirrored condition, compares 7B vs 32B separating citation
COVERAGE from citation CORRECTNESS, and compares raw map vs structured evidence.
All numbers come from verdict files already on disk.
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np, yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import CELL_NAMES

NBOOT, SEED = 10000, 20260918
MIRROR = {"top-left": "top-right", "top-center": "top-center",
          "top-right": "top-left", "center-left": "center-right",
          "center": "center", "center-right": "center-left",
          "bottom-left": "bottom-right", "bottom-center": "bottom-center",
          "bottom-right": "bottom-left"}


def boot(v, seed=SEED):
    a = np.asarray([x for x in v if x is not None], float)
    if not len(a):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (NBOOT, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def gt_cells(mask_path, thr=127):
    """GT mask -> 3x3 grid facts. Uses the frozen >127 binarization."""
    im = np.array(Image.open(mask_path).convert("L"))
    m = im > thr
    H, W = m.shape
    tot = m.sum()
    if tot == 0:
        return dict(cells=[], primary=None, centroid=None, area_ratio=0.0,
                    per_cell={})
    ys, xs = np.nonzero(m)
    cx, cy = float(xs.mean()) / W, float(ys.mean()) / H
    per = {}
    for r in range(3):
        for c in range(3):
            sub = m[r * H // 3:(r + 1) * H // 3, c * W // 3:(c + 1) * W // 3]
            if sub.sum():
                per[CELL_NAMES[r * 3 + c]] = float(sub.sum() / tot)
    cells = sorted(per, key=lambda k: -per[k])
    return dict(cells=[c for c in cells if per[c] >= 0.05],
                all_cells=cells, primary=cells[0] if cells else None,
                centroid=[round(cx, 3), round(cy, 3)],
                area_ratio=float(tot / (H * W)), per_cell=per)


def load(path, cond, label=None):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path):
        d = json.loads(l)
        if d["condition"] != cond:
            continue
        if label and d["label"] != label:
            continue
        out[(d["sample_id"], d["label"])] = d
    return out


def rel(pred, truth):
    """Set relations between two cell sets."""
    p, t = set(pred), set(truth)
    if not p:
        return dict(cited=0, hit=None, exact=None, prec=None, rec=None)
    return dict(cited=1, hit=int(bool(p & t)), exact=int(p == t),
                prec=len(p & t) / len(p), rec=(len(p & t) / len(t)) if t else None)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    thr = cfg["datasets"].get("mask_binarize_threshold", 127)

    conf = json.load(open(os.path.join(
        R, "trufor_conflict", "trufor_score_map_conflict_frozen.json")))
    MASK = {s["sample_id"]: os.path.join(root, s["mask"]) for s in conf["samples"]}
    ab = json.load(open(os.path.join(
        R, "trufor_abstraction", "trufor_evidence_abstraction_frozen.json")))
    ELIG = {k: set(v) for k, v in ab["e4_eligible_ids"].items()}

    print("PHASE X1 — RETROSPECTIVE EXPLAINABILITY ANALYSIS (no new inference)")
    print(f"mask binarization threshold >{thr} (frozen preprocessing choice)")

    # ---------------- A1: GT spatial objects ----------------
    print("\n" + "=" * 96)
    print("A1. GT SPATIAL OBJECTS (fake samples)")
    GT = {}
    for sid, mp in MASK.items():
        GT[sid] = gt_cells(mp, thr)
    ids = sorted(GT)
    nc = [len(GT[s]["cells"]) for s in ids]
    print(f"  fake samples {len(ids)}")
    print(f"  GT active cells (>=5% of mask): mean {np.mean(nc):.2f}  "
          f"dist {dict(sorted(Counter(nc).items()))}")
    print(f"  GT area_ratio median {np.median([GT[s]['area_ratio'] for s in ids]):.4f}")
    print(f"  GT primary cell distribution: "
          f"{dict(Counter(GT[s]['primary'] for s in ids).most_common(5))}")
    print(f"  NOTE: a GT mask spanning many cells makes any 'hit' metric easier;")
    print(f"  {sum(1 for s in ids if len(GT[s]['cells'])>=4)}/{len(ids)} samples span "
          f">=4 cells, so precision/recall are reported alongside hit rates.")

    # ---------------- conditions on disk ----------------
    C = {}
    C["7B_structured"] = load(os.path.join(R, "trufor_semantics", "verdicts.jsonl"),
                              "D0_baseline")
    C["7B_struct_rule"] = load(os.path.join(R, "trufor_semantics", "verdicts.jsonl"),
                               "D1_binary_rule")
    C["7B_mirrored"] = load(os.path.join(R, "trufor_abstraction", "verdicts.jsonl"),
                            "E4_shifted_struct")
    C["7B_struct_E2"] = load(os.path.join(R, "trufor_abstraction", "verdicts.jsonl"),
                             "E2_structured")
    C["7B_rawmap"] = load(os.path.join(R, "trufor_abstraction", "verdicts.jsonl"),
                          "E1_raw_map")
    C["7B_score_only"] = load(os.path.join(R, "trufor_fields", "verdicts.jsonl"),
                              "S0_score_only")
    C["32B_structured"] = load(os.path.join(R, "trufor_32b_final", "verdicts.jsonl"),
                               "B2_structured")
    C["32B_struct_rule"] = load(os.path.join(R, "trufor_32b_final", "verdicts.jsonl"),
                                "B3_structured_rule")
    C["32B_mirrored"] = load(os.path.join(R, "trufor_32b_final", "verdicts.jsonl"),
                             "B4_mirrored")
    C["32B_score_only"] = load(os.path.join(R, "trufor_32b_final", "verdicts.jsonl"),
                               "B0_score_only")
    print("\n" + "=" * 96)
    print("CONDITIONS LOADED FROM DISK")
    for k, v in C.items():
        nf = sum(1 for (p, l) in v if l == "fake")
        print(f"  {k:<18} cells {len(v):<5} fake {nf}")

    # ---------------- A2.1 Evidence -> GT ----------------
    print("\n" + "=" * 96)
    print("A2.1  EVIDENCE -> GT   (does TruFor structured evidence point at real")
    print("      tampering? independent of any MLLM)")
    ev_src = C["7B_structured"]
    rows = []
    for s in ids:
        r = ev_src.get((s, "fake"))
        if not r:
            continue
        ev = r["evidence_locations"] or []
        g = GT[s]
        rel_all = rel(ev, g["cells"])
        rows.append(dict(sid=s, ev=ev, top1=(ev[0] if ev else None),
                         gt_primary=g["primary"], **rel_all))
    print(f"  n={len(rows)}")
    print(f"  any-cell hit        {boot([r['hit'] for r in rows])[0]:.3f}")
    t1 = [int(r["top1"] in GT[r["sid"]]["cells"]) if r["top1"] else None
          for r in rows]
    print(f"  top-1 cell hit      {boot(t1)[0]:.3f}")
    ep = [int(r["top1"] == r["gt_primary"]) if r["top1"] else None for r in rows]
    print(f"  top-1 == GT primary {boot(ep)[0]:.3f}")
    print(f"  exact cell agreement {boot([r['exact'] for r in rows])[0]:.3f}")
    print(f"  multi-cell precision {boot([r['prec'] for r in rows])[0]:.3f}")
    print(f"  multi-cell recall    {boot([r['rec'] for r in rows])[0]:.3f}")
    print(f"  -> this is the CEILING for Explanation->GT under faithful following")

    # ---------------- A2.2/A2.3 Explanation -> Evidence / GT ----------------
    print("\n" + "=" * 96)
    print("A2.2 + A2.3  EXPLANATION -> EVIDENCE  and  EXPLANATION -> GT  (fake only)")
    hdr = (f"  {'condition':<18}{'cite':>7}{'E>Ev hit':>10}{'E>Ev exact':>12}"
           f"{'unsup':>8}{'E>GT hit':>10}{'E>GT top1':>11}{'prec':>7}{'rec':>7}"
           f"{'n':>6}")
    print(hdr)
    TAB1 = {}
    order = ["7B_score_only", "7B_rawmap", "7B_struct_E2", "7B_structured",
             "7B_struct_rule", "7B_mirrored", "32B_score_only", "32B_structured",
             "32B_struct_rule", "32B_mirrored"]
    for k in order:
        src = C[k]
        cite, evh, eve, uns, gth, gt1, pr, rc = [], [], [], [], [], [], [], []
        for s in ids:
            r = src.get((s, "fake"))
            if not r:
                continue
            ref = r["referenced_regions"]
            if len(ref) == 9:      # degenerate all-nine echo, carries no information
                continue
            cite.append(1 if ref else 0)
            ev = r.get("evidence_locations")
            if ev is not None and ref:
                a = rel(ref, ev)
                evh.append(a["hit"])
                eve.append(a["exact"])
                uns.append(int(bool(set(ref) - set(ev))))
            if ref:
                b = rel(ref, GT[s]["cells"])
                gth.append(b["hit"])
                gt1.append(int(ref[0] in GT[s]["cells"]))
                pr.append(b["prec"])
                rc.append(b["rec"])
        f = lambda v: (f"{boot(v)[0]:.3f}" if v else "  -  ")
        TAB1[k] = dict(cite=boot(cite), ev_hit=boot(evh) if evh else None,
                       ev_exact=boot(eve) if eve else None,
                       unsup=boot(uns) if uns else None,
                       gt_hit=boot(gth) if gth else None,
                       gt_top1=boot(gt1) if gt1 else None,
                       prec=boot(pr) if pr else None, rec=boot(rc) if rc else None,
                       n=len(cite))
        print(f"  {k:<18}{f(cite):>7}{f(evh):>10}{f(eve):>12}{f(uns):>8}"
              f"{f(gth):>10}{f(gt1):>11}{f(pr):>7}{f(rc):>7}{len(cite):>6}")
    print(f"\n  cite = fraction giving any region; all-nine echoes excluded entirely")
    print(f"  E>Ev metrics need evidence to exist, so score-only rows are blank")

    # ---------------- A3 mirrored comparison ----------------
    print("\n" + "=" * 96)
    print("A3. MIRRORED vs CORRECT STRUCTURED EVIDENCE (eligible subset only)")
    print("    eligible = GT-independent, frozen before any mirror result")
    for model, cor, mir in (("7B", "7B_struct_E2", "7B_mirrored"),
                            ("32B", "32B_struct_rule", "32B_mirrored")):
        use = [s for s in ids if s in ELIG["fake"]
               and (s, "fake") in C[cor] and (s, "fake") in C[mir]]
        print(f"\n  [{model}] n={len(use)}  (correct cell: {cor})")
        for nm, key in (("correct evidence", cor), ("mirrored evidence", mir)):
            evh, gth, cite = [], [], []
            for s in use:
                r = C[key][(s, "fake")]
                ref = r["referenced_regions"]
                if len(ref) == 9:
                    continue
                cite.append(1 if ref else 0)
                if not ref:
                    continue
                ev = r.get("evidence_locations") or []
                evh.append(rel(ref, ev)["hit"])
                gth.append(rel(ref, GT[s]["cells"])["hit"])
            print(f"    {nm:<20} cite {np.mean(cite):.3f}  "
                  f"Expl->Evidence {np.mean(evh) if evh else float('nan'):.3f}  "
                  f"Expl->GT {np.mean(gth) if gth else float('nan'):.3f}  "
                  f"n_scored {len(evh)}")
        # paired delta on the same sources
        pe, pg = [], []
        for s in use:
            a, b = C[cor][(s, "fake")], C[mir][(s, "fake")]
            ra, rb = a["referenced_regions"], b["referenced_regions"]
            if not ra or not rb or len(ra) == 9 or len(rb) == 9:
                continue
            pe.append(rel(rb, b.get("evidence_locations") or [])["hit"]
                      - rel(ra, a.get("evidence_locations") or [])["hit"])
            pg.append(rel(rb, GT[s]["cells"])["hit"]
                      - rel(ra, GT[s]["cells"])["hit"])
        if pe:
            de, dg = boot(pe), boot(pg)
            print(f"    paired mirrored - correct:  Expl->Evidence {de[0]:+.3f} "
                  f"[{de[1]:+.3f},{de[2]:+.3f}]   Expl->GT {dg[0]:+.3f} "
                  f"[{dg[1]:+.3f},{dg[2]:+.3f}]   n={len(pe)}")
    print(f"\n  descriptive only: if Expl->Evidence stays high while Expl->GT falls,")
    print(f"  the model follows the evidence it is given and its explanation")
    print(f"  correctness tracks the evidence's correctness.")

    # ---------------- E. 7B vs 32B, coverage vs correctness ----------------
    print("\n" + "=" * 96)
    print("E. 7B vs 32B — CITATION COVERAGE vs CITATION CORRECTNESS")
    print("   these are reported separately and never merged")
    print(f"  {'condition':<18}{'coverage':>10}{'cond E>Ev exact':>17}"
          f"{'cond E>GT hit':>15}{'n cited':>9}")
    for k in ("7B_structured", "32B_structured", "7B_struct_rule", "32B_struct_rule"):
        src = C[k]
        cov, ex, gh = [], [], []
        for s in ids:
            r = src.get((s, "fake"))
            if not r:
                continue
            ref = r["referenced_regions"]
            if len(ref) == 9:
                continue
            cov.append(1 if ref else 0)
            if ref:
                ev = r.get("evidence_locations") or []
                ex.append(rel(ref, ev)["exact"])
                gh.append(rel(ref, GT[s]["cells"])["hit"])
        print(f"  {k:<18}{np.mean(cov):>10.3f}{np.mean(ex):>17.3f}"
              f"{np.mean(gh):>15.3f}{len(ex):>9}")
    print(f"  coverage = willingness to name a region at all")
    print(f"  conditional metrics are computed ONLY on samples where a region was named")

    # ---------------- F. raw map vs structured ----------------
    print("\n" + "=" * 96)
    print("F. RAW MAP vs STRUCTURED EVIDENCE (7B, same sources, paired)")
    raw, st = C["7B_rawmap"], C["7B_struct_E2"]
    use = [s for s in ids if (s, "fake") in raw and (s, "fake") in st]
    print(f"  n={len(use)}")
    for nm, src in (("raw map (E1)", raw), ("structured (E2)", st)):
        cov, evh, gth, pr = [], [], [], []
        for s in use:
            r = src[(s, "fake")]
            ref = r["referenced_regions"]
            if len(ref) == 9:
                continue
            cov.append(1 if ref else 0)
            if not ref:
                continue
            ev = st[(s, "fake")].get("evidence_locations") or []
            evh.append(rel(ref, ev)["hit"])
            gth.append(rel(ref, GT[s]["cells"])["hit"])
            pr.append(rel(ref, GT[s]["cells"])["prec"])
        print(f"    {nm:<18} cite {np.mean(cov):.3f}  Expl->Evidence "
              f"{np.mean(evh):.3f}  Expl->GT {np.mean(gth):.3f}  "
              f"GT precision {np.mean(pr):.3f}  n {len(evh)}")
    pe, pg = [], []
    for s in use:
        a, b = raw[(s, "fake")], st[(s, "fake")]
        ra, rb = a["referenced_regions"], b["referenced_regions"]
        if not ra or not rb or len(ra) == 9 or len(rb) == 9:
            continue
        ev = b.get("evidence_locations") or []
        pe.append(rel(rb, ev)["hit"] - rel(ra, ev)["hit"])
        pg.append(rel(rb, GT[s]["cells"])["hit"] - rel(ra, GT[s]["cells"])["hit"])
    if pe:
        de, dg = boot(pe), boot(pg)
        print(f"    paired structured - raw:  Expl->Evidence {de[0]:+.3f} "
              f"[{de[1]:+.3f},{de[2]:+.3f}]   Expl->GT {dg[0]:+.3f} "
              f"[{dg[1]:+.3f},{dg[2]:+.3f}]   n={len(pe)}")
    print(f"  question: does structured evidence only make the model better at")
    print(f"  RESTATING evidence, or also bring the explanation closer to GT?")

    # ---------------- real-image citation sanity (spatial part only) ----------
    print("\n" + "=" * 96)
    print("REAL-IMAGE SPATIAL SANITY (no GT mask exists; counts only)")
    print(f"  {'condition':<18}{'named a region':>16}{'mean n cells':>14}{'n':>6}")
    for k in order:
        src = C[k]
        cov, ncell = [], []
        for s in ids:
            r = src.get((s, "real"))
            if not r:
                continue
            ref = r["referenced_regions"]
            if len(ref) == 9:
                continue
            cov.append(1 if ref else 0)
            ncell.append(len(ref))
        if cov:
            print(f"  {k:<18}{np.mean(cov):>16.3f}{np.mean(ncell):>14.2f}"
                  f"{len(cov):>6}")
    print(f"  naming a region on a real image is not automatically wrong, but it is")
    print(f"  the population Part B must audit for unsupported suspicion")

    od = os.path.join(R, "explainability")
    os.makedirs(od, exist_ok=True)
    json.dump(dict(gt=dict((s, GT[s]) for s in ids), table1=TAB1,
                   mask_threshold=thr, n_fake=len(ids)),
              open(os.path.join(od, "x1_spatial_analysis.json"), "w"), indent=1)
    print(f"\nwrote {od}/x1_spatial_analysis.json")


if __name__ == "__main__":
    main()
