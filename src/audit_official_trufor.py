"""Audit the official TGIF benchmark artifacts for TruFor feasibility evidence.

Central question: do the official files contain an IMAGE-LEVEL fake-vs-real
discrimination score, or only per-image LOCALIZATION quality metrics computed
against the ground-truth mask? Those are different things, and only the former
answers whether TruFor can serve as an image-level evidence source.

Read-only. No downloads.
"""
import csv, json, os, sys
from collections import Counter, defaultdict

BR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tgifmeta/benchmark-results"
GQ = os.path.join(BR, "generative_quality")


def load(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except Exception:
        return None


def summarize(name, rows):
    print(f"\n{'='*88}\nFILE: {name}   rows={len(rows)}")
    cols = list(rows[0].keys())
    print(f"  columns ({len(cols)}): {cols}")
    # what is actually in here
    for key in ("tool", "type", "tool_category", "label", "mask_variation"):
        if key in cols:
            c = Counter(r[key] for r in rows)
            top = dict(sorted(c.items(), key=lambda kv: -kv[1])[:12])
            print(f"  {key:<14} ({len(c)} distinct): {top}")
    # split inferred from Path
    if "parsed_path" in cols:
        sp = Counter(str(r["parsed_path"]).split("/")[0] for r in rows)
        print(f"  split (from parsed_path): {dict(sp)}")
    # coco_ids
    if "image_id" in cols:
        ids = {str(r["image_id"]).strip() for r in rows}
        print(f"  distinct coco_ids: {len(ids)}")
    return cols


def metric_stats(rows, cols, group_keys=("tool_category",)):
    """Report the numeric metric columns per subset."""
    METRICS = [c for c in ("F1", "IoU", "AuC", "Precision", "Recall") if c in cols]
    if not METRICS:
        print("  no localization metric columns found")
        return
    g = defaultdict(list)
    for r in rows:
        k = tuple(str(r.get(x, "")) for x in group_keys)
        g[k].append(r)
    print(f"\n  per-subset metric means ({', '.join(group_keys)}):")
    hdr = f"    {'subset':<26}{'n':>6}" + "".join(f"{m:>11}" for m in METRICS)
    print(hdr)
    for k in sorted(g):
        rs = g[k]
        line = f"    {'/'.join(k)[:26]:<26}{len(rs):>6}"
        for m in METRICS:
            vs = [num(r[m]) for r in rs]
            vs = [v for v in vs if v is not None]
            line += f"{(sum(vs)/len(vs) if vs else float('nan')):>11.4f}"
        print(line)


def main():
    print("#" * 88)
    print("OFFICIAL TGIF BENCHMARK AUDIT — is there an image-level TruFor score?")
    print("#" * 88)

    files = {}
    for f in ("trufor_orig.csv", "trufor_tgif2fr.csv"):
        p = os.path.join(GQ, f)
        if os.path.exists(p):
            files[f] = load(p)

    allcols = {}
    for f, rows in files.items():
        allcols[f] = summarize(f, rows)
        metric_stats(rows, allcols[f])

    print(f"\n{'='*88}")
    print("CRITICAL FIELD CHECK: image-level score vs localization metric")
    IMG_LEVEL_HINTS = ("score", "integrity", "detection", "conf", "pred",
                       "probability", "prob", "logit", "det_score", "image_score")
    for f, cols in allcols.items():
        hits = [c for c in cols if any(h in c.lower() for h in IMG_LEVEL_HINTS)]
        print(f"  {f}")
        print(f"    columns suggesting an image-level score: {hits or 'NONE'}")
        loc = [c for c in ("F1", "IoU", "AuC", "Precision", "Recall") if c in cols]
        print(f"    localization-quality columns present   : {loc}")

    # label semantics
    print(f"\n{'='*88}")
    print("LABEL SEMANTICS — does 'orig' file actually contain real images?")
    for f, rows in files.items():
        lc = Counter(str(r.get("label", "")).strip() for r in rows)
        print(f"  {f:<24} label distribution: {dict(lc)}")
        # are any rows pointing at *_orig.png as the analysed image?
        if "img_path" in rows[0]:
            oi = sum(1 for r in rows if "_orig" in str(r["img_path"]))
            print(f"    rows whose ANALYSED image is an original: {oi}/{len(rows)}")
        if "src_path" in rows[0]:
            print(f"    note: 'src_path' is the reference original used for "
                  f"PSNR/LPIPS, NOT an analysed real sample")

    print(f"\n{'='*88}")
    print("COVERAGE vs OUR TARGET (sd2-sp)")
    for f, rows in files.items():
        tc = Counter(str(r.get("tool_category", "")) for r in rows)
        has_sd2sp = [k for k in tc if "sd2" in k.lower() and "sp" in k.lower()]
        print(f"  {f}")
        print(f"    subsets present     : {sorted(tc)}")
        print(f"    sd2-sp present?     : {has_sd2sp or 'NO'}")


if __name__ == "__main__":
    main()
