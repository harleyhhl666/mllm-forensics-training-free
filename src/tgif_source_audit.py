"""TGIF source audit: which extra subsets exist, and how many coco_ids are
source-disjoint from everything already used in this project.

A source is a coco_id (an MS-COCO original image). Two fakes built from the SAME
coco_id by DIFFERENT generators are NOT independent sources, so subsets are scored
by coco_id disjointness, not by image count.

Reads only local metadata CSVs + our own frozen run artifacts. No downloads.
"""
import csv, json, os, sys
from collections import Counter, defaultdict

META = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tgifmeta/metadata"
RUNS = sys.argv[2] if len(sys.argv) > 2 else "/mnt/disk3/borui/fevi/runs"


def load(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def cid_of(row):
    for k in ("coco_id", "image_id", "id", "coco"):
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    # fall back: first numeric-looking field in a filename column
    for k, v in row.items():
        v = str(v)
        if v and "_" in v and v.split("_")[0].isdigit():
            return v.split("_")[0]
    return None


def main():
    print("=" * 88)
    print("1. METADATA FILES PRESENT")
    files = sorted(f for f in os.listdir(META) if f.endswith(".csv"))
    tables = {}
    for f in files:
        rows = load(os.path.join(META, f))
        tables[f] = rows
        cols = list(rows[0].keys()) if rows else []
        print(f"  {f:<24} rows={len(rows):<7} cols={cols[:8]}")

    print("\n" + "=" * 88)
    print("2. USED coco_ids IN THIS PROJECT (all prior stages)")
    used = set()
    srcs = {}
    p = os.path.join(RUNS, "used_coco_ids.json")
    if os.path.exists(p):
        s = set(json.load(open(p)))
        srcs["calib/pilot/primary/secondary"] = s
        used |= s
    for tag, rel in (("V4 validation", "validation_v4/validation_v4_frozen.json"),
                     ("mechanism", "mechanism/mechanism_frozen.json"),
                     ("mitigation", "mitigation/mitigation_frozen.json")):
        fp = os.path.join(RUNS, rel)
        if os.path.exists(fp):
            s = {x["coco_id"] for x in json.load(open(fp))["samples"]}
            srcs[tag] = s
            used |= s
    for k, v in srcs.items():
        print(f"  {k:<32} {len(v)} coco_ids")
    print(f"  {'TOTAL DISTINCT USED':<32} {len(used)} coco_ids")

    print("\n" + "=" * 88)
    print("3. PER-GENERATOR coco_id INVENTORY (from metadata)")
    print("   sp = spliced (inpainted region pasted back), fr = fully regenerated")
    gen_cids = {}
    for f, rows in tables.items():
        if f.startswith("orig"):
            continue
        gen = f.split("_")[0]
        split = f.split("_")[1].replace(".csv", "")
        cids = {c for c in (cid_of(r) for r in rows) if c}
        gen_cids[(gen, split)] = cids
    for (gen, split), cids in sorted(gen_cids.items()):
        inter = cids & used
        print(f"  {gen:<8} {split:<11} coco_ids={len(cids):<6} "
              f"overlap_with_used={len(inter):<6} disjoint={len(cids - used)}")

    print("\n" + "=" * 88)
    print("4. SOURCE-DISJOINT AVAILABILITY PER GENERATOR (union over splits)")
    per_gen = defaultdict(set)
    for (gen, split), cids in gen_cids.items():
        per_gen[gen] |= cids
    rank = []
    for gen, cids in sorted(per_gen.items()):
        dis = cids - used
        rank.append((len(dis), gen, len(cids), len(cids & used)))
        print(f"  {gen:<10} total={len(cids):<6} used={len(cids & used):<6} "
              f"TRULY DISJOINT={len(dis)}")
    rank.sort(reverse=True)

    print("\n" + "=" * 88)
    print("5. CROSS-GENERATOR coco_id SHARING (are subsets built on the same sources?)")
    gens = sorted(per_gen)
    print(f"  {'':<10}" + "".join(f"{g:>12}" for g in gens))
    for a in gens:
        row = f"  {a:<10}"
        for b in gens:
            row += f"{len(per_gen[a] & per_gen[b]):>12}"
        print(row)
    print("\n  -> if off-diagonal counts are close to the diagonal, the generators")
    print("     REUSE the same coco_ids and give NO new independent sources.")

    print("\n" + "=" * 88)
    print("6. RECOMMENDATION INPUT")
    for n, gen, tot, ov in rank:
        print(f"  {gen:<10} disjoint_sources={n}")
    best = rank[0] if rank else None
    if best:
        print(f"\n  largest source-disjoint pool: {best[1]} with {best[0]} coco_ids")


if __name__ == "__main__":
    main()
