"""Extract TruFor's rows from the official TGIF IFL summary, with the column
headers resolved, and audit what kind of metric each number actually is.

The distinction that matters: 'auc' in this workbook is computed per image over
PIXELS against the GT mask (a localization metric). A real image has no tampered
pixels, hence the real column is 0 for every method -- which is the signature of a
localization metric, not of image-level fake-vs-real discrimination.
"""
import sys
sys.path.insert(0, "/home/borui/haolin/fevi/src")
from xlsx_read import read_xlsx


def main(path, sid_paths):
    sh = read_xlsx(path)
    for nm, rows in sh.items():
        print("=" * 90)
        print(f"SHEET: {nm}  rows={len(rows)}")
        # header rows: row0 = dataset groups, row1 = real/fake/all
        g, sub = rows[0], rows[1]
        groups = []
        cur = ""
        for i in range(1, max(len(g), len(sub))):
            gv = g[i] if i < len(g) else ""
            if str(gv).strip():
                cur = str(gv).strip()
            s = str(sub[i]).strip() if i < len(sub) else ""
            groups.append((i, cur, s))
        print("\n  COLUMN LAYOUT:")
        seen = {}
        for i, ds, s in groups:
            seen.setdefault(ds, []).append(s)
        for ds, ss in seen.items():
            print(f"    dataset '{ds}': columns {ss}")

        print("\n  TRUFOR ROWS (all datasets):")
        for r in rows:
            if not r or not str(r[0]).strip().startswith("trufor"):
                continue
            metric = str(r[0]).strip()
            print(f"\n    {metric}")
            for i, ds, s in groups:
                v = r[i] if i < len(r) else ""
                if str(v).strip() == "":
                    continue
                try:
                    fv = float(v)
                    print(f"      {ds:<12} {s:<6} {fv:.4f}")
                except Exception:
                    print(f"      {ds:<12} {s:<6} {v}")

        print("\n  REAL-COLUMN CHECK across ALL methods (is 'real' always 0 for auc?)")
        for r in rows:
            n = str(r[0]).strip()
            if not n.endswith("_auc"):
                continue
            vals = []
            for i, ds, s in groups:
                if s == "real":
                    v = r[i] if i < len(r) else ""
                    vals.append(f"{ds}={v}")
            print(f"    {n:<22}{'  '.join(vals)}")

    for sp in sid_paths:
        print("\n" + "#" * 90)
        print("SID WORKBOOK:", sp)
        s2 = read_xlsx(sp)
        for nm, rows in s2.items():
            print(f"\n=== sheet '{nm}' rows={len(rows)} ===")
            for r in rows[:25]:
                print("   ", " | ".join(f"{str(c)[:22]:<22}" for c in r[:9]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
