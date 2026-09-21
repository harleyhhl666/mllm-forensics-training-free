"""Read .xlsx with the standard library only (xlsx = zip of XML).

Used to audit the official TGIF IFL/SID benchmark spreadsheets without installing
openpyxl into a verified environment.
"""
import re, sys, zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def col_to_idx(ref):
    m = re.match(r"([A-Z]+)", ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    # sheet names
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    names = [s.get("name") for s in wb.iter(f"{NS}sheet")]
    sheets = {}
    paths = sorted(n for n in z.namelist()
                   if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
    for i, sp in enumerate(paths):
        root = ET.fromstring(z.read(sp))
        rows = []
        for r in root.iter(f"{NS}row"):
            cells = {}
            for c in r.findall(f"{NS}c"):
                ref = c.get("r") or ""
                ci = col_to_idx(ref) if ref else len(cells)
                t = c.get("t")
                v = c.find(f"{NS}v")
                isn = c.find(f"{NS}is")
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr" and isn is not None:
                    val = "".join(x.text or "" for x in isn.iter(f"{NS}t"))
                elif v is not None:
                    val = v.text
                else:
                    val = ""
                cells[ci] = val
            if cells:
                width = max(cells) + 1
                rows.append([cells.get(j, "") for j in range(width)])
        nm = names[i] if i < len(names) else f"sheet{i+1}"
        sheets[nm] = rows
    return sheets


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("#" * 90)
        print("WORKBOOK:", p)
        sh = read_xlsx(p)
        for nm, rows in sh.items():
            print(f"\n=== sheet '{nm}'  rows={len(rows)} ===")
            for r in rows[:60]:
                cells = ["" if c is None else str(c) for c in r]
                line = " | ".join(f"{c[:20]:<20}" for c in cells[:12])
                print("   ", line)
            if len(rows) > 60:
                print(f"    ... {len(rows)-60} more rows")
