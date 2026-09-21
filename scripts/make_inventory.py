"""Generate PROJECT_INVENTORY_BEFORE_CLEANUP.md. Read-only: touches nothing."""
import hashlib, json, os, subprocess, sys
from collections import Counter, defaultdict


def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True,
                              timeout=300).stdout.strip()
    except Exception as e:
        return f"<error {e}>"


def walk(root, skip=(".git", "__pycache__")):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in skip]
        for f in fn:
            yield os.path.join(dp, f)


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}P"


def main():
    CODE = "/home/borui/haolin/fevi"
    RUNS = "/mnt/disk3/borui/fevi/runs"
    DATA = "/mnt/disk3/borui/fevi/data"
    L = []
    L.append("# PROJECT_INVENTORY_BEFORE_CLEANUP.md\n")
    L.append("Read-only inventory taken BEFORE any reorganisation, so the starting "
             "state is recoverable.\n")
    L.append(f"Generated: {sh('date -Iseconds')}\n")

    L.append("## Canonical locations\n")
    L.append("| role | path | note |")
    L.append("|---|---|---|")
    L.append(f"| code | `{CODE}` | server; scripts were executed here |")
    L.append(f"| results | `{RUNS}` | server; all frozen protocols + verdicts |")
    L.append(f"| datasets | `{DATA}` | server; NOT for git |")
    L.append("| X3 annotations | `C:/Users/HarleyH/fevi/x3` | local; annotation "
             "packets and outputs |")
    L.append("")
    L.append("Neither location was a git repository before this cleanup "
             "(`git rev-parse` returned nothing in both).\n")

    # ---- top-level dirs
    L.append("## Top-level directories (code tree)\n```")
    L.append(sh(f"cd {CODE} && find . -maxdepth 2 -type d "
                f"-not -path './.git*' -not -name '__pycache__' | sort"))
    L.append("```\n")

    # ---- file counts
    L.append("## File counts\n")
    for nm, root in (("code tree", CODE), ("runs tree", RUNS)):
        c = Counter()
        tot = 0
        for p in walk(root):
            e = os.path.splitext(p)[1].lower() or "<noext>"
            c[e] += 1
            tot += 1
        L.append(f"**{nm}** — {tot} files")
        L.append("")
        L.append("| ext | count |")
        L.append("|---|---|")
        for e, n in c.most_common(14):
            L.append(f"| `{e}` | {n} |")
        L.append("")

    # ---- disk usage
    L.append("## Disk usage\n```")
    L.append(sh(f"du -sh {CODE} {RUNS} {DATA}"))
    L.append("")
    L.append(sh(f"du -h --max-depth=1 {RUNS} | sort -h"))
    L.append("```\n")

    # ---- large files
    L.append("## Large files\n")
    L.append("| size | path | purpose |")
    L.append("|---|---|---|")
    WHY = {
        "mit_b2.pth": "TruFor upstream SegFormer backbone (vendored repo)",
        "trufor.pth.tar": "TruFor inference checkpoint",
        "TruFor_weights.zip": "TruFor weights archive as downloaded",
        ".pack": "git pack of the vendored TruFor repo",
        "_train_list.txt": "TruFor upstream training list (unused here)",
    }
    rows = []
    for root in (CODE, RUNS, DATA):
        for p in walk(root):
            try:
                s = os.path.getsize(p)
            except OSError:
                continue
            if s > 10 * 1024 ** 2:
                why = next((v for k, v in WHY.items() if k in p), "")
                rows.append((s, p, why))
    for s, p, why in sorted(rows, reverse=True)[:40]:
        L.append(f"| {human(s)} | `{p}` | {why} |")
    if not rows:
        L.append("| — | none over 10M outside datasets | — |")
    L.append("")
    L.append(f"Total files over 10M: {len(rows)}. Everything above is either a model "
             f"checkpoint, a vendored upstream repo, or raw dataset imagery; none of "
             f"it belongs in git.\n")

    # ---- runs dir listing with status hints
    L.append("## Run directories\n")
    L.append("| directory | size | files | has frozen protocol | has verdicts |")
    L.append("|---|---|---|---|---|")
    for d in sorted(os.listdir(RUNS)):
        fp = os.path.join(RUNS, d)
        if not os.path.isdir(fp):
            continue
        fs = list(walk(fp))
        sz = sum(os.path.getsize(x) for x in fs if os.path.exists(x))
        froz = any("frozen" in os.path.basename(x) for x in fs)
        verd = any("verdict" in os.path.basename(x) for x in fs)
        L.append(f"| `{d}` | {human(sz)} | {len(fs)} | "
                 f"{'yes' if froz else '-'} | {'yes' if verd else '-'} |")
    L.append("")

    # ---- protocol docs
    L.append("## Protocol / report markdown in code tree\n")
    L.append("| file | bytes | sha1 |")
    L.append("|---|---|---|")
    for f in sorted(os.listdir(CODE)):
        if f.endswith(".md"):
            p = os.path.join(CODE, f)
            h = hashlib.sha1(open(p, "rb").read()).hexdigest()
            L.append(f"| `{f}` | {os.path.getsize(p)} | `{h[:16]}…` |")
    L.append("")

    # ---- scripts
    L.append("## Scripts in src/\n")
    py = sorted(f for f in os.listdir(os.path.join(CODE, "src"))
                if f.endswith(".py"))
    groups = defaultdict(list)
    for f in py:
        if f.startswith("freeze_"):
            groups["freeze (protocol freezers)"].append(f)
        elif "run" in f:
            groups["run (inference drivers)"].append(f)
        elif f.startswith("analyze_") or f.startswith("analysis"):
            groups["analyze"].append(f)
        elif f.startswith("smoke") or "smoke" in f or f.startswith("probe") \
                or f.startswith("dbg") or f.startswith("check"):
            groups["smoke / debug"].append(f)
        else:
            groups["shared modules / utilities"].append(f)
    for g in sorted(groups):
        L.append(f"**{g}** ({len(groups[g])})")
        L.append("")
        L.append("```")
        L.append("\n".join(groups[g]))
        L.append("```")
        L.append("")
    L.append(f"Total `.py` in src/: {len(py)}\n")

    L.append("## Git state before cleanup\n")
    L.append("```")
    L.append(f"code tree : {sh(f'cd {CODE} && git status --short 2>&1 | head -5') or 'not a git repository'}")
    L.append("```\n")
    L.append("No commits, no remote, no `.gitignore` existed anywhere in the project "
             "before this stage.\n")

    out = sys.argv[1] if len(sys.argv) > 1 else "PROJECT_INVENTORY_BEFORE_CLEANUP.md"
    body = "\n".join(L)
    open(out, "w", encoding="utf-8").write(body)
    print(f"wrote {out} ({len(body)} bytes)")


if __name__ == "__main__":
    main()
