#!/usr/bin/env python3
"""Release gate for the training-free stage. Any FAIL blocks a push.

Checks: expected files exist; model snapshots match; frozen sample counts and pairing;
key prompt hashes; headline results recomputed from raw verdicts; invalid runs excluded
from the canonical manifest; git safety (no weights, secrets, .env or large tracked files).

Exit code is non-zero if any check FAILs.
"""
import glob, hashlib, json, os, re, subprocess, sys
from collections import Counter

RUNS = "runs"
P, W, F = [], [], []


def ok(m):
    P.append(m)
    print(f"  PASS  {m}")


def warn(m):
    W.append(m)
    print(f"  WARN  {m}")


def fail(m):
    F.append(m)
    print(f"  FAIL  {m}")


def sh(c):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, timeout=600)
        return (r.stdout or b"").decode("utf-8", "replace")
    except Exception:
        return ""


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def rows(path, cond=None):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8", errors="ignore"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if cond is None or d.get("condition") == cond:
            out.append(d)
    return out


def metrics(rr):
    f = [r for r in rr if r["label"] == "fake"]
    t = [r for r in rr if r["label"] == "real"]
    if not f or not t:
        return None
    rec = sum(r["final_verdict"] == "fake" for r in f) / len(f)
    fpr = sum(r["final_verdict"] == "fake" for r in t) / len(t)
    return rec, fpr, rec - fpr


def main():
    print("=" * 78)
    print("FINAL AUDIT — training-free stage")
    print("=" * 78)

    # ---------------- 1 files ----------------
    print("\n[1] EXPECTED FILES")
    need = ["README.md", ".gitignore", "docs/EXPERIMENT_INDEX.md",
            "docs/FINAL_EXPERIMENT_MANIFEST.md", "docs/CURRENT_FINDINGS.md",
            "docs/INVALID_AND_SUPERSEDED_RUNS.md", "docs/REPRODUCIBILITY.md",
            "docs/PROJECT_STRUCTURE.md", "docs/DATASET_AND_SPLITS.md",
            "docs/NEXT_STAGE.md", "docs/data_manifest.csv",
            "reports/TRAINING_FREE_EXPERIMENT_REPORT.md", "reports/CROSSCHECK.txt",
            "environment/SYSTEM_INFO.txt", "configs/exp01.yaml",
            "configs/exp01_32b.yaml", "runs/SHA256SUMS_LOCAL_ONLY.txt"]
    miss = [f for f in need if not os.path.exists(f)]
    ok(f"all {len(need)} expected documents present") if not miss else \
        fail(f"missing: {miss}")

    for e in ("qwen_vl", "qwen_vl_32b", "trufor"):
        a = f"environment/{e}_history.yml"
        b = f"environment/{e}_freeze.txt"
        if os.path.exists(a) and os.path.exists(b):
            ok(f"environment export present: {e}")
        else:
            fail(f"environment export missing: {e}")

    # frozen protocols
    froz = [p for p in glob.glob(f"{RUNS}/**/*frozen*.json", recursive=True)]
    ok(f"frozen protocol files present: {len(froz)}") if len(froz) >= 15 else \
        fail(f"only {len(froz)} frozen protocols found")

    # raw verdicts for the canonical rounds
    CANON = {"trufor_fields": 2208, "trufor_semantics": 1472, "trufor_2x2": 368,
             "trufor_32b_final": 1622, "trufor_framing": 2208,
             "trufor_abstraction": 1840, "trufor_conflict": 1656,
             "trufor_score_map": 1472, "trufor_32b": 736}
    for d, n in CANON.items():
        got = sum(len(rows(p)) for p in glob.glob(f"{RUNS}/{d}/*.jsonl")
                  if "INVALID" not in p)
        if got == n:
            ok(f"{d}: {got} rows as expected")
        else:
            fail(f"{d}: {got} rows, expected {n}")

    # annotations
    na = len(glob.glob(f"{RUNS}/explainability/x3/ann/*.json"))
    nr = len(glob.glob(f"{RUNS}/explainability/x3/ann_repeat/*.json"))
    ok(f"X3 annotations: {na} primary + {nr} repeat") if (na, nr) == (300, 60) else \
        fail(f"X3 annotations: {na} primary + {nr} repeat, expected 300 + 60")

    # ---------------- 2 models ----------------
    print("\n[2] MODEL SNAPSHOTS")
    S7 = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    S32 = "7cfb30d71a1f4f49a57592323337a4a4727301da"
    fin = jload(f"{RUNS}/trufor_32b_final/trufor_32b_final_capacity_frozen.json")
    got32 = (fin.get("model") or {}).get("snapshot")
    ok(f"32B snapshot pinned {S32}") if got32 == S32 else \
        fail(f"32B snapshot is {got32}, expected {S32}")
    sem = json.dumps(jload(f"{RUNS}/trufor_semantics/"
                           "trufor_task_semantics_frozen.json"))
    ok(f"7B snapshot {S7} present in semantics freeze") if S7 in sem else \
        fail("7B snapshot not found in semantics freeze")
    rm = jload(f"{RUNS}/trufor_feasibility/run_meta.json")
    ok("TruFor checkpoint md5 matches") \
        if rm.get("checkpoint_md5") == "55d7075dd1ff945e9c0f9437c5df9495" \
        else fail(f"TruFor checkpoint md5 is {rm.get('checkpoint_md5')}")

    # ---------------- 3 samples ----------------
    print("\n[3] FROZEN SAMPLES")
    conf = jload(f"{RUNS}/trufor_conflict/trufor_score_map_conflict_frozen.json")
    sm = conf["samples"]
    ok(f"frozen subset has {len(sm)} sources") if len(sm) == 184 else \
        fail(f"frozen subset has {len(sm)} sources, expected 184")
    cids = [s["coco_id"] for s in sm]
    ok("all coco_id unique (no duplicate sources)") if len(set(cids)) == len(cids) \
        else fail(f"duplicate coco_id: {len(cids)-len(set(cids))}")
    for d, cond in (("trufor_32b_final", "B0_score_only"),
                    ("trufor_semantics", "D0_baseline")):
        rr = rows(f"{RUNS}/{d}/verdicts.jsonl", cond)
        c = Counter(r["label"] for r in rr)
        ok(f"{d}/{cond} paired 184/184") if c.get("fake") == c.get("real") == 184 \
            else fail(f"{d}/{cond} unbalanced: {dict(c)}")
    fr = jload(f"{RUNS}/explainability/x3/x2_semantic_audit_frozen.json")
    sf = set(fr["sampling"]["fake_ids"])
    sr = set(fr["sampling"]["real_ids"])
    if len(sf) == 100 and len(sr) == 50 and not (sf & sr):
        ok("audit subset 100 fake + 50 real, disjoint")
    else:
        fail(f"audit subset {len(sf)}/{len(sr)}, overlap {len(sf & sr)}")

    # ---------------- 4 prompts ----------------
    print("\n[4] KEY PROMPT HASHES")
    EXPECT = {"A0/S0/P0 score-only": "8775b797feccec20",
              "A2/S5/D0 structured": "22b9ad9a944812ef",
              "A3/D1 structured+rule": "9167aa08c1a09d68",
              "A1 score+rule": "ef68309f567c7a61"}
    seen = {}
    for p in froz:
        j = jload(p)
        pr = j.get("prompts")
        if isinstance(pr, dict):
            for k, v in pr.items():
                if isinstance(v, str):
                    seen[hashlib.sha1(v.encode()).hexdigest()[:16]] = (
                        os.path.basename(p), k)
    for nm, h in EXPECT.items():
        ok(f"{nm}: {h} found in {seen[h][0]}") if h in seen else \
            fail(f"{nm}: prompt hash {h} NOT found in any frozen protocol")
    d0 = jload(f"{RUNS}/trufor_semantics/"
               "trufor_task_semantics_frozen.json")["prompts"]["D0_baseline"]
    b2 = fin["prompts"]["B2_structured"]
    ok("7B D0 and 32B B2 prompts byte-identical") if d0 == b2 else \
        fail("7B D0 and 32B B2 prompts DIFFER — X3 comparison invalid")

    # ---------------- 5 results ----------------
    print("\n[5] HEADLINE RESULTS RECOMPUTED")
    T7 = {"A0": (f"{RUNS}/trufor_fields/verdicts.jsonl", "S0_score_only",
                 .837, .011, .826),
          "A1": (f"{RUNS}/trufor_2x2/verdicts_A1.jsonl", None, .978, .082, .897),
          "A2": (f"{RUNS}/trufor_semantics/verdicts.jsonl", "D0_baseline",
                 .114, .000, .114),
          "A3": (f"{RUNS}/trufor_semantics/verdicts.jsonl", "D1_binary_rule",
                 .967, .065, .902)}
    for cell, (pth, cond, tr, tf, tj) in T7.items():
        m = metrics(rows(pth, cond))
        if not m:
            fail(f"{cell}: no rows")
            continue
        if all(abs(a - b) <= .002 for a, b in zip(m, (tr, tf, tj))):
            ok(f"7B {cell}: recall {m[0]:.3f} FPR {m[1]:.3f} J {m[2]:.3f}")
        else:
            fail(f"7B {cell}: got {tuple(round(x,3) for x in m)}, "
                 f"expected {(tr, tf, tj)}")
    T32 = {"B0_score_only": (.978, .060, .918), "B1_score_rule": (.989, .109, .880),
           "B2_structured": (.978, .076, .902),
           "B3_structured_rule": (1.0, .337, .663)}
    for cond, tgt in T32.items():
        m = metrics(rows(f"{RUNS}/trufor_32b_final/verdicts.jsonl", cond))
        if not m:
            fail(f"{cond}: no rows")
            continue
        if all(abs(a - b) <= .002 for a, b in zip(m, tgt)):
            ok(f"32B {cond}: recall {m[0]:.3f} FPR {m[1]:.3f} J {m[2]:.3f}")
        else:
            fail(f"32B {cond}: got {tuple(round(x,3) for x in m)}, expected {tgt}")
    fa = open(f"{RUNS}/trufor_feasibility/analysis.txt", encoding="utf-8",
              errors="replace").read()
    m = re.search(r"AUROC\s*=\s*([0-9.]+)", fa)
    ok(f"TruFor standalone AUROC {m.group(1)}") \
        if m and abs(float(m.group(1)) - .9845) < .002 else \
        fail("TruFor AUROC not 0.9845")

    # X3
    A = {}
    for q in glob.glob(f"{RUNS}/explainability/x3/ann/*.json"):
        try:
            A[os.path.basename(q)[:-5]] = jload(q)
        except Exception:
            pass
    MAP = fr["anonymization_map"]
    for mc, nm, tu, tm in (("M1", "7B", .610, .791), ("M2", "32B", .010, .044)):
        ids = [i for i in A if MAP.get(i, {}).get("label") == "fake"
               and MAP[i]["model_code"] == mc]
        if not ids:
            fail(f"X3 {nm}: no annotations")
            continue
        u = sum(A[i]["overall_semantic_support"] == "unsupported"
                for i in ids) / len(ids)
        man = [c["support"] for i in ids for c in A[i].get("claims", [])
               if c.get("claim_type") == "manipulation"
               and c.get("support") != "not_applicable"]
        mu = sum(s == "unsupported" for s in man) / len(man) if man else -1
        if abs(u - tu) <= .005 and abs(mu - tm) <= .005:
            ok(f"X3 fake {nm}: unsupported {u:.3f}, manipulation unsup {mu:.3f}")
        else:
            fail(f"X3 fake {nm}: unsupported {u:.3f} (want {tu}), "
                 f"manipulation {mu:.3f} (want {tm})")

    # ---------------- 6 invalid runs excluded ----------------
    print("\n[6] INVALID RUNS QUARANTINED")
    inval = glob.glob(f"{RUNS}/**/*INVALID*", recursive=True)
    ok(f"invalid artifacts retained on disk: {len(inval)}") if inval else \
        warn("no INVALID artifacts found — expected the voided semantics run")
    idx = open("docs/EXPERIMENT_INDEX.md", encoding="utf-8").read()
    bad = [os.path.basename(p) for p in inval if os.path.basename(p) in idx]
    ok("no INVALID artifact is cited in the experiment index") if not bad else \
        fail(f"INVALID artifacts referenced in index: {bad}")
    doc = open("docs/INVALID_AND_SUPERSEDED_RUNS.md", encoding="utf-8").read()
    ok("invalid runs are documented") if "INVALID_score_dup" in doc else \
        fail("voided run not documented in INVALID_AND_SUPERSEDED_RUNS.md")

    # ---------------- 7 git safety ----------------
    print("\n[7] GIT SAFETY")
    # core.quotepath=false：否则 git 会把中文文件名转义成 "\346..." 并加引号，
    # 导致后缀判断（.png 等）全部失效，产生假阴性。
    tracked = [l for l in sh("git -c core.quotepath=false ls-files").splitlines()
               if l.strip()]
    if not tracked:
        warn("nothing tracked yet (pre-commit run)")
    BADEXT = (".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".pth.tar")
    w = [f for f in tracked if f.lower().endswith(BADEXT)]
    ok("no model weights tracked") if not w else fail(f"weights tracked: {w[:5]}")
    env = [f for f in tracked if os.path.basename(f).startswith(".env")
           or f.lower().endswith((".key", ".pem")) or "id_rsa" in f]
    ok("no .env / key / pem tracked") if not env else fail(f"secrets tracked: {env}")
    big = []
    for f in tracked:
        try:
            s = os.path.getsize(f)
        except OSError:
            continue
        if s > 20 * 1024 ** 2:
            big.append((s, f))
    ok("no tracked file over 20 MB") if not big else \
        fail(f"large tracked files: {[(f'{s/1e6:.0f}MB', f) for s, f in big[:5]]}")
    ds = [f for f in tracked if f.startswith(("data/", "datasets/"))]
    ok("no raw dataset tracked") if not ds else fail(f"dataset tracked: {ds[:5]}")
    img = [f for f in tracked
           if f.lower().endswith((".png", ".jpg", ".jpeg", ".npz", ".npy"))]
    # reports/figures/ 下的图表是汇报产出物，刻意跟踪；其他图像一律不该进库。
    report_fig = [f for f in img if f.startswith("reports/figures/")]
    stray = [f for f in img if not f.startswith("reports/figures/")]
    if report_fig:
        ok(f"report figures tracked on purpose: {len(report_fig)}")
    ok("no dataset imagery / npz tracked") if not stray else \
        fail(f"{len(stray)} unexpected image/npz files tracked: {stray[:3]}")

    pat = re.compile(
        r"(api[_-]?key|secret[_-]?key|aws_secret|aws_access|ghp_[A-Za-z0-9]{20}"
        r"|sk-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{30}|BEGIN (RSA|OPENSSH) PRIVATE KEY)",
        re.I)
    hits = []
    SELF = os.path.basename(__file__)
    for f in tracked:
        # The scanner's own pattern literals would otherwise match themselves.
        if os.path.basename(f) == SELF:
            continue
        if f.lower().endswith((".md", ".py", ".txt", ".yaml", ".json", ".csv",
                               ".jsonl")):
            try:
                t = open(f, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for mm in pat.finditer(t):
                seg = t[max(0, mm.start() - 40):mm.end() + 40].replace("\n", " ")
                if "max_new_tokens" in seg or "visual_token" in seg:
                    continue
                hits.append((f, mm.group(0)[:24]))
    ok("no credential patterns in tracked files") if not hits else \
        fail(f"possible credentials: {hits[:5]}")

    # ---------------- summary ----------------
    print("\n" + "=" * 78)
    print(f"PASS {len(P)}   WARN {len(W)}   FAIL {len(F)}")
    if W:
        print("\nWARNINGS:")
        for m in W:
            print(f"  - {m}")
    if F:
        print("\nFAILURES — push is blocked:")
        for m in F:
            print(f"  - {m}")
    print("=" * 78)
    sys.exit(1 if F else 0)


if __name__ == "__main__":
    main()
