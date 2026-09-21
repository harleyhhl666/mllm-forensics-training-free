"""Stage-V capacity ablation analysis: 7B vs 32B, two separate analyses.

Analysis A -- strict historical pairing: A_fake_correct, B_fake_donor and
  C_legacy (real + fake-counterpart ELA). Inputs are byte-identical to what the
  7B M1 run consumed, so 7B-vs-32B is a clean paired contrast.

Analysis B -- semantically correct: A_fake_correct, B_fake_donor and C_own
  (real + its own ELA). This is the primary scientific analysis; the support
  discrimination metric uses C_own, never the legacy control.

Every sample is one coco_id (100 sources, one variant each), so cluster bootstrap
and plain paired bootstrap coincide; both are reported once and noted.
"""
import json, os, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
A, B = "A_fake_correct", "B_fake_donor"
CL, CO = "C_legacy_real_fakecounterpart", "C_own_real_own"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(vals, nb=NBOOT, seed=SEED):
    """Paired bootstrap over sources (one source per sample)."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (nb, len(v)))
    d = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load(path, key="condition"):
    R = defaultdict(dict)
    if not os.path.exists(path):
        return R
    for l in open(path):
        r = json.loads(l)
        R[r["pair_id"]][r[key]] = r
    return R


def merge(*srcs):
    out = defaultdict(dict)
    for s in srcs:
        for pid, d in s.items():
            out[pid].update(d)
    return out


def rates(rows):
    n = len(rows)
    if not n:
        return None
    m = sum(1 for r in rows if r["image_consistency"] == "matched")
    su = sum(1 for r in rows if r["forensic_support"] == "supported")
    st = Counter(r["state"] for r in rows)
    return dict(n=n, matched=m / n, mismatched=1 - m / n, supported=su / n,
                insufficient=1 - su / n, A=st.get("A", 0), B=st.get("B", 0),
                C=st.get("C", 0))


def show_block(title, D, pids, conds, labels):
    print(f"\n  {title}")
    print(f"    {'condition':<34}{'n':>4}{'matched':>9}{'mismatch':>10}"
          f"{'supported':>11}{'insuff':>9}   A/B/C")
    out = {}
    for c, lab in zip(conds, labels):
        rows = [D[p][c] for p in pids if c in D[p]]
        r = rates(rows)
        out[c] = r
        if not r:
            print(f"    {lab:<34} MISSING")
            continue
        print(f"    {lab:<34}{r['n']:>4}{r['matched']:>9.3f}{r['mismatched']:>10.3f}"
              f"{r['supported']:>11.3f}{r['insufficient']:>9.3f}   "
              f"{r['A']}/{r['B']}/{r['C']}")
    return out


def delta(D7, D32, pids, ca, cb, field, want):
    """Per-source difference of an indicator between two conditions, per model."""
    res = {}
    for nm, D in (("7B", D7), ("32B", D32)):
        v = []
        for p in pids:
            if ca in D[p] and cb in D[p]:
                v.append(int(D[p][ca][field] == want) - int(D[p][cb][field] == want))
        res[nm] = boot(v) + (len(v),)
    return res


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    hist = load(os.path.join(R, "mitigation_run", "stage_v.jsonl"), key="cue")
    # historical rows are keyed (label,cue); rebuild with condition names
    H7 = defaultdict(dict)
    for l in open(os.path.join(R, "mitigation_run", "stage_v.jsonl")):
        r = json.loads(l)
        k = {("fake", "correct"): A, ("fake", "donor"): B,
             ("real", "correct"): CL}.get((r["label"], r["cue"]))
        if k:
            H7[r["pair_id"]][k] = r
    N7 = load(os.path.join(R, "qwen7b_stagev_cown", "stage_v.jsonl"))
    D7 = merge(H7, N7)
    D32 = load(os.path.join(R, "qwen32b_stagev_ablation", "stage_v.jsonl"))

    print("STAGE-V CAPACITY ABLATION — Qwen2.5-VL 7B vs 32B")
    print(f"bootstrap {NBOOT} resamples, seed {SEED}")

    print("\n" + "=" * 90)
    print("1. INTEGRITY")
    n32 = sum(len(v) for v in D32.values())
    n7new = sum(len(v) for v in N7.values())
    pf32 = sum(1 for v in D32.values() for r in v.values()
               if "stage_v_parse_failed" in (r["notes"] or []))
    inv32 = sum(1 for v in D32.values() for r in v.values()
                if "invalid_stage_v_to_state_C" in (r["notes"] or []))
    pf7 = sum(1 for v in N7.values() for r in v.values()
              if "stage_v_parse_failed" in (r["notes"] or []))
    print(f"  32B rows                  : {n32} (expect 400)")
    print(f"  7B new C_own rows         : {n7new} (expect 100)")
    print(f"  32B parse failures        : {pf32}")
    print(f"  32B invalid -> State C    : {inv32}")
    print(f"  7B  parse failures        : {pf7}")
    cnt32 = Counter(c for v in D32.values() for c in v)
    print(f"  32B per-condition counts  : {dict(cnt32)}")
    # input identity: 32B legacy vs 7B history
    same = tot = 0
    for p, v in D32.items():
        for c in (A, B, CL):
            if c in v and c in H7[p]:
                tot += 1
                same += int(v[c]["visualization_sha1"] ==
                            H7[p][c]["visualization_sha1"])
    print(f"  32B/7B identical input hashes (A,B,C_legacy): {same}/{tot}")
    dist = sum(1 for p, v in D32.items() if CO in v and CL in v
               and v[CO]["visualization_sha1"] != v[CL]["visualization_sha1"])
    print(f"  C_own visualization differs from C_legacy   : {dist}/"
          f"{sum(1 for v in D32.values() if CO in v and CL in v)}")
    ph = {r["prompt_sha1"] for v in D32.values() for r in v.values()} | \
         {r["prompt_sha1"] for v in N7.values() for r in v.values()}
    print(f"  distinct Stage-V prompt hashes across all runs: {len(ph)} -> "
          f"{'IDENTICAL' if len(ph) == 1 else 'MISMATCH'}")

    pidsA = sorted(p for p in D32 if all(c in D32[p] for c in (A, B, CL))
                   and all(c in D7[p] for c in (A, B, CL)))
    pidsB = sorted(p for p in D32 if all(c in D32[p] for c in (A, B, CO))
                   and all(c in D7[p] for c in (A, B, CO)))
    print(f"  Analysis A complete sources: {len(pidsA)}")
    print(f"  Analysis B complete sources: {len(pidsB)}")
    print(f"  one variant per coco_id -> cluster bootstrap == paired bootstrap: "
          f"{len({D32[p][A]['coco_id'] for p in pidsA}) == len(pidsA)}")

    LAB_A = ["A  fake + correct ELA", "B  fake + donor ELA",
             "C_legacy real + fake-counterpart ELA"]
    LAB_B = ["A  fake + correct ELA", "B  fake + donor ELA",
             "C_own  real + its OWN ELA"]

    for title, pids, conds, labs in (
            ("ANALYSIS A — STRICT HISTORICAL PAIRING", pidsA, (A, B, CL), LAB_A),
            ("ANALYSIS B — SEMANTICALLY CORRECT (primary)", pidsB, (A, B, CO), LAB_B)):
        print("\n" + "=" * 90)
        print(f"{'2' if 'A —' in title else '3'}. {title}   n={len(pids)}")
        r7 = show_block("7B", D7, pids, conds, labs)
        r32 = show_block("32B", D32, pids, conds, labs)
        print("\n    paired changes per condition (7B -> 32B):")
        for c, lab in zip(conds, labs):
            mm = sum(1 for p in pids if D7[p][c]["image_consistency"] == "mismatched"
                     and D32[p][c]["image_consistency"] == "matched")
            mb = sum(1 for p in pids if D7[p][c]["image_consistency"] == "matched"
                     and D32[p][c]["image_consistency"] == "mismatched")
            si = sum(1 for p in pids if D7[p][c]["forensic_support"] == "insufficient"
                     and D32[p][c]["forensic_support"] == "supported")
            sb = sum(1 for p in pids if D7[p][c]["forensic_support"] == "supported"
                     and D32[p][c]["forensic_support"] == "insufficient")
            d, lo, hi = boot([int(D32[p][c]["image_consistency"] == "matched") -
                              int(D7[p][c]["image_consistency"] == "matched")
                              for p in pids])
            _, _, pm = mcnemar(
                [int(D7[p][c]["image_consistency"] == "matched") for p in pids],
                [int(D32[p][c]["image_consistency"] == "matched") for p in pids])
            ds, dlo, dhi = boot([int(D32[p][c]["forensic_support"] == "supported") -
                                 int(D7[p][c]["forensic_support"] == "supported")
                                 for p in pids])
            _, _, ps = mcnemar(
                [int(D7[p][c]["forensic_support"] == "supported") for p in pids],
                [int(D32[p][c]["forensic_support"] == "supported") for p in pids])
            print(f"      {lab}")
            print(f"        mismatched->matched {mm:>3}   matched->mismatched {mb:>3}"
                  f"   dMatched {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pm:.4g}")
            print(f"        insuff->supported   {si:>3}   supported->insuff   {sb:>3}"
                  f"   dSupported {ds:+.3f} [{dlo:+.3f},{dhi:+.3f}] p={ps:.4g}")

        print("\n    DISCRIMINATION METRICS")
        dm = delta(D7, D32, pids, conds[0], conds[1], "image_consistency", "matched")
        for nm in ("7B", "32B"):
            o, lo, hi, nn = dm[nm]
            print(f"      dMatch  ({nm:<3}) = P(matched|correct) - P(matched|donor)  "
                  f"{o:+.3f} [{lo:+.3f},{hi:+.3f}]  n={nn}")
        gain = boot([(int(D32[p][conds[0]]["image_consistency"] == "matched") -
                      int(D32[p][conds[1]]["image_consistency"] == "matched")) -
                     (int(D7[p][conds[0]]["image_consistency"] == "matched") -
                      int(D7[p][conds[1]]["image_consistency"] == "matched"))
                     for p in pids])
        print(f"      dMatch(32B) - dMatch(7B) = {gain[0]:+.3f} "
              f"[{gain[1]:+.3f},{gain[2]:+.3f}] -> "
              f"{'EXCLUDES zero' if (gain[1] > 0 or gain[2] < 0) else 'includes zero'}")
        ds = delta(D7, D32, pids, conds[0], conds[2], "forensic_support", "supported")
        for nm in ("7B", "32B"):
            o, lo, hi, nn = ds[nm]
            print(f"      dSupport({nm:<3}) = P(supported|fake-correct) - "
                  f"P(supported|{'legacy' if conds[2]==CL else 'real-own'})  "
                  f"{o:+.3f} [{lo:+.3f},{hi:+.3f}]  n={nn}")
        g2 = boot([(int(D32[p][conds[0]]["forensic_support"] == "supported") -
                    int(D32[p][conds[2]]["forensic_support"] == "supported")) -
                   (int(D7[p][conds[0]]["forensic_support"] == "supported") -
                    int(D7[p][conds[2]]["forensic_support"] == "supported"))
                   for p in pids])
        print(f"      dSupport(32B) - dSupport(7B) = {g2[0]:+.3f} "
              f"[{g2[1]:+.3f},{g2[2]:+.3f}] -> "
              f"{'EXCLUDES zero' if (g2[1] > 0 or g2[2] < 0) else 'includes zero'}")
        if "primary" in title:
            globals()["_PB"] = (r7, r32, dm, ds, gain, g2, pids)

    print("\n" + "=" * 90)
    print("4. REAL-OWN STRESS TEST (C_own, semantically correct)")
    for nm, D in (("7B", D7), ("32B", D32)):
        rows = [D[p][CO] for p in pidsB if CO in D[p]]
        r = rates(rows)
        print(f"  {nm:<4} n={r['n']}  P(matched)={r['matched']:.3f}  "
              f"P(supported)={r['supported']:.3f}  "
              f"P(insufficient)={r['insufficient']:.3f}  A/B/C={r['A']}/{r['B']}/{r['C']}")
    print("  ideal: matched is acceptable, but support should stay mostly insufficient")

    print("\n" + "=" * 90)
    print("5. PRE-REGISTERED CASE JUDGEMENT (from Analysis B)")
    r7, r32, dm, ds, gain, g2, pids = _PB
    mc7, mc32 = r7[A]["matched"], r32[A]["matched"]
    md7, md32 = r7[B]["matched"], r32[B]["matched"]
    sc7, sc32 = r7[A]["supported"], r32[A]["supported"]
    so7, so32 = r7[CO]["supported"], r32[CO]["supported"]
    print(f"  P(matched|correct)    7B {mc7:.3f} -> 32B {mc32:.3f}")
    print(f"  P(matched|donor)      7B {md7:.3f} -> 32B {md32:.3f}")
    print(f"  P(supported|correct)  7B {sc7:.3f} -> 32B {sc32:.3f}")
    print(f"  P(supported|real-own) 7B {so7:.3f} -> 32B {so32:.3f}")
    print(f"  dMatch   7B {dm['7B'][0]:+.3f} -> 32B {dm['32B'][0]:+.3f}"
          f"   (gain {gain[0]:+.3f} [{gain[1]:+.3f},{gain[2]:+.3f}])")
    print(f"  dSupport 7B {ds['7B'][0]:+.3f} -> 32B {ds['32B'][0]:+.3f}"
          f"   (gain {g2[0]:+.3f} [{g2[1]:+.3f},{g2[2]:+.3f}])")
    print()
    sel_match = gain[1] > 0
    sel_supp = g2[1] > 0
    permissive = (sc32 > sc7 + 0.2) and (r32[B]["supported"] > r7[B]["supported"] + 0.2) \
        and (so32 > so7 + 0.2)
    flat = (abs(dm["32B"][0]) < 0.10 and r32[A]["supported"] < 0.10)
    if permissive:
        print("  -> CASE 3: indiscriminate permissiveness. 32B grants support across")
        print("     correct, donor AND real-own; discrimination has not improved.")
        print("     Do NOT proceed to full 32B M1.")
    elif sel_match or sel_supp:
        print("  -> CASE 1: selective improvement -> the Stage-V failure contains a")
        print("     substantial model-capacity component.")
    elif flat:
        print("  -> CASE 2: no meaningful capacity improvement; model size alone does")
        print("     not resolve the Stage-V verification failure. Stop scaling.")
    else:
        print("  -> mixed/indeterminate: report descriptively, do not force a case.")

    print("\n" + "=" * 90)
    print("6. RUNTIME")
    for tag in ("qwen32b_stagev_ablation", "qwen7b_stagev_cown"):
        mp = os.path.join(R, tag, "run_meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        m = m if isinstance(m, list) else [m]
        for x in m:
            print(f"  {x['model_label']:<28} {x['n_inferences']:>4} inf  "
                  f"{x['total_minutes']:>6} min  mean {x['mean_seconds']}s  "
                  f"peak {x.get('peak_GB_per_gpu')}")


if __name__ == "__main__":
    main()
