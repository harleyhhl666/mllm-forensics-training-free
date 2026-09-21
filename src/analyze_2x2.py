"""Merge A1 with the frozen A0/A2/A3 cells and analyse the 2x2.

A0/A2/A3 verdicts are read from their original frozen JSONL files; nothing is
re-inferred. Reports the four simple effects and the factorial interaction on recall,
FPR and Youden J, and confronts the recall-ceiling caveat explicitly.
"""
import json, os, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
CELLS = ("A0", "A1", "A2", "A3")
LAB = {"A0": "A0 score-only", "A1": "A1 score-only + rule",
       "A2": "A2 structured", "A3": "A3 structured + rule"}
DESC = {"A0": "no structured, no rule", "A1": "no structured, RULE",
        "A2": "STRUCTURED, no rule", "A3": "STRUCTURED, RULE"}


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return 0, 0, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_stat(fn, ids, seed=SEED, nb=NBOOT):
    """Bootstrap a statistic that is a function of a resampled source list."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (nb, len(ids)))
    vals = np.empty(nb)
    arr = np.asarray(ids, dtype=object)
    for i in range(nb):
        vals[i] = fn(list(arr[idx[i]]))
    return float(fn(ids)), float(np.percentile(vals, 2.5)), \
        float(np.percentile(vals, 97.5))


def sig(t):
    return "EXCLUDES zero" if (t[1] > 0 or t[2] < 0) else "includes zero"


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    od = os.path.join(R, "trufor_2x2")
    F = json.load(open(os.path.join(od, "trufor_score_semantics_2x2_frozen.json")))

    def load(path, cond):
        out = {}
        for l in open(path):
            d = json.loads(l)
            if d["condition"] == cond:
                out[(d["sample_id"], d["label"])] = d
        return out

    V = {
        "A0": load(os.path.join(R, "trufor_fields", "verdicts.jsonl"),
                   "S0_score_only"),
        "A1": load(os.path.join(od, "verdicts_A1.jsonl"), "A1_score_rule"),
        "A2": load(os.path.join(R, "trufor_semantics", "verdicts.jsonl"),
                   "D0_baseline"),
        "A3": load(os.path.join(R, "trufor_semantics", "verdicts.jsonl"),
                   "D1_binary_rule"),
    }
    ids = sorted({p for (p, l) in V["A0"]}
                 & {p for (p, l) in V["A1"]}
                 & {p for (p, l) in V["A2"]}
                 & {p for (p, l) in V["A3"]})
    ids = [p for p in ids if all((p, l) in V[c] for c in CELLS
                                 for l in ("fake", "real"))]
    m = json.load(open(os.path.join(od, "run_meta_A1.json")))

    print("SCORE x TASK-SEMANTICS 2x2 CAUSAL CONTROL — 7B")
    print(f"complete sources {len(ids)}  bootstrap {NBOOT} seed {SEED}")
    print(f"A1 run: {m['n_inferences']} inf  {m['minutes']} min  "
          f"mean {m['mean_seconds']}s  {m['started']} -> {m['finished']}")
    print(f"A0/A2/A3 read from frozen files; nothing re-inferred")

    print("\n" + "=" * 94)
    print("1. INTEGRITY")
    a1 = list(V["A1"].values())
    print(f"  A1 rows {len(a1)}  parse failures "
          f"{sum(1 for r in a1 if 'json_parse_failed' in r['notes'])}"
          f"  bad verdicts {sum(1 for r in a1 if 'bad_verdict' in r['notes'])}")
    for c in CELLS:
        print(f"  {c} cells {len(V[c])}  unique sources "
              f"{len({p for (p,l) in V[c]})}")
    print(f"  A1 single-image and no structured payload: "
          f"{all(r['n_images']==1 and r['structured'] is False for r in a1)}")
    print(f"  score identical across cells: "
          f"{all(len({V[c][(p,l)]['score_shown'] for c in CELLS})==1 for p in ids for l in ('fake','real'))}")
    print(f"  protocol sha1 {m['protocol_sha1']}")

    fk = lambda c, p, l: int(V[c][(p, l)]["final_verdict"] == "fake")
    rec = lambda c: (lambda ii: float(np.mean([fk(c, p, "fake") for p in ii])))
    fpr = lambda c: (lambda ii: float(np.mean([fk(c, p, "real") for p in ii])))
    jj = lambda c: (lambda ii: float(np.mean([fk(c, p, "fake") for p in ii])
                                     - np.mean([fk(c, p, "real") for p in ii])))

    print("\n" + "=" * 94)
    print("2. FOUR CELLS")
    print(f"  {'cell':<26}{'recall':>9}{'95% CI':>18}{'FPR':>8}{'95% CI':>18}"
          f"{'spec':>7}{'J':>8}{'J 95% CI':>18}")
    SC = {}
    for c in CELLS:
        r = boot_stat(rec(c), ids)
        f = boot_stat(fpr(c), ids)
        j = boot_stat(jj(c), ids)
        SC[c] = dict(recall=r, fpr=f, j=j, spec=1 - f[0])
        print(f"  {LAB[c]:<26}{r[0]:>9.3f} [{r[1]:.3f},{r[2]:.3f}]"
              f"{f[0]:>8.3f} [{f[1]:.3f},{f[2]:.3f}]{1-f[0]:>7.3f}"
              f"{j[0]:>8.3f} [{j[1]:+.3f},{j[2]:+.3f}]")
    print(f"\n  factor layout:")
    for c in CELLS:
        print(f"    {c}  {DESC[c]}")
    print(f"\n  cross-check against the frozen known values:")
    for c, v in F["known_cells"].items():
        print(f"    {c} expected recall {v['recall']:.3f} / FPR {v['fpr']:.3f}  "
              f"observed {SC[c]['recall'][0]:.3f} / {SC[c]['fpr'][0]:.3f}  "
              f"{'OK' if abs(SC[c]['recall'][0]-v['recall'])<0.01 else 'MISMATCH'}")

    print("\n" + "=" * 94)
    print("3. SIMPLE EFFECTS")
    PAIRS = [("rule effect WITHOUT structured", "A1", "A0"),
             ("rule effect WITH structured", "A3", "A2"),
             ("structured effect WITHOUT rule", "A2", "A0"),
             ("structured effect WITH rule", "A3", "A1")]
    EFF = {}
    for nm, hi, lo in PAIRS:
        print(f"\n  {nm}   {hi} - {lo}")
        for w, fn in (("fake recall", rec), ("real FPR", fpr), ("J", jj)):
            d = boot_stat(lambda ii, h=hi, l=lo, g=fn: g(h)(ii) - g(l)(ii), ids)
            EFF[(nm, w)] = d
            line = f"    {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}"
            if w != "J":
                lab = "fake" if w.startswith("fake") else "real"
                _, _, pv = mcnemar([fk(lo, p, lab) for p in ids],
                                   [fk(hi, p, lab) for p in ids])
                line += f"  p={pv:.4g}"
            print(line)

    print("\n" + "=" * 94)
    print("4. FACTORIAL INTERACTION  (A3-A2) - (A1-A0)")
    INT = {}
    for w, fn in (("fake recall", rec), ("real FPR", fpr), ("J", jj)):
        d = boot_stat(lambda ii, g=fn: (g("A3")(ii) - g("A2")(ii))
                      - (g("A1")(ii) - g("A0")(ii)), ids)
        INT[w] = d
        print(f"  {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}")
    print(f"\n  equivalent form (A3-A1) - (A2-A0), must match:")
    for w, fn in (("fake recall", rec), ("real FPR", fpr), ("J", jj)):
        d = boot_stat(lambda ii, g=fn: (g("A3")(ii) - g("A1")(ii))
                      - (g("A2")(ii) - g("A0")(ii)), ids)
        print(f"  {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  "
              f"{'match' if abs(d[0]-INT[w][0])<1e-9 else 'DIFFERS'}")

    print("\n" + "=" * 94)
    print("5. RECALL CEILING — pre-registered caveat")
    print(f"  A0 recall is already {SC['A0']['recall'][0]:.3f}, so the score-only arm")
    print(f"  has at most {1-SC['A0']['recall'][0]:.3f} of head-room. The structured")
    print(f"  arm starts at {SC['A2']['recall'][0]:.3f} with "
          f"{1-SC['A2']['recall'][0]:.3f} of head-room.")
    hr0 = 1 - SC["A0"]["recall"][0]
    hr2 = 1 - SC["A2"]["recall"][0]
    g1 = SC["A1"]["recall"][0] - SC["A0"]["recall"][0]
    g3 = SC["A3"]["recall"][0] - SC["A2"]["recall"][0]
    print(f"  raw gains        A1-A0 {g1:+.3f}   A3-A2 {g3:+.3f}")
    print(f"  head-room        {hr0:.3f}          {hr2:.3f}")
    if hr0 > 0 and hr2 > 0:
        print(f"  gain / head-room {g1/hr0:+.3f}          {g3/hr2:+.3f}")
        print(f"  this normalised view is DESCRIPTIVE only — it is not a pre-registered")
        print(f"  metric and must not replace the raw interaction.")
    print(f"\n  A1 fake errors remaining: "
          f"{sum(1 for p in ids if not fk('A1', p, 'fake'))}/{len(ids)}")
    print(f"  A0 fake errors remaining: "
          f"{sum(1 for p in ids if not fk('A0', p, 'fake'))}/{len(ids)}")

    print("\n" + "=" * 94)
    print("6. DID THE RULE CHANGE POLICY IN THE SCORE-ONLY ARM?")
    print("  the decisive evidence is FPR movement, which has no ceiling problem")
    d = EFF[("rule effect WITHOUT structured", "real FPR")]
    dw = EFF[("rule effect WITH structured", "real FPR")]
    print(f"  A1 - A0 FPR {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}")
    print(f"  A3 - A2 FPR {dw[0]:+.3f} [{dw[1]:+.3f},{dw[2]:+.3f}] {sig(dw)}")
    n_flip_fr = sum(1 for p in ids if not fk("A0", p, "real") and fk("A1", p, "real"))
    n_flip_rf = sum(1 for p in ids if fk("A0", p, "real") and not fk("A1", p, "real"))
    print(f"  real images: A0 real -> A1 fake {n_flip_fr}   A0 fake -> A1 real "
          f"{n_flip_rf}")
    n_ff = sum(1 for p in ids if not fk("A0", p, "fake") and fk("A1", p, "fake"))
    n_fr = sum(1 for p in ids if fk("A0", p, "fake") and not fk("A1", p, "fake"))
    print(f"  fake images: A0 real -> A1 fake {n_ff}   A0 fake -> A1 real {n_fr}")
    tot = sum(1 for p in ids for l in ("fake", "real")
              if fk("A0", p, l) != fk("A1", p, l))
    print(f"  total verdict changes A0 -> A1: {tot}/{2*len(ids)} "
          f"({tot/(2*len(ids)):.3f})")
    print(f"  a non-trivial change count with flat recall means the rule DID shift")
    print(f"  policy in the score-only arm; zero changes would mean it did not.")

    print("\n" + "=" * 94)
    print("7. INTERPRETATION")
    ir = SC["A1"]["recall"][0] - SC["A0"]["recall"][0]
    ifp = EFF[("rule effect WITHOUT structured", "real FPR")]
    a3a1 = EFF[("structured effect WITH rule", "J")]
    a3a1r = EFF[("structured effect WITH rule", "fake recall")]
    print(f"  recall  A0 {SC['A0']['recall'][0]:.3f}  A1 {SC['A1']['recall'][0]:.3f}"
          f"  A2 {SC['A2']['recall'][0]:.3f}  A3 {SC['A3']['recall'][0]:.3f}")
    print(f"  J       A0 {SC['A0']['j'][0]:+.3f}  A1 {SC['A1']['j'][0]:+.3f}"
          f"  A2 {SC['A2']['j'][0]:+.3f}  A3 {SC['A3']['j'][0]:+.3f}")
    print(f"  interaction on J {INT['J'][0]:+.3f} [{INT['J'][1]:+.3f},"
          f"{INT['J'][2]:+.3f}] {sig(INT['J'])}")
    print()
    out = []
    flat_r = abs(ir) < 0.05
    flat_f = not (ifp[1] > 0 or ifp[2] < 0)
    if flat_r and flat_f and INT["J"][1] > 0:
        out.append("RULE MAINLY REPAIRS STRUCTURED FRAMING: A1 ~ A0 in both recall and "
                   "FPR while A3 >> A2, so the clarification acts on the "
                   "framing-induced policy shift rather than on the task generally")
    if flat_r and (ifp[1] > 0):
        out.append("RULE ACTS IN BOTH ARMS: score-only recall is ceiling-bound but FPR "
                   "rises, showing the rule also shifts policy without structured "
                   "evidence")
    if a3a1[1] > 0 or a3a1r[1] > 0:
        out.append("STRUCTURED EVIDENCE ADDS DECISION UTILITY UNDER CORRECT SEMANTICS "
                   "(A3 > A1)")
    elif not (a3a1[1] > 0 or a3a1[2] < 0):
        out.append("A3 ~ A1: structured evidence gives no extra discrimination once "
                   "semantics are correct, so its value is more likely explanation "
                   "faithfulness than added decision utility")
    if a3a1[2] < 0:
        out.append("A3 < A1: structured evidence REDUCES discrimination even under "
                   "correct semantics")
    for o in out:
        print(f"  -> {o}")

    json.dump(dict(cells={c: SC[c] for c in CELLS},
                   simple_effects={f"{k[0]} | {k[1]}": v for k, v in EFF.items()},
                   interaction=INT,
                   ceiling=dict(A0_headroom=hr0, A2_headroom=hr2,
                                A1_minus_A0_recall=g1, A3_minus_A2_recall=g3),
                   n_sources=len(ids)),
              open(os.path.join(od, "analysis_2x2.json"), "w"), indent=1)
    print(f"\n  wrote analysis_2x2.json")


if __name__ == "__main__":
    main()
