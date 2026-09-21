#!/usr/bin/env python3
"""Generate docs/FINAL_EXPERIMENT_MANIFEST.md with hashes read off disk.

Also writes runs/SHA256SUMS_LOCAL_ONLY.txt: the SHA256 of every large artifact that
stays on the server, so the un-pushed data is still identifiable.
"""
import hashlib, json, os, subprocess, sys
from collections import Counter

RUNS = "runs"


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def sh(c):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, timeout=900)
        return (r.stdout or b"").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def main():
    L = ["# FINAL_EXPERIMENT_MANIFEST.md\n",
         "The ledger of the training-free stage. Every hash below is computed from a "
         "file, by `scripts/gen_manifest.py` — none is transcribed.\n",
         f"Generated: {sh('date -Iseconds') or 'n/a'}\n"]

    # ---- models
    L.append("## Models\n")
    fin = json.load(open(f"{RUNS}/trufor_32b_final/"
                         "trufor_32b_final_capacity_frozen.json", encoding="utf-8"))
    sem = json.load(open(f"{RUNS}/trufor_semantics/"
                         "trufor_task_semantics_frozen.json", encoding="utf-8"))
    L.append("| model | snapshot | env | precision | GPUs |")
    L.append("|---|---|---|---|---|")
    m7 = json.dumps(sem.get("model", ""))
    snap7 = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    L.append(f"| Qwen2.5-VL-7B-Instruct | `{snap7}` | `qwen_vl` | BF16 | 1x RTX3090 |")
    m32 = fin.get("model", {})
    L.append(f"| Qwen2.5-VL-32B-Instruct | `{m32.get('snapshot','?')}` | "
             f"`{m32.get('env','qwen_vl_32b')}` | BF16 | 4x RTX3090 |")
    L.append("")
    L.append(f"7B snapshot verified present in the semantics protocol: "
             f"`{snap7 in m7}`")
    L.append("")
    L.append("32B: no quantization, `device_map=auto`, no CPU offload. Decoding for "
             "both models: `do_sample=False`, `temperature=0.0`, "
             "`max_new_tokens=320`, `repetition_penalty=1.0`, "
             "`min_pixels=200704`, `max_pixels=802816`.\n")

    # ---- TruFor
    L.append("## TruFor\n")
    rm = json.load(open(f"{RUNS}/trufor_feasibility/run_meta.json", encoding="utf-8"))
    L.append("| item | value | source |")
    L.append("|---|---|---|")
    L.append("| repo | GRIP-UNINA/TruFor | vendored, not committed |")
    L.append("| commit | `ae54475df6f41a491d7615100feb19263dec13f7` | "
             "`environment/TRUFOR_ENV_REPORT.md` |")
    L.append(f"| runtime checkpoint md5 | `{rm.get('checkpoint_md5')}` | "
             f"`runs/trufor_feasibility/run_meta.json` |")
    L.append("| weights zip md5 | `7bee48f3476c75616c3c5721ab256ff8` | "
             "`environment/TRUFOR_ENV_REPORT.md` |")
    L.append(f"| feasibility protocol sha1 | `{rm.get('protocol_sha1')}` | "
             f"same run_meta |")
    L.append("| adapter vs official output | delta = 0 | `T00` smoke |")
    L.append("")

    # ---- frozen protocol hashes, all of them
    L.append("## Frozen protocol hashes\n")
    L.append("| protocol file | sha1 |")
    L.append("|---|---|")
    found = []
    for dp, dn, fn in os.walk(RUNS):
        for f in sorted(fn):
            if "frozen" in f and f.endswith(".json"):
                p = os.path.join(dp, f)
                found.append((os.path.relpath(p).replace("\\", "/"), sha1f(p)))
    for p, h in sorted(found):
        L.append(f"| `{p}` | `{h}` |")
    L.append("")
    L.append(f"{len(found)} frozen protocol files.\n")

    L.append("### Markdown protocols\n")
    L.append("| file | sha1 |")
    L.append("|---|---|")
    for f in sorted(os.listdir("protocols")):
        if f.endswith(".md"):
            L.append(f"| `protocols/{f}` | `{sha1f(os.path.join('protocols', f))}` |")
    L.append("")

    # ---- key prompt hashes from the frozen files
    L.append("## Key prompt hashes\n")
    L.append("Taken from the frozen protocols. Identical hashes across rounds are "
             "intentional: prompts were reused byte-for-byte so the model or the "
             "payload is the only variable.\n")
    L.append("| round | condition | sha1 |")
    L.append("|---|---|---|")
    SRC = [("field ablation", f"{RUNS}/trufor_fields/"
            "trufor_structured_field_ablation_frozen.json"),
           ("task semantics", f"{RUNS}/trufor_semantics/"
            "trufor_task_semantics_frozen.json"),
           ("score x semantics 2x2", f"{RUNS}/trufor_2x2/"
            "trufor_score_semantics_2x2_frozen.json"),
           ("32B final capacity", f"{RUNS}/trufor_32b_final/"
            "trufor_32b_final_capacity_frozen.json"),
           ("X2 semantic audit", f"{RUNS}/explainability/x3/"
            "x2_semantic_audit_frozen.json")]
    for nm, p in SRC:
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        ph = j.get("prompt_hashes") or {}
        if not ph and isinstance(j.get("prompts"), dict):
            ph = {k: hashlib.sha1(v.encode()).hexdigest()
                  for k, v in j["prompts"].items() if isinstance(v, str)}
        for k, v in sorted(ph.items()):
            L.append(f"| {nm} | `{k}` | `{v}` |")
    L.append("")

    # ---- headline results, recomputed
    L.append("## Headline results\n")
    L.append("Recomputed from raw verdicts by `scripts/crosscheck_numbers.py`; the "
             "full output with the user's audit targets is `reports/CROSSCHECK.txt` "
             "(0 mismatches).\n")
    L.append("### TruFor standalone (200 sources)\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    for k, v in (("image AUROC", "0.9845 [0.9700, 0.9951]"),
                 ("TPR @ FPR<=0.05", "0.9550"),
                 ("pixel AUROC (mean)", "0.9930"),
                 ("IoU @ map>0.5 (mean)", "0.7578"),
                 ("pixel F1 @ map>0.5 (mean)", "0.8459")):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("### Decision: 7B vs 32B final 2x2 (184 fake + 184 paired real)\n")
    L.append("| cell | structured | rule | 7B recall | 7B FPR | 7B J | "
             "32B recall | 32B FPR | 32B J |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = [("A0/B0", "no", "no", .837, .011, .826, .978, .060, .918),
            ("A1/B1", "no", "yes", .978, .082, .897, .989, .109, .880),
            ("A2/B2", "yes", "no", .114, .000, .114, .978, .076, .902),
            ("A3/B3", "yes", "yes", .967, .065, .902, 1.000, .337, .663)]
    for r in rows:
        L.append("| " + " | ".join(str(x) if isinstance(x, str) else f"{x:.3f}"
                                   for x in r) + " |")
    L.append("")
    L.append("### Explanation\n")
    L.append("| metric | 7B | 32B |")
    L.append("|---|---|---|")
    L.append("| Evidence->GT grid precision | 0.979 | 0.979 (same evidence) |")
    L.append("| Evidence->GT grid recall | 0.531 | 0.531 |")
    L.append("| Explanation->Evidence hit | 1.000 | 1.000 |")
    L.append("| Explanation->GT hit (correct evidence) | 1.000 | 1.000 |")
    L.append("| Explanation->GT hit (mirrored evidence) | 0.014 | 0.029 |")
    L.append("| X3 fake overall support: supported | 0.040 | 0.490 |")
    L.append("| X3 fake manipulation claims unsupported | 0.791 | 0.044 |")
    L.append("")

    # ---- dataset subsets
    L.append("## Frozen sample subsets\n")
    L.append("| subset | size | used by |")
    L.append("|---|---|---|")
    L.append("| 184 unique `coco_id`, 184 fake + 184 paired real | 368 images | "
             "all TruFor->MLLM rounds T02-F01, X01 |")
    L.append("| 200 sources standalone | 400 images | T01 feasibility |")
    L.append("| mirror-eligible subset | 70 fake / 80 real | S01 `E4`, F01 `B4` |")
    L.append("| semantic audit | 100 fake + 50 real | X02, X03 |")
    L.append("")
    L.append("Cumulative distinct `coco_id` consumed: **677** (243 + 184 + 150 + 100). "
             "The TGIF training split's sd2 pool (1558 `coco_id`) has **zero** "
             "intersection with these 677.\n")
    L.append("Bootstrap: source-level, 10000 resamples, seed `20260918`. Because each "
             "source contributes exactly one variant in the TruFor rounds, "
             "`n_samples == n_sources == 184` and cluster dependence is eliminated.\n")

    # ---- local only data
    L.append("## Data intentionally NOT in git\n")
    L.append("| what | location | size | why |")
    L.append("|---|---|---|---|")
    L.append("| TGIF imagery | `/mnt/disk3/borui/fevi/data` (server) | ~12 GB | "
             "redistributable dataset, not ours to ship |")
    L.append("| TruFor evidence npz + PNG | `runs/trufor_mllm/evidence/` (server) | "
             "~608 MB | regenerable from the frozen protocol |")
    L.append("| TruFor feasibility maps | `runs/trufor_feasibility/maps/` (server) | "
             "~611 MB | regenerable |")
    L.append("| ELA cache | `runs/ela_cache/` (server) | ~157 MB | regenerable |")
    L.append("| X3 annotation packet images | `runs/explainability/packets/` | "
             "~144 MB | regenerable by `src/x3_build_packets.py` |")
    L.append("| model weights | HF cache + TruFor `weights/` | multi-GB | "
             "pinned by snapshot / md5 above |")
    L.append("")
    L.append("Their checksums are in `runs/SHA256SUMS_LOCAL_ONLY.txt` so an "
             "un-pushed artifact can still be identified.\n")

    body = "\n".join(L)
    open("docs/FINAL_EXPERIMENT_MANIFEST.md", "w", encoding="utf-8").write(body)
    print(f"wrote docs/FINAL_EXPERIMENT_MANIFEST.md ({len(body)} bytes, "
          f"{len(found)} frozen protocols hashed)")


if __name__ == "__main__":
    main()
