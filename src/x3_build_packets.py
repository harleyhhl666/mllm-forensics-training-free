"""Build the annotation packets frozen in Phase X2. Images only, no annotation.

Per fake unit: A the photograph, B the paired source, C the binary mask,
D the photograph with the masked region outlined.
Per real unit: A the photograph.

Filenames carry ONLY the anonymous EXPL id, never the source id, label or model,
so the annotator cannot infer condition or model identity from a path.
"""
import json, os, sys
import numpy as np, yaml
from PIL import Image, ImageDraw


def outline(img_path, mask_path, out_path, thr=127):
    im = Image.open(img_path).convert("RGB")
    m = np.array(Image.open(mask_path).convert("L").resize(im.size)) > thr
    d = ImageDraw.Draw(im)
    # outline = mask pixels adjacent to non-mask, drawn as a 2px red band
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    edge = m & ~er
    ys, xs = np.nonzero(edge)
    for x, y in zip(xs, ys):
        d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=(255, 0, 0))
    im.save(out_path, quality=92)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    root = cfg["datasets"]["tgif_sp"]["root"]
    thr = cfg["datasets"].get("mask_binarize_threshold", 127)
    fr = json.load(open(os.path.join(R, "explainability",
                                     "x2_semantic_audit_frozen.json")))
    conf = json.load(open(os.path.join(
        R, "trufor_conflict", "trufor_score_map_conflict_frozen.json")))
    SM = {s["sample_id"]: s for s in conf["samples"]}

    od = os.path.join(R, "explainability", "packets")
    os.makedirs(od, exist_ok=True)
    man = []
    for u in fr["annotation_units"]:
        a = u["anon_id"]
        s = SM[u["source"]]
        rec = dict(anon_id=a, label=u["label"], explanation=u["explanation"],
                   referenced_regions=u["referenced_regions"])
        if u["label"] == "fake":
            fp, rp = os.path.join(root, s["fake"]), os.path.join(root, s["real"])
            mp = os.path.join(root, s["mask"])
            pa = os.path.join(od, f"{a}_A.jpg")
            pb = os.path.join(od, f"{a}_B.jpg")
            pc = os.path.join(od, f"{a}_C.png")
            pd = os.path.join(od, f"{a}_D.jpg")
            Image.open(fp).convert("RGB").save(pa, quality=92)
            Image.open(rp).convert("RGB").save(pb, quality=92)
            m = (np.array(Image.open(mp).convert("L")) > thr).astype(np.uint8) * 255
            Image.fromarray(m).save(pc)
            outline(fp, mp, pd, thr)
            rec["images"] = dict(A=pa, B=pb, C=pc, D=pd)
        else:
            rp = os.path.join(root, s["real"])
            pa = os.path.join(od, f"{a}_A.jpg")
            Image.open(rp).convert("RGB").save(pa, quality=92)
            rec["images"] = dict(A=pa)
        man.append(rec)
    mp_out = os.path.join(R, "explainability", "packets_manifest.json")
    json.dump(man, open(mp_out, "w"), indent=1)
    n_img = sum(len(r["images"]) for r in man)
    print(f"units {len(man)}  images {n_img}")
    print(f"manifest {mp_out}")
    sz = sum(os.path.getsize(p) for r in man for p in r["images"].values())
    print(f"total bytes {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()
