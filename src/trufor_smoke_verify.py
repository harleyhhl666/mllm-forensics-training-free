"""Verify the adapter against the official script, and render smoke visualizations.

Two jobs:
 1. assert adapter outputs == official trufor_test.py outputs (score, map, conf)
 2. save per-image PNG panels: original | localization map | reliability map
    GT mask is rendered in a SEPARATE panel for human inspection only; it is never
    fed to the model and never used to alter any output.

Colour handling is recorded explicitly: maps are already probabilities in [0,1] and
are displayed with a FIXED 0..1 scale, not min-max stretched, so the picture cannot
exaggerate a weak response.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SMOKE = "/mnt/disk3/borui/fevi/trufor_smoke"
OFF = os.path.join(SMOKE, "output")
ADA = os.path.join(SMOKE, "adapter_out")
VIS = os.path.join(SMOKE, "vis")
TGIF = "/mnt/disk3/borui/fevi/data/TGIF"


def main():
    man = {os.path.basename(m["dst"]): m for m in
           json.load(open(os.path.join(SMOKE, "manifest.json")))}
    ada = json.load(open(os.path.join(ADA, "trufor_results.json")))
    os.makedirs(VIS, exist_ok=True)

    print("=" * 84)
    print("ADAPTER vs OFFICIAL SCRIPT EQUIVALENCE")
    ok = True
    for r in ada["results"]:
        base = os.path.basename(r["image_path"])
        off = np.load(os.path.join(OFF, base + ".npz"))
        stem = os.path.splitext(base)[0]
        mine = np.load(os.path.join(ADA, stem + "_trufor.npz"))
        ds = abs(float(off["score"]) - r["integrity_score"])
        dm = float(np.abs(off["map"] - mine["map"]).max())
        dc = float(np.abs(off["conf"] - mine["conf"]).max())
        good = ds < 1e-9 and dm < 1e-6 and dc < 1e-6
        ok &= good
        print(f"  {base:<44} |dscore|={ds:.3e} |dmap|max={dm:.3e} "
              f"|dconf|max={dc:.3e}  {'OK' if good else 'MISMATCH'}")
    print(f"  adapter reproduces the official pipeline: {ok}")

    print("\n" + "=" * 84)
    print("SMOKE RESULTS (interface check only -- 4 images prove nothing about "
          "performance)")
    print(f"  {'image':<44}{'GT':<6}{'score':>9}{'map>0.5 px%':>13}"
          f"{'conf mean':>11}")
    rows = []
    for r in ada["results"]:
        base = os.path.basename(r["image_path"])
        m = man[base]
        stem = os.path.splitext(base)[0]
        z = np.load(os.path.join(ADA, stem + "_trufor.npz"))
        loc, conf = z["map"], z["conf"]
        frac = float((loc > 0.5).mean())
        print(f"  {base:<44}{m['gt']:<6}{r['integrity_score']:>9.4f}"
              f"{frac*100:>12.3f}%{conf.mean():>11.4f}")
        rows.append(dict(image=base, gt=m["gt"], score=r["integrity_score"],
                         frac_map_gt_half=frac, conf_mean=float(conf.mean()),
                         loc_max=float(loc.max()), pair_id=m["pair_id"],
                         tamper_ratio=m["tamper_ratio"]))

        # ---- visualization
        img = np.array(Image.open(r["image_path"]).convert("RGB"))
        gtm = None
        mp = os.path.join(TGIF, m["mask"])
        if m["gt"] == "fake" and os.path.exists(mp):
            gg = np.array(Image.open(mp).convert("L"))
            gtm = (gg > 127).astype(float)
        n = 4 if gtm is not None else 3
        fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
        ax[0].imshow(img)
        ax[0].set_title(f"input ({m['gt']})\nscore={r['integrity_score']:.4f}")
        im1 = ax[1].imshow(loc, cmap="inferno", vmin=0.0, vmax=1.0)
        ax[1].set_title("localization map\nsoftmax[1], fixed scale 0-1")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)
        im2 = ax[2].imshow(conf, cmap="viridis", vmin=0.0, vmax=1.0)
        ax[2].set_title("reliability (conf) map\nsigmoid, fixed scale 0-1")
        plt.colorbar(im2, ax=ax[2], fraction=0.046)
        if gtm is not None:
            ax[3].imshow(gtm, cmap="gray", vmin=0, vmax=1)
            ax[3].set_title("GT mask (human check ONLY\nnot a model input)")
        for a in ax:
            a.axis("off")
        fig.suptitle(base, fontsize=9)
        fig.tight_layout()
        fp = os.path.join(VIS, stem + "_panel.png")
        fig.savefig(fp, dpi=95)
        plt.close(fig)

    json.dump(dict(adapter_matches_official=bool(ok), rows=rows,
                   colormap_note="localization: inferno, reliability: viridis, both "
                                 "displayed on a FIXED 0-1 scale; no min-max "
                                 "normalisation, so displayed intensity equals the "
                                 "model probability",
                   gt_usage="GT mask shown in a separate panel for human inspection "
                            "only; never a model input, never used to modify outputs"),
              open(os.path.join(SMOKE, "smoke_summary.json"), "w"), indent=1)
    print(f"\n  panels -> {VIS}")


if __name__ == "__main__":
    main()
