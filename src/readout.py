"""Blind Gate-1 read-out: frozen candidate families A, B, C.

Every function here sees ONLY an ELA anomaly map (and, for family C, a real-null
statistic built from calibration REAL images). No GT mask is ever an input.

Output contract for every read-out (per the protocol): not just a scalar, but
score + strongest candidate region + the scale it was found at.

FROZEN SEARCH SPACE -- declared before looking at any result, not extended later:
  A  multi_scale_scan      scales = {0.02, 0.05, 0.10, 0.20} area fractions
  B  region_contrast       region rule in {otsu, p99, p995}
  C  real_null_multiscale  = A + per-scale standardization against a real-null
                             distribution (mean/std of the max window response
                             observed on calibration REAL images at that scale)
Family C is the primary hypothesis: A supplies scale-matched localization, C
removes the area-dilution problem that made whole-image statistics fail.
"""
import numpy as np
import cv2

SCALES = (0.02, 0.05, 0.10, 0.20)          # FROZEN area fractions
REGION_RULES = ("otsu", "p99", "p995")     # FROZEN region rules


def _win_for_scale(h, w, frac):
    """Square window whose area is `frac` of the image area, odd-sized, >=8 px."""
    side = int(round(np.sqrt(h * w * frac)))
    side = max(8, min(side, min(h, w)))
    return side


def _box_mean(m, k):
    return cv2.blur(m, (k, k), borderType=cv2.BORDER_REFLECT)


def _argmax_xy(a):
    i = int(np.argmax(a))
    return i % a.shape[1], i // a.shape[1]


def _bbox_from_center(cx, cy, side, h, w):
    half = side // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    return [int(x1), int(y1), int(x2), int(y2)]


# ------------------------------------------------------------------ family A

def multi_scale_scan(m, scales=SCALES):
    """Strongest local response across frozen scales, standardized WITHIN the
    same scale (so a small hot region is not diluted by the whole image)."""
    h, w = m.shape
    best = dict(score=-np.inf, scale=None, bbox=None)
    per_scale = {}
    for f in scales:
        k = _win_for_scale(h, w, f)
        bm = _box_mean(m, k)
        mu, sd = float(bm.mean()), float(bm.std()) + 1e-6
        z = (float(bm.max()) - mu) / sd
        cx, cy = _argmax_xy(bm)
        per_scale[f] = z
        if z > best["score"]:
            best = dict(score=z, scale=f, bbox=_bbox_from_center(cx, cy, k, h, w))
    return dict(score=float(best["score"]), candidate_scale=best["scale"],
                candidate_bbox=best["bbox"], per_scale=per_scale,
                readout="multi_scale_scan")


# ------------------------------------------------------------------ family B

def region_contrast(m, rule="otsu"):
    """Segment a candidate region from the map itself, then score inside-vs-outside
    contrast. Blind: the region comes from the map, never from GT."""
    h, w = m.shape
    x = np.clip(m, 0, 1)
    if rule == "otsu":
        u8 = (x * 255).astype(np.uint8)
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = t / 255.0
    elif rule == "p99":
        thr = float(np.percentile(x, 99.0))
    elif rule == "p995":
        thr = float(np.percentile(x, 99.5))
    else:
        raise ValueError(rule)

    reg = x > thr
    if reg.sum() < 16 or (~reg).sum() < 16:
        return dict(score=0.0, candidate_scale=None, candidate_bbox=None,
                    readout=f"region_contrast_{rule}")
    # keep the largest connected component as the candidate region
    n, lab = cv2.connectedComponents(reg.astype(np.uint8))
    if n > 1:
        sizes = [(lab == i).sum() for i in range(1, n)]
        keep = 1 + int(np.argmax(sizes))
        reg = lab == keep
    inside = x[reg]; outside = x[~reg]
    sd = float(outside.std()) + 1e-6
    score = float((inside.mean() - outside.mean()) / sd)
    ys, xs = np.where(reg)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    area_frac = float(reg.mean())
    return dict(score=score, candidate_scale=area_frac, candidate_bbox=bbox,
                readout=f"region_contrast_{rule}")


# ------------------------------------------------------------------ family C

def build_real_null(real_maps, scales=SCALES):
    """Per-scale null distribution of the max window response, from REAL images.

    For each scale we record the mean and std ACROSS REAL IMAGES of the
    within-image standardized max response. A fake image's z is then expressed in
    units of how unusual it is for a genuine photograph at that same scale.
    """
    per = {f: [] for f in scales}
    for m in real_maps:
        m = np.asarray(m, np.float32)
        h, w = m.shape
        for f in scales:
            k = _win_for_scale(h, w, f)
            bm = _box_mean(m, k)
            mu, sd = float(bm.mean()), float(bm.std()) + 1e-6
            per[f].append((float(bm.max()) - mu) / sd)
    null = {}
    for f, v in per.items():
        v = np.asarray(v, float)
        null[f] = dict(mean=float(v.mean()), std=float(v.std()) + 1e-6,
                       n=int(v.size),
                       p95=float(np.percentile(v, 95)),
                       p99=float(np.percentile(v, 99)))
    return null


def real_null_multiscale(m, null, scales=SCALES):
    """Family C: multi-scale scan standardized against the real-null per scale."""
    h, w = m.shape
    best = dict(score=-np.inf, scale=None, bbox=None)
    per_scale = {}
    for f in scales:
        if f not in null:
            continue
        k = _win_for_scale(h, w, f)
        bm = _box_mean(m, k)
        mu, sd = float(bm.mean()), float(bm.std()) + 1e-6
        z_in = (float(bm.max()) - mu) / sd
        z = (z_in - null[f]["mean"]) / null[f]["std"]
        cx, cy = _argmax_xy(bm)
        per_scale[f] = z
        if z > best["score"]:
            best = dict(score=z, scale=f, bbox=_bbox_from_center(cx, cy, k, h, w))
    return dict(score=float(best["score"]), candidate_scale=best["scale"],
                candidate_bbox=best["bbox"], per_scale=per_scale,
                readout="real_null_multiscale")


# ------------------------------------------------------------------ evaluation

def bbox_hit(bbox, mask):
    """Does the blind candidate box intersect the GT mask? (evaluation only)"""
    if bbox is None:
        return dict(hit=0, iou=0.0, precision=0.0, recall=0.0, center_in_mask=0)
    x1, y1, x2, y2 = bbox
    sub = mask[y1:y2, x1:x2]
    inter = float(sub.sum())
    barea = max(1, (x2 - x1) * (y2 - y1))
    marea = float(mask.sum())
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cin = int(bool(mask[min(cy, mask.shape[0] - 1), min(cx, mask.shape[1] - 1)]))
    return dict(hit=int(inter > 0), iou=inter / max(1.0, barea + marea - inter),
                precision=inter / barea, recall=inter / max(marea, 1.0),
                center_in_mask=cin)


def expected_random_hit(bbox, mask):
    """Chance that a box of this size, placed uniformly at random, would hit the
    mask. Used so a hit on a large mask is not mistaken for understanding."""
    if bbox is None:
        return float("nan")
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    H, W = mask.shape
    if bw <= 0 or bh <= 0 or H <= bh or W <= bw:
        return float(mask.mean())
    # dilate mask by the box extent: any top-left corner inside the dilated
    # region yields an intersecting box
    kern = np.ones((min(bh * 2 - 1, H), min(bw * 2 - 1, W)), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), kern)
    valid = dil[: H - bh + 1, : W - bw + 1]
    return float(valid.mean())
