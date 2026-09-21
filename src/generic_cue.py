"""Generic salient cue (Control D) -- frozen implementation, no GT, no ELA.

Purpose: separate "the model trusts a FORENSIC visualization" from "the model
follows any bright blob in a second image".

Construction (fixed before any run, identical for every image):
  1. grayscale -> local-contrast saliency: |I - box_blur(I, k)| with k tied to the
     short side (6%), i.e. the same spatial scale the ELA patch statistics use.
  2. smooth with the same kernel so the result has ELA-like spatial granularity.
  3. min-max normalize, then apply cv2.COLORMAP_INFERNO -- the SAME colormap the
     ELA visualization uses, so the two conditions are visually comparable in
     palette and dynamic range.
  4. same pixel dimensions as the image under examination.

Why this is a fair control:
  * It carries no re-compression / forensic information whatsoever: it is a pure
    appearance statistic of the visible image.
  * It is derived from the CURRENT image, so unlike the donor swap it is spatially
    plausible, and it has no seams or resize artifacts.
  * It is highly salient (it peaks on edges and texture), so if the model merely
    chases bright regions it should adhere to this cue too.

CONFOUND DISCLOSED (must be stated in any write-up): because it is computed from
the current image, a generic saliency map correlates with image structure, and an
inpainted object is often a structural outlier. So generic-cue adherence is NOT a
pure "no information" baseline -- it can accidentally point at the edit. This makes
it a CONSERVATIVE control for the grounding question: if adherence to the generic
cue is as high as to the ELA cue, that argues for salience-following; but a lower
generic adherence does not by itself prove forensic-specific grounding. The donor
swap remains the primary mismatched-cue control precisely because it carries zero
information about the current image.
"""
import numpy as np, cv2

SALIENCY_FRAC = 0.06          # FROZEN: window size as fraction of short side
COLORMAP = cv2.COLORMAP_INFERNO   # FROZEN: same as ELA visualization


def generic_salient_map(rgb):
    """rgb: HxWx3 uint8 -> (map01 float32, vis_bgr uint8)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    k = max(8, int(round(min(h, w) * SALIENCY_FRAC)) | 1)
    local = np.abs(g - cv2.blur(g, (k, k)))
    local = cv2.blur(local, (k, k))
    lo, hi = float(local.min()), float(local.max())
    m = np.zeros_like(local) if hi - lo < 1e-9 else (local - lo) / (hi - lo)
    vis = cv2.applyColorMap((m * 255).astype(np.uint8), COLORMAP)
    return m, vis


def map_stats(m):
    """Comparability check against ELA maps: report the statistics we claim are
    similar, so the claim is verified rather than asserted."""
    f = m.ravel()
    hist, _ = np.histogram(f, bins=64, range=(0, 1))
    return dict(mean=float(f.mean()), std=float(f.std()),
                p99=float(np.percentile(f, 99)),
                dynamic_range=float(f.max() - f.min()),
                near_constant_ratio=float(hist.max() / f.size))


def candidate_region_p99(m):
    """Same frozen B_region_p99 read-out used for ELA, applied to any map, so the
    generic cue's candidate region is defined identically."""
    thr = float(np.percentile(m, 99.0))
    reg = (m > thr).astype(np.uint8)
    if reg.sum() < 16 or reg.sum() > 0.9 * reg.size:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(reg, connectivity=8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, ww, hh = (int(stats[big, cv2.CC_STAT_LEFT]), int(stats[big, cv2.CC_STAT_TOP]),
                    int(stats[big, cv2.CC_STAT_WIDTH]), int(stats[big, cv2.CC_STAT_HEIGHT]))
    return [x, y, x + ww, y + hh]
