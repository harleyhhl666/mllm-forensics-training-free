"""Forensic tools + Gate 0 (validity) + Gate 1 (blind objective anomaly score).

Hard rules encoded here:
  * No function in this file ever receives or reads a GT mask. Anomaly maps and
    anomaly scores are computed blind. GT is used only downstream, for
    evaluation (Gate 3 localization correctness, tamper_ratio analysis).
  * Validity (Gate 0) is decided from the tool's own output statistics and the
    image's format applicability, with thresholds fixed a priori in the config.
  * A tool that is invalid contributes no evidence, and the MLLM disbelieving an
    invalid tool is NOT counted as evidence rejection.
"""
import io
import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------- utilities

def _norm01(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x), 0.0
    return (x - lo) / (hi - lo), hi - lo


def _validity_stats(map01, raw_range):
    """Statistics used by Gate 0. map01 is the normalized anomaly map in [0,1]."""
    flat = map01.ravel()
    hist, _ = np.histogram(flat, bins=64, range=(0.0, 1.0))
    near_constant_ratio = float(hist.max() / max(flat.size, 1))
    return dict(
        variance=float(map01.var()),
        dynamic_range=float(raw_range),
        near_constant_ratio=near_constant_ratio,
        saturation_ratio=float((flat > 0.99).mean() + (flat < 0.01).mean() * 0.0),
        low_saturation_ratio=float((flat < 0.01).mean()),
    )


def _gate0(stats, cfg_v, applicable=True, reasons=None):
    reasons = list(reasons or [])
    if not applicable:
        reasons.append("not_applicable_to_format")
    if stats["variance"] < cfg_v["min_variance"]:
        reasons.append("variance_below_min")
    if stats["dynamic_range"] < cfg_v["min_dynamic_range"]:
        reasons.append("dynamic_range_below_min")
    if stats["near_constant_ratio"] > cfg_v["max_near_constant_ratio"]:
        reasons.append("near_constant_output")
    if stats["saturation_ratio"] > cfg_v["max_saturation_ratio"]:
        reasons.append("saturated_output")
    return (len(reasons) == 0), reasons


def blind_anomaly_score(map01, grid=3):
    """Blind Gate-1 score. NO GT mask involved.

    Split the normalized anomaly map into grid x grid cells, then measure how
    far the most anomalous cell deviates from the image's own cell population.
    A locally-confined anomaly gives a high score; a globally uniform texture
    response gives ~0. Returns (score, per_cell_means, argmax_cell_index).
    """
    h, w = map01.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    cells = np.zeros((grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            cells[i, j] = map01[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
    c = cells.ravel()
    score = float((c.max() - c.mean()) / (c.std() + 1e-6))
    return score, cells, int(c.argmax())


CELL_NAMES = ["top-left", "top-center", "top-right",
              "center-left", "center", "center-right",
              "bottom-left", "bottom-center", "bottom-right"]


def cell_index_to_name(idx):
    return CELL_NAMES[idx]


# ---------------------------------------------------------------- the tools

def tool_ela(bgr, pil_img, cfg, cfg_v):
    """Error Level Analysis: recompress at fixed quality, amplify the residual."""
    q = int(cfg["jpeg_quality"])
    scale = float(cfg["scale"])
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("ELA: jpeg encode failed")
    re = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, re).astype(np.float32) * scale
    gray = diff.mean(axis=2)
    map01, rng = _norm01(gray)
    stats = _validity_stats(map01, rng / 255.0)
    valid, reasons = _gate0(stats, cfg_v)
    vis = cv2.applyColorMap((map01 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return dict(name="ela", map01=map01, vis_bgr=vis, stats=stats,
                valid=valid, invalid_reasons=reasons, applicable=True)


def tool_noise_residual(bgr, pil_img, cfg, cfg_v):
    """High-pass / median-filter noise residual, then local residual energy."""
    k = int(cfg["kernel"])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = cv2.medianBlur(gray.astype(np.uint8), k).astype(np.float32)
    resid = gray - med
    # local energy of the residual (std in a 16x16 neighbourhood)
    mu = cv2.blur(resid, (16, 16))
    mu2 = cv2.blur(resid * resid, (16, 16))
    energy = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    map01, rng = _norm01(energy)
    stats = _validity_stats(map01, rng / 255.0)
    valid, reasons = _gate0(stats, cfg_v)
    vis = cv2.applyColorMap((map01 * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return dict(name="noise_residual", map01=map01, vis_bgr=vis, stats=stats,
                valid=valid, invalid_reasons=reasons, applicable=True)


def tool_dct_jpeg(bgr, pil_img, cfg, cfg_v):
    """Double-JPEG / blockiness analysis.

    Gate-0 APPLICABILITY: this tool reasons about JPEG quantization history. On a
    losslessly-stored PNG that never carried a JPEG history the analysis has no
    defined meaning, so the tool is marked NOT APPLICABLE (tool_valid=False) for
    PNG input when config sets dct_jpeg_requires_jpeg. This is a Phase-0 finding
    for CocoGlide (all PNG), not a workaround.
    """
    fmt = (getattr(pil_img, "format", None) or "").upper()
    is_jpeg = fmt in ("JPEG", "JPG", "MPO")
    applicable = True
    reasons = []
    if cfg_v.get("dct_jpeg_requires_jpeg", True) and not is_jpeg:
        applicable = False
        reasons.append(f"input_format_{fmt or 'unknown'}_has_no_jpeg_history")

    block = int(cfg["block"])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    H, W = (h // block) * block, (w // block) * block
    g = gray[:H, :W]
    # per-block DCT high-frequency energy -> blockwise map
    bh, bw = H // block, W // block
    hf = np.zeros((bh, bw), np.float32)
    for i in range(bh):
        row = g[i * block:(i + 1) * block]
        for j in range(bw):
            d = cv2.dct(row[:, j * block:(j + 1) * block])
            hf[i, j] = np.abs(d[block // 2:, block // 2:]).mean()
    up = cv2.resize(hf, (w, h), interpolation=cv2.INTER_NEAREST)
    map01, rng = _norm01(up)
    stats = _validity_stats(map01, rng / 255.0)
    valid, reasons = _gate0(stats, cfg_v, applicable=applicable, reasons=reasons)
    vis = cv2.applyColorMap((map01 * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return dict(name="dct_jpeg", map01=map01, vis_bgr=vis, stats=stats,
                valid=valid, invalid_reasons=reasons, applicable=applicable,
                source_format=fmt)


TOOLS = {"ela": tool_ela, "noise_residual": tool_noise_residual, "dct_jpeg": tool_dct_jpeg}


def run_tools(image_path, tool_cfg, grid=3):
    """Run every configured tool on one image. Returns a dict keyed by tool name."""
    pil = Image.open(image_path)
    fmt = pil.format
    rgb = np.array(pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cfg_v = tool_cfg["validity"]
    out = {}
    for name, fn in TOOLS.items():
        if name not in tool_cfg:
            continue
        r = fn(bgr, pil, tool_cfg[name], cfg_v)
        score, cells, amax = blind_anomaly_score(r["map01"], grid=grid)
        r["anomaly_score"] = score
        r["cell_means"] = cells.tolist()
        r["peak_cell"] = cell_index_to_name(amax)
        r["source_format"] = fmt
        out[name] = r
    return out, rgb
