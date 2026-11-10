#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic panel for --equi-region window: the whole panoramic prediction
with the scored window outlined on it, above the window itself beside the
Kinect ground truth and the error.

A STANDALONE MODULE. Drop it in Scripts/ beside equi_pinhole.py:

    from equi_panel import make_equi_window_panel

Nothing in equi_pinhole.py or theta_kinect_compare.py changes.

WHY THE EXISTING PANEL CANNOT SHOW THIS
    make_panel() is handed res["pred_m"], which in window mode is ALREADY
    the 518x518 projection -- compare_equi_pinhole consumes the 3840x1920
    radial map internally. The panorama is still in scope in run_eval as
    `pred`, so this panel takes it as an argument and does its own drawing.

HOW THE WINDOW OUTLINE IS FOUND
    Not by re-deriving the geometry -- by pushing a coordinate-index map
    through project_equi_to_pinhole itself, so the outline is sampled by the
    exact equi2pers call, rots and interpolation the scoring uses. If the
    conventions ever diverge, this panel diverges with them instead of
    drawing a plausible-looking lie over a projection that moved.

    project_equi_to_pinhole divides its output by s = r/z, which would
    corrupt an index map -- index 2,500,000 at the corner would come back
    as 1,937,984 and point at the wrong row. But s is fully determined by
    out_wh and fov_x, so it is multiplied straight back out (verified exact
    to 0 index errors over all 268,324 window pixels). Row and column go
    through as SEPARATE maps: a flat h*w index would be rounded by float32
    at those magnitudes, two indices under 4096 are exact.

COLOUR
    All depth tiles share ONE scale, taken from the ground truth's valid
    pixels. That is the point of the panel: a model reading 1.41x too far
    must LOOK too far next to the GT, not merely different. Separate
    per-tile autoscaling would hide exactly the error being investigated.
    The error tile has its own scale, stated in its label, because it is a
    different quantity.
"""

import numpy as np

try:
    import cv2
except ImportError as e:            # pragma: no cover
    raise ImportError("equi_panel needs OpenCV (cv2)") from e

# Viridis: monotone in lightness, so a brighter pixel is reliably a farther
# pixel, in print and for colour-blind readers. A rainbow map would invent
# banding at its hue turns and claim structure the depth map does not have.
# --------------------------------------------------------------------------
# Use the PROJECT'S drawing helpers when they are importable, so this panel
# is identical in font, label format and colour map to the make_panel() in
# theta_kinect_compare_equi.py. Copying the font size into this file would
# look the same today and drift the first time that one is edited; binding
# to the functions themselves cannot drift.
#
# The fallbacks below are only for running this module standalone (e.g. in
# a notebook); in the benchmark they are never reached. Which source is in
# use is printed once at import, so a silent fallback is impossible.
# --------------------------------------------------------------------------
_HELP = None
for _name in ("theta_kinect_compare_equi", "theta_kinect_compare"):
    try:
        _m = __import__(_name)
    except Exception:
        continue
    if hasattr(_m, "_label") and hasattr(_m, "_colorize_full"):
        _HELP = _m
        break
_HELP_SRC = _HELP.__name__ if _HELP is not None else "equi_panel fallbacks"


def _lbl(img, text):
    """The project's _label if available, else a local header bar."""
    if _HELP is not None:
        try:
            return _HELP._label(img, text)
        except Exception:
            pass
    return np.vstack([_header(img.shape[1], text), img])


def _cmap_full(a, lo, hi):
    if _HELP is not None:
        try:
            return _HELP._colorize_full(a, lo, hi)
        except Exception:
            pass
    return colorize(a, lo, hi)


def _cmap(a, valid, lo, hi):
    if _HELP is not None and hasattr(_HELP, "_colorize"):
        try:
            return _HELP._colorize(a, valid, lo, hi)
        except Exception:
            pass
    return colorize(a, lo, hi, valid)


def _fit_w(img, W, H):
    if _HELP is not None and hasattr(_HELP, "_fit_width"):
        try:
            return _HELP._fit_width(img, W, H)
        except Exception:
            pass
    h = max(int(round(img.shape[0] * W / float(img.shape[1]))), 1)
    return cv2.resize(img, (W, h), interpolation=cv2.INTER_AREA)


CMAP = cv2.COLORMAP_VIRIDIS
INVALID_BGR = (110, 110, 110)       # neutral grey, outside the ramp
OUTLINE_BGR = (60, 220, 255)        # amber; reads against all of viridis
FG = (245, 245, 245)
BG = (24, 24, 24)
HEADER_H = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pinhole_sec(out_wh, fov_x):
    """s(u, v) = r/z for every pixel of the pinhole window.

    f = (W/2)/tan(fov_x/2), principal point at ((W-1)/2, (H-1)/2). That
    choice is not free: it is what makes the corner 1.2901 rather than
    1.2910, and 1.290 is what equi_pinhole prints at run time.
    """
    W, H = out_wh
    f = (W / 2.0) / np.tan(np.radians(fov_x) / 2.0)
    u = (np.arange(W, dtype=np.float64) - (W - 1) / 2.0) / f
    v = (np.arange(H, dtype=np.float64) - (H - 1) / 2.0) / f
    return np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2), f


def window_footprint(project_fn, equi_hw, out_wh, fov_x, verbose=False):
    """Which panorama pixel did each window pixel read?

    Returns (rows, cols) int arrays shaped like the window, or (None, None)
    if the recovered values are not integers -- which would mean the
    sampling was not nearest, or a guard inside the projection altered the
    data, and in either case the outline would be wrong.
    """
    eh, ew = equi_hw
    s, _ = pinhole_sec(out_wh, fov_x)

    row_src = (np.arange(eh, dtype=np.float32) + 1.0)[:, None] * np.ones(
        (1, ew), np.float32)
    col_src = np.ones((eh, 1), np.float32) * (
        np.arange(ew, dtype=np.float32) + 1.0)[None, :]

    out = []
    for src in (row_src, col_src):
        z = project_fn(src)
        # project_equi_to_pinhole returns a TUPLE, (z, ...), not a bare
        # array -- take the first element. Also accept a torch tensor,
        # which has .shape but cannot be indexed by numpy downstream.
        if isinstance(z, (tuple, list)):
            z = z[0] if len(z) else None
        if z is None:
            return None, None
        if hasattr(z, "detach"):
            z = z.detach().cpu().numpy()
        z = np.asarray(z)
        if z.ndim == 3 and z.shape[0] == 1:
            z = z[0]
        if z.shape != (out_wh[1], out_wh[0]):
            return None, None
        back = np.asarray(z, np.float64) * s
        resid = float(np.abs(np.rint(back) - back).max())
        if verbose:
            print(f"[panel] footprint residual {resid:.2e} "
                  f"(must be ~0; non-integer means not nearest-sampled)")
        if resid > 0.05:
            return None, None
        out.append(np.rint(back).astype(np.int64) - 1)
    return out[0], out[1]


def colorize(a, vmin, vmax, valid=None):
    """Depth (metres) -> BGR, invalid pixels neutral grey."""
    a = np.asarray(a, np.float64)
    fin = np.isfinite(a)
    if valid is not None:
        fin &= np.asarray(valid, bool)
    span = max(float(vmax) - float(vmin), 1e-9)
    n = np.clip((np.nan_to_num(a, nan=vmin) - vmin) / span, 0.0, 1.0)
    img = cv2.applyColorMap((n * 255.0).astype(np.uint8), CMAP)
    img[~fin] = INVALID_BGR
    return img


def _header(width, text, h=HEADER_H):
    bar = np.full((h, width, 3), BG, np.uint8)
    cv2.putText(bar, text, (6, h - 7), FONT, 0.44, FG, 1, cv2.LINE_AA)
    return bar


def _stack(img, text):
    return np.vstack([_header(img.shape[1], text), img])


def _colorbar(width, vmin, vmax, h=36):
    # Through _cmap_full, so the bar matches whatever map the tiles used.
    # A hardcoded ramp here would read viridis while the tiles read the
    # project's map, and a mislabelled colour key is worse than none.
    ramp = np.repeat(np.linspace(vmin, vmax, width)[None, :], h - 16, axis=0)
    bar = _cmap_full(ramp, vmin, vmax)
    pad = np.full((16, width, 3), BG, np.uint8)
    out = np.vstack([bar, pad])
    cv2.putText(out, f"{vmin:.2f} m", (4, h - 4), FONT, 0.40, FG, 1,
                cv2.LINE_AA)
    t = f"{vmax:.2f} m"
    (tw, _), _ = cv2.getTextSize(t, FONT, 0.40, 1)
    cv2.putText(out, t, (width - tw - 4, h - 4), FONT, 0.40, FG, 1,
                cv2.LINE_AA)
    return out


def make_equi_window_panel(pred_equi_radial, res, project_fn,
                           out_wh=(518, 518), fov_x=60.0, label="",
                           equi_width=None, scale=1.0, verbose=False):
    """Build the panel. project_fn(equi) -> (H, W) window, i.e. a closure
    over project_equi_to_pinhole with this frame's yaw/pitch/device."""
    pred_w = np.asarray(res["pred_m"], np.float64)
    gt_w = np.asarray(res["gt_m"], np.float64)
    valid = np.asarray(res["valid"], bool)
    pano = np.asarray(pred_equi_radial, np.float64)

    tile_w, tile_h = out_wh
    total_w = tile_w * 3
    if equi_width is None:
        equi_width = total_w

    # One scale for every depth tile, from the GT's valid pixels.
    ok = valid & np.isfinite(gt_w) & (gt_w > 0)
    both = np.concatenate([gt_w[ok], pred_w[ok & np.isfinite(pred_w)]])
    if both.size:
        vmin, vmax = (float(np.percentile(both, 2)),
                      float(np.percentile(both, 98)))
    else:
        vmin, vmax = 0.0, 5.0
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    rows, cols = window_footprint(project_fn, pano.shape[:2], out_wh, fov_x,
                                  verbose=verbose)

    pf = pano[np.isfinite(pano) & (pano > 0)]
    if pf.size:
        pmin, pmax = float(np.percentile(pf, 2)), float(np.percentile(pf, 98))
    else:
        pmin, pmax = vmin, vmax
    if pmax - pmin < 1e-6:
        pmax = pmin + 1.0
    pano_img = _cmap_full(pano, pmin, pmax)
    eh, ew = pano.shape[:2]
    # NOT `scale` -- that is the function's output-size parameter, and
    # shadowing it here silently shrank every panel by equi_width/ew.
    pano_scale = equi_width / float(ew)
    pano_img = cv2.resize(pano_img,
                          (equi_width, max(int(round(eh * pano_scale)), 1)),
                          interpolation=cv2.INTER_AREA)

    if rows is not None:
        # Perimeter of the window only -- that is the footprint's boundary.
        pr = np.concatenate([rows[0, :], rows[-1, :], rows[:, 0], rows[:, -1]])
        pc = np.concatenate([cols[0, :], cols[-1, :], cols[:, 0], cols[:, -1]])
        py = np.clip((pr * pano_scale).astype(np.int32), 0,
                     pano_img.shape[0] - 1)
        px = np.clip((pc * pano_scale).astype(np.int32), 0,
                     pano_img.shape[1] - 1)
        # A 1 px amber line vanishes against viridis yellow. Dilate the
        # scatter twice: a dark halo first, then a bright core inside it, so
        # the outline reads on ANY background. Scatter rather than polylines
        # because a window straddling the +/-180 seam is two arcs, and a
        # closed polyline would draw a chord straight across the panorama.
        m = np.zeros(pano_img.shape[:2], np.uint8)
        m[py, px] = 255
        halo = cv2.dilate(m, np.ones((7, 7), np.uint8))
        core = cv2.dilate(m, np.ones((3, 3), np.uint8))
        pano_img[halo > 0] = (20, 20, 20)
        pano_img[core > 0] = OUTLINE_BGR
        note = (f"1. radial ||points|| {ew}x{eh}, raw  "
                f"{pmin:.2f}-{pmax:.2f} m  "
                f"(scored window outlined, {fov_x:g} deg; own scale)")
    else:
        note = (f"1. radial ||points|| {ew}x{eh}, raw  "
                f"{pmin:.2f}-{pmax:.2f} m  (OUTLINE UNAVAILABLE: the "
                f"projection did not return nearest-sampled indices)")

    top = _lbl(pano_img, f"{label}  {note}" if label else note)

    err = np.abs(pred_w - gt_w)
    e = err[valid & np.isfinite(err)]
    emax = float(np.percentile(e, 98)) if e.size else 1.0
    emax = max(emax, 1e-6)

    med_p = float(np.median(pred_w[valid])) if valid.any() else float("nan")
    med_g = float(np.median(gt_w[valid])) if valid.any() else float("nan")

    pv = pred_w[valid & np.isfinite(pred_w)]
    gv = gt_w[valid & np.isfinite(gt_w)]
    p_lo, p_hi = (float(pv.min()), float(pv.max())) if pv.size else (0.0, 0.0)
    g_lo, g_hi = (float(gv.min()), float(gv.max())) if gv.size else (0.0, 0.0)

    # Label format mirrors make_panel(): numbered stage, size, min-max in
    # metres, then the one fact that stage exists to show, in brackets.
    bottom = np.hstack([
        _lbl(_cmap(pred_w, valid, vmin, vmax),
             f"2. window prediction z {pred_w.shape[1]}x{pred_w.shape[0]}  "
             f"{p_lo:.2f}-{p_hi:.2f} m  (median {med_p:.3f} m)"),
        _lbl(_cmap(gt_w, valid, vmin, vmax),
             f"3. Azure warped into the window  {g_lo:.2f}-{g_hi:.2f} m  "
             f"(coverage {valid.mean()*100:.1f}%)"),
        _lbl(_cmap(err, valid, 0.0, emax),
             f"4. |pred - gt|  0.00-{emax:.2f} m  (own scale)"),
    ])

    if top.shape[1] != bottom.shape[1]:
        top = cv2.resize(top, (bottom.shape[1], top.shape[0]),
                         interpolation=cv2.INTER_AREA)

    ratio = (med_p / med_g) if (med_g and np.isfinite(med_g)) else float("nan")
    cb = _lbl(_colorbar(bottom.shape[1], vmin, vmax),
              f"rows 2-4 share {vmin:.2f}-{vmax:.2f} m (pred+GT p2-p98); "
              f"row 1 scaled separately   |   median pred/GT = {ratio:.3f}"
              f"   |   labels from {_HELP_SRC}")
    foot = np.zeros((0, bottom.shape[1], 3), np.uint8)
    out = np.vstack([top, bottom, cb, foot])
    if scale and scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return out