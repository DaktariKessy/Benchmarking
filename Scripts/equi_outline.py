#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where does the 60 deg pinhole window fall on an equirectangular image?

"""

import numpy as np

try:
    import cv2
except ImportError as e:            # pragma: no cover
    raise ImportError("equi_outline needs OpenCV (cv2)") from e


def pinhole_sec(out_wh, fov_x):
    """s(u, v) = r/z for every pixel of the pinhole window.

    f = (W/2)/tan(fov_x/2); principal point at ((W-1)/2, (H-1)/2). That
    choice is not arbitrary -- it is what puts the corner at 1.2901, and
    1.290 is what equi_pinhole prints at run time.
    """
    W, H = out_wh
    f = (W / 2.0) / np.tan(np.radians(fov_x) / 2.0)
    u = (np.arange(W, dtype=np.float64) - (W - 1) / 2.0) / f
    v = (np.arange(H, dtype=np.float64) - (H - 1) / 2.0) / f
    return np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2), f


def _unwrap(z, out_wh):
    """project_equi_to_pinhole returns a tuple; accept a tensor too."""
    if isinstance(z, (tuple, list)):
        z = z[0] if len(z) else None
    if z is None:
        return None
    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()
    z = np.asarray(z)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    return z if z.shape == (out_wh[1], out_wh[0]) else None


def equi_window_outline(project_fn, hw, out_wh=(518, 518), fov_x=60.0,
                        thickness=3, verbose=False):
    """Boolean mask of shape `hw` marking the window's border.

    project_fn : takes an equirect array of shape `hw`, returns the
                 projected window (a closure over project_equi_to_pinhole
                 with this run's yaw, pitch, device).
    hw         : (H, W) of the image you will draw on -- pass gt_m.shape so
                 the indices come back in that image's own coordinates and
                 need no rescaling.

    Returns None if the recovered coordinates are not whole numbers.

    The result depends only on yaw, pitch, fov and `hw` -- never on the
    frame -- so compute it ONCE before the loop, not per frame.
    """
    H, W = int(hw[0]), int(hw[1])
    s, _ = pinhole_sec(out_wh, fov_x)

    row_src = np.repeat((np.arange(H, dtype=np.float32) + 1.0)[:, None],
                        W, axis=1)
    col_src = np.repeat((np.arange(W, dtype=np.float32) + 1.0)[None, :],
                        H, axis=0)

    rc = []
    for src in (row_src, col_src):
        z = _unwrap(project_fn(src), out_wh)
        if z is None:
            if verbose:
                print("[outline] projection returned nothing usable")
            return None
        back = np.asarray(z, np.float64) * s
        resid = float(np.abs(np.rint(back) - back).max())
        if verbose:
            print(f"[outline] residual {resid:.2e} (must be ~0)")
        if resid > 0.05:
            if verbose:
                print("[outline] not nearest-sampled -- refusing to draw")
            return None
        rc.append(np.clip(np.rint(back).astype(np.int64) - 1, 0,
                          (H - 1) if src is row_src else (W - 1)))

    rows, cols = rc
    # The window's perimeter, not its interior. Scattered rather than drawn
    # as a polyline: a window straddling the +/-180 seam is two arcs, and a
    # closed polyline would draw a chord straight across the panorama.
    pr = np.concatenate([rows[0, :], rows[-1, :], rows[:, 0], rows[:, -1]])
    pc = np.concatenate([cols[0, :], cols[-1, :], cols[:, 0], cols[:, -1]])

    m = np.zeros((H, W), np.uint8)
    m[pr, pc] = 255
    k = max(int(thickness), 1)
    if k > 1:
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m > 0