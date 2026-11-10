#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score a PANORAMIC prediction inside the PINHOLE window (the R4.3 control).

A STANDALONE MODULE, not a block to paste. Drop it in Scripts/ beside
theta_kinect_compare_equi.py and import from it:

    from equi_pinhole import compare_equi_pinhole

Nothing in theta_kinect_compare.py or theta_kinect_compare_equi.py changes.

WHY IT EXISTS
    compare_equi scores a panoramic prediction over the Kinect's whole
    ~120x120 deg coverage; compare scores a pinhole prediction over a 60 deg
    window. Comparing those two changes TWO things at once -- the input
    regime AND the evaluation region -- so a difference in AbsRel is not
    attributable to either. This path holds the region fixed: the panoramic
    prediction is projected into the SAME window and handed to the SAME
    compare() the pinhole runs use, so the only thing that differs is what
    the model was shown.

    It calls compare() rather than masking compare_equi to a cone, so the
    ground truth goes through the identical warp, range gate and z-buffer as
    the pinhole runs. A control whose ground truth is built differently is
    not a control.
"""

import os
import sys

# Project paths, so this works from any directory and in any shell. Same
# block as theta_kinect_compare.py; harmless if that already ran.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_ROOT).lower() in ("scripts", "tools"):
    _ROOT = os.path.dirname(_ROOT)
for _sub in ("Processors", "Tools", "Scripts"):
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, '/opt/ros/humble/lib/python3.10/dist-packages')

import numpy as np
import torch

from equilib.equi2pers.torch_impl import run as equi2pers_run

from theta_kinect_compare import (compare, DEPTH_DISTORTION,
                                  OUT_W, OUT_H, FOV_X)


def project_equi_to_pinhole(radial_equi, yaw_deg=0.0, pitch_deg=0.0,
                            fov_x=None, out_wh=None, device="cpu",
                            verbose=False):
    """Sample an equirect RADIAL map into a pinhole window and return Z.

    Args:
        radial_equi : (h, w) float, distance from the optical centre. NOT a
                      z-depth -- on a sphere z is undefined past 90 deg.
        yaw/pitch   : the crop's aim, in the SAME convention and sign as
                      render_pseudo, so the window is the identical patch of
                      sphere the pinhole run used.
        fov_x, out_wh : default to FOV_X and (OUT_W, OUT_H) -- the pinhole
                      run's own virtual camera. Do not override them for the
                      control; the point is that they match.

    Returns (z_map, r_map): z along the crop's optical axis, and the radial
    map it came from, both (H, W) float32.

    TWO THINGS THAT WOULD SILENTLY CORRUPT THIS
      1. Converting radial->z on the SPHERE instead of here. Done here, the
         correction is exact for any yaw/pitch, because equi2pers has
         already resolved each output pixel to a ray and r is invariant to
         how the sphere is oriented. Done before projection it has to assume
         an axis, and gets the off-axis pixels wrong.
      2. Bilinear sampling. Across a depth discontinuity it invents values
         between foreground and background that belong to neither, and the
         crop boundary of a far wall seen past a near board is exactly such
         a discontinuity. NEAREST is requested, with a loud fallback.
    """
    fov_x = FOV_X if fov_x is None else float(fov_x)
    W, H = (OUT_W, OUT_H) if out_wh is None else out_wh

    t = torch.from_numpy(
        np.ascontiguousarray(radial_equi.astype(np.float32)))
    # A depth map is one channel; replicate so implementations that assume a
    # 3-channel image accept it, and take channel 0 back afterwards.
    t3 = t.unsqueeze(0).repeat(3, 1, 1).to(device)

    rots = [{"yaw": float(np.deg2rad(-yaw_deg)),
             "roll": 0.0,
             "pitch": float(np.deg2rad(-pitch_deg))}]

    def _run(equi, mode):
        with torch.inference_mode():
            return equi2pers_run(equi=equi, rots=rots, height=H, width=W,
                                 fov_x=fov_x, skew=0.0, z_down=False,
                                 mode=mode)

    mode = "nearest"
    for equi_in in (t3.unsqueeze(0), t3):      # (1,3,h,w) then (3,h,w)
        try:
            out = _run(equi_in, mode)
            break
        except (TypeError, ValueError, RuntimeError) as e:
            last = e
            try:
                out = _run(equi_in, "bilinear")
                print("[equi->pinhole] WARNING: 'nearest' rejected, used "
                      "bilinear. Values across depth edges are interpolated "
                      "and belong to neither surface.")
                break
            except (TypeError, ValueError, RuntimeError) as e2:
                last = e2
    else:
        raise RuntimeError(f"equi2pers rejected the depth map: {last}")

    arr = out.squeeze(0) if out.dim() == 4 else out
    r = arr[0].detach().cpu().numpy().astype(np.float32)

    # Guard: a value-preserving resample cannot leave the source's range.
    # Some image paths normalise to [0,1]; that would destroy metric depth
    # silently, producing a believable-looking map in the wrong units.
    src = radial_equi[np.isfinite(radial_equi) & (radial_equi > 0)]
    dst = r[np.isfinite(r) & (r > 0)]
    if src.size and dst.size and dst.max() > 1.001 * src.max():
        raise RuntimeError(
            f"projected depth exceeds the source range "
            f"({dst.max():.3f} > {src.max():.3f} m): equi2pers is not "
            f"value-preserving on this input.")
    if src.size and dst.size and dst.max() <= 1.0 < src.max():
        raise RuntimeError(
            "projected depth collapsed into [0, 1] while the source spans "
            f"up to {src.max():.3f} m: the projection normalised the map. "
            "Pass the raw float tensor, not preprocess()'s output.")

    # radial -> z, in the pinhole frame where it is exact.
    fx = (W / 2.0) / np.tan(np.radians(fov_x) / 2.0)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    xp = (u + 0.5 - W / 2.0) / fx
    yp = (v + 0.5 - H / 2.0) / fx
    sec = np.sqrt(1.0 + xp ** 2 + yp ** 2)      # r / z per pixel
    z = (r / sec).astype(np.float32)
    z[~np.isfinite(z)] = 0.0
    z[z < 0] = 0.0

    if verbose:
        print(f"[equi->pinhole] {radial_equi.shape[1]}x{radial_equi.shape[0]}"
              f" -> {W}x{H} at fov_x={fov_x:g}, yaw {yaw_deg:+g} "
              f"pitch {pitch_deg:+g}")
        print(f"[equi->pinhole] r/z ranges 1.000 (centre) to {sec.max():.3f} "
              f"(corner); skipping it would bias the corners by "
              f"{100*(sec.max()-1):.0f}%")
    return z, r


def compare_equi_pinhole(pred_equi_radial, depth_mm, kinect_intr,
                         yaw_deg=0.0, pitch_deg=0.0,
                         fov_x=None, out_wh=None, device="cpu",
                         metric_model=True, distortion=DEPTH_DISTORTION,
                         dyaw=0.0, dpitch=0.0, verbose=False):
 
    z, r = project_equi_to_pinhole(pred_equi_radial, yaw_deg, pitch_deg,
                                   fov_x, out_wh, device, verbose=verbose)
    W, H = (OUT_W, OUT_H) if out_wh is None else out_wh
    res = compare(z, depth_mm, kinect_intr, R_v=np.eye(3),
                  out_wh=(W, H), fov_x=FOV_X if fov_x is None else fov_x,
                  metric_model=metric_model, distortion=distortion,
                  dyaw=dyaw, dpitch=dpitch)
    res["pred_radial_window"] = r
    return res