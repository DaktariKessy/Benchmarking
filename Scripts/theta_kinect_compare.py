#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained Kinect<->360 calibration library + overlay tool.

This file now contains BOTH:
  * the comparison library that used to live in theta_kinect_compare.py
    (constants, undistort, the depth->colour->equirect chain, compare()), and
  * the overlay/visual-calibration tool (renders the equi2pers crop, warps
    the Kinect depth into it, draws the four-panel sheet).

It also owns the shared bag-reading and pseudo-rendering helpers
(open_bag, stream_pairs, render_pseudo) so that nothing here imports from
main_depthpro_eval.py. The dependency runs one way only:

    main_depthpro_eval.py  ->  calibrate_view.py

which avoids the circular import a naive split would create.

OVERLAY USAGE
-------------
    python3 calibrate_view.py --bag ~/Trials/0.5m --pitch 0 --fov 60
    python3 calibrate_view.py --bag ~/Trials/1.5m --pitch -10 --fov 60 --dpitch 10 --dyaw -0.4

CHAIN SELF-TEST (prints rotation/translation + distortion angles)
    python3 calibrate_view.py --selftest

THE CHAIN
---------
    p_depth  (from /depth/image_raw, undistorted with /depth/camera_info)
      -> depth camera to colour camera      5.53 deg,  32 mm
      -> colour camera to equirect          1.97 deg, 159 mm   (equiMpc)
      -> optional --dyaw / --dpitch nudge
      -> project with the virtual camera's fx

--dyaw and --dpitch exist because equilib's rots convention and the equirect
frame the calibration assumes do not necessarily share a zero. Whatever value
lines the overlay up is the offset between them.
"""

import argparse
import math
import sys
import os
from collections import deque


sys.path.insert(0, '/opt/ros/humble/lib/python3.10/dist-packages')

import cv2
import numpy as np
import torch

import rclpy.serialization as ser
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_ROOT).lower() in ("scripts", "tools"):
    _ROOT = os.path.dirname(_ROOT)
for _sub in ("Processors", "Tools", "Scripts"):
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from equilib import equi2pers as equi2pers_run          # adjust if different
from pipeline_nlf.utils import preprocess, compute_camera_parameters


# ==========================================================================
# Depth camera intrinsics + lens model, from the calibration log.
# Prefer the values in /depth/camera_info if you have them: `k` gives the
# intrinsics and `d` gives these eight coefficients in OpenCV order.
# ==========================================================================
DEPTH_INTRINSICS = (251.938, 252.073, 254.289, 255.167)   # fx, fy, cx, cy
DEPTH_DISTORTION = np.array([
    0.445236,      # k1
    -0.0272603,    # k2
    0.000118942,   # p1
    2.88387e-05,   # p2
    -0.00190938,   # k3
    0.786497,      # k4
    0.0487465,     # k5
    -0.0116415,    # k6
])

# ==========================================================================
# Kinect internal: depth camera -> colour camera.
# From the "Color: Extrinsics" block. Direction confirmed by composing the
# "Depth to IMU" and "IMU to Color" blocks, which reproduce it to 5e-7.
# ==========================================================================
R_DEPTH_TO_COLOR = np.array([
    [ 0.99999,    0.00306497, -0.00319136],
    [-0.00274355, 0.995352,    0.0962607 ],
    [ 0.00347156,-0.096251,    0.995351  ],
])
T_DEPTH_TO_COLOR = np.array([-32.0787, -2.20469, 3.8555]) / 1000.0

# ==========================================================================
# Calibration: Kinect colour camera -> 360 camera.
# ==========================================================================
equiMpc = np.array([
    [ 0.999424,   0.0225417,  0.0253641, -0.0673929],
    [-0.022692,   0.999727,   0.00565356, 0.0971137],
    [-0.0252297, -0.00622587, 0.999662,   0.106168],
    [ 0.0,        0.0,        0.0,        1.0],
])
R_EXTR = equiMpc[:3, :3]
T_EXTR = equiMpc[:3, 3]

DEPTH_IS_IN_COLOUR_FRAME = False  # True only for /depth_to_rgb/image_raw
MAX_RANGE_M = 2.88   # WFOV_2X2BINNED upper limit
MIN_RANGE_M = 0.40   # below this is sensor artefact, not scene

# ==========================================================================
# Bag topics. No leading or trailing whitespace.
# ==========================================================================
RGB_TOPIC = "/theta/image_raw"
DEPTH_TOPIC = "/depth/image_raw"
INFO_TOPIC = "/depth/camera_info"

# ==========================================================================
# Virtual camera defaults. 518x518 at 60 deg is what the overlay uses; the
# depthpro driver imports these so both stay on one source of truth.
# NOTE: the depthpro path is metric and sets absolute scale from fx, so the
# 60-vs-90 fov_x choice matters there. Settle it when we do depthpro.
# ==========================================================================
OUT_W, OUT_H = 518, 518
FOV_X = 60.0

_ray_cache = {}


def undistorted_rays(shape, intr, dist):
    """
    Normalised ray direction (x', y') for every pixel, lens distortion
    removed. A pixel at depth z becomes the 3D point (x'*z, y'*z, z).
    Cached, since it only depends on the camera, not the frame.
    """
    h, w = shape
    key = (h, w, tuple(intr), dist.tobytes() if dist is not None else None)
    if key in _ray_cache:
        return _ray_cache[key]

    fx, fy, cx, cy = intr
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))

    if dist is None:
        rays = np.stack([(u - cx) / fx, (v - cy) / fy], axis=-1)
    else:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        pts = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(pts.astype(np.float64), K,
                                   np.asarray(dist, dtype=np.float64))
        rays = rays.reshape(h, w, 2).astype(np.float32)

    _ray_cache[key] = rays
    return rays


def _rot_yp(yaw_deg, pitch_deg):
    """Yaw-then-pitch rotation, identical convention to the overlay's rot_yp."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx


def build_chain(R_v, dyaw=0.0, dpitch=0.0):
    """Full transform from the depth camera frame to the virtual camera.

    dyaw/dpitch are the SAME manual nudge the overlay applies (calibrate_view
    --dyaw/--dpitch). They left-multiply the depth->equirect chain, exactly as
    the overlay does (Rd @ R_ch), so passing the values you tuned in the
    overlay reproduces that alignment in the compare() path. At dyaw=dpitch=0
    this is the original chain unchanged.
    """
    if DEPTH_IS_IN_COLOUR_FRAME:
        R, t = R_EXTR, T_EXTR
    else:
        R = R_EXTR @ R_DEPTH_TO_COLOR
        t = R_EXTR @ T_DEPTH_TO_COLOR + T_EXTR

    if dyaw or dpitch:
        Rd = _rot_yp(dyaw, dpitch)
        R = Rd @ R
        t = Rd @ t

    R_v = np.asarray(R_v, dtype=np.float64)
    return R_v.T @ R, R_v.T @ t


# ==========================================================================
# STEP 3 - move the Kinect depth to the 360 camera's viewpoint
# ==========================================================================
def kinect_depth_to_virtual(depth_mm, kinect_intr, R_v, out_wh, fov_x,
                            distortion=DEPTH_DISTORTION,
                            splat=1, max_range_m=MAX_RANGE_M,
                            dyaw=0.0, dpitch=0.0):
    """
    depth_mm    : (h, w) uint16 from /depth/image_raw, millimetres
    kinect_intr : (fx, fy, cx, cy) from /depth/camera_info
    distortion  : 8 coefficients, or None to skip (do not skip for WFOV)
    R_v         : rotation from compute_camera_parameters()
    out_wh      : (width, height) of your equi2pers output
    dyaw/dpitch : manual nudge (degrees), same as calibrate_view --dyaw/--dpitch

    Returns (gt_metres, valid_mask) on the equi2pers grid.
    """
    W, H = out_wh
    fxv = (W / 2.0) / np.tan(np.radians(fov_x) / 2.0)
    fyv, cxv, cyv = fxv, W / 2.0, H / 2.0

    R_tot, t_tot = build_chain(R_v, dyaw=dyaw, dpitch=dpitch)

    z = depth_mm.astype(np.float32) / 1000.0
    ok = (z >= MIN_RANGE_M) & (z <= max_range_m)
    if not ok.any():
        return np.zeros((H, W), np.float32), np.zeros((H, W), bool)

    rays = undistorted_rays(depth_mm.shape, kinect_intr, distortion)

    zz_src = z[ok]
    P = np.stack([rays[..., 0][ok] * zz_src,
                  rays[..., 1][ok] * zz_src,
                  zz_src], axis=-1)

    P = P @ R_tot.T + t_tot
    P = P[P[:, 2] > 1e-6]

    uu = fxv * P[:, 0] / P[:, 2] + cxv
    vv = fyv * P[:, 1] / P[:, 2] + cyv
    zz = P[:, 2]

    buf = np.full(H * W, np.inf, np.float32)
    for dv in range(-splat, splat + 1):
        for du in range(-splat, splat + 1):
            ui = np.round(uu + du).astype(np.int32)
            vi = np.round(vv + dv).astype(np.int32)
            m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            np.minimum.at(buf, vi[m] * W + ui[m], zz[m])

    gt = buf.reshape(H, W)
    valid = np.isfinite(gt)
    gt[~valid] = 0.0
    return gt, valid


# ==========================================================================
# STEP 4 - put the AI output on the same scale as the Kinect
# ==========================================================================
def align_relative(pred, gt, mask):
    """MiDaS / Depth Anything: relative output, unknown scale and offset."""
    d_gt = 1.0 / np.clip(gt[mask], 1e-6, None)
    d_pr = pred[mask].astype(np.float64)
    A = np.stack([d_pr, np.ones_like(d_pr)], axis=1)

    if np.linalg.cond(A) > 1e10:
        raise ValueError("Scene has almost no depth variation; scale and "
                         "offset cannot be determined.")

    a, b = np.linalg.lstsq(A, d_gt, rcond=None)[0]
    return 1.0 / np.clip(a * pred + b, 1e-6, None), (float(a), float(b))


def align_metric(pred, gt, mask, apply_scale=False):
    """Metric models (Depth Pro / Metric3D / UniK3D): output is already metres.

    For a metric BENCHMARK we do NOT rescale -- the raw absolute error is the
    thing being measured. We still COMPUTE the scale that would align pred to
    gt and return it as a diagnostic (a value far from 1.0 means the model's
    absolute metres are off; e.g. ~0.72 => ~40% too far), but by default we do
    not apply it, so AbsRel/delta1 reflect the model's true metric accuracy.

    Set apply_scale=True to fall back to the old scale-fitting behaviour.
    """
    s = float(np.median(gt[mask]) / np.median(np.clip(pred[mask], 1e-6, None)))
    if apply_scale:
        return pred * s, (s,)
    return pred, (s,)   # raw pred; s reported for the record only


# ==========================================================================
# STEP 5 - the numbers
# ==========================================================================
def metrics(pred_m, gt_m, mask):
    p, g = pred_m[mask], gt_m[mask]
    # Exclude non-positive depths: a 0 means "no reading", not "0 m", and it
    # would divide-by-zero in the ratio. Keep only pixels valid in BOTH maps.
    good = (g > 0) & (p > 0)
    p, g = p[good], g[good]
    n = int(good.sum())
    if n == 0:
        return {"n_px": 0, "AbsRel": float("nan"), "RMSE_m": float("nan"),
                "MAE_m": float("nan"), "delta1": 0.0, "delta2": 0.0}
    ratio = np.maximum(p / g, g / p)
    return {
        "n_px":   n,
        "AbsRel": float(np.mean(np.abs(p - g) / g)),
        "RMSE_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "MAE_m":  float(np.mean(np.abs(p - g))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
    }


def compare(pred, depth_mm, kinect_intr, R_v, out_wh, fov_x,
            metric_model=False, distortion=DEPTH_DISTORTION,
            dyaw=0.0, dpitch=0.0):
    """Steps 3 + 4 + 5 in one call.

    dyaw/dpitch: manual nudge (degrees) forwarded to the warp, identical to
    calibrate_view --dyaw/--dpitch. Pass the values you confirmed in the
    overlay so the metrics use the same alignment you verified by eye.
    """
    gt, valid = kinect_depth_to_virtual(depth_mm, kinect_intr, R_v,
                                        out_wh, fov_x, distortion=distortion,
                                        dyaw=dyaw, dpitch=dpitch)
    if valid.sum() < 100:
        raise ValueError("Almost no overlap between the two views. "
                         "Check R_v and the calibration chain.")

    align = align_metric if metric_model else align_relative
    pred_m, params = align(pred, gt, valid)

    return {
        "metrics":   metrics(pred_m, gt, valid),
        "pred_m":    pred_m,
        "gt_m":      gt,
        "valid":     valid,
        "fit":       params,
        "error_map": np.where(valid, np.abs(pred_m - gt), 0.0),
    }


# ==========================================================================
# Shared bag reading + pseudo rendering (moved here so nothing imports from
# main_depthpro_eval.py -- keeps the dependency one-directional).
# ==========================================================================
def open_bag(bag_dir):
    """
    Humble's rosbag2 does not auto-detect the storage plugin from an empty
    storage_id, so it has to be named. Try sqlite3 first, then mcap.
    """
    last = None
    for sid in ("sqlite3", "mcap"):
        reader = SequentialReader()
        try:
            reader.open(
                StorageOptions(uri=str(bag_dir), storage_id=sid),
                ConverterOptions(input_serialization_format="cdr",
                                 output_serialization_format="cdr"),
            )
            return reader
        except RuntimeError as e:
            last = e
    raise RuntimeError(
        f"Could not open {bag_dir}. Check it is the bag DIRECTORY (the one "
        f"containing metadata.yaml), not the .db3 file. Last error: {last}")


# def stream_pairs(bag_dir, max_depth_buffer=8):
#     """
#     Yield (stamp_ns, equirect_bgr, depth_mm, kinect_intr, kinect_dist).

#     Streams instead of loading everything: the 748 Theta frames alone are
#     16.5 GB. Depth frames are only 524 kB each, so a short buffer of them
#     is kept and each Theta frame is matched to its nearest in time.
#     """
#     reader = open_bag(bag_dir)
#     bridge = CvBridge()

#     intr = dist = None
#     depth_buf = deque(maxlen=max_depth_buffer)
#     pending = []

#     def flush(force=False):
#         while pending:
#             t_stamp, rgb = pending[0]
#             if not depth_buf:
#                 return
#             # wait until a depth frame later than this Theta frame exists,
#             # otherwise "nearest" might still be ahead of us
#             if not force and depth_buf[-1][0] < t_stamp:
#                 return
#             j = min(range(len(depth_buf)),
#                     key=lambda i: abs(depth_buf[i][0] - t_stamp))
#             pending.pop(0)
#             yield t_stamp, rgb, depth_buf[j][1]

#     while reader.has_next():
#         topic, data, stamp_ns = reader.read_next()

#         if topic == INFO_TOPIC and intr is None:
#             m = ser.deserialize_message(data, CameraInfo)
#             intr = (m.k[0], m.k[4], m.k[2], m.k[5])
#             dist = np.array(m.d, dtype=np.float64) if len(m.d) else None

#         elif topic == DEPTH_TOPIC:
#             m = ser.deserialize_message(data, Image)
#             depth_buf.append((stamp_ns, bridge.imgmsg_to_cv2(m, "passthrough")))
#             for s, rgb, d in flush():
#                 yield s, rgb, d, intr, dist

#         elif topic == RGB_TOPIC:
#             m = ser.deserialize_message(data, Image)
#             pending.append((stamp_ns, bridge.imgmsg_to_cv2(m, "bgr8")))

#     for s, rgb, d in flush(force=True):
#         yield s, rgb, d, intr, dist
def stream_pairs(bag_dir, max_depth_buffer=8):
    """
    Yield (stamp_ns, equirect_bgr, depth_mm, kinect_intr, kinect_dist).

    Streams instead of loading everything. Each Theta frame is paired with
    the temporally nearest Azure depth frame. Synchronization statistics
    are reported when the generator closes.
    """
    reader = open_bag(bag_dir)
    bridge = CvBridge()

    intr = dist = None
    depth_buf = deque(maxlen=max_depth_buffer)
    pending = []

    # Store pairing errors (ms)
    pair_errors_ms = []

    def flush(force=False):
        while pending:
            t_stamp, rgb = pending[0]

            if not depth_buf:
                return

            # Wait until we have seen a depth frame after this RGB frame
            if not force and depth_buf[-1][0] < t_stamp:
                return

            # Find temporally closest depth frame
            j = min(
                range(len(depth_buf)),
                key=lambda i: abs(depth_buf[i][0] - t_stamp)
            )

            depth_stamp, depth = depth_buf[j]

            # Synchronization error
            dt_ms = abs(depth_stamp - t_stamp) / 1e6
            pair_errors_ms.append(dt_ms)

            pending.pop(0)

            yield t_stamp, rgb, depth

    try:
        while reader.has_next():
            topic, data, stamp_ns = reader.read_next()

            if topic == INFO_TOPIC and intr is None:
                m = ser.deserialize_message(data, CameraInfo)

                intr = (
                    m.k[0],   # fx
                    m.k[4],   # fy
                    m.k[2],   # cx
                    m.k[5]    # cy
                )

                dist = (
                    np.array(m.d, dtype=np.float64)
                    if len(m.d)
                    else None
                )

            elif topic == DEPTH_TOPIC:
                m = ser.deserialize_message(data, Image)

                depth_buf.append(
                    (
                        stamp_ns,
                        bridge.imgmsg_to_cv2(
                            m,
                            "passthrough"
                        )
                    )
                )

                for s, rgb, d in flush():
                    yield s, rgb, d, intr, dist

            elif topic == RGB_TOPIC:
                m = ser.deserialize_message(data, Image)

                pending.append(
                    (
                        stamp_ns,
                        bridge.imgmsg_to_cv2(
                            m,
                            "bgr8"
                        )
                    )
                )

        # Normal end-of-bag flush
        for s, rgb, d in flush(force=True):
            yield s, rgb, d, intr, dist

    finally:
        # Runs even when caller breaks with --max-frames
        if pair_errors_ms:
            arr = np.asarray(pair_errors_ms)

            print("\n========== RGB-Depth Pairing Statistics ==========")
            print(f"Pairs            : {len(arr)}")
            print(f"Mean Δt          : {arr.mean():.2f} ms")
            print(f"Median Δt        : {np.median(arr):.2f} ms")
            print(f"Std Dev Δt       : {arr.std():.2f} ms")
            print(f"95th Percentile  : {np.percentile(arr,95):.2f} ms")
            print(f"Maximum Δt       : {arr.max():.2f} ms")
        else:
            print("\n========== RGB-Depth Pairing Statistics ==========")
            print("No RGB-Depth pairs generated.")
            
def render_pseudo(equirect_bgr, yaw_deg, pitch_deg,
                  fov_x=None, w=None, h=None, device="cpu"):
    """equi2pers: rotation only, no depth.

    Returns (pseudo_bgr, K_np, R_np):
        pseudo_bgr : (h, w, 3) uint8 BGR crop
        K_np       : (3, 3) virtual-camera intrinsics from
                     compute_camera_parameters (replaces virtual_K)
        R_np       : (3, 3) virtual-camera orientation -- pass this into
                     compare() as R_v so off-centre crops warp correctly.
                     Built from the SAME yaw_deg/pitch_deg given to equi2pers,
                     so it is identity at yaw=pitch=0 and a real rotation
                     otherwise.

    NOTE on sign convention: equi2pers is fed NEGATED angles (-yaw_deg,
    -pitch_deg) while compute_camera_parameters is fed the un-negated
    yaw_deg/pitch_deg. Verify on a real off-centre frame that the overlay
    still lines up; if it slips, one of these two needs its sign flipped.
    """
    fov_x = FOV_X if fov_x is None else float(fov_x)
    w = OUT_W if w is None else int(w)
    h = OUT_H if h is None else int(h)

    equi_tensor = preprocess(equirect_bgr, device=torch.device(device))
    with torch.inference_mode():
        pseudo_camera = equi2pers_run(
            equi=equi_tensor,
            rots=[{"yaw": float(np.deg2rad(-yaw_deg)),
                   "roll": 0.0,
                   "pitch": float(np.deg2rad(-pitch_deg))}],
            height=h,
            width=w,
            fov_x=fov_x,
            skew=0.0,
            z_down=False,
            mode="bilinear",
        )

    input_tensor = pseudo_camera.to(dtype=torch.float32)

    # compute_camera_parameters(out_width, out_height, fov, yaw, pitch)
    # -> the 4th/5th args are the ACTUAL yaw/pitch, not zeros.
    K, R, t_cam = compute_camera_parameters(
        input_tensor.size(3),   # out_width
        input_tensor.size(2),   # out_height
        float(fov_x),
        float(yaw_deg),
        float(pitch_deg),
    )
    R_np = np.asarray(R, dtype=np.float32)
    K_np = np.asarray(K, dtype=np.float32)

    img = input_tensor[0] if input_tensor.dim() == 4 else input_tensor
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    pseudo_bgr = np.ascontiguousarray(img[:, :, ::-1])
    return pseudo_bgr, K_np, R_np


# ==========================================================================
# Overlay tool
# ==========================================================================
def rot_yp(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx


def warp_depth(depth_mm, intr, dist, R_tot, t_tot, W, H, fov_x, splat=1):
    rays = undistorted_rays(depth_mm.shape, intr, dist)
    z = depth_mm.astype(np.float32) / 1000.0
    ok = (z >= MIN_RANGE_M) & (z <= MAX_RANGE_M)
    zs = z[ok]
    P = np.stack([rays[..., 0][ok] * zs, rays[..., 1][ok] * zs, zs], axis=-1)

    fxv = (W / 2.0) / math.tan(math.radians(fov_x) / 2.0)
    Q = P @ R_tot.T + t_tot
    Q = Q[Q[:, 2] > 1e-6]
    uu = fxv * Q[:, 0] / Q[:, 2] + W / 2.0
    vv = fxv * Q[:, 1] / Q[:, 2] + H / 2.0

    buf = np.full(H * W, np.inf, np.float32)
    for dv in range(-splat, splat + 1):
        for du in range(-splat, splat + 1):
            ui = np.round(uu + du).astype(np.int32)
            vi = np.round(vv + dv).astype(np.int32)
            m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            np.minimum.at(buf, vi[m] * W + ui[m], Q[:, 2][m])

    d = buf.reshape(H, W)
    valid = np.isfinite(d)
    d[~valid] = 0.0
    return d, valid


def tag(img, text):
    o = img.copy()
    cv2.rectangle(o, (0, 0), (o.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(o, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return o


def chain_selftest():
    """Print the composed chain angle/offset and the distortion correction."""
    R, t = build_chain(np.eye(3))
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print("depth camera -> 360 camera")
    print(f"  rotation    {ang:.3f} deg")
    print(f"  translation {np.linalg.norm(t)*1000:.1f} mm")
    print(f"  DEPTH_IS_IN_COLOUR_FRAME = {DEPTH_IS_IN_COLOUR_FRAME}")

    rays_d = undistorted_rays((518, 518), DEPTH_INTRINSICS, DEPTH_DISTORTION)
    rays_p = undistorted_rays((518, 518), DEPTH_INTRINSICS, None)
    for r in (64, 128, 192, 250):
        a = np.append(rays_p[255, 254 + r], 1.0); a /= np.linalg.norm(a)
        b = np.append(rays_d[255, 254 + r], 1.0); b /= np.linalg.norm(b)
        d = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
        print(f"  distortion correction at r={r:3d} px: {d:5.2f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="print the chain + distortion diagnostic and exit")
    ap.add_argument("--bag")
    ap.add_argument("--blend-out", default=None,
                    help="save just the colored depth-on-photo blend, no grid/labels")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--height", type=int, default=518)
    ap.add_argument("--dyaw", type=float, default=0.0,
                    help="nudge the depth left/right, degrees")
    ap.add_argument("--dpitch", type=float, default=0.0,
                    help="nudge the depth up/down, degrees")
    ap.add_argument("--no-dc", action="store_true",
                    help="skip the depth->colour hop")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="depth opacity in the blended panel")
    ap.add_argument("--color", action="store_true",
                    help="colormap the standalone depth panel instead of greyscale")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="overlay.png")
    args = ap.parse_args()

    if args.selftest:
        chain_selftest()
        return
    if not args.bag:
        raise SystemExit("--bag is required (or use --selftest)")

    W, H, FOV = args.width, args.height, args.fov
    fx = (W / 2) / math.tan(math.radians(FOV) / 2)
    print(f"crop: yaw {args.yaw:g}  pitch {args.pitch:g}  fov {FOV:g}  {W}x{H}")
    print(f"  fx={fx:.1f}  fov_y={math.degrees(2*math.atan((H/2)/fx)):.2f} deg"
          f"  {W/FOV:.2f} px/deg")
    if args.dyaw or args.dpitch:
        print(f"  depth nudged by dyaw {args.dyaw:+g}  dpitch {args.dpitch:+g}")

    got = None
    for i, item in enumerate(stream_pairs(args.bag)):
        if i == args.frame:
            got = item
            break
    if got is None:
        raise SystemExit(f"frame {args.frame} not found")
    _, equirect, depth_mm, k_intr, k_dist = got

    pseudo, _K_np, _R_np = render_pseudo(equirect, args.yaw, args.pitch,
                                         FOV, W, H, args.device)

    if args.no_dc:
        R_ch, t_ch = R_EXTR, T_EXTR
    else:
        R_ch = R_EXTR @ R_DEPTH_TO_COLOR
        t_ch = R_EXTR @ T_DEPTH_TO_COLOR + T_EXTR

    Rd = rot_yp(args.dyaw, args.dpitch)
    dist = k_dist if k_dist is not None else DEPTH_DISTORTION
    d, valid = warp_depth(depth_mm, k_intr, dist,
                          Rd @ R_ch, Rd @ t_ch, W, H, FOV)

    if not valid.any():
        raise SystemExit("nothing landed in the view - check the nudge values")

    lo, hi = np.percentile(d[valid], [2, 98])
    print(f"depth in view: {d[valid].min():.3f}-{d[valid].max():.3f} m"
          f"  (p2 {lo:.3f}, p98 {hi:.3f})   coverage {valid.mean()*100:.1f}%")

    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)

    # Standalone depth panel: respects --color (grey by default).
    if args.color:
        dvis = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
    else:
        g = (255 - norm * 255).astype(np.uint8)      # near = white
        dvis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    dvis[~valid] = (0, 0, 0)

    # Blend panel: always Turbo colour, independent of --color.
    dvis_color = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                   cv2.COLORMAP_TURBO)
    dvis_color[~valid] = (0, 0, 0)
    blend = pseudo.copy()
    blend[valid] = cv2.addWeighted(pseudo, 1 - args.alpha,
                                   dvis_color, args.alpha, 0)[valid]

    core = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    dd = np.where(valid, d, 0).astype(np.float32)
    mag = np.hypot(cv2.Sobel(dd, cv2.CV_32F, 1, 0, 3),
                   cv2.Sobel(dd, cv2.CV_32F, 0, 1, 3))
    edges = (mag > 0.04 * np.maximum(dd, 0.3)) & core
    eov = pseudo.copy()
    eov[edges] = (0, 255, 0)

    nudge = (f"  d{args.dyaw:+g}/{args.dpitch:+g}"
             if (args.dyaw or args.dpitch) else "")
    panel = np.vstack([
        np.hstack([tag(pseudo, f" Pseudo perspective  pitch {args.pitch:g} fov {FOV:g}"),
                   tag(dvis, f"Azure Kinect depth  {lo:.2f}-{hi:.2f} m"
                             f"{'  near=white' if not args.color else ''}")]),
        np.hstack([tag(blend, f"blended  Masked"),
                   tag(eov, f"depth edges")]),
    ])

    if args.blend_out is not None:
        cv2.imwrite(args.blend_out, blend)
        print(f"wrote {args.blend_out}  (clean blend "
              f"{blend.shape[1]}x{blend.shape[0]})")
    cv2.imwrite(args.out, panel)
    print(f"wrote {args.out}  ({panel.shape[1]}x{panel.shape[0]})")


if __name__ == "__main__":
    main()