#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kinect <-> 360 comparison in EQUIRECTANGULAR space.

EQUIRECT PIXEL CONVENTION -- VERIFY THIS BEFORE TRUSTING ANY NUMBER
-------------------------------------------------------------------
A 3D point P = (X, Y, Z) in the 360 camera frame maps to
    r   = ||P||
    lon = atan2(X, Z)                  0 at +Z, growing toward +X
    lat = asin(Y / r)                  positive = DOWN (y-down convention)
    u   = (lon / 2pi + 0.5) * W
    v   = (lat / pi  + 0.5) * H
This matches the usual equilib / OpenCV y-down layout, but conventions differ
between libraries by a sign or a half-turn. Run the overlay before believing
the metrics:

    python3 theta_kinect_compare_equi.py --bag ~/0.5m --overlay equi_check.png

The Kinect patch must land on the SAME objects in the panorama it is drawn
over. If it is mirrored, flip --lon-sign; if upside down, flip --lat-sign; if
half a panorama off, use --lon-offset 180.

Usage as a library:
    from theta_kinect_compare_equi import compare_equi
    res = compare_equi(pred_equi, depth_mm, k_intr, eq_wh=(960, 480))
"""

import argparse
import sys

sys.path.insert(0, '/opt/ros/humble/lib/python3.10/dist-packages')

import cv2
import numpy as np

from theta_kinect_compare import (
    build_chain, undistorted_rays, metrics, align_metric, align_relative,
    stream_pairs, DEPTH_DISTORTION, MIN_RANGE_M, MAX_RANGE_M,
)


# ==========================================================================
# Kinect depth -> equirectangular grid
# ==========================================================================
def kinect_depth_to_equirect(depth_mm, kinect_intr, eq_wh,
                             distortion=DEPTH_DISTORTION,
                             splat=1, max_range_m=MAX_RANGE_M,
                             dyaw=0.0, dpitch=0.0,
                             lon_sign=1.0, lat_sign=1.0, lon_offset_deg=0.0):
    """Warp Azure depth onto the 360 camera's equirect grid.

    Args:
        depth_mm    : (h, w) uint16 from /depth/image_raw, millimetres
        kinect_intr : (fx, fy, cx, cy) from /depth/camera_info
        eq_wh       : (W, H) of the equirect evaluation grid
        dyaw/dpitch : the SAME manual nudge tuned in calibrate_view
        lon_sign / lat_sign / lon_offset_deg : convention escapes, see header

    Returns:
        (gt_radial, valid) on the equirect grid. gt_radial is ||P|| in metres,
        directly comparable to a Spherical-camera model's depth output.

    The chain is reused from theta_kinect_compare.build_chain with R_v =
    identity: the equirect grid IS the 360 camera frame, so there is no
    virtual-camera rotation to undo.
    """
    W, H = eq_wh
    R_tot, t_tot = build_chain(np.eye(3), dyaw=dyaw, dpitch=dpitch)

    z = depth_mm.astype(np.float32) / 1000.0
    ok = (z >= MIN_RANGE_M) & (z <= max_range_m)
    if not ok.any():
        return np.zeros((H, W), np.float32), np.zeros((H, W), bool)

    rays = undistorted_rays(depth_mm.shape, kinect_intr, distortion)
    zz = z[ok]
    P = np.stack([rays[..., 0][ok] * zz,
                  rays[..., 1][ok] * zz,
                  zz], axis=-1)

    # depth camera -> 360 camera frame
    P = P @ R_tot.T + t_tot

    r = np.linalg.norm(P, axis=1)
    good = r > 1e-6
    P, r = P[good], r[good]

    # 3D -> spherical -> equirect pixel
    lon = lon_sign * np.arctan2(P[:, 0], P[:, 2])
    lon = lon + np.radians(lon_offset_deg)
    lon = (lon + np.pi) % (2.0 * np.pi) - np.pi          # wrap to [-pi, pi)
    lat = lat_sign * np.arcsin(np.clip(P[:, 1] / r, -1.0, 1.0))

    uu = (lon / (2.0 * np.pi) + 0.5) * W
    vv = (lat / np.pi + 0.5) * H

    # nearest-surface z-buffer, with a small splat so the patch is dense
    buf = np.full(H * W, np.inf, np.float32)
    for dv in range(-splat, splat + 1):
        for du in range(-splat, splat + 1):
            ui = np.round(uu + du).astype(np.int32) % W   # wrap horizontally
            vi = np.round(vv + dv).astype(np.int32)
            m = (vi >= 0) & (vi < H)
            np.minimum.at(buf, vi[m] * W + ui[m], r[m])

    gt = buf.reshape(H, W)
    valid = np.isfinite(gt)
    gt[~valid] = 0.0
    return gt, valid


def resize_pred_to_eval(pred_equi, eq_wh):
    """Bring a model's equirect depth onto the evaluation grid.

    INTER_NEAREST on purpose: bilinear across a depth discontinuity invents
    values that lie between foreground and background and belong to neither.
    """
    W, H = eq_wh
    if pred_equi.shape[1] == W and pred_equi.shape[0] == H:
        return pred_equi.astype(np.float32)
    return cv2.resize(pred_equi.astype(np.float32), (W, H),
                      interpolation=cv2.INTER_NEAREST)


def equirect_radial_to_z(r_map, max_off_axis_deg=85.0):
    """Convert radial distance to z about the panorama's +Z axis.

    z = r * cos(lat) * cos(lon), i.e. the component along the direction that
    sits at the centre column of the equirect.

    ONLY MEANINGFUL IN THE FORWARD HEMISPHERE. Beyond 90 degrees off that axis
    the cosine goes negative and "z" stops describing a distance at all --
    which is precisely why the equirect path scores radial by default. Pixels
    beyond max_off_axis_deg are returned as 0 (invalid) rather than as a
    misleading small or negative number.

    Use this only to put metre-valued errors (RMSE_m, MAE_m) on the same
    footing as a pinhole run. AbsRel and delta1 need no conversion: they are
    ratios, so the per-pixel cosine cancels on both sides.
    """
    H, W = r_map.shape
    lon = (np.arange(W, dtype=np.float32) / W - 0.5) * 2.0 * np.pi
    lat = (np.arange(H, dtype=np.float32) / H - 0.5) * np.pi
    c = np.cos(lat)[:, None] * np.cos(lon)[None, :]
    z = r_map * c
    z[c < np.cos(np.radians(max_off_axis_deg))] = 0.0
    z[z < 0] = 0.0
    return z.astype(np.float32)


def compare_equi(pred_equi, depth_mm, kinect_intr, eq_wh=(960, 480),
                 metric_model=True, distortion=DEPTH_DISTORTION,
                 dyaw=0.0, dpitch=0.0, splat=1,
                 lon_sign=1.0, lat_sign=1.0, lon_offset_deg=0.0,
                 min_valid_px=100, metrics_in="radial"):
    """Equirect twin of theta_kinect_compare.compare().

    Returns the SAME dict shape (metrics / pred_m / gt_m / valid / fit /
    error_map) so the eval mains and CSV writing need no changes -- only the
    grid differs.

    metrics_in:
        "radial" (default) -- score distance from the optical centre. Correct
            on a sphere and defined everywhere.
        "z" -- convert BOTH sides to z about the +Z axis before scoring, so
            that RMSE_m / MAE_m are in the same units as a pinhole run and can
            sit in the same table. Costs coverage: everything beyond 85 deg
            off-axis is dropped, because z is undefined there.

    Note that AbsRel, delta1 and delta2 are RATIO metrics and are unchanged by
    this choice -- the per-pixel cosine cancels between prediction and ground
    truth. Only the metre-valued metrics move.
    """
    gt, valid = kinect_depth_to_equirect(
        depth_mm, kinect_intr, eq_wh, distortion=distortion, splat=splat,
        dyaw=dyaw, dpitch=dpitch, lon_sign=lon_sign, lat_sign=lat_sign,
        lon_offset_deg=lon_offset_deg)

    if valid.sum() < min_valid_px:
        raise ValueError(
            f"Only {int(valid.sum())} Kinect pixels landed on the equirect "
            f"grid. Check the lon/lat convention (run --overlay) and the "
            f"dyaw/dpitch nudge.")

    pred = resize_pred_to_eval(pred_equi, eq_wh)

    if metrics_in == "z":
        pred = equirect_radial_to_z(pred)
        gt = equirect_radial_to_z(gt)
        valid = valid & (gt > 0) & (pred > 0)
        if valid.sum() < min_valid_px:
            raise ValueError(
                "Converting to z left too few pixels: the Kinect's coverage "
                "must sit in the forward hemisphere for z to be defined. "
                "Score in radial instead.")

    align = align_metric if metric_model else align_relative
    pred_m, params = align(pred, gt, valid)

    # RAW absolute error only. This is a METRIC benchmark: the whole point is
    # how far the model's own metres are from the sensor's, so nothing is
    # rescaled before scoring. `scale` is still reported as a diagnostic --
    # the median gt/pred ratio, where 1.0 means the absolute metres are
    # correct -- but it is never applied.
    return {
        "metrics":        metrics(pred_m, gt, valid),
        "scale":          float(params[0]) if params else 1.0,
        "pred_m":         pred_m,
        "gt_m":           gt,
        "valid":          valid,
        "fit":            params,
        "error_map":      np.where(valid, np.abs(pred_m - gt), 0.0),
    }


# ==========================================================================
# Why the warped numbers differ from the raw sensor numbers
# ==========================================================================
def diagnose_warp(depth_mm, kinect_intr, gt_radial, valid,
                  distortion=DEPTH_DISTORTION, max_range_m=MAX_RANGE_M):
    """Decompose the difference between the raw Azure depth and the warped map.

    Three effects stack, and they are easy to mistake for a bug:

      1. z vs radial. The sensor stores z along its optical axis; the warp
         stores r = ||P||, distance from the 360 camera's centre. They are
         equal only on-axis and diverge as r = z / cos(theta) -- a factor of 2
         at the 60 deg edge of a 120 deg field of view.
      2. range gating. The warp drops z outside [MIN_RANGE_M, max_range_m]
         before projecting; a raw display of every z > 0 shows values the
         warped map never contained.
      3. extrinsics. The chain translates by roughly 19 cm, which shifts
         distances by a few percent depending on direction.

    Prints the raw stats, the same stats after gating, the warped stats, and
    the measured median r/z ratio next to the ratio predicted from the
    geometry. If measured and predicted agree, the change is the coordinate
    change and nothing is wrong.
    """
    z = depth_mm.astype(np.float32) / 1000.0
    pos = z > 0
    gate = (z >= MIN_RANGE_M) & (z <= max_range_m)

    print("\n--- raw sensor vs warped ---")
    if pos.any():
        print(f"raw z, all z>0      : {z[pos].min():.3f} .. "
              f"{z[pos].max():.3f} m   median {np.median(z[pos]):.3f}   "
              f"{int(pos.sum())} px")
    if gate.any():
        print(f"raw z, gated {MIN_RANGE_M:.2f}-{max_range_m:.2f}: "
              f"{z[gate].min():.3f} .. {z[gate].max():.3f} m   "
              f"median {np.median(z[gate]):.3f}   {int(gate.sum())} px "
              f"({gate.sum()/max(pos.sum(),1)*100:.0f}% of z>0 survive)")
    else:
        print("raw z, gated        : nothing survives the range gate")
        return

    if valid.any():
        g = gt_radial[valid]
        print(f"warped radial       : {g.min():.3f} .. {g.max():.3f} m   "
              f"median {np.median(g):.3f}   {int(valid.sum())} px")
        meas = np.median(g) / np.median(z[gate])
    else:
        print("warped radial       : nothing landed on the grid")
        return

    # predicted r/z from the sensor's own geometry: mean 1/cos(theta) over
    # the gated pixels, using the undistorted ray directions.
    rays = undistorted_rays(depth_mm.shape, kinect_intr, distortion)
    sec = np.sqrt(rays[..., 0] ** 2 + rays[..., 1] ** 2 + 1.0)
    pred = float(np.median(sec[gate]))

    print(f"median r/z measured : {meas:.3f}")
    print(f"median 1/cos(theta) : {pred:.3f}   "
          f"(pure z->radial prediction, before the ~19 cm extrinsics)")
    off = abs(meas - pred) / max(pred, 1e-6) * 100
    if off < 15:
        print(f"-> agree to {off:.0f}%: the difference IS the z->radial "
              f"coordinate change, not an error.")
    else:
        print(f"-> differ by {off:.0f}%: more than the coordinate change "
              f"explains. Check the extrinsics chain and the lon/lat "
              f"convention (run --overlay).")


# ==========================================================================
# Overlay: verify the equirect convention BEFORE trusting any metric
# ==========================================================================
def _overlay(equirect_bgr, gt, valid, alpha=0.6, outline=None):
    """Draw the warped Kinect patch over the panorama it should align with."""
    H, W = gt.shape
    base = cv2.resize(equirect_bgr, (W, H), interpolation=cv2.INTER_AREA)
    if not valid.any():
        return base
    lo, hi = np.percentile(gt[valid], [2, 98])
    norm = np.clip((gt - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out = base.copy()
    out[valid] = cv2.addWeighted(base, 1 - alpha, vis, alpha, 0)[valid]
    edges = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_GRADIENT,
                             np.ones((3, 3), np.uint8)) > 0
    out[edges] = (255, 255, 255)
    if outline is not None:
        out[outline] = (0,0, 255)      # amber: the 60 deg scored window
    return out



def _sweep(equirect, depth_mm, k_intr, k_dist, eq_wh, args):
    """Contact sheet over a dyaw/dpitch grid, so the nudge can be read off by
    eye instead of guessed.

    IMPORTANT: the dyaw/dpitch tuned in calibrate_view for a PITCHED CROP do
    not carry over here. In that path Rd is the only thing aiming the depth,
    so dpitch absorbs the crop's own pitch as well as the calibration offset
    (which is why it can be tens of degrees). On the sphere there is no crop
    to aim, so the equirect nudge is the calibration offset ALONE and is
    usually small. Re-tune it here rather than reusing the pinhole numbers.
    """
    ys = [args.sweep_yaw0 + i * args.sweep_step
          for i in range(-args.sweep_n, args.sweep_n + 1)]
    ps = [args.sweep_pitch0 + i * args.sweep_step
          for i in range(-args.sweep_n, args.sweep_n + 1)]

    dist = k_dist if k_dist is not None else DEPTH_DISTORTION
    W, H = eq_wh
    cells = []
    print(f"sweeping dyaw {ys[0]:+g}..{ys[-1]:+g}, "
          f"dpitch {ps[0]:+g}..{ps[-1]:+g}, step {args.sweep_step:g} deg")
    for pdeg in ps:
        row = []
        for ydeg in ys:
            gt, valid = kinect_depth_to_equirect(
                depth_mm, k_intr, eq_wh, distortion=dist, splat=args.splat,
                dyaw=ydeg, dpitch=pdeg, lon_sign=args.lon_sign,
                lat_sign=args.lat_sign, lon_offset_deg=args.lon_offset)
            cell = _overlay(equirect, gt, valid)
            cell = cv2.resize(cell, (W // 2, H // 2))
            cv2.rectangle(cell, (0, 0), (cell.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(cell, f"dyaw {ydeg:+.1f}  dpitch {pdeg:+.1f}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            row.append(cell)
        cells.append(np.hstack(row))
    sheet = np.vstack(cells)
    cv2.imwrite(args.sweep_out, sheet)
    print(f"wrote {args.sweep_out} ({sheet.shape[1]}x{sheet.shape[0]})")
    print("Pick the cell where the coloured patch sits on the SAME objects "
          "as the panorama under it, then pass those two numbers as "
          "--dyaw/--dpitch to the eval.")

def _window_outline(shape, args):
    """The 60 deg pinhole window's border, in this grid's own pixels.

    Depends only on the window's aim and size, never on the frame, so one
    call serves every frame and every sweep cell. Returns None when the
    projection cannot be trusted, in which case the overlay draws exactly
    as it did before, with no box and no crash.
    """
    if not args.win_box:
        return None
    try:
        from equi_outline import equi_window_outline
        from equi_pinhole import project_equi_to_pinhole
        from theta_kinect_compare import OUT_W, OUT_H, FOV_X
    except ImportError as e:
        print(f"[outline] skipped: {e}")
        return None
    return equi_window_outline(
        lambda a: project_equi_to_pinhole(
            a, yaw_deg=args.yaw, pitch_deg=args.pitch,
            fov_x=FOV_X, out_wh=(OUT_W, OUT_H), device="cpu"),
        hw=shape, out_wh=(OUT_W, OUT_H), fov_x=FOV_X, verbose=True)

def main():
    ap = argparse.ArgumentParser(
        description="Overlay the Kinect depth on the panorama to verify the "
                    "equirect convention, and sweep the dyaw/dpitch nudge.")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--eq-width", type=int, default=960)
    ap.add_argument("--eq-height", type=int, default=480)
    ap.add_argument("--dyaw", type=float, default=0.0)
    ap.add_argument("--dpitch", type=float, default=0.0)
    ap.add_argument("--splat", type=int, default=1)
    ap.add_argument("--lon-sign", type=float, default=1.0,
                    help="-1 if the patch is mirrored left-right")
    ap.add_argument("--lat-sign", type=float, default=1.0,
                    help="-1 if the patch is upside down")
    ap.add_argument("--lon-offset", type=float, default=0.0,
                    help="degrees; 180 if the patch is half a turn off")
    ap.add_argument("--overlay", default="equi_check.png")
    ap.add_argument("--sweep", action="store_true",
                    help="write a contact sheet over a dyaw/dpitch grid "
                         "instead of a single overlay")
    ap.add_argument("--sweep-n", type=int, default=2,
                    help="cells each side of centre (2 -> 5x5)")
    ap.add_argument("--sweep-step", type=float, default=2.0,
                    help="degrees between cells")
    ap.add_argument("--sweep-yaw0", type=float, default=0.0)
    ap.add_argument("--sweep-pitch0", type=float, default=0.0)
    ap.add_argument("--sweep-out", default="equi_sweep.png")
    ap.add_argument("--win-box", action="store_true",
                    help="draw the 60 deg / 518x518 window on the overlay")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="where the 60 deg window looks; same flag and same "
                         "value as main_eval.py --yaw")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="same flag and same value as main_eval.py --pitch")
    args = ap.parse_args()

    # Some bags emit the first depth image one message BEFORE the first
    # CameraInfo, so the earliest frames have no intrinsics. Skip those and
    # count --frame from the first USABLE one, rather than refusing to run --
    # the eval scripts skip them too, so both tools land on the same frame.
    got = None
    skipped = 0
    usable = 0
    for item in stream_pairs(args.bag):
        if item[3] is None:              # k_intr not seen yet
            skipped += 1
            continue
        if usable == args.frame:
            got = item
            break
        usable += 1
    if got is None:
        raise SystemExit(
            f"no frame with CameraInfo found (skipped {skipped} without it). "
            f"Check that {INFO_TOPIC} exists: ros2 bag info {args.bag}")
    if skipped:
        print(f"skipped {skipped} frame(s) before CameraInfo arrived")
    _, equirect, depth_mm, k_intr, k_dist = got

    eq_wh = (args.eq_width, args.eq_height)

    if args.sweep:
        _sweep(equirect, depth_mm, k_intr, k_dist, eq_wh, args)
        return

    gt, valid = kinect_depth_to_equirect(
        depth_mm, k_intr, eq_wh,
        distortion=k_dist if k_dist is not None else DEPTH_DISTORTION,
        splat=args.splat, dyaw=args.dyaw, dpitch=args.dpitch,
        lon_sign=args.lon_sign, lat_sign=args.lat_sign,
        lon_offset_deg=args.lon_offset)

    if not valid.any():
        raise SystemExit("nothing landed on the equirect grid")

    cov = valid.mean() * 100
    # solid angle actually covered, weighting rows by cos(lat)
    H, W = valid.shape
    lat = (np.arange(H) / H - 0.5) * np.pi
    wgt = np.cos(lat)[:, None] * np.ones((1, W))
    sa = (valid * wgt).sum() / wgt.sum() * 100

    print(f"equirect grid {W}x{H}")
    print(f"  Kinect depth range in view: "
          f"{gt[valid].min():.3f}-{gt[valid].max():.3f} m (radial)")
    print(f"  pixel coverage {cov:.2f}%   solid-angle coverage {sa:.2f}%")
    print(f"  convention: lon_sign {args.lon_sign:+g}  "
          f"lat_sign {args.lat_sign:+g}  lon_offset {args.lon_offset:+g} deg")

    diagnose_warp(depth_mm, k_intr, gt, valid,
                  distortion=k_dist if k_dist is not None
                  else DEPTH_DISTORTION)

    cv2.imwrite(args.overlay, _overlay(equirect, gt, valid,
                                       outline=_window_outline(gt.shape, args)))
    print(f"wrote {args.overlay} -- the coloured patch MUST sit on the same "
          f"objects as the panorama underneath it.")


if __name__ == "__main__":
    main()