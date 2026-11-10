#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared machinery for the monocular-depth benchmark.

One bag reader, one crop, one warp/compare, one CSV schema, one display
panel, one frame loop. Everything model-specific lives in adapters.py.
=========================================================================
"""

import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/opt/ros/humble/lib/python3.10/dist-packages')

# ==========================================================================
# Project layout
# ==========================================================================
# Expected tree (this file lives in Scripts/):
#     Benchmarking/
#         Data/  Models/  Output/  Processors/  Scripts/  Tools/
#
# PROJECT_ROOT is found from THIS file's location, not the shell's working
# directory, so the outputs land in Output/ no matter where you launch from.
# Processors/ and Tools/ are put on sys.path so the plain
# `from Depthpro_processor import ...` imports keep working without turning
# those folders into packages.
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (_THIS_DIR.parent
                if _THIS_DIR.name.lower() in {"scripts", "tools"}
                else _THIS_DIR)

for _sub in ("Processors", "Tools", "Scripts"):
    _p = PROJECT_ROOT / _sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_OUT_DIR = str(PROJECT_ROOT / "Output")

import cv2
import numpy as np
import torch

from theta_kinect_compare import (compare, DEPTH_DISTORTION,
                                  stream_pairs, render_pseudo,
                                  OUT_W, OUT_H, FOV_X)

from theta_kinect_compare_equi import compare_equi
from equi_panel import make_equi_window_panel
from equi_pinhole import compare_equi_pinhole,  project_equi_to_pinhole

from pixel_stability import (PixelStability, append_summary_csv,
                             append_depth_csv, finalize_eval_csv,
                             FRAME_COLUMNS)

# <<< set this to the module name of the file that defines ROSPublisher >>>
ROS_PUBLISHER_MODULE = "ros_publisher"

# The single source of truth for channel order. See the header block.
PSEUDO_IS_RGB = True

# 12 columns, identical across every model, so the CSVs concatenate for
# cross-model and distance-vs-error analysis. Do not reorder or extend
# without regenerating every existing run.
CSV_COLUMNS = ["frame", "stamp_ns", "t_infer_ms", "n_px", "scale",
               "AbsRel", "RMSE_m", "MAE_m", "delta1", "delta2",
               "pred_p50_m", "gt_p50_m"]+FRAME_COLUMNS


def to_model_rgb(pseudo):
    """The crop in RGB order, which is what every model in this benchmark
    expects. A no-op when render_pseudo already returns RGB."""
    return pseudo if PSEUDO_IS_RGB else cv2.cvtColor(pseudo,
                                                     cv2.COLOR_BGR2RGB)


def to_display_bgr(pseudo):
    """The crop in BGR order, for cv2.imshow / imwrite / the ROS bgr8
    publisher. A no-op when render_pseudo already returns BGR."""
    return cv2.cvtColor(pseudo, cv2.COLOR_RGB2BGR) if PSEUDO_IS_RGB else pseudo


# ==========================================================================
# Display panel
# ==========================================================================
def _label(img, text, scale=0.8, thickness=2):
    """Black title bar with white text across the top of an image.

    Bar height follows the font size, so larger text is never clipped.
    """
    bar_h = int(28 + scale * 18)
    o = img.copy()
    cv2.rectangle(o, (0, 0), (o.shape[1], bar_h), (0, 0, 0), -1)
    y = int(bar_h * 0.7)
    cv2.putText(o, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return o


def _colorize(depth_m, valid, lo, hi):
    """Turbo-colour a depth map on a fixed [lo, hi] metre scale (BGR out)."""
    norm = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = (0, 0, 0)
    return vis


def _blend(crop_bgr, depth_vis, valid, alpha=0.55):
    """Lay a colourised depth over the image, only where valid."""
    out = crop_bgr.copy()
    mixed = cv2.addWeighted(crop_bgr, 1 - alpha, depth_vis, alpha, 0)
    out[valid] = mixed[valid]
    return out


def make_panel(crop_bgr, pred_m, gt_m, valid, model_label, alpha=0.55):
    """Two rows, four views, identical styling for every model.

        [ (i)  Pseudo-Perspective RGB | (ii) Azure Kinect            ]
        [ (iii) <model>               | (iv) Kinect mask over RGB    ]

    crop_bgr must ALREADY be BGR -- pass to_display_bgr(pseudo).

    Ground truth and prediction are each coloured on their OWN
    5/95-percentile range, so a single stray pixel cannot flatten the
    contrast. They are NOT on a shared colour bar: read the metric offset
    from the CSV 'scale' column, never by comparing the two colourings.
    """
    if valid.any():
        lo, hi = np.percentile(gt_m[valid], [5,95])
        g_min, g_max = float(gt_m[valid].min()), float(gt_m[valid].max())
    else:
        lo, hi = 0.0, 1.0
        g_min = g_max = float("nan")

    pred_valid = np.isfinite(pred_m) & (pred_m > 0)
    if pred_valid.any():
        plo, phi = np.percentile(pred_m[pred_valid], [5,95])
        p_min, p_max = (float(pred_m[pred_valid].min()),
                        float(pred_m[pred_valid].max()))
    else:
        plo, phi = 0.0, 1.0
        p_min = p_max = float("nan")

    pred_vis = _colorize(pred_m, pred_valid, plo, phi)
    gt_vis = _colorize(gt_m, valid, lo, hi)
    gt_blend = _blend(crop_bgr, gt_vis, valid, alpha)

    # outline the valid mask so coverage is obvious at a glance
    edges = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_GRADIENT,
                             np.ones((3, 3), np.uint8)) > 0
    gt_blend[edges] = (255, 255, 255)

    top = np.hstack([
        _label(crop_bgr, " (i) Pseudo-Perspective RGB"),
        _label(gt_vis, f" (ii) Azure Kinect  {g_min:.2f}-{g_max:.2f} m"),
    ])
    bot = np.hstack([
        _label(pred_vis, f" (iii) {model_label}  {p_min:.2f}-{p_max:.2f} m"),
        _label(gt_blend, " (iv) Azure Kinect mask over RGB"),
    ])
    return np.vstack([top, bot])
# ---- paste into bench_core.py, right after make_panel() (line ~207) ----

def _fit_width(img, W, H):
    """Letterbox onto a W x H black canvas, aspect preserved.

    Rows of the equirect panel must be the same width to stack, but the
    Azure's native frame is nearly square and the prediction may be any
    size. Stretching either one to 2:1 would misrepresent the geometry --
    a square sensor frame drawn as a 2:1 strip looks like a wide-angle
    sensor. Black bars are honest; a stretch is not.
    """
    h, w = img.shape[:2]
    s = min(W / float(w), H / float(h))
    rw, rh = max(int(round(w * s)), 1), max(int(round(h * s)), 1)
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - rh) // 2, (W - rw) // 2
    out[y:y + rh, x:x + rw] = cv2.resize(img, (rw, rh),
                                         interpolation=cv2.INTER_AREA)
    return out


def _colorize_full(depth_m, lo, hi):
    """Turbo on a fixed [lo, hi] metre scale, no validity mask.

    Non-positive and non-finite pixels go black, matching _colorize, so a
    sensor's empty regions read the same way in every row.
    """
    d = np.asarray(depth_m, np.float32)
    ok = np.isfinite(d) & (d > 0)
    norm = np.clip((np.nan_to_num(d) - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~ok] = (0, 0, 0)
    return vis


def _range(a, mask=None):
    """(lo, hi) at 2/98 for colouring, plus (min, max) for the label."""
    v = np.asarray(a, np.float32)
    ok = np.isfinite(v) & (v > 0)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if not ok.any():
        return 0.0, 1.0, float("nan"), float("nan")
    s = v[ok]
    lo, hi = np.percentile(s, [2, 98])
    return float(lo), float(hi), float(s.min()), float(s.max())


def _common_label_scale(texts, width, base=0.8, thickness=2, pad=16,
                        floor=0.40):
    """One font size for every row: the largest that fits the LONGEST label.

    Shrinking only the row that overflows would leave the panel with two
    font sizes and make the smaller row look like a footnote rather than a
    peer. Below `floor` the text would be unreadable, so it is allowed to
    clip instead -- a clipped label is a visible fault; an illegible one
    silently is not.
    """
    s = base
    while s > floor:
        if all(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s,
                               thickness)[0][0] <= width - pad
               for t in texts):
            return s
        s -= 0.05
    return floor


def make_equi_panel(equi_bgr, pred_radial, depth_mm_raw, gt_m, valid,
                    model_label, outline=None, alpha=0.6):
    """Four rows, each one stage of the chain, nothing hidden.

        1. the panorama exactly as it came off the Theta
        2. the model's RADIAL depth at its native resolution, unmasked --
           before resize_pred_to_eval() and before any scale is fitted.
           Radial rather than a z-depth because distance from the optical
           centre cannot be negative, whereas z goes negative behind the
           image plane and is meaningless over half of a full 360.
        3. the Azure exactly as it came off the sensor, in its own
           geometry, letterboxed -- NOT warped, NOT aligned
        4. that same Azure warped onto the panorama: the alignment check.
           The patch must sit on the objects underneath it.

    Rows 1-2 share the sphere; row 3 is the sensor's native frame; row 4 is
    row 3 after the calibration chain. Read top to bottom, each row shows
    what one step did.

    Each row is coloured on its OWN 2/98 range, following make_panel(): a
    shared bar would let one stray pixel flatten every row at once. Read
    the metric offset from the CSV 'scale' column, never by comparing two
    colourings.
    """
    H, W = gt_m.shape
    base = cv2.resize(equi_bgr, (W, H), interpolation=cv2.INTER_AREA)

    rows = [(base, "1. Equirectangular RGB")]

    if pred_radial is not None:
        plo, phi, p_min, p_max = _range(pred_radial)
        rows.append((
            _fit_width(_colorize_full(pred_radial, plo, phi), W, H),
            f"2. {model_label} {p_min:.2f} - {p_max:.2f} m"))

    d_raw = depth_mm_raw.astype(np.float32) / 1000.0
    rlo, rhi, r_min, r_max = _range(d_raw)
    rows.append((
        _fit_width(_colorize_full(d_raw, rlo, rhi), W, H),
        f"3. Azure kinect {r_min:.2f}-{r_max:.2f} m"))

    glo, ghi, g_min, g_max = _range(gt_m, valid)
    blend = _blend(base, _colorize(gt_m, valid, glo, ghi), valid, alpha)
    edges = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_GRADIENT,
                             np.ones((3, 3), np.uint8)) > 0
    blend[edges] = (255, 255, 255)
    if outline is not None:
        blend[outline] = (0, 0, 255)
    rows.append((blend, "4. Azure Kinect mask over Equirectangular RGB")) 

    s = _common_label_scale([t for _, t in rows], W)
    return np.vstack([_label(img, t, scale=s) for img, t in rows])


# ==========================================================================
# Parameter counting
# ==========================================================================
def count_parameters(proc):
    """Best-effort parameter count for a benchmark processor.

    The processors store the underlying nn.Module under different attribute
    names, so probe the likely ones. Returns (n_params, source_attr) or
    (None, None).
    """
    import torch.nn as nn
    for attr in ("model", "net", "network", "depth_model", "_model"):
        m = getattr(proc, attr, None)
        if isinstance(m, nn.Module):
            return sum(p.numel() for p in m.parameters()), attr
    if isinstance(proc, nn.Module):
        return sum(p.numel() for p in proc.parameters()), "self"
    return None, None


def print_param_count(name, proc):
    n, src = count_parameters(proc)
    if n is None:
        print(f"[{name}] parameters: (could not locate nn.Module)")
    else:
        print(f"[{name}] parameters: {n:,} ({n/1e6:.1f} M)  [from .{src}]")


# ==========================================================================
# CLI arguments shared by every model
# ==========================================================================
def add_common_args(ap):
    ap.add_argument("--bag", required=True)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--dyaw", type=float, default=0.0,
                    help="manual depth nudge left/right, degrees "
                         "(use the value tuned in calibrate_view)")
    ap.add_argument("--dpitch", type=float, default=0.0,
                    help="manual depth nudge up/down, degrees "
                         "(use the value tuned in calibrate_view)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--save-display", action="store_true",
                    help="write the 4-panel figure to PNG per frame")
    ap.add_argument("--publish", action="store_true",
                    help="publish the predicted depth + RGB via ROSPublisher")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="where the CSV and panel PNGs are written; defaults "
                         "to the project's Output/ folder")
    ap.add_argument("--px-stability", action="store_true", default=True,
                    help="accumulate per-pixel temporal stability (J_px)")
    ap.add_argument("--no-px-stability", dest="px_stability",
                    action="store_false")
    ap.add_argument("--px-min-frac", type=float, default=0.95,
                    help="a pixel must be valid in this fraction of frames")
    ap.add_argument("--px-csv", default="px_stability.csv",
                    help="one summary row per run, APPENDED")
    ap.add_argument("--px-depth-csv", default="px_depth.csv",
                    help="one row per run/bin/edge-mode, APPENDED")
    ap.add_argument("--px-bins", type=float, nargs="+",
                    default=[0.3, 0.8, 1.35, 1.8, 2.3, 3.3],
                    help="Kinect-depth bin edges, metres. MUST be identical "
                         "across every run or the columns will not align")
    ap.add_argument("--eq-width", type=int, default=960)
    ap.add_argument("--eq-height", type=int, default=None)
    ap.add_argument("--splat", type=int, default=1)
    ap.add_argument("--lon-sign", type=float, default=1.0)
    ap.add_argument("--lat-sign", type=float, default=1.0)
    ap.add_argument("--lon-offset", type=float, default=0.0)
    ap.add_argument("--metrics-in", choices=["radial", "z"], default="radial")
    ap.add_argument("--equi-region", default="sphere",
                    choices=["sphere", "window"],
                    help="equirect regime only. 'sphere' scores over the "
                         "Kinect's full ~120 deg coverage. 'window' projects "
                         "the panoramic prediction into the SAME 60 deg "
                         "window the pinhole runs use and scores it with the "
                         "same compare() -- the R4.3 control, where only the "
                         "INPUT regime differs.")
    return ap


# ==========================================================================
# The one frame loop
# ==========================================================================
def run_eval(adapter, args):
    """Run one model over one bag and write the benchmark CSV.

    `adapter` supplies the five things that actually differ between models
    (see adapters.py): construction, the infer() call shape, the config tag,
    the panel label, and whether scoring is metric or relative.
    """
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU")
        device = "cpu"

    equi = getattr(adapter, "regime", "pinhole") == "equirect"
    window = equi and args.equi_region == "window"
    EQ = (args.eq_width, args.eq_height or args.eq_width // 2)
    if not equi:
        print(f"[VIRTUAL] {OUT_W}x{OUT_H} at fov_x={FOV_X} deg")
    elif window:
        print(f"[REGIME] EQUIRECT input, PINHOLE scoring: projected into "
              f"the same {OUT_W}x{OUT_H} / {FOV_X} deg window and scored by "
              f"the same compare(). Only the INPUT regime differs.")
        print(f"[REGIME] inside this window the panoramic prediction carries "
              f"~1/4 the angular sampling of a native crop. Report that.")
    else:
        print(f"[REGIME] EQUIRECT, scored on the sphere over the Kinect's "
              f"~120x120 deg coverage, {EQ[0]}x{EQ[1]}. NOT comparable to "
              f"the pinhole runs -- use --equi-region window for that.")
    if equi and args.publish:
        raise SystemExit("--publish is not supported in equirect mode.")
    print("  K and R_v come from compute_camera_parameters() per frame")
    print(f"  colour: render_pseudo returns "
          f"{'RGB' if PSEUDO_IS_RGB else 'BGR'}; models fed RGB, cv2 fed BGR")
    if args.dyaw or args.dpitch:
        print(f"  depth nudged by dyaw {args.dyaw:+g}  dpitch {args.dpitch:+g}")

    # ---- optional ROS publishing ----
    pub = None
    if args.publish:
        import importlib
        ros_mod = importlib.import_module(ROS_PUBLISHER_MODULE)
        pub = ros_mod.ROSPublisher(enable_ros=True)
        # Force the publisher's virtual camera to match what the model saw,
        # otherwise fx disagrees and the tracker scales depth wrong.
        pub.publish_camera_info(width=OUT_W, height=OUT_H, fov_deg=FOV_X)
        print(f"[PUBLISH] ROSPublisher up; CameraInfo forced to "
              f"{OUT_W}x{OUT_H} @ {FOV_X} deg")

    # ---- model ----
    proc = adapter.build(args, device)
    if proc is None or not proc.is_available():
        raise SystemExit(f"{adapter.display_name} failed to initialise "
                         f"(see the [{adapter.display_name}] messages above)")
    print_param_count(adapter.display_name, proc)
    desc = adapter.describe(args)
    if desc:
        print(f"[{adapter.display_name}] config: {desc}")

    metric_model = adapter.metric_model(args)
    print(f"[{adapter.display_name}] scored as "
          f"{'METRIC (raw, no rescale)' if metric_model else 'RELATIVE (scale fit)'}")

    # ---- output paths ----
    bag_tag = os.path.basename(os.path.normpath(args.bag))
    cfg = adapter.tag(args)
    region_tag = ("_win60" if (equi and args.equi_region == "window") else "")
    run_tag = f"{adapter.name}_{bag_tag}_{cfg}{region_tag}"

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"eval_{run_tag}.csv")

    # Panels are one PNG per frame, so a long run would bury the CSVs. Each
    # run gets its own subfolder: Output/panels/<model>_<bag>_<config>/
    panel_dir = os.path.join(out_dir, "panels", run_tag)
    
    if args.display or args.save_display:
        os.makedirs(panel_dir, exist_ok=True)
        if args.save_display:
            print(f"[PNG] {panel_dir}/")

            
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(CSV_COLUMNS)
    f.flush()
    print(f"[CSV] {csv_path}")

    # ---- per-pixel temporal stability (R4.2) ----
    _NO_PX = {"px_step_mm": float("nan"), "px_step_p95_mm": float("nan"),
              "px_step_n": 0}
    acc = None
    if args.px_stability:
        if not metric_model:
            print(f"[px] skipped: {adapter.display_name} is scored RELATIVE, "
                  f"so pred_m is refitted to the Kinect every frame. A "
                  f"per-pixel series built from that carries the sensor's "
                  f"noise, not the model's.")
        else:
            shape = (EQ[1], EQ[0]) if (equi and not window) else (OUT_H, OUT_W)
            acc = PixelStability(shape, min_frac=args.px_min_frac)
            print(f"[px] per-pixel stability on "
                  f"({OUT_H}x{OUT_W}, keep >= {args.px_min_frac:.0%} valid)")
    # The window's footprint depends only on its aim and the grid size,
    # never on the frame, so build it once on the first frame and reuse it.
    equi_outline_mask = None
    rows, n = [], 0
    skips = Counter()

    try:
        for stamp, equirect, depth_mm, k_intr, k_dist in stream_pairs(args.bag):
            n += 1
            if args.max_frames and n > args.max_frames:
                break
            if k_intr is None:
                # Some bags emit their first depth image one message BEFORE
                # the first CameraInfo, so the intrinsics are unknown when
                # the first pair completes. Skip that frame rather than
                # rejecting an otherwise valid bag over an ordering race.
                skips["CameraInfo not seen yet"] += 1
                continue

            if equi:
                model_in, K_np = cv2.cvtColor(equirect, cv2.COLOR_BGR2RGB), None
                # Display only. In window mode the scored arrays ARE the
                # 518x518 projection, so the RGB panel has to be that same
                # window -- which is exactly what render_pseudo returns.
                # Never fed to the model, and not rendered unless a panel is
                # actually being drawn, so the inference timing stays clean.
                if window and (args.display or args.save_display):
                    pseudo, _, _ = render_pseudo(equirect, args.yaw, args.pitch,
                                                 FOV_X, OUT_W, OUT_H,
                                                 device=device)
            else:
                pseudo, K_np, _ = render_pseudo(equirect, args.yaw, args.pitch,
                                                FOV_X, OUT_W, OUT_H,
                                                device=device)
                model_in = to_model_rgb(pseudo)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            pred = adapter.infer(proc, model_in, K_np, args)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t_start) * 1000

            if pred is None:
                skips["infer returned None"] += 1
                if skips["infer returned None"] <= 3:
                    print(f"[{n}] infer returned None -- see the "
                          f"[{adapter.display_name}] messages for the cause")
                continue
            if not np.isfinite(pred).any():
                skips["no finite pixels"] += 1
                continue

            # Sanitise non-finite pixels to 0 (the pipeline's "no reading";
            # metrics() drops p <= 0). inf is POSITIVE and would otherwise
            # slip past that guard, so this matters for MoGe in particular.
            pred = np.where(np.isfinite(pred), pred, 0.0).astype(np.float32)

            # render_pseudo has already aimed the crop, but compare() runs
            # with R_v = I and so knows nothing about it. Fold the aim out
            # here, and --dyaw/--dpitch then mean the same physical
            # calibration offset in every regime.
            try:
                if window:
                    res = compare_equi_pinhole(
                    pred, depth_mm, k_intr,
                    yaw_deg=args.yaw, pitch_deg=args.pitch,
                    out_wh=(OUT_W, OUT_H), device=device,
                    metric_model=metric_model,
                    distortion=k_dist if k_dist is not None
                    else DEPTH_DISTORTION,
                    # dyaw=args.dyaw, dpitch=args.dpitch,
                    # verbose=(len(rows) == 0))
                    dyaw=args.dyaw - args.yaw,
                    dpitch=args.dpitch - args.pitch,
                    verbose=(len(rows) == 0))
                elif equi:
                    res = compare_equi(
                    pred, depth_mm, k_intr, eq_wh=EQ,
                    metric_model=metric_model,
                    distortion=k_dist if k_dist is not None
                    else DEPTH_DISTORTION,
                    dyaw=args.dyaw, dpitch=args.dpitch, splat=args.splat,
                    lon_sign=args.lon_sign, lat_sign=args.lat_sign,
                    lon_offset_deg=args.lon_offset,
                    metrics_in=args.metrics_in)
                else:
                    res = compare(pred, depth_mm, k_intr, R_v=np.eye(3),
                              out_wh=(OUT_W, OUT_H), fov_x=FOV_X,
                              metric_model=metric_model,
                              distortion=k_dist if k_dist is not None
                              else DEPTH_DISTORTION,
                             dyaw=args.dyaw - args.yaw,
                              dpitch=args.dpitch - args.pitch)

            except ValueError as e:
                skips["compare rejected"] += 1
                if skips["compare rejected"] <= 3:
                    print(f"[{n}] compare rejected: {e}")
                continue

            m = res["metrics"]
            v = res["valid"]

            # frame_idx is the STREAM counter n, which keeps rising through
            # skipped frames. The accumulator only counts a frame-to-frame
            # step when prev_i == n-1, so a gap breaks the chain instead of
            # being billed as one interval of movement.
            st = (acc.update(res["pred_m"], gt_m=res["gt_m"], gt_valid=v,
                             frame_idx=n)
                  if acc is not None else _NO_PX)

            w.writerow([n, stamp, f"{t_ms:.2f}", m["n_px"],
                        f"{res['fit'][0]:.4f}",
                        f"{m['AbsRel']:.4f}", f"{m['RMSE_m']:.4f}",
                        f"{m['MAE_m']:.4f}", f"{m['delta1']:.4f}",
                        f"{m['delta2']:.4f}",
                        f"{np.median(res['pred_m'][v]):.4f}",
                        f"{np.median(res['gt_m'][v]):.4f}",
                        f"{st['px_step_mm']:.3f}",
                        f"{st['px_step_p95_mm']:.3f}", st["px_step_n"]])
            f.flush()
            rows.append(m)

            if pub is not None:
                # publish_synced wants bgr8 RGB + depth in METRES
                pub.publish_synced(to_display_bgr(pseudo), res["pred_m"])
                pub.spin_once(timeout_sec=0.0)

            if n <= 5 or n % 30 == 0:
                print(f"[{n}] {t_ms:6.1f} ms  AbsRel {m['AbsRel']:.4f}  "
                      f"d1 {m['delta1']:.3f}  scale {res['fit'][0]:.3f}  "
                      f"({m['n_px']} px)")

            if args.display or args.save_display:
                if equi and not window:
                    if equi_outline_mask is None:
                        try:
                            from equi_outline import equi_window_outline
                            equi_outline_mask = equi_window_outline(
                                lambda a: project_equi_to_pinhole(
                                    a, yaw_deg=args.yaw, pitch_deg=args.pitch,
                                    fov_x=FOV_X, out_wh=(OUT_W, OUT_H),
                                    device=device),
                                hw=res["gt_m"].shape, out_wh=(OUT_W, OUT_H),
                                fov_x=FOV_X, verbose=(n == 1))
                        except Exception as e:
                            print(f"[outline] skipped: {e}")
                            equi_outline_mask = False
                        if equi_outline_mask is None:
                            # The projection refused (not nearest-sampled).
                            # False, not None, or every frame retries it.
                            equi_outline_mask = False
                    panel = make_equi_panel(
                        equirect, pred, depth_mm, res["gt_m"], v,
                        adapter.panel_label,
                        outline=(None if equi_outline_mask is False
                                 else equi_outline_mask))
                elif equi:
                    panel = make_equi_window_panel(
                        pred, res,
                        project_fn=lambda a: project_equi_to_pinhole(
                            a, yaw_deg=args.yaw, pitch_deg=args.pitch,
                            fov_x=FOV_X, out_wh=(OUT_W, OUT_H),
                            device=device),
                        out_wh=(OUT_W, OUT_H), fov_x=FOV_X,
                        label=f"{adapter.panel_label}  frame {n}",
                        scale=0.6, verbose=(n == 1))
                else:
                    panel = make_panel(to_display_bgr(pseudo), res["pred_m"],
                                       res["gt_m"], v, adapter.panel_label)
                if args.display:
                    cv2.imshow(adapter.panel_label, panel)
                    if cv2.waitKey(1) == 27:
                        break
                if args.save_display:
                    cv2.imwrite(
                        os.path.join(panel_dir, f"panel_{n:04d}.png"), panel)

    except KeyboardInterrupt:
        print("\n[interrupted -- partial CSV kept]")
    finally:
        f.close()
        cv2.destroyAllWindows()
        if pub is not None:
            pub.shutdown()

    # ---- summary, including WHY frames were dropped ----
    print("\n" + "=" * 60)
    print(f"model={adapter.name}  config={cfg}  bag={bag_tag}")
    print(f"frames read: {n}   scored: {len(rows)}   "
          f"skipped: {sum(skips.values())}")
    for reason, cnt in skips.most_common():
        print(f"  skipped[{reason}] = {cnt}")

    if rows:
        for k in ("AbsRel", "RMSE_m", "MAE_m", "delta1", "delta2"):
            a = np.array([r[k] for r in rows])
            print(f"  {k:8s} median {np.median(a):.4f}   "
                  f"mean {a.mean():.4f}   std {a.std():.4f}")
        print(f"\nsaved to {csv_path}  ({len(rows)} rows)")
    else:
        print(f"\nNO ROWS WRITTEN. {csv_path} contains only the header.")
        print("Look at the skip reasons above:")
        print("  infer returned None -> the model produced nothing; check "
              "its load messages and whether it requires intrinsics.")
        print("  compare rejected    -> the warp found no Kinect/crop "
              "overlap; check --pitch/--dyaw/--dpitch and the bag topics.")

    if acc is not None and acc.n_frames:
        acc.print_report(label=f"{adapter.name}-{cfg} [{bag_tag}]")
        acc.print_by_depth(bins=tuple(args.px_bins))
        finalize_eval_csv(csv_path, acc)
        _p = lambda q: q if os.path.isabs(q) else os.path.join(out_dir, q)
        append_summary_csv(_p(args.px_csv), acc, model=adapter.name,
                           variant=cfg, bag=bag_tag)
        append_depth_csv(_p(args.px_depth_csv), acc, model=adapter.name,
                         variant=cfg, bag=bag_tag, bins=tuple(args.px_bins))

    return rows