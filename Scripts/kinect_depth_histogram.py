#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much depth range does the Kinect actually cover in the crop?

Run this BEFORE building the depth-binned instability figure. No model is
loaded and no inference happens -- it only reads the bag and warps the
Kinect onto the virtual camera.

The point: the depth-binned analysis only says something if the pixels
spread across several bins. If 90% of them land in one bin, the honest
result is a sentence saying the setup does not span enough depth to test
whether instability grows with distance -- a respectable finding, and much
better than a five-bin figure where three bins hold 400 pixels.

Bins on the per-pixel MEAN Kinect depth over the run, exactly as
PixelStability.by_depth() does, so the counts here are the counts you will
get there.

Usage (from anywhere -- PROJECT_ROOT is found from this file's location):
    python3 Scripts/kinect_depth_histogram.py --bag Data/1m
    python3 Scripts/kinect_depth_histogram.py --bag Data/0.51m \
        --dyaw -2.1 --dpitch -16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/opt/ros/humble/lib/python3.10/dist-packages')

# ==========================================================================
# Project layout -- the same block bench_core.py uses, with one change.
#
# bench_core matches the subfolder names exactly:
#     for _sub in ("Processors", "Tools", "Scripts"):
# which is case-SENSITIVE on Linux, so a folder actually named "tool" or
# "tools" is silently not added and the equilib import dies with
# ModuleNotFoundError several files later. Here the match is case- and
# plural-insensitive, and the chosen folders are printed under --verbose-path
# so a mismatch is visible instead of fatal.
#
# If this prints a name that differs from what bench_core.py looks for,
# FIX IT IN bench_core.py TOO -- it has the same failure waiting.
# ==========================================================================
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (_THIS_DIR.parent
                if _THIS_DIR.name.lower() in {"scripts", "tools", "tool"}
                else _THIS_DIR)

_WANT = {"processors", "processor", "tools", "tool", "scripts", "script"}
_ADDED = []
if PROJECT_ROOT.is_dir():
    for _p in sorted(PROJECT_ROOT.iterdir()):
        if _p.is_dir() and _p.name.lower() in _WANT:
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            _ADDED.append(_p.name)

import numpy as np

try:
    from theta_kinect_compare import (stream_pairs, kinect_depth_to_virtual,
                                      DEPTH_DISTORTION, OUT_W, OUT_H, FOV_X)
except ModuleNotFoundError as e:
    print(f"import failed: {e}")
    print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  folders added to sys.path: {_ADDED or '(none)'}")
    if PROJECT_ROOT.is_dir():
        print(f"  folders present: "
              f"{[p.name for p in sorted(PROJECT_ROOT.iterdir()) if p.is_dir()]}")
    print("  -> the module above lives in one of those folders; if its "
          "folder is not in the added list, add its name to _WANT.")
    raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--max-frames", type=int, default=200,
                    help="the scene is static, so a couple of hundred "
                         "frames estimate the per-pixel mean perfectly well")
    ap.add_argument("--dyaw", type=float, default=0.0,
                    help="same nudge you pass to the eval for this bag")
    ap.add_argument("--dpitch", type=float, default=0.0)
    ap.add_argument("--bins", type=float, nargs="+",
                    default=[0.3, 0.8, 1.2, 1.8, 2.3, 3],
                    help="must match --px-bins in the eval runs")
    ap.add_argument("--min-frac", type=float, default=0.95,
                    help="a pixel counts only if the Kinect saw it in this "
                         "fraction of frames; must match --px-min-frac")
    ap.add_argument("--verbose-path", action="store_true",
                    help="print PROJECT_ROOT and the sys.path additions")
    args = ap.parse_args()

    if args.verbose_path:
        print(f"[path] PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"[path] added        = {_ADDED}")

    print(f"[VIRTUAL] {OUT_W}x{OUT_H} at fov_x={FOV_X} deg")
    if args.dyaw or args.dpitch:
        print(f"  depth nudged by dyaw {args.dyaw:+g}  dpitch {args.dpitch:+g}")

    n_gt = np.zeros((OUT_H, OUT_W), np.int32)
    gt_sum = np.zeros((OUT_H, OUT_W), np.float64)
    n = 0

    for stamp, equirect, depth_mm, k_intr, k_dist in stream_pairs(args.bag):
        n += 1
        if n > args.max_frames:
            break
        if k_intr is None:
            continue

        # R_v = identity, matching bench_core: render_pseudo already aims
        # the crop, so the warp must not apply the rotation a second time.
        gt, valid = kinect_depth_to_virtual(
            depth_mm, k_intr, np.eye(3), (OUT_W, OUT_H), FOV_X,
            distortion=k_dist if k_dist is not None else DEPTH_DISTORTION,
            dyaw=args.dyaw, dpitch=args.dpitch)

        g = valid & np.isfinite(gt) & (gt > 0)
        if g.any():
            n_gt[g] += 1
            gt_sum[g] += gt[g]

    used = min(n, args.max_frames)
    if n_gt.max() == 0:
        raise SystemExit("the Kinect never landed in the crop -- check "
                         "--dyaw/--dpitch and the bag topics")

    seen = max(2, int(np.ceil(args.min_frac * used)))
    keep = n_gt >= seen
    depth = np.full((OUT_H, OUT_W), np.nan)
    depth[keep] = gt_sum[keep] / n_gt[keep]

    total_px = keep.size
    kept = int(keep.sum())
    print(f"\nframes used {used}")
    print(f"pixels the Kinect saw in >= {args.min_frac:.0%} of frames: "
          f"{kept} of {total_px}  ({100.0*kept/total_px:.1f}% of the crop)")
    if kept == 0:
        raise SystemExit("no pixel was seen often enough; lower --min-frac "
                         "or check the alignment nudge")

    d = depth[keep]
    print(f"depth range  min {d.min():.3f}  p5 {np.percentile(d,5):.3f}  "
          f"median {np.median(d):.3f}  p95 {np.percentile(d,95):.3f}  "
          f"max {d.max():.3f}  m")

    print(f"\n{'bin (m)':>14} {'pixels':>9} {'share':>8}")
    bins = args.bins
    counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        c = int(((d >= lo) & (d < hi)).sum())
        counts.append(c)
        bar = "#" * int(round(40 * c / max(kept, 1)))
        print(f"  {lo:5.2f}-{hi:<5.2f} {c:>9} {100.0*c/max(kept,1):>7.1f}%  {bar}")
    out = kept - sum(counts)
    if out:
        print(f"  {'outside':>12} {out:>9} {100.0*out/max(kept,1):>7.1f}%"
              f"   (widen --bins to include these)")

    # Deciles of the actual distribution. Equal-population bin edges for
    # THIS bag -- print them for all three bags, then choose ONE edge set
    # that is reasonable for all of them, since --px-bins must be identical
    # across every run or the summary CSV columns will not line up.
    qs = np.percentile(d, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print("\ndeciles of the depth distribution (m):")
    print("  " + "  ".join(f"{q:.2f}" for q in qs))

    # The verdict. Counting bins is not enough: a bin set can clear a
    # minimum-count test while one bin still holds nearly all the pixels,
    # which is a single point dressed up as a trend. Check the SHARE too,
    # and check for an empty interior bin -- an indoor scene is typically a
    # near surface and a far wall with nothing between, and a line drawn
    # across that gap interpolates through empty space.
    counts = np.array(counts)
    usable = int((counts >= 500).sum())
    share = counts / max(kept, 1)
    top = float(share.max()) if counts.size else 0.0
    interior_gap = any(counts[i] == 0 and counts[:i].sum() and counts[i+1:].sum()
                       for i in range(len(counts)))

    print(f"\nusable bins (>=500 px): {usable} of {len(counts)}   "
          f"largest bin holds {100*top:.1f}% of kept pixels")
    if usable < 2:
        print("VERDICT: not enough spread. Say so in one sentence and drop "
              "the figure.")
    elif top > 0.60:
        print(f"VERDICT: the bins are too COARSE for this scene -- "
              f"{100*top:.0f}% of pixels land in one bin, so a trend across "
              f"them would be one point with decoration. Re-run with finer "
              f"edges through the dense region, using the deciles above.")
    elif usable == 2:
        print("VERDICT: 2 usable bins. A trend across two points is weak -- "
              "report the two numbers in the text, not as a figure.")
    else:
        print("VERDICT: enough spread to plot a trend.")
    if interior_gap:
        print("NOTE: an interior bin is empty. The scene has a depth gap "
              "(near surface, then back wall). Report the groups separately; "
              "do not draw a connecting line across the gap.")


if __name__ == "__main__":
    main()