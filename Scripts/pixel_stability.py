#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-pixel temporal stability, accumulated online during an eval run.

Lives beside compare() in theta_kinect_compare.py, or as its own module in
Tools/ -- it depends on nothing in the comparison library.

WHAT THIS ANSWERS
    R4.2  CV of the frame median hides per-pixel instability. The frame
          median can be perfectly steady while individual pixels swing,
          because the median lands on whichever pixel happens not to move.
          J_px is the median over PIXELS of each pixel's own temporal CV.

DEPTH BINNING -- and the trap in it
    Does instability grow with distance? Measured on these bags, the answer
    is reachable but not in the obvious way. The scene is dominated by one
    plane facing the camera, so EQUAL-WIDTH bins put 88-99% of pixels in a
    single bin and the trend collapses to one point. That is a binning
    problem, not a data problem: the off-plane populations are large.

        bin (m)     0.51m        1m      1.5m    POOLED
        0.5-1.0     1,125         0         0     1,125
        1.0-1.5   233,034    11,651         0   244,685
        1.5-2.0         0   203,221     2,839   206,060
        2.0-2.5         0         0   131,903   131,903
        2.5-3.0         0     9,420     4,051    13,471

    The 1m bag ALONE spans 1.0-3.2 m with 11.6k / 203k / 9.4k pixels in
    three populations -- a within-run test needing no pooling. Pooled across
    bags, five bins cover 0.55-3.20 m.

    THE TRAP: those off-plane pixels are not a random sample of the scene.
    They are the far wall past the edge of the board, the object silhouette,
    and oblique surfaces. A pixel on a depth discontinuity flips between
    foreground and background between frames, so its CV is enormous for a
    reason that has nothing to do with distance. Bin without excluding them
    and you will measure "edges are unstable" and report it as "far things
    are unstable".

    exclude_edges=True (the default) drops any pixel whose local Kinect
    depth spans more than edge_rel of its depth within an edge_win window,
    which also drops pixels adjacent to no-data. Run by_depth BOTH ways: if
    the trend survives the edge mask it is about distance; if it vanishes,
    it was about geometry, and that is the finding.

WHY ONLINE
    Stacking every frame to compute this afterwards costs
    518*518*757*4 bytes = 0.81 GB per run, 41 GB across 51 runs. Welford's
    method keeps three frame-sized arrays and updates them as each frame
    arrives, so memory is flat in the number of frames. It also avoids the
    catastrophic cancellation of the sum-x / sum-x-squared shortcut: depths
    near 1 m varying by ~1% put the variance in the low bits of a float32
    subtraction of two nearly equal large numbers.

WHAT IT REPORTS, per pixel then medianed over pixels
    J_px     100 * std(d) / mean(d), percent -- total spread, order-blind
    jitter   mean |d[k] - d[k-1]|, mm        -- frame-to-frame movement
    drift    d[last] - d[first], mm, SIGNED  -- net travel over the run

    The last two exist because CV cannot tell a pixel that bounces from one
    that slides: the same values in a different order give the identical CV.
    Only differencing consecutive frames sees the ordering.

MASKS -- read this before quoting a number
    A model predicts densely over the whole 518x518 crop; the warped Kinect
    covers only part of it, and how much varies a lot by bag (87.3%, 85.6%,
    52.1% on these three). So there are two populations, NOT interchangeable:

      full     every pixel the model predicted. This is what ICP consumes,
               so it is the honest number for the tracking claim.
      overlap  only pixels the Kinect also saw. This is the region AbsRel
               is computed over.

    Put the OVERLAP number next to AbsRel in any table, so the two columns
    describe the same pixels. Quoting a full-frame J_px beside an
    overlap-region AbsRel repeats the mistake that put pinhole and equirect
    runs on one axis in Figure 8. Carry n_px_overlap alongside, because it
    changes by a factor of 1.7 between bags.
"""

import csv
import os

import cv2
import numpy as np


class PixelStability:
    """Welford accumulator over the virtual-camera grid.

    Feed it one frame at a time, in capture order, with the caller's STREAM
    index. Frames the caller skips must keep advancing that index, so a step
    across a gap is not counted as one interval of movement.
    """

    def __init__(self, shape, min_frac=0.95, eps=1e-9):
        h, w = shape
        self.shape = (h, w)
        self.min_frac = float(min_frac)
        self.eps = float(eps)
        self.n_frames = 0

        # --- Welford state over the PREDICTION, per pixel ---
        self.n = np.zeros((h, w), np.int32)          # valid frames seen
        self.mean = np.zeros((h, w), np.float64)
        self.M2 = np.zeros((h, w), np.float64)

        # --- ordering-aware state ---
        self.prev = np.zeros((h, w), np.float64)     # last valid value
        self.prev_i = np.full((h, w), -2, np.int32)  # its frame index
        self.sum_absd = np.zeros((h, w), np.float64)
        self.n_steps = np.zeros((h, w), np.int32)
        self.first = np.full((h, w), np.nan)
        self.last = np.full((h, w), np.nan)

        # --- Kinect: the overlap mask, and the run's measured depth ---
        self.n_gt = np.zeros((h, w), np.int32)
        self.gt_sum = np.zeros((h, w), np.float64)

    def update(self, pred_m, gt_m=None, gt_valid=None, frame_idx=None):
        """One frame. pred_m and gt_m are metres on the virtual grid.

        pred_m must be the RAW metric prediction -- compare()'s pred_m with
        metric_model=True. A relative model's pred_m has been refitted to
        the Kinect every frame, which injects the sensor's own noise into
        this series and makes J_px meaningless for it. Skip those runs
        rather than writing a contaminated number.
        """
        if pred_m.shape != self.shape:
            raise ValueError(f"expected {self.shape}, got {pred_m.shape}")
        i = self.n_frames if frame_idx is None else int(frame_idx)
        self.n_frames += 1

        frame = {"px_step_mm": float("nan"),
                 "px_step_p95_mm": float("nan"),
                 "px_step_n": 0}

        p = np.asarray(pred_m, dtype=np.float64)
        v = np.isfinite(p) & (p > 0)
        if v.any():
            # ---- Welford, in place, only on valid pixels ----
            self.n[v] += 1
            delta = p[v] - self.mean[v]
            self.mean[v] += delta / self.n[v]
            # NOTE: uses mean AFTER the update above. Swapping these two
            # lines silently computes the wrong variance.
            self.M2[v] += delta * (p[v] - self.mean[v])

            # ---- consecutive-frame step, only where the PREVIOUS frame
            # was also valid here AND was the immediately preceding one ----
            # Computed BEFORE self.prev is overwritten below -- moving
            # this after would difference the frame against itself.
            step_ok = v & (self.prev_i == i - 1)
            if step_ok.any():
                d_mm = np.abs(p[step_ok] - self.prev[step_ok]) * 1000.0
                self.sum_absd[step_ok] += d_mm / 1000.0
                self.n_steps[step_ok] += 1
                # per-FRAME summary of per-PIXEL movement. The median is the
                # typical pixel; p95 is the tail, because a model can look
                # steady on the median while a few percent of pixels thrash.
                frame = {"px_step_mm": float(np.median(d_mm)),
                         "px_step_p95_mm": float(np.percentile(d_mm, 95)),
                         "px_step_n": int(step_ok.sum())}

            newly = v & ~np.isfinite(self.first)
            self.first[newly] = p[newly]
            self.last[v] = p[v]
            self.prev[v] = p[v]
            self.prev_i[v] = i

        # ---- Kinect: overlap mask + measured depth ----
        if gt_m is not None and gt_valid is not None:
            g = gt_valid & np.isfinite(gt_m) & (gt_m > 0)
            if g.any():
                self.n_gt[g] += 1
                self.gt_sum[g] += gt_m[g]

        return frame

    # ------------------------------------------------------------------
    def _need(self):
        return max(2, int(np.ceil(self.min_frac * self.n_frames)))

    def maps(self):
        """Per-pixel maps: cv_pct, jitter_mm, drift_mm, gt_mean_m, keep."""
        keep = (self.n >= self._need()) & (self.n >= 2)

        var = np.full(self.shape, np.nan)
        np.divide(self.M2, np.maximum(self.n - 1, 1), out=var, where=keep)
        var[~keep] = np.nan

        cv = 100.0 * np.sqrt(var) / np.maximum(self.mean, self.eps)
        cv[~keep] = np.nan

        jit = np.full(self.shape, np.nan)
        has = keep & (self.n_steps > 0)
        jit[has] = 1000.0 * self.sum_absd[has] / self.n_steps[has]

        dr = np.full(self.shape, np.nan)
        dr[keep] = 1000.0 * (self.last[keep] - self.first[keep])

        gtm = np.full(self.shape, np.nan)
        hg = self.n_gt > 0
        gtm[hg] = self.gt_sum[hg] / self.n_gt[hg]

        return {"cv_pct": cv, "jitter_mm": jit, "drift_mm": dr,
                "gt_mean_m": gtm, "keep": keep}

    def summary(self):
        """J_px and friends, over the full frame and the Kinect overlap."""
        m = self.maps()
        keep = m["keep"]
        gt_seen = self.n_gt >= self._need()
        overlap = keep & gt_seen

        out = {"n_frames": self.n_frames,
               "n_px_total": int(keep.size),
               "n_px_full": int(keep.sum()),
               "n_px_overlap": int(overlap.sum())}

        # The run's MEASURED distance, for the x-axis of the J_px-vs-distance
        # figure. Use this, not the bag name: the bags are labelled 0.51/1/1.5
        # m but the Kinect measures the scene at 1.26/1.74/2.23 m, a constant
        # +0.74 m offset.
        out["gt_depth_m"] = (float(np.nanmedian(m["gt_mean_m"][overlap]))
                             if overlap.sum() else float("nan"))

        for tag, sel in (("full", keep), ("overlap", overlap)):
            if sel.sum() < 100:
                for k in ("J_px", "jitter", "drift"):
                    out[f"{k}_{tag}"] = float("nan")
                continue
            out[f"J_px_{tag}"] = float(np.nanmedian(m["cv_pct"][sel]))
            out[f"jitter_{tag}"] = float(np.nanmedian(m["jitter_mm"][sel]))
            out[f"drift_{tag}"] = float(np.nanmedian(m["drift_mm"][sel]))
        return out

    def print_report(self, label=""):
        s = self.summary()
        print(f"\n[pixel-stability] {label}".rstrip())
        print(f"  frames {s['n_frames']},  kept {s['n_px_full']} of "
              f"{s['n_px_total']} px full, {s['n_px_overlap']} px in the "
              f"Kinect overlap  (scene at {s['gt_depth_m']:.3f} m)")
        for tag in ("full", "overlap"):
            print(f"  {tag:<8} J_px {s[f'J_px_{tag}']:7.3f}%   "
                  f"jitter {s[f'jitter_{tag}']:7.3f} mm   "
                  f"drift {s[f'drift_{tag}']:+8.3f} mm")
        return s


    # ------------------------------------------------------------------
    def flat_mask(self, edge_rel=0.03, edge_win=5):
        """True where the Kinect depth is locally FLAT.

        A morphological span (dilate - erode) over the mean Kinect depth:
        wherever the neighbourhood spans more than edge_rel of the depth,
        the pixel sits on or beside a depth discontinuity. Invalid pixels
        are filled with 0 before the morphology, so a pixel next to a
        no-data hole shows a huge span and is dropped too -- which is what
        we want, since those are exactly the unreliable ones.
        """
        m = self.maps()
        gt = m["gt_mean_m"]
        valid = np.isfinite(gt)
        filled = np.where(valid, gt, 0.0).astype(np.float32)
        k = np.ones((int(edge_win), int(edge_win)), np.uint8)
        span = cv2.dilate(filled, k) - cv2.erode(filled, k)
        return valid & (span <= edge_rel * np.maximum(filled, 1e-6))

    def by_depth(self, bins=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), min_px=500,
                 exclude_edges=True, edge_rel=0.03, edge_win=5):
        """Instability per Kinect-depth bin.

        Binned by the KINECT's mean depth, never the model's: an unstable
        pixel is also a mis-estimated one, so binning by the prediction
        would sort pixels into the wrong bin BECAUSE they are unstable and
        manufacture a trend from nothing.

        sigma_mm and cv_pct are both returned on purpose. sigma almost
        always rises with depth for ordinary geometric reasons, so a rise
        there alone proves nothing; cv divides it out. If cv still rises,
        the model really is worse far away. If cv is flat, instability
        simply scales with depth -- itself a clean result.
        """
        m = self.maps()
        gt = m["gt_mean_m"]
        sel0 = m["keep"] & np.isfinite(gt)
        if exclude_edges:
            sel0 = sel0 & self.flat_mask(edge_rel, edge_win)

        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = sel0 & (gt >= lo) & (gt < hi)
            n = int(sel.sum())
            row = {"lo_m": lo, "hi_m": hi, "n_px": n, "thin": n < min_px}
            if n == 0:
                row.update(cv_pct=float("nan"), sigma_mm=float("nan"),
                           jitter_mm=float("nan"), drift_mm=float("nan"))
            else:
                sig = np.sqrt(self.M2[sel] / np.maximum(self.n[sel] - 1, 1))
                row.update(
                    cv_pct=float(np.nanmedian(m["cv_pct"][sel])),
                    sigma_mm=float(1000.0 * np.median(sig)),
                    jitter_mm=float(np.nanmedian(m["jitter_mm"][sel])),
                    drift_mm=float(np.nanmedian(m["drift_mm"][sel])))
            rows.append(row)
        return rows

    def print_by_depth(self, bins=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                       min_px=500, **kw):
        """Both ways, so the edge confound is visible rather than assumed."""
        for excl in (False, True):
            rows = self.by_depth(bins, min_px, exclude_edges=excl, **kw)
            tag = "edges EXCLUDED" if excl else "all pixels"
            print(f"  -- {tag} --")
            print(f"  {'depth bin':>12} {'px':>9} {'CV %':>8} "
                  f"{'sigma mm':>9} {'jitter mm':>10} {'drift mm':>9}")
            for r in rows:
                if r["n_px"] == 0:
                    continue
                flag = "  (thin)" if r["thin"] else ""
                print(f"  {r['lo_m']:5.1f}-{r['hi_m']:<5.1f} {r['n_px']:>9,} "
                      f"{r['cv_pct']:>8.3f} {r['sigma_mm']:>9.2f} "
                      f"{r['jitter_mm']:>10.3f} {r['drift_mm']:>+9.2f}{flag}")


# ----------------------------------------------------------------------
SUMMARY_COLUMNS = ["model", "variant", "bag", "gt_depth_m", "n_frames",
                   "n_px_full", "n_px_overlap",
                   "J_px_full", "jitter_full", "drift_full",
                   "J_px_overlap", "jitter_overlap", "drift_overlap"]


DEPTH_COLUMNS = ["model", "variant", "bag", "edges_excluded",
                 "lo_m", "hi_m", "n_px",
                 "cv_pct", "sigma_mm", "jitter_mm", "drift_mm"]


def append_depth_csv(path, acc, model, variant, bag,
                     bins=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), min_px=500,
                     **kw):
    """Long format: one row per (run, bin, edge-mode). Both modes are always
    written, because the comparison between them IS the result -- a trend
    that survives the edge mask is about distance, one that vanishes was
    about geometry. Long format also plots directly without reshaping."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEPTH_COLUMNS)
        if new:
            w.writeheader()
        for excl in (False, True):
            for r in acc.by_depth(bins, min_px, exclude_edges=excl, **kw):
                if r["n_px"] == 0:
                    continue
                w.writerow({"model": model, "variant": variant, "bag": bag,
                            "edges_excluded": int(excl),
                            "lo_m": r["lo_m"], "hi_m": r["hi_m"],
                            "n_px": r["n_px"], "cv_pct": r["cv_pct"],
                            "sigma_mm": r["sigma_mm"],
                            "jitter_mm": r["jitter_mm"],
                            "drift_mm": r["drift_mm"]})
    print(f"[pixel-stability] appended depth bins for "
          f"{model}-{variant} [{bag}] to {path}")


def append_summary_csv(path, acc, model, variant, bag):
    """One row per run, appended. Across all 51 runs this file IS the
    J_px figure: plot J_px_overlap against gt_depth_m, one line per model."""
    s = acc.summary()
    rec = {"model": model, "variant": variant, "bag": bag,
           "gt_depth_m": s["gt_depth_m"], "n_frames": s["n_frames"],
           "n_px_full": s["n_px_full"], "n_px_overlap": s["n_px_overlap"]}
    for tag in ("full", "overlap"):
        for k in ("J_px", "jitter", "drift"):
            rec[f"{k}_{tag}"] = s[f"{k}_{tag}"]

    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(rec)
    print(f"[pixel-stability] appended {model}-{variant} [{bag}] to {path}")
    return rec


# ----------------------------------------------------------------------
# Per-frame columns streamed during the run, and run-level columns appended
# to every row afterwards.
FRAME_COLUMNS = ["px_step_mm", "px_step_p95_mm", "px_step_n"]
RUN_COLUMNS = ["J_px_full", "J_px_overlap", "jitter_overlap",
               "drift_overlap", "n_px_overlap", "gt_depth_m"]


def finalize_eval_csv(path, acc):
    """Append the run-level columns to every row of an already-written
    per-frame CSV.

    Done as a REWRITE at the end rather than buffering rows in memory,
    because run_eval flushes after every row on purpose: an interrupted run
    must leave a readable partial CSV. If the run dies before this point,
    the file is still valid -- it simply lacks the run-level columns.

    The values repeat identically down every row. That is redundant, and it
    is the price of keeping one self-contained file per run; the same
    numbers also land once each in px_stability.csv.
    """
    if not os.path.exists(path) or acc.n_frames == 0:
        return
    s = acc.summary()
    extra = {k: s[k] for k in RUN_COLUMNS}

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    if RUN_COLUMNS[0] in rows[0]:
        return                      # already finalized; do not double-append

    cols = list(rows[0].keys()) + RUN_COLUMNS
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r.update(extra)
            w.writerow(r)
    os.replace(tmp, path)           # atomic: never a half-written CSV
    print(f"[pixel-stability] appended {len(RUN_COLUMNS)} run-level columns "
          f"to {os.path.basename(path)}")