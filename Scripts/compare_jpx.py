#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the instability coherent or per-pixel? One table over every run.

Reads the finalized eval_*.csv files -- the ones finalize_eval_csv() has
appended the run-level columns to -- and reports, per run:

    J_f     100 * std_t(pred_p50_m) / mean_t(pred_p50_m)
            the frame-median CV already in the paper, recomputed here from
            the per-frame column so both numbers come from one file
    J_px    median over pixels of each pixel's own temporal CV
    rho     J_px / J_f

WHAT RHO MEANS
    Every pixel wobbling independently -> the spatial median averages
    200k of them and barely moves -> J_f collapses and rho runs into the
    hundreds (simulated: 341x).
    The whole depth map breathing in and out together -> the median moves
    exactly as much as any single pixel -> rho = 1.00.
    So rho is a direct readout of WHICH failure mode a model has, and it
    decides whether a global scalar correction could fix it or whether
    spatial filtering is needed.

    rho < 1 is possible and is not an error: the per-pixel median can sit
    below the frame-median CV when the coherent component dominates and
    the typical pixel is slightly steadier than the spatial median.

NO JOIN
    An earlier version matched px_stability.csv against the eval files by
    reconstructing eval_<model>_<bag>_<variant>.csv from its columns. That
    is fragile -- one naming change and rows silently vanish. The eval CSVs
    already carry both numbers, so this reads one file per run.

Usage:
    python3 compare_jpx.py --dir Output
    python3 compare_jpx.py --dir Output --out coherence.csv --min-frames 100
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

NEEDED = ("pred_p50_m", "J_px_full", "J_px_overlap")


def parse_name(path):
    base = os.path.basename(path)
    m = re.match(r"^eval_([a-z0-9-]+)_([0-9]*\.?[0-9]+m)_(.+)\.csv$", base, re.I)
    if not m:
        return None
    return {"model": m.group(1), "bag": m.group(2), "variant": m.group(3)}


def read_run(path):
    meta = parse_name(path)
    if meta is None:
        return None, "unparsable filename"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, "empty"
    missing = [c for c in NEEDED if c not in rows[0]]
    if missing:
        return None, f"not finalized (missing {missing[0]})"

    p50 = np.array([float(r["pred_p50_m"]) for r in rows
                    if r["pred_p50_m"] not in ("", "nan")], dtype=float)
    p50 = p50[np.isfinite(p50) & (p50 > 0)]
    if p50.size < 3:
        return None, "fewer than 3 usable pred_p50_m values"

    g = lambda k: (float(rows[0][k])
                   if rows[0].get(k) not in (None, "", "nan") else np.nan)
    meta.update(
        n=int(p50.size),
        J_f=100.0 * float(p50.std(ddof=1)) / float(p50.mean()),
        J_px_full=g("J_px_full"),
        J_px=g("J_px_overlap"),
        jitter=g("jitter_overlap"),
        drift=g("drift_overlap"),
        n_px=g("n_px_overlap"),
        depth=g("gt_depth_m"),
    )
    meta["rho"] = meta["J_px"] / meta["J_f"] if meta["J_f"] > 0 else np.nan
    return meta, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Output")
    ap.add_argument("--bag", nargs="*", default=None)
    ap.add_argument("--min-frames", type=int, default=50,
                    help="skip runs shorter than this; a CV from a handful "
                         "of frames is noise, and rho is a ratio of two")
    ap.add_argument("--out", default=None, help="also write a CSV")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "eval_*.csv")))
    if not paths:
        raise SystemExit(f"no eval_*.csv in {args.dir}")

    runs, skipped = [], []
    for p in paths:
        r, why = read_run(p)
        if r is None:
            skipped.append((os.path.basename(p), why))
            continue
        if args.bag and r["bag"] not in set(args.bag):
            continue
        if r["n"] < args.min_frames:
            skipped.append((os.path.basename(p), f"only {r['n']} frames"))
            continue
        runs.append(r)

    if skipped:
        print(f"[skipped {len(skipped)}]")
        for name, why in skipped:
            print(f"   {name:<58} {why}")
    if not runs:
        raise SystemExit("\nnothing usable. Runs must be finalized by "
                         "finalize_eval_csv() -- check bench_core edit 6.")

    runs.sort(key=lambda r: (-r["rho"] if np.isfinite(r["rho"]) else -1e9))
    hdr = (f"\n{'model-variant':<34}{'bag':>7}{'n':>6}{'J_f %':>8}"
           f"{'J_px %':>8}{'rho':>7}{'jitter':>9}{'px':>9}")
    print(hdr); print("-" * len(hdr.strip("\n")))
    for r in runs:
        print(f"{r['model']+'-'+r['variant']:<34}{r['bag']:>7}{r['n']:>6}"
              f"{r['J_f']:>8.3f}{r['J_px']:>8.3f}{r['rho']:>7.2f}"
              f"{r['jitter']:>9.2f}{int(r['n_px']):>9,}")

    rho = np.array([r["rho"] for r in runs])
    rho = rho[np.isfinite(rho)]
    print(f"\nrho over {len(rho)} runs:  median {np.median(rho):.2f}   "
          f"range {rho.min():.2f} - {rho.max():.2f}")

    # The verdict, stated once rather than left to the reader's eye.
    if rho.max() < 2.0:
        print("COHERENT across every run: the instability is a global "
              "fluctuation, not per-pixel flicker. The frame-median CV in "
              "the submission was not understating it, and a single scalar "
              "correction per frame could remove most of it.")
    else:
        odd = [r for r in runs if np.isfinite(r["rho"]) and r["rho"] >= 2.0]
        print(f"{len(odd)} run(s) with rho >= 2 -- these have a real "
              f"per-pixel component the frame median hides:")
        for r in odd:
            print(f"   {r['model']}-{r['variant']} [{r['bag']}]  "
                  f"rho {r['rho']:.2f}")
        print("Report those separately; one number cannot cover both groups.")

    # ---- does CV_f RANK the models the same way CV_px does? ----
    # This, not the size of rho, is what licenses reporting CV_f alone.
    # The paper's claims are comparative -- which model is steadiest, which
    # three take the Pareto stars -- so what matters is whether the two
    # metrics put the models in the same ORDER. A rho that is constant
    # across models preserves the order exactly however large it is; a rho
    # that varies reorders models that sit close together, and most of
    # yours do.
    def spearman(a, b):
        """Rank correlation without scipy: Pearson on the ranks."""
        def rank(x):
            order = np.argsort(x, kind="mergesort")
            r = np.empty(len(x), float)
            r[order] = np.arange(len(x), dtype=float)
            # average ties so equal values do not get an arbitrary order
            _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
            for k in np.flatnonzero(cnt > 1):
                m = inv == k
                r[m] = r[m].mean()
            return r
        ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    print()
    for bag in sorted({r["bag"] for r in runs}, key=lambda b: float(b[:-1])):
        g = [r for r in runs if r["bag"] == bag]
        if len(g) < 4:
            print(f"rank check [{bag}]: only {len(g)} runs, skipped")
            continue
        rs = spearman([r["J_f"] for r in g], [r["J_px"] for r in g])
        by_f = sorted(g, key=lambda r: r["J_f"])
        by_px = sorted(g, key=lambda r: r["J_px"])
        same3 = ({(r["model"], r["variant"]) for r in by_f[:3]}
                 == {(r["model"], r["variant"]) for r in by_px[:3]})
        print(f"rank check [{bag}]: {len(g)} runs, Spearman {rs:+.3f}, "
              f"steadiest three {'IDENTICAL' if same3 else 'DIFFER'}")
        if not same3:
            print(f"    by CV_f : "
                  + ", ".join(f"{r['model']}-{r['variant']}" for r in by_f[:3]))
            print(f"    by CV_px: "
                  + ", ".join(f"{r['model']}-{r['variant']}" for r in by_px[:3]))
            print("    -> CV_f alone would recommend a different model. "
                  "Report both columns.")

    if args.out:
        cols = ["model", "variant", "bag", "n", "depth", "J_f", "J_px",
                "J_px_full", "rho", "jitter", "drift", "n_px"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in runs:
                w.writerow(r)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()