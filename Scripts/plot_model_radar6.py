#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Radar (spider) chart comparing depth models across multiple axes.

  *** VARIANT 8: L2 SELECTION ***
  Identical to plot_model_radar7.py EXCEPT for how a single "best" model is
  picked out of a Pareto frontier. v7 uses the arithmetic mean of the
  normalised objectives; v8 uses the L2 (Euclidean) distance to the ideal
  point (1,...,1) -- compromise programming. The Pareto FRONTIER itself is
  unchanged in both: domination is a component-wise comparison and uses no
  norm at all. Only the highlighted top-3 and the rank-heatmap ordering
  differ. Run both and compare; where they disagree, one objective is very
  weak and the mean is forgiving it.

Reads every eval_<model>_*.csv in a directory and plots each model on a
normalised radar chart. All axes are oriented so that OUTWARD = BETTER.

Axes:
  Accuracy     from median AbsRel          (lower AbsRel -> further out)
  Scale        from |median scale - 1|     (closer to 1.0 -> further out)
  Speed        from median steady latency  (faster -> further out)
  Stability    from std of AbsRel          (less fluctuation -> further out)
  Compactness  from parameter count        (smaller -> further out) [optional]

IMPORTANT CAVEAT ON "STABILITY":
  This is the frame-to-frame std of the ERROR METRIC, not true temporal
  jitter of the depth map. It captures how consistently a model performs,
  which is related to but NOT the same as depth flicker. Measuring real
  jitter (e.g. TAE) requires the per-frame depth maps, which the CSVs do
  not store. Label it accordingly in any figure caption.

NORMALISATION:
  Each axis is min-max scaled across the models present, so the chart shows
  RELATIVE standing within your benchmark -- not absolute quality. Adding or
  removing a model rescales everyone. State this in the caption.

Usage:
    python3 plot_model_radar.py --dir . --out radar.png
    python3 plot_model_radar.py --dir . --best-per-model   # one config each

  Paper-figure styles (all respect --bag / --models filters):
    # Fig 2  accuracy vs #params (size), one line per family
    python3 plot_model_radar.py --dir . --bag 1.5m --style scatter --out fig2.png
    # Fig "overall" radar, one BEST variant per family
    python3 plot_model_radar.py --dir . --bag 1.5m --best-per-family --out fig_radar.png
    # Fig 3  speed-accuracy Pareto frontier
    python3 plot_model_radar.py --dir . --bag 1.5m --style pareto --out fig3.png
    # Fig 4  scale bias bar (critical for a metric-sensor replacement)
    python3 plot_model_radar.py --dir . --bag 1.5m --style scalebar --out fig4.png
    # Fig 5  temporal-stability boxplot (frame-to-frame depth variation)
    python3 plot_model_radar.py --dir . --bag 1.5m --style stability --out fig5.png
    # 2x2 composite (a) size (b) Pareto (c) scale (d) stability
    python3 plot_model_radar.py --dir . --bag 1.5m --style composite --out fig_all.png

  NOTE: pin a single --bag for every figure above. Mixing distances puts
  0.51 m and 1.5 m points on the same axes, and (for the categorical
  figures) lets a model be represented by its best-scoring distance.
"""

import argparse
import csv
import glob
import os
import re
import textwrap
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Parameter counts (millions) captured from the eval logs. Fill in as you
# measure them; models missing here simply drop the Compactness axis value.
# Parameter counts (millions), measured from the eval logs.
# Keys are matched as SUBSTRINGS against the config tag first (so per-variant
# sizes are picked up), then against the bare model name as a fallback.
# Parameter counts (millions), measured from the eval logs.
# Keyed by "<model>-<variant>" so sizes never get confused with each other;
# a bare "<model>" key is the fallback for single-variant models.
# Parameter counts in millions.
# Keys may be written either as "<model>-<variant>" (e.g. "unik3d-vitb") or as
# any distinctive substring of the CSV name (e.g. "unidepth-v2-vitl14").
# lookup_params() tries several matches and reports anything it cannot find,
# so a wrong key shows up as a warning rather than a silently blank column.
PARAMS_M = {
    # UniK3D
    "unik3d-vits": 35.4,
    "unik3d-vitb": 117.2,
    "unik3d-vitl": 358.8,
    # MoGe-2
    "moge-2-vits": 35.1,
    "moge-2-vitb": 104.7,        # <- fill in from that run's startup log
    "moge-2-vitl": 330.9,
    # UniDepthV2
    "unidepth-v2-vits14": 34.2,
    "unidepth-v2-vitb14": 114.4,
    "unidepth-v2-vitl14": 353.8,
    # Video Depth Anything (files may be eval_vda_... or eval_videodepth_...)
    "videodepth-vits": 29.0,
    "videodepth-vitb": 114.4,
    "videodepth-vitl": 384.9,
    # Metric3D v2
    "metric3d-small": 37.5,
    "metric3d-large": 411.9,
    "metric3d-giant2": 1377.7,
    # single-variant models
    "unidac": 354.8,
    "depthpro": 952.0,
}
_PARAM_MISSES = set()

# --------------------------------------------------------------------------
# Family colours, sampled from the palette in the reference state scatter.
# That figure was flagged "bad" only because it stretched a qualitative scale
# over 50 categories; with ~7 families the same colours are easy to tell
# apart. One colour PER FAMILY, applied consistently across every panel so a
# model keeps its colour whether it's a dot, a bar, or a box. Edit freely; any
# family not in FAMILY_COLORS falls back to cycling PALETTE.
# --------------------------------------------------------------------------
PALETTE = [
    "#718eca",  # blue
    "#e09d73",  # orange
    "#5ca15f",  # green
    "#a779c5",  # purple
    "#3cc3cb",  # cyan
    "#f38da4",  # coral
    "#b8b05a",  # olive
    "#e889d4",  # magenta
]

FAMILY_COLORS = {
    "moge":       "#5ca15f",   # green
    "unik3d":     "#a779c5",   # purple
    "unidepth":   "#718eca",   # blue
    "videodepth": "#3cc3cb",   # cyan
    "metric3d":   "#e09d73",   # orange
    "depthpro":   "#f38da4",   # coral
    "unidac":     "#e889d4",   # magenta
}
_COLOR_FALLBACK = {}


def color_for(family):
    """Stable colour for a family, identical across all panels.

    "unidac-equi" resolves to "unidac": the panoramic run is the same network
    scored differently, so it keeps the family hue rather than being handed a
    new colour as though it were a different model.
    """
    if family.endswith("-equi"):
        family = family[:-5]
    if family in FAMILY_COLORS:
        return FAMILY_COLORS[family]
    if family not in _COLOR_FALLBACK:
        _COLOR_FALLBACK[family] = PALETTE[len(_COLOR_FALLBACK) % len(PALETTE)]
    return _COLOR_FALLBACK[family]


# variant -> linestyle, so multiple sizes of one family (which share a colour)
# stay distinguishable in the distance plots. Larger model = more solid.
_VARIANT_LS = {
    "vits": ":", "vitb": "--", "vitl": "-",
    "small": ":", "large": "--", "giant2": "-",
    "": "-",
}

# 17 maximally-distinct line colours for the distance plots, where every
# model x variant is its own SOLID line (tab10 + Dark2, pastels dropped so
# every line stays visible). Edit freely.
LINE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d",
]


def build_line_colors(pts):
    """Stable model x variant -> distinct colour map for the distance plots."""
    keys = sorted({(d["model"], d["variant"]) for d in pts})
    return {k: LINE_PALETTE[i % len(LINE_PALETTE)] for i, k in enumerate(keys)}


# Soft, low-saturation heatmap ramps. The heatmap encodes 0 (worst) -> 1
# (best), so 'soft' keeps the traffic-light reading (rose = worst, sage =
# best) but stays pale enough that black annotations remain legible. The
# others are gentle sequential ramps. Any matplotlib colormap name also works.
_CMAP_PRESETS = {
    "soft":        ["#e4aeaa", "#f5efe1", "#b6d2ab"],   # rose -> cream -> sage
    "soft-warm":   ["#e7b7ad", "#f4ecdf", "#e9d59a"],   # rose -> cream -> sand
    "soft-blue":   ["#eed9d3", "#e6ecef", "#a9c7df"],   # warm -> cool, light
    "soft-purple": ["#efe6f0", "#e7def0", "#c1acdd"],   # pale lavender ramp
    "soft-gray":   ["#efe3e1", "#eeeeea", "#c8d6c6"],   # very muted
}


def resolve_cmap(name):
    """A preset soft ramp, or any named matplotlib colormap."""
    if name in _CMAP_PRESETS:
        return LinearSegmentedColormap.from_list(name, _CMAP_PRESETS[name],
                                                 N=256)
    return plt.get_cmap(name)


def lookup_params(model, variant, config):
    """Find a parameter count, trying the most specific match first.

    1. exact "<model>-<variant>"        e.g. unik3d + vitb -> "unik3d-vitb"
    2. longest PARAMS_M key that is a substring of the config
                                       e.g. "unidepth-v2-vitl14"
    3. bare "<model>"                  e.g. "depthpro"

    Records anything unmatched so main() can warn instead of silently
    leaving the column blank.
    """
    table = {k.lower(): v for k, v in PARAMS_M.items()}
    cfg = config.lower()
    if variant:
        v = table.get(f"{model}-{variant}".lower())
        if v is not None:
            return v
    best = None
    for k, v in table.items():
        if v is not None and k in cfg and (best is None or len(k) > len(best[0])):
            best = (k, v)
    if best:
        return best[1]
    v = table.get(model.lower())
    if v is not None:
        return v
    _PARAM_MISSES.add(f"{model}-{variant}" if variant else model)
    return None


def variant_of(config):
    """Pull a size tag out of the config.

    Handles two naming styles seen in the CSVs:
      vit-prefixed : moge-2-vitb-normal, unidepth-v2-vitl14, unik3d-vits,
                     vda_1.5m_vitb_fp16   -> vitb / vitl / vits
      bare words   : metric3d_1.5m_small / _large / _giant2
                     -> small / large / giant2

    Returns "" when the name carries no size (e.g. unidac, depthpro), in
    which case the legend falls back to the bare model name.
    """
    m = re.search(r"vit[_-]?(small|base|large|s|b|l)", config, re.I)
    if m:
        w = m.group(1).lower()
        return {"small": "vits", "base": "vitb", "large": "vitl",
                "s": "vits", "b": "vitb", "l": "vitl"}[w]
    # bare size words, anchored so "normal" can't match "small" etc.
    m = re.search(r"[_-](giant2?|large|base|small|tiny)(?:[_-]|$)",
                  config, re.I)
    if m:
        return m.group(1).lower()
    return ""


SHOW_BAG = False          # set in main() when more than one bag is loaded
SHOW_MODE = False         # set in main() when BOTH evaluations are loaded


def label_of(d):
    """Legend label: model + variant + the distinguishing config bits."""
    v = d.get("variant", "")
    base = f"{d['model']}-{v}" if v else d["model"]
    extra = []
    mt = re.search(r"_nt(\d+)", d["config"])
    if mt:
        extra.append(f"{mt.group(1)}tok")
    mr = re.search(r"_rl(\d+)", d["config"])
    if mr:
        extra.append(f"rl{mr.group(1)}")
    if "normal" in d["config"]:
        extra.append("n")
    if SHOW_BAG and d.get("bag"):
        extra.append(f"@{d['bag']}")
    return base + (" " + " ".join(extra) if extra else "")


# def short_name(d):
#     """Compact model-variant label for crowded categorical axes/annotations."""
#     return f"{d['model']}-{d['variant']}" if d.get("variant") else d["model"]


def load_csv(path):
    base = os.path.basename(path)
    # Model token may carry hyphens ("unik3d-equi", "moge-2"). Capture the
    # whole token, then split a trailing "-equi" off as the evaluation MODE
    # rather than as a different model: it is the same network, scored on the
    # sphere instead of in a pinhole window.
    m = re.match(r"eval_([a-z0-9]+(?:-[a-z0-9]+)*)_", base)
    model = m.group(1) if m else "unknown"
    # Keep "-equi" in the model NAME so the panoramic run gets its own row
    # everywhere things are keyed by (model, variant); colour still resolves
    # to the family inside color_for(), so unidac and unidac-equi share a hue.
    mode = "equi" if model.endswith("-equi") else "pinhole"
    # everything between the model name and .csv is the config tag
    cfg = base[len("eval_"):-4]
    # bag/distance token, e.g. eval_unik3d_1.5m_... -> "1.5m"
    mb = re.match(r"eval_[a-z0-9]+(?:-[a-z0-9]+)*_([0-9]*\.?[0-9]+m)_", base)
    bag = mb.group(1) if mb else ""
    bag_val = float(bag[:-1]) if bag else float("nan")

    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            # pred_p50_m / gt_p50_m are per-frame DEPTH medians (metres),
            # not error metrics -- they give a real temporal signal.
            try:
                p50 = float(r["pred_p50_m"]); g50 = float(r["gt_p50_m"])
            except (KeyError, ValueError):
                p50 = g50 = float("nan")
            try:
                rmse = float(r["RMSE_m"]); mae = float(r["MAE_m"])
            except (KeyError, ValueError):
                rmse = mae = float("nan")
            rows.append((int(r["frame"]), float(r["t_infer_ms"]),
                         float(r["AbsRel"]), float(r["delta1"]),
                         float(r["scale"]), p50, g50, rmse, mae))
    if not rows:
        return None
    rows.sort()

    frames_arr = np.array([r[0] for r in rows], dtype=float)
    lat = np.array([r[1] for r in rows if r[0] > 1]) if len(rows) > 1 \
        else np.array([r[1] for r in rows])
    absrel = np.array([r[2] for r in rows])
    d1 = np.array([r[3] for r in rows])
    scale = np.array([r[4] for r in rows])

    # --- temporal stability from the per-frame depth medians ---
    # The scenes are static, so the true depth is constant over the run and
    # the median predicted depth should be too. No ground-truth ratio is
    # needed: any variation is the model's own temporal instability.
    p50 = np.array([r[5] for r in rows], dtype=float)
    g50 = np.array([r[6] for r in rows], dtype=float)   # kept for depth-trace
    okp = np.isfinite(p50) & (p50 > 0)
    if okp.sum() >= 3:
        med = p50[okp]                       # per-frame median predicted depth
        mean_d = float(np.mean(med))
        # TEMPORAL STABILITY = coefficient of variation (CV) of the per-frame
        # median depth. The scenes are static, so the true depth does not
        # change and any spread is the model's own instability -- no ground
        # truth is needed. ddof=1 (sample std) matches the CV convention.
        cv_m = float(np.std(med, ddof=1))                  # sigma, metres
        cv_pct = 100.0 * cv_m / max(mean_d, 1e-9)          # sigma/mu, percent
        # deviation of each frame from the run mean, as a % of depth: the
        # distribution whose (sample) std is the CV above; feeds the boxplot.
        steps_pct = (med - mean_d) / max(mean_d, 1e-9) * 100.0
        # kept for reference only: mean frame-to-frame movement, raw metres
        step_m = float(np.mean(np.abs(np.diff(med))))
    else:
        cv_m = cv_pct = step_m = float("nan")
        mean_d = float("nan")
        steps_pct = np.array([])

    # cv_pct is the canonical name; 'drift'/'jitter' are kept as ALIASES for
    # the same value so older commands (--sort jitter, --metric jitter) work.
    drift = jitter = cv_pct

    return {
        "model": model, "config": cfg, "variant": variant_of(cfg),
        "mode": mode,
        "bag": bag, "bag_m": bag_val,
        "n": len(rows),
        "drift": drift, "jitter": jitter, "steps_pct": steps_pct,
        "cv_pct": cv_pct, "cv_m": cv_m, "step_m": step_m,
        "stab_pct": cv_pct, "stab_m": cv_m, "jitter_m": step_m,
        "mean_depth_m": mean_d,
        "frames": frames_arr, "pred_p50": p50, "gt_p50": g50,
        "AbsRel": float(np.median(absrel)),
        "AbsRel_std": float(np.std(absrel)),
        "RMSE_m": float(np.median([r[7] for r in rows])),
        "MAE_m": float(np.median([r[8] for r in rows])),
        "delta1": float(np.median(d1)),
        "delta1_std": float(np.std(d1)),
        "scale": float(np.median(scale)),
        "latency_ms": float(np.median(lat)),
        "Fps": float(1000.0 / max(np.median(lat), 1e-6)),
        "params_M": lookup_params(model, variant_of(cfg), cfg),
    }


def norm_better_high(vals):
    """Map values to [0,1] where HIGHER input is better (outward = better)."""
    v = np.asarray(vals, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.ones_like(v) * 0.75
    return (v - lo) / (hi - lo)


def norm_better_low(vals):
    """Map values to [0,1] where LOWER input is better (outward = better)."""
    v = np.asarray(vals, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.ones_like(v) * 0.75
    return 1.0 - (v - lo) / (hi - lo)


def l2_to_ideal(cols, weights=None):
    """Compromise programming: distance to the ideal point (1,...,1).

        D_i = sqrt( sum_k w_k (1 - x_hat[i,k])^2 )

    Each column must already be normalised to [0,1] with 1 = best. Weights
    default to equal and are rescaled to mean 1, so D stays on the same
    scale as the unweighted version (and score_from_ideal's 1 - D/sqrt(K)
    still reads 0..1). A larger w_k makes a shortfall on axis k hurt more.
    Returns (D, order) with order[0] = best.
    """
    Mx = np.column_stack(cols)
    if weights is None:
        w = np.ones(Mx.shape[1])
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() * len(w)
    D = np.sqrt((w * (1.0 - Mx) ** 2).sum(1))
    return D, list(np.argsort(D))


def score_from_ideal(cols):
    """L2 distance rescaled to a 0-1 'higher is better' score."""
    D, _ = l2_to_ideal(cols)
    return 1.0 - D / np.sqrt(len(cols))


# --------------------------------------------------------------------------
# model-set selectors
# --------------------------------------------------------------------------
def best_per_variant(pts):
    """One row per (model, variant): the lowest-AbsRel config of that size.

    Collapses repeated configs (e.g. different token budgets) but keeps the
    small/base/large variants of a family distinct.
    """
    best = {}
    for d in pts:
        k = (d["model"], d["variant"])
        if k not in best or d["AbsRel"] < best[k]["AbsRel"]:
            best[k] = d
    return list(best.values())


def best_per_family(pts):
    """One row per family (model): the lowest-AbsRel variant of that family.

    This is the 'overall comparison' selection -- a single representative
    per model, so the radar / bars stay readable.
    """
    best = {}
    for d in pts:
        k = d["model"]
        if k not in best or d["AbsRel"] < best[k]["AbsRel"]:
            best[k] = d
    return list(best.values())


# --------------------------------------------------------------------------
# individual panels -- each draws onto a supplied Axes so the composite can
# reuse the exact same code as the standalone figures.
# --------------------------------------------------------------------------
def short_name(d):
    """Compact model-variant label for crowded categorical axes/annotations.

    When more than one distance is loaded the bag is appended. The pareto /
    scalebar / stability panels keep ONE row per (model, variant), chosen by
    best AbsRel, so each point silently represents whichever distance scored
    best -- naming it makes that choice visible instead of implied.
    """
    base = f"{d['model']}-{d['variant']}" if d.get("variant") else d["model"]
    if SHOW_BAG and d.get("bag"):
        base += f"@{d['bag']}"
    return base

def panel_metric_vs_size(ax, pts, key, ylabel, lower_better=True, title=""):
    """Scatter of ANY metric against parameter count, one connected line per
    family so the size -> metric trend within a family is visible. x is
    log-scaled because params span ~30 M to ~1.4 B.

    Generalises the old AbsRel-only panel_accuracy_size to any key (jitter/
    cv_pct, Fps, AbsRel, scale, ...) so the SAME "does bigger help" story can
    be told for speed and stability, not just accuracy -- this is the panel
    behind --style size.
    """
    fams = defaultdict(list)
    for d in pts:
        if d.get("params_M") and np.isfinite(d.get(key, np.nan)):
            fams[d["model"]].append(d)
    if not fams:
        ax.text(0.5, 0.5, "no parameter counts / metric values\n"
                          "(fill in PARAMS_M)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=10)
        return
    for fam, ds in sorted(fams.items()):
        ds.sort(key=lambda d: d["params_M"])
        c = color_for(fam)
        ax.plot([d["params_M"] for d in ds], [d[key] for d in ds],
                "-o", color=c, ms=6, lw=1.6, label=fam)
        for d in ds:
            ax.annotate(d["variant"] or "", (d["params_M"], d[key]),
                        fontsize=6, color=c, xytext=(4, 3),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel(ylabel + ("   (lower is better)" if lower_better
                            else "   (higher is better)"))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title(title, fontsize=10)


def panel_accuracy_size(ax, pts, title=""):
    """AbsRel vs parameter count. Kept as a thin wrapper around
    panel_metric_vs_size so the existing 'scatter' and 'composite' styles,
    which call this by name, are unaffected by the generalisation above."""
    panel_metric_vs_size(ax, pts, "AbsRel", "AbsRel", lower_better=True,
                         title=title)


def panel_pareto(ax, pts, title=""):
    """FPS (x, higher better) vs AbsRel (y, lower better). The non-dominated
    set -- models you cannot beat on BOTH speed and accuracy at once -- is
    outlined and joined by a dashed staircase; dominated models are faded."""
    P = [d for d in pts if np.isfinite(d["Fps"]) and np.isfinite(d["AbsRel"])]
    if len(P) < 2:
        ax.text(0.5, 0.5, "need >=2 models", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=10)
        return
    # frontier: sort by Fps desc, keep points that strictly lower running-min
    # AbsRel. Those are exactly the ones where trading speed buys accuracy.
    order = sorted(P, key=lambda d: (-d["Fps"], d["AbsRel"]))
    front, best = [], np.inf
    for d in order:
        if d["AbsRel"] < best - 1e-12:
            front.append(d)
            best = d["AbsRel"]
    front_ids = {id(d) for d in front}

    def col(d):
        return color_for(d["model"])

    for d in P:
        on = id(d) in front_ids
        ax.scatter(d["Fps"], d["AbsRel"], s=75 if on else 42, color=col(d),
                   edgecolor="black" if on else "none",
                   linewidth=1.2 if on else 0,
                   zorder=3 if on else 2, alpha=1.0 if on else 0.45)
        ax.annotate(short_name(d), (d["Fps"], d["AbsRel"]), fontsize=6,
                    color="black" if on else "0.5",
                    xytext=(4, 2), textcoords="offset points")
    fsort = sorted(front, key=lambda d: d["Fps"])
    ax.plot([d["Fps"] for d in fsort], [d["AbsRel"] for d in fsort],
            "--", color="black", lw=1.3, zorder=1)
    ax.set_xlabel("throughput (FPS, higher is better)")
    ax.set_ylabel("AbsRel   (lower is better)")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)


def panel_pareto3(ax, pts, title="", show_stars=True, weights=None):
    """3-objective Pareto over speed (FPS, up), accuracy (AbsRel, down) and
    steadiness (CV_f, down). Speed and accuracy are the axes; steadiness is
    the bubble size (bigger = steadier), so the ideal model is a large bubble
    in the bottom-right. A point is on the frontier if no other point beats it
    on ALL THREE at once; those are outlined, the dominated ones faded.
    Needs jitter, so any model without per-frame depth is dropped here.

    `weights` applies ONLY to the starred top-3 (the L2-to-ideal pick), in
    the order AbsRel, FPS, CV. The frontier is a component-wise comparison
    and carries no weights at all, so it is identical either way.
    """
    P = [d for d in pts if np.isfinite(d["Fps"]) and np.isfinite(d["AbsRel"])
         and np.isfinite(d["jitter"])]
    dropped = [short_name(d) for d in pts if not np.isfinite(d["jitter"])]
    if dropped:
        print(f"[pareto3] no jitter, dropped: {', '.join(dropped)}")
    if len(P) < 2:
        ax.text(0.5, 0.5, "need >=2 models with jitter", ha="center",
                va="center", transform=ax.transAxes, fontsize=12)
        if title:
            ax.set_title(title, fontsize=12)
        return [], []
    Fps = np.array([d["Fps"] for d in P])
    ab = np.array([d["AbsRel"] for d in P])
    jit = np.array([d["jitter"] for d in P])
    # non-dominated over (Fps up, ab down, jit down)
    nd = [True] * len(P)
    for i in range(len(P)):
        for j in range(len(P)):
            if i == j:
                continue
            if (Fps[j] >= Fps[i] and ab[j] <= ab[i] and jit[j] <= jit[i] and
                    (Fps[j] > Fps[i] or ab[j] < ab[i] or jit[j] < jit[i])):
                nd[i] = False
                break
    # bubble size: steadiest (lowest jitter) -> biggest
    jn = (jit - jit.min()) / max(np.ptp(jit), 1e-12)   # 0 steadiest, 1 shakiest
    smin, smax = 45.0, 340.0
    sizes = smax - jn * (smax - smin)
    # SELECTION RULE: L2 distance to the ideal point (see l2_to_ideal).
    # Squaring the shortfall means one weak objective dominates, so a model
    # must be decent on ALL THREE axes to be highlighted. Column order is
    # AbsRel, FPS, CV -- `weights` follows that order.
    _D, order = l2_to_ideal([norm_better_low(ab), norm_better_high(Fps),
                             norm_better_low(jit)], weights)
    if weights:
        print(f"[pareto3] weights AbsRel/FPS/CV = "
              f"{weights[0]:g}/{weights[1]:g}/{weights[2]:g} "
              f"(frontier unaffected)")
    top_ids = {id(P[k]) for k in order[:3]} if show_stars else set()
    for i, d in enumerate(P):
        on = nd[i]
        star = id(d) in top_ids
        ax.scatter(Fps[i], ab[i],
                   s=sizes[i] * 1.7 if star else sizes[i],
                   marker="*" if star else "o",
                   color=color_for(d["model"]),
                   edgecolor="black" if (on or star) else "none",
                   linewidth=1.6 if (on or star) else 0,
                   alpha=1.0 if (on or star) else 0.40,
                   zorder=5 if star else (3 if on else 2))
        ax.annotate(short_name(d), (Fps[i], ab[i]), fontsize=6,
                    color="black" if (on or star) else "0.5",
                    xytext=(5, 3), textcoords="offset points")
    front = [short_name(d) for i, d in enumerate(P) if nd[i]]
    print(f"[pareto3] non-dominated (speed/accuracy/steadiness): "
          f"{', '.join(front)}")
    print(f"[pareto3] top-3 by L2 distance to ideal: "
          f"{', '.join(f'{short_name(P[k])} (D={_D[k]:.3f})' for k in order[:3])}")
    # size + star legend, and a separate family-colour legend
    star_label = ("top-3 balanced" if not weights else
                  f"top-3, weights {weights[0]:g}/{weights[1]:g}/"
                  f"{weights[2]:g}")
    size_handles = [
        ax.scatter([], [], s=smax, color="0.6", edgecolor="none",
                   label=f"$\\mathrm{{CV_f}}$ {jit.min():.2f}% "
                         f"(most stable)"),
        ax.scatter([], [], s=smin, color="0.6", edgecolor="none",
                   label=f"$\\mathrm{{CV_f}}$ {jit.max():.2f}% "
                         f"(least stable)"),
        Line2D([], [], marker="*", ls="none", color="0.35", markersize=13,
               label=star_label),
    ]
    fam_handles = [mpatches.Patch(color=color_for(f), label=f)
                   for f in sorted({d["model"] for d in P})]
    ax.set_xlabel(r"FPS $\uparrow$")
    ax.set_ylabel(r"AbsRel $\downarrow$")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=12)
    return size_handles, fam_handles


def panel_scale(ax, pts, title=""):
    """Signed scale bias as a percentage: (median scale - 1) * 100. A metric
    sensor replacement cares about DIRECTION, so bars go above/below a zero
    line (positive = overestimates depth). Closest to zero wins."""
    ds = list(pts)
    if not ds:
        ax.text(0.5, 0.5, "no models", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=10)
        return
    x = np.arange(len(ds))
    err = [(d["scale"] - 1.0) * 100.0 for d in ds]
    colors = [color_for(d["model"]) for d in ds]
    ax.bar(x, err, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(d) for d in ds], rotation=45,
                       ha="right", fontsize=6)
    ax.set_ylabel("scale bias  (%)   + over / \u2212 under,  0 = metric")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(title, fontsize=10)
    for xi, e in zip(x, err):
        ax.annotate(f"{e:+.1f}", (xi, e), fontsize=5, ha="center",
                    va="bottom" if e >= 0 else "top",
                    xytext=(0, 2 if e >= 0 else -2),
                    textcoords="offset points")


def panel_stability(ax, pts, title=""):
    """Boxplot of the per-frame median predicted depth, expressed as %
    deviation from that run's mean. A TIGHT box = the depth reading barely
    moves over the sequence; a tall box = it wanders. The (sample) standard
    deviation of this distribution is the coefficient of variation reported
    in the table and on the radar. Caption should note this is GLOBAL
    (median-depth) stability, not per-pixel flicker."""
    ds = [d for d in pts if d.get("steps_pct") is not None
          and len(d["steps_pct"]) > 0]
    if not ds:
        ax.text(0.5, 0.5, "no per-frame depth\n(pred_p50/gt_p50 missing)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=10)
        return
    data = [np.asarray(d["steps_pct"]) for d in ds]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=color_for(ds[i]["model"]), alpha=0.6)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)
    ax.set_xticks(range(1, len(ds) + 1))
    ax.set_xticklabels([short_name(d) for d in ds], rotation=45,
                       ha="right", fontsize=6)
    ax.axhline(0, color="0.4", lw=0.9, ls="--", zorder=1)
    ax.set_ylabel("per-frame median depth\n(% deviation from run mean)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(title, fontsize=10)


def panel_bags(ax, pts, key, color_map=None):
    """One metric vs scene distance, a line per model x variant. With a
    color_map every line gets its own solid colour; otherwise it falls back to
    family colour + variant linestyle. A rising line means that model degrades
    with distance. Returns (handles, labels, n_lines) for a shared legend."""
    better_low = key in ("AbsRel", "jitter", "drift", "RMSE_m", "MAE_m")
    # Short axis labels with an arrow for direction, matching the paper's
    # table: CV_f for the stability metric, and no "(lower is better)" prose
    # eating width in a double-column figure.
    YLAB = {
        "AbsRel": r"AbsRel $\downarrow$",
        "jitter": r"$\mathrm{CV_f}$ (%) $\downarrow$",
        "drift":  r"$\mathrm{CV_f}$ (%) $\downarrow$",
        "cv_pct": r"$\mathrm{CV_f}$ (%) $\downarrow$",
        "Fps":    r"FPS $\uparrow$",
        "RMSE_m": r"RMSE (m) $\downarrow$",
        "MAE_m":  r"MAE (m) $\downarrow$",
        "scale":  r"scale (1 = unbiased)",
    }
    groups = defaultdict(list)
    for d in pts:
        # group across bags by everything EXCEPT the bag
        gk = (d["model"], d["variant"],
              re.sub(r"^[^_]+_[0-9.]+m_", "", d["config"]))
        groups[gk].append(d)
    nseen = 0
    for gk, ds in sorted(groups.items()):
        ds = [d for d in ds if np.isfinite(d["bag_m"])]
        if len(ds) < 2:
            continue          # a single point is not a trend
        ds.sort(key=lambda d: d["bag_m"])
        mv = (ds[0]["model"], ds[0]["variant"])
        if color_map is not None:
            c, ls = color_map.get(mv, "#333333"), "-"
        else:
            c, ls = color_for(ds[0]["model"]), _VARIANT_LS.get(mv[1], "-")
        # Deliberately NOT short_name(): that appends "@<bag>" when several
        # distances are loaded, and a line here spans every bag, so tagging
        # it with one distance would be wrong.
        base = (f"{ds[0]['model']}-{ds[0]['variant']}"
                if ds[0]["variant"] else ds[0]["model"])
        ax.plot([d["bag_m"] for d in ds], [d[key] for d in ds],
                marker="o", ms=5, lw=1.8, color=c, ls=ls,
                label=base)
        nseen += 1
    if key == "scale":
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel(YLAB.get(
        key, f"{key}" + ("   (lower is better)" if better_low
                         else "   (higher is better)")))
    ax.set_xlabel("Scene distance (m)")
    ax.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    return h, l, nseen


def panel_dist_heatmap(fig, ax, pts, metrics, bags, cmap_name="soft",
                       fontsize=7, with_params=True, fps_target=20.0,
                       sort_by="name"):
    """Models x (metric x distance) grid, the layout of a benchmark sheet.
    Rows are model-variants; columns are grouped by metric, and within each
    metric run left-to-right over the distances. Parameter count, being a
    property of the model rather than of a distance, gets a SINGLE leading
    column instead of one per bag. Colour is normalised WITHIN EACH COLUMN so
    1 = best model in that column (same convention as the single-bag
    heatmap); the printed number is the RAW value. A blank cell means that
    model was not run at that distance.
    """
    # index every (model, variant) x bag we have
    by = {}
    for d in pts:
        by[((d["model"], d["variant"]), d["bag"])] = d
    # params: one value per row, taken from whichever bag has it
    prm = {}
    for d in pts:
        k = (d["model"], d["variant"])
        if d.get("params_M") and k not in prm:
            prm[k] = d["params_M"]

    # ---- row order -------------------------------------------------------
    # A row spans several bags, so every key except params (which is a
    # property of the model, not of a distance) is the MEDIAN across the bags
    # that model was actually run on. Rows missing a value sort last rather
    # than crashing or silently landing at the top.
    keys = sorted({(d["model"], d["variant"]) for d in pts})
    per_row = {k: [d for d in pts if (d["model"], d["variant"]) == k]
               for k in keys}

    def _med(k, field, transform=None):
        vals = [d.get(field, np.nan) for d in per_row[k]]
        if transform:
            vals = [transform(v) for v in vals]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else np.inf

    _order = {
        "params":   lambda k: prm.get(k, np.inf),
        "accuracy": lambda k: _med(k, "AbsRel"),
        "speed":    lambda k: -_med(k, "Fps"),
        "scale":    lambda k: _med(k, "scale", lambda v: abs(v - 1.0)),
        "jitter":   lambda k: _med(k, "cv_pct"),
        "name":     lambda k: k,
    }
    rows = sorted(keys, key=_order.get(sort_by, _order["name"]))
    if sort_by != "name":
        print(f"[dist-heatmap] rows ordered by {sort_by} (best/smallest first)")

    has_params = with_params and len(prm) > 0
    # RMSE_m/MAE_m ARE scored/ranked columns here, unlike on the plain
    # 'heatmap' style. dist-heatmap never combines its columns into one
    # score (no L2-to-ideal, no radar), so there is no double-counting risk
    # from also ranking RMSE/MAE alongside AbsRel -- they are just two more
    # independent columns on the sheet.
    better_low = {"AbsRel": True, "RMSE_m": True, "MAE_m": True,
                  "cv_pct": True, "stab_pct": True,
                  "Fps": False,
                  "scale_bias": True}
    fmt = {"AbsRel": "{:.3f}", "RMSE_m": "{:.3f}", "MAE_m": "{:.3f}",
           "cv_pct": "{:.2f}", "stab_pct": "{:.2f}",
           "Fps": "{:.1f}",
           "scale_bias": "{:+.1f}"}
    nice = {"AbsRel": "AbsRel", "RMSE_m": "RMSE (m)", "MAE_m": "MAE (m)",
            "cv_pct": "Stability, $\\mathrm{CV_f}$ (%)",
            "stab_pct": "Stability, $\\mathrm{CV_f}$ (%)",
            "Fps": (f"FPS  (green $\\geq$ {fps_target:g})" if fps_target
                    else "FPS"),
            "scale_bias": "Scale bias (%)"}

    def value(d, key):
        if d is None:
            return float("nan")
        if key == "scale_bias":
            return (d["scale"] - 1.0) * 100.0
        return d.get(key, float("nan"))

    # column plan: optional params column, then metric x distance
    pairs = [(m, b) for m in metrics for b in bags]
    npar = 1 if has_params else 0
    ncol = npar + len(pairs)
    raw = np.full((len(rows), ncol), np.nan)
    if has_params:
        for i, rk in enumerate(rows):
            raw[i, 0] = prm.get(rk, np.nan)
    for j, (mk, bg) in enumerate(pairs):
        for i, rk in enumerate(rows):
            raw[i, npar + j] = value(by.get((rk, bg)), mk)

    # ---- diagnose "flat" columns BEFORE drawing ----
    # A column with (near) zero spread colours every row the SAME shade (the
    # fallback below) -- that can look like normalisation "isn't working"
    # even though it is; there is simply nothing to rank. Separately, the
    # Fps column uses an ABSOLUTE threshold (--fps-target), not per-column
    # min/max, so it saturates solid green (or stays near the red end)
    # whenever every model lands on the same side of that line -- by design,
    # not a bug, but it looks identical to a flat column from the picture
    # alone. Printed once per column so the cause is never a guessing game.
    for j, (mk, bg) in enumerate(pairs):
        col = raw[:, npar + j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        lo, hi = np.nanmin(col[ok]), np.nanmax(col[ok])
        if hi - lo < 1e-9:
            print(f"[dist-heatmap] flat column {mk}@{bg}: all {int(ok.sum())} "
                  f"values ~= {lo:.4g} -- every cell will be the same "
                  f"colour, nothing to rank")
        elif mk == "Fps" and fps_target:
            n_clip = int((col[ok] / float(fps_target) >= 1.0).sum())
            if n_clip == ok.sum():
                print(f"[dist-heatmap] Fps@{bg}: all {n_clip} model(s) clear "
                      f"--fps-target {fps_target:g} -> column is fully "
                      f"saturated green (absolute threshold, not relative "
                      f"ranking). Pass --fps-target 0 for per-column "
                      f"min/max colouring instead.")
            elif n_clip == 0:
                print(f"[dist-heatmap] Fps@{bg}: NO model clears "
                      f"--fps-target {fps_target:g} -> column stays near "
                      f"the red end for everyone. Lower --fps-target or "
                      f"pass 0 for relative colouring.")

    # normalise per column: 1 = best model in that column
    norm = np.full_like(raw, np.nan)

    def fill(j, col, low_is_better, use_abs=False):
        ok = np.isfinite(col)
        if ok.sum() == 0:
            return
        v = np.abs(col) if use_abs else col
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi - lo < 1e-12:
            norm[ok, j] = 0.75
        elif low_is_better:
            norm[ok, j] = 1.0 - (v[ok] - lo) / (hi - lo)
        else:
            norm[ok, j] = (v[ok] - lo) / (hi - lo)

    if has_params:
        fill(0, raw[:, 0], True)              # smaller model = better
    for j, (mk, _bg) in enumerate(pairs):
        if mk == "Fps" and fps_target:
            # ABSOLUTE scale: >= fps_target is fully green (usable in real
            # time), 0 FPS is fully red, linear in between. This is a fixed
            # threshold, NOT a ranking, so a column where nothing clears the
            # bar stays red rather than crowning a relative winner.
            col = raw[:, npar + j]
            ok = np.isfinite(col)
            norm[ok, npar + j] = np.clip(col[ok] / float(fps_target), 0.0, 1.0)
        else:
            fill(npar + j, raw[:, npar + j], better_low[mk],
                 use_abs=(mk == "scale_bias"))

    im = ax.imshow(norm, cmap=resolve_cmap(cmap_name), vmin=0, vmax=1,
                   aspect="auto")

    # cell text (raw values); NaN -> blank
    for i in range(len(rows)):
        if has_params and np.isfinite(raw[i, 0]):
            ax.text(0, i, f"{raw[i, 0]:.0f}", ha="center", va="center",
                    fontsize=fontsize, color="black")
        for j, (mk, _bg) in enumerate(pairs):
            v = raw[i, npar + j]
            ax.text(npar + j, i,
                    "" if not np.isfinite(v) else fmt[mk].format(v),
                    ha="center", va="center", fontsize=fontsize, color="black")

         # Box the single winner of each metric GROUP, chosen on the RAW value.
    # One box per metric, on the best cell at whichever distance achieved it,
    # rather than one per column. Reading `raw` instead of `norm` also means
    # the FPS column's absolute clipping (everything at/above --fps-target
    # normalises to exactly 1.0) cannot produce a tie.
    # CAUTION for RMSE_m/MAE_m: those grow with true distance even for a
    # perfect model, so the nearest bag wins their group for reasons that
    # have nothing to do with the model.
    if has_params:
        i_best = int(np.nanargmin(raw[:, 0]))
        ax.add_patch(plt.Rectangle((-0.5, i_best - 0.5), 1, 1, fill=False,
                                   edgecolor="#3a3a3a", linewidth=1.5,
                                   zorder=5))
    for g, mk in enumerate(metrics):
        j0 = npar + g * len(bags)
        sub = raw[:, j0:j0 + len(bags)]
        if not np.any(np.isfinite(sub)):
            continue
        v = np.abs(sub) if mk == "scale_bias" else sub
        flat = np.nanargmin(v) if better_low[mk] else np.nanargmax(v)
        i_best, dj = np.unravel_index(int(flat), sub.shape)
        print(f"[box] {mk} -> {rows[i_best]} @ {bags[dj]} = "
              f"{sub[i_best, dj]:.3f}")
        ax.add_patch(plt.Rectangle((j0 + dj - 0.5, i_best - 0.5), 1, 1,
                                   fill=False, edgecolor="#3a3a3a",
                                   linewidth=1.5, zorder=5))
    # distance ticks on the bottom, metric group labels above
    ax.set_xticks(range(ncol))
    ax.set_xticklabels((["M"] if has_params else [])
                       + [("" if len(bags) == 1 else b)
                          for _m in metrics for b in bags],
                       fontsize=fontsize)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{m}-{v}" if v else m for m, v in rows],
                       fontsize=fontsize)
    if has_params:                                   # separate params block
        ax.axvline(0.5, color="white", lw=3)
    for g in range(1, len(metrics)):                 # separators between groups
        ax.axvline(npar + g * len(bags) - 0.5, color="white", lw=3)
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(([0] if has_params else [])
                   + [npar + g * len(bags) + (len(bags) - 1) / 2
                      for g in range(len(metrics))])
    sec.set_xticklabels((["Params"] if has_params else [])
                        + [nice[m] for m in metrics], fontsize=fontsize + 1)
    sec.tick_params(length=0)
    return im

def panel_pareto4(ax, pts, title="", show_stars=True):
    """4-objective frontier: speed (Fps, up), accuracy (AbsRel, down),
    stability (CV, down) and metric scale bias (|scale-1|, down).
    x = Fps, y = AbsRel, bubble SIZE = stability (bigger = steadier),
    bubble EDGE COLOUR = scale bias (dark edge = closer to metric truth).
    A point is non-dominated if no other point is at least as good on ALL
    FOUR objectives and strictly better on one. Top-3 by the balanced score
    (equal-weight mean of the four normalised objectives) get a gold ring."""
    P = [d for d in pts if np.isfinite(d["Fps"]) and np.isfinite(d["AbsRel"])
         and np.isfinite(d.get("cv_pct", np.nan))
         and np.isfinite(d.get("scale", np.nan))]
    if len(P) < 2:
        ax.text(0.5, 0.5, "need >=2 models with all four metrics",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        return [], []
    Fps = np.array([d["Fps"] for d in P])
    ab = np.array([d["AbsRel"] for d in P])
    cv = np.array([d["cv_pct"] for d in P])
    sb = np.array([abs(d["scale"] - 1.0) * 100.0 for d in P])   # |bias| in %
    # --- domination over four objectives ---
    nd = [True] * len(P)
    for i in range(len(P)):
        for j in range(len(P)):
            if i == j:
                continue
            atleast = (Fps[j] >= Fps[i] and ab[j] <= ab[i]
                       and cv[j] <= cv[i] and sb[j] <= sb[i])
            strict = (Fps[j] > Fps[i] or ab[j] < ab[i]
                      or cv[j] < cv[i] or sb[j] < sb[i])
            if atleast and strict:
                nd[i] = False
                break
    # bubble size from stability (steadiest = biggest)
    cvn = (cv - cv.min()) / max(np.ptp(cv), 1e-12)
    smin, smax = 45.0, 340.0
    sizes = smax - cvn * (smax - smin)
    # edge shade from scale bias (least biased = darkest edge)
    sbn = (sb - sb.min()) / max(np.ptp(sb), 1e-12)
    # SELECTION RULE: L2 distance to the ideal point (see l2_to_ideal).
    _D, order = l2_to_ideal([norm_better_low(ab), norm_better_high(Fps),
                             norm_better_low(cv), norm_better_low(sb)])
    top_ids = ({id(P[k]) for k in order[:3]} if show_stars else set())
    for i, d in enumerate(P):
        edge = str(0.15 + 0.7 * sbn[i])        # 0.15 dark .. 0.85 pale
        ax.scatter(Fps[i], ab[i], s=sizes[i], marker="o",
                   color=color_for(d["model"]),
                   edgecolor=edge, linewidth=2.2 if nd[i] else 1.2,
                   alpha=1.0 if (nd[i] or id(d) in top_ids) else 0.45,
                   zorder=3 if nd[i] else 2)
        if id(d) in top_ids:
            ax.scatter(Fps[i], ab[i], s=sizes[i] * 2.1, marker="o",
                       facecolor="none", edgecolor="#d4a017", linewidth=2.4,
                       zorder=6)
        ax.annotate(short_name(d), (Fps[i], ab[i]), fontsize=6,
                    color="black" if nd[i] else "0.5",
                    xytext=(5, 3), textcoords="offset points")
    print(f"[pareto4] non-dominated (speed/accuracy/stability/scale): "
          f"{', '.join(short_name(d) for i, d in enumerate(P) if nd[i])}")
    print(f"[pareto4] top-3 by L2 distance to ideal: "
          f"{', '.join(f'{short_name(P[k])} (D={_D[k]:.3f})' for k in order[:3])}")
    size_handles = [
        ax.scatter([], [], s=smax, color="0.6", edgecolor="none",
                   label=f"CV {cv.min():.2f}% (steadiest)"),
        ax.scatter([], [], s=smin, color="0.6", edgecolor="none",
                   label=f"CV {cv.max():.2f}% (least steady)"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="0.75",
               markeredgecolor="0.15", markeredgewidth=2.2, markersize=11,
               label=f"scale bias {sb.min():.1f}% (dark edge)"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="0.75",
               markeredgecolor="0.85", markeredgewidth=2.2, markersize=11,
               label=f"scale bias {sb.max():.1f}% (pale edge)"),
    ]
    if show_stars:
        size_handles.append(
            Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                   markeredgecolor="#d4a017", markeredgewidth=2.2,
                   markersize=13, label="top-3 closest to ideal (L2)"))
    fam_handles = [mpatches.Patch(color=color_for(f), label=f)
                   for f in sorted({d["model"] for d in P})]
    ax.set_xlabel("Throughput (FPS)\n[higher is better]")
    ax.set_ylabel("AbsRel [lower is better]")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    return size_handles, fam_handles


def panel_pareto5(ax, pts, title="", show_stars=True, annotate_fps=True):
    """2-objective QUALITY frontier: accuracy vs temporal stability, no speed.
    x = temporal stability (CV of the per-frame median depth, %) -> lower better
    y = accuracy (AbsRel)                                        -> lower better
    Both objectives are minimised, so the frontier is the lower-left staircase:
    a model is dominated when another is at least as accurate AND at least as
    stable, and strictly better on one. Throughput is deliberately excluded --
    this answers "which model gives the best depth if speed is not a
    constraint?", the counterpart to the speed-aware pareto/pareto3/pareto4.
    FPS is still printed next to each label so the reader can see what the
    quality leaders would cost in throughput."""
    P = [d for d in pts if np.isfinite(d["AbsRel"])
         and np.isfinite(d.get("cv_pct", np.nan))]
    if len(P) < 2:
        ax.text(0.5, 0.5, "need >=2 models with AbsRel and CV", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        return []
    ab = np.array([d["AbsRel"] for d in P])
    cv = np.array([d["cv_pct"] for d in P])
    # non-dominated over (AbsRel down, CV down)
    nd = [True] * len(P)
    for i in range(len(P)):
        for j in range(len(P)):
            if i == j:
                continue
            if (ab[j] <= ab[i] and cv[j] <= cv[i]
                    and (ab[j] < ab[i] or cv[j] < cv[i])):
                nd[i] = False
                break
    for i, d in enumerate(P):
        on = nd[i]
        ax.scatter(cv[i], ab[i], s=88 if on else 46,
                   color=color_for(d["model"]),
                   edgecolor="black" if on else "none",
                   linewidth=1.3 if on else 0,
                   alpha=1.0 if on else 0.45, zorder=3 if on else 2)
        lab = short_name(d)
        if annotate_fps and np.isfinite(d.get("Fps", np.nan)):
            lab += f"  ({d['Fps']:.0f} FPS)"
        ax.annotate(lab, (cv[i], ab[i]), fontsize=6,
                    color="black" if on else "0.5",
                    xytext=(5, 3), textcoords="offset points")
    # staircase through the frontier, sorted by CV
    fs = sorted([P[i] for i in range(len(P)) if nd[i]],
                key=lambda d: d["cv_pct"])
    ax.plot([d["cv_pct"] for d in fs], [d["AbsRel"] for d in fs],
            "--", color="black", lw=1.3, zorder=1)
    front = [short_name(d) for d in fs]
    print(f"[pareto5] non-dominated (accuracy/stability, no FPS): "
          f"{', '.join(front)}")
    if show_stars:
        # SELECTION RULE: L2 distance to the ideal point (see l2_to_ideal).
        _D, _order = l2_to_ideal([norm_better_low(ab), norm_better_low(cv)])
        best3 = _order[:3]
        for i in best3:
            ax.scatter(cv[i], ab[i], s=250, marker="o", facecolor="none",
                       edgecolor="#d4a017", linewidth=2.4, zorder=6)
        print(f"[pareto5] top-3 by L2 distance to ideal: "
              f"{', '.join(f'{short_name(P[i])} (D={_D[i]:.3f})' for i in best3)}")
    ax.set_xlabel("Temporal stability, CV (%)\n[lower is better]")
    ax.set_ylabel("AbsRel [lower is better]")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    ax.annotate("best", xy=(0.03, 0.03), xycoords="axes fraction",
                fontsize=8, color="0.35")
    return [mpatches.Patch(color=color_for(f), label=f)
            for f in sorted({d["model"] for d in P})]


def panel_pareto6(ax, pts, title="", show_stars=True, annotate_fps=True):
    """3-objective QUALITY frontier -- no speed.
    x = metric scale error |s-1| (%)                  -> lower better
    y = accuracy AbsRel                               -> lower better
    bubble SIZE = temporal instability CV (%)         -> lower better (bigger)
    All three are minimised, so the ideal model is a LARGE bubble in the
    bottom-left. Throughput is deliberately excluded: this is the
    "which model gives the best depth if speed is not a constraint?" figure,
    the quality-only counterpart to pareto4. FPS is still printed beside each
    label so the reader sees what the quality leaders would cost.
    A model is non-dominated when no other is at least as accurate AND at
    least as metrically unbiased AND at least as stable, with one strict.
    """
    P = [d for d in pts if np.isfinite(d["AbsRel"])
         and np.isfinite(d.get("cv_pct", np.nan))
         and np.isfinite(d.get("scale", np.nan))]
    if len(P) < 2:
        ax.text(0.5, 0.5, "need >=2 models with AbsRel, CV and scale",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        return [], []
    ab = np.array([d["AbsRel"] for d in P])
    cv = np.array([d["cv_pct"] for d in P])
    sb = np.array([abs(d["scale"] - 1.0) * 100.0 for d in P])
    # --- domination over the three quality objectives (all minimised) ---
    nd = [True] * len(P)
    for i in range(len(P)):
        for j in range(len(P)):
            if i == j:
                continue
            if (ab[j] <= ab[i] and cv[j] <= cv[i] and sb[j] <= sb[i]
                    and (ab[j] < ab[i] or cv[j] < cv[i] or sb[j] < sb[i])):
                nd[i] = False
                break
    # bubble size from stability (most stable = biggest)
    cvn = (cv - cv.min()) / max(np.ptp(cv), 1e-12)
    smin, smax = 45.0, 340.0
    sizes = smax - cvn * (smax - smin)
    top_ids = set()
    if show_stars:
        # L2 distance to the ideal point (see l2_to_ideal)
        _D, _order = l2_to_ideal([norm_better_low(ab), norm_better_low(cv),
                                  norm_better_low(sb)])
        top_ids = {id(P[k]) for k in _order[:3]}
    for i, d in enumerate(P):
        on = nd[i]
        top = id(d) in top_ids
        ax.scatter(sb[i], ab[i], s=sizes[i], marker="o",
                   color=color_for(d["model"]),
                   edgecolor="black" if on else "none",
                   linewidth=1.4 if on else 0,
                   alpha=1.0 if (on or top) else 0.42,
                   zorder=3 if on else 2)
        if top:
            ax.scatter(sb[i], ab[i], s=sizes[i] * 2.1, marker="o",
                       facecolor="none", edgecolor="#d4a017", linewidth=2.4,
                       zorder=6)
        lab = short_name(d)
        if annotate_fps and np.isfinite(d.get("Fps", np.nan)):
            lab += f"  ({d['Fps']:.0f} FPS)"
        ax.annotate(lab, (sb[i], ab[i]), fontsize=6,
                    color="black" if (on or top) else "0.5",
                    xytext=(5, 3), textcoords="offset points")
    print(f"[pareto6] non-dominated (accuracy/stability/scale, no FPS): "
          f"{', '.join(short_name(d) for i, d in enumerate(P) if nd[i])}")
    if show_stars:
        print(f"[pareto6] top-3 by L2 distance to ideal: "
              f"{', '.join(f'{short_name(P[k])} (D={_D[k]:.3f})' for k in _order[:3])}")
    size_handles = [
        ax.scatter([], [], s=smax, color="0.6", edgecolor="none",
                   label=f"CV {cv.min():.2f}% (most stable)"),
        ax.scatter([], [], s=smin, color="0.6", edgecolor="none",
                   label=f"CV {cv.max():.2f}% (least stable)"),
    ]
    if show_stars:
        size_handles.append(
            Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                   markeredgecolor="#d4a017", markeredgewidth=2.2,
                   markersize=13, label="top-3 closest to ideal (L2)"))
    fam_handles = [mpatches.Patch(color=color_for(f), label=f)
                   for f in sorted({d["model"] for d in P})]
    ax.set_xlabel("Metric scale error |s-1| (%)\n[lower is better]")
    ax.set_ylabel("AbsRel [lower is better]")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    ax.annotate("best", xy=(0.03, 0.03), xycoords="axes fraction",
                fontsize=8, color="0.35")
    return size_handles, fam_handles


def panel_pareto2(ax, pts, title="", show_stars=True, annotate_absrel=True):
    """2-objective frontier: temporal stability vs speed. Accuracy excluded.
    x = throughput FPS                          -> higher better
    y = temporal instability CV (%)             -> lower better
    The frontier is the lower-right staircase: a model is dominated when
    another is at least as fast AND at least as stable, with one strict.
    Accuracy is deliberately left out -- this isolates the runtime question
    "can a model hold a steady depth while running fast?", which is what a
    closed-loop tracker feels most directly. AbsRel is still printed beside
    each label so the reader can see the accuracy each point comes with.
    """
    P = [d for d in pts if np.isfinite(d["Fps"])
         and np.isfinite(d.get("cv_pct", np.nan))]
    if len(P) < 2:
        ax.text(0.5, 0.5, "need >=2 models with FPS and CV", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        return []
    fp = np.array([d["Fps"] for d in P])
    cv = np.array([d["cv_pct"] for d in P])
    # non-dominated over (FPS up, CV down)
    nd = [True] * len(P)
    for i in range(len(P)):
        for j in range(len(P)):
            if i == j:
                continue
            if (fp[j] >= fp[i] and cv[j] <= cv[i]
                    and (fp[j] > fp[i] or cv[j] < cv[i])):
                nd[i] = False
                break
    top_ids = set()
    if show_stars:
        _D, _order = l2_to_ideal([norm_better_high(fp),
                                  norm_better_low(cv)])
        top_ids = {id(P[k]) for k in _order[:3]}
    for i, d in enumerate(P):
        on = nd[i]
        top = id(d) in top_ids
        ax.scatter(fp[i], cv[i], s=88 if on else 46,
                   color=color_for(d["model"]),
                   edgecolor="black" if on else "none",
                   linewidth=1.3 if on else 0,
                   alpha=1.0 if (on or top) else 0.45,
                   zorder=3 if on else 2)
        if top:
            ax.scatter(fp[i], cv[i], s=250, marker="o", facecolor="none",
                       edgecolor="#d4a017", linewidth=2.4, zorder=6)
        lab = short_name(d)
        if annotate_absrel and np.isfinite(d.get("AbsRel", np.nan)):
            lab += f"  ({d['AbsRel']:.3f})"
        ax.annotate(lab, (fp[i], cv[i]), fontsize=6,
                    color="black" if (on or top) else "0.5",
                    xytext=(5, 3), textcoords="offset points")
    fs = sorted([P[i] for i in range(len(P)) if nd[i]], key=lambda d: d["Fps"])
    ax.plot([d["Fps"] for d in fs], [d["cv_pct"] for d in fs],
            "--", color="black", lw=1.3, zorder=1)
    print(f"[pareto2] non-dominated (speed/stability, no accuracy): "
          f"{', '.join(short_name(d) for d in fs)}")
    if show_stars:
        print(f"[pareto2] top-3 by L2 distance to ideal: "
              f"{', '.join(f'{short_name(P[k])} (D={_D[k]:.3f})' for k in _order[:3])}")
    ax.set_xlabel("Throughput (FPS)\n[higher is better]")
    ax.set_ylabel("Temporal stability, CV (%)\n[lower is better]")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    ax.annotate("best", xy=(0.96, 0.04), xycoords="axes fraction",
                fontsize=8, color="0.35", ha="right")
    handles = [mpatches.Patch(color=color_for(f), label=f)
               for f in sorted({d["model"] for d in P})]
    if show_stars:
        handles.append(Line2D([], [], marker="o", ls="none",
                              markerfacecolor="none", markeredgecolor="#d4a017",
                              markeredgewidth=2.2, markersize=13,
                              label="top-3 closest to ideal (L2)"))
    return handles


def panel_consistency(ax, pts, metrics=None, cmap_name="soft", fontsize=7,
                      min_bags=2):
    """Models x metrics grid where each cell is the SPREAD (max - min) of
    that metric across every distance the model was actually run at --
    directly answering "does this model behave the same at every range, or
    does it wander?" for every model at once. Companion to dist-heatmap,
    which shows the raw per-distance values this figure collapses into one
    number per (model, metric).

    WHAT GOES IN, AND WHY:
      AbsRel, cv_pct (jitter), Fps : spread = max - min, own units.
      scale : spread of the RAW (signed) value, NOT of |scale-1|. Taking
        |scale-1| per distance first and then spreading THAT would let a
        model reading +5% at one range and -5% at another look perfectly
        consistent (both |.|s equal 0.05, spread 0) when it has actually
        flipped between over- and under-estimating -- precisely the
        inconsistency this figure exists to catch. Displayed as raw-scale
        range x100 (percentage points) to match the scale-bias convention
        used elsewhere.
      RMSE_m / MAE_m : excluded on purpose. Their absolute scale grows with
        true distance even for a flawless model (a 5 cm error is 10% at
        0.5 m but 3.3% at 1.5 m), so their spread would measure the SCENE,
        not the model -- the same reason dist-heatmap's own console note
        gives for not comparing them across bags.

    A model run at only ONE distance has NO spread to report ('-'), which
    is deliberately different from 0: zero would claim perfect consistency
    for a condition that was never actually tested twice. Such rows sort
    last rather than crashing or landing arbitrarily.

    Rows are sorted by MEAN spread across the available metrics, smallest
    first -- reading top to bottom IS the consistency ranking, most
    steady-across-range model first.
    """
    metrics = metrics or ["AbsRel", "cv_pct", "scale", "Fps"]
    nice = {"AbsRel": "AbsRel range", "cv_pct": "CV range (pp)",
            "scale": "Scale range (%)", "Fps": "FPS range"}
    fmt = {"AbsRel": "{:.3f}", "cv_pct": "{:.2f}", "scale": "{:.1f}",
           "Fps": "{:.1f}"}
    pct100 = {"scale"}   # computed/shown x100 (percentage points)

    keys = sorted({(d["model"], d["variant"]) for d in pts})
    by_key = defaultdict(list)
    for d in pts:
        by_key[(d["model"], d["variant"])].append(d)

    raw = np.full((len(keys), len(metrics)), np.nan)
    n_bags_row = []
    for i, k in enumerate(keys):
        ds = by_key[k]
        nb = len({d["bag"] for d in ds if d["bag"]})
        n_bags_row.append(nb)
        if nb < min_bags:
            continue
        for j, mk in enumerate(metrics):
            vals = [d[mk] for d in ds if np.isfinite(d.get(mk, np.nan))]
            if len(vals) < 2:
                continue
            spread = max(vals) - min(vals)
            if mk in pct100:
                spread *= 100.0
            raw[i, j] = spread

    # LOW spread = 1 (greenest) for every column, uniformly -- unlike
    # dist-heatmap there is no "higher is better" metric here; a spread is
    # always something you want small.
    norm = np.full_like(raw, np.nan)
    for j in range(len(metrics)):
        col = raw[:, j]
        ok = np.isfinite(col)
        if ok.sum() == 0:
            continue
        lo, hi = np.nanmin(col[ok]), np.nanmax(col[ok])
        norm[ok, j] = (0.75 if hi - lo < 1e-12
                       else 1.0 - (col[ok] - lo) / (hi - lo))

    def _rowkey(i):
        vals = raw[i, :]
        ok = np.isfinite(vals)
        return float(np.mean(vals[ok])) if ok.any() else np.inf
    order = sorted(range(len(keys)), key=_rowkey)
    keys = [keys[i] for i in order]
    raw, norm = raw[order], norm[order]
    n_bags_row = [n_bags_row[i] for i in order]

    im = ax.imshow(norm, cmap=resolve_cmap(cmap_name), vmin=0, vmax=1,
                   aspect="auto")
    for i in range(len(keys)):
        for j, mk in enumerate(metrics):
            v = raw[i, j]
            ax.text(j, i, "-" if not np.isfinite(v) else fmt[mk].format(v),
                    ha="center", va="center", fontsize=fontsize,
                    color="black")
    for j in range(len(metrics)):
        col = norm[:, j]
        if not np.any(np.isfinite(col)):
            continue
        i_best = int(np.nanargmax(col))
        ax.add_patch(plt.Rectangle((j - 0.5, i_best - 0.5), 1, 1,
                                   fill=False, edgecolor="#3a3a3a",
                                   linewidth=1.5, zorder=5))
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([nice[m] for m in metrics], fontsize=fontsize + 2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(
        [f"{m}-{v}" if v else m for m, v in keys], fontsize=fontsize + 1)
    for i, nb in enumerate(n_bags_row):
        if nb < min_bags:
            ax.text(-0.62, i, "*", ha="center", va="center", fontsize=9,
                    color="#b5473f")
    print(f"[consistency] most consistent first: " +
          ", ".join(f"{m}-{v}" if v else m for m, v in keys[:5]) + " > ...")
    n_incomplete = sum(1 for nb in n_bags_row if nb < min_bags)
    if n_incomplete:
        print(f"[consistency] {n_incomplete} row(s) marked * ran at fewer "
              f"than {min_bags} distances -- no spread to report, sorted "
              f"last")
    return im


def panel_rank_heatmap(fig, ax, pts, cmap_name="soft", fontsize=7,
                       fps_target=20.0):
    """Models ranked BEST (top) to WORST (bottom) by a balanced score.
    Columns are the four objectives plus the score itself. Every objective is
    first mapped to [0,1] with 1 = best in this set; the score is their
    equal-weight mean. Rows are then sorted by that score, so the reading
    order of the figure IS the ranking."""
    rows = list(pts)
    ab = np.array([d["AbsRel"] for d in rows])
    Fps = np.array([d["Fps"] for d in rows])
    cv = np.array([d.get("cv_pct", np.nan) for d in rows])
    sb = np.array([abs(d["scale"] - 1.0) * 100.0 for d in rows])
    nab, ncv, nsb = (norm_better_low(ab), norm_better_low(cv),
                     norm_better_low(sb))
    nfps = (np.clip(Fps / float(fps_target), 0, 1) if fps_target
            else norm_better_high(Fps))
    # SELECTION RULE: L2 distance to ideal, rescaled so higher = better.
    score = score_from_ideal([nab, nfps, ncv, nsb])
    idx = np.argsort(-score)                     # best first
    rows = [rows[i] for i in idx]
    cols = ["AbsRel", "FPS", "Stability CV (%)", "|Scale bias| (%)",
            "SCORE\n(1 - D/2)"]
    norm = np.column_stack([nab[idx], nfps[idx], ncv[idx], nsb[idx],
                            score[idx]])
    raw = [[f"{ab[i]:.3f}", f"{Fps[i]:.1f}", f"{cv[i]:.2f}", f"{sb[i]:.1f}",
            f"{score[i]:.2f}"] for i in idx]
    im = ax.imshow(norm, cmap=resolve_cmap(cmap_name), vmin=0, vmax=1,
                   aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, raw[i][j], ha="center", va="center",
                    fontsize=fontsize + (1 if j == 4 else 0),
                    fontweight="bold" if j == 4 else "normal", color="black")
    ax.axvline(3.5, color="white", lw=3)          # separate the score column
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=fontsize + 2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{k+1}. {short_name(d)}"
                        for k, d in enumerate(rows)], fontsize=fontsize + 1)
    print("[rank] " + " > ".join(short_name(d) for d in rows[:5]) + " > ...")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="radar.png")
    ap.add_argument("--best-per-model", action="store_true",
                    help="keep only the lowest-AbsRel config per (model,size)")
    ap.add_argument("--best-per-family", action="store_true",
                    help="keep ONE variant per family (lowest AbsRel); use "
                         "for the overall radar and the composite's bars")
    ap.add_argument("--with-params", action="store_true",
                    help="include the Compactness axis (needs PARAMS_M)")
    ap.add_argument("--style", default="auto",
                    choices=["auto", "radar", "grid", "heatmap",
                             "parallel", "bags", "bags-grid",
                             "scatter", "size", "pareto", "pareto3", "scalebar",
                             "stability", "composite", "depth-trace",
                             "dist-heatmap", "consistency",
                             "pareto4", "pareto2", "pareto5", "pareto6",
                             "rank-heatmap"],
                    help="auto: radar if <=6 models, else heatmap. "
                         "grid = small-multiple radars (one per model). "
                         "heatmap = models x metrics, best for many models. "
                         "parallel = parallel-coordinates lines. "
                         "bags = one metric vs distance, a line per model. "
                         "scatter = AbsRel vs #params (size). "
                         "pareto = speed-accuracy frontier. "
                         "scalebar = signed scale-bias bars. "
                         "stability = frame-to-frame depth boxplot. "
                         "composite = 2x2 of scatter/pareto/scale/stability.")
    ap.add_argument("--font", default="default",
                    choices=["default", "serif", "times", "sans", "cm"],
                    help="figure font. 'serif'/'times' match a LaTeX body "
                         "text; 'cm' uses Computer Modern (pdflatex default); "
                         "'default' leaves matplotlib alone.")
    ap.add_argument("--font-size", type=float, default=None,
                    help="base font size in pt; match your paper's caption "
                         "size (often 8-9pt for two-column)")
    ap.add_argument("--cmap", default="soft",
                    help="heatmap colormap. Soft presets: soft (default, "
                         "rose->cream->sage), soft-warm, soft-blue, "
                         "soft-purple, soft-gray. Or any matplotlib name "
                         "(e.g. RdYlGn, PuBuGn).")
    ap.add_argument("--sort", default="speed",
                    choices=["speed", "accuracy", "scale", "jitter", "params",
                             "name"],
                    help="row/legend order, best first. Default 'speed' "
                         "matches the leading axis; 'accuracy' sorts by AbsRel.")
    ap.add_argument("--top", type=int, default=None,
                    help="keep only the N best by AbsRel (useful for radar)")
    ap.add_argument("--bag", nargs="*", default=None,
                    help="keep only these bags/distances, e.g. --bag 1.5m 1m. "
                         "Omit to load all; labels then carry '@<bag>'.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="substring filter on the model name, "
                         "e.g. --models metric3d moge")
    ap.add_argument("--mode", choices=["pinhole", "equi", "both"],
                    default="pinhole",
                    help="which evaluation to plot. 'pinhole' (default) = the "
                         "60 deg virtual-camera window in z-depth, which every "
                         "model has. 'equi' = the panoramic runs, scored over "
                         "the Kinect's full ~120 deg coverage in RADIAL "
                         "metres; only the camera-agnostic models have these. "
                         "'both' overlays them and is for the input-regime "
                         "ablation ONLY -- see the warning it prints.")
    ap.add_argument("--metric", default="AbsRel",
                    choices=["AbsRel", "scale", "Fps", "jitter", "drift",
                             "RMSE_m", "MAE_m"],
                    help="which metric the 'bags' style plots vs distance")
    ap.add_argument("--metrics", nargs="*", default=None,
                    choices=["AbsRel", "scale", "Fps", "jitter", "drift",
                             "RMSE_m", "MAE_m"],
                    help="metrics for 'bags-grid', one subplot each "
                         "(default: AbsRel scale jitter).")
    ap.add_argument("--ncols", type=int, default=2,
                    help="columns in the 'bags-grid' layout (default 2, so "
                         "4 metrics -> 2x2, 3 metrics -> 2x2 with one blank).")
    ap.add_argument("--legend", default="auto",
                    choices=["auto", "bottom", "cell", "right"],
                    help="bags-grid legend placement: 'auto' (bottom row -- "
                         "cheapest on a fixed-width double-column page), "
                         "'bottom' (one wide row beneath the panels, wraps "
                         "at 5 entries/row), 'cell' (force into a leftover "
                         "blank grid cell), 'right' (center-right beside "
                         "the panels -- widens the figure).")
    ap.add_argument("--panel-titles", nargs="*", default=None,
                    help="titles drawn on the bags-grid panels, in order, e.g. "
                         "--panel-titles '(a) Accuracy vs distance' "
                         "'(b) Temporal stability' '(c) Speed'")
    ap.add_argument("--caption", default=None,
                    help="a caption paragraph drawn (wrapped) below the "
                         "bags-grid figure")
    ap.add_argument("--hm-metrics", nargs="*", default=None,
                    choices=["AbsRel", "RMSE_m", "MAE_m", "cv_pct",
                             "stab_pct", "Fps", "scale_bias"],
                    help="metric groups for 'dist-heatmap', in order "
                         "(default: AbsRel RMSE_m MAE_m cv_pct Fps "
                         "scale_bias). Unlike the standalone 'heatmap' "
                         "style, RMSE_m/MAE_m ARE scored/ranked columns "
                         "here -- dist-heatmap never combines columns into "
                         "one score, so there is no double-counting risk.")
    ap.add_argument("--fig-width", type=float, default=None,
                    help="dist-heatmap: draw the figure at its PRINTED width "
                         "in inches (7.16 = IEEE two-column \\textwidth, "
                         "3.5 = one column), so --font-size is the true "
                         "printed size. Omit for the old, larger canvas.")
    ap.add_argument("--row-height", type=float, default=0.17,
                    help="dist-heatmap with --fig-width: height of one model "
                         "row in inches (default 0.17).")
    ap.add_argument("--fps-target", type=float, default=20.0,
                    help="FPS at/above which the dist-heatmap FPS cells are "
                         "fully green (absolute threshold, default 20). "
                         "Set 0 to fall back to relative min-max colouring.")
    ap.add_argument("--all-configs", action="store_true",
                    help="compute the Pareto frontier over EVERY tested "
                         "configuration, not just the lowest-AbsRel config "
                         "per (model,variant). Use this when you have swept "
                         "settings (token count, precision) that trade speed "
                         "or stability against accuracy -- pre-filtering by "
                         "AbsRel can discard genuinely Pareto-optimal points.")
    ap.add_argument("--no-stars", action="store_true",
                    help="hide the equal-weight top-3 rings on pareto3/"
                         "pareto4, leaving the frontier alone")
    ap.add_argument("--no-params", action="store_true",
                    help="drop the Params column from 'dist-heatmap'")
    ap.add_argument("--no-grid", action="store_true",
                    help="remove grid lines from all panels")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="skip CSVs with fewer than this many frames")
    ap.add_argument("--weights", nargs=3, type=float, default=None,
                    metavar=("W_ABSREL", "W_FPS", "W_CV"),
                    help="weights for the pareto3 L2-to-ideal pick, in the "
                         "order AbsRel FPS CV. Equal by default. e.g. "
                         "--weights 1 1 3 prioritises low CV. Affects only "
                         "the starred top-3, never the frontier.")
    ap.add_argument("--title", default=None,
                    help="override the figure suptitle/title")
    args = ap.parse_args()

    # ---- figure font ----
    # Matplotlib's default (DejaVu Sans) reads as foreign next to LaTeX body
    # text. These presets approximate the common document fonts; the figure
    # is still rendered by matplotlib, not by LaTeX, so glyphs are close but
    # not identical unless you enable usetex (which needs a local TeX).
    if args.font != "default":
        fam = {
            "serif": ("serif", ["DejaVu Serif", "Times New Roman", "Nimbus Roman"]),
            "times": ("serif", ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]),
            "cm":    ("serif", ["CMU Serif", "Computer Modern Roman",
                                "DejaVu Serif"]),
            "sans":  ("sans-serif", ["Helvetica", "Arial", "DejaVu Sans"]),
        }[args.font]
        matplotlib.rcParams["font.family"] = fam[0]
        matplotlib.rcParams[f"font.{fam[0]}"] = fam[1]
        matplotlib.rcParams["mathtext.fontset"] = (
            "cm" if fam[0] == "serif" else "dejavusans")
        print(f"[font] {args.font} -> {fam[0]} {fam[1][0]}")
    if args.font_size:
        matplotlib.rcParams["font.size"] = args.font_size

    paths = sorted(glob.glob(os.path.join(args.dir, "eval_*.csv")))
    pts = [d for d in (load_csv(p) for p in paths) if d is not None]
    pts = [d for d in pts if d["n"] >= args.min_frames]

    if args.bag:
        want = {b.lower() for b in args.bag}
        pts = [d for d in pts if d["bag"].lower() in want]
    if args.models:
        # Match against BOTH the raw filename config tag (e.g.
        # "metric3d_1m_giant2") AND the clean "model-variant" form used
        # everywhere else in this script's own output (short_name(), e.g.
        # "metric3d-giant2"). Matching only the raw tag means a token typed
        # in the display convention -- hyphenated -- can never match, since
        # the raw tag is underscore-joined; that mismatch silently filtered
        # out every row for a hyphenated token instead of erroring, which is
        # worse than a crash because nothing signals the filter did nothing.
        tokens = [t.lower() for t in args.models]

        def _matches(d):
            clean = f"{d['model']}-{d['variant']}" if d['variant'] else d['model']
            hay = f"{d['config']} {clean}".lower()
            return any(t in hay for t in tokens)

        before = len(pts)
        pts = [d for d in pts if _matches(d)]
        if not pts:
            print(f"[warn] --models {args.models} matched 0 of {before} rows "
                  f"-- check spelling; run without --models to see the full "
                  f"list of model-variant names actually present.")
    if args.mode != "both":
        pts = [d for d in pts if d.get("mode", "pinhole") == args.mode]

    if not pts:
        raise SystemExit(f"no usable CSVs in {args.dir} after filtering "
                         f"(mode={args.mode}; equirect files are named "
                         f"eval_<model>-equi_..., pinhole eval_<model>_...)")

    global SHOW_MODE
    SHOW_MODE = len({d.get("mode", "pinhole") for d in pts}) > 1
    if SHOW_MODE:
        print("\n[warn] this figure MIXES pinhole and equirect evaluations.")
        print("  pinhole -> 60 deg window,            z-depth metres")
        print("  equi    -> Kinect's full ~120 deg,   RADIAL metres")
        print("  AbsRel / delta1 / scale are ratios and survive the z-vs-")
        print("  radial difference, but they are still measured over "
              "DIFFERENT")
        print("  REGIONS. RMSE_m and MAE_m are not comparable at all.")
        print("  Use --mode pinhole or --mode equi for any figure that RANKS")
        print("  models; keep 'both' for the input-regime ablation.")

    # Preserve the full (all-variant) set BEFORE any best-per collapse, so the
    # scatter / Pareto / composite panels can always show every size while the
    # radar / bars use a collapsed set.
    pts_full = list(pts)

    # Only tag labels with the bag when more than one distance is loaded,
    # otherwise every label carries a redundant "@1.5m".
    global SHOW_BAG
    bags = sorted({d["bag"] for d in pts if d["bag"]},
                  key=lambda b: float(b[:-1]))
    SHOW_BAG = len(bags) > 1
    if bags:
        print(f"[bags] {len(bags)} distance(s): {', '.join(bags)}")
        if SHOW_BAG:
            print("  NOTE: RMSE/MAE are in metres and are NOT comparable "
                  "across distances\n  (a 5 cm error is 10% at 0.5 m but "
                  "3.3% at 1.5 m). Compare AbsRel,\n  scale and jitter "
                  "across bags; keep RMSE/MAE within a bag.")

    if args.best_per_family:
        pts = best_per_family(pts)
    elif args.best_per_model:
        best = {}
        for d in pts:
            # keep each SIZE separately: moge-vits and moge-vitl are different
            # models for comparison purposes, not two configs of one model.
            key = (d["model"], d["variant"])
            if key not in best or d["AbsRel"] < best[key]["AbsRel"]:
                best[key] = d
        pts = list(best.values())

    # Row order is independent of axis order. Sorted BEST FIRST on the chosen
    # key; --top then keeps the leading N *of that ordering*, so --sort speed
    # --top 6 gives the six fastest, not the six most accurate.
    sort_keys = {
        "speed":    lambda d: -d["Fps"],              # higher Fps first
        "accuracy": lambda d: d["AbsRel"],            # lower error first
        "scale":    lambda d: abs(d["scale"] - 1.0),  # closer to 1.0 first
        "jitter":   lambda d: (np.inf if not np.isfinite(d["jitter"])
                               else d["jitter"]),     # steadier first
        "params":   lambda d: (np.inf if not d.get("params_M")
                               else d["params_M"]),   # smaller first
    }
    pts.sort(key=sort_keys[args.sort])
    if args.top:
        pts = pts[:args.top]
    print(f"[sort] rows ordered by {args.sort} (best first)")

    style = args.style
    NEWSTYLES = {"scatter", "size", "pareto", "pareto3", "scalebar",
                 "stability", "composite", "depth-trace"}
    if style in NEWSTYLES and SHOW_BAG:
        print(f"\n[warn] style '{style}' with {len(bags)} distances loaded: "
              f"the axes will MIX distances,\n  and each model may be drawn "
              f"at whichever distance scored best. Pin a single\n  --bag "
              f"(e.g. --bag {bags[-1]}) for a clean, comparable figure.")

    # ---- build the radar/heatmap axes (all oriented outward = better) ----
    acc = norm_better_low([d["AbsRel"] for d in pts])
    sca = norm_better_low([abs(d["scale"] - 1.0) for d in pts])
    # Speed as FPS (higher = better). NOTE this is not just a relabel:
    # Fps = 1000/latency is nonlinear, so min-max scaling on Fps spaces
    # the fast models further apart and compresses the slow ones,
    # unlike scaling on latency. The ordering is identical either way.
    spd = norm_better_high([d["Fps"] for d in pts])
    jit_raw = [d["jitter"] for d in pts]
    if np.all(np.isnan(jit_raw)):
        # no p50 columns -> fall back to the error-fluctuation proxy
        stb = norm_better_low([d["AbsRel_std"] for d in pts])
        stb_label = "Stability\n(AbsRel std)"
    else:
        filled = np.where(np.isnan(jit_raw), np.nanmax(jit_raw), jit_raw)
        stb = norm_better_low(filled)
        stb_label = "Stability\n(CV of depth, %)"
    # Axis order: Speed -> Accuracy -> Steadiness -> Scale -> [Compactness].
    # Keep labels/data/raws (heatmap annotations) in exactly this order.
    labels = ["Speed\n(FPS)", "Accuracy\n(AbsRel)", stb_label,
              "Scale\n(|s-1|)"]
    data = [spd, acc, stb, sca]
    if args.with_params and any(d["params_M"] for d in pts):
        pv = [d["params_M"] if d["params_M"] else np.nan for d in pts]
        if not np.all(np.isnan(pv)):
            filled = np.where(np.isnan(pv), np.nanmax(pv), pv)
            data.append(norm_better_low(filled))
            labels.append("Compactness\n(params)")

    # ---- print the underlying table so the chart is auditable ----
    print(f"{'label':>22} {'params':>8} {'AbsRel':>8} {'RMSE_m':>8} "
          f"{'MAE_m':>8} {'scale':>7} {'CV(%)':>8} {'sigma(mm)':>10} "
          f"{'step(mm)':>9} {'lat(ms)':>8} {'Fps':>6} {'n':>4}")
    for d in pts:
        pm = f"{d['params_M']:.1f}M" if d.get("params_M") else "-"
        sm = d.get("cv_m", float("nan")) * 1000.0
        jm = d.get("step_m", float("nan")) * 1000.0
        print(f"{label_of(d):>22} {pm:>8} {d['AbsRel']:>8.4f} "
              f"{d['RMSE_m']:>8.4f} {d['MAE_m']:>8.4f} "
              f"{d['scale']:>7.3f} {d['cv_pct']:>8.3f} {sm:>10.2f} "
              f"{jm:>9.2f} "
              f"{d['latency_ms']:>8.1f} {d['Fps']:>6.1f} {d['n']:>4}")
    if _PARAM_MISSES:
        print(f"\n  [params] no entry for: {', '.join(sorted(_PARAM_MISSES))}"
              f"\n  add them to PARAMS_M at the top of this script "
              f"(the eval prints the count at startup).")
    print("\n  RMSE_m/MAE_m are in metres and reported in the table (and, on"
          "\n  the 'heatmap' style, as two reference-only columns) -- they"
          "\n  correlate strongly with AbsRel, so they are deliberately NOT"
          "\n  folded into any ranked/coloured axis: doing so would count"
          "\n  accuracy several times over against the genuinely independent"
          "\n  axes (speed, stability, scale). RMSE >> MAE indicates a"
          "\n  heavy-tailed error distribution.")
    print("\n  TEMPORAL STABILITY = COEFFICIENT OF VARIATION (CV) of the")
    print("    per-frame MEDIAN PREDICTED DEPTH over the run:")
    print("        CV = 100 * sigma / mu   [%]   (sigma = sample std, ddof=1)")
    print("    NO ground truth is used: the scenes are static, so the true")
    print("    depth does not change and any spread is model instability.")
    print("    e.g. CV 0.44 = the depth reading varies by 0.44% of its mean")
    print("    (sigma ~ 4.4 mm on a 1 m scene). Lower = steadier; 0 = a")
    print("    perfectly constant prediction, as the Azure Kinect should give.")
    print("    Dividing by mu is what makes the value comparable across the")
    print("    0.51 / 1 / 1.5 m bags. NOTE mu is the model's OWN mean")
    print("    predicted depth, not the nominal distance.")
    print("  step(mm) (table only) = mean |frame-to-frame change| in mm,")
    print("    i.e. per-frame movement rather than overall spread.")
    print("  The keys 'jitter' and 'drift' are ALIASES of CV(%) so older")
    print("    commands (--sort jitter, --metric jitter) still work.")
    print("  NOTE: this is a GLOBAL (median-depth) measure. A model can hold a"
          "\n  steady median while individual pixels flicker; per-pixel"
          "\n  stability would need the saved depth maps.")

    # ---- draw ----
    if style == "auto":
        style = "radar" if len(pts) <= 6 else "heatmap"
        if style != args.style:
            print(f"\n[style] {len(pts)} models -> using '{style}' "
                  f"(radar becomes unreadable past ~6; override with --style)")

    K = len(labels)
    short_labels = [l.replace("\n", " ") for l in labels]
    M = np.array(data)                 # (K axes, N models)

    if style == "heatmap":
        # models x metrics grid. Scales to any number of models and the
        # values stay individually readable, unlike overlapping polygons.
        #
        # RMSE_m and MAE_m are appended as TWO EXTRA, UNSCORED columns, not
        # as additional ranked axes. All three of AbsRel, RMSE_m and MAE_m
        # summarise the SAME per-pixel error; folding all of them into the
        # scored grid would count accuracy two or three times against the
        # genuinely independent axes (speed, stability, scale bias) -- see
        # the console note above. They render on a flat neutral grey (a
        # masked array + cmap.set_bad, not part of the 0-1 colour scale) and
        # sit behind a divider, the same "extra info, not a vote" treatment
        # the Params column gets in panel_dist_heatmap.
        #
        # The raw-value text for the SCORED columns is looked up by the
        # column's own label (raw_for[...]) instead of a second, independently
        # -ordered list. A hardcoded parallel list is exactly what caused the
        # previous version of this block to silently show RMSE/MAE numbers
        # under the Stability/Scale columns' colours -- see the comment this
        # replaces if you are diffing against an older copy.
        n_extra = 2
        Kx = K + n_extra
        M_ext = np.vstack([M, np.full((n_extra, len(pts)), np.nan)])
        cmap = resolve_cmap(args.cmap)
        try:
            cmap = cmap.copy()
        except AttributeError:
            pass   # older matplotlib: falls back to mutating the shared cmap
        cmap.set_bad(color="#ececec")

        fig, ax = plt.subplots(figsize=(1.6 * Kx + 3.5, 0.42 * len(pts) + 2.2))
        im = ax.imshow(np.ma.masked_invalid(M_ext.T), cmap=cmap, vmin=0,
                       vmax=1, aspect="auto")
        ax.set_xticks(range(Kx))
        ax.set_xticklabels(short_labels + ["RMSE (m)", "MAE (m)"], fontsize=9)
        ax.set_yticks(range(len(pts)))
        ax.set_yticklabels([label_of(d) for d in pts], fontsize=8)

        # scored columns: raw text keyed off the SAME label driving that
        # column's colour, so the two cannot drift apart again
        raw_for = {
            "Speed":       lambda d: f"{d['Fps']:.1f}",
            "Accuracy":    lambda d: f"{d['AbsRel']:.3f}",
            "Stability":   lambda d: ("-" if not np.isfinite(d["jitter"])
                                      else f"{d['jitter']:.2f}"),
            "Scale":       lambda d: f"{d['scale']:.2f}",
            "Compactness": lambda d: ("-" if not d.get("params_M")
                                      else f"{d['params_M']:.0f}"),
        }
        for j, lab in enumerate(labels):
            fn = raw_for[lab.split("\n")[0]]
            for i, d in enumerate(pts):
                ax.text(j, i, fn(d), ha="center", va="center",
                        fontsize=7, color="black")
        # extra, unscored reference columns
        for i, d in enumerate(pts):
            ax.text(K, i, f"{d['RMSE_m']:.3f}", ha="center", va="center",
                    fontsize=7, color="#444444")
            ax.text(K + 1, i, f"{d['MAE_m']:.3f}", ha="center", va="center",
                    fontsize=7, color="#444444")

        # divider marking the reference-only columns as separate from the
        # scored ones
        ax.axvline(K - 0.5, color="white", lw=3, zorder=5)

        # Outline the best cell in each SCORED column only (M[j] is already
        # normalised so 1.0 = best; the two reference columns have no
        # "winner" box since they are not ranked here).
        for j in range(K):
            col = np.asarray(M[j], dtype=float)
            if not np.any(np.isfinite(col)):
                continue
            i_best = int(np.nanargmax(col))
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i_best - 0.5), 1, 1,
                fill=False, edgecolor="#3a3a3a", linewidth=1.6, zorder=6))

        ax.set_title(args.title or "", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.6,
                     label="normalised (1 = best)\n(grey = reference only)")
        fig.tight_layout()

    elif style == "bags":
        # INTRA-MODEL view: one metric against distance, a line per model.
        lc = build_line_colors(pts_full)
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        h, l, nseen = panel_bags(ax, pts_full, args.metric, lc)
        if nseen == 0:
            raise SystemExit("style 'bags' needs the same model on >=2 bags; "
                             "none found (check the CSV names carry the "
                             "distance, e.g. eval_<model>_1.5m_...)")
        ax.set_title(args.title or "", fontsize=11)
        ax.legend(h, l, fontsize=7, ncol=1 if nseen <= 10 else 2,
                  bbox_to_anchor=(1.01, 1.0), loc="upper left")
        fig.tight_layout()

    elif style == "bags-grid":
        # Several metrics vs distance in one figure, one subplot each, sharing
        # a single legend. This is the "all distances in one figure" view.
        #
        # Sizing is tuned for a DOUBLE-COLUMN paper: 3.4in/panel width means
        # a 2-panel row lands close to IEEEtran's full \textwidth (~7.16in);
        # go to --ncols 1 for single-column-width panels if space is even
        # tighter. The old 4.9in/panel default was print-unusable at more
        # than one panel per row.
        keys = args.metrics or ["AbsRel", "jitter", "Fps"]
        lc = build_line_colors(pts_full)
        ncol = max(1, args.ncols)
        nrow = int(np.ceil(len(keys) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(3.4 * ncol, 5.6* nrow),
                                 squeeze=False)
        axes = axes.ravel()
        hl = {}
        seen_any = 0
        ptitles = args.panel_titles or []
        for i, key in enumerate(keys):
            h, l, ns = panel_bags(axes[i], pts_full, key, lc)
            if i < len(ptitles):
                axes[i].text(0.5, -0.18, ptitles[i],
                             transform=axes[i].transAxes,
                             ha="center", va="top", fontsize=12)
            seen_any = max(seen_any, ns)
            for hh, ll in zip(h, l):          # merge legend entries by label
                hl.setdefault(ll, hh)
        if seen_any == 0:
            raise SystemExit("style 'bags-grid' needs the same model on >=2 "
                             "bags; none found. Do NOT pin a single --bag, and "
                             "check the CSV names carry the distance.")
        empties = list(range(len(keys), len(axes)))
        place = args.legend
        # "bottom" is now the auto fallback: in a double-column paper, width
        # is the scarce resource (fixed by the column), height is the one
        # you can still spend a little of. A legend beside the panels
        # ("right") or crammed into a leftover cell ("cell") both eat into
        # that fixed width; a row below the panels does not.
        if place == "auto":
            place = "bottom"
        if place == "cell" and not empties:
            place = "bottom"                     # no blank cell to use
        if place == "cell":
            # drop the shared legend into the first blank cell (e.g. the
            # bottom-right panel when 3 metrics fill a 2x2 grid)
            lax = axes[empties[0]]
            lax.axis("off")
            lax.legend(list(hl.values()), list(hl.keys()), loc="center",
                       fontsize=12, ncol=1 if len(hl) <= 9 else 2,
                       frameon=False)
            for j in empties[1:]:
                axes[j].axis("off")
            if args.title:
                fig.suptitle(args.title, fontsize=12)
            fig.tight_layout(h_pad=2.0)
        elif place == "right":
            # center-right, beside the panels (blank cells left empty) --
            # widens the figure; use only if height is the tighter budget.
            for j in empties:
                axes[j].axis("off")
            fig.legend(list(hl.values()), list(hl.keys()), fontsize=12,
                       loc="center left", bbox_to_anchor=(1.0, 0.5),
                       ncol=1 if len(hl) <= 12 else 2)
            if args.title:
                fig.suptitle(args.title, fontsize=11)
            fig.tight_layout(rect=(0, 0, 0.82, 1), h_pad=3.0)
        else:  # "bottom": legend left-aligned under panel (a)
            for j in empties:
                axes[j].axis("off")
            ncol_legend = min(len(hl), 4)
            n_legend_rows = int(np.ceil(len(hl) / ncol_legend))
            bottom = 0.03 + 0.025 * n_legend_rows
            if args.title:
                fig.suptitle(args.title, fontsize=11)
            fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=2.6)
            # panel (a)'s left edge, read AFTER layout settles -- anchoring
            # to a live coordinate instead of a guessed bbox_to_anchor x
            x0 = axes[0].get_position().x0
            fig.legend(list(hl.values()), list(hl.keys()), fontsize=8,
                       loc="lower left", ncol=ncol_legend, frameon=False,
                       bbox_to_anchor=(x0, 0.0))
            if args.title:
                fig.suptitle(args.title, fontsize=11)
            fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=2.6)
        if args.caption:
            wrapped = "\n".join(textwrap.wrap(
                args.caption, width=int(fig.get_figwidth() * 12)))
            fig.text(0.5, -0.01, wrapped, ha="center", va="top", fontsize=14)

    elif style == "parallel":
        # one line per model across the axes; handles many series far better
        # than radar because nothing is hidden behind a filled polygon.
        fig, ax = plt.subplots(figsize=(1.9 * K + 3, 5.4))
        cmap = plt.get_cmap("turbo")
        for i, d in enumerate(pts):
            ax.plot(range(K), M[:, i], "-o", markersize=4, linewidth=1.6,
                    color=cmap(i / max(len(pts) - 1, 1)),
                    label=label_of(d), alpha=0.85)
        ax.set_xticks(range(K))
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel("normalised (1 = best)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title(args.title or "", fontsize=11)
        ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1.0),
                  loc="upper left")
        fig.tight_layout()

    elif style == "grid":
        # small multiples: one little radar per model, identical axes.
        ncol = min(5, len(pts))
        nrow = int(np.ceil(len(pts) / ncol))
        angles = np.linspace(0, 2 * np.pi, K, endpoint=False).tolist()
        angles += angles[:1]
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.8 * nrow),
                                 subplot_kw=dict(polar=True))
        axes = np.atleast_1d(axes).ravel()
        cmap = plt.get_cmap("tab20")
        for i, d in enumerate(pts):
            a = axes[i]
            vals = list(M[:, i]) + [M[0, i]]
            a.plot(angles, vals, "-", linewidth=1.8, color=cmap(i % 20))
            a.fill(angles, vals, color=cmap(i % 20), alpha=0.25)
            a.set_xticks(angles[:-1])
            a.set_xticklabels([l.split("\n")[0] for l in labels], fontsize=6)
            a.set_yticklabels([])
            a.set_ylim(0, 1.05)
            a.set_title(f"{label_of(d)}\n{d['AbsRel']:.3f}", fontsize=7, pad=8)
            a.grid(alpha=0.3)
        for j in range(len(pts), len(axes)):
            axes[j].axis("off")
        if args.title:
            fig.suptitle(args.title, fontsize=12)
        fig.tight_layout()

    elif style == "scatter":
        # Fig 2: accuracy vs #params. Show every variant (needs multiple
        # sizes per family to read as a trend), collapsing duplicate configs.
        S = best_per_variant(pts_full)
        print(f"[scatter] {len(S)} variants across "
              f"{len({d['model'] for d in S})} families")
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        panel_accuracy_size(ax, S, title=args.title or "")
        fig.tight_layout()

    elif style == "size":
        # "Does bigger help?" -- three panels, ALL against parameter count
        # (log x): temporal stability, accuracy, throughput. One connected
        # line per family; a family that benefits from scale traces a
        # FALLING line on (a) and (b). Panel (c) is throughput, and for
        # every family measured in this benchmark so far, bigger has meant
        # SLOWER -- a rising line on (c) alongside falling lines on (a)/(b)
        # would be the expected, unremarkable shape. If (c) instead falls
        # alongside (a)/(b) (bigger = faster too), that is surprising enough
        # to double-check before reporting it: look for a warm-up frame
        # still in the latency median, a GPU shared with another process
        # during the small-model run, or a resolution/precision difference
        # between the configs being compared.
        S = best_per_variant(pts_full)
        print(f"[size] {len(S)} variants across "
              f"{len({d['model'] for d in S})} families")
        fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.8))
        panel_metric_vs_size(axs[0], S, "jitter",
                             "Temporal stability, CV (%)", lower_better=True,
                             title="(a) Stability vs size")
        panel_metric_vs_size(axs[1], S, "AbsRel", "AbsRel",
                             lower_better=True, title="(b) Accuracy vs size")
        panel_metric_vs_size(axs[2], S, "Fps", "Throughput (FPS)",
                             lower_better=False, title="(c) Speed vs size")
        if args.title:
            fig.suptitle(args.title, fontsize=13, y=1.03)
        fig.tight_layout()

    elif style == "pareto":
        # Fig 3: speed-accuracy frontier.
        S = pts_full if args.all_configs else best_per_variant(pts_full)
        print(f"[pareto] {len(S)} variants")
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        panel_pareto(ax, S, title=args.title or "")
        fig.tight_layout()

    elif style == "pareto3":
        # 3-objective frontier: speed x accuracy x steadiness (bubble size).
        S = best_per_variant(pts_full)
        print(f"[pareto3] {len(S)} variants")
        fig, ax = plt.subplots(figsize=(9.8, 6.0))
        mh, fh = panel_pareto3(ax, S, title=args.title or "",
                               show_stars=not args.no_stars,
                               weights=args.weights)
        if fh:
            fig.legend(fh, [h.get_label() for h in fh], title="family",
                       loc="upper left", bbox_to_anchor=(0.775, 0.96),
                       fontsize=7, title_fontsize=7, framealpha=0.9)
        if mh:
            fig.legend(mh, [h.get_label() for h in mh],
                       title="bubble = $\\mathrm{{CV_f}}$", loc="lower left",
                       bbox_to_anchor=(0.775, 0.04), fontsize=7,
                       title_fontsize=7, framealpha=0.9, labelspacing=1.5)
        fig.tight_layout(rect=(0, 0, 0.76, 1))

    elif style == "pareto4":
        # 4-objective frontier: speed x accuracy x stability x scale bias.
        S = pts_full if args.all_configs else best_per_variant(pts_full)
        print(f"[pareto4] {len(S)} variants")
        fig, ax = plt.subplots(figsize=(10.2, 6.0))
        mh, fh = panel_pareto4(ax, S, title=args.title or "",
                               show_stars=not args.no_stars)
        if fh:
            fig.legend(fh, [h.get_label() for h in fh], title="family",
                       loc="upper left", bbox_to_anchor=(0.75, 0.97),
                       fontsize=7, title_fontsize=7, framealpha=0.9)
        if mh:
            fig.legend(mh, [h.get_label() for h in mh], title="encoding",
                       loc="lower left", bbox_to_anchor=(0.75, 0.03),
                       fontsize=7, title_fontsize=7, framealpha=0.9,
                       labelspacing=1.4)
        fig.tight_layout(rect=(0, 0, 0.74, 1))

    elif style == "pareto5":
        # 2-objective QUALITY frontier: accuracy vs temporal stability.
        # Deliberately excludes FPS -- the counterpart to pareto/3/4.
        S = pts_full if args.all_configs else best_per_variant(pts_full)
        print(f"[pareto5] {len(S)} entries")
        fig, ax = plt.subplots(figsize=(8.4, 5.8))
        fh = panel_pareto5(ax, S, title=args.title or "",
                           show_stars=not args.no_stars)
        if fh:
            fig.legend(fh, [h.get_label() for h in fh], title="family",
                       loc="center left", bbox_to_anchor=(0.79, 0.5),
                       fontsize=7, title_fontsize=7, framealpha=0.9)
        fig.tight_layout(rect=(0, 0, 0.78, 1))

    elif style == "pareto6":
        # 3-objective QUALITY frontier: accuracy x stability x scale bias.
        # Deliberately excludes FPS.
        S = pts_full if args.all_configs else best_per_variant(pts_full)
        print(f"[pareto6] {len(S)} entries")
        fig, ax = plt.subplots(figsize=(9.6, 6.0))
        mh, fh = panel_pareto6(ax, S, title=args.title or "",
                               show_stars=not args.no_stars)
        if fh:
            fig.legend(fh, [h.get_label() for h in fh], title="family",
                       loc="upper left", bbox_to_anchor=(0.77, 0.96),
                       fontsize=7, title_fontsize=7, framealpha=0.9)
        if mh:
            fig.legend(mh, [h.get_label() for h in mh],
                       title="bubble = Temporal\nStability (CV %)",
                       loc="lower left", bbox_to_anchor=(0.77, 0.04),
                       fontsize=7, title_fontsize=7, framealpha=0.9,
                       labelspacing=1.4)
        fig.tight_layout(rect=(0, 0, 0.76, 1))

    elif style == "pareto2":
        # 2-objective frontier: speed x temporal stability (no accuracy).
        S = pts_full if args.all_configs else best_per_variant(pts_full)
        print(f"[pareto2] {len(S)} entries")
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        fh = panel_pareto2(ax, S, title=args.title or "",
                           show_stars=not args.no_stars)
        if fh:
            fig.legend(fh, [h.get_label() for h in fh], title="family",
                       loc="center left", bbox_to_anchor=(0.79, 0.5),
                       fontsize=7, title_fontsize=7, framealpha=0.9)
        fig.tight_layout(rect=(0, 0, 0.78, 1))

    elif style == "rank-heatmap":
        # models ranked best -> worst by the balanced score
        S = best_per_variant(pts_full)
        fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(S) + 2.0))
        im = panel_rank_heatmap(fig, ax, S, args.cmap,
                                fps_target=args.fps_target)
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02,
                     label="normalised (1 = best)")
        if args.title:
            fig.suptitle(args.title, fontsize=12)
        fig.tight_layout()

    elif style == "scalebar":
        # Fig 4: signed scale bias. Use family-best or all-variants, ordered
        # so the least biased model is leftmost.
        S = best_per_family(pts_full) if args.best_per_family \
            else best_per_variant(pts_full)
        if args.top:
            S = sorted(S, key=lambda d: d["AbsRel"])[:args.top]
        S.sort(key=lambda d: abs(d["scale"] - 1.0))
        print(f"[scalebar] {len(S)} bars (least-biased first)")
        fig, ax = plt.subplots(figsize=(max(6.5, 0.6 * len(S) + 2), 4.8))
        panel_scale(ax, S, title=args.title or "")
        fig.tight_layout()

    elif style == "stability":
        # Fig 5: frame-to-frame depth boxplot. Ordered steadiest-first.
        S = best_per_family(pts_full) if args.best_per_family \
            else best_per_variant(pts_full)
        if args.top:
            S = sorted(S, key=lambda d: d["AbsRel"])[:args.top]
        S.sort(key=lambda d: (np.inf if not np.isfinite(d["jitter"])
                              else d["jitter"]))
        print(f"[stability] {len(S)} boxes (steadiest first)")
        fig, ax = plt.subplots(figsize=(max(6.5, 0.6 * len(S) + 2), 4.8))
        panel_stability(ax, S, title=args.title or "")
        fig.tight_layout()

    elif style == "composite":
        # 2x2 paper figure. (a),(b) show every variant; (c),(d) use one
        # representative per family, ordered by accuracy so the same model
        # sits in the same column position across both categorical panels.
        Sab = best_per_variant(pts_full)
        Scd = sorted(best_per_family(pts_full), key=lambda d: d["AbsRel"])
        print(f"[composite] scatter/pareto: {len(Sab)} variants | "
              f"scale/stability: {len(Scd)} families")
        fig, axs = plt.subplots(2, 2, figsize=(13, 10))
        panel_accuracy_size(axs[0, 0], Sab)
        panel_pareto(axs[0, 1], Sab)
        panel_scale(axs[1, 0], Scd)
        panel_stability(axs[1, 1], Scd)
        if args.title:
            fig.suptitle(args.title, fontsize=13, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
        else:
            fig.tight_layout()

    elif style == "dist-heatmap":
        # Benchmark-sheet layout: rows = models, columns = metric x distance.
        # Uses ALL bags, so do NOT pin --bag. Collapse duplicate configs
        # WITHIN each (model, variant, bag) so every distance survives.
        best_mvb = {}
        for d in pts_full:
            k = (d["model"], d["variant"], d["bag"])
            if k not in best_mvb or d["AbsRel"] < best_mvb[k]["AbsRel"]:
                best_mvb[k] = d
        S = list(best_mvb.values())
        mets = args.hm_metrics or ["AbsRel", "RMSE_m", "MAE_m", "cv_pct",
                                  "Fps", "scale_bias"]
        bgs = bags if bags else [""]
        nrows = len({(d["model"], d["variant"]) for d in S})
        print(f"[dist-heatmap] {nrows} models x {len(mets)} metrics "
              f"x {len(bgs)} distances")
        if args.fig_width:
            # PRINTED size: the PDF is placed at 100 %, so the font sizes
            # below are the real sizes on the page. The old canvas below was
            # ~15 in wide and got shrunk to half when placed in the paper,
            # which is why its 7 pt text printed at ~3.5 pt.
            figsize = (args.fig_width, args.row_height * nrows + 1.0)
        else:
            figsize = (1.15 * (len(mets) * len(bgs) + 1) + 3.2,
                       0.34 * nrows + 2.0)
        fig, ax = plt.subplots(figsize=figsize)
        im = panel_dist_heatmap(fig, ax, S, mets, bgs, args.cmap,
                                fontsize=args.font_size or 7,
                                with_params=not args.no_params,
                                fps_target=args.fps_target,
                                sort_by=args.sort)
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02,
                     label="normalised per column (1 = best)")
        if args.title:
            fig.suptitle(args.title, fontsize=12)
        fig.tight_layout()

    elif style == "consistency":
        # "Do models behave the same at every distance?" One row per model,
        # one column per metric, cell = spread (max-min) of that metric
        # across the distances it was run at. Companion to dist-heatmap,
        # which shows the raw per-distance values this collapses into a
        # single number. Uses ALL bags -- do NOT pin --bag, or every row
        # collapses to a single point with nothing to spread.
        if len(bags) < 2:
            print(f"\n[warn] style 'consistency' needs >=2 distances loaded "
                  f"to compute a spread; found {len(bags)}. Do not pass "
                  f"--bag, or pass several: --bag 0.51m 1m 1.5m")
        best_mvb = {}
        for d in pts_full:
            k = (d["model"], d["variant"], d["bag"])
            if k not in best_mvb or d["AbsRel"] < best_mvb[k]["AbsRel"]:
                best_mvb[k] = d
        S = list(best_mvb.values())
        nrows = len({(d["model"], d["variant"]) for d in S})
        print(f"[consistency] {nrows} models across {len(bags)} distances")
        fig, ax = plt.subplots(figsize=(8.6, 0.34 * nrows + 2.0))
        im = panel_consistency(ax, S, cmap_name=args.cmap)
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02,
                     label="normalised per column\n(1 = most consistent)")
        ax.annotate("* fewer than 2 distances -- no spread to report",
                    xy=(0, 1.01), xycoords="axes fraction", fontsize=7,
                    color="#b5473f")
        if args.title:
            fig.suptitle(args.title, fontsize=12)
        fig.tight_layout()

    elif style == "depth-trace":
        # Per-frame predicted median depth in raw metres over the whole run,

        # one line per model, with the Azure Kinect ground truth on top. Shows
        # the actual frame-to-frame depth shift that 'jitter' reduces to a
        # single number. Pin one --bag; filter with --models to compare a few.
        S = best_per_variant(pts_full)
        S = [d for d in S
             if np.isfinite(np.asarray(d["pred_p50"], dtype=float)).sum() >= 2]
        if not S:
            raise SystemExit("style 'depth-trace' needs per-frame "
                             "pred_p50_m/gt_p50_m columns in the CSVs.")
        lc = build_line_colors(pts_full)
        fig, ax = plt.subplots(figsize=(12, 4.6))
        # Kinect GT once (same recording -> same gt across models)
        ref = max(S, key=lambda d: np.isfinite(
            np.asarray(d["gt_p50"], dtype=float)).sum())
        gf = np.asarray(ref["frames"])
        gv = np.asarray(ref["gt_p50"], dtype=float)
        og = np.isfinite(gv)
        ax.plot(gf[og], gv[og], color="black", lw=2.4, zorder=6,
                label="Azure Kinect (GT)")
        for d in S:
            f = np.asarray(d["frames"])
            v = np.asarray(d["pred_p50"], dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            ax.plot(f[ok], v[ok], lw=1.1, alpha=0.9,
                    color=lc.get((d["model"], d["variant"]), "#333333"),
                    label=short_name(d))
        ax.set_xlabel("Frame")
        ax.set_ylabel("Predicted median depth (m)")
        ax.set_title(args.title or "", fontsize=12)
        ax.legend(fontsize=7, ncol=1 if len(S) <= 12 else 2,
                  loc="lowerright", framealpha=0.9)
        fig.tight_layout()

    else:  # radar
        if len(pts) > 8:
            print(f"\n[warn] {len(pts)} models on one radar will be hard to "
                  f"read; consider --style heatmap, --style grid, or --top 6")
        angles = np.linspace(0, 2 * np.pi, K, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7.6, 7.0), subplot_kw=dict(polar=True))
        cmap = plt.get_cmap("tab20" if len(pts) > 10 else "tab10")
        ncolors = 20 if len(pts) > 10 else 10
        for i, d in enumerate(pts):
            vals = list(M[:, i]) + [M[0, i]]
            c = cmap(i % ncolors)
            ax.plot(angles, vals, "-", linewidth=2, color=c, marker="*",
                    markersize=8, label=f"{label_of(d)}  ({d['AbsRel']:.3f})")
            ax.fill(angles, vals, color=c, alpha=0.06)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.35)
        ax.set_title(args.title or "", fontsize=11)
        ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=7,
                  ncol=1 if len(pts) <= 10 else 2)
        fig.tight_layout()

    if args.no_grid:
        for _a in fig.get_axes():
            _a.grid(False, which="both")

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.out}  (style={style}, {len(pts)} models)")


if __name__ == "__main__":
    main()
