#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-panel distance sweep comparing chosen models/variants: temporal
stability (CV%), accuracy (AbsRel), and speed (FPS) vs scene distance.

WHY THREE SEPARATE PANELS, NOT ONE
-----------------------------------
The three metrics answer different questions, share no units, and do not
always move together with distance. Three panels sharing only the x-axis
shows all three without distorting any of them.

LAYOUT -- READ THIS BEFORE CHANGING --figsize
---------------------------------------------
A figure is NOT resized by \\includegraphics without cost: scaling a 12 in
wide PDF into a 3.33 in column shrinks every label by the same 3.6x, so
10 pt axis text renders at under 3 pt. The fix is to AUTHOR the figure at
its final width and never let LaTeX scale it.

    --layout column  (default)  3.33 in wide, panels stacked VERTICALLY.
                                Drop into \\begin{figure} with
                                \\includegraphics[width=\\linewidth]{...}
                                and the scale factor is 1.0.
    --layout wide               12 in wide, panels side by side. For a
                                full-width \\begin{figure*}, a slide, or
                                the appendix -- NOT for one column.

A column is tall and narrow, so the figure must be too. Three panels in a
row can never be legible at 3.33 in no matter what font size is chosen.

TWO WAYS TO USE THIS FILE
--------------------------
  1. Compare DIFFERENT models, each in one mode -- the default.
        python3 compare_panel.py --dir . --models unik3d-vitl unidac \
            --mode equi
  2. Compare crop vs 360-input for the SAME model(s) -- --mode both, solid
     = panorama, dashed = crop; circle marker = panorama, square = crop.
     COLOUR identifies the SERIES (family+variant), not the mode, so a
     family contributing several sizes does not collapse into shades of
     one hue -- linestyle/marker alone carry the crop-vs-360 distinction.
        python3 compare_panel.py --dir . \
            --models unidac unik3d-vits unik3d-vitb unik3d-vitl \
            --mode both --out fig_crop_vs_equi.pdf

METHODOLOGY -- kept identical to the rest of the benchmark
------------------------------------------------------------
  * AbsRel and CV%: median/CV over ALL frames including the first --
    matches plot_model_radar8.py's convention.
  * FPS: median of t_infer_ms with frame 1 (warm-up) EXCLUDED -- also
    matches the existing convention.
  * CV% = coefficient of variation of the per-frame median predicted depth
    (pred_p50_m). No ground truth needed: the scene is static.

USAGE
    python3 compare_panel.py --dir .
    python3 compare_panel.py --dir . --layout wide --font times
"""
import argparse
import csv
import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# IEEE two-column: \columnwidth is ~3.33 in (252 pt) and \textwidth (both
# columns) is ~7.16 in (516 pt). Measure your own with \the\columnwidth and
# \the\textwidth in the document if the class differs.
COL_IN = 3.4      # (\textwidth - \columnsep)/2 = (7.0 - 0.2)/2
SPAN_IN = 7.0     # \textwidth, IEEEtran V1.6b

FAMILY_COLORS = {
    "moge": "#5ca15f", "unik3d": "#a779c5", "unidepth": "#718eca",
    "videodepth": "#3cc3cb", "vda": "#3cc3cb", "metric3d": "#e09d73",
    "depthpro": "#f38da4", "unidac": "#e889d4",
}
PALETTE = ["#718eca", "#e09d73", "#5ca15f", "#a779c5", "#3cc3cb",
           "#f38da4", "#b8b05a", "#e889d4", "#4c72b0", "#c44e52",
           "#8172b3", "#937860", "#da8bc3", "#8c8c8c", "#ccb974"]
_FALLBACK = {}

# The SAME 17 bold line colours plot_model_radar6.py uses for its distance
# plots (tab10 + Dark2, pastels dropped). Copied verbatim and assigned the
# same way (sorted model x variant -> index), so a model keeps its colour
# between the bags-grid figure and this one.
LINE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d",
]


def build_radar_colors(pts, plotted, scope, mode):
    """Bold colours, indexed exactly like radar6's build_line_colors().

    radar6 sorts every (model, variant) it has loaded and hands out
    LINE_PALETTE in that order, so the colour of a model depends on WHICH
    CSVs that run loaded. `scope` says which radar run to reproduce:

      radar     all pinhole CSVs in --dir  -> radar6 run with its defaults
                (no --models, --mode pinhole), i.e. the main bags-grid fig.
      mode      all CSVs of THIS --mode    -> radar6 run with the same --mode
                (its equi runs are named '<model>-equi', which shifts the
                sort order, so this differs from 'radar').
      selected  only the series plotted here -> radar6 run with the same
                --models filter.
    """
    if scope == "mode" and mode == "both":
        print("[color] --color-scope mode needs a single --mode; "
              "using 'selected'")
        scope = "selected"
    if scope == "radar":
        ref = {(d["family"], d["variant"]): (d["family"], d["variant"])
               for d in pts if d["mode"] == "pinhole"}
    elif scope == "mode":
        suf = "-equi" if mode == "equi" else ""
        ref = {(d["family"] + suf, d["variant"]): (d["family"], d["variant"])
               for d in pts if d["mode"] == mode}
    else:
        ref = {k: k for k in plotted}
    colors = {}
    for i, rk in enumerate(sorted(ref)):
        colors[ref[rk]] = LINE_PALETTE[i % len(LINE_PALETTE)]
    # series radar6 would not have had in that run (e.g. a 360-only model
    # under scope 'radar'): give them colours no plotted series is using
    used = {colors[k] for k in plotted if k in colors}
    spare = [c for c in LINE_PALETTE if c not in used] or LINE_PALETTE
    for j, k in enumerate(sorted(k for k in plotted if k not in colors)):
        colors[k] = spare[j % len(spare)]
        print(f"[color] {k[0]}{'-' + k[1] if k[1] else ''} is outside "
              f"--color-scope {scope}; assigned a spare colour")
    return colors


def color_for(family):
    fam = family[:-5] if family.endswith("-equi") else family
    if fam in FAMILY_COLORS:
        return FAMILY_COLORS[fam]
    if fam not in _FALLBACK:
        _FALLBACK[fam] = PALETTE[len(_FALLBACK) % len(PALETTE)]
    return _FALLBACK[fam]


VARIANT_TOKENS = "vits|vitb|vitl|small|large|giant2|tiny|base"
_SIZE_RANK = {"vits": 0, "tiny": 0, "small": 0, "vitb": 1, "base": 1,
              "vitl": 2, "large": 2, "giant2": 3, "": 1}


def parse_spec(spec):
    """'unik3d-vitl' -> ('unik3d','vitl') ; 'unidac' -> ('unidac','')."""
    m = re.match(rf"^(.*)-({VARIANT_TOKENS})$", spec, re.I)
    return (m.group(1).lower(), m.group(2).lower()) if m else (spec.lower(), "")


def build_colors(fam_var_pairs):
    """One DISTINCT colour per (family, variant) actually being plotted.

    Families contributing exactly one variant keep that family's usual
    identity colour -- consistent with every other figure in the paper.
    Families contributing SEVERAL variants get one palette colour each, so
    unik3d-vits/vitb/vitl are distinct at a glance rather than three
    lightnesses of one hue. Crop-vs-360 stays encoded by linestyle and
    marker shape alone, independent of colour.
    """
    by_family = {}
    for fam, var in fam_var_pairs:
        by_family.setdefault(fam, set()).add(var)

    colors = {}
    used = set()

    for fam, variants in sorted(by_family.items()):
        if len(variants) == 1:
            base = color_for(fam)
            colors[(fam, next(iter(variants)))] = base
            used.add(base)

    pool = [c for c in PALETTE if c not in used]
    pool += [c for c in PALETTE if c in used]
    pool_iter = iter(pool)
    for fam, variants in sorted(by_family.items()):
        if len(variants) <= 1:
            continue
        for v in sorted(variants, key=lambda v: _SIZE_RANK.get(v, 1)):
            c = next(pool_iter)
            while c in used:
                c = next(pool_iter)
            colors[(fam, v)] = c
            used.add(c)
    return colors


def load_csv(path):
    """One eval CSV -> family, mode, variant, bag (numeric), three metrics."""
    base = os.path.basename(path)
    m = re.match(r"eval_([a-z0-9]+(?:-[a-z0-9]+)*)_", base)
    if not m:
        return None
    token = m.group(1)
    mode = "equi" if token.endswith("-equi") else "pinhole"
    family = token[:-5] if mode == "equi" else token
    cfg = base[len("eval_"):-4]
    mb = re.match(r"eval_[a-z0-9]+(?:-[a-z0-9]+)*_([0-9]*[._]?[0-9]+m)_", base)
    if not mb:
        return None
    bag_m = float(mb.group(1).replace("_", ".")[:-1])
    vm = re.search(rf"(?:vit[_-]?)?({VARIANT_TOKENS})", cfg, re.I)
    variant = ""
    if vm:
        w = vm.group(1).lower()
        variant = {"small": "vits", "base": "vitb", "large": "vitl",
                   "s": "vits", "b": "vitb", "l": "vitl"}.get(w, w)
    ab, p50, lat = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                fr = int(r["frame"])
                ab.append(float(r["AbsRel"]))
                p50.append(float(r["pred_p50_m"]))
                if fr > 1:
                    lat.append(float(r["t_infer_ms"]))
            except (KeyError, ValueError):
                continue
    if not ab:
        return None
    p50 = np.array(p50, dtype=float)
    ok = np.isfinite(p50) & (p50 > 0)
    cv = (100.0 * np.std(p50[ok], ddof=1) / max(np.mean(p50[ok]), 1e-9)
          if ok.sum() >= 3 else np.nan)
    lat = np.array(lat) if lat else np.array([np.nan])
    return {
        "family": family, "mode": mode, "variant": variant, "bag_m": bag_m,
        "AbsRel": float(np.median(ab)), "cv_pct": float(cv),
        "Fps": float(1000.0 / max(np.median(lat), 1e-9)),
        "n": len(ab),
    }


def series_for(pts, family, variant, mode):
    rows = [d for d in pts if d["family"] == family and d["variant"] == variant
            and d["mode"] == mode]
    rows.sort(key=lambda d: d["bag_m"])
    return rows


def label_for(family, variant, mode, show_mode):
    base = f"{family}-{variant}" if variant else family
    return base + (" (360°)" if (show_mode and mode == "equi") else
                   " (crop)" if (show_mode and mode == "pinhole") else "")


def style_for_layout(layout, user_font_size):
    """Font and element sizes appropriate to the final printed width.

    In column layout the figure is authored at its final size, so these ARE
    the sizes the reader sees -- no scaling happens afterwards. 7 pt axis
    text against 9-10 pt body text is the usual convention and stays
    legible in print.
    """
    if layout == "column":
        s = dict(font=7, label=7, tick=6, legend=10, lw=1.1, ms=3.0,
                 annot=5, grid=0.3)
    elif layout == "span":
        s = dict(font=14, label=8, tick=7, legend=12, lw=1.3, ms=3.6,
                 annot=6, grid=0.3)
    elif layout == "grid":
        # Same look as your Fig. distanceVsq (plot_model_radar6.py
        # --style bags-grid --font-size 14): 14 pt ticks and axis labels,
        # 8 pt legend, 12 pt panel titles, lw 1.8, marker size 5.
        s = dict(font=14, label=14, tick=14, legend=8, lw=1.8, ms=5.0,
                 annot=8, grid=0.3)
    else:
        s = dict(font=11, label=11, tick=10, legend=10, lw=1.9, ms=6.0,
                 annot=9, grid=0.3)
    if user_font_size:
        scale = user_font_size / s["font"]
        # grid: like bags-grid, --font-size changes ticks and labels only;
        # the legend stays at 8 pt
        keys = ("font", "label", "tick", "annot") if layout == "grid" \
            else ("font", "label", "tick", "legend", "annot")
        for k in keys:
            s[k] *= scale
    matplotlib.rcParams.update({
        "font.size": s["font"], "axes.labelsize": s["label"],
        "axes.titlesize": s["label"], "xtick.labelsize": s["tick"],
        "ytick.labelsize": s["tick"], "legend.fontsize": s["legend"],
    })
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--models", nargs="+", default=["unik3d-vitl", "unidac"],
                    help="model tokens to compare, e.g. unik3d-vitl unidac")
    ap.add_argument("--mode", choices=["equi", "pinhole", "both"],
                    default="equi",
                    help="'equi' = full-panorama runs, 'pinhole' = the 60 deg "
                         "crop runs, 'both' overlays the two for EACH named "
                         "model")
    ap.add_argument("--layout", choices=["span", "column", "wide", "grid"],
                    default="grid",
                    help="'grid' (default): same canvas, fonts, line width "
                         "and markers as plot_model_radar6.py --style "
                         "bags-grid --font-size 14. 'span': 7.16 in, panels "
                         "SIDE BY SIDE, "
                         "short, for \\begin{figure*} at width=\\textwidth. "
                         "Costs the least VERTICAL space, which is what a "
                         "page limit actually constrains. 'column': 3.33 in, "
                         "panels stacked, for \\begin{figure}. 'wide': 12 in "
                         "for slides -- do NOT scale this one into a paper.")
    ap.add_argument("--col-width", type=float, default=COL_IN,
                    help="single-column width in inches; check "
                         "\\the\\columnwidth (1 pt = 1/72 in)")
    ap.add_argument("--span-width", type=float, default=SPAN_IN,
                    help="both-column width in inches; check "
                         "\\the\\textwidth")
    ap.add_argument("--height", type=float, default=None,
                    help="total figure height in inches, overriding the "
                         "default. Width is fixed by the column it must fit, "
                         "so height is the only free dimension -- raise it "
                         "to give the panels more room.")
    ap.add_argument("--font", default="default",
                    choices=["default", "serif", "times", "sans", "cm"])
    ap.add_argument("--font-size", type=float, default=None,
                    help="override the base size; everything else scales "
                         "with it")
    ap.add_argument("--annotate", action="store_true",
                    help="print each point's value beside the marker (busy "
                         "in column layout)")
    ap.add_argument("--grid-width", type=float, default=10.2,
                    help="'grid' layout: total width in inches (default "
                         "10.2 = 3 panels x 3.4 in, like bags-grid)")
    ap.add_argument("--panel-titles", nargs="*", default=None,
                    help="'grid' layout: titles drawn under the three panels, "
                         "in order, e.g. --panel-titles '(a) Stability' "
                         "'(b) Accuracy' '(c) Speed'")
    ap.add_argument("--palette", choices=["bold", "soft"], default="bold",
                    help="'bold' (default): the saturated line colours of "
                         "plot_model_radar6.py's distance plots. 'soft': the "
                         "old pastel family colours.")
    ap.add_argument("--color-scope", choices=["radar", "mode", "selected"],
                    default="radar",
                    help="which radar6 run the bold colours should match: "
                         "'radar' (default) = radar6 with its defaults, all "
                         "pinhole models; 'mode' = radar6 with the same "
                         "--mode; 'selected' = radar6 with the same --models.")
    ap.add_argument("--out", default="fig_panel3.pdf")
    args = ap.parse_args()

    if args.font != "default":
        # Same font lists as plot_model_radar6.py, so both figures match.
        # "DejaVu Serif" is the fallback when Times is not installed (e.g.
        # on Colab); without it matplotlib drops to a sans-serif font.
        fam = {"serif": ("serif", ["DejaVu Serif", "Times New Roman",
                                   "Nimbus Roman"]),
               "times": ("serif", ["Times New Roman", "Nimbus Roman",
                                   "DejaVu Serif"]),
               "cm": ("serif", ["CMU Serif", "Computer Modern Roman",
                                "DejaVu Serif"]),
               "sans": ("sans-serif", ["Helvetica", "Arial",
                                       "DejaVu Sans"])}[args.font]
        matplotlib.rcParams["font.family"] = fam[0]
        matplotlib.rcParams[f"font.{fam[0]}"] = fam[1]
        matplotlib.rcParams["mathtext.fontset"] = (
            "cm" if fam[0] == "serif" else "dejavusans")
    st = style_for_layout(args.layout, args.font_size)

    paths = sorted(glob.glob(os.path.join(args.dir, "eval_*.csv")))
    pts = [d for d in (load_csv(p) for p in paths) if d is not None]
    if not pts:
        sys.exit(f"no eval_*.csv found in {args.dir}")

    modes = ["equi", "pinhole"] if args.mode == "both" else [args.mode]
    show_mode = args.mode == "both"

    resolved = []
    for spec in args.models:
        fam, var = parse_spec(spec)
        for mode in modes:
            rows = series_for(pts, fam, var, mode)
            if not rows:
                print(f"[warn] no '{mode}' CSVs for '{spec}' "
                      f"(family={fam!r} variant={var!r}) -- skipped.")
                continue
            resolved.append((fam, var, mode, rows))
    if not resolved:
        sys.exit("nothing to plot -- no requested model/mode combination "
                 "matched any CSV")

    fam_var_pairs = {(fam, var) for fam, var, _m, _r in resolved}
    if args.palette == "bold":
        colors = build_radar_colors(pts, fam_var_pairs, args.color_scope,
                                    args.mode)
    else:
        colors = build_colors(fam_var_pairs)
    per_family = {}
    for fam, var in fam_var_pairs:
        per_family.setdefault(fam, set()).add(var)
    for fam, vs in per_family.items():
        if len(vs) > 1 and args.palette == "soft":
            print(f"[color] {fam}: {len(vs)} variants each get a DISTINCT "
                  f"colour ({', '.join(sorted(vs, key=lambda v: _SIZE_RANK.get(v,1)))})")

    lines = []
    for fam, var, mode, rows in resolved:
        ls, marker = ("-", "o") if (not show_mode or mode == "equi") \
            else ("--", "s")
        lines.append((label_for(fam, var, mode, show_mode),
                      colors[(fam, var)], ls, marker, rows))

    print(f"\n{'series':>26} {'bag(m)':>7} {'AbsRel':>8} {'CV(%)':>7} "
          f"{'FPS':>6} {'n':>4}")
    for lab, _c, _ls, _mk, rows in lines:
        for r in rows:
            print(f"{lab:>26} {r['bag_m']:>7.2f} {r['AbsRel']:>8.4f} "
                  f"{r['cv_pct']:>7.3f} {r['Fps']:>6.1f} {r['n']:>4}")

    # ---- layout ----------------------------------------------------------
    # Column: stacked, shared x, one x-label at the bottom, short y-labels
    # (a 3.33 in panel has no room for "(lower is better)" -- the arrow
    # carries it, and the caption can say it in words).
    column = args.layout == "column"
    span = args.layout == "span"
    grid = args.layout == "grid"
    if grid:
        # Copy of plot_model_radar6.py --style bags-grid with --ncols 3:
        # 3.4 in per panel, 5.6 in tall, short arrow y-labels.
        fig, axs = plt.subplots(1, 3, figsize=(args.grid_width,
                                               args.height or 5.6))
        panels = [
            ("cv_pct", r"$\mathrm{CV_f}$ (%) $\downarrow$", axs[0]),
            ("AbsRel", r"AbsRel $\downarrow$", axs[1]),
            ("Fps",    r"FPS $\uparrow$",      axs[2]),
        ]
    elif span:
        # Side by side at \textwidth. Height is deliberately small: in a
        # page-limited paper the scarce resource is VERTICAL space, and a
        # 2.9 in band across both columns costs less of it than a 6 in
        # figure in one column, while giving each panel more than twice
        # the width.
        n_leg_rows = int(np.ceil(len(lines) / 4))
        height = args.height or (3.05 + 0.18 * n_leg_rows)
        fig, axs = plt.subplots(1, 3, figsize=(args.span_width, height))
        panels = [
            ("cv_pct", r"$\mathrm{CV_f}$ (%) $\downarrow$", axs[0]),
            ("AbsRel", r"AbsRel $\downarrow$", axs[1]),
            ("Fps",    r"FPS $\uparrow$",      axs[2]),
        ]
    elif column:
        n_leg_rows = int(np.ceil(len(lines) / 2))
        height = args.height or (5.2 + 0.19 * n_leg_rows)
        fig, axs = plt.subplots(3, 1, figsize=(args.col_width, height),
                                sharex=True)
        panels = [
            ("cv_pct", r"$\mathrm{CV_f}$ (%) $\downarrow$", axs[0]),
            ("AbsRel", r"AbsRel $\downarrow$", axs[1]),
            ("Fps",    r"FPS $\uparrow$",      axs[2]),
        ]
    else:
        fig, axs = plt.subplots(1, 3, figsize=(12.0, 6.4))
        panels = [
            ("cv_pct", r"Temporal stability, $\mathrm{CV_f}$ (%) "
                       r"(lower is better)", axs[0]),
            ("AbsRel", "AbsRel (lower is better)", axs[1]),
            ("Fps", "FPS   (higher is better)", axs[2]),
        ]

    for key, ylab, ax in panels:
        for lab, c, ls, marker, rows in lines:
            x = [r["bag_m"] for r in rows]
            y = [r[key] for r in rows]
            ax.plot(x, y, marker=marker, ms=st["ms"], lw=st["lw"], color=c,
                    ls=ls, label=lab)
            if args.annotate:
                for xi, yi in zip(x, y):
                    ax.annotate(f"{yi:.3g}", (xi, yi), fontsize=st["annot"],
                                xytext=(0, 5), textcoords="offset points",
                                ha="center")
        ax.set_ylabel(ylab)
        ax.grid(alpha=st["grid"])
        if not column:
            ax.set_xlabel("Scene distance (m)")
        if span:
            ax.set_xticks(sorted({r["bag_m"] for _l, _c, _ls, _m, rr
                                  in lines for r in rr}))
    if column:
        axs[-1].set_xlabel("Scene distance (m)")
        axs[-1].set_xticks(sorted({r["bag_m"] for _l, _c, _ls, _m, rows
                                   in lines for r in rows}))
    if grid and args.panel_titles:
        # same position and size as the bags-grid panel titles
        for ax, t in zip(axs, args.panel_titles):
            ax.text(0.5, -0.18, t, transform=ax.transAxes,
                    ha="center", va="top", fontsize=12)

    handles, labels_ = axs[0].get_legend_handles_labels()
    ncol = 2 if column else (4 if span else min(len(lines), 4))
    n_legend_rows = int(np.ceil(len(lines) / ncol))
    if grid:
        # bags-grid "bottom" legend: left-aligned under panel (a)
        bottom = 0.03 + 0.025 * n_legend_rows
        fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=2.6)
        x0 = axs[0].get_position().x0
        fig.legend(handles, labels_, fontsize=st["legend"], loc="lower left",
                   ncol=ncol, frameon=False, bbox_to_anchor=(x0, 0.0))
        fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=2.6)
    elif span:
        fig.tight_layout(rect=(0, 0.02 + 0.075 * n_legend_rows, 1, 1))
        fig.legend(handles, labels_, loc="lower center", ncol=ncol,
                   frameon=False, handlelength=2.0, columnspacing=1.2,
                   borderaxespad=0.1)
    elif column:
        fig.tight_layout(rect=(0, 0.008 + 0.022 * n_legend_rows, 1, 1))
        fig.legend(handles, labels_, loc="lower center", ncol=ncol,
                   frameon=False, handlelength=2.2, columnspacing=1.0,
                   borderaxespad=0.1)
    else:
        bottom = 0.06 + 0.035 * (n_legend_rows - 1)
        fig.legend(handles, labels_, loc="lower center", ncol=ncol,
                   bbox_to_anchor=(0.5, -bottom + 0.02), frameon=False)
        fig.tight_layout(rect=(0, bottom, 1, 1))

    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    w_in, h_in = fig.get_size_inches()
    print(f"\nwrote {args.out}  ({len(lines)} series, mode={args.mode}, "
          f"layout={args.layout})")
    print(f"figure is {w_in:.2f} x {h_in:.2f} in "
          f"({w_in*72:.0f} x {h_in*72:.0f} pt). In LaTeX use:")
    if column:
        print(r"    \includegraphics[width=\linewidth]{%s}   "
              r"inside \begin{figure}" % os.path.basename(args.out))
        print("  -- the scale factor is 1.0, so the font sizes above are "
              "what the reader sees.")
    elif grid:
        print(r"    \includegraphics[width=\textwidth]{%s}  "
              r"inside \begin{figure*}[t!]" % os.path.basename(args.out))
        print("  -- same size as the bags-grid figure, so the two print "
              "with matching text.")
    elif span:
        print(r"    \includegraphics[width=\textwidth]{%s}  "
              r"inside \begin{figure*}[t!]" % os.path.basename(args.out))
        print("  -- the scale factor is 1.0. figure* floats to the top of a "
              "page; 'h' is ignored.")
    else:
        print(r"    \includegraphics[width=\textwidth]{%s}  "
              r"inside \begin{figure*}" % os.path.basename(args.out))


if __name__ == "__main__":
    main()
