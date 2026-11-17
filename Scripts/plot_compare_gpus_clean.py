#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compact paper-ready comparison of MMDE inference throughput on two GPUs.

One row = one model-scale configuration.
Throughput for each configuration is the median FPS across the available
operating distances.

Visual encoding
---------------
Colour        = model family
Open circle   = RTX 4060 Laptop GPU
Filled circle = RTX 4080

Designed for IEEE double-column figures (~7.1 inch width).
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================================
# Model-family colours
# ============================================================================

FAMILY_COLORS = {
    "unidepth":   "#718eca",
    "metric3d":   "#e09d73",
    "moge":       "#5ca15f",
    "unik3d":     "#a779c5",
    "videodepth": "#3cc3cb",
    "depthpro":   "#f38da4",
    "unidac":     "#e889d4",
}

PALETTE = [
    "#718eca", "#e09d73", "#5ca15f", "#a779c5",
    "#3cc3cb", "#f38da4", "#e889d4"
]

_FALLBACK_COLORS = {}


def color_for(family):
    fam = family.lower()

    if fam == "vda":
        fam = "videodepth"

    if fam in FAMILY_COLORS:
        return FAMILY_COLORS[fam]

    if fam not in _FALLBACK_COLORS:
        _FALLBACK_COLORS[fam] = \
            PALETTE[len(_FALLBACK_COLORS) % len(PALETTE)]

    return _FALLBACK_COLORS[fam]


# ============================================================================
# Filename normalization
# ============================================================================

FAMILY_ALIASES = {
    "video-depth": "videodepth",
    "video_depth": "videodepth",
    "vda": "videodepth",
    "depth-pro": "depthpro",
    "metric3dv2": "metric3d",
    "metric3d-v2": "metric3d",
    "unidepth-v2": "unidepth",
    "moge2": "moge",
    "moge-2": "moge",
}


def normalize_family(family):
    f = family.lower().strip().replace(" ", "")
    return FAMILY_ALIASES.get(f, f)


def normalize_bag(bag):
    b = bag.lower().replace("_", ".")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)m", b)

    if not m:
        return b

    return f"{float(m.group(1)):g}m"


def _variant_token(text):
    s = text.lower().replace("_", "-")

    m = re.search(r"\bvit([sbl])14\b", s)
    if m:
        return f"vit{m.group(1)}14"

    m = re.search(r"\bvit-?(small|base|large|s|b|l)\b", s)
    if m:
        return {
            "small": "vits",
            "base": "vitb",
            "large": "vitl",
            "s": "vits",
            "b": "vitb",
            "l": "vitl",
        }[m.group(1)]

    m = re.search(
        r"(?:^|-)(giant2?|large|small|base|tiny)(?:-|$)",
        s
    )
    if m:
        return m.group(1)

    return ""


def normalize_config(family, cfg):
    fam = normalize_family(family)

    s = cfg.lower().replace("_", "-")
    s = re.sub(r"-+", "-", s).strip("-")

    noise = {
        "fp16", "fp32", "bf16",
        "metric", "rl1", "eval",
        "inference", "infer",
    }

    parts = [p for p in s.split("-") if p and p not in noise]
    s = "-".join(parts)

    if fam in {"depthpro", "unidac"}:
        return ""

    if fam == "metric3d":
        for v in ("giant2", "giant", "large", "small"):
            if re.search(rf"(?:^|-){v}(?:-|$)", s):
                return "giant2" if v == "giant" else v
        return ""

    if fam == "videodepth":
        v = _variant_token(s)
        return {
            "vits14": "vits",
            "vitb14": "vitb",
            "vitl14": "vitl",
        }.get(v, v)

    if fam == "unidepth":
        m = re.search(
            r"unidepth-v2-(vits14|vitb14|vitl14)",
            s
        )

        if m:
            return {
                "vits14": "vits",
                "vitb14": "vitb",
                "vitl14": "vitl",
            }[m.group(1)]

        v = _variant_token(s)

        return {
            "vits14": "vits",
            "vitb14": "vitb",
            "vitl14": "vitl",
        }.get(v, v)

    if fam in {"unik3d", "moge"}:
        v = _variant_token(s)

        return {
            "vits14": "vits",
            "vitb14": "vitb",
            "vitl14": "vitl",
        }.get(v, v)

    return _variant_token(s)


def parse_filename(path):
    """
    Example:
      eval_unik3d-equi_1m_vitl_rl1.csv

    -> family=unik3d, mode=equi, bag=1m, variant=vitl
    """

    base = os.path.basename(path)

    stem = (
        base[:-4]
        if base.lower().endswith(".csv")
        else base
    )

    m = re.match(
        r"^eval_([a-z0-9-]+)_([0-9]+(?:[._][0-9]+)?m)_(.+)$",
        stem,
        re.I
    )

    if not m:
        return None

    raw_family, raw_bag, raw_cfg = m.groups()

    raw_family = raw_family.lower()

    if raw_family.endswith("-equi"):
        mode = "equi"
        raw_family = raw_family[:-5]
    else:
        mode = "crop"

    family = normalize_family(raw_family)
    bag = normalize_bag(raw_bag)
    variant = normalize_config(family, raw_cfg)

    return {
        "file": base,
        "family": family,
        "mode": mode,
        "bag": bag,
        "variant": variant,
        "cfg": raw_cfg,
        "key": (family, mode, bag, variant),
    }


# ============================================================================
# Build run maps
# ============================================================================

def build_run_map(folder, label):
    paths = sorted(
        glob.glob(
            os.path.join(folder, "eval_*.csv")
        )
    )

    groups = defaultdict(list)

    for path in paths:
        meta = parse_filename(path)

        if meta is None:
            print(
                f"[warn] cannot parse: "
                f"{os.path.basename(path)}"
            )
            continue

        groups[meta["key"]].append(path)

    result = {}

    for key, candidates in groups.items():

        if len(candidates) == 1:
            result[key] = candidates[0]

        else:
            print(
                f"[warn] duplicate key in {label}: {key}"
            )

            for p in candidates:
                print("       ", os.path.basename(p))

    return result


# ============================================================================
# Timing
# ============================================================================

def read_frame_rows(path):
    rows = {}

    with open(path, newline="") as f:

        for r in csv.DictReader(f):

            try:
                frame = int(r["frame"])
                latency = float(r["t_infer_ms"])

            except (KeyError, ValueError, TypeError):
                continue

            rows[frame] = latency

    return rows


def paired_run(path_a, path_b):
    """
    Throughput is evaluated only on frame IDs common to both runs.
    Frame 1 is excluded as warm-up.
    """

    ma = parse_filename(path_a)
    mb = parse_filename(path_b)

    if ma is None or mb is None:
        return None, None

    ra = read_frame_rows(path_a)
    rb = read_frame_rows(path_b)

    common = sorted(set(ra) & set(rb))

    timing_frames = [
        f for f in common
        if f > 1
    ]

    if not timing_frames:
        return None, None

    def summarize(rows, meta):

        lat = np.asarray(
            [rows[f] for f in timing_frames],
            dtype=float
        )

        ok = np.isfinite(lat) & (lat > 0)

        if not ok.any():
            fps = np.nan
        else:
            fps = float(
                1000.0 / np.median(lat[ok])
            )

        return {
            "model": meta["family"],
            "variant": meta["variant"],
            "mode": meta["mode"],
            "bag": meta["bag"],
            "Fps": fps,
            "n_common": len(common),
        }

    return summarize(ra, ma), summarize(rb, mb)


# ============================================================================
# GPU-folder matching
# ============================================================================

def build_matches(A, B):
    matches = []

    for key in sorted(set(A) & set(B)):
        matches.append(
            (key, A[key], B[key])
        )

    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))

    if only_a:
        print(f"[warn] {len(only_a)} unmatched runs in A")

    if only_b:
        print(f"[warn] {len(only_b)} unmatched runs in B")

    return matches


# ============================================================================
# Aggregate across distances
# ============================================================================

def aggregate_distances(pairs):
    """
    One row per model-scale-input configuration.

    FPS_A = median FPS across tested distances on GPU A
    FPS_B = median FPS across tested distances on GPU B
    """

    grouped = defaultdict(list)

    for da, db in pairs:

        key = (
            da["model"],
            da["variant"],
            da["mode"],
        )

        grouped[key].append(
            (da, db)
        )

    out = []

    for (model, variant, mode), runs in grouped.items():

        a = np.asarray(
            [
                da["Fps"]
                for da, _ in runs
                if np.isfinite(da["Fps"])
            ],
            dtype=float
        )

        b = np.asarray(
            [
                db["Fps"]
                for _, db in runs
                if np.isfinite(db["Fps"])
            ],
            dtype=float
        )

        if not len(a) or not len(b):
            continue

        fps_a = float(np.median(a))
        fps_b = float(np.median(b))

        out.append({
            "model": model,
            "variant": variant,
            "mode": mode,
            "Fps_A": fps_a,
            "Fps_B": fps_b,
            "speedup":
                fps_b / max(fps_a, 1e-12),
            "n_distances": len(runs),
        })

    return out


# ============================================================================
# Compact paper labels
# ============================================================================

FAMILY_NAMES = {
    "depthpro": "Depth Pro",
    "metric3d": "Metric3D",
    "moge": "MoGe-2",
    "unidepth": "UniDepth",
    "unik3d": "UniK3D",
    "videodepth": "VDA",
    "unidac": "UniDAC",
}

VARIANT_NAMES = {
    "vits": "S",
    "vitb": "B",
    "vitl": "L",
    "small": "S",
    "large": "L",
    "giant2": "Giant2",
}


def display_label(d):
    fam = FAMILY_NAMES.get(
        d["model"],
        d["model"]
    )

    var = VARIANT_NAMES.get(
        d["variant"],
        d["variant"]
    )

    if var:
        label = f"{fam}-{var}"
    else:
        label = fam

    if d["mode"] == "equi":
        label += " (360°)"

    return label


# ============================================================================
# Compact full-width dumbbell plot
# ============================================================================

def plot_dumbbell(ax, configs, log_x=False):

    P = [
        d for d in configs
        if np.isfinite(d["Fps_A"])
        and np.isfinite(d["Fps_B"])
    ]

    if not P:
        ax.text(
            0.5, 0.5,
            "No matched throughput results",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        return []

    # Primary platform = RTX 4060 Laptop GPU.
    # Highest throughput at top.
    P.sort(
        key=lambda d: d["Fps_A"],
        reverse=True
    )

    y = np.arange(len(P))

    for i, d in enumerate(P):

        xa = d["Fps_A"]
        xb = d["Fps_B"]
        c = color_for(d["model"])

        # connecting segment
        ax.plot(
            [xa, xb],
            [i, i],
            color=c,
            lw=1.25,
            alpha=0.58,
            zorder=1
        )

        # RTX 4060 Laptop GPU
        ax.scatter(
            xa,
            i,
            s=29,
            facecolor="white",
            edgecolor=c,
            linewidth=1.35,
            zorder=3
        )

        # RTX 4080
        ax.scatter(
            xb,
            i,
            s=31,
            facecolor=c,
            edgecolor=c,
            linewidth=0.7,
            zorder=4
        )

    # ------------------------------------------------------------------
    # Y labels
    # ------------------------------------------------------------------

    ax.set_yticks(y)

    ax.set_yticklabels(
        [display_label(d) for d in P],
        fontsize=6.4
    )

    ax.invert_yaxis()

    ax.tick_params(
        axis="y",
        length=0,
        pad=2
    )

    # ------------------------------------------------------------------
    # X axis
    # ------------------------------------------------------------------

    ax.set_xlabel(
        "Inference throughput (FPS)",
        fontsize=7.8,
        labelpad=2
    )

    ax.tick_params(
        axis="x",
        labelsize=6.7
    )

    if log_x:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0)

    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.55
    )

    ax.set_axisbelow(True)

    # Cleaner IEEE style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Single summary annotation
    # ------------------------------------------------------------------

    ratios = np.asarray(
        [d["speedup"] for d in P],
        dtype=float
    )

    median_speedup = float(
        np.median(ratios)
    )

    ax.text(
        0.985,
        0.035,
        f"Median speedup: {median_speedup:.2f}$\\times$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="0.78",
            linewidth=0.55,
            alpha=0.95
        )
    )

    return P


# ============================================================================
# GPU legend
# ============================================================================

def gpu_legend_handles(label_a, label_b):

    return [
        Line2D(
            [], [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.30",
            markeredgewidth=1.25,
            markersize=5.2,
            label=label_a
        ),

        Line2D(
            [], [],
            marker="o",
            linestyle="none",
            markerfacecolor="0.30",
            markeredgecolor="0.30",
            markersize=5.2,
            label=label_b
        ),
    ]


# ============================================================================
# Console summary
# ============================================================================

def print_summary(configs):

    P = sorted(
        configs,
        key=lambda d: d["Fps_A"],
        reverse=True
    )

    print(
        f"\n{'configuration':24s}"
        f"{'4060 FPS':>10s}"
        f"{'4080 FPS':>10s}"
        f"{'speedup':>9s}"
    )

    for d in P:

        print(
            f"{display_label(d):24s}"
            f"{d['Fps_A']:10.2f}"
            f"{d['Fps_B']:10.2f}"
            f"{d['speedup']:8.2f}x"
        )

    ratios = np.asarray(
        [d["speedup"] for d in P]
    )

    print(
        f"\nMedian speedup: "
        f"{np.median(ratios):.2f}x"
    )


# ============================================================================
# Main
# ============================================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--a",
        required=True,
        help="CSV directory for RTX 4060 Laptop GPU"
    )

    ap.add_argument(
        "--b",
        required=True,
        help="CSV directory for RTX 4080"
    )

    ap.add_argument(
        "--label-a",
        default="RTX 4060 Laptop GPU"
    )

    ap.add_argument(
        "--label-b",
        default="RTX 4080"
    )

    ap.add_argument(
        "--bag",
        nargs="*",
        default=None
    )

    ap.add_argument(
        "--models",
        nargs="*",
        default=None
    )

    ap.add_argument(
        "--mode",
        choices=["crop", "equi", "both"],
        default="crop",
        help=(
            "crop = 17 perspective configurations; "
            "equi = panoramic only; "
            "both = all configurations"
        )
    )

    ap.add_argument(
        "--font",
        choices=[
            "default",
            "times",
            "serif",
            "sans"
        ],
        default="times"
    )

    ap.add_argument(
        "--log-x",
        action="store_true"
    )

    ap.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None
    )

    ap.add_argument(
        "--out",
        default="Fig_gpu_throughput.pdf"
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # IEEE-compatible font
    # ------------------------------------------------------------------

    if args.font == "times":

        matplotlib.rcParams["font.family"] = "serif"

        matplotlib.rcParams["font.serif"] = [
            "Times New Roman",
            "Nimbus Roman",
            "Times",
            "DejaVu Serif"
        ]

        matplotlib.rcParams["mathtext.fontset"] = "stix"

    elif args.font == "serif":
        matplotlib.rcParams["font.family"] = "serif"

    elif args.font == "sans":
        matplotlib.rcParams["font.family"] = "sans-serif"

    # Small but still readable after insertion at full IEEE text width
    matplotlib.rcParams["font.size"] = 10.0

    # ------------------------------------------------------------------
    # Match GPU runs
    # ------------------------------------------------------------------

    A = build_run_map(
        args.a,
        args.label_a
    )

    B = build_run_map(
        args.b,
        args.label_b
    )

    matches = build_matches(A, B)

    # input regime
    if args.mode != "both":

        matches = [
            r for r in matches
            if r[0][1] == args.mode
        ]

    # distance subset
    if args.bag:

        wanted = {
            normalize_bag(x)
            for x in args.bag
        }

        matches = [
            r for r in matches
            if r[0][2] in wanted
        ]

    # model subset
    if args.models:

        tokens = [
            x.lower()
            for x in args.models
        ]

        matches = [
            r for r in matches
            if any(
                token in "-".join(r[0]).lower()
                for token in tokens
            )
        ]

    if not matches:
        sys.exit(
            "No matched runs remain after filtering."
        )

    # ------------------------------------------------------------------
    # Frame-matched timing
    # ------------------------------------------------------------------

    pairs = []

    for key, path_a, path_b in matches:

        da, db = paired_run(
            path_a,
            path_b
        )

        if da is None or db is None:

            print(
                f"[warn] {key}: "
                "no usable common frames"
            )

            continue

        pairs.append(
            (da, db)
        )

    if not pairs:
        sys.exit(
            "No usable paired runs."
        )

    # ------------------------------------------------------------------
    # Collapse distances
    # ------------------------------------------------------------------

    configs = aggregate_distances(
        pairs
    )

    if not configs:
        sys.exit(
            "No configurations after aggregation."
        )

    print(
        f"\n{len(pairs)} distance-level runs "
        f"-> {len(configs)} configurations"
    )

    print_summary(configs)

    # ------------------------------------------------------------------
    # Compact double-column figure
    # ------------------------------------------------------------------

    if args.figsize:
        figsize = tuple(args.figsize)

    else:
        # IEEE full-width figure:
        # width ≈ 7.0 in, deliberately limited height
        #
        # 17 configs -> ~4.55 in
        # 21 configs -> ~5.1 in
        height = (
            4.55
            if len(configs) <= 17
            else 5.10
        )

        figsize = (
            7.0,
            height
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    plot_dumbbell(
        ax,
        configs,
        log_x=args.log_x
    )

    # ------------------------------------------------------------------
    # Bottom-centre legend
    # ------------------------------------------------------------------

    handles = gpu_legend_handles(
        args.label_a,
        args.label_b
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        fontsize=10,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.6,
        borderaxespad=0
    )

    # very little wasted whitespace
    fig.subplots_adjust(
        left=0.185,
        right=0.985,
        top=0.985,
        bottom=0.13
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    fig.savefig(
        args.out,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(
        f"\nwrote {args.out} "
        f"({len(configs)} configurations)"
    )


if __name__ == "__main__":
    main()
