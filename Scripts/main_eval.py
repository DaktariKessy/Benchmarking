#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Usage:
     python3 Scripts/main_eval.py --model moge --moge-model Ruicheng/moge-2-vitl  --moge-num-tokens 3600   --bag Data/1.5m --display --max-frames 30
     python3 Scripts/main_eval.py --model depthpro --bag Data/1.5m  --display --max-frames 30
     python3 Scripts/main_eval.py --model unidac   --bag Data/1.5m  --display --max-frames 30
     python3 Scripts/main_eval.py --model unik3d   --bag Data/1.5m  --unik3d-model lpiccinelli/unik3d-vitl        --display --max-frames 30
     python3 Scripts/main_eval.py --model unidepth --bag Data/1.5m  --unidepth-model lpiccinelli/unidepth-v2-vits --display --max-frames 30
     python3 Scripts/main_eval.py --model metric3d --bag Data/1.5m  --metric3d-variant metric3d_vit_large         --display --max-frames 30
     python3 Scripts/main_eval.py --model vda      --bag Data/1.5m  --vda-encoder vitb                            --display --max-frames 30

    
Sweeping is a shell loop, not a code edit:
    for m in depthpro metric3d moge unidepth unidac vda; do
        python3 main_eval.py --model $m --bag ~/Trials/1.5m --pitch -10
    done

Every run writes eval_<model>_<bag>_<config>.csv with the SAME 12 columns,
so the files concatenate directly for cross-model comparison.
"""

import argparse
import sys

from adapters import REGISTRY
from bench_core import add_common_args, run_eval



def main():
    ap = argparse.ArgumentParser(
        description="Bag -> equi2pers -> depth model -> compare vs Kinect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY),
                    help="which model to benchmark")
    add_common_args(ap)

    # Every adapter's flags are registered, not just the chosen one. They are
    # all prefixed (--moge-*, --vda-*, ...) so they cannot collide, and this
    # way `--help` shows the full surface and a sweep script can pass the same
    # flag set to several models without argparse rejecting it.
    for adapter in REGISTRY.values():
        adapter.add_args(ap)

    args = ap.parse_args()
    adapter = REGISTRY[args.model]

    print(f"[MODEL] {adapter.display_name}")
    rows = run_eval(adapter, args)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
