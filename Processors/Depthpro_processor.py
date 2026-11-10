#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apple Depth Pro depth estimation processor.

Mirrors the UniDepthV2/Metric3D/UniDAC/LR2Depth processor interface:
availability flags, local clone loading, infer(rgb, K) -> float32
depth in meters or None.

Depth Pro traits relevant to the benchmark:
- Metric z-depth, sharp boundaries; internally always runs at
  1536x1536 (input is resized, output is resized back).
- Focal-length aware WITHOUT retraining: if K is provided, fx is used
  to scale metric depth; if K is None, the model's own FOV head
  estimates the focal length (self-calibrated condition). The last
  estimated/used focal is stored in `last_f_px` for logging.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


# Global state for Depth Pro availability
DEPTHPRO_AVAILABLE = False
DEPTHPRO_PATH = None


def load_depthpro_from_path(local_path):
    """Load Depth Pro from a local cloned directory (src/ layout)."""
    global DEPTHPRO_AVAILABLE, DEPTHPRO_PATH

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[DepthPro] Local path not found: {local_path}")
        return False

    # repo uses a src/ layout; support both installed and raw clone
    sys.path.insert(0, str(local_path / "src"))
    try:
        import depth_pro  # noqa: F401
        DEPTHPRO_AVAILABLE = True
        DEPTHPRO_PATH = local_path
        print(f"[DepthPro] Loaded from local clone: {local_path}")
        return True
    except ImportError as e:
        print(f"[DepthPro] Failed to import from {local_path}: {e}")
        return False


def is_depthpro_available():
    """Live check (avoids stale `from ... import DEPTHPRO_AVAILABLE`)."""
    return DEPTHPRO_AVAILABLE


class DepthProProcessor:
    """Wrapper for Apple Depth Pro."""

    def __init__(self, checkpoint, device="cuda", precision="fp16"):
        """
        Args:
            checkpoint: path to depth_pro.pt
            device: 'cuda' or 'cpu'
            precision: 'fp16' (default) or 'fp16' (faster, less VRAM;
                       validate against fp16 once before trusting)
        """
        self.device = torch.device(device)
        self._available = False
        self._mode_announced = False
        self.last_f_px = None   # focal actually used on the last frame

        try:
            from depth_pro.depth_pro import (
                create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT)

            cfg = replace(DEFAULT_MONODEPTH_CONFIG_DICT,
                          checkpoint_uri=str(checkpoint))
            dtype = torch.half if precision == "fp16" else torch.float32

            print(f"[DepthPro] Loading model (precision={precision})...")
            self.model, self.transform = create_model_and_transforms(
                config=cfg, device=self.device, precision=dtype)
            self.model.eval()
            self._available = True
            print("[DepthPro] \u2713 Model loaded successfully")
        except Exception as e:
            print(f"[DepthPro] \u2717 Failed to load: {e}")
            import traceback
            traceback.print_exc()
            self._available = False

    def is_available(self):
        """Check if model is available."""
        return self._available

    def infer(self, rgb_image, K=None):
        """
        Estimate metric depth from an RGB image.

        Args:
            rgb_image: numpy uint8 array (H, W, 3) in RGB format
            K: optional 3x3 intrinsics at this image's resolution.
               If given, fx drives the metric scaling (oracle
               intrinsics). If None, Depth Pro estimates the focal
               length itself (self-calibrated condition).

        Returns:
            depth_map: float32 numpy array (H, W), z-depth in meters,
                       or None on failure.
        """
        if not self._available:
            return None

        try:
            if not self._mode_announced:
                self._mode_announced = True
                if K is not None:
                    print("[DepthPro] \u2713 Using provided intrinsics "
                          f"(f_px={float(np.asarray(K)[0, 0]):.1f})")
                else:
                    print("[DepthPro] !! No intrinsics provided; using the "
                          "model's own focal-length estimate !!")

            x = self.transform(rgb_image).to(self.device)

            f_px = None
            if K is not None:
                f_px = torch.tensor(float(np.asarray(K)[0, 0]),
                                    device=self.device)

            with torch.no_grad():
                pred = self.model.infer(x, f_px=f_px)

            depth = pred["depth"].detach().float().cpu().numpy()
            while depth.ndim > 2:
                depth = depth[0]

            fl = pred.get("focallength_px", f_px)
            if fl is not None:
                self.last_f_px = float(torch.as_tensor(fl).flatten()[0])

            return depth.astype(np.float32)

        except Exception as e:
            print(f"[DepthPro] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None