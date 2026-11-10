#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoGe depth estimation processor.

Changes from the previous version:
  * num_tokens is now a real constructor arg and is FORWARDED to
    model.infer(). Per MoGe's API, when num_tokens is not None it OVERRIDES
    resolution_level. This is what makes main_moge_eval's --moge-num-tokens
    flag actually take effect (the paper's token-count operating point).
  * verbose_intrinsics: when True, prints the intrinsics/FOV MoGe actually
    used on the first frame. Use it once to confirm MoGe HONORS the fov_x you
    pass (fov_x=60) rather than estimating its own -- a mismatch there shows
    up as a metric scale offset.
"""

import sys
from pathlib import Path

import torch
import cv2
import numpy as np

# Global state for MoGe availability
MoGe_AVAILABLE = False
MoGe_PATH = None


def load_MoGe_from_path(local_path):
    """Load MoGe from a local cloned directory.

    Args:
        local_path: Path to MoGe clone root

    Returns:
        bool: True if load succeeded, False otherwise
    """
    global MoGe_AVAILABLE, MoGe_PATH

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[MoGe] Local path not found: {local_path}")
        return False

    sys.path.insert(0, str(local_path))
    try:
        from moge.model.v2 import MoGeModel  # noqa: F401
        MoGe_AVAILABLE = True
        MoGe_PATH = local_path
        print(f"[MoGe] Loaded from local clone: {local_path}")
        return True
    except ImportError as e:
        print(f"[MoGe] Failed to import from {local_path}: {e}")
        return False


# Try installed package first
try:
    from moge.model.v2 import MoGeModel  # noqa: F401
    MoGe_AVAILABLE = True
    print("[MoGe] Loaded from installed package")
except ImportError:
    pass

if not MoGe_AVAILABLE:
    print("[MoGe] Not available yet (will check --MoGe-path)")


def is_moge_available():
    """Live check of module-level availability (safe to call from other
    modules; avoids the stale-value problem of `from ... import MoGe_AVAILABLE`)."""
    return MoGe_AVAILABLE


class MoGeProcessor:
    """Manages MoGe model and inference for hand frames."""

    def __init__(self, model_id="Ruicheng/moge-2-vitb", device="cuda:0",
                 use_fp16=True, resolution_level=9, fov_x=None,
                 num_tokens=None, verbose_intrinsics=False):
        """Initialize MoGe processor.

        Args:
            model_id: HuggingFace model identifier (or local path).
            device: Torch device (e.g., "cuda:0" or "cpu").
            use_fp16: Use half precision for faster inference (GPU only).
            resolution_level: MoGe internal resolution, 0-9. Higher = more
                tokens, sharper detail, slower. MoGe's own default is 9.
                IGNORED by MoGe if num_tokens is set.
            fov_x: Horizontal field of view in degrees. If None, MoGe
                ESTIMATES the FOV itself (scale is then not anchored to your
                camera). Pass your crop FOV to anchor the metric scale.
            num_tokens: Explicit token count (~1200-2500). If not None, it
                OVERRIDES resolution_level. This is the paper's operating-
                point axis.
            verbose_intrinsics: Print the intrinsics MoGe used on the first
                frame (to confirm it honors fov_x).
        """
        self.device = torch.device(device)
        self.model = None
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.resolution_level = resolution_level
        self.fov_x = fov_x
        self.num_tokens = num_tokens
        self.verbose_intrinsics = verbose_intrinsics
        self._printed_intrinsics = False

        try:
            # Import here so it uses the path set by load_MoGe_from_path()
            from moge.model.v2 import MoGeModel

            print(f"[MoGe] Loading model: {model_id}")
            self.model = MoGeModel.from_pretrained(model_id).to(self.device).eval()
            print("[MoGe] Model loaded successfully")
        except Exception as e:
            print(f"[MoGe] Failed to load model: {e}")
            self.model = None

    def is_available(self):
        """Check if model is available and loaded."""
        return self.model is not None

    def infer(self, frame_rgb_uint8):
        """Process a single frame to depth.

        Args:
            frame_rgb_uint8: uint8 RGB image (H, W, 3)

        Returns:
            (depth_metric, depth_vis):
                depth_metric: float32 (H, W) depth in meters, inf on invalid
                              pixels (e.g. sky). None if inference failed.
                depth_vis:    uint8 (H, W) min-max normalized visualization.
                              None if inference failed.
        """
        if self.model is None:
            return None, None

        try:
            # To tensor (3, H, W), RGB in [0, 1]. No resizing needed:
            # MoGe handles arbitrary resolutions/aspect ratios internally,
            # and forcing a square would distort the geometry.
            tensor = (
                torch.from_numpy(frame_rgb_uint8 / 255.0)
                .float()
                .permute(2, 0, 1)
                .to(self.device)
            )

            # num_tokens (when set) overrides resolution_level inside MoGe.
            output = self.model.infer(
                tensor,
                fov_x=self.fov_x,
                use_fp16=self.use_fp16,
                resolution_level=self.resolution_level,
                num_tokens=self.num_tokens,
            )

            # One-time check: confirm MoGe honored the fov_x we passed.
            if self.verbose_intrinsics and not self._printed_intrinsics:
                self._printed_intrinsics = True
                if "intrinsics" in output:
                    K = output["intrinsics"].detach().cpu().numpy()
                    print(f"[MoGe] intrinsics used (normalised):\n{K}")
                    # fx is K[0,0] in normalised units; fov_x = 2*atan(0.5/fx)
                    try:
                        fx = float(np.asarray(K).reshape(3, 3)[0, 0])
                        fov = np.degrees(2 * np.arctan(0.5 / fx))
                        print(f"[MoGe] implied fov_x = {fov:.2f} deg "
                              f"(you passed {self.fov_x})")
                    except Exception:
                        pass

            depth = output["depth"].cpu().numpy()  # (H, W) float32, meters

            # Visualization: normalize only over valid (finite) pixels
            valid = np.isfinite(depth)
            vis = np.zeros(depth.shape, dtype=np.uint8)
            if valid.any():
                d = depth[valid]
                d_min, d_max = d.min(), d.max()
                if d_max > d_min:
                    vis[valid] = (
                        (depth[valid] - d_min) / (d_max - d_min) * 255
                    ).astype(np.uint8)

            return depth, vis

        except Exception as e:
            print(f"[MoGe] Inference failed: {e}")
            return None, None