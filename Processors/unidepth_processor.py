#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UniDepthV2 depth estimation processor.

Changes from the previous version:
  * CAMERA CONDITIONING FIXED. UniDepthV2's infer() expects a `Pinhole`
    camera object, NOT a bare 3x3 K (their demo: `camera = Pinhole(
    K=intrinsics.unsqueeze(0))`, and only V1/V2old take the raw K). Passing
    a bare tensor made V2 raise and silently fall back to UNCALIBRATED
    inference, so the metric scale came from the model's own FOV estimate
    instead of the crop's. Conditioning is worth it: the paper's Table XI
    reports delta1 49.3 -> 54.5 with camera conditioning.
  * resolution_level / interpolation_mode exposed (V2-only knobs, per their
    demo). The paper (Fig. 4) shows V2 is nearly invariant to input shape,
    so these trade speed for detail with little accuracy cost.
  * The per-pixel uncertainty V2 predicts is kept in `last_confidence`
    (None if the model doesn't emit it). Not used for scoring; handy if you
    later want to mask low-confidence pixels before feeding a tracker.

Expected parameter counts (paper Table VIII) -- use these to confirm the
right checkpoint loaded:  Small 34.18M, Base 114.4M, Large 353.8M.
"""

import sys
from pathlib import Path

import torch
import numpy as np


# Global state for UniDepthV2 availability
UNIDEPTH_AVAILABLE = False
UNIDEPTH_PATH = None


def load_unidepth_from_path(local_path):
    """Load UniDepthV2 from a local cloned directory."""
    global UNIDEPTH_AVAILABLE, UNIDEPTH_PATH

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[UniDepthV2] Local path not found: {local_path}")
        return False

    sys.path.insert(0, str(local_path))
    try:
        from unidepth.models import UniDepthV2  # noqa: F401
        UNIDEPTH_AVAILABLE = True
        UNIDEPTH_PATH = local_path
        print(f"[UniDepthV2] Loaded from local clone: {local_path}")
        return True
    except ImportError as e:
        print(f"[UniDepthV2] Failed to import from {local_path}: {e}")
        return False


def is_unidepth_available():
    """Live check (avoids stale `from ... import UNIDEPTH_AVAILABLE`)."""
    return UNIDEPTH_AVAILABLE


# Try installed package first
try:
    from unidepth.models import UniDepthV2  # noqa: F401
    UNIDEPTH_AVAILABLE = True
    print("[UniDepthV2] Loaded from installed package")
except ImportError:
    pass

if not UNIDEPTH_AVAILABLE:
    print("[UniDepthV2] Not available yet (will check --unidepth-path)")


class UniDepthV2Processor:
    """Wrapper for UniDepthV2 depth estimation."""

    def __init__(self, model_name="lpiccinelli/unidepth-v2-vits14",
                 device="cuda", resolution_level=None,
                 interpolation_mode="bilinear"):
        """
        Args:
            model_name: HuggingFace id, e.g. lpiccinelli/unidepth-v2-vit{s,b,l}14
            device: 'cuda' or 'cpu'
            resolution_level: V2-only int knob (higher = finer/slower).
                None leaves the library default.
            interpolation_mode: V2-only; their demo sets "bilinear".
        """
        self.device = device
        self.model_name = model_name
        self._available = False
        self._intrinsics_ok = None   # None = untested, True/False after first try
        self.last_confidence = None  # (H, W) float32 uncertainty, or None

        try:
            from unidepth.models import UniDepthV2
            print(f"[UniDepthV2] Loading {model_name}...")
            self.model = UniDepthV2.from_pretrained(model_name)

            # V2-only inference knobs (set before .to()/eval, like the demo)
            if resolution_level is not None:
                try:
                    self.model.resolution_level = int(resolution_level)
                    print(f"[UniDepthV2] resolution_level={resolution_level}")
                except Exception as e:
                    print(f"[UniDepthV2] resolution_level not settable: {e}")
            if interpolation_mode is not None:
                try:
                    self.model.interpolation_mode = interpolation_mode
                except Exception as e:
                    print(f"[UniDepthV2] interpolation_mode not settable: {e}")

            self.model = self.model.to(device).eval()
            self._available = True
            print("[UniDepthV2] \u2713 Model loaded successfully")
        except Exception as e:
            print(f"[UniDepthV2] \u2717 Failed to load: {e}")
            import traceback
            traceback.print_exc()
            self._available = False

    def is_available(self):
        """Check if model is available."""
        return self._available

    @staticmethod
    def _to_hw(t):
        """Tensor/array of shape (1,1,H,W) / (1,H,W) / (H,W) -> (H, W) float32."""
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            t = t.detach().float().cpu().numpy()
        t = np.asarray(t)
        while t.ndim > 2:
            t = t[0]
        return t.astype(np.float32)

    def _extract_depth(self, output):
        """Pull depth (and uncertainty, if present) out of the model output."""
        if isinstance(output, dict):
            depth = output.get("depth")
            if depth is None:
                depth = output.get("metric_depth")
            if depth is None:
                depth = next(iter(output.values()))
            # V2 also predicts a per-pixel uncertainty/confidence
            for key in ("confidence", "uncertainty", "sigma"):
                if key in output:
                    self.last_confidence = self._to_hw(output[key])
                    break
            else:
                self.last_confidence = None
        else:
            depth = output
            self.last_confidence = None

        return self._to_hw(depth)

    def _build_camera(self, K):
        """Wrap a 3x3 K as the camera object UniDepthV2's infer() expects.

        V2 takes a `Pinhole` (batched K); V1/V2old take the bare K matrix.
        Returns (camera_obj_or_tensor, used_pinhole: bool).
        """
        K_t = torch.as_tensor(np.asarray(K, dtype=np.float32),
                              device=self.device)
        try:
            from unidepth.utils.camera import Pinhole
            return Pinhole(K=K_t.unsqueeze(0)), True
        except Exception:
            # older API: pass the bare matrix
            return K_t, False

    def infer(self, rgb_image, K=None):
        """
        Estimate metric depth from an RGB image.

        Official usage: raw uint8 tensor (C, H, W) -- no manual normalisation,
        the model handles it internally.

        Args:
            rgb_image: numpy uint8 array (H, W, 3) in RGB format
            K: optional 3x3 pinhole intrinsics. When given, the model is
               conditioned on the true camera instead of self-estimating one.
               UniDepthV2 still works WITHOUT it (unlike UniDAC) -- it just
               estimates its own camera, so the metric scale is less anchored.

        Returns:
            depth_map: float32 numpy array (H, W) in meters, or None
        """
        if not self._available:
            return None

        try:
            if isinstance(rgb_image, np.ndarray):
                rgb = torch.from_numpy(
                    np.ascontiguousarray(rgb_image)).permute(2, 0, 1)
            else:
                rgb = rgb_image
            rgb = rgb.to(self.device)

            with torch.no_grad():
                if K is not None and self._intrinsics_ok is not False:
                    camera, used_pinhole = self._build_camera(K)
                    try:
                        output = self.model.infer(rgb, camera)
                        if self._intrinsics_ok is None:
                            self._intrinsics_ok = True
                            how = "Pinhole" if used_pinhole else "raw K"
                            print(f"[UniDepthV2] \u2713 Conditioning on provided "
                                  f"intrinsics ({how})")
                    except (TypeError, RuntimeError) as e:
                        # try the other calling convention once before giving up
                        try:
                            fallback = (camera.K.squeeze(0) if used_pinhole
                                        else camera)
                            output = self.model.infer(rgb, fallback)
                            self._intrinsics_ok = True
                            print("[UniDepthV2] \u2713 Conditioning on provided "
                                  "intrinsics (raw K fallback)")
                        except (TypeError, RuntimeError):
                            self._intrinsics_ok = False
                            print(f"[UniDepthV2] !! Intrinsics rejected ({e}); "
                                  f"running UNCALIBRATED (scale may be off) !!")
                            output = self.model.infer(rgb)
                else:
                    if K is None and self._intrinsics_ok is None:
                        self._intrinsics_ok = False
                        print("[UniDepthV2] !! No intrinsics provided; "
                              "model will self-estimate the camera !!")
                    output = self.model.infer(rgb)

            return self._extract_depth(output)

        except Exception as e:
            print(f"[UniDepthV2] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None