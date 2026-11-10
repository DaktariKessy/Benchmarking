#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UniK3D depth estimation processor (metric depth + point cloud + rays).

Supports optional camera conditioning: pass a 3x3 intrinsics matrix K to
infer_metric() for correct metric scale when the camera geometry is known
(e.g. a virtual pinhole crop from an equirectangular frame).

VERIFIED BEHAVIOUR (checked against this repo, not assumed):
  * predictions["depth"] is Z-DEPTH, not radial distance. Confirmed by
    max|depth - points[2]| == 0 and a corner/centre ratio of 1.000 where
    radial would have shown ~1.291 at 60 deg. So it feeds compare() directly.
  * Camera conditioning is FUNCTIONAL. Pinhole(params=[fx,fy,cx,cy]) is a
    valid constructor here (it accepts either params= or K=), and the model
    responds to it: fx 449 -> 967 moved median depth 1.964 -> 2.550 m.

resolution_level (added):
  UniK3D resizes the input into a pixel-count band chosen by an integer in
  [0, 9], which partitions shape_constraints pixels_min..pixels_max into ten
  equal steps. On this checkpoint that range is 200,000-600,000 px
  (0.20-0.60 MP), so:
      level 0: 0.20-0.24 MP      level 5: 0.40-0.44 MP
      level 1: 0.24-0.28 MP      level 9: 0.56-0.60 MP
  A 518x518 crop is 0.268 MP -> native to LEVEL 1. Left as None, the model
  keeps the full 0.2-0.6 MP range and a 0.268 MP crop passes through
  unresized (it also emits a "resolution_level not set" warning). Setting a
  level above 1 only UPSAMPLES your crop: more compute, no new information.
"""

import sys
from pathlib import Path

import torch
import cv2
import numpy as np

# Global state for UniK3D availability
UNIK3D_AVAILABLE = False
UNIK3D_PATH = None


def load_unik3d_from_path(local_path):
    """Load UniK3D from a local cloned directory."""
    global UNIK3D_AVAILABLE, UNIK3D_PATH

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[UniK3D] Local path not found: {local_path}")
        return False

    sys.path.insert(0, str(local_path))
    try:
        from unik3d.models.unik3d import UniK3D  # noqa: F401
        UNIK3D_AVAILABLE = True
        UNIK3D_PATH = local_path
        print(f"[UniK3D] Loaded from local clone: {local_path}")
        return True
    except ImportError as e:
        print(f"[UniK3D] Failed to import from {local_path}: {e}")
        return False


def is_unik3d_available():
    """Live check of module-level availability."""
    return UNIK3D_AVAILABLE


# Try installed package first
try:
    from unik3d.models.unik3d import UniK3D  # noqa: F401
    UNIK3D_AVAILABLE = True
    print("[UniK3D] Loaded from installed package")
except ImportError:
    pass

if not UNIK3D_AVAILABLE:
    print("[UniK3D] Not available yet (will check --unik3d-path)")


class UniK3DProcessor:
    """Manages UniK3D model and inference.

    Provides metric depth (meters, z-depth), 3D point clouds in camera
    coordinates, and unprojected rays, following the official UniK3D API.
    """

    def __init__(self, model_id="lpiccinelli/unik3d-vits", device="cuda:0",
                 resolution_level=None):
        """Initialize UniK3D processor.

        Args:
            model_id: HuggingFace id, e.g. lpiccinelli/unik3d-vit{s,b,l}
            device: Torch device (e.g. "cuda:0" or "cpu")
            resolution_level: int in [0, 9] selecting the input pixel-count
                band (see module docstring). None = library default (full
                0.2-0.6 MP range, no narrowing). For a 518x518 crop the
                native band is level 1.
        """
        self.device = device
        self.model = None
        self.resolution_level = resolution_level

        try:
            from unik3d.models.unik3d import UniK3D

            print(f"[UniK3D] Loading model: {model_id}")
            self.model = UniK3D.from_pretrained(model_id)

            if resolution_level is not None:
                if not (0 <= int(resolution_level) < 10):
                    raise ValueError("resolution_level must be in [0, 9]")
                try:
                    self.model.resolution_level = int(resolution_level)
                    print(f"[UniK3D] resolution_level={resolution_level}")
                except Exception as e:
                    print(f"[UniK3D] resolution_level not settable: {e}")

            self.model = self.model.to(device).eval()
            print("[UniK3D] Model loaded successfully")
        except Exception as e:
            print(f"[UniK3D] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def is_available(self):
        """Check if model is available and loaded."""
        return self.model is not None

    def pixel_bounds(self):
        """Return (pixels_min, pixels_max) for the ACTIVE configuration.

        Useful for reporting the operating point: with resolution_level set,
        this is the narrowed band; otherwise the model's full range.
        """
        if self.model is None:
            return None
        try:
            sc = self.model.shape_constraints
            lo, hi = sc["pixels_min"], sc["pixels_max"]
        except Exception:
            return None
        if self.resolution_level is None:
            return (lo, hi)
        iv = (hi - lo) / 10.0
        L = int(self.resolution_level)
        return (L * iv + lo, (L + 1) * iv + lo)

    def _make_camera(self, K):
        """Build a UniK3D Pinhole camera from a 3x3 intrinsics matrix.

        UniK3D's Pinhole accepts params=[fx, fy, cx, cy] (or K=). Verified
        functional: the model's depth responds to changes in fx.
        """
        try:
            from unik3d.utils.camera import Pinhole
        except ImportError:
            print("[UniK3D] Camera conditioning unavailable "
                  "(unik3d.utils.camera not found); running uncalibrated.")
            return None

        K = np.asarray(K, dtype=np.float32)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        params = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        return Pinhole(params=params).to(self.device)

    # ------------------------------------------------------------------
    # Core inference: metric depth, point cloud, rays
    # ------------------------------------------------------------------
    def infer_metric(self, frame_rgb_uint8, K=None,
                     return_points=True, return_rays=False):
        """Run UniK3D on a single frame and return metric outputs.

        Official API: the raw uint8 RGB tensor (C, H, W) is passed directly to
        model.infer() -- normalization and internal resizing are handled by
        the model. Do NOT pre-normalize or resize.

        Args:
            frame_rgb_uint8: uint8 RGB image (H, W, 3)
            K: optional 3x3 pinhole intrinsics for the input frame. When
               provided, UniK3D is conditioned on the true camera, which
               anchors the metric scale. UniK3D also works WITHOUT it (it
               self-predicts the camera), so a silent failure here shows up
               only as a scale offset -- watch the scale column.
            return_points: include the 3D point cloud (camera coords)
            return_rays: include the unprojected rays

        Returns:
            dict or None:
                'depth'  : float32 Z-DEPTH in meters, (H, W), resized to the
                           input frame size
                'points' : float32 (3, H', W') camera-coordinate point cloud
                'rays'   : float32 (3, H', W') unprojected rays
        """
        if self.model is None:
            return None

        try:
            rgb = torch.from_numpy(np.ascontiguousarray(frame_rgb_uint8)) \
                       .permute(2, 0, 1).to(self.device)  # C, H, W

            camera = self._make_camera(K) if K is not None else None

            with torch.no_grad():
                if camera is not None:
                    predictions = self.model.infer(rgb, camera)
                else:
                    predictions = self.model.infer(rgb)

            depth = predictions["depth"].squeeze().float().cpu().numpy()

            # Resize back to the original frame size. Values are metric
            # z-depth; interpolation preserves the scale.
            h, w = frame_rgb_uint8.shape[:2]
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h),
                                   interpolation=cv2.INTER_LINEAR)

            result = {"depth": depth}

            if return_points and "points" in predictions:
                result["points"] = (predictions["points"].squeeze(0)
                                    .float().cpu().numpy())

            if return_rays and "rays" in predictions:
                result["rays"] = (predictions["rays"].squeeze(0)
                                  .float().cpu().numpy())

            return result

        except Exception as e:
            print(f"[UniK3D] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Backwards-compatible API: normalized uint8 depth for visualization
    # ------------------------------------------------------------------
    def infer(self, frame_rgb_uint8, K=None):
        """Normalized uint8 depth map, for VISUALISATION ONLY.

        WARNING: this returns 0-255, NOT metres. Benchmarking must use
        infer_metric()["depth"].
        """
        result = self.infer_metric(frame_rgb_uint8, K=K,
                                   return_points=False, return_rays=False)
        if result is None:
            return None

        depth = result["depth"]
        if depth.size == 0:
            return None

        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max > depth_min:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min)
                          * 255.0).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        return depth_norm
    
    def infer_equirect_metric(self, rgb_equi, return_points=False,
                              return_rays=False):
        """Run UniK3D on a full equirectangular image (Spherical camera).
 
        Args:
            rgb_equi: (H, W, 3) uint8 RGB panorama, full 360 horizontally.
                      W should be 2*H for a standard equirect.
            return_points / return_rays: include those outputs in the dict.
 
        Returns:
            dict with "depth" (H, W) float32 RADIAL distance in metres,
            optionally "points" (H, W, 3) and "rays" (H, W, 3), or None.
        """
        if not self.is_available:
            return None
 
        try:
            import numpy as _np
            import torch as _torch
            from unik3d.utils.camera import Spherical
 
            H, W = rgb_equi.shape[:2]
            if abs(W - 2 * H) > 2:
                print(f"[UniK3D] WARNING: {W}x{H} is not 2:1; the Spherical "
                      f"model assumes a full 360x180 panorama.")
 
            rgb_t = _torch.from_numpy(
                _np.ascontiguousarray(rgb_equi)).permute(2, 0, 1)
 
            # full equirect: horizontal half-FOV = pi, vertical = pi*H/W
            hfov_half = _np.pi
            vfov_half = _np.pi * H / W
            if vfov_half > _np.pi / 2 + 1e-6:
                print(f"[UniK3D] WARNING: vfov_half {vfov_half:.3f} > pi/2; "
                      f"image is taller than a hemisphere.")
 
            camera = Spherical(params=_torch.tensor(
                [1.0] * 4 + [float(W), float(H),
                             float(hfov_half), float(vfov_half)]))
 
            with _torch.no_grad():
                outputs = self.model.infer(rgb=rgb_t, camera=camera,
                                           normalize=True)
 
            out = {"depth": outputs["depth"].squeeze()
                   .float().cpu().numpy().astype(_np.float32)}
            if return_points and "points" in outputs:
                out["points"] = (outputs["points"].squeeze()
                                 .permute(1, 2, 0).float().cpu().numpy())
            if return_rays and "rays" in outputs:
                out["rays"] = (outputs["rays"].squeeze()
                               .permute(1, 2, 0).float().cpu().numpy())
            return out
 
        except Exception as e:
            print(f"[UniK3D] equirect inference error: {e}")
            import traceback
            traceback.print_exc()
            return None
 