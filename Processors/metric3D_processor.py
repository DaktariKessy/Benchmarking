#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric3D v2 (ViT) depth estimation processor.

Interface matches the other benchmark processors: availability flags,
local clone loading, and infer(rgb, K) -> float32 depth in meters or None.

DE-CANONICAL SCALE (important):
Metric3D predicts depth in a canonical camera space (focal = 1000). The
network sees the image AFTER the keep-ratio resize to INPUT_SIZE, so the
focal length it effectively observes is fx * scale. Following the
official hubconf recipe -- which scales the intrinsics alongside the
image ("remember to scale intrinsic, hold depth") and then divides by the
canonical focal -- the de-canonical factor is:

    depth_metric = depth_canonical * (fx * scale) / 1000

Upsampling the prediction back to the original resolution resamples it
spatially but does NOT change depth values, so the factor is unaffected
by that step.

A thin unpad-boundary ring can leak near-zero depth (padding bleeding one
or two pixels into the valid area). It is trimmed, and depths below the
model's valid range are rejected, so no garbage points reach the
downstream point cloud / ICP.

Surface normals (a Metric3D v2 feature the other benchmark models do not
provide) are extracted when available and stored in `last_normal` for
optional use in point-to-plane ICP.
"""

import sys
from pathlib import Path

import torch
import cv2
import numpy as np


# Global state for Metric3D availability
METRIC3D_AVAILABLE = False
METRIC3D_PATH = None


def load_metric3d_from_path(local_path):
    """Register a local Metric3D clone so models can be built from it."""
    global METRIC3D_AVAILABLE, METRIC3D_PATH

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[Metric3D] Local path not found: {local_path}")
        return False
    if not (local_path / "hubconf.py").exists():
        print(f"[Metric3D] No hubconf.py in {local_path} - is this the "
              f"Metric3D repo root?")
        return False

    METRIC3D_AVAILABLE = True
    METRIC3D_PATH = local_path
    print(f"[Metric3D] Using local clone: {local_path}")
    return True


def is_metric3d_available():
    """Live check (avoids stale `from ... import METRIC3D_AVAILABLE`)."""
    return METRIC3D_AVAILABLE


class Metric3DProcessor:
    """Wrapper for Metric3D v2 depth estimation."""

    # ViT (v2) models expect this input size; ConvNeXt (v1) uses (544, 1216)
    INPUT_SIZE = (616, 1064)  # (h, w)
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    CANONICAL_FOCAL = 1000.0

    # post-processing cleanup
    BORDER_TRIM_PX = 0      # trim this many pixels off each edge
    MIN_VALID_DEPTH = 0  # meters; ViT models' depth bins start at 0.1 m

    def __init__(self, variant="metric3d_vit_large", checkpoint=None,
                 device="cuda", want_normals=True):
        """
        Args:
            variant: metric3d_vit_small | metric3d_vit_large |
                     metric3d_vit_giant2
            checkpoint: optional path to a local .pth. If None, weights
                        are downloaded via torch.hub (pretrain=True).
            device: 'cuda' or 'cpu'
            want_normals: keep the predicted surface normal map in
                          `last_normal` (Metric3D v2 predicts it anyway).
        """
        self.device = device
        self.variant = variant
        self.want_normals = want_normals
        self.last_normal = None   # (H, W, 3) float32 unit normals, or None
        self._available = False
        self._warned_no_K = False

        try:
            print(f"[Metric3D] Loading {variant}...")
            if METRIC3D_PATH is not None:
                # Build from the local clone
                self.model = torch.hub.load(
                    str(METRIC3D_PATH), variant, source="local",
                    pretrain=(checkpoint is None), trust_repo=True,
                )
            else:
                # Fall back to fetching the repo from GitHub
                self.model = torch.hub.load(
                    "yvanyin/metric3d", variant,
                    pretrain=(checkpoint is None), trust_repo=True,
                )

            if checkpoint is not None:
                ckpt = torch.load(checkpoint, map_location="cpu")
                sd = ckpt.get("model_state_dict", ckpt)
                # strip DDP 'module.' prefixes if present
                sd = {(k[7:] if k.startswith("module.") else k): v
                      for k, v in sd.items()}
                missing, unexpected = self.model.load_state_dict(
                    sd, strict=False)
                if missing:
                    print(f"[Metric3D] load_state_dict: {len(missing)} "
                          f"missing keys (first: {missing[0]})")
                if unexpected:
                    print(f"[Metric3D] load_state_dict: {len(unexpected)} "
                          f"unexpected keys (first: {unexpected[0]})")
                print(f"[Metric3D] Loaded checkpoint: {checkpoint}")

            self.model.eval().to(device)
            self._available = True
            print("[Metric3D] \u2713 Model loaded successfully")
        except Exception as e:
            print(f"[Metric3D] \u2717 Failed to load: {e}")
            import traceback
            traceback.print_exc()
            self._available = False

    def is_available(self):
        """Check if model is available."""
        return self._available

    def infer(self, rgb_image, K=None):
        """
        Estimate metric depth from an RGB image.

        Official hubconf recipe: keep-ratio resize to INPUT_SIZE, pad with
        the channel means, normalize, forward, unpad, upsample back to the
        original resolution, de-canonicalize with (fx * scale) / 1000.

        Args:
            rgb_image: numpy uint8 array (H, W, 3) in RGB format
            K: 3x3 pinhole intrinsics at THIS image's resolution.
               REQUIRED for metric scale (fx = K[0,0]). If None, canonical
               depth is returned and the scale is NOT metric.

        Returns:
            depth_map: float32 numpy array (H, W) in meters (0 = invalid),
                       or None on failure.
        """
        if not self._available:
            return None

        try:
            h0, w0 = rgb_image.shape[:2]

            # --- keep-ratio resize to fit the model input ---
            ih, iw = self.INPUT_SIZE
            scale = min(ih / h0, iw / w0)
            nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
            resized = cv2.resize(rgb_image, (nw, nh),
                                 interpolation=cv2.INTER_LINEAR)

            # --- pad to INPUT_SIZE with channel means ---
            pad_h, pad_w = ih - nh, iw - nw
            pad_top, pad_left = pad_h // 2, pad_w // 2
            pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
            padded = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=self.MEAN.tolist(),
            )

            # --- normalize, to tensor (1, C, H, W) ---
            x = (padded.astype(np.float32) - self.MEAN) / self.STD
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            x = x.to(self.device)

            with torch.no_grad():
                pred_depth, confidence, output_dict = self.model.inference(
                    {"input": x})

            # --- unpad, then upsample back to the ORIGINAL resolution ---
            d = pred_depth.squeeze()
            d = d[pad_top: d.shape[0] - pad_bottom,
                  pad_left: d.shape[1] - pad_right]
            d = torch.nn.functional.interpolate(
                d[None, None], size=(h0, w0), mode="bilinear",
                align_corners=False,
            ).squeeze()

            # --- de-canonical: the network saw the RESIZED image, whose
            #     focal length is fx * scale (official hubconf recipe) ---
            if K is not None:
                fx = float(np.asarray(K)[0, 0])
                d = d * (fx * scale / self.CANONICAL_FOCAL)
            else:
                if not self._warned_no_K:
                    self._warned_no_K = True
                    print("[Metric3D] !! No intrinsics provided; returning "
                          "CANONICAL-space depth (scale is NOT metric) !!")

            d = d.clamp(0, 300).float().cpu().numpy().astype(np.float32)

            # --- optional: surface normals (Metric3D v2 only) ---
            self.last_normal = None
            if self.want_normals and "prediction_normal" in output_dict:
                try:
                    n = output_dict["prediction_normal"][:, :3].squeeze()
                    n = n[:, pad_top: n.shape[1] - pad_bottom,
                          pad_left: n.shape[2] - pad_right]
                    n = torch.nn.functional.interpolate(
                        n[None], size=(h0, w0), mode="bilinear",
                        align_corners=False).squeeze(0)
                    n = torch.nn.functional.normalize(n, dim=0)
                    self.last_normal = (n.permute(1, 2, 0).float().cpu()
                                        .numpy().astype(np.float32))
                except Exception as ne:
                    print(f"[Metric3D] normal extraction skipped: {ne}")

            # --- cleanup: trim unpad-boundary ring + reject near garbage ---
            m = self.BORDER_TRIM_PX
            if m > 0 and d.shape[0] > 2 * m and d.shape[1] > 2 * m:
                d[:m, :] = 0.0
                d[-m:, :] = 0.0
                d[:, :m] = 0.0
                d[:, -m:] = 0.0
            d[d < self.MIN_VALID_DEPTH] = 0.0  # 0 = invalid downstream

            return d

        except Exception as e:
            print(f"[Metric3D] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None