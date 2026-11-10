#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UniDAC depth estimation processor (CVPR 2026).

Mirrors the UniDepthV2/Metric3D processor interface: availability
flags, local clone loading, and infer(rgb, K) -> float32 depth in
meters (z-depth) or None.

UniDAC operates in ERP (equirectangular) space. For a pinhole crop the
pipeline is:
    crop -> gnomonic projection to ERP patch (cam_to_erp_patch_fast)
         -> resize to fwd_sz -> model -> * pred_scale_factor
         -> unpad -> resize back to patch size
         -> inverse projection to the camera view (erp_patch_to_cam_fast)
         -> euclidean-to-z conversion (model predicts RADIAL distance)

K (fx, fy, cx, cy at the crop resolution) is REQUIRED: it drives both
the ERP projection and the euclid->z conversion. Without it, None is
returned.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF



# Global state for UniDAC availability
UNIDAC_AVAILABLE = False
UNIDAC_PATH = None


def load_unidac_from_path(local_path):
    """Load UniDAC from a local cloned directory."""
    global UNIDAC_AVAILABLE, UNIDAC_PATH

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[UniDAC] Local path not found: {local_path}")
        return False

    sys.path.insert(0, str(local_path))
    try:
        from unidac.models.unidac import UniDAC  # noqa: F401
        UNIDAC_AVAILABLE = True
        UNIDAC_PATH = local_path
        print(f"[UniDAC] Loaded from local clone: {local_path}")
        return True
    except ImportError as e:
        print(f"[UniDAC] Failed to import from {local_path}: {e}")
        return False


def is_unidac_available():
    """Live check (avoids stale `from ... import UNIDAC_AVAILABLE`)."""
    return UNIDAC_AVAILABLE


class UniDACProcessor:
    """Wrapper for UniDAC depth estimation on pinhole crops."""

    NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    def __init__(self, checkpoint, config_file=None, device="cuda",
                 fov_buffer_deg=20.0):
        """
        Args:
            checkpoint: path to unidac.pt (from HF girish1511/UniDAC)
            config_file: model config json. Defaults to the indoor NYU
                         test config inside the local clone (pinhole).
            device: 'cuda' or 'cpu'
            fov_buffer_deg: extra angular margin added to the crop FOV
                            when sizing the ERP patch (README: original
                            FOV + some buffer).
        """
        self.device = device
        self.fov_buffer_deg = float(fov_buffer_deg)
        self._available = False
        self._warned_no_K = False

        try:
            from unidac.models.unidac import UniDAC

            if config_file is None:
                if UNIDAC_PATH is None:
                    raise RuntimeError(
                        "call load_unidac_from_path() first or pass "
                        "config_file explicitly")
                config_file = str(
                    UNIDAC_PATH / "configs/test/"
                    "dac_dinov3l+dpt_indoor_test_nyu.json")
            with open(config_file, "r") as f:
                self.config = json.load(f)

            self.cano_sz = self.config["data"]["cano_sz"]     # e.g. [1400, 1400]
            self.fwd_sz = self.config["data"]["fwd_sz"]       # e.g. [512, 704]

            print(f"[UniDAC] Loading model (config={Path(config_file).name}, "
                  f"cano_sz={self.cano_sz}, fwd_sz={self.fwd_sz})...")
            self.model = UniDAC.build(self.config)
            self.model.load_pretrained(checkpoint)
            self.model = self.model.to(device)
            self.model.eval()
            self._available = True
            print("[UniDAC] \u2713 Model loaded successfully")
        except Exception as e:
            print(f"[UniDAC] \u2717 Failed to load: {e}")
            import traceback
            traceback.print_exc()
            self._available = False

    def is_available(self):
        """Check if model is available."""
        return self._available

    def infer(self, rgb_image, K=None):
        """
        Estimate metric depth from a pinhole RGB crop.

        Args:
            rgb_image: numpy uint8 array (H, W, 3) in RGB format
            K: 3x3 pinhole intrinsics at THIS image's resolution.
               REQUIRED (ERP projection + euclid->z need it).

        Returns:
            depth_map: float32 numpy array (H, W), Z-DEPTH in meters
                       (ready for standard pinhole back-projection),
                       or None on failure.
        """
        if not self._available:
            return None
        if K is None:
            if not self._warned_no_K:
                self._warned_no_K = True
                print("[UniDAC] !! Intrinsics are required for the ERP "
                      "projection; returning None !!")
            return None

        try:
            from unidac.utils.erp_geometry import (
                cam_to_erp_patch_fast, erp_patch_to_cam_fast)
            from unidac.dataloaders.dataset import resize_for_input

            K = np.asarray(K, dtype=np.float64)
            h0, w0 = rgb_image.shape[:2]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # 'dataset' must avoid the fisheye/MEI branches; any other
            # name plus fx/fy/cx/cy selects the pinhole path.
            cam_params = {"dataset": "pinhole_crop",
                          "fx": fx, "fy": fy, "cx": cx, "cy": cy}

            # --- size the ERP patch: crop FOV + buffer ---
            fov_x = math.degrees(2 * math.atan((w0 / 2.0) / fx))
            crop_wfov = min(fov_x + self.fov_buffer_deg, 180.0)
            cano_h = self.cano_sz[0]
            patch_w = int(cano_h * crop_wfov / 180)
            patch_h = int(patch_w * self.fwd_sz[0] / self.fwd_sz[1])

            # --- pinhole crop -> ERP patch (gnomonic) ---
            img = rgb_image.astype(np.float32) / 255.0
            dummy_depth = np.zeros((h0, w0, 1), dtype=np.float32)
            dummy_mask = np.ones((h0, w0, 1), dtype=np.float32)
            theta = 0.0
            phi = np.array(0, dtype=np.float32)
            roll = np.array(0, dtype=np.float32)

            erp_img, erp_depth, _, erp_mask, lat, lon = cam_to_erp_patch_fast(
                img, dummy_depth, dummy_mask, theta, phi,
                patch_h, patch_w, cano_h, cano_h * 2,
                cam_params, roll, scale_fac=None,
            )
            lat_range = torch.tensor(
                [float(np.min(lat)), float(np.max(lat))])
            long_range = torch.tensor(
                [float(np.min(lon)), float(np.max(lon))])

            # --- resize to model input ---
            (erp_img_r, _, pad, pred_scale_factor, attn_mask,
             lat_grid, long_grid) = resize_for_input(
                (erp_img * 255.0).astype(np.uint8), erp_depth,
                self.fwd_sz, None, [erp_img.shape[0], erp_img.shape[1]],
                1.0, padding_rgb=[0, 0, 0], mask=erp_mask,
                lat_grid=lat, long_grid=lon,
            )

            # --- to tensors, forward ---
            x = TF.normalize(TF.to_tensor(erp_img_r), **self.NORM)
            x = x.unsqueeze(0).to(self.device)
            attn = TF.to_tensor(
                (attn_mask > 0).astype(np.float32)).unsqueeze(0).to(self.device)
            lat_g = torch.tensor(lat_grid).unsqueeze(0).to(self.device)
            lat_r = lat_range.unsqueeze(0).to(self.device)
            long_r = long_range.unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds, _, _ = self.model(x, lat_r, long_r,
                                         attn_mask=attn, lat_grid=lat_g)
            preds = preds * pred_scale_factor

            # --- unpad (pad = [top, bottom, left, right]) ---
            d = preds.squeeze()
            top, bottom, left, right = pad
            d = d[top: d.shape[0] - bottom, left: d.shape[1] - right]

            # --- back to ERP patch resolution ---
            d = torch.nn.functional.interpolate(
                d[None, None], size=(patch_h, patch_w), mode="nearest",
            ).squeeze(0)  # (1, patch_h, patch_w)

            # --- ERP patch -> camera view ---
            img_erp_t = torch.from_numpy(
                erp_img.transpose(2, 0, 1)).float().to(d.device)
            mask_t = torch.ones_like(d)
            out = erp_patch_to_cam_fast(
                img_erp_t, d, mask_t, theta, float(phi),
                h0, w0, cano_h, cano_h * 2, cam_params,
            )
            depth_cam = out[1].squeeze().cpu().numpy()  # euclidean dist

            # --- euclidean (radial) -> z-depth for back-projection ---
            u, v = np.meshgrid(np.arange(w0), np.arange(h0))
            xn = (u - cx) / fx
            yn = (v - cy) / fy
            ray_norm = np.sqrt(xn ** 2 + yn ** 2 + 1.0)
            depth_z = depth_cam / ray_norm

            return depth_z.astype(np.float32)

        except Exception as e:
            print(f"[UniDAC] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

        # ============================================================================
# ADD THIS METHOD to UniDACProcessor in UniDAC_processor.py
# ============================================================================
# Native ERP inference: the panorama goes in as-is. This mirrors the demo's
# `sample["erp"] is True` branch (INDOOR_SAMPLE_1 / matterport3d), which
# differs from the pinhole path in four ways:
#
#   1. NO cam_to_erp_patch_fast. The input is already equirectangular, so
#      there is nothing to gnomonically project.
#   2. lat/long ranges are the WHOLE sphere: lat [-pi/2, pi/2],
#      long [-pi, pi], with grids from linspace over the full image.
#   3. attn_mask is all ones -- every pixel of a full panorama is valid,
#      unlike an ERP patch where only the projected wedge is.
#   4. NO erp_patch_to_cam_fast and NO euclid->z at the end. The prediction
#      stays on the sphere, where depth is RADIAL distance. Converting to z
#      would be meaningless: there is no optical axis on a panorama, and z
#      goes negative beyond 90 degrees off whatever axis you picked.
#
# CONFIG: this needs the ERP config, not the pinhole one. Build the processor
# with configs/test/dac_dinov3l+dpt_indoor_test_m3d.json, whose fwd_sz is
# (512, 1024) -- 2:1, matching a panorama. The NYU pinhole config's (512, 704)
# would squash the aspect ratio.
#
# ONE DEVIATION FROM THE DEMO, deliberate: the demo builds attn_mask at the
# ORIGINAL image size and never resizes it. That works there only because
# matterport3d is already 512x1024 = exactly fwd_sz, so the shapes coincide.
# A 3840x1920 Theta frame is not, and the un-resized mask would be the wrong
# shape for the batch. Here the mask is built at fwd_sz directly.

    def infer_erp(self, rgb_equi, out_hw=None):
        """Estimate metric depth from a FULL equirectangular image.

        Args:
            rgb_equi: numpy uint8 (H, W, 3) RGB panorama, W = 2*H.
            out_hw: (h, w) to resize the prediction to. None keeps the
                    model's own output resolution, which is what you want if
                    you are inspecting the raw output.

        Returns:
            float32 (h, w) RADIAL distance in metres -- distance from the
            optical centre along each ray, NOT z. None on failure.
        """
        if not self.is_available():
            return None

        try:
            from unidac.dataloaders.dataset import resize_for_input

            H, W = rgb_equi.shape[:2]
            if abs(W - 2 * H) > 2:
                print(f"[UniDAC] WARNING: {W}x{H} is not 2:1; the ERP path "
                      f"assumes a full 360x180 panorama.")

            fwd_h, fwd_w = int(self.fwd_sz[0]), int(self.fwd_sz[1])
            cano_h = int(self.cano_sz[0])

            # whole sphere, exactly as the demo's ERP branch
            lat_range = torch.tensor([-np.pi / 2, np.pi / 2],
                                     dtype=torch.float32)
            long_range = torch.tensor([-np.pi, np.pi], dtype=torch.float32)
            latitude, longitude = np.meshgrid(
                np.linspace(-np.pi / 2, np.pi / 2, H),
                np.linspace(-np.pi, np.pi, W), indexing="ij")

            dummy_depth = np.zeros((H, W), dtype=np.float32)
            to_cano_ratio = cano_h / float(H)

            (img_r, _, pad, pred_scale_factor,
             lat_grid, long_grid) = resize_for_input(
                rgb_equi, dummy_depth, self.fwd_sz, None,
                self.cano_sz, to_cano_ratio,
                lat_grid=latitude, long_grid=longitude)

            # every pixel of a full panorama is valid; built at fwd_sz so the
            # batch shapes agree for any input resolution (see note above).
            attn_np = np.ones((img_r.shape[0], img_r.shape[1]),
                              dtype=np.float32)

            x = TF.normalize(TF.to_tensor(img_r), **self.NORM)
            x = x.unsqueeze(0).to(self.device)
            attn = TF.to_tensor(attn_np).unsqueeze(0).to(self.device)
            lat_g = torch.tensor(lat_grid).unsqueeze(0).to(self.device)
            lat_r = lat_range.unsqueeze(0).to(self.device)
            long_r = long_range.unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds, _, _ = self.model(x, lat_r, long_r,
                                         attn_mask=attn, lat_grid=lat_g)
            preds = preds * pred_scale_factor

            # unpad (pad = [top, bottom, left, right])
            d = preds.squeeze()
            top, bottom, left, right = pad
            if top or bottom or left or right:
                d = d[top: d.shape[0] - bottom, left: d.shape[1] - right]

            if out_hw is not None:
                d = torch.nn.functional.interpolate(
                    d[None, None], size=(int(out_hw[0]), int(out_hw[1])),
                    mode="nearest").squeeze()

            # RADIAL distance on the sphere. No euclid->z here, on purpose.
            return d.float().cpu().numpy().astype(np.float32)

        except Exception as e:
            print(f"[UniDAC] ERP inference error: {e}")
            import traceback
            traceback.print_exc()
            return None