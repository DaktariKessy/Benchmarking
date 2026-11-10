#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video Depth Anything processor with verified checkpoint loading.

CHANGES vs the previous version (all in response to review of the checkpoint
loading path):

  1. `_try_load_checkpoint` is renamed `_load_checkpoint` and now builds the
     filename from BOTH `self.encoder` and `self.metric`. The hardcoded
     '..._vits.pth' path is gone, so the backbone no longer has to be selected
     by hand-editing this file between runs.

  2. Loading failure is FATAL. A missing file, a shape mismatch under
     strict=True, or any other exception raises CheckpointLoadError. The old
     code caught the exception, printed a line, and left `self._available =
     True` with randomly initialised weights, so a mismatched backbone would
     silently benchmark an untrained network.

  3. Loading is VERIFIED, not assumed. We hash the model's parameters before
     and after `load_state_dict`; if the hash is unchanged, nothing was
     actually loaded and we raise. This detects the exact failure mode above
     even if some future refactor reintroduces a swallowed exception.

  4. Provenance is RECORDED. `self.provenance` carries the checkpoint path,
     the SHA-256 of the checkpoint file on disk, the SHA-256 of the loaded
     model parameters, the parameter count, dtype, encoder and metric flag.
     Callers write this next to their results so that any reported number can
     be traced back to the bytes that produced it.

  5. `reset_temporal_state()` exposes the stream model's temporal buffer so a
     caller can guarantee no state carries across sequences.

Public API used by the eval script:
    VideoDepthAnythingProcessor(encoder, device, metric, fp32, checkpoint_dir)
    .is_available() -> bool
    .provenance     -> dict
    .infer(rgb, input_size) -> np.ndarray | None
    .reset_temporal_state() -> bool
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


VIDEO_DEPTH_ANYTHING_AVAILABLE = False
VIDEO_DEPTH_ANYTHING_PATH = None


class CheckpointLoadError(RuntimeError):
    """Raised when weights could not be loaded, or could not be shown to load."""


def load_video_depth_anything_from_path(local_path):
    """Make a local Video-Depth-Anything clone importable."""
    global VIDEO_DEPTH_ANYTHING_AVAILABLE, VIDEO_DEPTH_ANYTHING_PATH

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[VideoDepthAnything] Local path not found: {local_path}")
        return False

    sys.path.insert(0, str(local_path))
    try:
        from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: F401
        VIDEO_DEPTH_ANYTHING_AVAILABLE = True
        VIDEO_DEPTH_ANYTHING_PATH = local_path
        print(f"[VideoDepthAnything] Loaded from local clone: {local_path}")
        return True
    except ImportError as e:
        print(f"[VideoDepthAnything] Failed to import from {local_path}: {e}")
        return False


try:
    from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: F401
    VIDEO_DEPTH_ANYTHING_AVAILABLE = True
    print("[VideoDepthAnything] Loaded from installed package")
except ImportError:
    pass


# ----------------------------------------------------------------------------
# hashing helpers
# ----------------------------------------------------------------------------

def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 of a file on disk. Identifies the checkpoint bytes themselves."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_bytes(t):
    """Deterministic bytes for a tensor, handling dtypes numpy cannot hold."""
    t = t.detach().to("cpu")
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.float()
    return np.ascontiguousarray(t.numpy()).tobytes()


def sha256_state(module_or_state):
    """SHA-256 over parameter names, shapes, dtypes and values.

    Applied to a live nn.Module this answers "what weights are actually in the
    network right now", which is the question the file hash alone cannot
    settle (the file could exist and still never have been read).
    """
    if isinstance(module_or_state, torch.nn.Module):
        state = module_or_state.state_dict()
    else:
        state = module_or_state

    h = hashlib.sha256()
    for k in sorted(state.keys()):
        v = state[k]
        h.update(k.encode("utf-8"))
        if torch.is_tensor(v):
            h.update(str(tuple(v.shape)).encode("utf-8"))
            h.update(str(v.dtype).encode("utf-8"))
            h.update(_tensor_bytes(v))
        else:
            h.update(repr(v).encode("utf-8"))
    return h.hexdigest()


# ----------------------------------------------------------------------------
# processor
# ----------------------------------------------------------------------------

class VideoDepthAnythingProcessor:
    """Video Depth Anything wrapper with verified, encoder-aware weight loading."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,
                 "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128,
                 "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256,
                 "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, encoder="vits", device=None, metric=True, fp32=False,
                 checkpoint_dir="./checkpoints", checkpoint_path=None,
                 strict=True, verbose=True):
        """
        Args:
            encoder: 'vits' | 'vitb' | 'vitl'. Selects BOTH the architecture and
                the checkpoint filename; the two can no longer disagree.
            metric: True loads metric_video_depth_anything_<encoder>.pth,
                False loads video_depth_anything_<encoder>.pth.
            checkpoint_path: explicit override. When given, its filename is
                checked against `encoder`/`metric` and a mismatch raises.
            strict: passed to load_state_dict. Leave True.

        Raises:
            CheckpointLoadError: model built but weights not verifiably loaded.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.metric = bool(metric)
        self.fp32 = bool(fp32)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.strict = bool(strict)
        self.verbose = bool(verbose)

        self.model = None
        self._available = False
        self.provenance = {}
        self._frames_seen = 0

        if encoder not in self.MODEL_CONFIGS:
            raise CheckpointLoadError(
                f"Unknown encoder {encoder!r}; expected one of "
                f"{sorted(self.MODEL_CONFIGS)}")

        from video_depth_anything.video_depth_stream import VideoDepthAnything

        cfg = self.MODEL_CONFIGS[encoder]
        if self.verbose:
            kind = "metric" if self.metric else "relative"
            print(f"[VideoDepthAnything] Building {encoder} ({kind}) ...")

        self.model = VideoDepthAnything(**cfg)
        self.model.eval()
        self.model = self.model.to(self.device)

        # Fatal on failure; sets self.provenance on success.
        self._load_checkpoint(explicit_path=checkpoint_path)

        for p in self.model.parameters():
            p.requires_grad = False

        self._available = True

        if self.verbose:
            p = self.provenance
            print(f"[VideoDepthAnything] checkpoint : {p['checkpoint_name']}")
            print(f"[VideoDepthAnything] file sha256: {p['checkpoint_sha256']}")
            print(f"[VideoDepthAnything] weights sha: {p['loaded_weights_sha256']}")
            print(f"[VideoDepthAnything] parameters : "
                  f"{p['n_parameters']:,} ({p['n_parameters'] / 1e6:.1f} M)")

    # ------------------------------------------------------------------ load

    def expected_checkpoint_name(self):
        prefix = "metric_video_depth_anything" if self.metric \
            else "video_depth_anything"
        return f"{prefix}_{self.encoder}.pth"

    def _load_checkpoint(self, explicit_path=None):
        expected = self.expected_checkpoint_name()

        if explicit_path is not None:
            path = Path(explicit_path)
            # An explicit path is allowed, but it must be consistent with the
            # architecture we just built. Silent inconsistency here is exactly
            # what produced the original problem.
            if path.name != expected:
                raise CheckpointLoadError(
                    f"checkpoint_path {path.name!r} does not match the "
                    f"requested configuration (encoder={self.encoder}, "
                    f"metric={self.metric}), which expects {expected!r}. "
                    f"Rename the file or fix the arguments; the loader will "
                    f"not guess.")
        else:
            path = self.checkpoint_dir / expected

        if not path.exists():
            raise CheckpointLoadError(
                f"Checkpoint not found: {path}\n"
                f"  encoder={self.encoder} metric={self.metric} "
                f"expects {expected}\n"
                f"  (Refusing to run with randomly initialised weights.)")

        file_sha = sha256_file(path)
        sha_before = sha256_state(self.model)

        try:
            state = torch.load(str(path), map_location="cpu")
        except Exception as e:
            raise CheckpointLoadError(f"torch.load failed for {path}: {e}") from e

        # Some releases wrap the weights one level down.
        if isinstance(state, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        try:
            incompatible = self.model.load_state_dict(state, strict=self.strict)
        except Exception as e:
            raise CheckpointLoadError(
                f"load_state_dict failed for {path} with encoder="
                f"{self.encoder}.\n"
                f"  This usually means the checkpoint and the architecture "
                f"disagree.\n"
                f"  Underlying error: {e}") from e

        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        if missing or unexpected:
            raise CheckpointLoadError(
                f"Incomplete load for {path}: {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys.")

        sha_after = sha256_state(self.model)
        if sha_after == sha_before:
            # load_state_dict returned without error but changed nothing.
            raise CheckpointLoadError(
                f"Model parameters are unchanged after loading {path}. "
                f"The network still holds its initialisation; refusing to run.")

        n_params = sum(p.numel() for p in self.model.parameters())

        self.provenance = {
            "model": "video-depth-anything",
            "encoder": self.encoder,
            "metric_checkpoint": self.metric,
            "precision": "fp32" if self.fp32 else "fp16",
            "device": str(self.device),
            "checkpoint_path": str(path.resolve()),
            "checkpoint_name": path.name,
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": file_sha,
            "weights_sha256_before_load": sha_before,
            "loaded_weights_sha256": sha_after,
            "n_parameters": int(n_params),
            "strict_load": self.strict,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    # ------------------------------------------------------------- inference

    def is_available(self):
        """True only if weights were verifiably loaded."""
        return self._available and bool(self.provenance)

    def reset_temporal_state(self):
        """Clear the streaming temporal buffer, if the model exposes one.

        Returns True if a reset hook was found. Callers should treat False as
        "construct a fresh processor per sequence instead".
        """
        for name in ("reset", "reset_states", "clear_cache", "reset_stream"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                fn()
                self._frames_seen = 0
                return True
        for attr in ("frame_buffer", "cache", "_cache", "history"):
            buf = getattr(self.model, attr, None)
            if isinstance(buf, list):
                buf.clear()
                self._frames_seen = 0
                return True
        return False

    @property
    def frames_seen(self):
        """Frames pushed through the temporal buffer since construction/reset."""
        return self._frames_seen

    def infer(self, rgb_image, input_size=518):
        """One RGB frame in, one depth map out. STATEFUL: order matters."""
        if not self.is_available():
            raise CheckpointLoadError(
                "infer() called on a processor without verified weights.")
        try:
            depth = self.model.infer_video_depth_one(
                rgb_image, input_size=input_size,
                device=self.device, fp32=self.fp32)
            self._frames_seen += 1
            return depth
        except Exception as e:
            print(f"[VideoDepthAnything] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def infer_batch(self, rgb_images, input_size=518):
        """Sequential inference. Retains temporal state across the list."""
        if isinstance(rgb_images, np.ndarray) and rgb_images.ndim == 4:
            images = [rgb_images[i] for i in range(rgb_images.shape[0])]
        elif isinstance(rgb_images, np.ndarray):
            images = [rgb_images]
        else:
            images = list(rgb_images)
        return [self.infer(im, input_size) for im in images]

    # ------------------------------------------------------------ utilities

    def write_provenance(self, path):
        """Write the provenance record as JSON."""
        with open(path, "w") as fh:
            json.dump(self.provenance, fh, indent=2, sort_keys=True)
        return path

    def process_video(self, input_video, output_dir="./outputs", input_size=518,
                      max_res=1280, target_fps=-1, max_len=-1, grayscale=False,
                      save_npz=False):
        """Convenience path for whole video files (unchanged in spirit)."""
        from utils.dc_utils import save_video

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_video}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        width, height = ow, oh
        if max_res > 0 and max(oh, ow) > max_res:
            scale = max_res / max(oh, ow)
            height, width = round(oh * scale), round(ow * scale)

        fps = original_fps if target_fps < 0 else target_fps
        stride = max(round(original_fps / fps), 1)

        depths, frame_count = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_len > 0 and frame_count >= max_len):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (width, height) != (ow, oh):
                    frame = cv2.resize(frame, (width, height))
                d = self.infer(frame, input_size=input_size)
                if d is not None:
                    depths.append(d)
            frame_count += 1
        cap.release()

        if not depths:
            print("[VideoDepthAnything] No frames processed")
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_video).stem
        depths = np.stack(depths, axis=0)
        save_video(depths, str(out / f"{stem}_vis.mp4"), fps=fps,
                   is_depths=True, grayscale=grayscale)
        if save_npz:
            np.savez(str(out / f"{stem}_depth.npz"), depth=depths)
        self.write_provenance(out / f"{stem}_provenance.json")


def create_depth_processor(encoder="vits", device=None, metric=True, fp32=False,
                           checkpoint_dir="./checkpoints"):
    return VideoDepthAnythingProcessor(encoder=encoder, device=device,
                                       metric=metric, fp32=fp32,
                                       checkpoint_dir=checkpoint_dir)