#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video Depth Anything depth estimation processor with checkpoint loading and video processing."""

import sys
import os
import time
from pathlib import Path

import torch
import cv2
import numpy as np


# Global state for Video Depth Anything availability
VIDEO_DEPTH_ANYTHING_AVAILABLE = False
VIDEO_DEPTH_ANYTHING_PATH = None


def load_video_depth_anything_from_path(local_path):
    """Load Video Depth Anything from a local cloned directory."""
    global VIDEO_DEPTH_ANYTHING_AVAILABLE, VIDEO_DEPTH_ANYTHING_PATH
    
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[VideoDepthAnything] Local path not found: {local_path}")
        return False

    sys.path.insert(0, str(local_path))
    try:
        from video_depth_anything.video_depth_stream import VideoDepthAnything
        VIDEO_DEPTH_ANYTHING_AVAILABLE = True
        VIDEO_DEPTH_ANYTHING_PATH = local_path
        print(f"[VideoDepthAnything] Loaded from local clone: {local_path}")
        return True
    except ImportError as e:
        print(f"[VideoDepthAnything] Failed to import from {local_path}: {e}")
        return False


# Try installed package first
try:
    from video_depth_anything.video_depth_stream import VideoDepthAnything
    VIDEO_DEPTH_ANYTHING_AVAILABLE = True
    print("[VideoDepthAnything] Loaded from installed package")
except ImportError:
    pass

if not VIDEO_DEPTH_ANYTHING_AVAILABLE:
    print("[VideoDepthAnything] Not available yet (will check --video-depth-path)")


class VideoDepthAnythingProcessor:
    """Wrapper for Video Depth Anything depth estimation with checkpoint loading."""
    
    def __init__(self, encoder="vits", device=None, metric=False, fp32=False, 
                 checkpoint_dir="/home/kessy/Trials/Video-Depth-Anything/checkpoints"):
        """
        Initialize Video Depth Anything processor with optional checkpoint loading.
        
        Args:
            encoder: 'vits' (fast), 'vitb' (balanced), or 'vitl' (quality)
            device: 'cuda', 'cpu', or None (auto-detect)
            metric: Use metric depth models (real-world measurements)
            fp32: Use FP32 precision (slower but more accurate)
            checkpoint_dir: Directory containing model checkpoints
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.metric = metric
        self.fp32 = fp32
        self.checkpoint_dir = Path(checkpoint_dir)
        self._available = False
        self.model = None
        
        # Model configuration (from run.py)
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        if encoder not in self.model_configs:
            print(f"[VideoDepthAnything] ✗ Unknown encoder: {encoder}")
            return
        
        try:
            from video_depth_anything.video_depth_stream import VideoDepthAnything
            
            model_type = "metric" if metric else "relative"
            print(f"[VideoDepthAnything] Loading {encoder.upper()} ({model_type} depth)...")
            
            config = self.model_configs[encoder]
            self.model = VideoDepthAnything(**config)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Try to load checkpoint if available
            self._try_load_checkpoint()
            
            # Disable gradients
            for param in self.model.parameters():
                param.requires_grad = False
            
            self._available = True
            # print(f"[VideoDepthAnything] ✓ Model loaded successfully")
            # print(f"[VideoDepthAnything]   Device: {self.device}")
            # print(f"[VideoDepthAnything]   Precision: {'FP32' if fp32 else 'FP16'}")
            
        except Exception as e:
            print(f"[VideoDepthAnything] ✗ Failed to load: {e}")
            import traceback
            traceback.print_exc()
            self._available = False
    
    # def _try_load_checkpoint(self):
    #     """Try to load model checkpoint from disk (from run.py)."""
    #     if not self.checkpoint_dir.exists():
    #         print(f"[VideoDepthAnything] Checkpoint directory not found: {self.checkpoint_dir}")
    #         return
        
    #     # Determine checkpoint name based on metric setting (from run.py)
    #     checkpoint_prefix = 'metric_video_depth_anything' if self.metric else 'video_depth_anything'
    #     checkpoint_name = f"{checkpoint_prefix}_{self.encoder}.pth"
    #     checkpoint_path = self.checkpoint_dir / checkpoint_name
        
    #     if checkpoint_path.exists():
    #         try:
    #             print(f"[VideoDepthAnything] Loading checkpoint: {checkpoint_path}")
    #             state_dict = torch.load(str(checkpoint_path), map_location='cpu')
    #             self.model.load_state_dict(state_dict, strict=True)
    #             print(f"[VideoDepthAnything] ✓ Checkpoint loaded: {checkpoint_name}")
    #         except Exception as e:
    #             print(f"[VideoDepthAnything] ✗ Failed to load checkpoint: {e}")
    #     else:
    #         print(f"[VideoDepthAnything] Checkpoint not found: {checkpoint_path}")
    #         print(f"[VideoDepthAnything] Model will run without pre-trained weights")
    
    def _try_load_checkpoint(self):
        """Load model checkpoint."""
        checkpoint_path = Path("/home/kessy/Trials/Video-Depth-Anything/checkpoints/metric_video_depth_anything_vits.pth")

        if not checkpoint_path.exists():
            print(f"[VideoDepthAnything] Checkpoint not found: {checkpoint_path}")
            return

        try:
            # print(f"[VideoDepthAnything] Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            # print("[VideoDepthAnything] ✓ Checkpoint loaded")
        except Exception as e:
            print(f"[VideoDepthAnything] ✗ Failed to load checkpoint: {e}")

    def is_available(self):
        """Check if model is available and loaded."""
        return self._available
    
    def infer(self, rgb_image, input_size=518):
        """
        Estimate depth from RGB image.
        
        Args:
            rgb_image: numpy array (H, W, 3) in RGB format, values 0-255
            input_size: target input size for model (default: 518)
        
        Returns:
            depth_map: numpy array (H, W) with depth values, or None if failed
        """
        if not self._available or self.model is None:
            return None
        
        try:
            # Use the actual infer_video_depth_one method from the model
            depth = self.model.infer_video_depth_one(
                rgb_image,
                input_size=input_size,
                device=self.device,
                fp32=self.fp32
            )
            return depth
        
        except Exception as e:
            print(f"[VideoDepthAnything] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def infer_batch(self, rgb_images, input_size=518):
        """
        Estimate depth from multiple RGB images (batch processing).
        
        Args:
            rgb_images: list of numpy arrays or single array (B, H, W, 3)
            input_size: target input size for model
        
        Returns:
            depth_maps: list of numpy arrays or None if failed
        """
        if not self._available or self.model is None:
            return None
        
        try:
            # Handle both list and array inputs
            if isinstance(rgb_images, np.ndarray):
                if rgb_images.ndim == 4:
                    images_list = [rgb_images[i] for i in range(rgb_images.shape[0])]
                else:
                    images_list = [rgb_images]
            else:
                images_list = rgb_images
            
            depth_maps = []
            for rgb_image in images_list:
                depth = self.infer(rgb_image, input_size)
                depth_maps.append(depth)
            
            return depth_maps
        
        except Exception as e:
            print(f"[VideoDepthAnything] Batch inference error: {e}")
            return None
    
    def process_video(self, input_video, output_dir="./outputs", input_size=518, 
                     max_res=1280, target_fps=-1, max_len=-1, grayscale=False,
                     save_npz=False, save_exr=False):
        """
        Process video with depth estimation (based on run.py).
        
        Args:
            input_video: path to input video file
            output_dir: directory to save output
            input_size: model input size (default: 518)
            max_res: maximum resolution (default: 1280)
            target_fps: target fps (-1 means original)
            max_len: maximum frame count (-1 means no limit)
            grayscale: save grayscale depth (no colormap)
            save_npz: save depth as NPZ format
            save_exr: save depth as EXR format
        """
        if not self._available or self.model is None:
            print("[VideoDepthAnything] Model not available")
            return
        
        try:
            from utils.util import save_video
        except ImportError:
            print("[VideoDepthAnything] Could not import save_video utility")
            print("[VideoDepthAnything] Make sure Video-Depth-Anything is in path")
            return
        
        # Open video (from run.py)
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"[VideoDepthAnything] Could not open video: {input_video}")
            return
        
        # Get video properties (from run.py)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[VideoDepthAnything] Processing video: {input_video}")
        print(f"  Resolution: {original_width}x{original_height}")
        print(f"  FPS: {original_fps}")
        print(f"  Total frames: {total_frames}")
        
        # Handle resolution scaling (from run.py)
        width, height = original_width, original_height
        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)
            print(f"  Scaled to: {width}x{height}")
        
        # Handle FPS (from run.py)
        fps = original_fps if target_fps < 0 else target_fps
        stride = max(round(original_fps / fps), 1)
        print(f"  Output FPS: {fps}")
        print(f"  Frame stride: {stride}")
        
        # Process frames (from run.py)
        depths = []
        frame_count = 0
        start = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_len > 0 and frame_count >= max_len):
                break
            
            # Process every stride-th frame (from run.py)
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed (from run.py)
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))
                
                # Inference depth (from run.py)
                depth = self.infer(frame, input_size=input_size)
                
                if depth is not None:
                    depths.append(depth)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  [{frame_count}/{total_frames}] frames processed")
        
        end = time.time()
        cap.release()
        
        print(f"[VideoDepthAnything] Time: {end - start:.2f}s")
        
        # Save results (from run.py)
        if len(depths) > 0:
            video_name = Path(input_video).stem
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Stack depth maps (from run.py)
            depths = np.stack(depths, axis=0)
            print(f"[VideoDepthAnything] Depth stack shape: {depths.shape}")
            
            # Save visualization video (from run.py)
            depth_vis_path = str(output_path / f"{video_name}_vis.mp4")
            print(f"[VideoDepthAnything] Saving visualization: {depth_vis_path}")
            try:
                save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=grayscale)
            except Exception as e:
                print(f"[VideoDepthAnything] Could not save video: {e}")
            
            # Save NPZ if requested
            if save_npz:
                npz_path = str(output_path / f"{video_name}_depth.npz")
                print(f"[VideoDepthAnything] Saving NPZ: {npz_path}")
                np.savez(npz_path, depth=depths)
            
            # Save EXR if requested
            if save_exr:
                try:
                    import OpenEXR
                    for i, depth_map in enumerate(depths):
                        exr_path = str(output_path / f"{video_name}_depth_{i:06d}.exr")
                        # Note: Proper EXR writing requires additional setup
                        print(f"[VideoDepthAnything] EXR support requires additional setup")
                except ImportError:
                    print("[VideoDepthAnything] OpenEXR not installed, skipping EXR save")
        else:
            print("[VideoDepthAnything] No frames processed")
        
        print("[VideoDepthAnything] ✓ Video processing complete")


# Convenience function for YOLO pipeline integration
def create_depth_processor(encoder="vits", device=None, metric=False, fp32=False, 
                          checkpoint_dir="/home/kessy/Trials/Video-Depth-Anything/checkpoints"):
    """
    Factory function to create VideoDepthAnythingProcessor.
    
    Args:
        encoder: 'vits', 'vitb', or 'vitl'
        device: 'cuda', 'cpu', or None
        metric: Use metric depth models
        fp32: Use FP32 precision
        checkpoint_dir: Directory containing model checkpoints
    
    Returns:
        VideoDepthAnythingProcessor instance
    """
    return VideoDepthAnythingProcessor(
        encoder=encoder, 
        device=device, 
        metric=metric, 
        fp32=fp32,
        checkpoint_dir=checkpoint_dir
    )