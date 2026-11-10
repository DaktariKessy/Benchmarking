#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-model adapters for the depth benchmark.

Each adapter supplies the ONLY five things that differ between models:

    add_args(ap)            its CLI flags (all prefixed, so they coexist)
    build(args, device)     construct the processor
    tag(args)               filesystem-safe config tag for CSV/PNG names
    infer(proc, rgb, K, a)  normalise the call shape -> float32 (H, W) or None
    metric_model(args)      True = score raw metres, False = fit scale+offset
"""

import os
import re
import numpy as np 

from theta_kinect_compare import FOV_X

# ---- default clone / checkpoint locations (edit for your machine) ----
DEFAULT_DEPTHPRO_PATH = '/home/kessy/Projects/Benchmarking/Models/ml-depth-pro/'
DEFAULT_DEPTHPRO_CKPT = '/home/kessy/Projects/Benchmarking/Models/ml-depth-pro/checkpoints/depth_pro.pt'

DEFAULT_METRIC3D_PATH = '/home/kessy/Projects/Benchmarking/Models/Metric3D/'
DEFAULT_METRIC3D_VARIANT = 'metric3d_vit_small'
DEFAULT_METRIC3D_CKPT = None

DEFAULT_MOGE_PATH = '/home/kessy/Projects/Benchmarking/Models/MoGe/'
DEFAULT_MOGE_MODEL = 'Ruicheng/moge-2-vitb-normal'

DEFAULT_UNIDEPTH_PATH = '/home/kessy/Projects/Benchmarking/Models/UniDepth/'
DEFAULT_UNIDEPTH_MODEL = 'lpiccinelli/unidepth-v2-vitb14'

DEFAULT_UNIDAC_PATH = '/home/kessy/Projects/Benchmarking/Models/UniDAC/'
DEFAULT_UNIDAC_CKPT = '/home/kessy/Projects/Benchmarking/Models/UniDAC/checkpoints/unidac.pt'
DEFAULT_UNIDAC_ERP_CFG = ('/home/kessy/Projects/Benchmarking/Models/UniDAC/'
                          'configs/test/dac_dinov3l+dpt_indoor_test_m3d.json')

DEFAULT_VDA_PATH = '/home/kessy/Projects/Benchmarking/Models/Video-Depth-Anything/'
DEFAULT_VDA_CKPT_DIR = '/home/kessy/Projects/Benchmarking/Models/Video-Depth-Anything/checkpoints'

DEFAULT_UNIK3D_PATH = '/home/kessy/Projects/Benchmarking/Models/UniK3D/'
DEFAULT_UNIK3D_MODEL = 'lpiccinelli/unik3d-vitl'

def _slug(s):
    """Filesystem-safe fragment for output filenames."""
    return re.sub(r'[^A-Za-z0-9]+', '-', str(s)).strip('-')


def _stem(path):
    return _slug(os.path.splitext(os.path.basename(str(path)))[0])


class Adapter:
    """Base class. Subclasses override what they need."""
    name = "model"
    display_name = "Model"
    panel_label = "Model"
    regime = "pinhole"

    @staticmethod
    def add_args(ap):
        pass

    def build(self, args, device):
        raise NotImplementedError

    def tag(self, args):
        return "default"

    def infer(self, proc, rgb, K, args):
        return proc.infer(rgb, K=K)

    def metric_model(self, args):
        return True

    def describe(self, args):
        return ""


# ==========================================================================
class DepthProAdapter(Adapter):
    name = "depthpro"
    display_name = "Depth Pro"
    panel_label = "Depth Pro"

    @staticmethod
    def add_args(ap):
        ap.add_argument("--depthpro-path", default=DEFAULT_DEPTHPRO_PATH)
        ap.add_argument("--depthpro-ckpt", default=DEFAULT_DEPTHPRO_CKPT)
        ap.add_argument("--depthpro-fp32", action="store_true",
                        help="run Depth Pro in FP32 (default FP16)")

    def build(self, args, device):
        from Depthpro_processor import (DepthProProcessor,
                                        load_depthpro_from_path,
                                        is_depthpro_available)
        if not is_depthpro_available():
            load_depthpro_from_path(args.depthpro_path)
        precision = 'fp32' if args.depthpro_fp32 else 'fp16'
        return DepthProProcessor(checkpoint=args.depthpro_ckpt,
                                 device=device, precision=precision)

    def tag(self, args):
        precision = 'fp32' if args.depthpro_fp32 else 'fp16'
        return f"{_stem(args.depthpro_ckpt)}_{precision}"

    def describe(self, args):
        precision = 'fp32' if args.depthpro_fp32 else 'fp16'
        return f"ckpt={args.depthpro_ckpt}  precision={precision}"


# ==========================================================================
class Metric3DAdapter(Adapter):
    name = "metric3d"
    display_name = "Metric3D"
    panel_label = "Metric3D"

    @staticmethod
    def add_args(ap):
        ap.add_argument("--metric3d-path", default=DEFAULT_METRIC3D_PATH)
        ap.add_argument("--metric3d-variant", default=DEFAULT_METRIC3D_VARIANT,
                        help="metric3d_vit_small | metric3d_vit_large | "
                             "metric3d_vit_giant2")
        ap.add_argument("--metric3d-ckpt", default=DEFAULT_METRIC3D_CKPT,
                        help="optional local .pth; omitted = torch.hub weights")

    def build(self, args, device):
        from metric3D_processor import (Metric3DProcessor,
                                        load_metric3d_from_path,
                                        is_metric3d_available)
        if not is_metric3d_available():
            load_metric3d_from_path(args.metric3d_path)
        return Metric3DProcessor(variant=args.metric3d_variant,
                                 checkpoint=args.metric3d_ckpt,
                                 device=device)

    def tag(self, args):
        return args.metric3d_variant.replace("metric3d_vit_", "")

    def describe(self, args):
        return (f"variant={args.metric3d_variant}  "
                f"ckpt={args.metric3d_ckpt or 'torch.hub'}")


# ==========================================================================
class MoGeAdapter(Adapter):
    name = "moge"
    display_name = "MoGe"
    panel_label = "MoGe-2"

    @staticmethod
    def add_args(ap):
        ap.add_argument("--moge-path", default=DEFAULT_MOGE_PATH)
        ap.add_argument("--moge-model", default=DEFAULT_MOGE_MODEL,
                        help="HF id or local path, e.g. Ruicheng/moge-2-vitl "
                             "or Ruicheng/moge-2-vitl-normal")
        ap.add_argument("--moge-resolution", type=int, default=9,
                        help="resolution_level 0-9. Ignored when "
                             "--moge-num-tokens is set.")
        ap.add_argument("--moge-num-tokens", type=int, default=None,
                        help="explicit token count (~1200-2500); overrides "
                             "resolution_level")
        ap.add_argument("--moge-fp32", action="store_true",
                        help="run MoGe in FP32 (default FP16)")

    def build(self, args, device):
        from MoGeDepthProcessor import (MoGeProcessor, load_MoGe_from_path,
                                        is_moge_available)
        if not is_moge_available():
            load_MoGe_from_path(args.moge_path)
        # MoGe takes no per-frame K: fov_x at BUILD time is its only metric
        # anchor to the virtual camera. Passing FOV_X is what stops it from
        # estimating its own FOV and drifting the scale.
        kw = dict(model_id=args.moge_model, device=device, fov_x=FOV_X,
                  resolution_level=args.moge_resolution,
                  use_fp16=not args.moge_fp32,
                  verbose_intrinsics=True)
        if args.moge_num_tokens is not None:
            import inspect
            if "num_tokens" in inspect.signature(
                    MoGeProcessor.__init__).parameters:
                kw["num_tokens"] = args.moge_num_tokens
            else:
                print("[MoGe] WARNING: processor has no 'num_tokens' arg; "
                      "--moge-num-tokens ignored.")
        return MoGeProcessor(**kw)

    def infer(self, proc, rgb, K, args):
        # MoGe takes NO K and returns (depth, vis). Keep the depth.
        out = proc.infer(rgb)
        return out[0] if isinstance(out, tuple) else out

    def tag(self, args):
        model_short = _slug(args.moge_model.split('/')[-1])
        res = (f"nt{args.moge_num_tokens}" if args.moge_num_tokens is not None
               else f"rl{args.moge_resolution}")
        prec = 'fp32' if args.moge_fp32 else 'fp16'
        return f"{model_short}_{res}_{prec}"

    def describe(self, args):
        res = (f"num_tokens={args.moge_num_tokens}"
               if args.moge_num_tokens is not None
               else f"resolution_level={args.moge_resolution}")
        return (f"model={args.moge_model}  {res}  "
                f"precision={'fp32' if args.moge_fp32 else 'fp16'}  "
                f"fov_x={FOV_X}")


# ==========================================================================
class UniDepthAdapter(Adapter):
    name = "unidepth"
    display_name = "UniDepthV2"
    panel_label = "UniDepthV2"

    @staticmethod
    def add_args(ap):
        ap.add_argument("--unidepth-path", default=DEFAULT_UNIDEPTH_PATH)
        ap.add_argument("--unidepth-model", default=DEFAULT_UNIDEPTH_MODEL,
                        help="HF id: lpiccinelli/unidepth-v2-vit{s,b,l}14")
        ap.add_argument("--unidepth-resolution", type=int, default= 1,
                        help="V2-only resolution_level; default leaves the "
                             "library setting")

    def build(self, args, device):
        from unidepth_processor import (UniDepthV2Processor,
                                        load_unidepth_from_path,
                                        is_unidepth_available)
        if not is_unidepth_available():
            load_unidepth_from_path(args.unidepth_path)
        return UniDepthV2Processor(model_name=args.unidepth_model,
                                   device=device,
                                   resolution_level=args.unidepth_resolution)

    def tag(self, args):
        model_short = _slug(args.unidepth_model.split('/')[-1])
        res = ("" if args.unidepth_resolution is None
               else f"_rl{args.unidepth_resolution}")
        return f"{model_short}{res}"

    def describe(self, args):
        return (f"model={args.unidepth_model}  "
                f"resolution_level={args.unidepth_resolution}")


# ==========================================================================
class UniDACAdapter(Adapter):
    name = "unidac"
    display_name = "UniDAC"
    panel_label = "UniDAC"

    @staticmethod
    def add_args(ap):
        ap.add_argument("--unidac-path", default=DEFAULT_UNIDAC_PATH)
        ap.add_argument("--unidac-ckpt", default=DEFAULT_UNIDAC_CKPT)
        ap.add_argument("--unidac-config", default=None,
                        help="optional test-config json; default is the "
                             "indoor NYU pinhole config inside the clone")

    def build(self, args, device):
        from UniDAC_processor import (UniDACProcessor, load_unidac_from_path,
                                      is_unidac_available)
        if not is_unidac_available():
            load_unidac_from_path(args.unidac_path)
        kw = dict(checkpoint=args.unidac_ckpt, device=device)
        if args.unidac_config is not None:
            kw["config_file"] = args.unidac_config
        return UniDACProcessor(**kw)

    def tag(self, args):
        return _stem(args.unidac_ckpt)

    def describe(self, args):
        return (f"ckpt={args.unidac_ckpt}  "
                f"config={args.unidac_config or 'indoor NYU pinhole'}  "
                f"(REQUIRES intrinsics; returns None without K)")


# ==========================================================================
class VDAAdapter(Adapter):
    """Video Depth Anything -- the one STATEFUL model here.

    Its infer() remembers recent frames, so depth is temporally consistent.
    Frames must be fed in bag order (stream_pairs already does that) and
    must not be shuffled or skipped. The first few frames are warming up the
    temporal buffer and are less representative; keep that in mind when
    reading early-frame numbers.
    """
    name = "vda"
    display_name = "Video-DA"
    panel_label = "Video-DA"

    @staticmethod
    def add_args(ap):
        ap.add_argument("--vda-path", default=DEFAULT_VDA_PATH)
        ap.add_argument("--vda-ckpt-dir", default=DEFAULT_VDA_CKPT_DIR)
        ap.add_argument("--vda-encoder", default="vitl",
                        choices=["vits", "vitb", "vitl"])
        ap.add_argument("--vda-fp32", action="store_true",
                        help="FP32 (default FP16)")
        ap.add_argument("--vda-relative", action="store_true",
                        help="score as a RELATIVE model (scale+offset fit); "
                             "use only if you loaded relative weights")
        ap.add_argument("--vda-input-size", type=int, default=518)

    def build(self, args, device):
        # Prefer the FIXED processor: it derives the checkpoint filename from
        # `encoder`, verifies the load, and FAILS LOUDLY. The older
        # videoDepthAnythingProcessor.py hardcodes the vits checkpoint,
        # swallows the load error and still reports itself available -- which
        # benchmarks a randomly initialised network with any other encoder.
        try:
            from Videodepthanythingprocessor import (
                VideoDepthAnythingProcessor,
                load_video_depth_anything_from_path)
        except ImportError:
            from videoDepthAnythingProcessor import (
                VideoDepthAnythingProcessor,
                load_video_depth_anything_from_path)
            print("[VDA] !! WARNING: using the OLD processor "
                  "(videoDepthAnythingProcessor.py). It hardcodes the vits "
                  "checkpoint and does not fail on a bad load. Verify the "
                  "weights actually loaded before trusting these numbers. !!")
        try:
            load_video_depth_anything_from_path(args.vda_path)
        except Exception as e:
            print(f"[VDA] load_from_path note: {e}")
        return VideoDepthAnythingProcessor(
            encoder=args.vda_encoder, device=device,
            metric=(not args.vda_relative), fp32=args.vda_fp32,
            checkpoint_dir=args.vda_ckpt_dir)

    def infer(self, proc, rgb, K, args):
        # No K; takes input_size instead. Stateful: one frame in, one out.
        return proc.infer(rgb, input_size=args.vda_input_size)

    def metric_model(self, args):
        return not args.vda_relative

    def tag(self, args):
        prec = 'fp32' if args.vda_fp32 else 'fp16'
        mode = 'rel' if args.vda_relative else 'metric'
        return f"{args.vda_encoder}_{prec}_{mode}_is{args.vda_input_size}"

    def describe(self, args):
        return (f"encoder={args.vda_encoder}  "
                f"precision={'fp32' if args.vda_fp32 else 'fp16'}  "
                f"input_size={args.vda_input_size}  "
                f"STATEFUL (early frames are temporal warm-up)")



# ==========================================================================
class UniK3DAdapter(Adapter):
    name = "unik3d"
    display_name = "UniK3D"
    panel_label = "UniK3D"

    @staticmethod
    def add_args(ap):
        ap.add_argument("--unik3d-path", default=DEFAULT_UNIK3D_PATH)
        ap.add_argument("--unik3d-model", default=DEFAULT_UNIK3D_MODEL,
                        help="HF id: lpiccinelli/unik3d-vit{s,b,l}")
        # UniDepth defaults this to None; UniK3D does not, on purpose.
        # 518x518 is 0.268 MP, native to level 1 on this checkpoint, and the
        # published CSVs carry an _rl1 suffix -- so 1 is what actually ran.
        # None would still work but drops the suffix from tag(), and the
        # re-run would no longer sit beside the numbers in the paper.
        # CONFIRM against a real filename before starting the sweep.
        ap.add_argument("--unik3d-resolution", type=int, default=1,
                        help="input pixel-count band 0-9; 1 is native for a "
                             "518x518 crop, higher only upsamples")

    def build(self, args, device):
        from unik3d_processor import (UniK3DProcessor,
                                      load_unik3d_from_path,
                                      is_unik3d_available)
        if not is_unik3d_available():
            load_unik3d_from_path(args.unik3d_path)
        return UniK3DProcessor(model_id=args.unik3d_model,
                               device=device,
                               resolution_level=args.unik3d_resolution)

    def infer(self, proc, rgb, K, args):
        """UniK3D is the ONE processor whose infer() is not metric.

        UniK3DProcessor.infer() returns a 0-255 uint8 visualisation, so the
        base class's `proc.infer(rgb, K=K)` would push normalised bytes into
        compare() as if they were metres -- no exception, just a nonsense
        column. infer_metric()["depth"] is the metric one.

        That depth is already Z-DEPTH along the optical axis (the processor
        docstring verifies it against points[2], and against a corner/centre
        ratio of 1.000 where radial would have given ~1.291 at 60 deg), so it
        feeds compare() with no radial->z conversion. The equirect path is
        the opposite -- radial -- which is the other reason not to share code
        between them.

        K matters and fails silently. Without it UniK3D self-predicts the
        camera and still returns a plausible map; the only symptom is a shift
        in the scale column. Check scale against the published run.
        """
        out = proc.infer_metric(rgb, K=K,
                                return_points=False, return_rays=False)
        return None if out is None else out.get("depth")

    def tag(self, args):
        model_short = _slug(args.unik3d_model.split('/')[-1])
        res = ("" if args.unik3d_resolution is None
               else f"_rl{args.unik3d_resolution}")
        return f"{model_short}{res}"

    def describe(self, args):
        return (f"model={args.unik3d_model}  "
                f"resolution_level={args.unik3d_resolution}  "
                f"(z-depth, K-conditioned; infer_metric not infer)")

# ==========================================================================

 
class _EquiAdapter(Adapter):
    """Shared bits of the two panoramic regimes.
 
    regime = "equirect" is the ONLY thing run_eval branches on. It changes
    two things and nothing else: the model is handed the whole panorama
    instead of a render_pseudo crop, and scoring goes through compare_equi
    on the sphere instead of compare in a pinhole window.
    """
    regime = "equirect"
 
    # Both sides are RADIAL distance from the optical centre, so there is no
    # radial->z correction anywhere in this path. compare_equi expects that.
    def metric_model(self, args):
        return True
 
 
class UniK3DEquiAdapter(_EquiAdapter):
    name = "unik3d-equi"
    display_name = "UniK3D (equirect)"
    panel_label = "UniK3D-equi"
 
    @staticmethod
    def add_args(ap):
        # Note: no --unik3d-equi-path/-model. The flags are shared with the
        # pinhole UniK3D adapter on purpose, so one --unik3d-model picks the
        # checkpoint for both regimes and they cannot silently diverge.
        pass
 
    def build(self, args, device):
        from unik3d_processor import (UniK3DProcessor,
                                      load_unik3d_from_path,
                                      is_unik3d_available)
        if not is_unik3d_available():
            load_unik3d_from_path(args.unik3d_path)
        proc = UniK3DProcessor(model_id=args.unik3d_model, device=device,
                               resolution_level=args.unik3d_resolution)
        if not hasattr(proc, "infer_equirect_metric"):
            raise SystemExit(
                "UniK3DProcessor has no infer_equirect_metric(). Paste it "
                "from unik3d_equirect_snippet.py first.")
        return proc
 
    def infer(self, proc, rgb, K, args):
        """rgb is the WHOLE panorama; K is None (a sphere has no pinhole K).
 
        Returns RADIAL distance rebuilt from the 3D points, not
        outputs["depth"]. On a full 360 a z-depth goes negative behind the
        image plane and is meaningless over half the sphere; ||points|| is
        non-negative by construction and is what compare_equi expects.
        """
        out = proc.infer_equirect_metric(rgb, return_points=True)
        if out is None:
            return None
        pts = out.get("points")
        if pts is not None and pts.ndim == 3 and pts.shape[2] == 3:
            return np.linalg.norm(pts, axis=2).astype(np.float32)
        return out.get("depth")      # fallback; watch the scale column
 
    def tag(self, args):
        model_short = _slug(args.unik3d_model.split('/')[-1])
        res = ("" if args.unik3d_resolution is None
               else f"_rl{args.unik3d_resolution}")
        return f"{model_short}{res}"
 
    def describe(self, args):
        return (f"model={args.unik3d_model}  "
                f"resolution_level={args.unik3d_resolution}  "
                f"Spherical camera, radial output, scored on the sphere")
 
 
class UniDACEquiAdapter(_EquiAdapter):
    name = "unidac-equi"
    display_name = "UniDAC (native ERP)"
    panel_label = "UniDAC-equi"
 
    @staticmethod
    def add_args(ap):
        # Only the ERP config is its own flag: the checkpoint and clone path
        # are shared with the pinhole UniDAC adapter, but the CONFIG must
        # differ -- the pinhole one's fwd_sz is (512, 704) and would squash
        # a 2:1 panorama.
        ap.add_argument("--unidac-erp-config", default=DEFAULT_UNIDAC_ERP_CFG,
                        help="ERP config, fwd_sz 2:1 (the m3d one). NOT the "
                             "NYU pinhole config.")
 
    def build(self, args, device):
        from UniDAC_processor import (UniDACProcessor, load_unidac_from_path,
                                      is_unidac_available)
        if not is_unidac_available():
            load_unidac_from_path(args.unidac_path)
        proc = UniDACProcessor(checkpoint=args.unidac_ckpt,
                               config_file=args.unidac_erp_config,
                               device=device)
        if not hasattr(proc, "infer_erp"):
            raise SystemExit(
                "UniDACProcessor has no infer_erp(). Paste it from "
                "unidac_erp_snippet.py first.")
        fw = getattr(proc, "fwd_sz", None)
        if fw is not None:
            ratio = fw[1] / max(fw[0], 1)
            print(f"[UniDAC-equi] fwd_sz={list(fw)}  aspect {ratio:.2f}")
            if abs(ratio - 2.0) > 0.2:
                print("[UniDAC-equi] *** fwd_sz is not 2:1 -- this looks "
                      "like a PINHOLE config. A panorama pushed through it "
                      "is squashed. Use the ERP (m3d) config. ***")
        return proc
 
    def infer(self, proc, rgb, K, args):
        """Native ERP path: no cam_to_erp_patch_fast in, no
        erp_patch_to_cam_fast + euclid->z out. Both round trips are
        resampling steps; removing them is the point of this regime."""
        return proc.infer_erp(rgb)
 
    def tag(self, args):
        return _stem(args.unidac_erp_config)
 
    def describe(self, args):
        return (f"ckpt={args.unidac_ckpt}  config={args.unidac_erp_config}  "
                f"native ERP, radial output, no crop round trip")
 

REGISTRY = {a.name: a for a in (
    DepthProAdapter(), Metric3DAdapter(), MoGeAdapter(),
    UniDepthAdapter(), UniDACAdapter(), VDAAdapter(), UniK3DAdapter(),
    UniK3DEquiAdapter(), UniDACEquiAdapter(),
)}
