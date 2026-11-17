# Benchmarking Monocular Metric Depth Estimation Models for Robotics

This repository provides the code and evaluation framework for benchmarking monocular metric depth estimation (MMDE) models for robotic applications.

## 1. Prerequisites

### System requirements

| Requirement | Notes |
|---|---|
| Ubuntu 22.04 | ROS 2 Humble targets it. `bench_core.py` hardcodes `/opt/ros/humble/lib/python3.10/dist-packages`. |
| Python 3.10 | Must match the ROS 2 Humble interpreter. |
| ROS 2 Humble | Required to read ROS 2 `.db3` bags, even without `--publish`. |
| NVIDIA GPU + CUDA | CPU fallback is available with a warning, but full evaluation may be impractical. |
| X11 display | Required only for `--display`. Use `--save-display` on a headless machine. |

### Python packages

Install the main dependencies:

```bash
pip install torch torchvision numpy opencv-python
pip install huggingface_hub==0.25.2
```

For additional functionality:

- `--publish` requires `rclpy`, `cv_bridge`, and `scipy`.
- `rclpy` and `cv_bridge` are provided by the ROS 2 installation, not pip.
- Install `xformers` matching your PyTorch build if available. It can substantially improve inference speed.

**Hugging Face compatibility note:** Version `1.28.0` of `huggingface_hub` causes `UniDepthV2.from_pretrained` to fail with `TypeError: missing 1 required positional argument: 'config'`. Use an older compatible release or the fallback loader in `unidepth_load.py`.

## 2. Download the models

Clone each model's repository and fetch its pretrained weights, then place the
code and checkpoints in the corresponding directory inside `Models/`.

| Model | Repository | Checkpoint source |
|---|---|---|
| UniDepth V2 | https://github.com/lpiccinelli-eth/UniDepth | `lpiccinelli/unidepth-v2-vit{s,b,l}14` (Hugging Face) |
| UniK3D | https://github.com/lpiccinelli-eth/UniK3D | `lpiccinelli/unik3d-vit{s,b,l}` (Hugging Face) |
| MoGe-2 | https://github.com/microsoft/MoGe | `Ruicheng/moge-2-vit{s,b}-normal`, `Ruicheng/moge-2-vitl` |
| Metric3D V2 | https://github.com/YvanYin/Metric3D | Release checkpoints; the variant must match the configuration |
| Video Depth Anything | https://github.com/DepthAnything/Video-Depth-Anything | Official release checkpoints |
| Depth Pro | https://github.com/apple/ml-depth-pro | Apple release checkpoint |
| UniDAC | https://girish1511.github.io/UniDAC/ | See project page; set `"pretrained": null` in `configs/test/*.json` |

Refer to the original repositories for licensing information.

## 3. Download the dataset

Download the ROS 2 bag datasets from Zenodo:

[huggingface.co/datasets/kibteff/mmde-benchmark](https://huggingface.co/datasets/kibteff/mmde-benchmark)

The dataset contains two recording directories:

Data/
├── 0.51m/
│   ├── 0.51m_0.db3
│   └── metadata.yaml
├── 1m/
│   ├── 1m_0.db3
│   └── metadata.yaml
└── 1.5m/
    ├── 1.5m_0.db3
    └── metadata.yaml

Download each bag and place it in its corresponding directory. Keep the `metadata.yaml` file alongside the bag.
```bash
hf download kibteff/mmde-benchmark --repo-type dataset --local-dir Data/
```

### Required input data

Each bag must contain time-synchronised recordings of:

- Ricoh THETA X equirectangular colour images.
- Azure Kinect depth images (16-bit, millimetres).
- Azure Kinect `CameraInfo`.

**Important:**
- Name each bag directory using its scene distance, e.g. `Data/1.5m/`.
- The analysis scripts extract the distance token from CSV filenames. Bags without this naming convention may be silently skipped during later analysis.
- `stream_pairs()` tolerates a depth frame arriving before the first `CameraInfo` by skipping that frame. However, a bag without any `CameraInfo` messages will produce an empty CSV.

## 4. Project layout

The scripts locate project resources relative to `bench_core.py`, rather than the shell's working directory. Keep the directory structure as follows:

```text
Benchmarking/
├── Data/          # ROS 2 bags and metadata
├── Models/        # Downloaded model checkpoints
├── Output/        # CSV results and generated panels
├── Processors/    # Per-model processing wrappers
├── Scripts/       # Evaluation and analysis scripts
└── Tools/         # Supporting utilities, including equilib
```

`Processors/`, `Tools/`, and `Scripts/` are added to `sys.path` at import time. Required local modules must be located in one of these three directories.

### Required local modules

| Module | Provides |
|---|---|
| `theta_kinect_compare` | `compare`, `stream_pairs`, `render_pseudo`, `OUT_W`, `OUT_H`, `FOV_X`, `DEPTH_DISTORTION` |
| `theta_kinect_compare_equi` | `compare_equi` — spherical scoring |
| `equi_pinhole` | `compare_equi_pinhole`, `project_equi_to_pinhole` — 60° window control |
| `equi_panel` | `make_equi_window_panel` |
| `pixel_stability` | `PixelStability` and per-pixel stability CSV writers |
| `adapters.py` | One adapter per model family |
| `equi_outline` | Optional; draws the red evaluation-window box. Skipped with a warning if missing. |
| `ros_publisher` | Required only for `--publish`. Set `ROS_PUBLISHER_MODULE` in `bench_core.py` if the file has another name. |

## 5. Run the evaluation

The main evaluation script is:

```bash
python3 Scripts/main_eval.py
```

Use the following command structure to select a model, its variant (where applicable), the input bag, and the number of frames:

```bash
python3 Scripts/main_eval.py --model MODEL [VARIANT_OPTIONS] --bag DATA_PATH --display --max-frames NUMBER
```

### Example

Run UniDAC on the 1.5 m dataset using 30 frames:

```bash
python3 Scripts/main_eval.py --model unidac --bag Data/1.5m --display --max-frames 30
```

Replace `MODEL` with the desired model name and provide the appropriate variant options for models that support multiple configurations.

Evaluation results and generated visualizations are saved under `Output/`.

## 6. Output

The `Output/` directory is used for:

- Depth evaluation metrics (CSV files).
- Temporal stability measurements.
- Generated comparison panels and visualizations.

The output files are generated automatically during evaluation.

## 7. Notes

- Ensure that the required model checkpoints and dataset files are in the correct directories before running the evaluation.
- Use Python 3.10 with the ROS 2 Humble environment.
- The `--display` option requires an available graphical display.
- Check the original model repositories for their respective licences and checkpoint requirements.

## Citation

If you use this code or dataset, please cite:

> S. Kessy, M. D'Haene, A. Andre, G. Caron.
> "Benchmarking Monocular Metric Depth Estimation Models for Robotics:
> Potential and Limitations."
> IEEE/SICE International Symposium on System Integration (SII),
> Kobe, Japan, January 2027.

```bibtex
@inproceedings{kessy2027benchmarking,
  title     = {Benchmarking Monocular Metric Depth Estimation Models for
               Robotics: Potential and Limitations},
  author    = {Kessy, S. and D'Haene, M. and Andre, A. and Caron, G.},
  booktitle = {IEEE/SICE International Symposium on System Integration (SII)},
  address   = {Kobe, Japan},
  month     = jan,
  year      = {2027}
}
```
