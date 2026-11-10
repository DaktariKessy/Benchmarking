# Benchmarking Monocular Metric Depth Estimation Models

## 1. Download the models

Download the pretrained weights for the following models:

- UniK3D
- UniDepth V2
- MoGe-2
- Metric3D V2
- UniDAC
- Video Depth Anything
- Depth Pro

Place each downloaded model in its corresponding directory inside the `Models/` folder.

## 2. Download the dataset

Download the ROS bag datasets from Zenodo and place them in their corresponding directories:

```text
Data/
├── 1m/
│   └── 1m_0.db3
└── 1.5m/
    └── 1.5m_0.db3
```

Keep the corresponding `metadata.yaml` files in each directory.

**Zenodo dataset:** [Add Zenodo link here]

## 3. Run the evaluation

Use the following command structure:

```bash
python3 Scripts/main_eval.py --model MODEL [MODEL_VARIANT_OPTIONS] --bag DATA_PATH --display --max-frames NUMBER
```

Example:

```bash
python3 Scripts/main_eval.py --model unidac --bag Data/1.5m --display --max-frames 30
```

Replace `MODEL` with the model name and provide the appropriate variant options when required.

