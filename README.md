# tubule_segmentation

Instance segmentation of kidney tubules in histopathology images using distance transform prediction with OpenMidnight + SAM2.

## Overview

This pipeline predicts instance segmentation masks for kidney tubules by:

1. **Distance Transform Prediction**: A lightweight model combines OpenMidnight (DINOv2-G) and SAM2 features to predict a distance transform map, where pixel intensity indicates distance from tubule boundaries
2. **Instance Separation**: Otsu thresholding + erosion separates touching instances into seeds
3. **SAM2 Refinement**: Each seed is refined using SAM2 with box + mask prompts for precise boundaries

## Architecture
```
Input Image (1024x1024)
│
├──► OpenMidnight ──► 1536-dim features (73x73)
│                            │
│                            ▼
│                     AlignmentBridge ──► 256-dim fused (64x64)
│                            ▲
├──► SAM2 Encoder ──► 256-dim features (64x64)
│
│                     DistanceHead
│                            │
│                            ▼
│                   Distance Map (256x256)
│                            │
│                            ▼
│                   Otsu + Erode + Dilate
│                            │
│                            ▼
│                    Rough Instance Masks
│                            │
└──► SAM2 Predictor ◄────────┘
│
▼
Refined Instance Masks
│
▼
GeoJSON Output
```

## Files

| File | Description |
|------|-------------|
| `data_preprocessing.py` | Resizes images and combines proximal/distal tubule masks into a unified dataset |
| `train.py` | Trains the distance transform prediction model using frozen OpenMidnight + SAM2 backbones |
| `patch_inference.py` | Runs inference on a single image patch and outputs GeoJSON annotations |
| `wsi_inference.py` | Runs batched tile-based inference on whole slide images (WSI) |
| `merge_segmentation.py` | Post-processing step to resolve artifact overlaps from the WSI tiling process |
| `environment.yaml` | Conda environment specification |

## Installation
```
conda env create -f environment.yaml
conda activate tubule_segmentation

download sam2.1_hiera_base_plus.pt from Meta's SAM2 repository and put it in the checkpoints folder
```

## Data preparation
### Input Structure
```
  data/
├── he-pt-data/
│   ├── image1.png
│   ├── image1_mask_proximal.png
│   └── ...
├── he-dt-data/
│   ├── image1.png
│   ├── image1_mask_distal.png
│   └── ...
├── pas-pt-data/
├── pas-dt-data/
├── sil-pt-data/
├── sil-dt-data/
├── tri-pt-data/
└── tri-dt-data/
```
```
Stain types: he, pas, sil, tri
pt = proximal tubule, dt = distal tubule
Masks named: {image_name}_mask_proximal.png or {image_name}_mask_distal.png
```
### Run Preprocessing
python data_preprocessing.py ./data ./data_combined

### Output Structure
```
data_combined/
├── images/
│   ├── he_image1.png
│   ├── pas_image1.png
│   └── ...
└── masks/
    ├── he_image1.png
    ├── pas_image1.png
    └── ...
```

## Training
python train.py

### Configuration
```
Edit constants at top of train.py:

Parameter	Default	Description
IMG_SIZE	1024	Input image size
OUTPUT_SIZE	256	Distance map output size
BATCH_SIZE	8	Training batch size
EPOCHS	30	Number of epochs
LR	1e-4	Learning rate
MAX_DISTANCE	50	Max distance for normalization
VISUALIZE_EVERY	10	Save visualization every N batches
```

### Outputs
```
checkpoints/best_distance_model.pt - Best model weights
distance_outputs/ - Training visualizations
```

### Training Details
```
Frozen backbones: OpenMidnight and SAM2 encoder weights are frozen
Trainable parameters: ~2M (AlignmentBridge + DistanceHead only)
Loss: MSE + peak-weighted MSE (emphasizes tubule centers)
Augmentation: Elastic deformation, color jitter, blur/noise
```

## Inference

> **Note on Hardware:** Both `patch_inference.py` and `wsi_inference.py` have `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` hardcoded at the top of the file. You may need to change or remove this line to map to `0` or another ID depending on your system's GPU setup.

### Patch Inference
Run inference on a single image patch:
```bash
python patch_inference.py <input_image> <output_geojson>
# example: python patch_inference.py patch.png output.geojson
```

#### Patch Configuration
Edit constants at the top of `patch_inference.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ERODE_ITER` | 5 | Erosion iterations for instance separation |
| `MIN_AREA` | 30 | Minimum instance area in pixels |
| `THRESH_FACTOR`| 0.8 | Otsu threshold multiplier |


### WSI Inference
Run batched, tile-based inference on a Whole Slide Image (WSI):
```bash
python wsi_inference.py --input input_wsi.tif --output-dir wsi_tubule_output
```

**Common WSI CLI Arguments:**
* `--tile-size` (default: 512): Output size of the tile
* `--overlap` (default: 256): Tile overlap in pixels
* `--batch-size` (default: 8): Number of tiles to process in parallel
* `--min-tissue` (default: 0.1): Minimum background threshold to process a tissue tile
* `--visualize` (default: 0): Number of preview tile segmentations to save

#### WSI Configuration
Edit constants at the top of `wsi_inference.py`. This script shares `ERODE_ITER`, `MIN_AREA`, and `THRESH_FACTOR` with patch inference, plus the following:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_TUBULE_AREA` | 2000 | Minimum tubule area (in pixels) for WSI area filtering |
| `MAX_TUBULE_AREA` | 400000 | Maximum tubule area (in pixels) for WSI area filtering |


## Post-Processing (WSI Only)
Use `merge_segmentation.py` to resolve touching tile-edges and eliminate artifact overlaps resulting from the WSI tiling process:

```bash
python merge_segmentation.py -i wsi_tubule_output/tubules.geojson -o wsi_tubule_output/tubules_processed.geojson --merge 0.8 --water 0.3
```

**Common Post-Processing Arguments:**
* `--merge` (default: 0.75): Area overlap ratio threshold required to union/merge adjacent features.
* `--water` (default: 0.3): Overlap ratio required to split features using a watershed algorithm to separate complex touching boundaries.

### Output Format
```
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "0",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y2], ...]]
      },
      "properties": {
        "classification": {"name": "Tubule"},
        "score": 0.9523
      }
    }
  ]
}
```