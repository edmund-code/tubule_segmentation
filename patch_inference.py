import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json
import argparse
import sys
from typing import List, Dict, Tuple
import warnings
from shapely.geometry import Polygon as ShapePolygon
warnings.filterwarnings('ignore')

# ============== Configuration ==============
IMG_SIZE = 1024
DINO_SIZE = 1022
OUTPUT_SIZE = 256
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DISTANCE_MODEL_PATH = "checkpoints/best_distance_model.pt"

ERODE_ITER = 5
MIN_AREA = 30
THRESH_FACTOR = 0.8

# Tubule size filters (in pixels at full resolution)
MIN_TUBULE_AREA = 2000
MAX_TUBULE_AREA = 400000


# ============== Model Definitions ==============

class AlignmentBridge(nn.Module):
    """
    Fuses high-dimensional features from OpenMidnight (1536-dim) and SAM2 (256-dim)
    into a unified representation space for distance transform prediction.
    """
    def __init__(self, midnight_dim=1536, sam_dim=256, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(midnight_dim + sam_dim, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, out_dim, kernel_size=1)
        )
    
    def forward(self, m_feat, s_feat):
        m_up = F.interpolate(m_feat, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.proj(torch.cat([m_up, s_feat], dim=1))


class DistanceHead(nn.Module):
    """
    Decodes the fused features from the AlignmentBridge into a 
    single-channel predicted distance transform map using transpose convolutions.
    """
    def __init__(self, in_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.head(x)


class DistanceModel(nn.Module):
    """
    End-to-end network combining the AlignmentBridge and DistanceHead to 
    predict distance maps from OpenMidnight and SAM2 feature embeddings.
    """
    def __init__(self):
        super().__init__()
        self.bridge = AlignmentBridge()
        self.head = DistanceHead()
    
    def forward(self, m_feat, s_feat):
        return self.head(self.bridge(m_feat, s_feat))


# ============== Core Functions ==============

def load_models():
    """Load all required models."""
    print("Loading models...")
    
    dl_loc = hf_hub_download(repo_id="SophontAI/OpenMidnight", filename="teacher_checkpoint_load.pt")
    midnight = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=False)
    cp = torch.load(dl_loc, map_location="cuda", weights_only=False)
    midnight.pos_embed = nn.Parameter(cp["pos_embed"])
    midnight.load_state_dict(cp)
    midnight = midnight.cuda().eval()
    print("  ✓ Midnight loaded")
    
    sam2_enc = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device="cuda").image_encoder.eval()
    sam2_predictor = SAM2ImagePredictor(build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device="cuda"))
    print("  ✓ SAM2 loaded")
    
    model = DistanceModel().cuda().eval()
    model.load_state_dict(torch.load(DISTANCE_MODEL_PATH, map_location="cuda", weights_only=False)['model'])
    print("  ✓ Distance model loaded")
    
    return midnight, sam2_enc, sam2_predictor, model


@torch.inference_mode()
def extract_features(midnight, sam2_enc, img_tensor):
    """
    Extract feature embeddings from both OpenMidnight and SAM2 encoders.
    """
    img_midnight = F.interpolate(img_tensor, size=(DINO_SIZE, DINO_SIZE), mode='bilinear', align_corners=False)
    tokens = midnight.forward_features(img_midnight)["x_norm_patchtokens"]
    grid = int(tokens.shape[1] ** 0.5)
    m_feat = tokens.permute(0, 2, 1).reshape(1, 1536, grid, grid).contiguous()
    
    output = sam2_enc(img_tensor)
    s_feat = output["backbone_fpn"][-1]
    if s_feat.dim() == 3:
        if s_feat.shape[-1] == 256:
            s_feat = s_feat.permute(0, 2, 1)
        H = int(s_feat.shape[-1] ** 0.5)
        s_feat = s_feat.reshape(1, -1, H, H).contiguous()
    
    return m_feat, s_feat


def distance_to_instances(distance_map):
    """Convert distance map to instance masks."""
    # Convert float map to uint8 for OpenCV processing
    dist_uint8 = (distance_map * 255).astype(np.uint8)
    
    # Calculate global Otsu threshold, then apply a factor for leniency
    thresh_val, _ = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(dist_uint8, int(thresh_val * THRESH_FACTOR), 255, cv2.THRESH_BINARY)
    binary = (binary > 0).astype(np.uint8)
    
    # 1. Erosion: Shrink masks to separate touching tubule boundaries (seeds)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=ERODE_ITER)
    
    # Identify distinct tubule candidates
    num_labels, labels = cv2.connectedComponents(eroded)
    
    masks = []
    # 2. Re-expansion: Dilate the seeds back to their original size 
    # but strictly contained within the original binary mask
    for i in range(1, num_labels):
        seed = (labels == i).astype(np.uint8)
        dilated = cv2.dilate(seed, kernel, iterations=ERODE_ITER)
        final = dilated & binary
        if final.sum() >= MIN_AREA:
            masks.append(final)
    
    # 3. Collision Resolution: Handle pixels claimed by multiple dilated instances
    if len(masks) > 1:
        combined = sum((m > 0).astype(np.int32) for m in masks)
        if (combined > 1).any():
            # Calculate the geometric centers (centroids) of each candidate instance
            centroids = [(np.where(m > 0)[1].mean(), np.where(m > 0)[0].mean()) for m in masks]
            new_masks = [m.copy() for m in masks]
            
            # Resolve overlapping pixels by assigning them to the nearest centroid
            for y, x in zip(*np.where(combined > 1)):
                best = np.argmin([(x - cx)**2 + (y - cy)**2 for cx, cy in centroids])
                for i in range(len(new_masks)):
                    new_masks[i][y, x] = 1 if i == best else 0
            masks = new_masks
    
    return masks


def refine_with_sam2(image, rough_masks, sam2_predictor):
    """Refine rough masks using SAM2."""
    if not rough_masks:
        return [], []
    
    sam2_predictor.set_image(image)
    refined, scores = [], []
    scale = IMG_SIZE / OUTPUT_SIZE
    
    for mask in rough_masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        
        # Format the mask hint for SAM2's expected input (-3 to +3 logic)
        mask_hint = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_tensor = torch.from_numpy((mask_hint * 6.0) - 3.0).unsqueeze(0).unsqueeze(0).cuda()
        
        # Create a bounding box prompt tightly enclosing the rough instance
        box = np.array([xs.min() * scale, ys.min() * scale, xs.max() * scale, ys.max() * scale])
        
        # Predict 3 multimasks and select the best scoring one
        preds, score_preds, _ = sam2_predictor.predict(box=box, mask_input=mask_tensor, multimask_output=True)
        best = score_preds.argmax()
        refined.append(preds[best])
        scores.append(float(score_preds[best]))
    
    return refined, scores


def save_geojson_safe(geojson_data: dict, output_path: str) -> bool:
    """
    Safely save GeoJSON with validation.
    Writes to temp file first, then moves to final location.
    """
    import tempfile
    import shutil
    
    output_path = Path(output_path)
    
    # Validate structure before saving
    print(f"   Validating {len(geojson_data['features'])} features...")
    
    valid_features = []
    for i, feature in enumerate(geojson_data['features']):
        try:
            coords = feature['geometry']['coordinates'][0]
            
            # Check each coordinate is [x, y] only (2D)
            valid_coords = []
            for pt in coords:
                if len(pt) >= 2:
                    # Force 2D - take only first two values
                    valid_coords.append([float(pt[0]), float(pt[1])])
                else:
                    print(f"   Warning: Feature {i} has invalid coordinate: {pt}")
                    continue
            
            # Ensure closed polygon
            if valid_coords and valid_coords[0] != valid_coords[-1]:
                valid_coords.append([valid_coords[0][0], valid_coords[0][1]])
            
            # Need at least 4 points
            if len(valid_coords) < 4:
                print(f"   Warning: Feature {i} has too few coordinates ({len(valid_coords)}), skipping")
                continue
            
            # Update feature with validated coordinates
            feature['geometry']['coordinates'] = [valid_coords]
            valid_features.append(feature)
            
        except Exception as e:
            print(f"   Warning: Feature {i} validation failed: {e}")
            continue
    
    print(f"   Valid features: {len(valid_features)} / {len(geojson_data['features'])}")
    
    geojson_data['features'] = valid_features
    
    # Write to temp file first
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as tmp:
            json.dump(geojson_data, tmp)
            tmp_path = tmp.name
        
        # Verify temp file is valid JSON
        with open(tmp_path, 'r') as f:
            _ = json.load(f)  # This will raise if invalid
        
        # Move to final location
        shutil.move(tmp_path, output_path)
        print(f"   ✓ Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to save: {e}")
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        return False


def masks_to_geojson_features(masks, scores, offset_x=0, offset_y=0, id_prefix="",
                               min_area=MIN_TUBULE_AREA, max_area=MAX_TUBULE_AREA):
    """Convert binary masks to QuPath-compatible GeoJSON features with validation."""
    features = []
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        try:
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # 1. Preliminary area check and simplification
            # Simplification (epsilon=0.5) removes micro-artifacts from SAM2
            poly = cv2.approxPolyDP(contour, 0.5, True)
            if len(poly) < 3:
                continue
            
            # 2. Extract and Round coordinates (1 decimal place is plenty for WSI)
            points = poly.reshape(-1, 2)
            raw_coords = []
            for pt in points:
                x = round(float(pt[0] + offset_x), 1)
                y = round(float(pt[1] + offset_y), 1)
                raw_coords.append((x, y))
            
            if len(raw_coords) < 3:
                continue

            # 3. Use Shapely to fix "Reduction" / Topology errors
            # buffer(0) is a standard trick to fix self-intersecting polygons
            shape = ShapePolygon(raw_coords).buffer(0)
            
            if shape.is_empty or not shape.is_valid:
                continue

            # Handle potential MultiPolygons if buffer(0) split a self-intersector
            if shape.geom_type == 'MultiPolygon':
                shape = max(shape.geoms, key=lambda a: a.area)
            
            # Final area filter using validated geometry
            if shape.area < min_area or shape.area > max_area:
                continue

            # 4. Final GeoJSON Coordinate formatting
            # exterior.coords provides CCW winding order by default
            final_coords = [list(pt) for pt in shape.exterior.coords]

            features.append({
                "type": "Feature",
                "id": f"{id_prefix}{i}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [final_coords]
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": "Tubule",
                        "colorRGB": -65536  # QuPath Red
                    },
                    "isLocked": False,
                    "measurements": [
                        {"name": "Confidence", "value": round(float(score), 4)},
                        {"name": "Area px", "value": round(float(shape.area), 1)}
                    ]
                }
            })
            
        except Exception as e:
            # Silently skip truly broken geometries
            continue
    
    return features


# ============== Main ==============

@torch.inference_mode()
def run(image_path, output_path):
    """
    Main entry point for patch-level inference pipeline:
    1. Loads models and image
    2. Runs feature extraction and distance prediction
    3. Converts distance map to separated tubule instances
    4. Refines instances with SAM2
    5. Saves output to a GeoJSON file
    """
    midnight, sam2_enc, sam2_predictor, model = load_models()
    
    print(f"Loading image: {image_path}")
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    
    # 1. Resize image to model resolution
    h, w = image.shape[:2]
    if h != IMG_SIZE or w != IMG_SIZE:
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    else:
        image_resized = image
    
    # Format according to ImageNet standard
    img_norm = (image_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().cuda()
    
    # 2. Forward pass: extract features and predict distance
    m_feat, s_feat = extract_features(midnight, sam2_enc, img_tensor)
    distance_map = model(m_feat, s_feat)[0, 0].cpu().numpy()
    
    # 3. Post-processing: isolate distinct seeds and refine boundaries
    rough_masks = distance_to_instances(distance_map)
    refined_masks, scores = refine_with_sam2(image_resized, rough_masks, sam2_predictor)
    
    # Scale masks back to original image size
    if h != IMG_SIZE or w != IMG_SIZE:
        scaled_masks = []
        for mask in refined_masks:
            scaled = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            scaled_masks.append(scaled.astype(bool))
        refined_masks = scaled_masks
    
    features = masks_to_geojson_features(refined_masks, scores, min_area=MIN_TUBULE_AREA, max_area=MAX_TUBULE_AREA)
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    save_geojson_safe(geojson_data, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <input_image> <output_geojson>")
        sys.exit(1)
        
    run(sys.argv[1], sys.argv[2])