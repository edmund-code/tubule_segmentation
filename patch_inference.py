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
import sys

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


# ============== Model ==============

class AlignmentBridge(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.bridge = AlignmentBridge()
        self.head = DistanceHead()
    
    def forward(self, m_feat, s_feat):
        return self.head(self.bridge(m_feat, s_feat))


# ============== Core Functions ==============

@torch.inference_mode()
def extract_features(midnight, sam2_enc, img_tensor):
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
    dist_uint8 = (distance_map * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(dist_uint8, int(thresh_val * THRESH_FACTOR), 255, cv2.THRESH_BINARY)
    binary = (binary > 0).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=ERODE_ITER)
    num_labels, labels = cv2.connectedComponents(eroded)
    
    masks = []
    for i in range(1, num_labels):
        seed = (labels == i).astype(np.uint8)
        dilated = cv2.dilate(seed, kernel, iterations=ERODE_ITER)
        final = dilated & binary
        if final.sum() >= MIN_AREA:
            masks.append(final)
    
    if len(masks) > 1:
        combined = sum((m > 0).astype(np.int32) for m in masks)
        if (combined > 1).any():
            centroids = [(np.where(m > 0)[1].mean(), np.where(m > 0)[0].mean()) for m in masks]
            new_masks = [m.copy() for m in masks]
            for y, x in zip(*np.where(combined > 1)):
                best = np.argmin([(x - cx)**2 + (y - cy)**2 for cx, cy in centroids])
                for i in range(len(new_masks)):
                    new_masks[i][y, x] = 1 if i == best else 0
            masks = new_masks
    
    return masks


def refine_with_sam2(image, rough_masks, sam2_predictor):
    if not rough_masks:
        return [], []
    
    sam2_predictor.set_image(image)
    refined, scores = [], []
    scale = IMG_SIZE / OUTPUT_SIZE
    
    for mask in rough_masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        
        mask_hint = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_tensor = torch.from_numpy((mask_hint * 6.0) - 3.0).unsqueeze(0).unsqueeze(0).cuda()
        box = np.array([xs.min() * scale, ys.min() * scale, xs.max() * scale, ys.max() * scale])
        
        preds, score_preds, _ = sam2_predictor.predict(box=box, mask_input=mask_tensor, multimask_output=True)
        best = score_preds.argmax()
        refined.append(preds[best])
        scores.append(float(score_preds[best]))
    
    return refined, scores


def masks_to_geojson(masks, scores):
    features = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        contour = cv2.approxPolyDP(contour, 1.0, True)
        if len(contour) < 3:
            continue
        coords = contour.squeeze().tolist()
        coords.append(coords[0])
        features.append({
            "type": "Feature",
            "id": str(i),
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"classification": {"name": "Tubule"}, "score": round(score, 4)}
        })
    return {"type": "FeatureCollection", "features": features}


# ============== Load Models ==============

def load_models():
    dl_loc = hf_hub_download(repo_id="SophontAI/OpenMidnight", filename="teacher_checkpoint_load.pt")
    midnight = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=False)
    cp = torch.load(dl_loc, map_location="cuda", weights_only=False)
    midnight.pos_embed = nn.Parameter(cp["pos_embed"])
    midnight.load_state_dict(cp)
    midnight = midnight.cuda().eval()
    
    sam2_enc = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device="cuda").image_encoder.eval()
    sam2_predictor = SAM2ImagePredictor(build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device="cuda"))
    
    model = DistanceModel().cuda().eval()
    model.load_state_dict(torch.load(DISTANCE_MODEL_PATH, map_location="cuda", weights_only=False)['model'])
    
    return midnight, sam2_enc, sam2_predictor, model


# ============== Main ==============

@torch.inference_mode()
def run(image_path, output_path):
    midnight, sam2_enc, sam2_predictor, model = load_models()
    
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    img_norm = (image_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().cuda()
    
    m_feat, s_feat = extract_features(midnight, sam2_enc, img_tensor)
    distance_map = model(m_feat, s_feat)[0, 0].cpu().numpy()
    
    rough_masks = distance_to_instances(distance_map)
    refined_masks, scores = refine_with_sam2(image_resized, rough_masks, sam2_predictor)
    
    geojson = masks_to_geojson(refined_masks, scores)
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    
    print(f"{len(refined_masks)} instances -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <input_image> <output_geojson>")
        sys.exit(1)
    
    run(sys.argv[1], sys.argv[2])