"""
data_preprocessing.py

Resize and combine histopathology image datasets.

Usage:
    python data_preprocessing.py ./data ./data_combined
    python data_preprocessing.py ./data ./data_combined --size 512

Input structure:
    data/
    ├── he-pt-data/
    │   ├── image1.png
    │   ├── image1_mask_proximal.png
    │   ├── image2.png
    │   ├── image2_mask_proximal.png
    │   └── ...
    ├── he-dt-data/
    │   ├── image1.png
    │   ├── image1_mask_distal.png
    │   └── ...
    ├── pas-pt-data/
    │   └── ...
    ├── pas-dt-data/
    │   └── ...
    ├── sil-pt-data/
    │   └── ...
    ├── sil-dt-data/
    │   └── ...
    ├── tri-pt-data/
    │   └── ...
    └── tri-dt-data/
        └── ...

    Stain types: he, pas, sil, tri
    pt = proximal tubule, dt = distal tubule
    Masks must be named: {image_name}_mask_proximal.png or {image_name}_mask_distal.png

Output structure:
    data_combined/
    ├── images/
    │   ├── he_image1.png
    │   ├── he_image2.png
    │   ├── pas_image1.png
    │   └── ...
    └── masks/
        ├── he_image1.png
        ├── he_image2.png
        ├── pas_image1.png
        └── ...

    - Images are resized to specified size (default 1024x1024)
    - Masks are combined (proximal + distal) and binarized
    - Output filenames are prefixed with stain type
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_SIZE = 1024
OVERLAP_THRESHOLD = 0.3
STAINS = ["he", "pas", "sil", "tri"]


# -----------------------------------------------------------------------------
# Resize
# -----------------------------------------------------------------------------

def process_file(args):
    """Resize a single file."""
    src_path, dst_path, is_mask, size = args
    
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(img, (size, size), interpolation=interp)
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), resized)
    return True


def resize_dataset(source_dir, target_dir, size):
    """Resize all images in directory."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    files = []
    for root, _, filenames in os.walk(source_dir):
        for f in filenames:
            if f.endswith(".png"):
                src = Path(root) / f
                dst = target_dir / src.relative_to(source_dir)
                is_mask = "_mask_" in f
                files.append((src, dst, is_mask, size))
    
    if not files:
        print(f"No PNG files found in {source_dir}")
        return
    
    print(f"Resizing {len(files)} files to {size}x{size}...")
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, files), total=len(files)))


# -----------------------------------------------------------------------------
# Combine
# -----------------------------------------------------------------------------

def combine_dataset(input_root, output_root, size):
    """Combine proximal and distal masks into single dataset."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    images_dir = output_root / "images"
    masks_dir = output_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    total_saved = 0
    
    print("Combining masks...")
    
    for stain in STAINS:
        pt_folder = input_root / f"{stain}-pt-data"
        dt_folder = input_root / f"{stain}-dt-data"
        
        # Get image names
        pt_images = set()
        dt_images = set()
        
        if pt_folder.exists():
            pt_images = {f for f in os.listdir(pt_folder) 
                        if f.endswith(".png") and "_mask_" not in f}
        
        if dt_folder.exists():
            dt_images = {f for f in os.listdir(dt_folder) 
                        if f.endswith(".png") and "_mask_" not in f}
        
        all_images = sorted(pt_images | dt_images)
        
        for img_name in tqdm(all_images, desc=f"{stain.upper()}", leave=False):
            base = img_name.replace(".png", "")
            out_name = f"{stain}_{base}"
            
            # Paths
            pt_img_path = pt_folder / img_name
            dt_img_path = dt_folder / img_name
            pt_mask_path = pt_folder / f"{base}_mask_proximal.png"
            dt_mask_path = dt_folder / f"{base}_mask_distal.png"
            
            # Find image
            if pt_img_path.exists():
                img_path = pt_img_path
            elif dt_img_path.exists():
                img_path = dt_img_path
            else:
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Load masks
            pt_mask = None
            dt_mask = None
            
            if pt_mask_path.exists():
                pt_mask = cv2.imread(str(pt_mask_path), cv2.IMREAD_GRAYSCALE)
            if dt_mask_path.exists():
                dt_mask = cv2.imread(str(dt_mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Combine masks
            if pt_mask is not None and dt_mask is not None:
                pt_bin = pt_mask > 127
                dt_bin = dt_mask > 127
                intersection = np.logical_and(pt_bin, dt_bin).sum()
                union = np.logical_or(pt_bin, dt_bin).sum()
                overlap = intersection / union if union > 0 else 0
                
                if overlap < OVERLAP_THRESHOLD:
                    combined_mask = np.maximum(pt_mask, dt_mask)
                else:
                    combined_mask = pt_mask
            elif pt_mask is not None:
                combined_mask = pt_mask
            elif dt_mask is not None:
                combined_mask = dt_mask
            else:
                continue
            
            # Resize
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            combined_mask = cv2.resize(combined_mask, (size, size), 
                                       interpolation=cv2.INTER_NEAREST)
            
            # Binarize
            combined_mask = ((combined_mask > 127) * 255).astype(np.uint8)
            
            # Save
            cv2.imwrite(str(images_dir / f"{out_name}.png"), img)
            cv2.imwrite(str(masks_dir / f"{out_name}.png"), combined_mask)
            total_saved += 1
    
    print(f"Saved {total_saved} samples to {output_root}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess histopathology dataset: resize and combine masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input data folder")
    parser.add_argument("output", type=Path, help="Output folder")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, 
                        help=f"Image size (default: {DEFAULT_SIZE})")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input folder does not exist: {args.input}")
        sys.exit(1)
    
    # Temporary folder for resized data
    temp_dir = Path(f"./_temp_resized_{args.size}")
    
    print(f"\n{'='*50}")
    print("Step 1: Resizing")
    print(f"{'='*50}")
    resize_dataset(args.input, temp_dir, args.size)
    
    print(f"\n{'='*50}")
    print("Step 2: Combining")
    print(f"{'='*50}")
    combine_dataset(temp_dir, args.output, args.size)
    
    # Cleanup temp folder
    shutil.rmtree(temp_dir)
    
    print(f"\n{'='*50}")
    print(f"Done! Output: {args.output}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()