"python merge_segmentation.py -i HUK1_COR1_segmentations/tubules.geojson -o HUK1_COR1_segmentations/tubules_merged.geojson -t 0.8"

import json
import argparse
import numpy as np
import os
from pathlib import Path
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

import rasterio
from rasterio import features
from skimage.segmentation import watershed
from scipy import ndimage as sho

def is_tile_edge_artifact(poly, length_threshold=70, tolerance=5):
    """
    Detects artifact edges by looking at cumulative distance.
    Checks if a sequence of points stays within a narrow corridor over a long distance,
    even if broken into many tiny segments.
    """
    if not isinstance(poly, Polygon):
        return False
    
    coords = np.array(poly.exterior.coords)
    n_points = len(coords)
    
    for i in range(n_points):
        for j in range(i + 2, n_points):
            p1, p2 = coords[i], coords[j]
            dx = abs(p1[0] - p2[0])
            dy = abs(p1[1] - p2[1])
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > length_threshold:
                segment_subset = coords[i:j+1]
                
                # Check Horizontal Corridor
                if dy <= tolerance:
                    y_range = np.max(segment_subset[:, 1]) - np.min(segment_subset[:, 1])
                    if y_range <= tolerance:
                        return True
                
                # Check Vertical Corridor
                if dx <= tolerance:
                    x_range = np.max(segment_subset[:, 0]) - np.min(segment_subset[:, 0])
                    if x_range <= tolerance:
                        return True
                
                if dx > tolerance and dy > tolerance:
                    break
    return False

def resolve_touching_pair(poly_a, poly_b, res=0.5):
    """
    Separates two overlapping/touching polygons using a watershed algorithm.
    Rasterizes the union of the two polygons and computes a distance transform 
    to robustly split them apart along complex boundaries.
    """
    merged = poly_a.union(poly_b)
    minx, miny, maxx, maxy = merged.bounds
    pad = res * 4  
    width = int(np.ceil((maxx - minx + 2*pad) / res))
    height = int(np.ceil((maxy - miny + 2*pad) / res))
    
    if width <= 5 or height <= 5:
        return poly_a, poly_b

    transform = rasterio.transform.from_origin(minx-pad, maxy+pad, res, res)
    # Rasterize the combined shapes to create a binary mask for the watershed
    combined_mask = features.rasterize([(merged, 1)], out_shape=(height, width), transform=transform, fill=0)
    
    # Place markers at the representative centers or 'centroids' of the original separated polygons
    markers = np.zeros((height, width), dtype=np.int32)
    for i, p in enumerate([poly_a, poly_b], 1):
        try:
            pt = p.representative_point()
            col, row = ~transform * (pt.x, pt.y)
            markers[np.clip(int(row), 0, height-1), np.clip(int(col), 0, width-1)] = i
        except:
            continue

    # Compute Euclidean distance transform from the mask boundaries inwards
    distance = sho.distance_transform_edt(combined_mask)
    
    # Perform watershed starting from the markers, letting the boundary settle along the distance ridge
    labels = watershed(-distance, markers, mask=combined_mask)

    new_polys = []
    for val in [1, 2]:
        mask = (labels == val)
        shapes_gen = features.shapes(labels.astype(np.int32), mask=mask, transform=transform)
        parts = [shape(g) for g, v in shapes_gen]
        if parts:
            new_p = unary_union(parts)
            if new_p.has_z:
                if isinstance(new_p, Polygon):
                    new_p = Polygon([(pt[0], pt[1]) for pt in new_p.exterior.coords])
                else:
                    new_p = MultiPolygon([Polygon([(pt[0], pt[1]) for pt in p.exterior.coords]) for p in new_p.geoms])
            new_polys.append(new_p)
            
    return (new_polys[0], new_polys[1]) if len(new_polys) == 2 else (poly_a, poly_b)

def merge_and_separate(input_path, output_path, merge_thresh, water_thresh):
    """
    Main algorithmic pipeline for cleaning WSI tiled tubule segmentations:
    1. Filters out tile-edge artifacts.
    2. Uses a spatial index (STRtree) to find neighboring tubules.
    3. If overlap is large (>= merge_thresh), merges them into a single tubule.
    4. If overlap is moderate (>= water_thresh), attempts watershed separation.
    """
    import ijson
    valid_geoms, valid_props = [], []
    total_count = 0

    print("Loading and filtering features incrementally...")
    with open(input_path, 'rb') as f:
        for feat in ijson.items(f, 'features.item'):
            total_count += 1
            poly = shape(feat['geometry']).buffer(0)
            if not is_tile_edge_artifact(poly, length_threshold=70, tolerance=5):
                valid_geoms.append(poly)
                valid_props.append(feat.get('properties', {}))
    
    removed_count = total_count - len(valid_geoms)
    print(f"Removed {removed_count} artifact features.")
    
    geoms, props = valid_geoms, valid_props
    
    if not geoms:
        print("No valid features found.")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write('{"type":"FeatureCollection","features":[]}\n')
        return

    indices = sorted(range(len(geoms)), key=lambda i: geoms[i].area, reverse=True)
    processed = [False] * len(geoms)

    print("Building spatial tree...")
    tree = STRtree(geoms)
    
    saved_count = 0
    print(f"Processing and streaming out features to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write('{"type":"FeatureCollection","features":[\n')
        first_feature = True

        for i in indices:
            if processed[i]: continue
            current_geom = geoms[i]
            
            # Extremely fast O(log N) bounding-box nearest neighbor queries
            neighbor_indices = tree.query(current_geom)
            
            for n_idx in neighbor_indices:
                n_idx = int(n_idx)
                if n_idx == i or processed[n_idx]: continue
                
                neighbor_geom = geoms[n_idx]
                
                # Exact intersection geometry verification
                if not current_geom.intersects(neighbor_geom): continue
                
                # Calculate intersection over the smaller geometry's area
                inter_area = current_geom.intersection(neighbor_geom).area
                smaller_area = min(current_geom.area, neighbor_geom.area)
                overlap_ratio = inter_area / (smaller_area + 1e-9)

                if overlap_ratio >= merge_thresh:
                    # Overlap is too high, assume it is the same tubule
                    current_geom = current_geom.union(neighbor_geom).buffer(0)
                    processed[n_idx] = True
                elif overlap_ratio < water_thresh and overlap_ratio > 0.001:
                    # Overlap is moderate, attempt to separate them via watershed prediction
                    res_a, res_b = resolve_touching_pair(current_geom, neighbor_geom, res=0.5)
                    current_geom, geoms[n_idx] = res_a, res_b  

            out_feat = {"type": "Feature", "geometry": mapping(current_geom), "properties": props[i]}
            
            # Write feature
            separator = ",\n" if not first_feature else ""
            out_f.write(separator + json.dumps(out_feat))
            first_feature = False
            
            processed[i] = True
            saved_count += 1

        out_f.write('\n]}\n')
    
    print(f"✓ Success: Saved {saved_count} features.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--merge', type=float, default=0.75)
    parser.add_argument('--water', type=float, default=0.3)
    args = parser.parse_args()
    
    merge_and_separate(args.input, args.output, args.merge, args.water)