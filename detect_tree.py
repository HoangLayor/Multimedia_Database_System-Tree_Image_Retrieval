#!/usr/bin/env python3
"""
Tree detection via SegFormer semantic segmentation (ADE20K).

Usage:
    python3 detect_tree.py <image_path> [image_path2 ...]
    python3 detect_tree.py --batch <file_with_paths>

Outputs JSON array to stdout with tree bounding boxes for each image.
"""

import sys
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

# ADE20K class indices for vegetation
TREE_CLASS = 4       # "tree"
PLANT_CLASS = 17     # "plant, flora"
GRASS_CLASS = 9      # "grass"

# Classes to consider as "tree" in segmentation
TREE_CLASSES = {TREE_CLASS, PLANT_CLASS}

# Padding around detected tree box (fraction of box size)
BOX_PADDING = 0.05

# Minimum area fraction for a tree region to be considered valid
MIN_TREE_AREA_FRACTION = 0.005  # 0.5% of image


def load_model():
    """Load SegFormer model and processor (cached after first download ~14MB)."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    return processor, model


def segment_image(image: Image.Image, processor, model) -> np.ndarray:
    """Run segmentation and return class map at original resolution."""
    import torch
    import torch.nn.functional as F

    original_size = image.size  # (W, H)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, num_classes, H/4, W/4)

    # Upsample to original image size
    upsampled = F.interpolate(
        logits,
        size=(original_size[1], original_size[0]),  # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)
    return seg_map


def find_tree_regions(seg_map: np.ndarray, image_width: int, image_height: int):
    """
    Find connected tree regions in segmentation map.
    Returns list of bounding boxes sorted by area (largest first).
    """
    from scipy import ndimage

    # Create binary mask for tree classes
    tree_mask = np.isin(seg_map, list(TREE_CLASSES)).astype(np.uint8)

    total_pixels = image_width * image_height
    min_area = int(total_pixels * MIN_TREE_AREA_FRACTION)

    if tree_mask.sum() < min_area:
        return []

    # Find connected components
    labeled, num_features = ndimage.label(tree_mask)

    boxes = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        area = component.sum()

        if area < min_area:
            continue

        # Get bounding box: (min_row, min_col, max_row, max_col)
        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]

        y_min, y_max = int(rows[0]), int(rows[-1])
        x_min, x_max = int(cols[0]), int(cols[-1])

        w = x_max - x_min + 1
        h = y_max - y_min + 1

        # Add padding
        pad_x = int(w * BOX_PADDING)
        pad_y = int(h * BOX_PADDING)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(image_width - 1, x_max + pad_x)
        y_max = min(image_height - 1, y_max + pad_y)

        boxes.append({
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min + 1,
            "h": y_max - y_min + 1,
            "area": int(area),
        })

    # Sort by area descending (largest tree first)
    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes


def detect_tree(image_path: str, processor, model) -> dict:
    """Detect trees in a single image. Returns detection result dict."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "file": image_path,
            "error": str(e),
            "tree_found": False,
        }

    width, height = image.size

    try:
        seg_map = segment_image(image, processor, model)
        boxes = find_tree_regions(seg_map, width, height)
    except Exception as e:
        return {
            "file": image_path,
            "width": width,
            "height": height,
            "error": str(e),
            "tree_found": False,
        }

    if not boxes:
        return {
            "file": image_path,
            "width": width,
            "height": height,
            "tree_found": False,
            "tree_box": None,
            "tree_coverage": 0.0,
            "all_tree_boxes": [],
        }

    primary = boxes[0]
    total_tree_area = sum(b["area"] for b in boxes)
    coverage = total_tree_area / (width * height)

    # Remove 'area' key from output boxes (internal use only)
    clean_boxes = [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]} for b in boxes]

    return {
        "file": image_path,
        "width": width,
        "height": height,
        "tree_found": True,
        "tree_box": clean_boxes[0],
        "tree_coverage": round(coverage, 4),
        "all_tree_boxes": clean_boxes,
    }


def main():
    args = sys.argv[1:]

    if not args:
        print(json.dumps({"error": "No image paths provided"}), file=sys.stderr)
        sys.exit(1)

    # Handle --batch mode
    if args[0] == "--batch":
        if len(args) < 2:
            print(json.dumps({"error": "--batch requires a file path"}), file=sys.stderr)
            sys.exit(1)
        with open(args[1]) as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        image_paths = args

    # Validate paths
    valid_paths = []
    for p in image_paths:
        if os.path.isfile(p):
            valid_paths.append(p)
        else:
            print(f"Warning: skipping {p} (not found)", file=sys.stderr)

    if not valid_paths:
        print(json.dumps([]), file=sys.stdout)
        sys.exit(0)

    # Load model once
    print(f"Loading SegFormer model...", file=sys.stderr)
    processor, model = load_model()
    print(f"Model loaded. Processing {len(valid_paths)} image(s)...", file=sys.stderr)

    results = []
    for i, path in enumerate(valid_paths):
        result = detect_tree(path, processor, model)
        results.append(result)
        status = "✓ tree found" if result.get("tree_found") else "✗ no tree"
        print(f"  [{i+1}/{len(valid_paths)}] {Path(path).name}: {status}", file=sys.stderr)

    print(json.dumps(results, indent=2), file=sys.stdout)


if __name__ == "__main__":
    main()
