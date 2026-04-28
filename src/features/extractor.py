"""
extractor.py — Orchestrator trích chọn 21 features

Thiết kế: mỗi nhóm feature là một module riêng biệt.
Thêm/bỏ/thay feature chỉ cần sửa một file, không ảnh hưởng các module khác.
"""
from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path

from .geometry import extract_geometry
from .color import extract_color
from .texture import extract_texture
from .shape import extract_shape


FEATURE_NAMES = [
    # Geometry (5)
    "height_width_ratio", "canopy_density", "trunk_width_ratio",
    "canopy_symmetry", "branch_angle_variance",
    # Color (9)
    "primary_green_hue", "green_saturation", "leaf_color_variance",
    "brown_gray_ratio",
    "color_hist_bin_1", "color_hist_bin_2", "color_hist_bin_3",
    "color_hist_bin_4", "color_hist_bin_5",
    # Texture (4)
    "texture_coarseness", "contour_complexity", "edge_density", "texture_entropy",
    # Shape (3 Hu + solidity + circularity = 5, tổng 21 không phải 17)
    "hu_moment_1", "hu_moment_2", "hu_moment_3", "solidity", "circularity",
]

DIM = len(FEATURE_NAMES)  # 21


def extract(image_path: str | Path) -> np.ndarray:
    """
    Trích chọn vector 21 features từ một file ảnh.
    Ảnh phải đã qua preprocessing (512x512, BGR).

    Returns: np.ndarray shape (21,), float32
    Raises: ValueError nếu ảnh không đọc được
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    geo  = extract_geometry(img)   # 5 values
    col  = extract_color(img)      # 9 values
    tex  = extract_texture(img)    # 4 values
    shp  = extract_shape(img)      # 5 values (3 Hu + solidity + circularity)

    vector = np.array(geo + col + tex + shp, dtype=np.float32)
    assert len(vector) == DIM, f"Feature dim mismatch: {len(vector)} != {DIM}"
    return vector


def extract_batch(image_paths: list, verbose: bool = False) -> tuple[np.ndarray, list]:
    """
    Trích chọn features cho nhiều ảnh.
    Returns: (matrix NxDIM, list of valid paths)
    Bỏ qua ảnh lỗi thay vì raise exception.
    """
    vectors = []
    valid_paths = []
    for i, path in enumerate(image_paths):
        if verbose and i % 100 == 0:
            print(f"  [{i}/{len(image_paths)}] extracting...")
        try:
            vec = extract(path)
            vectors.append(vec)
            valid_paths.append(path)
        except Exception as e:
            print(f"  Skip {path}: {e}")
    return np.array(vectors, dtype=np.float32), valid_paths
