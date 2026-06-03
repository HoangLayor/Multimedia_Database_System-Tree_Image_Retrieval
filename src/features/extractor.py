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
    # Nhóm 1: Hình học (5)
    "tỷ_lệ_cao_rộng", "mật_độ_tán", "đường_kính_thân",
    "độ_đối_xứng", "phương_sai_góc_cành",
    # Nhóm 2: Màu sắc (9)
    "sắc_xanh_chủ_đạo", "độ_bão_hòa_xanh", "phương_sai_màu_lá",
    "tỷ_lệ_nâu_xám",
    "biểu_đồ_màu_bin_1", "biểu_đồ_màu_bin_2", "biểu_đồ_màu_bin_3",
    "biểu_đồ_màu_bin_4", "biểu_đồ_màu_bin_5",
    # Nhóm 3: Kết cấu (4)
    "độ_nhám_kết_cấu", "độ_phức_tạp_đường_viền", "mật_độ_cạnh", "entropy_kết_cấu",
    # Nhóm 4: Hình dạng (9)
    "moment_hu_1", "moment_hu_2", "moment_hu_3", "moment_hu_4",
    "moment_hu_5", "moment_hu_6", "moment_hu_7",
    "độ_đặc", "độ_tròn",
]

DIM = len(FEATURE_NAMES)  # 27


def extract(image_path: str | Path) -> np.ndarray:
    """
    Trích chọn vector 27 features từ một file ảnh.
    Ảnh phải đã qua preprocessing (512x512, BGR).

    Returns: np.ndarray shape (27,), float32
    Raises: ValueError nếu ảnh không đọc được
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    geo  = extract_geometry(img)   # 5 values
    col  = extract_color(img)      # 9 values
    tex  = extract_texture(img)    # 4 values
    shp  = extract_shape(img)      # 9 values (7 Hu + solidity + circularity)

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
