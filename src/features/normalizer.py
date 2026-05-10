"""
normalizer.py — Z-score normalization

Quan trọng: phải dùng CÙNG scaler khi index và khi query.
Scaler được lưu vào disk sau khi fit, load lại mỗi lần query.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


class Normalizer:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, matrix: np.ndarray) -> "Normalizer":
        """Tính mean và std từ training set."""
        if matrix.size == 0 or matrix.ndim < 2:
            # Tránh lỗi khi không có dữ liệu
            self.mean_ = np.zeros((1, 27)) # Hoặc giá trị mặc định
            self.std_ = np.ones((1, 27))
            return self

        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # tránh chia 0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Chuẩn hóa vector hoặc matrix."""
        assert self.mean_ is not None, "Call fit() trước"
        return (x - self.mean_) / self.std_

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        return self.fit(matrix).transform(matrix)

    def save(self, directory: str) -> None:
        d = Path(directory)
        np.save(str(d / "scaler_mean.npy"), self.mean_)
        np.save(str(d / "scaler_std.npy"), self.std_)

    def load(self, directory: str) -> bool:
        """Load scaler từ disk. Trả về True nếu thành công."""
        d = Path(directory)
        mean_path = d / "scaler_mean.npy"
        std_path = d / "scaler_std.npy"
        
        if not mean_path.exists() or not std_path.exists():
            return False
            
        self.mean_ = np.load(str(mean_path))
        self.std_ = np.load(str(std_path))
        return True
