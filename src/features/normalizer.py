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

    def load(self, directory: str) -> "Normalizer":
        d = Path(directory)
        self.mean_ = np.load(str(d / "scaler_mean.npy"))
        self.std_ = np.load(str(d / "scaler_std.npy"))
        return self
