"""
engine.py — Search Engine chính

Nhận ảnh query → trả về TOP-K kết quả từ CSDL.
Tách biệt với storage layer để dễ test và thay thế.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from ..features.extractor import extract
from ..features.normalizer import Normalizer
from ..storage.vector_store import VectorStore
from ..storage.db import get_session
from ..storage.models import Image


class SearchEngine:
    def __init__(self, config: dict):
        self.config = config
        self.vector_store = VectorStore(
            index_path=config["storage"]["faiss_index_path"],
            ids_path=config["storage"]["image_ids_path"],
        )
        self.normalizer = Normalizer()
        self.normalizer.load(config["data"]["features_dir"])
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.vector_store.load()
            self._loaded = True

    def search(self, image_path: str | Path, top_k: int = 5) -> list[dict]:
        """
        Tìm TOP-K ảnh giống nhất với ảnh query.

        Returns: list of {
            "rank": int,
            "image_id": int,
            "filename": str,
            "species": str,
            "age_class": str,
            "similarity": float,   # [0, 1]
            "thumbnail_path": str,
        }
        """
        self._ensure_loaded()

        # 1. Trích chọn features
        raw_vec = extract(image_path)

        # 2. Chuẩn hóa bằng cùng scaler đã dùng khi build index (nếu có)
        if self.normalizer.mean_ is not None:
            norm_vec = self.normalizer.transform(raw_vec)
        else:
            norm_vec = raw_vec

        # 3. Vector search
        raw_results = self.vector_store.search(norm_vec, top_k=top_k)

        # 4. Enrich với metadata từ SQLite
        results = []
        with get_session() as session:
            for rank, r in enumerate(raw_results, start=1):
                img = session.get(Image, r["image_id"])
                if img is None:
                    continue
                results.append({
                    "rank": rank,
                    "image_id": img.id,
                    "filename": img.filename,
                    "species": img.species,
                    "common_name": img.common_name,
                    "age_class": img.age_class,
                    "similarity": round(r["score"], 4),
                    "file_path": img.file_path,
                })
        return results
