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

    def search(self, image_path: str | Path, top_k: int = 5, ef_search: int = 16) -> list[dict]:
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
        raw_results = self.vector_store.search(norm_vec, top_k=top_k, ef_search=ef_search)

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

    def search_db_sequential(self, image_path: str | Path, top_k: int = 5) -> list[dict]:
        """
        Phương án 1: Tìm kiếm tuần tự trong Database (Linear Scan).
        Được dùng để so sánh hiệu năng với FAISS.
        """
        from src.storage.db import get_session
        from src.storage.models import Feature, Image
        from scipy.spatial.distance import cdist
        
        # 1. Trích xuất và chuẩn hóa
        feat = extract(image_path)
        norm_feat = self.normalizer.transform(feat.reshape(1, -1))
        
        # 2. Lấy toàn bộ vector từ Database
        with get_session() as session:
            db_features = session.query(Feature).all()
            if not db_features:
                return []
            
            db_vectors = np.array([f.vector for f in db_features], dtype=np.float32)
            db_image_ids = [f.image_id for f in db_features]
            
            # 3. Tính toán khoảng cách Cosine tuần tự
            # cdist với metric='cosine' trả về Cosine Distance (1 - Cosine Similarity)
            distances = cdist(norm_feat, db_vectors, metric='cosine').flatten()
            
            # Tính điểm tương đồng: Cosine Similarity = 1 - Cosine Distance
            similarities = 1.0 - distances
            
            # 4. Sắp xếp và lấy Top-K (Sắp xếp giảm dần theo similarity)
            indices = np.argsort(-similarities)[:top_k]
            
            results = []
            for rank, idx in enumerate(indices):
                img_id = db_image_ids[idx]
                img = session.get(Image, img_id)
                results.append({
                    "image_id": img.id,
                    "file_path": img.file_path,
                    "species": img.species,
                    "common_name": img.common_name,
                    "age_class": img.age_class,
                    "similarity": round(float(similarities[idx]), 4),
                    "rank": rank + 1
                })
            return results

    def rebuild_index(self, M: int = 32, ef_construction: int = 40):
        """Build lại FAISS index từ SQLite database với tham số mới"""
        from src.storage.db import get_session
        from src.storage.models import Feature
        
        with get_session() as session:
            db_features = session.query(Feature).all()
            if not db_features:
                raise ValueError("Không có vector nào trong cơ sở dữ liệu để build index.")
            
            db_vectors = np.array([f.vector for f in db_features], dtype=np.float32)
            db_image_ids = np.array([f.image_id for f in db_features], dtype=np.int64)
            
            # Khởi tạo lại vector_store với config hiện tại
            self.vector_store = VectorStore(
                index_path=self.config["storage"]["faiss_index_path"],
                ids_path=self.config["storage"]["image_ids_path"]
            )
            # Build
            self.vector_store.build(db_vectors, db_image_ids, M=M, ef_construction=ef_construction)
            self._loaded = True
