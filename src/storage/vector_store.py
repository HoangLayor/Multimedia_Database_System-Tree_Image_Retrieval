"""
vector_store.py — FAISS wrapper

Tại sao FAISS thay vì brute-force numpy?
- 1000 ảnh: không cần thiết, nhưng code pattern đúng từ đầu
- 10000+ ảnh: FAISS IVFFlat nhanh hơn brute-force ~50x
- Nâng cấp lên Milvus/Qdrant sau: chỉ cần thay class này, không sửa gì khác
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


class VectorStore:
    """
    Interface tối giản cho vector similarity search.
    Hiện tại dùng FAISS. Để swap sang Milvus/Qdrant,
    chỉ cần tạo class mới với cùng interface này.
    """

    def __init__(self, index_path: str, ids_path: str, dim: int = 27):
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.dim = dim
        self._index = None
        self._image_ids: np.ndarray | None = None

    def build(self, vectors: np.ndarray, image_ids: np.ndarray) -> None:
        """Xây dựng FAISS index từ ma trận features (N x dim)."""
        import faiss

        assert vectors.shape[1] == self.dim, f"Expected dim={self.dim}, got {vectors.shape[1]}"
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)  # chuẩn hóa để cosine = inner product

        # Sử dụng HNSW (Hierarchical Navigable Small World) theo yêu cầu
        # M = 32: Số lượng liên kết tối đa trên mỗi node (thường 16-64)
        M = 32
        index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_INNER_PRODUCT)
        
        # Tùy chỉnh tham số HNSW (tùy chọn)
        # efConstruction: Ảnh hưởng thời gian build và chất lượng (cao = tốt hơn nhưng build chậm)
        # index.hnsw.efConstruction = 40 

        index.add(vectors)

        faiss.write_index(index, str(self.index_path))
        np.save(str(self.ids_path), image_ids)
        self._index = index
        self._image_ids = image_ids

    def load(self) -> bool:
        """Load index từ disk. Trả về True nếu thành công."""
        import faiss
        if not self.index_path.exists() or not self.ids_path.exists():
            return False
        self._index = faiss.read_index(str(self.index_path))
        self._image_ids = np.load(str(self.ids_path))
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Returns: list of {"image_id": int, "score": float}
        score = cosine similarity [0, 1], cao hơn = giống hơn
        """
        if self._index is None:
            if not self.load():
                return []  # Không có index, không có kết quả

        import faiss
        q = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        scores, indices = self._index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "image_id": int(self._image_ids[idx]),
                "score": float(score),
            })
        return results
