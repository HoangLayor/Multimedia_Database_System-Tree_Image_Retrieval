#!/usr/bin/env python3
"""
build_index.py — Pipeline xây dựng FAISS index từ toàn bộ ảnh

Chạy sau khi đã thu thập và validate ảnh:
    python scripts/build_index.py

Quy trình:
  1. Đọc tất cả ảnh từ data/processed/
  2. Trích chọn 21 features mỗi ảnh
  3. Fit Z-score scaler, transform toàn bộ
  4. Build FAISS index + lưu
  5. Insert metadata vào SQLite
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from src.features.extractor import extract_batch
from src.features.normalizer import Normalizer
from src.storage.vector_store import VectorStore
from src.storage.db import init_db, get_session
from src.storage.models import Image, Feature


def main():
    config = yaml.safe_load(open("configs/default.yaml"))
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir  = Path(config["data"]["features_dir"])
    features_dir.mkdir(exist_ok=True)

    print("=== Step 1: Collecting image paths ===")
    image_paths = sorted(processed_dir.glob("*.jpg"))
    print(f"  Found {len(image_paths)} images")

    print("=== Step 2: Extracting features ===")
    matrix, valid_paths = extract_batch(image_paths, verbose=True)
    print(f"  Extracted: {matrix.shape}")  # (N, 21)

    print("=== Step 3: Normalizing features ===")
    normalizer = Normalizer()
    normalizer.fit(matrix)
    norm_matrix = normalizer.transform(matrix)
    normalizer.save(str(features_dir))

    print("=== Step 4: Building FAISS index ===")
    init_db(config["storage"]["db_path"])
    with get_session() as session:
        image_ids = []
        for path in valid_paths:
            img = session.query(Image).filter_by(filename=path.name).first()
            if img:
                image_ids.append(img.id)
        image_ids = np.array(image_ids, dtype=np.int64)

    store = VectorStore(
        index_path=config["storage"]["faiss_index_path"],
        ids_path=config["storage"]["image_ids_path"],
    )
    store.build(norm_matrix, image_ids)
    print(f"  Index built: {len(image_ids)} vectors")

    print("=== Done ===")


if __name__ == "__main__":
    main()
