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


import argparse

def main():
    parser = argparse.ArgumentParser(description="Build Search Index")
    parser.add_argument("--method", type=str, choices=["faiss", "db", "both"], default="both",
                        help="Lưu vào đâu? (faiss, db, both)")
    args = parser.parse_args()

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
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

    print(f"=== Step 4: Saving features (Method: {args.method}) ===")
    init_db()
    
    image_ids = []
    # 1. Lưu vào DB nếu được yêu cầu
    if args.method in ["db", "both"]:
        with get_session() as session:
            print("  Saving vectors to Database...")
            session.query(Feature).delete()
            for i, path in enumerate(valid_paths):
                img = session.query(Image).filter_by(filename=path.name).first()
                if img:
                    image_ids.append(img.id)
                    session.add(Feature(image_id=img.id, vector=norm_matrix[i].tolist()))
            session.commit()
    else:
        # Nếu chỉ build FAISS, vẫn cần lấy list Image IDs để mapping
        with get_session() as session:
            for path in valid_paths:
                img = session.query(Image).filter_by(filename=path.name).first()
                if img: image_ids.append(img.id)

    # 2. Build FAISS nếu được yêu cầu
    if args.method in ["faiss", "both"]:
        print("  Building FAISS index...")
        store = VectorStore(
            index_path=config["storage"]["faiss_index_path"],
            ids_path=config["storage"]["image_ids_path"],
        )
        store.build(norm_matrix, np.array(image_ids, dtype=np.int64))
    
    print(f"=== Done: Processed {len(image_ids)} images ===")

    print("=== Done ===")


if __name__ == "__main__":
    main()
