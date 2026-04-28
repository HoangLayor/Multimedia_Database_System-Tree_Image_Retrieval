# Tree Image Search System
> Hệ CSDL lưu trữ và tìm kiếm ảnh cây — Bài tập lớn môn Hệ CSDL Phân tán

## Kiến trúc tổng quan

```
Thu thập ảnh → Validation → Preprocessing → Feature Extraction → Storage → Search API
 (iNaturalist)  (filter)    (resize/norm)   (21-dim vector)  (SQLite+FAISS) (FastAPI)
```

## Cấu trúc thư mục

```
tree-image-search/
├── configs/                  # Cấu hình hệ thống (YAML)
│   ├── default.yaml          # Config mặc định
│   └── features.yaml         # Định nghĩa 21 features + trọng số
│
├── data/                     # Dữ liệu (không commit ảnh lên git)
│   ├── raw/                  # Ảnh gốc sau khi download
│   ├── processed/            # Ảnh đã chuẩn hóa (512x512)
│   ├── features/             # Feature matrix (.npy) + FAISS index
│   └── thumbnails/           # Ảnh nhỏ 128x128 cho UI
│
├── src/                      # Source code chính
│   ├── ingestion/            # Thu thập & làm sạch dữ liệu
│   │   ├── collector.py      # Download từ iNaturalist/GBIF API
│   │   ├── validator.py      # Lọc ảnh (kích thước, màu, aspect ratio)
│   │   ├── preprocessor.py   # Resize, CLAHE, chuẩn hóa màu
│   │   └── deduplicator.py   # Loại ảnh trùng (perceptual hash)
│   │
│   ├── features/             # Trích chọn đặc trưng
│   │   ├── extractor.py      # Orchestrator → vector 21-dim
│   │   ├── geometry.py       # Nhóm 1: hình học (5 features)
│   │   ├── color.py          # Nhóm 2: màu sắc (9 features)
│   │   ├── texture.py        # Nhóm 3: texture (4 features)
│   │   ├── shape.py          # Nhóm 4: hình dạng (3 features)
│   │   └── normalizer.py     # Z-score fit/transform, lưu scaler
│   │
│   ├── storage/              # Lưu trữ
│   │   ├── models.py         # SQLAlchemy models: Image, Feature, SearchLog
│   │   ├── db.py             # SQLite session, CRUD
│   │   ├── vector_store.py   # FAISS: build/save/load/search
│   │   └── migrations.py     # Schema migrations
│   │
│   ├── search/               # Engine tìm kiếm
│   │   ├── engine.py         # SearchEngine: ảnh → TOP-K results
│   │   ├── metrics.py        # Cosine, Euclidean, Weighted distance
│   │   └── ranker.py         # Re-ranking, dedup
│   │
│   └── api/                  # Web API
│       ├── app.py            # FastAPI app factory
│       ├── routes.py         # POST /search, GET /images/{id}
│       └── schemas.py        # Pydantic request/response models
│
├── scripts/                  # CLI độc lập
│   ├── collect_data.py       # --species oak --limit 200
│   ├── build_index.py        # Tính features + build FAISS index
│   └── evaluate.py           # precision@5, query time benchmark
│
├── notebooks/                # EDA & demo
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_exploration.ipynb
│   └── 03_search_demo.ipynb
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   └── test_search.py
│   └── integration/
│       └── test_pipeline.py
│
├── docs/
│   ├── architecture.md
│   └── features.md
│
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/collect_data.py --limit 1000
python scripts/build_index.py
uvicorn src.api.app:app --reload
```
