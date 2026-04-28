# Kiến trúc hệ thống

## Nguyên tắc thiết kế

1. **Tối giản**: chỉ thêm dependency khi thực sự cần. 1000 ảnh không cần Kafka hay Kubernetes.
2. **Scalability qua interface**: mỗi tầng giao tiếp qua interface, không hard-code implementation.
3. **Linh hoạt backend**: nâng cấp từ SQLite → PostgreSQL, FAISS → Milvus chỉ cần đổi config.

## Sơ đồ tầng

```
[API Layer]         FastAPI — nhận ảnh upload, trả JSON
      |
[Search Layer]      SearchEngine — điều phối query pipeline
      |
[Feature Layer]     Extractor — tính 21-dim vector từ ảnh
      |
[Storage Layer]     SQLite (metadata) + FAISS (vectors)
      |
[Ingestion Layer]   Collector + Validator + Preprocessor + Deduplicator
```

## Quyết định thiết kế

### Tại sao SQLite chứ không phải PostgreSQL?
Với 1000 ảnh, SQLite đủ dùng và không cần setup server.
Nâng cấp: thay `db_path` bằng `postgres://...` trong `default.yaml`.

### Tại sao FAISS chứ không phải brute-force numpy?
- Cùng accuracy nhưng nhanh hơn 20-100x khi N > 5000
- Pattern đúng từ đầu: nếu sau này tăng lên 50k ảnh, chỉ cần đổi index type từ `IndexFlatIP` sang `IndexIVFFlat`

### Tại sao không dùng MongoDB?
Metadata của ảnh cây là structured (tên loài, tuổi, species) — phù hợp SQL hơn document store.
MongoDB hữu ích khi metadata không đồng nhất hoặc cần horizontal sharding > 1M records.

### Tại sao features lưu cả SQL lẫn FAISS?
- SQL: dễ debug, filter, audit ("tìm tất cả ảnh có canopy_density > 0.8")
- FAISS: vector search nhanh
- Hai nơi luôn đồng bộ nhau qua `build_index.py`
