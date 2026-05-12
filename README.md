# 🌳 Tree Image Search System
> Hệ thống CSDL đa phương tiện lưu trữ và tìm kiếm ảnh cây thông minh.

Hệ thống cho phép tìm kiếm ảnh cây dựa trên nội dung (CBIR) bằng cách sử dụng bộ đặc trưng 27 chiều (Geometry, Color, Texture, Shape) kết hợp với FAISS Index để truy xuất tốc độ cao.

## ✨ Tính năng chính
- **Trích xuất 27 đặc trưng**: Phân tích chi tiết về hình dáng, màu sắc và kết cấu của cây.
- **Tiền xử lý thông minh**: Tự động tách cây khỏi nền bằng thuật toán Otsu Adaptive.
- **Tìm kiếm tương đồng**: Hỗ trợ độ đo Cosine, Euclidean và Weighted Metric.
- **Lưu trữ Hybrid**: 
  - Metadata: SQLite (SQLAlchemy).
  - Feature Index: FAISS (Facebook AI Similarity Search).
  - Image Storage: **MinIO (S3 Compatible)**.
- **Giao diện hiện đại**: Streamlit UI với khả năng trực quan hóa quá trình trích xuất đặc trưng.

## 🛠 Cấu trúc thư mục
```text
tree-image-search/
├── configs/                  # Cấu hình hệ thống (YAML)
├── data/                     # Dữ liệu local (raw, processed, features)
├── src/                      # Mã nguồn chính
│   ├── features/             # Module trích xuất đặc trưng (27 chiều)
│   ├── storage/              # Lưu trữ (DB, FAISS, MinIO/S3)
│   ├── search/               # Engine tìm kiếm và tính toán độ tương đồng
│   └── ingestion/            # Thu thập và tiền xử lý ảnh
├── scripts/                  # Script thực thi CLI (Ingest, Build Index)
├── app.py                    # Giao diện Streamlit chính
└── requirements.txt          # Danh sách thư viện phụ thuộc
```

## 🚀 Hướng dẫn cài đặt

### 1. Cài đặt môi trường
```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Thiết lập MinIO (Storage)
1. Chạy MinIO bằng Docker:
   ```bash
   docker run -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"
   ```
2. Truy cập `http://localhost:9001` (User/Pass: `minioadmin`), tạo bucket tên: `tree-images`.
3. Cấu hình thông tin trong `configs/default.yaml` (phần `minio`).

### 3. Khởi tạo dữ liệu
Đặt các ảnh gốc vào thư mục `data/raw/` (định dạng `.jpg` hoặc `.png`). Sau đó chạy:

```bash
# Bước 1: Tiền xử lý, resize và upload lên MinIO
python -m scripts.ingest_images

# Bước 2: Trích xuất đặc trưng và xây dựng FAISS Index
python scripts/build_index.py
```

### 4. Chạy ứng dụng
```bash
streamlit run app.py
```

## 📊 Bộ đặc trưng (27 chiều)
1. **Geometry (5)**: Tỷ lệ cao/rộng, Mật độ tán, Đường kính thân, Độ đối xứng, Phương sai góc cành.
2. **Color (9)**: Mean & Std của các kênh HSV, Độ rực màu (Saturation).
3. **Texture (4)**: Độ thô ráp (LBP), Độ phức tạp biên, Mật độ cạnh, Entropy.
4. **Shape (9)**: 7 Momen Hu bất biến, Độ đặc (Solidity), Độ tròn (Circularity).

## 📄 Giấy phép
Dự án được phát triển phục vụ mục đích học tập môn Hệ CSDL Đa phương tiện.
