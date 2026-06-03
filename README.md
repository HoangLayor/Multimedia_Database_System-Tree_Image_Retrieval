<div align="center">
  <img src="https://socialify.git.ci/HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval/image?description=1&font=Inter&language=1&name=1&owner=1&pattern=Solid&theme=Auto" alt="Tree Image Search System" width="640" height="320" />
  <br/>
  <h1>🌳 Tree Image Search System</h1>
  <p>
    <strong>Hệ thống Cơ sở dữ liệu Đa phương tiện lưu trữ và tìm kiếm ảnh cây thông minh</strong>
  </p>
  
  <p>
    <a href="https://github.com/HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval/stargazers"><img src="https://img.shields.io/github/stars/HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval?style=flat-square&color=yellow" alt="Stars" /></a>
    <a href="https://github.com/HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval/network/members"><img src="https://img.shields.io/github/forks/HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval?style=flat-square&color=blue" alt="Forks" /></a>
    <a href="https://github.com/HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval/issues"><img src="https://img.shields.io/github/issues/HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval?style=flat-square&color=red" alt="Issues" /></a>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat-square&logo=Streamlit&logoColor=white" alt="Streamlit" /></a>
  </p>
</div>

<hr />

## 📖 Giới thiệu

Hệ thống cho phép tìm kiếm ảnh cây dựa trên nội dung (**CBIR** - Content-Based Image Retrieval) bằng cách sử dụng bộ đặc trưng **27 chiều** (Geometry, Color, Texture, Shape) kết hợp với **FAISS Index** để truy xuất tốc độ cao.

Dự án được phát triển nhằm mục đích cung cấp một giải pháp toàn diện từ khâu tiền xử lý, trích xuất đặc trưng, lưu trữ phân tán cho đến tìm kiếm tương đồng.

---

## ✨ Tính năng nổi bật

- 🔍 **Trích xuất 27 đặc trưng đa chiều**: Phân tích chuyên sâu về hình dáng, màu sắc và kết cấu của cây.
- ✂️ **Tiền xử lý thông minh**: Tự động phân tách cây khỏi nền bằng thuật toán **Otsu Adaptive**.
- ⚡ **Tìm kiếm siêu tốc**: Tích hợp **FAISS** (Facebook AI Similarity Search) hỗ trợ truy xuất trên tập dữ liệu lớn.
- 🧮 **Đa dạng độ đo tương đồng**: Hỗ trợ tính toán bằng *Cosine*, *Euclidean* và *Weighted Metric*.
- 💾 **Lưu trữ Hybrid hiện đại**:
  - **Metadata**: SQLite (SQLAlchemy)
  - **Feature Index**: FAISS
  - **Image Storage**: MinIO (S3 Compatible)
- 🎨 **Giao diện trực quan**: Ứng dụng Web xây dựng bằng **Streamlit**, cho phép trực quan hóa toàn bộ quá trình trích xuất đặc trưng.

---

## 🛠 Công nghệ sử dụng

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/MinIO-C7202C?style=for-the-badge&logo=minio&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

---

## 📁 Cấu trúc thư mục

```text
tree-image-search/
├── configs/                  # Cấu hình hệ thống (YAML)
├── data/                     # Dữ liệu local (raw, processed, features)
├── src/                      # Mã nguồn chính
│   ├── features/             # Module trích xuất đặc trưng (27 chiều)
│   ├── storage/              # Module lưu trữ (DB, FAISS, MinIO/S3)
│   ├── search/               # Engine tìm kiếm & tính toán tương đồng
│   └── ingestion/            # Pipeline thu thập & tiền xử lý ảnh
├── scripts/                  # CLI Scripts (Ingest, Build Index)
├── app.py                    # Giao diện Streamlit chính
└── requirements.txt          # Danh sách thư viện phụ thuộc
```

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Cài đặt môi trường

Khuyến nghị sử dụng `venv` hoặc `conda` để quản lý môi trường:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate
# Kích hoạt môi trường (Linux/macOS)
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Thiết lập Storage (MinIO)

1. Khởi chạy MinIO container thông qua Docker:
   ```bash
   docker run -d -p 9000:9000 -p 9001:9001 --name minio \
     -e "MINIO_ROOT_USER=minioadmin" \
     -e "MINIO_ROOT_PASSWORD=minioadmin" \
     quay.io/minio/minio server /data --console-address ":9001"
   ```
2. Truy cập MinIO Console tại `http://localhost:9001` (Tài khoản/Mật khẩu: `minioadmin`).
3. Tạo một bucket mới với tên: `tree-images`.
4. Cập nhật thông tin kết nối trong file `configs/default.yaml` (tại mục `minio`).

### 3. Khởi tạo dữ liệu

Chuẩn bị tập dữ liệu ảnh gốc (định dạng `.jpg` hoặc `.png`) và đặt vào thư mục `data/raw/`. Sau đó thực thi các script sau:

```bash
# Bước 1: Tiền xử lý, chuẩn hóa kích thước và upload lên MinIO
python -m scripts.ingest_images

# Bước 2: Trích xuất đặc trưng 27 chiều và xây dựng FAISS Index
python scripts/build_index.py
```

### 4. Khởi chạy Giao diện Web

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở ứng dụng tại địa chỉ `http://localhost:8501`.

---

## 📊 Chi tiết Bộ đặc trưng (27 chiều)

Hệ thống trích xuất tổng cộng 27 đặc trưng phân làm 4 nhóm chính:

| Nhóm Đặc Trưng | Số lượng | Chi tiết |
|:---|:---:|:---|
| 📐 **Geometry** | 5 | Tỷ lệ cao/rộng, Mật độ tán, Đường kính thân, Độ đối xứng, Phương sai góc cành |
| 🎨 **Color** | 9 | Giá trị Mean & Std của các kênh HSV, Độ rực màu (Saturation) |
| 🧶 **Texture** | 4 | Độ thô ráp (LBP), Độ phức tạp biên, Mật độ cạnh, Entropy |
| 🟢 **Shape** | 9 | 7 Momen Hu bất biến, Độ đặc (Solidity), Độ tròn (Circularity) |

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval&type=Date)](https://star-history.com/#HoangLayor/Multimedia_Database_System-Tree_Image_Retrieval&Date)

---

## 📄 Giấy phép & Tác giả

Dự án được phát triển phục vụ mục đích học tập và nghiên cứu môn học **Hệ Cơ sở dữ liệu Đa phương tiện**. 

Nếu bạn thấy dự án hữu ích, hãy để lại một ⭐️ để ủng hộ tác giả nhé!
