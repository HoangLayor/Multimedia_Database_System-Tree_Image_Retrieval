import streamlit as st
import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image as PILImage
import time

from src.search.engine import SearchEngine
from src.features.extractor import extract, FEATURE_NAMES
from src.storage.db import get_session
from src.storage.models import Image as DBImage

# Cấu hình trang
st.set_page_config(
    page_title="Tree Image Search - Hệ thống tìm kiếm cây thông minh",
    page_icon="🌳",
    layout="wide"
)

# Load cấu hình
@st.cache_resource
def load_engine():
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        # Fallback nếu không có file config
        config = {
            "data": {
                "features_dir": "data/features"
            },
            "storage": {
                "faiss_index_path": "data/features/faiss.index",
                "image_ids_path": "data/features/image_ids.npy"
            }
        }
    return SearchEngine(config)

engine = load_engine()

# --- Giao diện Sidebar ---
st.sidebar.title("Cấu hình Tìm kiếm")

# Kiểm tra trạng thái hệ thống
with st.sidebar.expander("Trạng thái Hệ thống", expanded=True):
    if engine.vector_store.load():
        st.success("✅ CSDL Vector: Sẵn sàng")
    else:
        st.warning("⚠️ CSDL Vector: Chưa tạo index (data/features/faiss.index)")
        
    if engine.normalizer.mean_ is not None:
        st.success("✅ Scaler: Sẵn sàng")
    else:
        st.warning("⚠️ Scaler: Thiếu file scaler_mean.npy (Sẽ bỏ qua chuẩn hóa)")

top_k = st.sidebar.slider("Số lượng kết quả (Top-K)", 1, 20, 5)
show_intermediate = st.sidebar.checkbox("Hiển thị kết quả trung gian", value=True)

# --- Giao diện Chính ---
st.title("🌳 Tree Image Search System")
st.markdown("### Hệ thống tìm kiếm và đánh giá đặc trưng cây dựa trên nội dung (CBIR)")

uploaded_file = st.sidebar.file_uploader("Tải ảnh cây cần tìm kiếm...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize về 512x512 theo spec
    img_bgr_resized = cv2.resize(img_bgr, (512, 512))
    img_rgb_resized = cv2.resize(img_rgb, (512, 512))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Query")
        st.image(img_rgb_resized, use_container_width=True)
        
    if show_intermediate:
        with col2:
            st.subheader("Bước 1: Tiền xử lý (Binarized Mask)")
            from src.features.utils import get_tree_mask
            mask = get_tree_mask(img_bgr_resized)
            st.image(mask, use_container_width=True, caption="Mask tự động tách cây (Smart Detection)")
            
    # Lưu ảnh query tạm thời để engine.py có thể đọc (vì engine.py nhận path)
    temp_path = Path("temp_query.jpg")
    cv2.imwrite(str(temp_path), img_bgr_resized)
    
    # --- Bước trung gian 2: Trích chọn đặc trưng ---
    st.divider()
    if show_intermediate:
        st.subheader("Bước 2: Trích chọn đặc trưng (Feature Extraction)")
        with st.spinner("Đang trích xuất 27 đặc trưng..."):
            start_time = time.time()
            features = extract(temp_path)
            extract_time = (time.time() - start_time) * 1000
            
            # Hiển thị vector đặc trưng theo nhóm
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.write("**Geometry (5)**")
                st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(5)})
            with c2:
                st.write("**Color (9)**")
                st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(5, 14)})
            with c3:
                st.write("**Texture (4)**")
                st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(14, 18)})
            with c4:
                st.write("**Shape (9)**")
                st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(18, 27)})
            
            st.info(f"Thời gian trích xuất: {extract_time:.2f} ms")

    # --- Bước 3: Tìm kiếm và hiển thị kết quả ---
    st.divider()
    st.subheader(f"Kết quả Tìm kiếm (Top {top_k})")
    
    with st.spinner("Đang tìm kiếm trong CSDL..."):
        start_time = time.time()
        results = engine.search(temp_path, top_k=top_k)
        search_time = (time.time() - start_time) * 1000
        
    st.success(f"Tìm thấy {len(results)} kết quả trong {search_time:.2f} ms")
    
    # Hiển thị kết quả dạng lưới (grid)
    if results:
        cols = st.columns(3)
        for i, res in enumerate(results):
            with cols[i % 3]:
                # Load ảnh kết quả
                res_path = Path(res["file_path"])
                if res_path.exists():
                    res_img = PILImage.open(res_path)
                    st.image(res_img, use_container_width=True, caption=f"Rank {res['rank']} - Similarity: {res['similarity']:.4f}")
                else:
                    st.error(f"Không tìm thấy file: {res_path.name}")
                
                with st.expander("Chi tiết Metadata"):
                    st.write(f"**ID:** {res['image_id']}")
                    st.write(f"**Loài:** {res['species']}")
                    st.write(f"**Tên phổ thông:** {res['common_name']}")
                    st.write(f"**Độ tuổi:** {res['age_class']}")
    else:
        st.warning("Không tìm thấy kết quả nào phù hợp.")

else:
    st.info("Vui lòng tải lên một ảnh cây ở thanh bên trái để bắt đầu tìm kiếm.")

# Dọn dẹp file tạm khi thoát (tùy chọn)
if Path("temp_query.jpg").exists():
    pass # Để lại để debug nếu cần
