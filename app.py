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
from src.storage.s3 import MinioStorage
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

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
storage = MinioStorage(engine.config)

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
search_method = st.sidebar.selectbox(
    "Phương pháp tìm kiếm",
    ["FAISS (Vector Index)", "Database (Sequential Scan)", "So sánh hiệu năng (Benchmark)"],
    index=2
)
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
            
            # Sử dụng Tabs để phân tách nội dung
            tab1, tab2 = st.tabs(["🔢 Chỉ số đặc trưng", "📊 Phân tích Trực quan"])
            
            with tab1:
                # Hiển thị vector đặc trưng theo nhóm
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Geometry & Texture**")
                    st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(5)})
                    st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(14, 18)})
                with c2:
                    st.write("**Color & Shape**")
                    st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(5, 14)})
                    st.json({FEATURE_NAMES[i]: round(float(features[i]), 4) for i in range(18, 27)})
                st.info(f"Thời gian trích xuất: {extract_time:.2f} ms")

            with tab2:
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.write("**Hồ sơ Hình thái (Shape Profile)**")
                    radar_labels = ['Mật độ tán', 'Độ đối xứng', 'Độ đặc (Solidity)', 'Độ tròn']
                    radar_values = [features[1], features[3], features[25], features[26]]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_values, theta=radar_labels, fill='toself', line_color='#2E7D32'
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False, margin=dict(l=40, r=40, t=40, b=40), height=350
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with v_col2:
                    st.write("**Màu sắc chủ đạo (Dominant Colors)**")
                    tree_pixels = img_rgb_resized[mask > 0]
                    if len(tree_pixels) > 100:
                        sample_size = min(len(tree_pixels), 5000)
                        sample_pixels = tree_pixels[np.random.choice(len(tree_pixels), sample_size, replace=False)]
                        kmeans = KMeans(n_clusters=5, n_init=10)
                        kmeans.fit(sample_pixels)
                        colors = kmeans.cluster_centers_.astype(int)
                        percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
                        hex_colors = [f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in colors]
                        
                        fig_pie = px.pie(values=percentages, names=hex_colors, color=hex_colors,
                                       color_discrete_map={h: h for h in hex_colors})
                        fig_pie.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=350)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("Không đủ dữ liệu phân tích màu.")

                # Trực quan hóa Phân đoạn (Segmentation)
                st.divider()
                st.write("**Trực quan hóa Phân đoạn (Segmentation)**")
                vis_img = img_rgb_resized.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                st.image(vis_img, use_container_width=True)

    # --- Bước 3: Tìm kiếm và hiển thị kết quả ---
    st.divider()
    st.subheader(f"Kết quả Tìm kiếm (Top {top_k})")
    
    # Logic tìm kiếm dựa trên setting
    with st.spinner(f"Đang tìm kiếm bằng {search_method}..."):
        if search_method == "FAISS (Vector Index)":
            start_time = time.time()
            results_faiss = engine.search(temp_path, top_k=top_k)
            exec_time = (time.time() - start_time) * 1000
            st.success(f"Tìm thấy kết quả bằng FAISS trong {exec_time:.2f} ms")
            
        elif search_method == "Database (Sequential Scan)":
            start_time = time.time()
            results_faiss = engine.search_db_sequential(temp_path, top_k=top_k)
            exec_time = (time.time() - start_time) * 1000
            st.success(f"Tìm thấy kết quả bằng DB Scan trong {exec_time:.2f} ms")
            
        else: # Benchmark mode
            start_faiss = time.time()
            results_faiss = engine.search(temp_path, top_k=top_k)
            time_faiss = (time.time() - start_faiss) * 1000
            
            start_db = time.time()
            results_db = engine.search_db_sequential(temp_path, top_k=top_k)
            time_db = (time.time() - start_db) * 1000
            
            # Hiển thị Benchmark
            with st.expander("⏱️ So sánh hiệu năng (Performance Benchmark)", expanded=True):
                b_col1, b_col2 = st.columns([1, 2])
                with b_col1:
                    st.metric("FAISS (Index)", f"{time_faiss:.2f} ms", delta="Nhanh nhất", delta_color="normal")
                    st.metric("DB (Sequential)", f"{time_db:.2f} ms", delta=f"{time_db/time_faiss:.1f}x chậm hơn", delta_color="inverse")
                with b_col2:
                    bench_data = {"Phương pháp": ["FAISS (Index)", "DB (Sequential)"], "Thời gian (ms)": [time_faiss, time_db]}
                    st.bar_chart(bench_data, x="Phương pháp", y="Thời gian (ms)", color=["#2E7D32", "#FF4B4B"])
            st.success(f"Kết quả FAISS ({time_faiss:.2f} ms) vs DB Scan ({time_db:.2f} ms)")
    
    # Hiển thị kết quả dạng lưới (grid)
    if results_faiss:
        cols = st.columns(3)
        for i, res in enumerate(results_faiss):
            with cols[i % 3]:
                res_path = res["file_path"]
                # Nếu res_path không phải là đường dẫn tuyệt đối (không có dấu \ hoặc / của Windows/Linux) 
                # thì đó là S3 Key
                if not ("\\" in res_path or "/" in res_path):
                    img_url = storage.get_url(res_path)
                    if img_url:
                        st.image(img_url, use_container_width=True, caption=f"Rank {res['rank']} - Similarity: {res['similarity']:.4f}")
                    else:
                        st.error("Lỗi lấy ảnh từ MinIO")
                else:
                    # Fallback dùng local file
                    res_path_obj = Path(res_path)
                    if res_path_obj.exists():
                        res_img = PILImage.open(res_path_obj)
                        st.image(res_img, use_container_width=True, caption=f"Rank {res['rank']} - Similarity: {res['similarity']:.4f}")
                    else:
                        st.error(f"Không tìm thấy file: {res_path_obj.name}")
                
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
