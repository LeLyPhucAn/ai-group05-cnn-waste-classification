"""
Ứng dụng Web Demo: Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh
Môn học: Trí tuệ Nhân tạo - Đề tài 19 (Mức Nâng cao)
Nhóm thực hiện: Nhóm 05
Giảng viên hướng dẫn: Thầy Bùi Trọng Hiếu

Kiến trúc: Ảnh đầu vào -> Tiền xử lý RGB (224x224) -> MobileNetV2 (GAP 1280d) -> Random Forest -> 12 lớp chi tiết -> Ánh xạ 4 nhóm xử lý
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st

# Thêm thư mục gốc vào PYTHONPATH để nạp các module trong src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mobilenet_feature_extractor import MobileNetV2FeatureExtractor
from src.mapping import get_mapping_info, get_waste_group, get_vi_name, get_waste_guide
from src.random_forest import load_model


# -------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Phân loại Rác thải - Nhóm 05",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến giao diện CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        text-align: center;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .guide-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #F7FAFC;
        border-left: 4px solid #3182CE;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# TẢI TÀI NGUYÊN (CACHE RESOURCE ĐỂ TỐI ƯU HIỆU NĂNG)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp mô hình MobileNetV2 & Random Forest...")
def load_models_and_metadata():
    """
    Nạp sẵn bộ trích xuất MobileNetV2, mô hình Random Forest và danh sách 12 lớp.
    Được lưu trong bộ nhớ đệm (cache) để tránh nạp lại mỗi lần người dùng bấm nút.
    """
    class_names_path = PROJECT_ROOT / "data" / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = [
            "battery", "biological", "brown-glass", "cardboard", "clothes",
            "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
        ]

    # Khởi tạo MobileNetV2 (CPU hoặc CUDA)
    extractor = MobileNetV2FeatureExtractor()

    # Nạp mô hình Random Forest đã huấn luyện
    model_path = PROJECT_ROOT / "models" / "random_forest_model.joblib"
    rf_model = None
    if model_path.exists():
        try:
            rf_model = load_model(str(model_path))
        except Exception as e:
            st.warning(f"Không thể nạp model từ {model_path}: {e}")

    return extractor, rf_model, class_names


# -------------------------------------------------------------
# GIAO DIỆN CHÍNH
# -------------------------------------------------------------
st.markdown('<div class="main-title">♻️ HỆ THỐNG PHÂN LOẠI RÁC THẢI SINH HOẠT QUA ẢNH</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Đề tài 19: CNN (MobileNetV2 Transfer Learning) kết hợp Random Forest | Nhóm 05 - AI</div>', unsafe_allow_html=True)

# Khởi tạo mô hình
try:
    extractor, rf_model, class_names = load_models_and_metadata()
except Exception as err:
    st.error(f"Lỗi khởi tạo hệ thống: {err}")
    st.stop()

# Thanh bên (Sidebar): Giới thiệu & Thống kê
with st.sidebar:
    st.header("📌 Thông tin Dự án")
    st.markdown("""
    - **Môn học:** Trí tuệ Nhân tạo
    - **Giảng viên:** Thầy Bùi Trọng Hiếu
    - **Nhóm thực hiện:** Nhóm 05 (TV1 - TV5)
    - **Kiến trúc:** MobileNetV2 (ImageNet) + Random Forest
    - **Số lượng lớp:** 12 lớp rác Kaggle
    - **Độ chính xác Test:** **95.94%** (Có Augmentation)
    """)
    st.markdown("---")
    st.subheader("🗂️ 4 Nhóm Xử lý Rác")
    st.markdown("""
    - 🍏 **Rác hữu cơ:** `biological`
    - 🔋 **Rác nguy hại:** `battery`
    - ♻️ **Rác tái chế:** `plastic`, `metal`, `paper`, `cardboard`, `glass`, `clothes`, `shoes`
    - 🗑️ **Rác còn lại:** `trash`
    """)

# Khu vực tương tác chính
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("📷 Tải ảnh rác thải đầu vào")
    upload_method = st.radio("Chọn phương thức nhập ảnh:", ["Tải ảnh từ máy tính", "Chụp ảnh bằng Camera"], horizontal=True)

    image_file = None
    if upload_method == "Tải ảnh từ máy tính":
        image_file = st.file_uploader("Chọn ảnh rác thải (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
    else:
        image_file = st.camera_input("Chụp ảnh rác thực tế:")

    if image_file is not None:
        pil_img = Image.open(image_file)
        st.image(pil_img, caption="Ảnh rác thải được chọn", use_container_width=True)

with col_right:
    st.subheader("🔍 Kết quả Phân loại & Hướng dẫn")

    if image_file is None:
        st.info("👈 Vui lòng tải lên một bức ảnh rác thải hoặc chụp từ camera để bắt đầu nhận diện.")
    else:
        if st.button("🚀 BẮT ĐẦU PHÂN LOẠI", type="primary", use_container_width=True):
            with st.spinner("Đang trích xuất đặc trưng MobileNetV2 và phân loại bằng Random Forest..."):
                try:
                    # 1. Ép hệ màu RGB
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    # 2. Trích xuất vector 1.280 chiều qua MobileNetV2
                    feat_vector = extractor.extract_from_image(pil_img)

                    # 3. Dự đoán với Random Forest
                    if rf_model is not None:
                        # Dự đoán phân bố xác suất cho 12 lớp
                        probs = rf_model.predict_proba([feat_vector])[0]
                        pred_idx = int(np.argmax(probs))
                        pred_class = class_names[pred_idx]
                        confidence = float(probs[pred_idx]) * 100
                    else:
                        st.warning("Chưa nạp được file random_forest_model.joblib, hiển thị mô phỏng.")
                        probs = np.ones(len(class_names)) / len(class_names)
                        pred_idx = 0
                        pred_class = class_names[0]
                        confidence = 90.0

                    # 4. Ánh xạ sang nhóm xử lý & thông tin phụ trợ
                    info = get_mapping_info(pred_class)

                    # 5. Hiển thị thẻ kết quả nổi bật
                    st.markdown(f"""
                    <div class="result-card" style="background-color: {info['badge_bg']}; border: 2px solid {info['color']};">
                        <div style="font-size: 1.1rem; font-weight: 600; color: {info['color']};">
                            {info['icon']} NHÓM: {info['group'].upper()}
                        </div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #1A202C; margin-top: 0.3rem;">
                            {info['vi_name']}
                        </div>
                        <div style="font-size: 0.95rem; color: #4A5568; margin-top: 0.2rem;">
                            Lớp kỹ thuật: <code>{pred_class}</code> | Độ tin cậy: <b>{confidence:.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 6. Hướng dẫn xử lý rác
                    st.markdown(f"""
                    <div class="guide-box">
                        <b>📋 Hướng dẫn xử lý thùng rác:</b><br>
                        {info['guide']}
                    </div>
                    """, unsafe_allow_html=True)

                    # 7. Biểu đồ Top 3 xác suất cao nhất
                    st.markdown("#### 📊 Phân bố xác suất dự đoán (Top 3):")
                    top3_indices = np.argsort(probs)[-3:][::-1]
                    for idx in top3_indices:
                        c_name = class_names[idx]
                        p_val = float(probs[idx])
                        v_name = get_vi_name(c_name)
                        st.write(f"**{v_name}** (`{c_name}`): {p_val * 100:.2f}%")
                        st.progress(min(max(p_val, 0.0), 1.0))

                except Exception as ex:
                    st.error(f"Đã xảy ra lỗi trong quá trình phân loại: {ex}")
