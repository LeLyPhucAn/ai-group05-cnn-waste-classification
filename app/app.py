"""
Ứng dụng Web Demo: Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh
Môn học: Trí tuệ Nhân tạo - Đề tài 19 (Mức Nâng cao)
Nhóm thực hiện: Nhóm 05
Giảng viên hướng dẫn: Thầy Bùi Trọng Hiếu

Kiến trúc:
  Ảnh đầu vào -> Ép RGB & Tiền xử lý (src/preprocessing.py) -> MobileNetV2 (GAP 1280d)
  -> Random Forest predict_proba() -> 12 lớp chi tiết -> Ánh xạ 4 nhóm xử lý (src/mapping.py)
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Thêm thư mục gốc vào PYTHONPATH để nạp các module trong src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mapping import get_mapping_info, get_vi_name
from src.mobilenet_feature_extractor import MobileNetV2FeatureExtractor
from src.preprocessing import load_and_preprocess_image, load_class_names, to_rgb
from src.random_forest import load_model


# -------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Phân loại Rác thải - Nhóm 05",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    Nạp sẵn bộ trích xuất MobileNetV2, mô hình Random Forest và danh sách 12 lớp từ data/class_names.json.
    """
    class_names_path = PROJECT_ROOT / "data" / "class_names.json"
    class_names = load_class_names(class_names_path)

    extractor = MobileNetV2FeatureExtractor()

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
st.markdown(
    '<div class="sub-title">Đề tài 19: CNN (MobileNetV2 Transfer Learning) kết hợp Random Forest | Nhóm 05 - AI</div>',
    unsafe_allow_html=True,
)

try:
    extractor, rf_model, class_names = load_models_and_metadata()
except Exception as err:
    st.error(f"Lỗi khởi tạo hệ thống: {err}")
    st.stop()

# Thanh bên (Sidebar)
with st.sidebar:
    st.header("📌 Thông tin Dự án")
    st.markdown("""
    - **Môn học:** Trí tuệ Nhân tạo
    - **Giảng viên:** Thầy Bùi Trọng Hiếu
    - **Nhóm thực hiện:** Nhóm 05 (TV1 - TV5)
    - **Kiến trúc:** MobileNetV2 (ImageNet) + Random Forest
    - **Số lượng lớp:** 12 lớp rác Kaggle
    """)
    st.markdown("---")
    st.subheader("🗂️ 4 Nhóm Xử lý Rác")
    st.markdown("""
    - 🍏 **Rác hữu cơ:** `biological`
    - 🔋 **Rác nguy hại:** `battery`
    - ♻️ **Rác tái chế:** `plastic`, `metal`, `paper`, `cardboard`, `white-glass`, `green-glass`, `brown-glass`, `clothes`, `shoes`
    - 🗑️ **Rác còn lại:** `trash`
    """)

col_left, col_right = st.columns([1, 1], gap="medium")

pil_img = None
with col_left:
    st.subheader("📷 Tải ảnh rác thải đầu vào")
    upload_method = st.radio(
        "Chọn phương thức nhập ảnh:",
        ["Tải ảnh từ máy tính", "Chụp ảnh bằng Camera"],
        horizontal=True,
    )

    image_file = (
        st.file_uploader("Chọn ảnh rác thải (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
        if upload_method == "Tải ảnh từ máy tính"
        else st.camera_input("Chụp ảnh rác thực tế:")
    )

    if image_file is not None:
        try:
            pil_img = to_rgb(Image.open(image_file))
            st.image(pil_img, caption="Ảnh rác thải được chọn", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("Tệp tải lên bị hỏng hoặc không phải định dạng ảnh hợp lệ. Vui lòng chọn ảnh khác.")
            pil_img = None

with col_right:
    st.subheader("🔍 Kết quả Phân loại & Hướng dẫn")

    if pil_img is None:
        st.info("👈 Vui lòng tải lên một bức ảnh rác thải hoặc chụp từ camera để bắt đầu nhận diện.")
    else:
        if st.button("🚀 BẮT ĐẦU PHÂN LOẠI", type="primary", use_container_width=True):
            with st.spinner("Đang tiền xử lý, trích xuất đặc trưng MobileNetV2 và phân loại bằng Random Forest..."):
                try:
                    # 1. Tiền xử lý chuẩn từ src/preprocessing.py (TV2) -> Tensor (1, 3, 224, 224)
                    tensor = load_and_preprocess_image(pil_img)

                    # 2. Trích xuất vector 1.280 chiều qua MobileNetV2 (TV1)
                    feat_vector = extractor.extract_from_batch(tensor)[0]

                    # 3. Dự đoán phân bố xác suất 12 lớp với Random Forest (TV3)
                    if rf_model is not None:
                        probs = rf_model.predict_proba([feat_vector])[0]
                        pred_idx = int(np.argmax(probs))
                        pred_class = class_names[pred_idx]
                        confidence = float(probs[pred_idx]) * 100
                    else:
                        st.warning("Chưa tìm thấy file models/random_forest_model.joblib.")
                        st.stop()

                    # 4. Ánh xạ sang 4 nhóm xử lý (src/mapping.py - TV4)
                    info = get_mapping_info(pred_class)

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

                    st.markdown(f"""
                    <div class="guide-box">
                        <b>📋 Hướng dẫn xử lý thùng rác:</b><br>
                        {info['guide']}
                    </div>
                    """, unsafe_allow_html=True)

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
