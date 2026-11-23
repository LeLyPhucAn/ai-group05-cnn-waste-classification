"""
Ứng dụng Web Demo: Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh
Môn học: Trí tuệ Nhân tạo - Đề tài 19 (Mức Nâng cao)
Nhóm thực hiện: Nhóm 05 | Giảng viên hướng dẫn: Thầy Bùi Trọng Hiếu

Kiến trúc:
  Ảnh đầu vào -> Ép RGB & Tiền xử lý (src/preprocessing.py) -> MobileNetV2 (GAP 1.280d)
  -> Random Forest predict_proba() -> 12 lớp chi tiết -> Ánh xạ 4 nhóm xử lý (src/mapping.py)

Chạy ứng dụng:
  streamlit run app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mapping import get_mapping_info, get_vi_name
from src.mobilenet_feature_extractor import MobileNetV2FeatureExtractor
from src.preprocessing import load_and_preprocess_image, load_class_names, to_rgb
from src.random_forest import load_model


# -------------------------------------------------------------
# CẤU HÌNH TRANG & GIAO DIỆN CSS (Phong cách xanh môi trường hiện đại)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Phân loại Rác thải AI - Nhóm 05",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #43b04a 0%, #2196f3 100%);
        padding: 1.8rem 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(33, 150, 243, 0.22);
    }
    .hero-banner h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        color: white;
    }
    .hero-banner p {
        font-size: 1.02rem;
        margin: 0;
        opacity: 0.96;
    }
    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        margin-top: 0.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .guide-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background-color: #f8fafc;
        border-left: 5px solid #2196f3;
        margin-top: 1rem;
        color: #1e293b;
    }
    .group-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.4rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #43b04a, #2e7d32);
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# TẢI TÀI NGUYÊN (CACHE RESOURCE)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp mô hình MobileNetV2 & Random Forest...")
def load_models_and_metadata():
    class_names = load_class_names(PROJECT_ROOT / "data" / "class_names.json")
    extractor = MobileNetV2FeatureExtractor()

    model_path = PROJECT_ROOT / "models" / "random_forest_model.joblib"
    rf_model = load_model(str(model_path)) if model_path.exists() else None
    return extractor, rf_model, class_names


def load_eval_summary():
    csv_path = PROJECT_ROOT / "results" / "metrics" / "evaluation_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def list_demo_images():
    demo_root = PROJECT_ROOT / "demo_samples"
    if not demo_root.is_dir():
        return {}
    samples = {}
    for cls_dir in sorted(demo_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for p in imgs[:3]:  # lấy tối đa 3 ảnh đại diện mỗi nhóm để chọn nhanh
            label = f"{get_vi_name(cls_dir.name)} — {p.name}"
            samples[label] = p
    return samples


def predict_waste(pil_img: Image.Image, extractor, rf_model, class_names):
    tensor = load_and_preprocess_image(pil_img)
    feat_vector = extractor.extract_from_batch(tensor)[0]
    probs = rf_model.predict_proba([feat_vector])[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    return pred_class, confidence, probs


def render_prediction_result(pred_class: str, confidence: float, probs: np.ndarray, class_names):
    info = get_mapping_info(pred_class)
    st.markdown(f"""
    <div class="result-card" style="background-color: {info['badge_bg']}; border: 2px solid {info['color']};">
        <div class="group-pill" style="background-color: {info['color']}; color: white;">
            {info['icon']} {info['group'].upper()}
        </div>
        <div style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-top: 0.25rem;">
            {info['vi_name']}
        </div>
        <div style="font-size: 0.98rem; color: #334155; margin-top: 0.25rem;">
            Lớp kỹ thuật (12 lớp): <code>{pred_class}</code> &nbsp;|&nbsp;
            Độ tin cậy: <b>{confidence:.2f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if confidence < 55.0:
        st.warning(
            "Độ tin cậy thấp (< 55%): Ảnh có thể bị mờ, thiếu sáng, nền phức tạp "
            "hoặc vật thể dễ nhầm giữa các nhóm nhựa / kim loại / thủy tinh."
        )

    st.markdown(f"""
    <div class="guide-box">
        <b>📋 Hướng dẫn phân loại & bỏ thùng rác:</b><br>
        {info['guide']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Phân bố xác suất dự đoán (Top 5 lớp gần nhất)")
    top_indices = np.argsort(probs)[-5:][::-1]
    for idx in top_indices:
        c_name = class_names[idx]
        p_val = float(probs[idx])
        v_name = get_vi_name(c_name)
        st.write(f"**{v_name}** (`{c_name}`): `{p_val * 100:.2f}%`")
        st.progress(min(max(p_val, 0.0), 1.0))


# -------------------------------------------------------------
# HEADER & SIDEBAR
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>♻️ HỆ THỐNG PHÂN LOẠI RÁC THẢI THÔNG MINH</h1>
    <p>Đề tài 19 (Nâng cao): MobileNetV2 (Transfer Learning) kết hợp Random Forest | 12 lớp chi tiết &rarr; 4 nhóm xử lý</p>
</div>
""", unsafe_allow_html=True)

try:
    extractor, rf_model, class_names = load_models_and_metadata()
except Exception as err:
    st.error(f"Lỗi khởi tạo hệ thống: {err}")
    st.stop()

eval_df = load_eval_summary()

with st.sidebar:
    st.header("📌 Thông tin Đề tài")
    st.markdown("""
    - **Môn học:** Trí tuệ Nhân tạo
    - **Giảng viên:** Thầy Bùi Trọng Hiếu
    - **Nhóm thực hiện:** Nhóm 05
    - **Kiến trúc:** MobileNetV2 (1.280d) + Random Forest
    - **Dữ liệu:** Kaggle Garbage Classification (12 lớp – 15.496 ảnh sạch)
    """)
    if eval_df is not None and len(eval_df) >= 2:
        aug_row = eval_df.iloc[-1]
        st.success(
            f"**Test Accuracy (Có Augment):** {aug_row['Test Accuracy (%)']}%\n\n"
            f"**Macro F1-Score:** {aug_row['Macro F1-Score (%)']}%"
        )
    st.markdown("---")
    st.subheader("🗂️ Ánh xạ 4 Nhóm Xử lý")
    st.markdown("""
    - 🍏 **Rác hữu cơ:** `biological`
    - 🔋 **Rác nguy hại:** `battery`
    - ♻️ **Rác tái chế:** `plastic`, `metal`, `paper`, `cardboard`, `white-glass`, `green-glass`, `brown-glass`, `clothes`, `shoes`
    - 🗑️ **Rác còn lại:** `trash`
    """)

if rf_model is None:
    st.error("Chưa tìm thấy `models/random_forest_model.joblib`. Hãy chạy `python src/tune_and_evaluate.py` trước.")
    st.stop()


# -------------------------------------------------------------
# 3 TAB CHỨC NĂNG CHÍNH (Lấy cảm hứng từ DO_AN_NHOM11 + mở rộng thực nghiệm)
# -------------------------------------------------------------
tab_upload, tab_camera, tab_eval = st.tabs([
    "📁 Nhận diện từ Ảnh tải lên",
    "📷 Chụp ảnh bằng Camera",
    "📊 Kết quả Thực nghiệm & Ma trận Nhầm lẫn",
])

with tab_upload:
    col_left, col_right = st.columns([1, 1.15], gap="large")
    pil_img = None

    with col_left:
        st.subheader("1. Chọn ảnh rác thải")
        src_mode = st.radio(
            "Nguồn ảnh:",
            ["Tải ảnh từ máy tính", "Chọn nhanh ảnh mẫu (demo_samples)"],
            horizontal=True,
        )

        if src_mode == "Tải ảnh từ máy tính":
            uploaded = st.file_uploader(
                "Kéo thả hoặc chọn ảnh (JPG, PNG, JPEG):",
                type=["jpg", "png", "jpeg"],
            )
            if uploaded is not None:
                try:
                    pil_img = to_rgb(Image.open(uploaded))
                except (UnidentifiedImageError, OSError):
                    st.error("Tệp ảnh bị lỗi hoặc không đúng định dạng. Vui lòng chọn ảnh khác.")
        else:
            demo_map = list_demo_images()
            if demo_map:
                chosen = st.selectbox("Chọn ảnh mẫu để kiểm thử nhanh:", list(demo_map.keys()))
                pil_img = to_rgb(Image.open(demo_map[chosen]))
            else:
                st.info("Thư mục `demo_samples/` hiện chưa có ảnh mẫu cục bộ.")

        if pil_img is not None:
            st.image(pil_img, caption="Ảnh đầu vào (đã chuẩn hóa RGB)", width="stretch")

    with col_right:
        st.subheader("2. Kết quả Phân loại")
        if pil_img is None:
            st.info("👈 Vui lòng tải ảnh lên hoặc chọn ảnh mẫu ở cột bên trái để bắt đầu nhận diện.")
        else:
            if st.button("🚀 PHÂN LOẠI NGAY", type="primary", width="stretch", key="btn_upload"):
                with st.spinner("Đang trích xuất đặc trưng MobileNetV2 (1.280d) và dự đoán Random Forest..."):
                    pred_cls, conf, probs = predict_waste(pil_img, extractor, rf_model, class_names)
                render_prediction_result(pred_cls, conf, probs, class_names)

with tab_camera:
    cam_left, cam_right = st.columns([1, 1.15], gap="large")
    cam_img = None

    with cam_left:
        st.subheader("1. Chụp ảnh trực tiếp từ Webcam")
        cam_file = st.camera_input("Đưa vật thể rác vào khung hình và nhấn chụp:")
        if cam_file is not None:
            try:
                cam_img = to_rgb(Image.open(cam_file))
            except (UnidentifiedImageError, OSError):
                st.error("Không đọc được ảnh từ camera.")

    with cam_right:
        st.subheader("2. Kết quả Phân loại")
        if cam_img is None:
            st.info("👈 Nhấn nút chụp ảnh từ webcam để hệ thống nhận diện loại rác.")
        else:
            with st.spinner("Đang phân loại ảnh chụp từ camera..."):
                pred_cls, conf, probs = predict_waste(cam_img, extractor, rf_model, class_names)
            render_prediction_result(pred_cls, conf, probs, class_names)

with tab_eval:
    st.subheader("📈 So sánh Hiệu năng: Không Augmentation vs Có Augmentation (Tập Test độc lập)")
    if eval_df is not None:
        st.dataframe(eval_df, width="stretch", hide_index=True)
    else:
        st.info("Chưa có file `results/metrics/evaluation_results.csv`.")

    c1, c2 = st.columns(2)
    cmp_chart = PROJECT_ROOT / "results" / "charts" / "augmentation_comparison.png"
    f1_chart = PROJECT_ROOT / "results" / "charts" / "class_wise_f1.png"
    if cmp_chart.exists():
        c1.image(str(cmp_chart), caption="So sánh 2 kịch bản Data Augmentation", width="stretch")
    if f1_chart.exists():
        c2.image(str(f1_chart), caption="So sánh F1-Score chi tiết 12 lớp", width="stretch")

    st.markdown("---")
    st.subheader("🔎 Ma trận Nhầm lẫn 12 lớp (Phân tích các cặp dễ nhầm: nhựa, kim loại, thủy tinh...)")
    cm1, cm2 = st.columns(2)
    cm_raw = PROJECT_ROOT / "results" / "confusion_matrix" / "confusion_matrix.png"
    cm_norm = PROJECT_ROOT / "results" / "confusion_matrix" / "confusion_matrix_normalized.png"
    if cm_raw.exists():
        cm1.image(str(cm_raw), caption="Ma trận nhầm lẫn (Số lượng mẫu)", width="stretch")
    if cm_norm.exists():
        cm2.image(str(cm_norm), caption="Ma trận nhầm lẫn chuẩn hóa (% Recall)", width="stretch")

    analysis_path = PROJECT_ROOT / "results" / "metrics" / "confusion_analysis.txt"
    if analysis_path.exists():
        with st.expander("📋 Xem phân tích các cặp lớp nhầm lẫn nhiều nhất"):
            st.code(analysis_path.read_text(encoding="utf-8"), language="text")
