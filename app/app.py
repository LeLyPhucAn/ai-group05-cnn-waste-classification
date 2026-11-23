"""
Ứng dụng Web Demo: Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh (Đề tài 19)
Phụ trách giao diện: TV4 Quí | Tích hợp kiến trúc: TV1 Lê Lý Phúc An
Mục đích:
  Tầng giao diện Streamlit đơn giản, trực quan. Mọi xử lý AI được gọi qua lớp WastePredictor (src/predictor.py).
Chạy:
  streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHARTS_DIR, CM_ANALYSIS_PATH, CM_DIR, DEMO_DIR, EVAL_CSV_PATH
from src.mapping import get_vi_name
from src.predictor import WastePredictor
from src.preprocessing import to_rgb

# -------------------------------------------------------------
# 1. CẤU HÌNH TRANG & CSS ĐƠN GIẢN
# -------------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Phân loại Rác thải AI - Nhóm 05",
    page_icon="♻️",
    layout="wide",
)

st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #43b04a 0%, #2196f3 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .hero-banner h1 { font-size: 2rem; margin: 0 0 0.3rem 0; color: white; }
    .hero-banner p { font-size: 1rem; margin: 0; }
    .result-card {
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        margin-top: 0.4rem;
    }
    .group-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.88rem;
        color: white;
    }
    .guide-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8fafc;
        border-left: 5px solid #2196f3;
        margin-top: 0.9rem;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. NẠP BỘ DỰ ĐOÁN AI (CACHE RESOURCE 1 LẦN DUY NHẤT)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải mô hình MobileNetV2 & Random Forest...")
def get_predictor() -> WastePredictor:
    return WastePredictor()


def list_demo_images() -> dict:
    """Lấy danh sách ảnh mẫu đại diện từ demo_samples/ để chọn test nhanh."""
    if not DEMO_DIR.is_dir():
        return {}
    samples = {}
    for cls_dir in sorted(DEMO_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for p in imgs[:3]:
            samples[f"{get_vi_name(cls_dir.name)} — {p.name}"] = p
    return samples


def show_prediction(res: dict) -> None:
    """Hiển thị thẻ kết quả phân loại, hướng dẫn bỏ thùng rác và Top 5 xác suất."""
    st.markdown(f"""
    <div class="result-card" style="background-color: {res['badge_bg']}; border: 2px solid {res['color']};">
        <div class="group-pill" style="background-color: {res['color']};">
            {res['icon']} {res['group'].upper()}
        </div>
        <div style="font-size: 1.75rem; font-weight: 800; color: #0f172a; margin-top: 0.3rem;">
            {res['vi_name']}
        </div>
        <div style="font-size: 0.96rem; color: #334155; margin-top: 0.2rem;">
            Lớp kỹ thuật (12 lớp): <code>{res['class_name']}</code> &nbsp;|&nbsp;
            Độ tin cậy: <b>{res['confidence']:.2f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if res["is_low_confidence"]:
        st.warning("Độ tin cậy thấp (< 55%): Ảnh có thể thiếu sáng, nền phức tạp hoặc thuộc cặp vật liệu dễ nhầm.")

    st.markdown(f"""
    <div class="guide-box">
        <b>📋 Hướng dẫn bỏ thùng rác:</b><br>{res['guide']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Top 5 lớp có xác suất cao nhất")
    for item in res["top_k"]:
        st.write(f"**{item['vi_name']}** (`{item['class_name']}`): `{item['prob']:.2f}%`")
        st.progress(min(max(item["prob"] / 100.0, 0.0), 1.0))


# -------------------------------------------------------------
# 3. BỐ CỤC GIAO DIỆN CHÍNH
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>♻️ HỆ THỐNG PHÂN LOẠI RÁC THẢI THÔNG MINH</h1>
    <p>Đề tài 19 (Nâng cao): MobileNetV2 (1.280d) + Random Forest | 12 lớp Kaggle &rarr; 4 nhóm xử lý</p>
</div>
""", unsafe_allow_html=True)

try:
    predictor = get_predictor()
except Exception as err:
    st.error(f"Không thể khởi tạo mô hình AI: {err}")
    st.stop()

eval_df = pd.read_csv(EVAL_CSV_PATH) if EVAL_CSV_PATH.exists() else None

with st.sidebar:
    st.header("📌 Thông tin Đề tài")
    st.markdown("""
    - **Môn học:** Trí tuệ Nhân tạo
    - **Giảng viên:** Thầy Bùi Trọng Hiếu
    - **Nhóm thực hiện:** Nhóm 05
    - **Mô hình:** MobileNetV2 + Random Forest
    - **Dữ liệu:** 15.496 ảnh sạch (12 lớp)
    """)
    if eval_df is not None and len(eval_df) >= 2:
        aug_row = eval_df.iloc[-1]
        st.success(f"**Test Accuracy:** {aug_row['Test Accuracy (%)']}%\n\n**Macro F1:** {aug_row['Macro F1-Score (%)']}%")
    st.markdown("---")
    st.markdown("""
    **🗂️ 4 Nhóm Xử lý:**
    - 🍏 **Hữu cơ:** `biological`
    - 🔋 **Nguy hại:** `battery`
    - ♻️ **Tái chế:** `plastic`, `metal`, `paper`, `cardboard`, `glass`, `clothes`, `shoes`
    - 🗑️ **Còn lại:** `trash`
    """)

tab_upload, tab_camera, tab_eval = st.tabs([
    "📁 Nhận diện từ Ảnh tải lên",
    "📷 Chụp ảnh bằng Camera",
    "📊 Kết quả Thực nghiệm & Ma trận Nhầm lẫn",
])

with tab_upload:
    col_left, col_right = st.columns([1, 1.15], gap="large")
    pil_img = None

    with col_left:
        st.subheader("1. Chọn ảnh đầu vào")
        mode = st.radio("Nguồn ảnh:", ["Tải ảnh từ máy tính", "Chọn nhanh ảnh mẫu (demo_samples)"], horizontal=True)
        if mode == "Tải ảnh từ máy tính":
            uploaded = st.file_uploader("Chọn file ảnh (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
            if uploaded is not None:
                try:
                    pil_img = to_rgb(Image.open(uploaded))
                except (UnidentifiedImageError, OSError):
                    st.error("File ảnh bị lỗi hoặc không đúng định dạng.")
        else:
            demos = list_demo_images()
            if demos:
                chosen = st.selectbox("Chọn ảnh mẫu:", list(demos.keys()))
                pil_img = to_rgb(Image.open(demos[chosen]))
            else:
                st.info("Chưa có ảnh mẫu trong thư mục demo_samples/.")

        if pil_img is not None:
            st.image(pil_img, caption="Ảnh rác thải cần phân loại", width="stretch")

    with col_right:
        st.subheader("2. Kết quả Phân loại")
        if pil_img is None:
            st.info("👈 Vui lòng chọn hoặc tải lên một bức ảnh ở cột bên trái.")
        elif st.button("🚀 PHÂN LOẠI NGAY", type="primary", width="stretch"):
            with st.spinner("Đang trích xuất đặc trưng MobileNetV2 và phân loại..."):
                result = predictor.predict(pil_img)
            show_prediction(result)

with tab_camera:
    cam_left, cam_right = st.columns([1, 1.15], gap="large")
    cam_img = None

    with cam_left:
        st.subheader("1. Chụp ảnh từ Webcam")
        cam_file = st.camera_input("Đưa rác thải vào giữa khung hình và nhấn chụp:")
        if cam_file is not None:
            try:
                cam_img = to_rgb(Image.open(cam_file))
            except (UnidentifiedImageError, OSError):
                st.error("Không thể đọc ảnh từ camera.")

    with cam_right:
        st.subheader("2. Kết quả Phân loại")
        if cam_img is None:
            st.info("👈 Nhấn chụp ảnh từ webcam để nhận diện.")
        else:
            with st.spinner("Đang phân loại ảnh chụp..."):
                result = predictor.predict(cam_img)
            show_prediction(result)

with tab_eval:
    st.subheader("📈 So sánh Kịch bản Không Augmentation vs Có Augmentation (Tập Test 2.325 ảnh)")
    if eval_df is not None:
        st.dataframe(eval_df, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    if (CHARTS_DIR / "augmentation_comparison.png").exists():
        c1.image(str(CHARTS_DIR / "augmentation_comparison.png"), caption="So sánh hiệu năng Augmentation", width="stretch")
    if (CHARTS_DIR / "class_wise_f1.png").exists():
        c2.image(str(CHARTS_DIR / "class_wise_f1.png"), caption="F1-Score chi tiết 12 lớp", width="stretch")

    st.markdown("---")
    st.subheader("🔎 Ma trận Nhầm lẫn 12 lớp (Confusion Matrix)")
    cm1, cm2 = st.columns(2)
    if (CM_DIR / "confusion_matrix.png").exists():
        cm1.image(str(CM_DIR / "confusion_matrix.png"), caption="Ma trận nhầm lẫn (Số mẫu)", width="stretch")
    if (CM_DIR / "confusion_matrix_normalized.png").exists():
        cm2.image(str(CM_DIR / "confusion_matrix_normalized.png"), caption="Ma trận nhầm lẫn (% Recall)", width="stretch")

    if CM_ANALYSIS_PATH.exists():
        with st.expander("📋 Xem phân tích chi tiết các cặp lớp dễ nhầm lẫn"):
            st.code(CM_ANALYSIS_PATH.read_text(encoding="utf-8"), language="text")
