"""
Ứng dụng Web Demo: Hệ thống Phân loại Rác thải (Đề tài 19 - Nhóm 05)
Cấu trúc tầng giao diện tách riêng:
  - app/app.py                   : Điều phối giao diện và gọi WastePredictor
  - app/static/style.css         : Toàn bộ CSS định dạng giao diện
  - app/static/script.js         : JavaScript hỗ trợ phía trình duyệt
  - app/templates/header.html    : Khung tiêu đề chính
  - app/templates/result_card.html: Khung thẻ kết quả dự đoán

Chạy ứng dụng:
  streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHARTS_DIR, CM_ANALYSIS_PATH, CM_DIR, DEMO_DIR, EVAL_CSV_PATH
from src.mapping import get_vi_name
from src.predictor import WastePredictor
from src.preprocessing import to_rgb

# -------------------------------------------------------------
# 1. CẤU HÌNH TRANG & NẠP MODULE HTML / CSS / JS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Phân loại Rác thải",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def inject_assets() -> None:
    """Nạp file CSS và JS tách riêng từ app/static/."""
    css = load_text(APP_DIR / "static" / "style.css")
    js = load_text(APP_DIR / "static" / "script.js")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    components.html(f"<script>{js}</script>", height=0)


@st.cache_resource(show_spinner="Đang tải mô hình AI...")
def get_predictor() -> WastePredictor:
    return WastePredictor()


def list_demo_images() -> dict:
    """Lấy tối đa 3 ảnh đại diện mỗi lớp trong demo_samples/ để test nhanh."""
    if not DEMO_DIR.is_dir():
        return {}
    samples = {}
    for cls_dir in sorted(DEMO_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for p in imgs[:3]:
            samples[f"{get_vi_name(cls_dir.name)} - {p.name}"] = p
    return samples


def show_prediction(res: dict) -> None:
    """Render kết quả phân loại bằng template HTML riêng biệt."""
    tpl = load_text(APP_DIR / "templates" / "result_card.html")
    st.markdown(tpl.format(**res), unsafe_allow_html=True)

    if res["is_low_confidence"]:
        st.warning("Độ tin cậy thấp (< 55%): Ảnh có thể thiếu sáng, nền phức tạp hoặc thuộc cặp vật liệu dễ nhầm.")

    st.markdown("#### Top 5 lớp có xác suất cao nhất")
    for item in res["top_k"]:
        st.write(f"**{item['vi_name']}** (`{item['class_name']}`): `{item['prob']:.2f}%`")
        st.progress(min(max(item["prob"] / 100.0, 0.0), 1.0))


# -------------------------------------------------------------
# 2. HIỂN THỊ GIAO DIỆN CHÍNH
# -------------------------------------------------------------
inject_assets()
st.markdown(load_text(APP_DIR / "templates" / "header.html"), unsafe_allow_html=True)

try:
    predictor = get_predictor()
except Exception as err:
    st.error(f"Không thể khởi tạo mô hình AI: {err}")
    st.stop()

tab_upload, tab_camera, tab_eval = st.tabs([
    "Tải ảnh lên",
    "Chụp ảnh Camera",
    "Kết quả thực nghiệm",
])

with tab_upload:
    col_left, col_right = st.columns([1, 1.1], gap="large")
    pil_img = None

    with col_left:
        st.subheader("1. Chọn ảnh đầu vào")
        mode = st.radio("Nguồn ảnh:", ["Tải ảnh từ máy tính", "Chọn ảnh mẫu (demo_samples)"], horizontal=True)
        if mode == "Tải ảnh từ máy tính":
            uploaded = st.file_uploader("Chọn ảnh (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
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
                st.info("Chưa có ảnh mẫu trong demo_samples/.")

        if pil_img is not None:
            st.image(pil_img, caption="Ảnh đầu vào", width="stretch")

    with col_right:
        st.subheader("2. Kết quả phân loại")
        if pil_img is None:
            st.info("Vui lòng chọn hoặc tải lên một bức ảnh ở cột bên trái.")
        elif st.button("Phân loại ngay", type="primary", width="stretch"):
            with st.spinner("Đang phân tích ảnh..."):
                result = predictor.predict(pil_img)
            show_prediction(result)

with tab_camera:
    cam_left, cam_right = st.columns([1, 1.1], gap="large")
    cam_img = None

    with cam_left:
        st.subheader("1. Chụp ảnh từ Camera")
        cam_file = st.camera_input("Đưa vật thể rác vào khung hình và nhấn chụp:")
        if cam_file is not None:
            try:
                cam_img = to_rgb(Image.open(cam_file))
            except (UnidentifiedImageError, OSError):
                st.error("Không đọc được ảnh từ camera.")

    with cam_right:
        st.subheader("2. Kết quả phân loại")
        if cam_img is None:
            st.info("Nhấn nút chụp ảnh từ camera để nhận diện.")
        else:
            with st.spinner("Đang phân tích ảnh..."):
                result = predictor.predict(cam_img)
            show_prediction(result)

with tab_eval:
    st.subheader("So sánh Không Augmentation và Có Augmentation (Tập Test 2.325 ảnh)")
    if EVAL_CSV_PATH.exists():
        st.dataframe(pd.read_csv(EVAL_CSV_PATH), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    if (CHARTS_DIR / "augmentation_comparison.png").exists():
        c1.image(str(CHARTS_DIR / "augmentation_comparison.png"), caption="So sánh hiệu năng Augmentation", width="stretch")
    if (CHARTS_DIR / "class_wise_f1.png").exists():
        c2.image(str(CHARTS_DIR / "class_wise_f1.png"), caption="F1-Score từng lớp (12 lớp)", width="stretch")

    st.markdown("---")
    st.subheader("Ma trận nhầm lẫn 12 lớp (Confusion Matrix)")
    cm1, cm2 = st.columns(2)
    if (CM_DIR / "confusion_matrix.png").exists():
        cm1.image(str(CM_DIR / "confusion_matrix.png"), caption="Ma trận nhầm lẫn (Số mẫu)", width="stretch")
    if (CM_DIR / "confusion_matrix_normalized.png").exists():
        cm2.image(str(CM_DIR / "confusion_matrix_normalized.png"), caption="Ma trận nhầm lẫn (% Recall)", width="stretch")

    if CM_ANALYSIS_PATH.exists():
        with st.expander("Xem phân tích các cặp lớp dễ nhầm lẫn"):
            st.code(CM_ANALYSIS_PATH.read_text(encoding="utf-8"), language="text")
