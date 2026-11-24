# Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh bằng CNN kết hợp Random Forest

**Môn học:** Trí tuệ Nhân tạo | **Đề tài:** Đề tài 19 (Mức Nâng cao)  
**Giảng viên hướng dẫn:** Thầy Bùi Trọng Hiếu | **Nhóm thực hiện:** Nhóm 05  

---

## 1. Tổng quan Dự án

Hệ thống nhận diện ảnh rác thải sinh hoạt thành **12 lớp chi tiết** (bộ Kaggle *Garbage Classification*, 15.496 ảnh sạch) và ánh xạ sang **4 nhóm xử lý tại nguồn** (*Rác hữu cơ, Rác nguy hại, Rác tái chế, Rác còn lại*) kèm hướng dẫn bỏ thùng rác.

* **Kiến trúc mô hình:**
  $$\text{Ảnh đầu vào} \xrightarrow{\text{RGB, 224}\times\text{224}} \text{MobileNetV2 (ImageNet GAP)} \xrightarrow{\text{1.280 chiều}} \text{Random Forest} \longrightarrow \text{12 Lớp} \longrightarrow \text{4 Nhóm xử lý}$$
* **Công nghệ sử dụng:** PyTorch & Torchvision (MobileNetV2), scikit-learn & joblib (Random Forest), Pillow, Pandas, Matplotlib/Seaborn, Streamlit (HTML/CSS/JS tách riêng).

---

## 2. Kết quả Thực nghiệm (Tập Test độc lập 2.325 ảnh)

Mô hình tối ưu chọn trên tập Validation (`n_estimators=200`, `max_depth=25`, `min_samples_split=2`, `class_weight='balanced'`), lưu tại `models/random_forest_model.joblib` (36,72 MB):

| Kịch bản Dữ liệu | Số mẫu Train | Thời gian Train | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không Augmentation (Baseline)** | 10.847 | 4.63s | **93.08%** | **90.92%** | **89.64%** | **90.20%** |
| **Có Augmentation (Tăng cường)** | 32.541 | 17.16s | **94.06%** | **91.96%** | **91.21%** | **91.52%** |
| **Mức cải thiện ($\Delta$)** | **+21.694** | +12.53s | **+0.98%** | **+1.04%** | **+1.57%** | **+1.32%** |

* **Kiểm thử ngoài tập Kaggle (`demo_samples/`):** Đúng lớp chi tiết **46/65 (70.77%)**, đúng nhóm xử lý **56/65 (86.15%)**.
* **Biểu đồ & Ma trận nhầm lẫn:** Lưu tại `results/charts/`, `results/confusion_matrix/` và `results/metrics/confusion_analysis.txt`.

---

## 3. Phân công Thành viên

| Thành viên | Vai trò chính | Sản phẩm cốt lõi |
| :--- | :--- | :--- |
| **TV1 – Lê Lý Phúc An (Nhóm trưởng)** | Kiến trúc hệ thống, MobileNetV2, RF baseline & tổng hợp báo cáo | `src/config.py`, `src/mobilenet_feature_extractor.py`, `src/extract_features.py`, `src/random_forest.py`, `src/train_baseline.py`, `src/predictor.py` |
| **TV2 – Trần Lâm** | Dữ liệu 12 lớp, tiền xử lý & Data Augmentation | `src/prepare_data.py`, `src/preprocessing.py`, `src/augmentation.py`, `src/make_stats.py`, `data/data_statistics.csv`, `results/charts/*.png` |
| **TV3 – Nguyễn Văn Khương** | Tối ưu siêu tham số RF, đánh giá Test & Confusion Matrix | `src/evaluation.py`, `src/tune_and_evaluate.py`, `models/random_forest_model.joblib`, `results/metrics/*`, `results/confusion_matrix/*` |
| **TV4 – Quí** | Ứng dụng Web Demo Streamlit & ánh xạ nhãn | `app/app.py`, `app/templates/*`, `app/static/*`, `src/mapping.py`, `requirements.txt` |
| **TV5 – Hoàng** | Chương 1, sơ đồ 2.1, thuật toán 2.2, kiểm thử ảnh mẫu & Chương 3 | `demo_samples/`, `src/test_demo_samples.py`, `demo_samples/real_test_cases.csv` & tài liệu báo cáo |

---

## 4. Cấu trúc Thư mục

```plaintext
ai-group05-cnn-waste-classification/
├── .streamlit/
│   └── config.toml                    # Cấu hình giao diện tối giản (ẩn toolbar mặc định)
├── app/
│   ├── app.py                         # Ứng dụng Web Demo Streamlit
│   ├── templates/                     # Giao diện HTML tách riêng (header.html, result_card.html)
│   └── static/                        # Định dạng CSS & JS tách riêng (style.css, script.js)
├── src/
│   ├── config.py                      # Đường dẫn tuyệt đối & hằng số kỹ thuật dùng chung
│   ├── prepare_data.py                # Bước 1: Lọc ảnh lỗi/trùng MD5, chia Stratified 70/15/15
│   ├── make_stats.py                  # Bước 2: Kiểm tra rò rỉ MD5 & vẽ biểu đồ phân bố 12 lớp
│   ├── preprocessing.py               # Tiền xử lý RGB, Resize 224x224, Normalize ImageNet
│   ├── augmentation.py                # Tăng cường dữ liệu riêng cho tập Train & vẽ ảnh demo
│   ├── mobilenet_feature_extractor.py # MobileNetV2 (ImageNet GAP -> vector 1.280 chiều)
│   ├── extract_features.py            # Bước 3: Trích xuất 4 tập vector .npy lưu vào data/
│   ├── random_forest.py               # Khởi tạo, huấn luyện, lưu & nạp Random Forest (.joblib)
│   ├── evaluation.py                  # Tính Accuracy/P/R/F1, vẽ Confusion Matrix & biểu đồ
│   ├── train_baseline.py              # Bước 4A: Train Random Forest baseline kiểm chứng nhanh
│   ├── tune_and_evaluate.py           # Bước 4B: Tối ưu siêu tham số Val, đánh giá Test, lưu model
│   ├── mapping.py                     # Ánh xạ 12 lớp Kaggle -> 4 nhóm xử lý & hướng dẫn
│   ├── predictor.py                   # Lớp WastePredictor đóng gói trọn luồng dự đoán 1 ảnh
│   └── test_demo_samples.py           # Bước 5: Kiểm thử hàng loạt ảnh trong demo_samples/
├── data/                              # class_names.json, data_statistics.csv, *.npy, train/val/test
├── models/                            # random_forest_model.joblib (mô hình tối ưu cuối cùng)
├── demo_samples/                      # 101 ảnh mẫu (đủ 12 lớp) & bảng real_test_cases.csv
├── results/                           # charts/, confusion_matrix/, metrics/
├── requirements.txt
└── README.md
```

---

## 5. Hướng dẫn Cài đặt & Chạy

```bash
# 1. Cài đặt thư viện (Python 3.10 - 3.12)
pip install -r requirements.txt

# 2. Chạy ứng dụng Web Demo (http://localhost:8501)
streamlit run app/app.py
```

### Chạy lại quy trình huấn luyện từng bước (nếu cần train lại)
```bash
python src/prepare_data.py        # Bước 1: Lọc ảnh lỗi/trùng và chia 70/15/15 (từ ảnh gốc)
python src/make_stats.py          # Bước 2: Kiểm tra rò rỉ dữ liệu & vẽ biểu đồ phân bố
python src/extract_features.py    # Bước 3: Trích xuất vector MobileNetV2 1.280d ra data/*.npy
python src/train_baseline.py      # Bước 4A: Train Random Forest baseline
python src/tune_and_evaluate.py   # Bước 4B: Tối ưu Val, chấm Test & lưu random_forest_model.joblib
python src/test_demo_samples.py   # Bước 5: Kiểm thử hàng loạt ảnh trong demo_samples/
streamlit run app/app.py          # Bước 6: Mở giao diện Web Demo
```
