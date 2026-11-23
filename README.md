# Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh bằng CNN kết hợp Random Forest

**Môn học:** Trí tuệ Nhân tạo / Deep Learning  
**Đề tài:** Đề tài 19 (Mức Nâng cao)  
**Giảng viên hướng dẫn:** Thầy Bùi Trọng Hiếu  
**Nhóm thực hiện:** Nhóm 05  

---

## 1. Tổng quan Đề tài

Dự án xây dựng hệ thống nhận diện và phân loại rác thải sinh hoạt qua hình ảnh, hỗ trợ phân loại rác tại nguồn thành 4 nhóm xử lý từ 12 lớp rác chi tiết.

* **Kiến trúc mô hình cốt lõi:**
  $$\text{Ảnh đầu vào} \xrightarrow{\text{Tiền xử lý (224}\times\text{224 RGB)}} \text{MobileNetV2 (ImageNet)} \xrightarrow{\text{GAP}} \text{Vector 1.280 chiều} \xrightarrow{\text{Random Forest}} \text{12 Lớp chi tiết} \rightarrow \text{4 Nhóm xử lý}$$
* **Thư viện thống nhất toàn nhóm:** PyTorch + Torchvision + Pillow, scikit-learn + joblib, Streamlit (chạy trực tiếp trên máy cá nhân).

---

## 2. Dữ liệu & Quy tắc Phân chia (TV2 – Trần Lâm)

* **Bộ dữ liệu:** Kaggle *"Garbage Classification"* (12 lớp, 15.515 ảnh gốc). Sau khi lọc ảnh lỗi và loại **19 ảnh trùng MD5** (`src/prepare_data.py`), còn **15.496 ảnh hợp lệ**, xác nhận **0 ảnh rò rỉ** giữa 3 tập (`src/make_stats.py`).
* **Chia phân tầng cố định (`stratify`, `seed=42`):**
  * **Train (70%):** 10.847 ảnh gốc $\rightarrow$ **32.541 mẫu** khi áp dụng Data Augmentation (`N_AUG = 2`: giữ 100% ảnh gốc + 2 biến thể/ảnh).
  * **Validation (15%):** 2.324 ảnh (không tăng cường).
  * **Test (15%):** 2.325 ảnh (không tăng cường).
* **Ánh xạ 12 lớp sang 4 nhóm xử lý:**
  1. 🍏 **Rác hữu cơ:** `biological`
  2. 🔋 **Rác nguy hại:** `battery`
  3. ♻️ **Rác tái chế:** `plastic`, `metal`, `paper`, `cardboard`, `white-glass`, `green-glass`, `brown-glass`, `clothes`, `shoes`
  4. 🗑️ **Rác còn lại (Khó tái chế):** `trash`

---

## 3. Kết quả Thực nghiệm Thực tế (Tập Test độc lập 2.325 ảnh)

Mô hình tối ưu chọn trên Validation (`n_estimators=200`, `max_depth=25`, `min_samples_split=2`, `class_weight='balanced'`), lưu tại `models/random_forest_model.joblib` (36,72 MB):

| Kịch bản Dữ liệu | Số mẫu Train | Thời gian Train | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không Augmentation (Baseline)** | 10.847 | 4.63s | **93.08%** | **90.92%** | **89.64%** | **90.20%** |
| **Có Augmentation (Tăng cường)** | 32.541 | 17.16s | **94.06%** | **91.96%** | **91.21%** | **91.52%** |
| **Mức cải thiện ($\Delta$)** | **+21.694** | +12.53s | **+0.98%** | **+1.04%** | **+1.57%** | **+1.32%** |

* Kiểm thử trên 65 ảnh ngoài tập Kaggle (`demo_samples/`): đúng loại rác (12 lớp) **46/65 (70.77%)**, đúng nhóm xử lý (4 nhóm) **56/65 (86.15%)**.
* Phân tích chi tiết các cặp lớp dễ nhầm lẫn (nhựa/thủy tinh/kim loại/giấy-carton) lưu tại `results/metrics/confusion_analysis.txt` và `results/confusion_matrix/`.

---

## 4. Phân công 5 Thành viên

| Thành viên | Vai trò chính | Sản phẩm code & dữ liệu trong repo |
| :--- | :--- | :--- |
| **TV1 – Lê Lý Phúc An (Nhóm trưởng)** | Kiến trúc nền móng, trích xuất MobileNetV2, RF baseline & tổng hợp báo cáo | `src/config.py`, `src/mobilenet_feature_extractor.py`, `src/extract_features.py`, `src/random_forest.py`, `src/train_baseline.py`, `src/predictor.py` |
| **TV2 – Trần Lâm** | Dữ liệu 12 lớp, tiền xử lý & Data Augmentation | `src/prepare_data.py`, `src/preprocessing.py`, `src/augmentation.py`, `src/make_stats.py`, `data/data_statistics.csv`, `results/charts/*.png` |
| **TV3 – Nguyễn Văn Khương** | Tối ưu siêu tham số RF, đánh giá Test & Confusion Matrix | `src/evaluation.py`, `src/tune_and_evaluate.py`, `models/random_forest_model.joblib`, `results/metrics/*`, `results/confusion_matrix/*` |
| **TV4 – Quí** | Ứng dụng Web Demo Streamlit & ánh xạ nhãn | `app/app.py`, `src/mapping.py`, `src/predictor.py`, `requirements.txt` |
| **TV5 – Hoàng** | Chương 1, sơ đồ 2.1, thuật toán 2.2, kiểm thử ảnh thật & Chương 3 | `demo_samples/`, `src/test_demo_samples.py`, `demo_samples/real_test_cases.csv` & các phần báo cáo Word |

---

## 5. Kiến trúc Dự án (Chia 4 Tầng từ Nền lên Đỉnh)

```plaintext
ai-group05-cnn-waste-classification/
├── src/                                    # PACKAGE LÕI
│   ├── __init__.py
│   ├── config.py                           # [Tầng 0 - Nền móng] Đường dẫn & hằng số kỹ thuật chung
│   ├── preprocessing.py                    # [Tầng 1 - Dữ liệu] Ép RGB, Resize 224x224, Normalize ImageNet
│   ├── augmentation.py                     # [Tầng 1 - Dữ liệu] Tăng cường dữ liệu tập Train & xuất demo
│   ├── prepare_data.py                     # [Bước 1] Lọc ảnh hỏng, khử trùng MD5 & chia 70/15/15
│   ├── make_stats.py                       # [Bước 2] Kiểm tra rò rỉ MD5 & vẽ biểu đồ phân bố 12 lớp
│   ├── mobilenet_feature_extractor.py      # [Tầng 2 - CNN] MobileNetV2 GAP xuất vector 1.280 chiều
│   ├── extract_features.py                 # [Bước 3] Trích xuất 4 tập ra data/X_*.npy, y_*.npy
│   ├── random_forest.py                    # [Tầng 3 - ML] Khởi tạo, huấn luyện, lưu/nạp Random Forest
│   ├── evaluation.py                       # [Tầng 3 - Đánh giá] Metrics, Classification Report & Confusion Matrix
│   ├── train_baseline.py                   # [Bước 4A] Train RF baseline kiểm chứng nhanh pipeline
│   ├── tune_and_evaluate.py                # [Bước 4B] Tối ưu siêu tham số trên Val & đánh giá Test
│   ├── mapping.py                          # [Tầng 4 - Nghiệp vụ] Ánh xạ 12 lớp -> 4 nhóm xử lý & hướng dẫn
│   ├── predictor.py                        # [Tầng 4 - Cầu nối] Lớp WastePredictor đóng gói luồng dự đoán 1 ảnh
│   └── test_demo_samples.py                # [Bước 5] Kiểm thử hàng loạt 65 ảnh demo_samples/
├── app/
│   └── app.py                              # [Giao diện Demo] Ứng dụng Streamlit gọi WastePredictor
├── data/                                   # class_names.json, data_statistics.csv, *.npy, train/, val/, test/
├── models/
│   └── random_forest_model.joblib          # Mô hình Random Forest tối ưu nhất (36,72 MB)
├── demo_samples/                           # Bộ ảnh kiểm thử & real_test_cases.csv
├── results/                                # charts/, confusion_matrix/, metrics/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Hướng dẫn Chạy Dự án Theo Từng Bước (Bước 1 -> Bước 6)

```bash
# Cài đặt thư viện (Python 3.10 - 3.12)
pip install -r requirements.txt

# Bước 1 (TV2): Lọc ảnh hỏng/trùng MD5 và chia Stratified 70/15/15 (tuỳ chọn nếu chia lại từ ảnh gốc)
python src/prepare_data.py

# Bước 2 (TV2): Kiểm tra rò rỉ dữ liệu & xuất biểu đồ phân bố
python src/make_stats.py

# Bước 3 (TV1): Trích xuất vector MobileNetV2 1.280 chiều cho 4 tập ra data/*.npy
python src/extract_features.py

# Bước 4 (TV1 & TV3): Train baseline, tối ưu siêu tham số trên Validation & đánh giá Test
python src/train_baseline.py
python src/tune_and_evaluate.py

# Bước 5 (TV5): Kiểm thử mô hình trên 65 ảnh mẫu trong demo_samples/
python src/test_demo_samples.py

# Bước 6 (TV4): Khởi chạy giao diện Web Demo Streamlit
streamlit run app/app.py
```
