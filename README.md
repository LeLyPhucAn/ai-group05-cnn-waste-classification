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
| **TV1 – Lê Lý Phúc An (Nhóm trưởng)** | Kiến trúc, trích xuất MobileNetV2, RF baseline & tổng hợp báo cáo | `src/mobilenet_feature_extractor.py`, `src/extract_features.py`, `src/random_forest.py`, `src/train_baseline.py` |
| **TV2 – Trần Lâm** | Dữ liệu 12 lớp, tiền xử lý & Data Augmentation | `src/prepare_data.py`, `src/preprocessing.py`, `src/augmentation.py`, `src/make_stats.py`, `data/data_statistics.csv`, `results/charts/*.png` |
| **TV3 – Nguyễn Văn Khương** | Tối ưu siêu tham số RF, đánh giá Test & Confusion Matrix | `src/evaluation.py`, `src/tune_and_evaluate.py`, `models/random_forest_model.joblib`, `results/metrics/*`, `results/confusion_matrix/*` |
| **TV4 – Quí** | Ứng dụng Web Demo Streamlit & tích hợp AI | `app/app.py`, `src/mapping.py`, `requirements.txt` |
| **TV5 – Hoàng** | Chương 1, sơ đồ 2.1, thuật toán 2.2, kiểm thử ảnh thật & Chương 3 | `demo_samples/`, `demo_samples/real_test_cases.csv` & các phần báo cáo Word |

---

## 5. Cấu trúc Thư mục Dự án

```plaintext
ai-group05-cnn-waste-classification/
├── data/
│   ├── README.md                        # Quy chuẩn chia tập 70/15/15 & mô tả 12 lớp
│   ├── class_names.json                 # Thứ tự 12 nhãn lớp chuẩn hóa
│   ├── data_statistics.csv              # Thống kê chuẩn 15.496 ảnh của TV2
│   ├── train/                           # Ảnh tập Train (cục bộ, .gitignore)
│   ├── val/                             # Ảnh tập Validation (cục bộ, .gitignore)
│   └── test/                            # Ảnh tập Test (cục bộ, .gitignore)
├── src/
│   ├── prepare_data.py                  # (TV2) Lọc ảnh lỗi/trùng MD5 & chia 70/15/15
│   ├── preprocessing.py                 # (TV2) Tiền xử lý chuẩn ImageNet (224x224 RGB)
│   ├── augmentation.py                  # (TV2) Tăng cường dữ liệu tập Train & xuất demo
│   ├── make_stats.py                    # (TV2) Kiểm tra rò rỉ MD5 & vẽ biểu đồ dữ liệu
│   ├── mobilenet_feature_extractor.py   # (TV1) MobileNetV2 GAP xuất vector 1.280 chiều
│   ├── extract_features.py              # (TV1) Trích xuất 4 tập ra data/*.npy
│   ├── random_forest.py                 # (TV1/TV3) Khởi tạo, huấn luyện, lưu/nạp RF
│   ├── train_baseline.py                # (TV1) Train RF baseline kiểm chứng pipeline
│   ├── evaluation.py                    # (TV3) Metrics, Classification Report & Confusion Matrix
│   ├── tune_and_evaluate.py             # (TV3) Tối ưu siêu tham số Validation & đánh giá Test
│   ├── generate_dataset_vectors.py      # (Legacy/Mock) Sinh vector giả lập kiểm thử nhanh
│   └── mapping.py                       # (TV4) Ánh xạ 12 lớp -> 4 nhóm xử lý & hướng dẫn
├── models/
│   └── random_forest_model.joblib       # Mô hình Random Forest tối ưu nhất
├── app/
│   └── app.py                           # Ứng dụng Web Demo Streamlit
├── demo_samples/                        # Bộ ảnh kiểm thử (ảnh nặng .gitignore, giữ CSV)
├── results/
│   ├── charts/                          # Biểu đồ phân bố dữ liệu, Augmentation & F1
│   ├── confusion_matrix/                # Ma trận nhầm lẫn 12 lớp
│   └── metrics/                         # Bảng chỉ số CSV & classification report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Hướng dẫn Cài đặt & Chạy Pipeline End-to-End

### 6.1. Cài đặt môi trường (Python 3.10 – 3.12)
```bash
pip install -r requirements.txt
```

### 6.2. Chuẩn bị dữ liệu & Thống kê (TV2)
```bash
# (Tuỳ chọn nếu chạy lại từ bộ ảnh gốc Kaggle)
python src/prepare_data.py --raw-dir data/raw/garbage_classification --out-dir data

# Kiểm tra rò rỉ MD5 & vẽ lại biểu đồ phân bố
python src/make_stats.py --split-dir data
```

### 6.3. Trích xuất vector 1.280 chiều & Train baseline (TV1)
```bash
# Trích xuất 4 tập (Train gốc, Train + 2 bản biến đổi Augmentation, Val, Test) ra data/*.npy
python src/extract_features.py --split-dir data --aug-copies 2

# Train Random Forest baseline kiểm chứng pipeline
python src/train_baseline.py
```

### 6.4. Tối ưu siêu tham số & Xuất kết quả đánh giá (TV3)
```bash
python src/tune_and_evaluate.py
```

### 6.5. Kiểm thử bộ ảnh mẫu & Khởi chạy Ứng dụng Web Demo (TV4 & TV5)
```bash
python src/test_demo_samples.py
streamlit run app/app.py
```
