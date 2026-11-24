# Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh bằng CNN kết hợp Random Forest

**Môn học:** Trí tuệ Nhân tạo | **Đề tài:** Đề tài 19 
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

| Họ và tên | MSSV | Công việc |
| :--- | :---: | :--- |
| **Lê Lý Phúc An**<br>*(Nhóm trưởng)* | `080206002048` | - Thiết kế kiến trúc hệ thống, quản lý kho mã nguồn GitHub và điều phối nhóm.<br>- Lập trình trích xuất đặc trưng MobileNetV2, xuất vector đặc trưng (`.npy`) và huấn luyện mô hình Random Forest gốc.<br>- Kiểm thử tích hợp, tổng hợp, đối chiếu số liệu và hoàn thiện báo cáo cuối. |
| **Nguyễn Phúc Hoàng** | `049206002481` | - Viết phần Mở đầu, Chương 1, mục 2.1–2.2 (sơ đồ quy trình, lý thuyết, mã giả), mục 2.5 (giao diện), Chương 3 và Tài liệu tham khảo.<br>- Thu thập ảnh rác thật, kiểm thử ứng dụng thực tế, chụp ảnh minh họa demo. |
| **Nguyễn Văn Khương** | `066206009437` | - Xây dựng module đánh giá (`evaluation.py`) và tối ưu siêu tham số cho mô hình Random Forest.<br>- Thực nghiệm so sánh có/không Data Augmentation trên tập Test, xuất chỉ số đánh giá và vẽ Confusion Matrix.<br>- Viết mục Quá trình tối ưu và Kết quả đánh giá trong Chương 2. |
| **Trần Sơn Lâm** | `080206007896` | - Thu thập, làm sạch và chia bộ dữ liệu 12 lớp theo tỷ lệ 70/15/15.<br>- Lập trình tiền xử lý ảnh (`preprocessing.py`), tăng cường dữ liệu (`augmentation.py`) và thống kê, vẽ biểu đồ dữ liệu.<br>- Viết mục Dữ liệu, Tiền xử lý và Data Augmentation trong Chương 2. |
| **Đào Minh Quí** | `051207019415` | - Xây dựng ứng dụng demo Streamlit (`app.py`) và module ánh xạ 12 lớp rác sang nhóm xử lý (`mapping.py`).<br>- Tích hợp mô hình MobileNetV2 + Random Forest vào ứng dụng, sửa lỗi và đóng gói thư viện (`requirements.txt`).<br>- Viết mục Triển khai ứng dụng và sơ đồ luồng xử lý kỹ thuật trong Chương 2. |

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

## 5. Hướng dẫn Cài đặt & Chạy Từng bước

> **Yêu cầu:** Python **3.10 – 3.12** (khuyến nghị **Python 3.12** để tương thích ổn định với PyTorch).  
> Luôn gọi qua `-m pip` / `-m streamlit` để tránh lỗi `not recognized` hoặc nhầm phiên bản khi máy cài nhiều bản Python.

### Bước 1: Kiểm tra phiên bản Python & Cài đặt thư viện

#### Cách 1 (Khuyên dùng – tạo môi trường ảo `.venv`, chạy chuẩn trên mọi máy cài nhiều Python)
```powershell
# 1. Xem các bản Python đang cài trên Windows (chọn 3.12 hoặc 3.11 / 3.10)
py -0p

# 2. Tạo môi trường ảo bằng Python 3.12
py -3.12 -m venv .venv

# 3. Kích hoạt môi trường ảo (Windows PowerShell)
.venv\Scripts\Activate.ps1
# (Nếu dùng CMD: .venv\Scripts\activate.bat | macOS/Linux: source .venv/bin/activate)

# 4. Cài đặt toàn bộ thư viện
python -m pip install -r requirements.txt
```

#### Cách 2 (Chạy trực tiếp không cần `.venv` trên Windows có nhiều bản Python)
```powershell
# Chỉ đích danh Python 3.12 để cài thư viện (thay 3.12 bằng 3.11/3.10 nếu máy dùng bản đó)
py -3.12 -m pip install -r requirements.txt
```
*(Nếu máy chỉ cài 1 bản Python duy nhất, thay `py -3.12` bằng `python`.)*

---

### Bước 2: Chạy ứng dụng Web Demo (dùng mô hình đã huấn luyện sẵn)
```powershell
# Nếu đã kích hoạt .venv (hoặc máy chỉ có 1 bản Python)
python -m streamlit run app/app.py

# Nếu chạy trực tiếp trên Windows cài nhiều bản Python
py -3.12 -m streamlit run app/app.py
```
Mở trình duyệt tại **`http://localhost:8501`**. Nhấn `Ctrl + C` trong terminal để tắt.

---

### Bước 3: Chạy lại quy trình Huấn luyện & Thực nghiệm từng bước (tuỳ chọn)

*(Thay `py -3.12` bằng `python` nếu bạn đang ở trong `.venv` hoặc máy chỉ có 1 bản Python.)*

```powershell
# Bước 3.1 (Tuỳ chọn): Lọc ảnh hỏng/trùng MD5 và chia Stratified 70/15/15 từ ảnh Kaggle gốc
py -3.12 src/prepare_data.py --raw-dir data/raw/garbage_classification --out-dir data

# Bước 3.2: Kiểm tra rò rỉ dữ liệu MD5 & xuất biểu đồ phân bố 12 lớp
py -3.12 src/make_stats.py

# Bước 3.3: Trích xuất vector MobileNetV2 (1.280 chiều) cho 4 tập ra data/*.npy (~15 phút trên CPU)
py -3.12 src/extract_features.py --split-dir data --aug-copies 2 --batch-size 64

# Bước 3.4A: Train nhanh Random Forest baseline kiểm chứng 2 kịch bản (~15 giây)
py -3.12 src/train_baseline.py

# Bước 3.4B: Tối ưu siêu tham số trên Validation, đánh giá Test, vẽ Confusion Matrix & lưu model cuối (~1-2 phút)
py -3.12 src/tune_and_evaluate.py

# Bước 3.5: Chấm điểm tự động bộ ảnh mẫu demo_samples/
py -3.12 src/test_demo_samples.py

# Bước 3.6: Mở lại ứng dụng Web Demo với mô hình vừa train
py -3.12 -m streamlit run app/app.py
```
