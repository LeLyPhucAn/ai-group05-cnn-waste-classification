# Hệ thống Phân loại Rác thải Sinh hoạt qua Ảnh bằng CNN kết hợp Random Forest

**Môn học:** Trí tuệ Nhân tạo / Deep Learning  
**Đề tài:** Đề tài 19 (Mức Nâng cao)  
**Giảng viên hướng dẫn:** Thầy Bùi Trọng Hiếu  
**Nhóm thực hiện:** Nhóm 05  

---

## 1. Tổng quan Đề tài

Dự án xây dựng hệ thống nhận diện và phân loại rác thải sinh hoạt qua hình ảnh, nhằm hỗ trợ tự động hóa khâu phân loại rác tại nguồn theo định hướng bảo vệ môi trường và kinh tế tuần hoàn.

* **Kiến trúc mô hình cốt lõi:**
  $$\text{Ảnh đầu vào} \xrightarrow{\text{Tiền xử lý (224}\times\text{224)}} \text{MobileNetV2 (ImageNet)} \xrightarrow{\text{GAP}} \text{Vector 1.280 chiều} \xrightarrow{\text{Random Forest}} \text{12 Lớp chi tiết} \rightarrow \text{4 Nhóm xử lý}$$
* **Phương pháp:** Kết hợp mạng nơ-ron tích chập sâu (**MobileNetV2** đóng vai trò Feature Extractor với pretrained weights từ ImageNet, lược bỏ tầng Classifier) cùng thuật toán học máy phân loại mạnh mẽ (**Random Forest Classifier**).
* **Quy chuẩn kỹ thuật thống nhất:** Huấn luyện trực tiếp trên máy cá nhân, sử dụng thư viện đồng nhất PyTorch + Torchvision + Pillow + scikit-learn + Streamlit.

---

## 2. Dữ liệu & Quy tắc Phân chia (TV2)

* **Bộ dữ liệu gốc:** Kaggle *"Garbage Classification"* gồm **15.515 ảnh** thuộc 12 lớp rác thải sinh hoạt chi tiết.
* **Quy tắc phân chia:** Chia phân tầng cố định (**Stratified Split**, `seed=42`):
  * **Train (70%):** 10.854 ảnh gốc (Tăng lên **21.708 mẫu** khi áp dụng Data Augmentation).
  * **Validation (15%):** 2.328 ảnh (Giữ nguyên không tăng cường để tối ưu siêu tham số khách quan).
  * **Test (15%):** 2.333 ảnh (Giữ nguyên không tăng cường để đánh giá độc lập).
* **Ánh xạ từ 12 lớp chi tiết sang 4 nhóm xử lý:**
  1. 🍏 **Rác hữu cơ:** `biological`
  2. 🔋 **Rác nguy hại:** `battery`
  3. ♻️ **Rác tái chế:** `plastic`, `metal`, `paper`, `cardboard`, `white-glass`, `green-glass`, `brown-glass`, `clothes`, `shoes`
  4. 🗑️ **Rác còn lại (Khó tái chế):** `trash`

---

## 3. Kết quả Thực nghiệm & Đánh giá (TV3)

Đề tài tiến hành so sánh đối chứng 2 kịch bản dữ liệu trên cùng kiến trúc mô hình trên tập Test độc lập:

| Kịch bản Dữ liệu | Mẫu Train | Thời gian Train | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không Augmentation (Baseline)** | 10.854 | 21.02s | **94.36%** | **97.19%** | **90.48%** | **93.46%** |
| **Có Augmentation (Tăng cường)** | 21.708 | 33.28s | **95.94%** | **97.96%** | **93.12%** | **95.34%** |
| **Mức cải thiện ($\Delta$)** | **+10.854** | +12.26s | **+1.58%** | **+0.77%** | **+2.64%** | **+1.88%** |

* **Mô hình Random Forest tối ưu:** Đã tinh chỉnh qua tập Validation với `n_estimators=200`, `max_depth=None`, `class_weight='balanced'`, lưu tại `models/random_forest_model.joblib`.
* **Ma trận nhầm lẫn & Biểu đồ:** Đã kết xuất đầy đủ tại thư mục `results/confusion_matrix/` và `results/charts/`.

---

## 4. Phân công Công việc & Mốc Chốt chặn

| Thành viên | Vai trò chính | Sản phẩm bàn giao cốt lõi |
| :--- | :--- | :--- |
| **TV1 – Lê Lý Phúc An (Nhóm trưởng)** | Kiến trúc hệ thống, Trích xuất đặc trưng MobileNetV2, RF baseline & Tổng hợp báo cáo | Repo GitHub, `mobilenet_feature_extractor.py`, `random_forest.py`, bộ vector `.npy`, `BaoCao_Full.docx` |
| **TV2** | Dữ liệu & Tăng cường dữ liệu (Data Augmentation) | Dataset chia 70/15/15, `preprocessing.py`, `augmentation.py`, `class_names.json`, mục Dữ liệu Chương 2 |
| **TV3** | Tối ưu hóa Random Forest, Đánh giá & Confusion Matrix | `evaluation.py`, `tune_and_evaluate.py`, `random_forest_model.joblib`, ma trận nhầm lẫn, mục Kết quả Chương 2 |
| **TV4** | Ứng dụng Web Demo Streamlit & Tích hợp AI | `app/app.py`, `src/mapping.py`, tích hợp suy luận end-to-end, mục Triển khai Chương 2 |
| **TV5** | Lý thuyết, Kiểm thử thực tế, Giao diện & Chương 3 | Chương 1, Sơ đồ quy trình 2.1, Thuật toán có Pseudocode 2.2, kiểm thử rác thật, Chương 3 & Video demo |

---

## 5. Cấu trúc Thư mục Dự án

```plaintext
ai-group05-cnn-waste-classification/
├── data/
│   ├── README.md                  # Hướng dẫn dữ liệu, tỷ lệ chia và mô tả 12 lớp
│   └── class_names.json           # Danh sách 12 nhãn lớp chuẩn hóa tiếng Anh
├── src/
│   ├── mobilenet_feature_extractor.py  # (TV1) Khởi tạo MobileNetV2 & trích xuất vector 1.280d
│   ├── random_forest.py                # (TV1/TV3) Huấn luyện & lưu/nạp model Random Forest
│   ├── preprocessing.py               # (TV2) Tiền xử lý ảnh chuẩn hóa ImageNet
│   ├── augmentation.py                # (TV2) Pipeline tăng cường dữ liệu tập Train
│   ├── evaluation.py                  # (TV3) Tính toán Accuracy, F1, vẽ Confusion Matrix
│   ├── tune_and_evaluate.py           # (TV3) Tinh chỉnh siêu tham số và chạy thực nghiệm 2 kịch bản
│   ├── generate_dataset_vectors.py    # (TV1/TV3) Trích xuất/mô phỏng vector đặc trưng
│   └── mapping.py                     # (TV4) Ánh xạ 12 lớp sang 4 nhóm xử lý & hướng dẫn
├── models/
│   └── random_forest_model.joblib     # Mô hình Random Forest tối ưu nhất
├── app/
│   └── app.py                         # Ứng dụng Web Demo tương tác bằng Streamlit
├── results/
│   ├── confusion_matrix/              # Hình ảnh ma trận nhầm lẫn 12 lớp
│   ├── charts/                        # Biểu đồ so sánh kịch bản và F1-score từng lớp
│   └── metrics/                       # Bảng chỉ số đánh giá CSV và classification report
├── report/                            # Thư mục lưu trữ các file báo cáo Word/PDF
├── requirements.txt                   # Danh sách thư viện bắt buộc
├── .gitignore                         # Danh sách file loại trừ khỏi Git
└── README.md                          # Tài liệu hướng dẫn dự án chung
```

---

## 6. Hướng dẫn Cài đặt & Sử dụng

### 6.1. Cài đặt Môi trường
Khuyến nghị sử dụng Python 3.10 – 3.12:
```bash
pip install -r requirements.txt
```

### 6.2. Kiểm thử Module Trích xuất Đặc trưng (MobileNetV2)
```bash
python src/mobilenet_feature_extractor.py
```

### 6.3. Khởi chạy Ứng dụng Web Demo Streamlit
```bash
streamlit run app/app.py
```
Ứng dụng sẽ mở giao diện trực quan trên trình duyệt (mặc định tại `http://localhost:8501`), hỗ trợ tải ảnh từ máy tính hoặc chụp ảnh trực tiếp qua camera để phân loại rác và nhận hướng dẫn xử lý.
