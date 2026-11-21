# Waste-Classification: Hệ thống phân loại rác thải sinh hoạt qua ảnh

Đề tài 19 (Mức Nâng cao) - Môn học: Trí tuệ nhân tạo / Deep Learning  
Giảng viên hướng dẫn: **Thầy Bùi Trọng Hiếu**

## 1. Giới thiệu Đề tài
Hệ thống phân loại rác thải sinh hoạt qua ảnh bằng phương pháp kết hợp mạng nơ-ron tích chập (CNN - MobileNetV2 pretrained ImageNet) và thuật toán Rừng ngẫu nhiên (Random Forest).

* **Kiến trúc mô hình:**
  $$\text{Ảnh đầu vào} \xrightarrow{\text{Tiền xử lý (224x224)}} \text{MobileNetV2 (ImageNet)} \xrightarrow{\text{GAP}} \text{Vector 1.280 chiều} \xrightarrow{\text{Random Forest}} \text{12 Lớp chi tiết} \rightarrow \text{4 Nhóm xử lý}$$
* **Bộ dữ liệu:** Kaggle "Garbage Classification" gồm 12 lớp rác thải (Tổng cộng 15.515 ảnh).
* **Chia tập:** Train 70% (10.854 ảnh) – Validation 15% (2.321 ảnh) – Test 15% (2.340 ảnh).

---

## 2. Kết quả thực nghiệm chính (Thành viên 3 - TV3)

| Kịch bản Dữ liệu | Số mẫu Train | Thời gian Train | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không Augmentation (Baseline)** | 10.854 | 21.02s | **94.36%** | **97.19%** | **90.48%** | **93.46%** |
| **Có Augmentation (Tăng cường)** | 21.708 | 33.28s | **95.94%** | **97.96%** | **93.12%** | **95.34%** |
| **Mức cải thiện (Δ)** | **+10.854** | +12.26s | **+1.58%** | **+0.77%** | **+2.64%** | **+1.88%** |

* **Mô hình tối ưu:** Đã lưu tại `models/random_forest_model.joblib` (`n_estimators=200`, `max_depth=None`, `min_samples_split=2`, `class_weight='balanced'`).
* **Ma trận nhầm lẫn & Biểu đồ:** Lưu tại thư mục `results/`.

---

## 3. Hướng dẫn tích hợp cho các thành viên

### Dành cho TV4 (Ứng dụng Streamlit `app/app.py`):
```python
import joblib
import numpy as np

# 1. Nạp mô hình đã train của TV3
model = joblib.load("models/random_forest_model.joblib")

# 2. Dự đoán xác suất cho vector 1.280 chiều trích xuất từ MobileNetV2
probs = model.predict_proba(features_1280)[0]   # Mảng xác suất 12 lớp
pred_id = model.predict(features_1280)[0]       # ID lớp dự đoán (0 đến 11)
confidence = probs[pred_id] * 100               # Độ tin cậy (%)
```

### Dành cho TV2 (Dữ liệu & Tiền xử lý):
* Thứ tự 12 lớp chuẩn hóa tại `data/class_names.json`.
* Module đánh giá tại `src/evaluation.py`.

---

## 4. Cài đặt môi trường

```bash
pip install -r requirements.txt
```
