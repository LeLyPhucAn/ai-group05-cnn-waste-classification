# Hướng dẫn & Quy chuẩn Dữ liệu (Garbage Classification 12 lớp)

Thành viên phụ trách chính: **Thành viên 2 (Trần Lâm - Data & Augmentation)**  
Phối hợp kỹ thuật: **Thành viên 1 (Lê Lý Phúc An - Kiến trúc & Trích xuất đặc trưng)**

---

## 1. Nguồn dữ liệu & Cấu trúc phân loại
- **Bộ dữ liệu gốc:** Kaggle *"Garbage Classification"* gồm 12 lớp rác thải sinh hoạt chi tiết (15.515 ảnh gốc).
- **Sau khi làm sạch (`src/prepare_data.py`):** Đã lọc ảnh lỗi (0 ảnh) và loại bỏ **19 ảnh trùng lặp nội dung MD5** (`clothes`: -1, `green-glass`: -15, `white-glass`: -3) $\rightarrow$ còn **15.496 ảnh hợp lệ**, đã xác minh **0 ảnh rò rỉ** giữa Train / Val / Test (`src/make_stats.py`).
- **Thứ tự 12 lớp chuẩn hóa:** Lưu tại `data/class_names.json` (sắp xếp theo thứ tự bảng chữ cái):
  1. `battery` (Pin / Ắc quy)
  2. `biological` (Rác hữu cơ / Thức ăn thừa)
  3. `brown-glass` (Thủy tinh nâu)
  4. `cardboard` (Bìa carton)
  5. `clothes` (Quần áo cũ / Vải)
  6. `green-glass` (Thủy tinh xanh lá)
  7. `metal` (Kim loại / Lon nước)
  8. `paper` (Giấy)
  9. `plastic` (Nhựa / Chai nhựa)
  10. `shoes` (Giày dép cũ)
  11. `trash` (Rác thải khó tái chế khác)
  12. `white-glass` (Thủy tinh trong suốt)

---

## 2. Quy tắc chia tập dữ liệu (70% - 15% - 15%)
Chia phân tầng (**Stratified Split**) với `random_state = 42` (chi tiết từng lớp tại `data/data_statistics.csv` và `results/charts/data_statistics.csv`):

| Tập dữ liệu | Tỷ lệ | Số lượng ảnh (Gốc) | Số lượng ảnh (Có Augmentation) | Vị trí cục bộ |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | **70%** | **10.847** ảnh | **32.541** ảnh *(gốc + 2 biến thể)* | `data/train/` |
| **Validation** | **15%** | **2.324** ảnh | **2.324** ảnh *(không tăng cường)* | `data/val/` |
| **Test** | **15%** | **2.325** ảnh | **2.325** ảnh *(không tăng cường)* | `data/test/` |
| **Tổng cộng** | **100%** | **15.496** ảnh | **37.190** mẫu | |

---

## 3. Quy tắc Data Augmentation (`src/augmentation.py`)
> [!IMPORTANT]
> **Chỉ áp dụng Data Augmentation trên tập Train!** Tuyệt đối giữ nguyên tập Validation và Test.

- **Quy tắc sinh ảnh:** Giữ nguyên 100% ảnh gốc (`10.847`) + sinh thêm `N_AUG = 2` biến thể cho mỗi ảnh (`21.694`) $\rightarrow$ tổng **32.541 mẫu Train**.
- **Các phép biến đổi (`seed = 42`):**
  - `RandomHorizontalFlip(p=0.5)`: Lật ngang ngẫu nhiên.
  - `RandomRotation(degrees=20)`: Xoay ngẫu nhiên $\pm 20^\circ$.
  - `RandomResizedCrop(224, scale=(0.7, 1.0))`: Cắt/phóng ngẫu nhiên.
  - `ColorJitter(brightness=0.3, contrast=0.3)`: Thay đổi độ sáng & độ tương phản.
- **Ảnh minh họa:** `results/charts/augmentation_demo.png`.
