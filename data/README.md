# Hướng dẫn & Quy chuẩn Dữ liệu (Garbage Classification 12 lớp)

Thành viên phụ trách chính: **Thành viên 2 (TV2 - Data & Augmentation)**  
Phối hợp kỹ thuật: **Thành viên 1 (TV1 - Kiến trúc & Trích xuất đặc trưng)**

---

## 1. Nguồn dữ liệu & Cấu trúc phân loại
- **Bộ dữ liệu:** Kaggle *"Garbage Classification"* gồm 12 lớp rác thải sinh hoạt chi tiết.
- **Tổng số lượng ảnh gốc:** 15.515 ảnh (định dạng JPG/PNG).
- **Thứ tự 12 lớp chuẩn hóa:** Lưu tại `data/class_names.json` (sắp xếp theo thứ tự chữ cái ABC):
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
Để đảm bảo tính khách quan và kiểm soát rò rỉ dữ liệu (Data Leakage), việc chia tập phải thực hiện phân tầng (**Stratified Split**) với `random_state = 42`:

| Tập dữ liệu | Tỷ lệ | Số lượng ảnh (Gốc) | Số lượng ảnh (Có Augmentation) | Mục đích sử dụng |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | **70%** | **10.854** ảnh | **21.708** ảnh | Huấn luyện mô hình Random Forest |
| **Validation** | **15%** | **2.328** ảnh | **2.328** ảnh | Tối ưu siêu tham số Random Forest |
| **Test** | **15%** | **2.333** ảnh | **2.333** ảnh | Đánh giá độc lập & phân tích nhầm lẫn |
| **Tổng cộng** | **100%** | **15.515** ảnh | **26.369** mẫu | |

---

## 3. Quy tắc Data Augmentation
> [!IMPORTANT]
> **Chỉ áp dụng Data Augmentation trên tập Train!**  
> Tuyệt đối **KHÔNG** áp dụng Augmentation lên tập Validation và Test để kết quả đánh giá phản ánh chân thực năng lực khái quát hóa trên dữ liệu thực tế.

- **Kịch bản thực nghiệm:**
  1. **Kịch bản 1 (Không Augmentation - Raw Baseline):** 10.854 vector Train gốc.
  2. **Kịch bản 2 (Có Augmentation - Tăng cường):** 100% ảnh gốc + sinh thêm 1 bản biến đổi ngẫu nhiên cho mỗi ảnh, nâng quy mô tập Train lên 21.708 vector.
- **Các phép biến đổi áp dụng:**
  - `RandomHorizontalFlip(p=0.5)`: Lật ngang ngẫu nhiên.
  - `RandomRotation(degrees=15)`: Xoay góc nhẹ $\pm 15^\circ$.
  - `RandomResizedCrop(224, scale=(0.85, 1.0))`: Cắt phóng ngẫu nhiên.
  - `ColorJitter(brightness=0.15, contrast=0.15)`: Thay đổi nhẹ độ sáng và độ tương phản.

---

## 4. Ánh xạ 12 lớp sang 4 nhóm xử lý rác
Mô hình AI huấn luyện trên toàn bộ **12 lớp chi tiết** để phục vụ phân tích ma trận nhầm lẫn (Confusion Matrix). Trên ứng dụng giao diện (Streamlit), kết quả sẽ được ánh xạ sang **4 nhóm xử lý**:

| Nhóm xử lý | Các lớp tương ứng trong bộ 12 lớp | Màu nhận diện | Hướng dẫn phân loại |
| :--- | :--- | :---: | :--- |
| **Rác hữu cơ** | `biological` | Xanh lá | Bỏ vào thùng rác hữu cơ để làm phân compost. |
| **Rác nguy hại** | `battery` | Đỏ / Cam | Gom riêng đến điểm thu hồi chất thải nguy hại, không vứt bừa bãi. |
| **Rác tái chế** | `plastic`, `metal`, `paper`, `cardboard`, `white-glass`, `green-glass`, `brown-glass`, `clothes`, `shoes` | Xanh dương | Làm sạch sơ bộ, phân loại theo vật liệu và bỏ vào thùng tái chế. |
| **Rác còn lại** | `trash` | Xám | Bỏ vào thùng rác vô cơ / rác sinh hoạt thông thường. |
