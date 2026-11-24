# Thư mục Ảnh Kiểm thử Demo (`demo_samples/`)

**Người phụ trách chính:** Thành viên 5 (Hoàng)  
**Lưu ý Git:** Các file ảnh PNG/JPG dung lượng lớn (~140 MB) được loại trừ khỏi Git qua `.gitignore`; repo chỉ theo dõi file `README.md` và bảng `real_test_cases.csv`.

---

## 1. Cấu trúc Ảnh Hiện Có (65 ảnh từ TV5)
Ảnh đã được sắp xếp lại thành 6 thư mục tên lớp tiếng Anh chuẩn theo `data/class_names.json`:
- `demo_samples/battery/` (`battery_01.png` – `battery_10.png`): 10 ảnh (từ `Pin`)
- `demo_samples/biological/` (`biological_01.png` – `biological_10.png`): 10 ảnh (từ `Vỏ trái cây`)
- `demo_samples/cardboard/` (`cardboard_01.png` – `cardboard_11.png`): 11 ảnh (từ `Giấy_carton`)
- `demo_samples/metal/` (`metal_01.png` – `metal_10.png`): 10 ảnh (từ `Lon nước`)
- `demo_samples/plastic/` (`plastic_01.png` – `plastic_10.png`): 10 ảnh (từ `Chai nhựa`)
- `demo_samples/white-glass/` (`white-glass_01.png` – `white-glass_14.png`): 14 ảnh (từ `Chai thủy tinh`)

---

## 2. Lưu ý Kiểm thử Thực tế
- Danh sách toàn bộ 65 ảnh đã được lập sẵn tại `demo_samples/real_test_cases.csv` (để trống cột dự đoán cho tới khi chạy trên app Streamlit bản dùng model huấn luyện từ dữ liệu thật).
- **Khuyến nghị cho TV5:** Trong 65 ảnh hiện tại có nhiều ảnh tạo sinh bằng ChatGPT (`Ảnh ChatGPT ...`). Theo ghi chú của chính bảng kiểm thử, TV5 nên chụp thêm **15–20 ảnh rác tự chụp ngoài đời bằng điện thoại** để báo cáo Mục 2.5 & Hạn chế Chương 3 mang tính thực tế thuyết phục nhất khi bảo vệ.
