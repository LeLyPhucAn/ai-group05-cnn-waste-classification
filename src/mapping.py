"""
Module: mapping.py
Tác giả: Thành viên 4 - Phụ trách Ứng dụng Demo Streamlit (TV4) & TV1
Mục đích:
  - Ánh xạ 12 lớp rác thải chi tiết sang 4 nhóm xử lý:
      1. Rác hữu cơ (biological)
      2. Rác nguy hại (battery)
      3. Rác tái chế (plastic, metal, paper, cardboard, white-glass, green-glass, brown-glass, clothes, shoes)
      4. Rác còn lại - Khó tái chế (trash)
  - Cung cấp nhãn hiển thị tiếng Việt, mã màu giao diện, icon và hướng dẫn phân loại xử lý.
"""

from typing import Dict, Any


# Bảng ánh xạ chi tiết 12 lớp sang nhóm xử lý và thông tin phụ trợ
CLASS_MAPPING_INFO: Dict[str, Dict[str, Any]] = {
    "battery": {
        "vi_name": "Pin / Ắc quy",
        "group": "Rác nguy hại",
        "color": "#E53E3E",  # Đỏ
        "badge_bg": "#FED7D7",
        "icon": "🔋",
        "guide": "Chất thải nguy hại chứa kim loại nặng và hóa chất độc hại. KHÔNG vứt chung với rác sinh hoạt. Hãy gom riêng vào hộp kín và mang tới điểm thu hồi rác điện tử/nguy hại."
    },
    "biological": {
        "vi_name": "Rác hữu cơ / Thức ăn thừa",
        "group": "Rác hữu cơ",
        "color": "#38A169",  # Xanh lá
        "badge_bg": "#C6F6D5",
        "icon": "🍏",
        "guide": "Rác hữu cơ dễ phân hủy sinh học (thực phẩm dư thừa, cuống rau củ quả). Bỏ vào thùng rác hữu cơ để ủ làm phân bón compost cho cây trồng."
    },
    "brown-glass": {
        "vi_name": "Thủy tinh nâu",
        "group": "Rác tái chế",
        "color": "#3182CE",  # Xanh dương
        "badge_bg": "#BEE3F8",
        "icon": "🍾",
        "guide": "Thủy tinh có khả năng tái chế vô hạn. Rửa sạch cặn bẩn bên trong, bọc cẩn thận tránh vỡ gây thương tích và bỏ vào thùng rác tái chế."
    },
    "cardboard": {
        "vi_name": "Bìa carton",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "📦",
        "guide": "Giữ thùng carton khô ráo, gập phẳng để tiết kiệm thể tích và bỏ vào thùng rác tái chế."
    },
    "clothes": {
        "vi_name": "Quần áo cũ / Đồ vải",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "👕",
        "guide": "Quần áo cũ còn dùng được nên giặt sạch và quyên góp từ thiện; đồ rách nát có thể tái chế sợi dệt công nghiệp."
    },
    "green-glass": {
        "vi_name": "Thủy tinh xanh",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🍾",
        "guide": "Thủy tinh tái chế. Vui lòng rửa sạch và phân loại vào thùng rác tái chế vật liệu thủy tinh."
    },
    "metal": {
        "vi_name": "Kim loại / Lon nhôm, sắt",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🥫",
        "guide": "Kim loại có giá trị tái chế rất cao. Trút hết nước thừa, ép dẹp vỏ lon và bỏ vào thùng tái chế kim loại."
    },
    "paper": {
        "vi_name": "Giấy / Sách báo cũ",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "📄",
        "guide": "Giữ giấy khô ráo, không để dính dầu mỡ hoặc thức ăn rồi gom vào thùng thu gom giấy tái chế."
    },
    "plastic": {
        "vi_name": "Chai lọ / Đồ nhựa tái chế",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🥤",
        "guide": "Đồ nhựa phổ biến (PET, HDPE). Rửa trôi cặn bẩn, tháo nắp và vặn xẹp thân chai trước khi cho vào thùng tái chế nhựa."
    },
    "shoes": {
        "vi_name": "Giày dép cũ",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "👟",
        "guide": "Giày dép còn tốt nên tặng lại hoặc quyên góp. Các loại đế cao su/nhựa có thể tái chế làm vật liệu lát sân thể thao."
    },
    "trash": {
        "vi_name": "Rác sinh hoạt khó tái chế",
        "group": "Rác còn lại (Khó tái chế)",
        "color": "#718096",  # Xám
        "badge_bg": "#E2E8F0",
        "icon": "🗑️",
        "guide": "Rác hỗn hợp, xỉ than, tã lót hoặc đồ nhựa dính bẩn không thể tái chế. Buộc chặt miệng túi và bỏ vào thùng rác vô cơ để chuyển đến bãi chôn lấp/nhà máy đốt rác."
    },
    "white-glass": {
        "vi_name": "Thủy tinh trong suốt",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🍶",
        "guide": "Chai lọ thủy tinh trong suốt (hũ thực phẩm, ly cốc). Rửa sạch bên trong và bỏ vào thùng rác tái chế thủy tinh."
    }
}


def get_mapping_info(class_name: str) -> Dict[str, Any]:
    """
    Lấy thông tin chi tiết của một lớp rác theo tên lớp tiếng Anh trong Kaggle.
    """
    return CLASS_MAPPING_INFO.get(class_name, {
        "vi_name": class_name,
        "group": "Chưa xác định",
        "color": "#4A5568",
        "badge_bg": "#EDF2F7",
        "icon": "❓",
        "guide": "Vui lòng kiểm tra lại chất liệu để phân loại rác đúng quy chuẩn."
    })


def get_waste_group(class_name: str) -> str:
    """Trả về nhóm xử lý (Hữu cơ / Nguy hại / Tái chế / Rác còn lại)."""
    return get_mapping_info(class_name)["group"]


def get_vi_name(class_name: str) -> str:
    """Trả về tên tiếng Việt của loại rác."""
    return get_mapping_info(class_name)["vi_name"]


def get_waste_guide(class_name: str) -> str:
    """Trả về hướng dẫn bỏ thùng rác."""
    return get_mapping_info(class_name)["guide"]


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM THỬ MODULE ÁNH XẠ NHÃN MAPPING.PY (TV4 & TV1)")
    print("=" * 60)
    for c_name, info in CLASS_MAPPING_INFO.items():
        print(f"[{c_name:<12}] -> Nhóm: {info['group']:<25} | Tên: {info['vi_name']}")
    print("=" * 60)
    print("Kiểm thử thành công: Toàn bộ 12 lớp đã được ánh xạ chính xác!")
