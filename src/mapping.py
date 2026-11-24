"""
Module: mapping.py (Tầng 4 - Ánh xạ Nghiệp vụ | Phụ trách: TV4 Quí)
Mục đích:
  Ánh xạ 12 lớp rác chi tiết (Kaggle) sang 4 nhóm xử lý tại nguồn:
    1. Rác hữu cơ (biological)
    2. Rác nguy hại (battery)
    3. Rác tái chế (plastic, metal, paper, cardboard, white/green/brown-glass, clothes, shoes)
    4. Rác còn lại - Khó tái chế (trash)
"""

from typing import Any, Dict

CLASS_MAPPING_INFO: Dict[str, Dict[str, Any]] = {
    "battery": {
        "vi_name": "Pin / Ắc quy",
        "group": "Rác nguy hại",
        "color": "#E53E3E",
        "badge_bg": "#FED7D7",
        "icon": "🔋",
        "guide": "Chất thải nguy hại chứa kim loại nặng. KHÔNG vứt chung với rác sinh hoạt; hãy gom riêng và mang tới điểm thu hồi pin/rác điện tử.",
    },
    "biological": {
        "vi_name": "Rác hữu cơ / Thức ăn thừa",
        "group": "Rác hữu cơ",
        "color": "#38A169",
        "badge_bg": "#C6F6D5",
        "icon": "🍏",
        "guide": "Rác hữu cơ dễ phân hủy sinh học. Bỏ vào thùng rác hữu cơ để ủ làm phân bón compost.",
    },
    "brown-glass": {
        "vi_name": "Thủy tinh nâu",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🍾",
        "guide": "Rửa sạch cặn bẩn bên trong, bọc cẩn thận tránh vỡ và bỏ vào thùng rác tái chế thủy tinh.",
    },
    "cardboard": {
        "vi_name": "Bìa carton",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "📦",
        "guide": "Giữ bìa carton khô ráo, gấp phẳng tiết kiệm diện tích và bỏ vào thùng rác tái chế.",
    },
    "clothes": {
        "vi_name": "Quần áo cũ / Đồ vải",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "👕",
        "guide": "Quần áo còn tốt nên giặt sạch để quyên góp; đồ vải cũ hỏng phân loại vào nhóm tái chế sợi dệt.",
    },
    "green-glass": {
        "vi_name": "Thủy tinh xanh",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🍾",
        "guide": "Tráng sạch chai lọ và bỏ vào thùng thu gom thủy tinh tái chế.",
    },
    "metal": {
        "vi_name": "Kim loại / Lon nhôm, sắt",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🥫",
        "guide": "Đổ hết nước thừa, ép dẹp vỏ lon và bỏ vào thùng rác tái chế kim loại.",
    },
    "paper": {
        "vi_name": "Giấy / Sách báo cũ",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "📄",
        "guide": "Giữ giấy khô sạch, không dính dầu mỡ và bỏ vào thùng tái chế giấy.",
    },
    "plastic": {
        "vi_name": "Chai lọ / Đồ nhựa tái chế",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🥤",
        "guide": "Tráng sạch, tháo nắp, bóp dẹp thân chai nhựa và bỏ vào thùng rác tái chế nhựa.",
    },
    "shoes": {
        "vi_name": "Giày dép cũ",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "👟",
        "guide": "Giày dép còn dùng được nên quyên góp; đế cao su/nhựa có thể thu gom tái chế vật liệu.",
    },
    "trash": {
        "vi_name": "Rác sinh hoạt khó tái chế",
        "group": "Rác còn lại (Khó tái chế)",
        "color": "#718096",
        "badge_bg": "#E2E8F0",
        "icon": "🗑️",
        "guide": "Rác hỗn hợp hoặc vật liệu dính bẩn không thể tái chế. Buộc kín túi và bỏ vào thùng rác sinh hoạt còn lại.",
    },
    "white-glass": {
        "vi_name": "Thủy tinh trong suốt",
        "group": "Rác tái chế",
        "color": "#3182CE",
        "badge_bg": "#BEE3F8",
        "icon": "🍶",
        "guide": "Chai lọ thủy tinh trong suốt. Tráng sạch và bỏ vào thùng tái chế thủy tinh.",
    },
}


def get_mapping_info(class_name: str) -> Dict[str, Any]:
    """Lấy đầy đủ thông tin nhóm xử lý, tên tiếng Việt, màu sắc và hướng dẫn của lớp rác."""
    return CLASS_MAPPING_INFO.get(
        class_name,
        {
            "vi_name": class_name,
            "group": "Chưa xác định",
            "color": "#4A5568",
            "badge_bg": "#EDF2F7",
            "icon": "❓",
            "guide": "Vui lòng kiểm tra lại chất liệu rác thải.",
        },
    )


def get_waste_group(class_name: str) -> str:
    return get_mapping_info(class_name)["group"]


def get_vi_name(class_name: str) -> str:
    return get_mapping_info(class_name)["vi_name"]


def get_waste_guide(class_name: str) -> str:
    return get_mapping_info(class_name)["guide"]


if __name__ == "__main__":
    assert len(CLASS_MAPPING_INFO) == 12
    print("[Mapping] Kiểm thử OK! Đủ 12 lớp ánh xạ sang 4 nhóm xử lý.")
