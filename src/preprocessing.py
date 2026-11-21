"""
Module: preprocessing.py
Tác giả: Thành viên 2 (TV2 - Data & Augmentation) & TV1 (Nhóm trưởng)
Mục đích:
  - Cung cấp pipeline tiền xử lý chuẩn hóa ảnh đầu vào cho toàn bộ hệ thống.
  - Sử dụng torchvision.transforms:
      1. Ép hệ màu sang RGB (xử lý định dạng RGBA, Grayscale, CMYK).
      2. Resize về kích thước chuẩn (224, 224) phù hợp đầu vào MobileNetV2.
      3. Chuyển tensor dạng float32 trong khoảng [0, 1].
      4. Chuẩn hóa theo thống kê ImageNet:
         mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
"""

from typing import Union
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


def get_base_transform():
    """
    Trả về Compose pipeline tiền xử lý ảnh cho tập Validation, Test và Inference trên Web App.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_and_preprocess_image(image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
    """
    Nạp ảnh và tiền xử lý hoàn chỉnh, trả về Tensor kích thước (1, 3, 224, 224).
    Tự động ép ảnh RGBA (PNG 4 kênh) hoặc ảnh xám về 3 kênh RGB.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Không hỗ trợ kiểu dữ liệu ảnh: {type(image_input)}")

    # Xử lý dứt điểm trường hợp ảnh PNG có kênh alpha
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = get_base_transform()
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor


if __name__ == "__main__":
    print("[Preprocessing] Đang kiểm tra module tiền xử lý...")
    sample_img = Image.new("RGBA", (300, 400), color=(100, 150, 200, 255))
    tensor = load_and_preprocess_image(sample_img)
    print(f"[Preprocessing] Tensor đầu ra: {tensor.shape} (Kỳ vọng: (1, 3, 224, 224))")
    assert tensor.shape == (1, 3, 224, 224), "Sai kích thước tensor!"
    print("[Preprocessing] Kiểm tra thành công!")
