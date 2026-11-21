"""
Module: augmentation.py
Tác giả: Thành viên 2 (TV2 - Data & Augmentation) & TV1 (Nhóm trưởng)
Mục đích:
  - Cung cấp pipeline tăng cường dữ liệu (Data Augmentation) áp dụng RIÊNG cho tập Train.
  - Bao gồm các phép biến đổi hình học và quang học:
      + RandomResizedCrop (cắt ngẫu nhiên tỷ lệ 85% - 100%, resize 224x224)
      + RandomHorizontalFlip (lật ngang ngẫu nhiên p=0.5)
      + RandomRotation (xoay nhẹ +/-15 độ)
      + ColorJitter (thay đổi độ sáng, tương phản 15%)
  - Giữ nguyên 100% ảnh gốc và sinh thêm các biến thể biến đổi nhằm chống overfitting cho Random Forest.
"""

from typing import List, Union
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


def get_train_augmentation_transform():
    """
    Trả về Compose pipeline biến đổi tăng cường dữ liệu cho tập Train.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_pil_augmentation_transform():
    """
    Pipeline biến đổi chỉ trả về ảnh PIL (dùng để vẽ ảnh minh họa hoặc lưu ảnh mẫu).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15)
    ])


def generate_augmented_variants(image_input: Union[str, Path, Image.Image], num_variants: int = 1) -> List[torch.Tensor]:
    """
    Tạo ra danh sách các tensor ảnh biến thể từ 1 ảnh gốc.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Không hỗ trợ định dạng ảnh: {type(image_input)}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = get_train_augmentation_transform()
    variants = [transform(img).unsqueeze(0) for _ in range(num_variants)]
    return variants


if __name__ == "__main__":
    print("[Augmentation] Đang kiểm tra module tăng cường dữ liệu...")
    dummy = Image.new("RGB", (250, 250), color=(120, 80, 200))
    variants = generate_augmented_variants(dummy, num_variants=2)
    print(f"[Augmentation] Số biến thể sinh ra: {len(variants)} | Tensor shape: {variants[0].shape}")
    assert len(variants) == 2 and variants[0].shape == (1, 3, 224, 224)
    print("[Augmentation] Kiểm tra thành công!")
