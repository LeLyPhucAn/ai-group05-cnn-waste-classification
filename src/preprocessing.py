"""
Module: preprocessing.py (Tầng 1 - Tiền xử lý dữ liệu | Phụ trách: TV2 Trần Lâm)
Mục đích:
  Chuẩn hóa ảnh đầu vào đồng nhất cho cả quá trình huấn luyện và suy luận trên App:
  Ép hệ màu RGB -> Resize (224, 224) -> ToTensor [0, 1] -> Normalize theo ImageNet.
"""

import json
import sys
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES_PATH, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


def to_rgb(img: Image.Image) -> Image.Image:
    """Ép ảnh về hệ màu 3 kênh RGB (xử lý ảnh PNG 4 kênh RGBA hoặc ảnh xám)."""
    return img.convert("RGB") if img.mode != "RGB" else img


# Pipeline tiền xử lý chuẩn dùng cho tập Train gốc, Val, Test và suy luận trên App
eval_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_base_transform() -> transforms.Compose:
    """Trả về pipeline tiền xử lý chuẩn ImageNet."""
    return eval_transform


def load_class_names(path: Union[str, Path] = CLASS_NAMES_PATH) -> List[str]:
    """Đọc danh sách 12 lớp rác chuẩn hóa (thứ tự alphabet 0..11) từ class_names.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_dataset(split_dir: Union[str, Path]) -> datasets.ImageFolder:
    """Tạo đối tượng ImageFolder từ thư mục tập dữ liệu (ví dụ: data/train, data/val)."""
    return datasets.ImageFolder(str(split_dir), transform=eval_transform)


def load_and_preprocess_image(image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
    """Đọc 1 ảnh (đường dẫn hoặc PIL Image) và trả về Tensor (1, 3, 224, 224)."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Không hỗ trợ kiểu dữ liệu ảnh: {type(image_input)}")

    return eval_transform(to_rgb(img)).unsqueeze(0)


if __name__ == "__main__":
    sample = Image.new("RGBA", (300, 400), color=(100, 150, 200, 255))
    out = load_and_preprocess_image(sample)
    assert out.shape == (1, 3, 224, 224)
    print(f"[Preprocessing] Kiểm thử OK! Tensor shape: {out.shape}")
