"""
Module: preprocessing.py
Phụ trách: Thành viên 2 (Trần Lâm - Data & Augmentation) & Thành viên 1 (Phúc An)
Mục đích:
  - Tiền xử lý ảnh chuẩn cho toàn nhóm (PyTorch / torchvision):
      1. Ép hệ màu RGB (xử lý ảnh RGBA/PNG 4 kênh, ảnh xám, ảnh palette).
      2. Resize về kích thước (224, 224) phù hợp đầu vào MobileNetV2.
      3. Chuyển thành Tensor [0, 1].
      4. Chuẩn hóa theo mean/std của ImageNet.
  - Cung cấp các hàm dùng chung cho TV1 (trích xuất), TV2 (augmentation), và TV4 (Streamlit app).
"""

import json
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "data" / "class_names.json"

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_rgb(img: Image.Image) -> Image.Image:
    """Ép ảnh về hệ màu RGB (xử lý ảnh RGBA, ảnh xám, ảnh palette)."""
    return img.convert("RGB") if img.mode != "RGB" else img


# Pipeline chuẩn dùng chung cho Train (sau tăng cường), Val, Test và suy luận (App)
eval_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_base_transform() -> transforms.Compose:
    """Trả về pipeline tiền xử lý chuẩn (tương thích với mobilenet_feature_extractor.py)."""
    return eval_transform


def load_class_names(path: Union[str, Path] = DEFAULT_CLASS_NAMES_PATH) -> List[str]:
    """Nạp danh sách 12 tên lớp chuẩn hóa từ file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_dataset(split_dir: Union[str, Path]) -> datasets.ImageFolder:
    """
    Tạo ImageFolder dataset từ thư mục đã chia, ví dụ make_dataset('data/val').
    ImageFolder sắp tên lớp theo bảng chữ cái, trùng với data/class_names.json.
    """
    return datasets.ImageFolder(str(split_dir), transform=eval_transform)


def load_and_preprocess_image(image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
    """
    Nạp ảnh và tiền xử lý hoàn chỉnh, trả về Tensor kích thước (1, 3, 224, 224).
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Không hỗ trợ kiểu dữ liệu ảnh: {type(image_input)}")

    tensor = eval_transform(to_rgb(img)).unsqueeze(0)
    return tensor


if __name__ == "__main__":
    print("Resize:", IMAGE_SIZE, "| mean:", IMAGENET_MEAN, "| std:", IMAGENET_STD)
    sample_img = Image.new("RGBA", (300, 400), color=(100, 150, 200, 255))
    tensor = load_and_preprocess_image(sample_img)
    assert tensor.shape == (1, 3, 224, 224), f"Sai kích thước tensor: {tensor.shape}"
    print("[Preprocessing] Tensor đầu ra:", tensor.shape, "-> Kiểm tra thành công!")
