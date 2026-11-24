"""
Module: augmentation.py (Tầng 1 - Tăng cường dữ liệu | Phụ trách: TV2 Trần Lâm)
Mục đích:
  Áp dụng Data Augmentation CHỈ cho tập Train (giữ nguyên 100% ảnh gốc + sinh N_AUG bản biến đổi).
  Tập Validation và Test tuyệt đối giữ nguyên không tăng cường.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AUG_DATA_DIR,
    CHARTS_DIR,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    N_AUG,
    SEED,
    VALID_EXTS,
)
from src.preprocessing import to_rgb

DEFAULT_DEMO_PATH = CHARTS_DIR / "augmentation_demo.png"

# Các phép biến đổi ảnh PIL cho tập Train: Lật ngang, Xoay +-20 độ, Cắt phóng 70-100%, Đổi sáng/tương phản
aug_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])


def get_pil_augmentation_transform() -> transforms.Compose:
    """Trả về pipeline biến đổi ảnh PIL (dùng khi lưu ảnh hoặc vẽ hình minh họa)."""
    return aug_transform


def get_train_augmentation_transform() -> transforms.Compose:
    """Trả về pipeline Augmentation nối tiếp ToTensor & Normalize (dùng khi trích xuất vector)."""
    return transforms.Compose([
        aug_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def generate_augmented_variants(
    image_input: Union[str, Path, Image.Image],
    num_variants: int = N_AUG,
) -> List[torch.Tensor]:
    """Sinh danh sách num_variants tensor biến thể (1, 3, 224, 224) từ 1 ảnh gốc."""
    img = Image.open(image_input) if isinstance(image_input, (str, Path)) else image_input
    tf = get_train_augmentation_transform()
    img = to_rgb(img)
    return [tf(img).unsqueeze(0) for _ in range(num_variants)]


def build_augmented_train(
    split_dir: Union[str, Path] = DATA_DIR,
    aug_dir: Union[str, Path] = AUG_DATA_DIR,
    n_aug: int = N_AUG,
    seed: int = SEED,
) -> None:
    """Sinh thư mục ảnh Train tăng cường offline (100% ảnh gốc + n_aug ảnh biến đổi/ảnh)."""
    random.seed(seed)
    torch.manual_seed(seed)

    src_root = Path(split_dir) / "train"
    dst_root = Path(aug_dir) / "train"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục train: {src_root}")
    if dst_root.exists():
        shutil.rmtree(dst_root)

    n_orig = n_new = 0
    for cls in sorted(os.listdir(src_root)):
        cls_src = src_root / cls
        if not cls_src.is_dir():
            continue
        cls_dst = dst_root / cls
        cls_dst.mkdir(parents=True, exist_ok=True)
        for fn in sorted(os.listdir(cls_src)):
            if not fn.lower().endswith(VALID_EXTS):
                continue
            src_path = cls_src / fn
            shutil.copy2(src_path, cls_dst / fn)
            n_orig += 1
            stem = Path(fn).stem
            with Image.open(src_path) as im:
                for k in range(1, n_aug + 1):
                    aug_transform(im).save(cls_dst / f"{stem}_aug{k}.jpg", quality=95)
                    n_new += 1

    print(f"[Augmentation] Gốc: {n_orig} + Sinh thêm: {n_new} = {n_orig + n_new} ảnh -> {dst_root}")


def make_demo(
    split_dir: Union[str, Path] = DATA_DIR,
    out_path: Union[str, Path] = DEFAULT_DEMO_PATH,
    seed: int = SEED,
) -> None:
    """Xuất ảnh ghép minh họa (1 ảnh gốc + 4 biến thể tăng cường) phục vụ báo cáo."""
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    src_root = Path(split_dir) / "train"
    classes = sorted(d for d in os.listdir(src_root) if (src_root / d).is_dir())
    cls = classes[0]
    files = sorted(f for f in os.listdir(src_root / cls) if f.lower().endswith(VALID_EXTS))

    img = to_rgb(Image.open(src_root / cls / files[0]))
    imgs = [img.resize(IMAGE_SIZE)] + [aug_transform(img) for _ in range(4)]
    titles = ["Ảnh gốc", "Biến thể 1", "Biến thể 2", "Biến thể 3", "Biến thể 4"]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.suptitle(f"Minh họa Data Augmentation (lớp {cls})", fontsize=13)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("[Augmentation] Đã lưu ảnh minh họa:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TV2 - Tăng cường dữ liệu cho tập Train")
    ap.add_argument("--split-dir", default=str(DATA_DIR))
    ap.add_argument("--aug-dir", default=str(AUG_DATA_DIR))
    ap.add_argument("--n-aug", type=int, default=N_AUG)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--demo-out", default=str(DEFAULT_DEMO_PATH))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        dummy = Image.new("RGB", (250, 250), color=(120, 80, 200))
        v = generate_augmented_variants(dummy, num_variants=2)
        assert len(v) == 2 and v[0].shape == (1, 3, 224, 224)
        print("[Augmentation] Kiểm thử OK! Shape:", v[0].shape)
    elif args.demo:
        make_demo(args.split_dir, args.demo_out)
    else:
        build_augmented_train(args.split_dir, args.aug_dir, args.n_aug)
