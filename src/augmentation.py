"""
Module: augmentation.py
Phụ trách: Thành viên 2 (Trần Lâm - Data & Augmentation)
Mục đích:
  - Tăng cường dữ liệu CHỈ áp dụng cho tập Train (Val và Test giữ nguyên 100%).
  - Giữ nguyên 100% ảnh gốc và sinh thêm N_AUG bản biến đổi (mặc định 2) cho mỗi ảnh.
  - Các phép biến đổi chuẩn hóa (khớp báo cáo & ảnh minh họa của TV2):
      + RandomHorizontalFlip(p=0.5)
      + RandomRotation(degrees=20)
      + RandomResizedCrop(224, scale=(0.7, 1.0))
      + ColorJitter(brightness=0.3, contrast=0.3)

Chạy từ thư mục gốc dự án:
  python src/augmentation.py                  # sinh data/augmented/train từ data/train
  python src/augmentation.py --demo           # xuất ảnh minh họa (1 gốc + 4 biến thể)
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

from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, to_rgb

SEED = 42
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data"
DEFAULT_AUG_DIR = PROJECT_ROOT / "data" / "augmented"
DEFAULT_DEMO_PATH = PROJECT_ROOT / "results" / "charts" / "augmentation_demo.png"
N_AUG = 2
EXTS = (".jpg", ".jpeg", ".png")


# Pipeline tăng cường trả về ảnh PIL (dùng lưu ảnh offline hoặc vẽ minh họa)
aug_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])


def get_pil_augmentation_transform() -> transforms.Compose:
    """Trả về pipeline biến đổi ảnh PIL cho tập Train."""
    return aug_transform


def get_train_augmentation_transform() -> transforms.Compose:
    """
    Trả về pipeline biến đổi đầy đủ (Augmentation + ToTensor + Normalize ImageNet)
    dùng trực tiếp khi trích xuất đặc trưng online (src/extract_features.py).
    """
    return transforms.Compose([
        aug_transform,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def generate_augmented_variants(
    image_input: Union[str, Path, Image.Image],
    num_variants: int = N_AUG,
) -> List[torch.Tensor]:
    """Tạo danh sách tensor biến thể từ 1 ảnh gốc."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Không hỗ trợ định dạng ảnh: {type(image_input)}")

    tf = get_train_augmentation_transform()
    img = to_rgb(img)
    return [tf(img).unsqueeze(0) for _ in range(num_variants)]


def build_augmented_train(
    split_dir: Union[str, Path] = DEFAULT_SPLIT_DIR,
    aug_dir: Union[str, Path] = DEFAULT_AUG_DIR,
    n_aug: int = N_AUG,
    seed: int = SEED,
) -> None:
    """Sinh tập Train đã tăng cường offline (100% ảnh gốc + n_aug bản biến đổi/ảnh)."""
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
            if not fn.lower().endswith(EXTS):
                continue
            src_path = cls_src / fn
            shutil.copy2(src_path, cls_dst / fn)
            n_orig += 1
            stem = Path(fn).stem
            with Image.open(src_path) as im:
                for k in range(1, n_aug + 1):
                    out = aug_transform(im)
                    out.save(cls_dst / f"{stem}_aug{k}.jpg", quality=95)
                    n_new += 1

    print(
        f"[Augmentation] Train gốc: {n_orig} | ảnh sinh thêm: {n_new} | "
        f"tổng sau tăng cường: {n_orig + n_new} -> {dst_root}"
    )


def make_demo(
    split_dir: Union[str, Path] = DEFAULT_SPLIT_DIR,
    out_path: Union[str, Path] = DEFAULT_DEMO_PATH,
    seed: int = SEED,
) -> None:
    """Ghép 1 ảnh gốc + 4 biến thể trên cùng 1 hình minh họa."""
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    src_root = Path(split_dir) / "train"
    classes = sorted(d for d in os.listdir(src_root) if (src_root / d).is_dir())
    if not classes:
        raise FileNotFoundError(f"Không tìm thấy thư mục lớp nào trong {src_root}")

    cls = classes[0]
    files = sorted(f for f in os.listdir(src_root / cls) if f.lower().endswith(EXTS))
    img = Image.open(src_root / cls / files[0]).convert("RGB")
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
    ap.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Thư mục chứa train/ val/ test/")
    ap.add_argument("--aug-dir", default=str(DEFAULT_AUG_DIR), help="Thư mục lưu tập Train tăng cường")
    ap.add_argument("--n-aug", type=int, default=N_AUG, help="Số bản biến đổi cho mỗi ảnh gốc")
    ap.add_argument("--demo", action="store_true", help="Xuất ảnh minh họa 1 gốc + 4 biến thể")
    ap.add_argument("--demo-out", default=str(DEFAULT_DEMO_PATH), help="Đường dẫn ảnh minh họa")
    ap.add_argument("--self-test", action="store_true", help="Kiểm thử nhanh với ảnh giả lập")
    args = ap.parse_args()

    if args.self_test:
        dummy = Image.new("RGB", (250, 250), color=(120, 80, 200))
        variants = generate_augmented_variants(dummy, num_variants=2)
        assert len(variants) == 2 and variants[0].shape == (1, 3, 224, 224)
        print("[Augmentation] Self-test thành công! Shape:", variants[0].shape)
    elif args.demo:
        make_demo(args.split_dir, args.demo_out)
    else:
        build_augmented_train(args.split_dir, args.aug_dir, args.n_aug)
