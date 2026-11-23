"""
Script: extract_features.py  (TV1 - Ngày 3)
Mục đích:
  Trích xuất vector đặc trưng 1.280 chiều (MobileNetV2) cho 4 tập và lưu .npy vào data/:
      X_train_raw.npy / y_train_raw.npy   -> Train gốc (không Augmentation)
      X_train_aug.npy / y_train_aug.npy   -> Train có Augmentation
      X_val.npy       / y_val.npy         -> Validation (không tăng cường)
      X_test.npy      / y_test.npy        -> Test (không tăng cường)

Cấu trúc thư mục đầu vào (ImageFolder, mỗi lớp 1 thư mục con):
    <split-dir>/train/<class>/*.jpg
    <split-dir>/val/<class>/*.jpg
    <split-dir>/test/<class>/*.jpg
    <aug-dir>/train/<class>/*.jpg      (tuỳ chọn - bộ Train đã tăng cường của TV2)

Hai cách tạo tập "Train có Augmentation":
  A) --aug-dir <thư mục>: dùng bộ ảnh đã tăng cường của TV2. Thư mục này được coi là TOÀN BỘ tập
     Train có Augmentation (gồm ảnh gốc + các bản biến đổi).
  B) Không có --aug-dir: tự sinh ngay khi chạy từ src/augmentation.py: giữ 100% vector ảnh gốc
     và sinh thêm --aug-copies (mặc định 2) bản biến đổi cho mỗi ảnh (seed cố định = 42).

Ví dụ:
    python src/extract_features.py --split-dir D:/dataset_split --aug-dir D:/dataset_aug
    python src/extract_features.py --split-dir D:/dataset_split --aug-copies 2
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mobilenet_feature_extractor import MobileNetV2FeatureExtractor, save_feature_vectors
from src.augmentation import get_train_augmentation_transform


def load_class_names(data_dir):
    with open(os.path.join(data_dir, "class_names.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def check_split_dir(root, class_names, name):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Không tìm thấy thư mục {name}: {root}")
    found = {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    missing = [c for c in class_names if c not in found]
    extra = sorted(found - set(class_names))
    if extra:
        raise ValueError(f"[{name}] Có thư mục lớp lạ không nằm trong class_names.json: {extra}")
    if missing:
        print(f"[CẢNH BÁO] [{name}] thiếu lớp: {missing}")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser(description="TV1 - Trích xuất vector MobileNetV2 ra file .npy")
    ap.add_argument("--split-dir", default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục chứa train/ val/ test/ (mặc định: data/)")
    ap.add_argument("--aug-dir", default=None, help="(Tuỳ chọn) Thư mục chứa train/ đã tăng cường của TV2")
    ap.add_argument("--aug-copies", type=int, default=2, help="Số bản biến đổi/ảnh khi tự sinh augmentation (1-2)")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data"), help="Nơi lưu file .npy")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-aug", action="store_true", help="Bỏ qua bước Train có Augmentation")
    args = ap.parse_args()

    seed_everything(args.seed)
    class_names = load_class_names(os.path.join(PROJECT_ROOT, "data"))
    print(f"[Extract] {len(class_names)} lớp (class_names.json): {class_names}")

    for s in ("train", "val", "test"):
        check_split_dir(os.path.join(args.split_dir, s), class_names, s)

    extractor = MobileNetV2FeatureExtractor()
    t_all = time.time()
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)

    # 1-3: Train gốc, Val, Test
    X_tr, y_tr, _ = extractor.extract_from_directory(os.path.join(args.split_dir, "train"), **kw)
    save_feature_vectors(X_tr, y_tr, args.out_dir, "train_raw")
    X_va, y_va, _ = extractor.extract_from_directory(os.path.join(args.split_dir, "val"), **kw)
    save_feature_vectors(X_va, y_va, args.out_dir, "val")
    X_te, y_te, _ = extractor.extract_from_directory(os.path.join(args.split_dir, "test"), **kw)
    save_feature_vectors(X_te, y_te, args.out_dir, "test")

    # 4: Train có Augmentation
    if not args.skip_aug:
        if args.aug_dir:
            check_split_dir(os.path.join(args.aug_dir, "train"), class_names, "aug/train")
            X_au, y_au, _ = extractor.extract_from_directory(os.path.join(args.aug_dir, "train"), **kw)
            if len(y_au) <= len(y_tr):
                print(f"[CẢNH BÁO] Train aug ({len(y_au)}) không nhiều hơn Train gốc ({len(y_tr)}). "
                      f"Kiểm tra lại thư mục --aug-dir có chứa bản biến đổi không.")
            print(f"[Extract] Train aug từ TV2: {len(y_au)} mẫu (gốc {len(y_tr)})")
        else:
            print(f"[Extract] Tự sinh Augmentation: giữ {len(y_tr)} ảnh gốc + {args.aug_copies} bản biến đổi/ảnh")
            aug_tf = get_train_augmentation_transform()
            xs, ys = [X_tr], [y_tr]
            for k in range(args.aug_copies):
                torch.manual_seed(args.seed + 1 + k)
                Xk, yk, _ = extractor.extract_from_directory(
                    os.path.join(args.split_dir, "train"), transform=aug_tf, **kw)
                xs.append(Xk)
                ys.append(yk)
            X_au, y_au = np.vstack(xs), np.concatenate(ys)
        save_feature_vectors(X_au, y_au, args.out_dir, "train_aug")

    # Kiểm tra tổng thể
    print("\n" + "=" * 60)
    print("TỔNG KẾT TRÍCH XUẤT")
    print("=" * 60)
    for name in ("train_raw", "train_aug", "val", "test"):
        xp = os.path.join(args.out_dir, f"X_{name}.npy")
        if os.path.exists(xp):
            X = np.load(xp, mmap_mode="r")
            y = np.load(os.path.join(args.out_dir, f"y_{name}.npy"))
            assert X.shape[1] == 1280 and len(X) == len(y)
            assert not np.isnan(np.asarray(X)).any(), f"{name} có NaN"
            print(f"  {name:10s}: X{tuple(X.shape)}  y{tuple(y.shape)}  | số lớp có mặt: {len(np.unique(y))}/{len(class_names)}")
    print(f"Thời gian tổng: {time.time() - t_all:.1f}s | Lưu tại: {args.out_dir}")


if __name__ == "__main__":
    main()
