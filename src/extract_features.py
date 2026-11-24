"""
Script Bước 3: extract_features.py (Phụ trách: TV1 Lê Lý Phúc An)
Mục đích:
  Trích xuất vector đặc trưng 1.280 chiều (MobileNetV2) cho 4 tập dữ liệu và lưu .npy vào data/:
    1. X_train_raw.npy / y_train_raw.npy  (Train gốc - 10.847 mẫu)
    2. X_train_aug.npy / y_train_aug.npy  (Train có Augmentation - 32.541 mẫu)
    3. X_val.npy       / y_val.npy        (Validation - 2.324 mẫu)
    4. X_test.npy      / y_test.npy       (Test - 2.325 mẫu)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation import get_train_augmentation_transform
from src.config import DATA_DIR, FEATURE_DIM, N_AUG, SEED
from src.mobilenet_feature_extractor import MobileNetV2FeatureExtractor, save_feature_vectors
from src.preprocessing import load_class_names


def check_split_dir(root: Path, class_names, split_name: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục {split_name}: {root}")
    found = {d for d in os.listdir(root) if (root / d).is_dir()}
    extra = sorted(found - set(class_names))
    if extra:
        raise ValueError(f"[{split_name}] Có thư mục lớp lạ không nằm trong class_names.json: {extra}")


def main():
    ap = argparse.ArgumentParser(description="Bước 3 (TV1): Trích xuất vector MobileNetV2 ra file .npy")
    ap.add_argument("--split-dir", default=str(DATA_DIR), help="Thư mục chứa train/ val/ test/")
    ap.add_argument("--aug-dir", default=None, help="(Tuỳ chọn) Thư mục chứa train/ đã tăng cường offline")
    ap.add_argument("--aug-copies", type=int, default=N_AUG, help="Số bản biến đổi/ảnh khi tự sinh Augmentation")
    ap.add_argument("--out-dir", default=str(DATA_DIR), help="Nơi lưu các file .npy")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-aug", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    class_names = load_class_names()
    for s in ("train", "val", "test"):
        check_split_dir(split_dir / s, class_names, s)

    extractor = MobileNetV2FeatureExtractor()
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    t0 = time.time()

    # 1. Trích xuất Train gốc, Validation, Test
    X_tr, y_tr, _ = extractor.extract_from_directory(split_dir / "train", **kw)
    save_feature_vectors(X_tr, y_tr, out_dir, "train_raw")

    X_va, y_va, _ = extractor.extract_from_directory(split_dir / "val", **kw)
    save_feature_vectors(X_va, y_va, out_dir, "val")

    X_te, y_te, _ = extractor.extract_from_directory(split_dir / "test", **kw)
    save_feature_vectors(X_te, y_te, out_dir, "test")

    # 2. Trích xuất Train có Augmentation
    if not args.skip_aug:
        if args.aug_dir:
            aug_train_dir = Path(args.aug_dir) / "train"
            check_split_dir(aug_train_dir, class_names, "aug/train")
            X_au, y_au, _ = extractor.extract_from_directory(aug_train_dir, **kw)
        else:
            print(f"[Bước 3 - Extract] Sinh Augmentation: {len(y_tr)} ảnh gốc + {args.aug_copies} biến thể/ảnh")
            aug_tf = get_train_augmentation_transform()
            xs, ys = [X_tr], [y_tr]
            for k in range(args.aug_copies):
                torch.manual_seed(args.seed + 1 + k)
                Xk, yk, _ = extractor.extract_from_directory(split_dir / "train", transform=aug_tf, **kw)
                xs.append(Xk)
                ys.append(yk)
            X_au, y_au = np.vstack(xs), np.concatenate(ys)
        save_feature_vectors(X_au, y_au, out_dir, "train_aug")

    # 3. Kiểm tra tính toàn vẹn của các file .npy vừa lưu
    print("\n" + "=" * 60)
    print("TỔNG KẾT TRÍCH XUẤT ĐẶC TRƯNG (.NPY)")
    print("=" * 60)
    for name in ("train_raw", "train_aug", "val", "test"):
        xp = out_dir / f"X_{name}.npy"
        if xp.exists():
            X = np.load(xp, mmap_mode="r")
            y = np.load(out_dir / f"y_{name}.npy")
            assert X.shape[1] == FEATURE_DIM and len(X) == len(y)
            assert not np.isnan(np.asarray(X)).any()
            print(f"  {name:10s}: X{tuple(X.shape)} | y{tuple(y.shape)} | Lớp: {len(np.unique(y))}/{len(class_names)}")
    print(f"Hoàn tất trong {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
