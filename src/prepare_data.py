"""
Script: prepare_data.py
Phụ trách: Thành viên 2 (Trần Lâm - Data & Augmentation)
Mục đích (Bước 1 - Ngày 1):
  - Quét bộ dữ liệu gốc Kaggle "Garbage Classification" (12 lớp).
  - Lọc bỏ ảnh lỗi/hỏng (Image.verify + convert RGB) và ảnh trùng nội dung (MD5).
  - Xuất danh sách 12 lớp chuẩn hóa ra data/class_names.json.
  - Chia phân tầng (stratified, seed=42) theo tỷ lệ Train 70% / Val 15% / Test 15%.

Chạy từ thư mục gốc dự án:
  python src/prepare_data.py --raw-dir data/raw/garbage_classification --out-dir data
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "garbage_classification"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"
SEED = 42
EXTS = (".jpg", ".jpeg", ".png")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_dataset(raw_dir: Path, out_dir: Path, seed: int = SEED) -> None:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh gốc: {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chốt thứ tự 12 lớp theo bảng chữ cái
    class_names = sorted(d for d in os.listdir(raw_dir) if (raw_dir / d).is_dir())
    class_names_path = out_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"[Prepare] {len(class_names)} lớp:", class_names)

    # 2. Lọc ảnh lỗi + ảnh trùng nội dung MD5
    paths, labels, bad, dups = [], [], [], []
    seen = {}
    for idx, cls in enumerate(class_names):
        folder = raw_dir / cls
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(EXTS):
                continue
            p = folder / fn
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.convert("RGB").load()
            except Exception as e:
                bad.append(f"{p}\t{e}")
                continue

            h = md5_of(p)
            if h in seen:
                dups.append(f"{p}\ttrùng với\t{seen[h]}")
                continue
            seen[h] = p
            paths.append(p)
            labels.append(idx)

    (out_dir / "bad_images.txt").write_text("\n".join(bad), encoding="utf-8")
    (out_dir / "duplicate_images.txt").write_text("\n".join(dups), encoding="utf-8")
    print(f"[Prepare] Ảnh hợp lệ: {len(paths)} | ảnh lỗi: {len(bad)} | ảnh trùng bị loại: {len(dups)}")

    # 3. Chia 70/15/15 phân tầng (stratify, seed=42)
    tr_p, tmp_p, tr_y, tmp_y = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    va_p, te_p, va_y, te_y = train_test_split(
        tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=seed
    )

    for split, ps, ys in (("train", tr_p, tr_y), ("val", va_p, va_y), ("test", te_p, te_y)):
        split_root = out_dir / split
        if split_root.exists():
            shutil.rmtree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)
        for p, y in zip(ps, ys):
            dst = split_root / class_names[y]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
        print(f"[Prepare] {split}: {len(ps)} ảnh -> {split_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TV2 - Lọc ảnh và chia Train/Val/Test 70/15/15")
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Thư mục chứa 12 lớp ảnh Kaggle gốc")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Thư mục lưu train/ val/ test/ và class_names.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    prepare_dataset(Path(args.raw_dir), Path(args.out_dir), args.seed)
