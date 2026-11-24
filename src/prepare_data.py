"""
Script Bước 1: prepare_data.py (Phụ trách: TV2 Trần Lâm)
Mục đích:
  1. Quét bộ ảnh gốc Kaggle (12 lớp), lọc bỏ ảnh hỏng và ảnh trùng lặp nội dung (MD5).
  2. Lưu danh sách 12 lớp chuẩn theo thứ tự alphabet vào data/class_names.json.
  3. Chia dữ liệu phân tầng (Stratified 70% Train - 15% Val - 15% Test, seed=42).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RAW_DATA_DIR, SEED, VALID_EXTS


def md5_of(path: Path) -> str:
    """Tính mã băm MD5 của file ảnh để phát hiện ảnh trùng nội dung."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_dataset(raw_dir: Path = RAW_DATA_DIR, out_dir: Path = DATA_DIR, seed: int = SEED) -> None:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh gốc: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chốt thứ tự 12 lớp theo bảng chữ cái
    class_names = sorted(d for d in os.listdir(raw_dir) if (raw_dir / d).is_dir())
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"[Bước 1 - Prepare] {len(class_names)} lớp:", class_names)

    # 2. Lọc ảnh lỗi và ảnh trùng MD5
    paths, labels, bad, dups = [], [], [], []
    seen = {}
    for idx, cls in enumerate(class_names):
        folder = raw_dir / cls
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(VALID_EXTS):
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

    print(f"[Bước 1 - Prepare] Hợp lệ: {len(paths)} | Lỗi: {len(bad)} | Trùng bị loại: {len(dups)}")

    # 3. Chia Stratified 70% Train / 15% Val / 15% Test
    tr_p, tmp_p, tr_y, tmp_y = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    va_p, te_p, va_y, te_y = train_test_split(tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=seed)

    for split, ps, ys in (("train", tr_p, tr_y), ("val", va_p, va_y), ("test", te_p, te_y)):
        split_root = out_dir / split
        if split_root.exists():
            shutil.rmtree(split_root)
        for p, y in zip(ps, ys):
            dst = split_root / class_names[y]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
        print(f"  -> {split:5s}: {len(ps)} ảnh")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bước 1 (TV2): Lọc ảnh và chia Train/Val/Test 70/15/15")
    ap.add_argument("--raw-dir", default=str(RAW_DATA_DIR))
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    prepare_dataset(Path(args.raw_dir), Path(args.out_dir), args.seed)
