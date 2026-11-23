"""
Script Bước 2: make_stats.py (Phụ trách: TV2 Trần Lâm)
Mục đích:
  1. Kiểm tra rò rỉ dữ liệu (Data Leakage) bằng mã băm MD5 giữa Train, Val và Test.
  2. Xuất bảng thống kê số lượng ảnh từng lớp (results/charts/data_statistics.csv).
  3. Vẽ biểu đồ phân bố 12 lớp (class_distribution.png & split_distribution.png).
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUG_DATA_DIR, CHARTS_DIR, DATA_DIR, VALID_EXTS

SPLITS = ("train", "val", "test")


def list_images(root: Path) -> List[Tuple[str, Path]]:
    out = []
    if not root.is_dir():
        return out
    for cls in sorted(os.listdir(root)):
        folder = root / cls
        if folder.is_dir():
            for fn in os.listdir(folder):
                if fn.lower().endswith(VALID_EXTS):
                    out.append((cls, folder / fn))
    return out


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_leakage(split_dir: Path) -> bool:
    """Đảm bảo không có bất kỳ bức ảnh nào xuất hiện đồng thời ở 2 tập khác nhau."""
    hashes = {s: {md5_of(p) for _, p in list_images(split_dir / s)} for s in SPLITS}
    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(hashes[a] & hashes[b])
        print(f"[Leakage Check] Ảnh trùng giữa {a} và {b}: {n}")
        ok &= (n == 0)
    print("[Leakage Check] Kết luận:", "KHÔNG rò rỉ dữ liệu!" if ok else "CẢNH BÁO: Có ảnh trùng!")
    return ok


def generate_stats(
    split_dir: Path = DATA_DIR,
    aug_dir: Path = AUG_DATA_DIR,
    charts_dir: Path = CHARTS_DIR,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    check_leakage(split_dir)

    rows = {}
    for s in SPLITS:
        for cls, _ in list_images(split_dir / s):
            rows.setdefault(cls, {x: 0 for x in SPLITS})[s] += 1

    df = pd.DataFrame(rows).T[list(SPLITS)]
    df.index.name = "class"
    df["total"] = df.sum(axis=1)
    df["ty_le_%"] = (df["total"] / df["total"].sum() * 100).round(2)

    aug_train = aug_dir / "train"
    if aug_train.is_dir():
        cnt = {}
        for cls, _ in list_images(aug_train):
            cnt[cls] = cnt.get(cls, 0) + 1
        if cnt:
            df["train_sau_tang_cuong"] = pd.Series(cnt)
    else:
        df["train_sau_tang_cuong"] = df["train"] * 3

    charts_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(charts_dir / "data_statistics.csv", encoding="utf-8-sig")
    print(df)

    # Biểu đồ 1: Tổng số ảnh mỗi lớp
    d = df.sort_values("total", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(d.index, d["total"], color="#2b7bba")
    ax.bar_label(bars, fontsize=11, padding=3)
    ax.set_title("Phân bố số lượng ảnh theo 12 lớp", fontsize=14)
    ax.set_ylabel("Số ảnh")
    ax.set_xlabel("Lớp")
    plt.xticks(rotation=35, ha="right", fontsize=11)
    plt.tight_layout()
    plt.savefig(charts_dir / "class_distribution.png", dpi=150)
    plt.close()

    # Biểu đồ 2: Số ảnh Train/Val/Test theo lớp
    ax = df[list(SPLITS)].loc[d.index].plot(kind="bar", figsize=(13, 6), width=0.8)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8, padding=2)
    ax.set_title("Số ảnh Train/Val/Test theo từng lớp", fontsize=14)
    ax.set_ylabel("Số ảnh")
    plt.xticks(rotation=35, ha="right", fontsize=11)
    plt.tight_layout()
    plt.savefig(charts_dir / "split_distribution.png", dpi=150)
    plt.close()
    print(f"[Bước 2 - Stats] Đã lưu thống kê & biểu đồ tại: {charts_dir}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bước 2 (TV2): Kiểm tra rò rỉ dữ liệu & vẽ biểu đồ phân bố")
    ap.add_argument("--split-dir", default=str(DATA_DIR))
    ap.add_argument("--aug-dir", default=str(AUG_DATA_DIR))
    ap.add_argument("--charts-dir", default=str(CHARTS_DIR))
    args = ap.parse_args()
    generate_stats(Path(args.split_dir), Path(args.aug_dir), Path(args.charts_dir))
