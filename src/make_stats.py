"""
Script: make_stats.py
Phụ trách: Thành viên 2 (Trần Lâm - Data & Augmentation)
Mục đích (Ngày 3):
  - Kiểm tra rò rỉ dữ liệu (Data Leakage) bằng mã băm MD5 giữa Train / Val / Test.
  - Thống kê số lượng ảnh từng lớp ra results/charts/data_statistics.csv.
  - Vẽ 2 biểu đồ:
      + results/charts/class_distribution.png  (Phân bố tổng số ảnh 12 lớp)
      + results/charts/split_distribution.png  (Số ảnh Train/Val/Test từng lớp)

Chạy từ thư mục gốc dự án:
  python src/make_stats.py --split-dir data --aug-dir data/augmented
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data"
DEFAULT_AUG_DIR = PROJECT_ROOT / "data" / "augmented"
DEFAULT_CHARTS_DIR = PROJECT_ROOT / "results" / "charts"
EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")


def list_images(root: Path) -> List[Tuple[str, Path]]:
    out = []
    if not root.is_dir():
        return out
    for cls in sorted(os.listdir(root)):
        folder = root / cls
        if folder.is_dir():
            for fn in os.listdir(folder):
                if fn.lower().endswith(EXTS):
                    out.append((cls, folder / fn))
    return out


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_leakage(split_dir: Path) -> bool:
    hashes = {s: {md5_of(p) for _, p in list_images(split_dir / s)} for s in SPLITS}
    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(hashes[a] & hashes[b])
        print(f"[Leakage Check] Ảnh trùng giữa {a} và {b}: {n}")
        ok &= (n == 0)
    print("[Leakage Check] KẾT LUẬN:", "KHÔNG rò rỉ dữ liệu!" if ok else "CÓ ẢNH TRÙNG, cần xử lý!")
    return ok


def generate_stats(split_dir: Path, aug_dir: Path, charts_dir: Path) -> pd.DataFrame:
    check_leakage(split_dir)

    rows = {}
    for s in SPLITS:
        for cls, _ in list_images(split_dir / s):
            rows.setdefault(cls, {x: 0 for x in SPLITS})[s] += 1

    if not rows:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong {split_dir}/{{train,val,test}}")

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

    charts_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = charts_dir / "data_statistics.csv"
    df.to_csv(stats_csv, encoding="utf-8-sig")
    print(df)
    print("[Stats] Tổng:", df[list(SPLITS)].sum().to_dict(), "| tổng ảnh:", int(df["total"].sum()))

    # Biểu đồ 1: tổng số ảnh mỗi lớp
    import matplotlib.pyplot as plt

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

    # Biểu đồ 2: số ảnh Train/Val/Test theo lớp
    ax = df[list(SPLITS)].loc[d.index].plot(kind="bar", figsize=(13, 6), width=0.8)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8, padding=2)
    ax.set_title("Số ảnh Train/Val/Test theo từng lớp", fontsize=14)
    ax.set_ylabel("Số ảnh")
    plt.xticks(rotation=35, ha="right", fontsize=11)
    plt.tight_layout()
    plt.savefig(charts_dir / "split_distribution.png", dpi=150)
    plt.close()
    print(f"[Stats] Đã lưu thống kê & biểu đồ vào: {charts_dir}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TV2 - Kiểm tra rò rỉ dữ liệu & vẽ biểu đồ phân bố 12 lớp")
    ap.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Thư mục chứa train/ val/ test/")
    ap.add_argument("--aug-dir", default=str(DEFAULT_AUG_DIR), help="Thư mục chứa train/ sau tăng cường (tuỳ chọn)")
    ap.add_argument("--charts-dir", default=str(DEFAULT_CHARTS_DIR), help="Thư mục lưu CSV và biểu đồ")
    args = ap.parse_args()
    generate_stats(Path(args.split_dir), Path(args.aug_dir), Path(args.charts_dir))
