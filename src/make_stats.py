"""
Ngày 3: kiểm tra rò rỉ dữ liệu + thống kê + biểu đồ phân bố 12 lớp.
Chạy sau prepare_data.py (và augmentation.py nếu muốn thống kê sau tăng cường).

Xuất: data_statistics.csv, results/charts/*.png
"""
import os, hashlib
import pandas as pd
import matplotlib.pyplot as plt

SPLIT_DIR = "/content/dataset_split"
AUG_DIR = "/content/dataset_aug"
EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")


def list_images(root):
    out = []
    for cls in sorted(os.listdir(root)):
        folder = os.path.join(root, cls)
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if fn.lower().endswith(EXTS):
                    out.append((cls, os.path.join(folder, fn)))
    return out


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def check_leakage():
    hashes = {}
    for s in SPLITS:
        hashes[s] = {md5_of(p) for _, p in list_images(os.path.join(SPLIT_DIR, s))}
    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(hashes[a] & hashes[b])
        print(f"Ảnh trùng giữa {a} và {b}: {n}")
        ok &= (n == 0)
    print("KẾT LUẬN:", "không rò rỉ dữ liệu" if ok else "CÓ ẢNH TRÙNG, cần xử lý!")
    return ok


def main():
    check_leakage()

    rows = {}
    for s in SPLITS:
        for cls, _ in list_images(os.path.join(SPLIT_DIR, s)):
            rows.setdefault(cls, {x: 0 for x in SPLITS})[s] += 1
    df = pd.DataFrame(rows).T[list(SPLITS)]
    df.index.name = "class"
    df["total"] = df.sum(axis=1)
    df["ty_le_%"] = (df["total"] / df["total"].sum() * 100).round(2)

    aug_train = os.path.join(AUG_DIR, "train")
    if os.path.isdir(aug_train):
        cnt = {}
        for cls, _ in list_images(aug_train):
            cnt[cls] = cnt.get(cls, 0) + 1
        df["train_sau_tang_cuong"] = pd.Series(cnt)

    df.to_csv("data_statistics.csv", encoding="utf-8-sig")
    print(df)
    print("Tổng:", df[list(SPLITS)].sum().to_dict(), "| tổng ảnh:", int(df["total"].sum()))

    os.makedirs("results/charts", exist_ok=True)

    # Biểu đồ 1: tổng số ảnh mỗi lớp
    d = df.sort_values("total", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(d.index, d["total"], color="#2b7bba")
    ax.bar_label(bars, fontsize=11, padding=3)
    ax.set_title("Phân bố số lượng ảnh theo 12 lớp", fontsize=14)
    ax.set_ylabel("Số ảnh"); ax.set_xlabel("Lớp")
    plt.xticks(rotation=35, ha="right", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/charts/class_distribution.png", dpi=150)
    plt.close()

    # Biểu đồ 2: số ảnh Train/Val/Test theo lớp
    ax = df[list(SPLITS)].loc[d.index].plot(kind="bar", figsize=(13, 6), width=0.8)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8, padding=2)
    ax.set_title("Số ảnh Train/Val/Test theo từng lớp", fontsize=14)
    ax.set_ylabel("Số ảnh")
    plt.xticks(rotation=35, ha="right", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/charts/split_distribution.png", dpi=150)
    plt.close()
    print("Đã lưu biểu đồ vào results/charts/")


if __name__ == "__main__":
    main()
