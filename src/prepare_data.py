"""
Bước 1 (Ngày 1): lọc ảnh lỗi, loại ảnh trùng, tạo class_names.json,
chia Train 70% / Val 15% / Test 15% (stratify, seed=42).

Chạy:  python prepare_data.py
"""
import os, json, shutil, hashlib
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = "/content/data/garbage_classification"   # thư mục chứa 12 thư mục lớp
OUT_DIR = "/content/dataset_split"                 # nơi lưu bộ đã chia
SEED = 42
EXTS = (".jpg", ".jpeg", ".png")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. Chốt thứ tự 12 lớp
    class_names = sorted(d for d in os.listdir(RAW_DIR)
                         if os.path.isdir(os.path.join(RAW_DIR, d)))
    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"{len(class_names)} lớp:", class_names)

    # 2. Lọc ảnh lỗi + ảnh trùng (trùng hoàn toàn nội dung file)
    paths, labels, bad, dups = [], [], [], []
    seen = {}
    for idx, c in enumerate(class_names):
        folder = os.path.join(RAW_DIR, c)
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            if not fn.lower().endswith(EXTS):
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:      # mở lại để chắc chắn đọc được
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

    with open("bad_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(bad))
    with open("duplicate_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dups))
    print(f"Ảnh hợp lệ giữ lại: {len(paths)} | ảnh lỗi: {len(bad)} | ảnh trùng bị loại: {len(dups)}")

    # 3. Chia 70/15/15 theo từng lớp (stratify, seed=42)
    tr_p, tmp_p, tr_y, tmp_y = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    va_p, te_p, va_y, te_y = train_test_split(
        tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=SEED)

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for split, ps, ys in (("train", tr_p, tr_y), ("val", va_p, va_y),
                          ("test", te_p, te_y)):
        for p, y in zip(ps, ys):
            dst = os.path.join(OUT_DIR, split, class_names[y])
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
        print(f"{split}: {len(ps)} ảnh")
    print("Xong. Bộ dữ liệu đã chia nằm ở:", OUT_DIR)


if __name__ == "__main__":
    main()
