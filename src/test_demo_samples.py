"""
Script Bước 5: test_demo_samples.py (Phụ trách: TV5 Hoàng & TV1 Phúc An)
Mục đích:
  Chạy kiểm thử hàng loạt toàn bộ ảnh trong thư mục demo_samples/ thông qua WastePredictor,
  cập nhật kết quả dự đoán và độ tin cậy vào demo_samples/real_test_cases.csv.
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEMO_CSV_PATH, DEMO_DIR
from src.predictor import WastePredictor


def main():
    if not DEMO_CSV_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy bảng kịch bản test: {DEMO_CSV_PATH}")

    predictor = WastePredictor()
    with open(DEMO_CSV_PATH, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    n_ok_class = n_ok_group = n_tested = 0
    glass_set = {"white-glass", "green-glass", "brown-glass"}

    for r in rows:
        img_path = DEMO_DIR / r["Ten_File"]
        if not img_path.exists():
            continue

        res = predictor.predict(img_path)
        pred_cls = res["class_name"]
        pred_grp = res["group"]
        true_cls = r["Lop_Thuc_Te"]
        true_grp = r["Nhom_Xu_Ly"]

        is_class_ok = (pred_cls == true_cls) or (true_cls in glass_set and pred_cls in glass_set)
        is_group_ok = (pred_grp == true_grp)

        r["Du_Doan_Lop"] = pred_cls
        r["Du_Doan_Nhom"] = pred_grp
        r["Do_Tin_Cay_Phan_Tram"] = f"{res['confidence']:.2f}"
        r["Ket_Qua_Dung_Sai"] = "Đúng" if is_class_ok else ("Đúng nhóm" if is_group_ok else "Sai")

        n_tested += 1
        n_ok_class += int(is_class_ok)
        n_ok_group += int(is_group_ok)

    with open(DEMO_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[Bước 5 - Demo Test] Đã kiểm thử {n_tested} ảnh mẫu:")
    print(f"  - Đúng loại rác (12 lớp): {n_ok_class}/{n_tested} ({n_ok_class / max(n_tested, 1) * 100:.2f}%)")
    print(f"  - Đúng nhóm xử lý (4 nhóm): {n_ok_group}/{n_tested} ({n_ok_group / max(n_tested, 1) * 100:.2f}%)")


if __name__ == "__main__":
    main()
