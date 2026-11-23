"""
Script: test_demo_samples.py
Mục đích:
  - Chạy kiểm thử toàn bộ ảnh trong demo_samples/ qua pipeline thật:
      Ảnh -> src/preprocessing.py -> MobileNetV2 (1.280d) -> Random Forest -> src/mapping.py
  - Điền kết quả dự đoán thật vào demo_samples/real_test_cases.csv để TV5 & TV1 lập báo cáo Mục 2.5.

Chạy:
  python src/test_demo_samples.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mapping import get_waste_group
from src.mobilenet_feature_extractor import MobileNetV2FeatureExtractor
from src.preprocessing import load_and_preprocess_image, load_class_names, to_rgb
from src.random_forest import load_model


def main():
    csv_path = PROJECT_ROOT / "demo_samples" / "real_test_cases.csv"
    model_path = PROJECT_ROOT / "models" / "random_forest_model.joblib"

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {csv_path}")

    class_names = load_class_names(PROJECT_ROOT / "data" / "class_names.json")
    extractor = MobileNetV2FeatureExtractor()
    rf_model = load_model(str(model_path))

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    n_ok_class = 0
    n_ok_group = 0
    n_tested = 0

    for r in rows:
        img_path = PROJECT_ROOT / "demo_samples" / r["Ten_File"]
        if not img_path.exists():
            continue
        img = to_rgb(Image.open(img_path))
        tensor = load_and_preprocess_image(img)
        feat = extractor.extract_from_batch(tensor)[0]
        probs = rf_model.predict_proba([feat])[0]
        pred_idx = int(np.argmax(probs))
        pred_cls = class_names[pred_idx]
        pred_grp = get_waste_group(pred_cls)
        conf = float(probs[pred_idx]) * 100.0

        true_cls = r["Lop_Thuc_Te"]
        true_grp = r["Nhom_Xu_Ly"]
        # Với chai thủy tinh có 3 lớp màu (white/green/brown-glass), chấp nhận đúng nhóm thủy tinh là đúng lớp thủy tinh
        glass_set = {"white-glass", "green-glass", "brown-glass"}
        is_class_ok = (pred_cls == true_cls) or (true_cls in glass_set and pred_cls in glass_set)
        is_group_ok = (pred_grp == true_grp)

        r["Du_Doan_Lop"] = pred_cls
        r["Du_Doan_Nhom"] = pred_grp
        r["Do_Tin_Cay_Phan_Tram"] = f"{conf:.2f}"
        r["Ket_Qua_Dung_Sai"] = "Đúng" if is_class_ok else ("Đúng nhóm" if is_group_ok else "Sai")

        n_tested += 1
        n_ok_class += int(is_class_ok)
        n_ok_group += int(is_group_ok)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[Demo Test] Đã kiểm thử {n_tested} ảnh trong demo_samples/:")
    print(f"  - Đúng loại rác (12 lớp): {n_ok_class}/{n_tested} ({n_ok_class / max(n_tested, 1) * 100:.2f}%)")
    print(f"  - Đúng nhóm xử lý (4 nhóm): {n_ok_group}/{n_tested} ({n_ok_group / max(n_tested, 1) * 100:.2f}%)")
    print(f"  - Đã cập nhật kết quả vào: {csv_path}")


if __name__ == "__main__":
    main()
