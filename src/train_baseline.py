"""
Script: train_baseline.py  (TV1 - Ngày 3)
Mục đích:
  Huấn luyện RandomForestClassifier baseline (n_jobs=-1, random_state=42) trên các file .npy
  để kiểm chứng pipeline MobileNetV2 -> Random Forest, cho 2 kịch bản:
    - Không Augmentation (X_train_raw.npy)
    - Có Augmentation    (X_train_aug.npy)
  Đánh giá nhanh trên Validation và Test, lưu kết quả sơ bộ vào results/metrics/baseline_results.csv.

Lưu ý: KHÔNG ghi đè models/random_forest_model.joblib (đó là mô hình tối ưu của TV3).
       Model baseline lưu ở models/baseline/ (đã .gitignore).

Chạy:  python src/train_baseline.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.random_forest import build_random_forest, train_random_forest, save_model


def main(data_dir=None, out_csv=None, model_dir=None, n_estimators=100):
    data_dir = data_dir or os.path.join(PROJECT_ROOT, "data")
    out_csv = out_csv or os.path.join(PROJECT_ROOT, "results", "metrics", "baseline_results.csv")
    model_dir = model_dir or os.path.join(PROJECT_ROOT, "models", "baseline")

    load = lambda n: (np.load(os.path.join(data_dir, f"X_{n}.npy")), np.load(os.path.join(data_dir, f"y_{n}.npy")))
    X_val, y_val = load("val")
    X_test, y_test = load("test")

    rows = []
    for label, name in (("Không Augmentation", "train_raw"), ("Có Augmentation", "train_aug")):
        if not os.path.exists(os.path.join(data_dir, f"X_{name}.npy")):
            print(f"[Bỏ qua] Chưa có X_{name}.npy")
            continue
        X_tr, y_tr = load(name)
        rf = build_random_forest(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        t0 = time.time()
        train_random_forest(rf, X_tr, y_tr)
        dt = time.time() - t0
        row = {"Kịch bản": label, "Số mẫu Train": len(y_tr), "Thời gian Train (s)": round(dt, 2)}
        for split, (X, y) in (("Val", (X_val, y_val)), ("Test", (X_test, y_test))):
            p = rf.predict(X)
            row[f"{split} Accuracy (%)"] = round(accuracy_score(y, p) * 100, 2)
            row[f"{split} Macro F1 (%)"] = round(f1_score(y, p, average="macro", zero_division=0) * 100, 2)
        rows.append(row)
        save_model(rf, os.path.join(model_dir, f"rf_baseline_{name}.joblib"))
        print(f"[Baseline] {label}: {row}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))
    print(f"\nĐã lưu: {out_csv}")
    return df


if __name__ == "__main__":
    main()
