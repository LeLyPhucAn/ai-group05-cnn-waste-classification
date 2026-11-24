"""
Script Bước 4A: train_baseline.py (Phụ trách: TV1 Lê Lý Phúc An)
Mục đích:
  Huấn luyện nhanh RandomForestClassifier baseline (100 cây, mặc định) trên các file .npy
  để kiểm chứng luồng MobileNetV2 -> Random Forest trước khi TV3 tinh chỉnh siêu tham số.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASELINE_CSV_PATH, BASELINE_MODELS_DIR, DATA_DIR, SEED
from src.mobilenet_feature_extractor import load_feature_vectors
from src.random_forest import build_random_forest, save_model, train_random_forest


def main(
    data_dir: Path = DATA_DIR,
    out_csv: Path = BASELINE_CSV_PATH,
    model_dir: Path = BASELINE_MODELS_DIR,
    n_estimators: int = 100,
) -> pd.DataFrame:
    data_dir, out_csv, model_dir = Path(data_dir), Path(out_csv), Path(model_dir)
    X_val, y_val = load_feature_vectors(data_dir, "val")
    X_test, y_test = load_feature_vectors(data_dir, "test")

    rows = []
    for label, prefix in (("Không Augmentation", "train_raw"), ("Có Augmentation", "train_aug")):
        if not (data_dir / f"X_{prefix}.npy").exists():
            continue
        X_tr, y_tr = load_feature_vectors(data_dir, prefix)
        rf = build_random_forest(n_estimators=n_estimators, random_state=SEED, n_jobs=-1)

        t0 = time.time()
        train_random_forest(rf, X_tr, y_tr)
        dt = time.time() - t0

        row = {"Kịch bản": label, "Số mẫu Train": len(y_tr), "Thời gian Train (s)": round(dt, 2)}
        for split_name, (X, y) in (("Val", (X_val, y_val)), ("Test", (X_test, y_test))):
            pred = rf.predict(X)
            row[f"{split_name} Accuracy (%)"] = round(accuracy_score(y, pred) * 100, 2)
            row[f"{split_name} Macro F1 (%)"] = round(f1_score(y, pred, average="macro", zero_division=0) * 100, 2)

        rows.append(row)
        save_model(rf, model_dir / f"rf_baseline_{prefix}.joblib")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
