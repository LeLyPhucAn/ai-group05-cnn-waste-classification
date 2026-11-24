"""
Module: random_forest.py (Tầng 3 - Mô hình Học máy | Phụ trách: TV1 Phúc An & TV3 Khương)
Mục đích:
  Khởi tạo, huấn luyện, lưu và nạp mô hình RandomForestClassifier nhận đầu vào vector 1.280 chiều.
"""

import os
import sys
from pathlib import Path
from typing import Union

import joblib
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RF_MODEL_PATH, SEED


def build_random_forest(
    n_estimators: int = 100,
    max_depth=None,
    min_samples_split: int = 2,
    criterion: str = "gini",
    random_state: int = SEED,
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestClassifier:
    """Khởi tạo mô hình RandomForestClassifier theo cấu hình chỉ định."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )


def train_random_forest(model: RandomForestClassifier, X_train, y_train) -> RandomForestClassifier:
    """Huấn luyện Random Forest trên ma trận đặc trưng X_train (N, 1280) và nhãn y_train (N,)."""
    model.fit(X_train, y_train)
    return model


def save_model(
    model: RandomForestClassifier,
    filepath: Union[str, Path] = RF_MODEL_PATH,
    compress: int = 3,
) -> None:
    """Lưu mô hình đã huấn luyện ra file .joblib (mặc định nén mức 3 gọn nhẹ)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath, compress=compress)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[Random Forest] Đã lưu model: {filepath} ({size_mb:.2f} MB)")


def load_model(filepath: Union[str, Path] = RF_MODEL_PATH) -> RandomForestClassifier:
    """Nạp mô hình Random Forest từ file .joblib."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {filepath}")
    return joblib.load(filepath)
