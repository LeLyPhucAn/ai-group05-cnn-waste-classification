"""
Module: random_forest.py
Mục đích:
  - Cung cấp lớp bọc huấn luyện mô hình Random Forest từ vector đặc trưng 1.280 chiều của MobileNetV2.
  - Hỗ trợ nạp và lưu mô hình (.joblib).
  - Phục vụ việc tái sử dụng giữa các thành viên TV1, TV3 và TV4.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    criterion="gini",
    random_state=42,
    n_jobs=-1,
    **kwargs
):
    """
    Khởi tạo mô hình RandomForestClassifier theo quy chuẩn kỹ thuật của đề tài.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    return model


def train_random_forest(model, X_train, y_train):
    """
    Huấn luyện mô hình Random Forest trên ma trận đặc trưng X_train và nhãn y_train.
    """
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath="models/random_forest_model.joblib"):
    """
    Lưu mô hình đã huấn luyện ra file .joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[Random Forest] Đã lưu mô hình thành công tại: {filepath}")


def load_model(filepath="models/random_forest_model.joblib"):
    """
    Nạp mô hình đã lưu từ file .joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại {filepath}")
    model = joblib.load(filepath)
    print(f"[Random Forest] Đã tải mô hình thành công từ: {filepath}")
    return model
