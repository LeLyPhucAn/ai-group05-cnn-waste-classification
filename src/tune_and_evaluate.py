"""
Script Bước 4B: tune_and_evaluate.py (Phụ trách: TV3 Nguyễn Văn Khương)
Mục đích:
  1. Khảo sát siêu tham số Random Forest trên tập Validation (X_val.npy).
  2. Chọn bộ tham số và kịch bản có Validation Macro F1 cao nhất (đúng chuẩn Học máy).
  3. Đánh giá độc lập cả 2 kịch bản (Không Augment vs Có Augment) trên tập Test (X_test.npy).
  4. Xuất model cuối (.joblib), bảng CSV, biểu đồ so sánh và Ma trận nhầm lẫn 12 lớp.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHARTS_DIR,
    CM_ANALYSIS_PATH,
    CM_DIR,
    DATA_DIR,
    EVAL_CSV_PATH,
    METRICS_DIR,
    RF_MODEL_PATH,
    SEED,
    TUNING_CSV_PATH,
)
from src.evaluation import (
    calculate_metrics,
    get_classification_report,
    plot_class_wise_f1,
    plot_comparison_chart,
    plot_confusion_matrix,
    save_evaluation_results,
)
from src.mobilenet_feature_extractor import load_feature_vectors
from src.preprocessing import load_class_names
from src.random_forest import build_random_forest, save_model, train_random_forest


def load_dataset_vectors(data_dir: Path = DATA_DIR) -> dict:
    """Nạp 4 bộ vector .npy và danh sách 12 lớp chuẩn."""
    class_names = load_class_names()
    X_train_raw, y_train_raw = load_feature_vectors(data_dir, "train_raw")
    X_train_aug, y_train_aug = load_feature_vectors(data_dir, "train_aug")
    X_val, y_val = load_feature_vectors(data_dir, "val")
    X_test, y_test = load_feature_vectors(data_dir, "test")

    print(f"[Bước 4B - Load] Đã nạp 12 lớp | Raw: {X_train_raw.shape} | Aug: {X_train_aug.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return {
        "class_names": class_names,
        "X_train_raw": X_train_raw, "y_train_raw": y_train_raw,
        "X_train_aug": X_train_aug, "y_train_aug": y_train_aug,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def run_hyperparameter_tuning(data: dict):
    """Khảo sát các tổ hợp siêu tham số trên tập Validation để chọn cấu hình tối ưu."""
    param_grid = [
        {"n_estimators": 50, "max_depth": 15, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 25, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": 25, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "class_weight": "balanced"},
    ]

    scenarios = [
        ("Không Augmentation", data["X_train_raw"], data["y_train_raw"]),
        ("Có Augmentation", data["X_train_aug"], data["y_train_aug"]),
    ]

    records = []
    best_val_f1 = -1.0
    best_params = None
    best_scenario = None

    for scenario_name, X_tr, y_tr in scenarios:
        print(f"\n--- Khảo sát trên Validation: {scenario_name} ({len(y_tr)} mẫu) ---")
        for p in param_grid:
            t0 = time.time()
            rf = build_random_forest(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_split=p["min_samples_split"],
                class_weight=p["class_weight"],
                random_state=SEED,
            )
            train_random_forest(rf, X_tr, y_tr)
            dt = time.time() - t0

            val_metrics = calculate_metrics(data["y_val"], rf.predict(data["X_val"]))
            f1_m = val_metrics["f1_macro"]
            acc = val_metrics["accuracy"]

            records.append({
                "Scenario": scenario_name,
                "n_estimators": p["n_estimators"],
                "max_depth": str(p["max_depth"]),
                "min_samples_split": p["min_samples_split"],
                "Train_Time_sec": round(dt, 3),
                "Val_Accuracy": round(acc, 4),
                "Val_Precision_Macro": round(val_metrics["precision_macro"], 4),
                "Val_Recall_Macro": round(val_metrics["recall_macro"], 4),
                "Val_F1_Macro": round(f1_m, 4),
            })
            print(f"  [n_est={p['n_estimators']}, depth={str(p['max_depth']):4s}, split={p['min_samples_split']}] -> Val Acc: {acc*100:.2f}% | Val F1: {f1_m*100:.2f}%")

            if f1_m > best_val_f1:
                best_val_f1 = f1_m
                best_params = p
                best_scenario = scenario_name

    df_tuning = pd.DataFrame(records)
    TUNING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_tuning.to_csv(TUNING_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[Tuning] Cấu hình tối ưu trên Validation: {best_scenario} | {best_params} (Val Macro F1 = {best_val_f1*100:.2f}%)")
    return df_tuning, best_params, best_scenario


def write_confusion_analysis(cm_counts: np.ndarray, class_names: list) -> None:
    """Xuất file phân tích chi tiết các cặp lớp dễ nhầm lẫn (nhựa, kim loại, thủy tinh...)."""
    pairs = [
        (int(cm_counts[i, j]), class_names[i], class_names[j])
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and cm_counts[i, j] > 0
    ]
    pairs.sort(reverse=True)

    lines = [
        "PHÂN TÍCH MA TRẬN NHẦM LẪN 12 LỚP RÁC THẢI (TẬP TEST ĐỘC LẬP)",
        "=" * 68,
        f"Tổng số mẫu Test: {int(cm_counts.sum())} | Dự đoán đúng: {int(np.trace(cm_counts))} | Nhầm lẫn: {int(cm_counts.sum() - np.trace(cm_counts))}",
        "",
        "TOP 15 CẶP LỚP NHẦM LẪN NHIỀU NHẤT (Nhãn thật -> Dự đoán nhầm):",
    ]
    for cnt, t_cls, p_cls in pairs[:15]:
        lines.append(f"  - {t_cls:12s} -> {p_cls:12s}: {cnt:3d} mẫu")

    lines.extend(["", "CÁC NHÓM DỄ NHẦM TRỌNG TÂM THEO YÊU CẦU ĐỀ TÀI:"])
    focus_pairs = [
        ("plastic", "metal"), ("metal", "plastic"),
        ("plastic", "white-glass"), ("white-glass", "plastic"),
        ("white-glass", "green-glass"), ("green-glass", "white-glass"),
        ("brown-glass", "green-glass"), ("brown-glass", "white-glass"),
        ("paper", "cardboard"), ("cardboard", "paper"),
    ]
    for a, b in focus_pairs:
        if a in class_names and b in class_names:
            ia, ib = class_names.index(a), class_names.index(b)
            lines.append(f"  - {a:12s} -> {b:12s}: {int(cm_counts[ia, ib]):3d} mẫu")

    CM_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CM_ANALYSIS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Evaluation] Đã lưu phân tích nhầm lẫn: {CM_ANALYSIS_PATH}")


def evaluate_test_and_generate_artifacts(data: dict, best_params: dict, best_val_scenario: str):
    """Đánh giá độc lập 2 kịch bản trên tập Test và lưu model tối ưu (chọn theo Validation)."""
    class_names = data["class_names"]

    # 1. Kịch bản Không Augmentation
    rf_no_aug = build_random_forest(**best_params, random_state=SEED)
    t0 = time.time()
    train_random_forest(rf_no_aug, data["X_train_raw"], data["y_train_raw"])
    time_no_aug = time.time() - t0
    pred_no_aug = rf_no_aug.predict(data["X_test"])
    metrics_no_aug = calculate_metrics(data["y_test"], pred_no_aug)
    rep_no_aug_dict = get_classification_report(data["y_test"], pred_no_aug, class_names, output_dict=True)

    # 2. Kịch bản Có Augmentation
    rf_aug = build_random_forest(**best_params, random_state=SEED)
    t1 = time.time()
    train_random_forest(rf_aug, data["X_train_aug"], data["y_train_aug"])
    time_aug = time.time() - t1
    pred_aug = rf_aug.predict(data["X_test"])
    metrics_aug = calculate_metrics(data["y_test"], pred_aug)
    rep_aug_dict = get_classification_report(data["y_test"], pred_aug, class_names, output_dict=True)

    # 3. Lưu bảng so sánh 2 kịch bản trên tập Test
    df_eval = pd.DataFrame([
        {
            "Kịch bản": "Không Augmentation (Baseline)",
            "Số mẫu Train": len(data["y_train_raw"]),
            "Thời gian Train (s)": round(time_no_aug, 2),
            "Test Accuracy (%)": round(metrics_no_aug["accuracy"] * 100, 2),
            "Macro Precision (%)": round(metrics_no_aug["precision_macro"] * 100, 2),
            "Macro Recall (%)": round(metrics_no_aug["recall_macro"] * 100, 2),
            "Macro F1-Score (%)": round(metrics_no_aug["f1_macro"] * 100, 2),
            "Weighted F1-Score (%)": round(metrics_no_aug["f1_weighted"] * 100, 2),
        },
        {
            "Kịch bản": "Có Augmentation (Tăng cường)",
            "Số mẫu Train": len(data["y_train_aug"]),
            "Thời gian Train (s)": round(time_aug, 2),
            "Test Accuracy (%)": round(metrics_aug["accuracy"] * 100, 2),
            "Macro Precision (%)": round(metrics_aug["precision_macro"] * 100, 2),
            "Macro Recall (%)": round(metrics_aug["recall_macro"] * 100, 2),
            "Macro F1-Score (%)": round(metrics_aug["f1_macro"] * 100, 2),
            "Weighted F1-Score (%)": round(metrics_aug["f1_weighted"] * 100, 2),
        },
    ])
    save_evaluation_results(df_eval, EVAL_CSV_PATH)
    print("\n" + df_eval.to_string(index=False))

    # 4. Lưu Classification Report chi tiết và vẽ biểu đồ / Confusion Matrix
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "classification_report_raw.txt").write_text(
        get_classification_report(data["y_test"], pred_no_aug, class_names), encoding="utf-8"
    )
    (METRICS_DIR / "classification_report_aug.txt").write_text(
        get_classification_report(data["y_test"], pred_aug, class_names), encoding="utf-8"
    )

    plot_comparison_chart(metrics_no_aug, metrics_aug, CHARTS_DIR / "augmentation_comparison.png")
    plot_class_wise_f1(rep_no_aug_dict, rep_aug_dict, class_names, CHARTS_DIR / "class_wise_f1.png")

    cm_counts = plot_confusion_matrix(
        data["y_test"], pred_aug, class_names,
        output_path=CM_DIR / "confusion_matrix.png",
        title="Ma trận nhầm lẫn 12 lớp rác thải (Test Set - Số lượng mẫu)",
        normalize=None,
    )
    plot_confusion_matrix(
        data["y_test"], pred_aug, class_names,
        output_path=CM_DIR / "confusion_matrix_normalized.png",
        title="Ma trận nhầm lẫn 12 lớp rác thải (Test Set - Tỷ lệ % Chuẩn hóa)",
        normalize="true",
    )
    write_confusion_analysis(cm_counts, class_names)

    # 5. Lưu model tốt nhất dựa trên kết quả Validation (không dùng Test để chọn model)
    best_rf = rf_aug if best_val_scenario == "Có Augmentation" else rf_no_aug
    save_model(best_rf, RF_MODEL_PATH)
    return df_eval


def main():
    data = load_dataset_vectors(DATA_DIR)
    _, best_params, best_val_scenario = run_hyperparameter_tuning(data)
    evaluate_test_and_generate_artifacts(data, best_params, best_val_scenario)


if __name__ == "__main__":
    main()
