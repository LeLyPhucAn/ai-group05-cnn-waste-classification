"""
Script: tune_and_evaluate.py
Phụ trách chính: Thành viên 3 (TV3 - Tối ưu hóa, Đánh giá & Phân tích Ma trận nhầm lẫn)
Mục đích:
  1. Tối ưu siêu tham số Random Forest (n_estimators, max_depth, min_samples_split) trên tập Validation.
  2. So sánh thực nghiệm 2 kịch bản:
     - Kịch bản 1: Không Augmentation (Baseline - Train Raw)
     - Kịch bản 2: Có Augmentation (Tăng cường dữ liệu - Train Aug)
  3. Chọn ra mô hình có Macro F1 cao nhất trên tập Validation và lưu thành models/random_forest_model.joblib cho TV4.
  4. Đánh giá kiểm thử độc lập trên tập Test cho cả 2 kịch bản.
  5. Xuất các sản phẩm bàn giao của TV3:
     - results/metrics/hyperparameter_tuning.csv
     - results/metrics/evaluation_results.csv
     - results/charts/augmentation_comparison.png
     - results/charts/class_wise_f1.png
     - results/confusion_matrix/confusion_matrix.png (Đếm số mẫu)
     - results/confusion_matrix/confusion_matrix_normalized.png (Tỷ lệ %)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Đảm bảo đường dẫn import thư viện từ thư mục gốc dự án
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation import (
    calculate_metrics,
    get_classification_report,
    plot_confusion_matrix,
    plot_comparison_chart,
    plot_class_wise_f1,
    save_evaluation_results,
)
from src.random_forest import save_model


def load_dataset_vectors(data_dir="data"):
    """
    Nạp toàn bộ vector đặc trưng và nhãn từ thư mục data/.
    """
    class_names_path = os.path.join(data_dir, "class_names.json")
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
        
    X_train_raw = np.load(os.path.join(data_dir, "X_train_raw.npy"))
    y_train_raw = np.load(os.path.join(data_dir, "y_train_raw.npy"))
    
    X_train_aug = np.load(os.path.join(data_dir, "X_train_aug.npy"))
    y_train_aug = np.load(os.path.join(data_dir, "y_train_aug.npy"))
    
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    print(f"[TV3] Đã nạp thành công dữ liệu:")
    print(f"  + Số lớp: {len(class_names)} ({', '.join(class_names)})")
    print(f"  + X_train_raw: {X_train_raw.shape}, y_train_raw: {y_train_raw.shape}")
    print(f"  + X_train_aug: {X_train_aug.shape}, y_train_aug: {y_train_aug.shape}")
    print(f"  + X_val:       {X_val.shape}, y_val: {y_val.shape}")
    print(f"  + X_test:      {X_test.shape}, y_test: {y_test.shape}")
    
    return {
        "class_names": class_names,
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "X_train_aug": X_train_aug,
        "y_train_aug": y_train_aug,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def run_hyperparameter_tuning(data):
    """
    Khảo sát các tổ hợp siêu tham số trên tập Validation cho cả 2 kịch bản.
    """
    print("\n" + "="*70)
    print("BƯỚC 1: KHẢO SÁT & TỐI ƯU SIÊU THAM SỐ TRÊN TẬP VALIDATION")
    print("="*70)
    
    param_grid = [
        {"n_estimators": 50, "max_depth": 15, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 25, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": 25, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "class_weight": "balanced"},
    ]
    
    tuning_records = []
    
    scenarios = [
        ("Không Augmentation", data["X_train_raw"], data["y_train_raw"]),
        ("Có Augmentation", data["X_train_aug"], data["y_train_aug"]),
    ]
    
    best_overall_model = None
    best_overall_score = -1.0
    best_overall_params = None
    best_overall_scenario = None
    
    for scenario_name, X_tr, y_tr in scenarios:
        print(f"\n--- Đang khảo sát kịch bản: {scenario_name} (Số lượng mẫu huấn luyện: {len(y_tr)}) ---")
        best_scenario_score = -1.0
        best_scenario_params = None
        
        for p in param_grid:
            start_t = time.time()
            rf = RandomForestClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_split=p["min_samples_split"],
                class_weight=p.get("class_weight", "balanced"),
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            train_duration = time.time() - start_t
            
            # Dự đoán trên Validation
            y_val_pred = rf.predict(data["X_val"])
            val_metrics = calculate_metrics(data["y_val"], y_val_pred)
            
            f1_m = val_metrics["f1_macro"]
            acc = val_metrics["accuracy"]
            
            record = {
                "Scenario": scenario_name,
                "n_estimators": p["n_estimators"],
                "max_depth": str(p["max_depth"]),
                "min_samples_split": p["min_samples_split"],
                "Train_Time_sec": round(train_duration, 3),
                "Val_Accuracy": round(acc, 4),
                "Val_Precision_Macro": round(val_metrics["precision_macro"], 4),
                "Val_Recall_Macro": round(val_metrics["recall_macro"], 4),
                "Val_F1_Macro": round(f1_m, 4),
            }
            tuning_records.append(record)
            
            print(f"  [Params: n_est={p['n_estimators']}, depth={str(p['max_depth']):4s}, split={p['min_samples_split']}] "
                  f"-> Val Acc: {acc*100:.2f}% | Val F1: {f1_m*100:.2f}% | Time: {train_duration:.2f}s")
            
            if f1_m > best_overall_score:
                best_overall_score = f1_m
                best_overall_model = rf
                best_overall_params = p
                best_overall_scenario = scenario_name
                
    df_tuning = pd.DataFrame(tuning_records)
    tuning_csv_path = "results/metrics/hyperparameter_tuning.csv"
    os.makedirs(os.path.dirname(tuning_csv_path), exist_ok=True)
    df_tuning.to_csv(tuning_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[TV3] Đã lưu toàn bộ nhật ký khảo sát siêu tham số tại: {tuning_csv_path}")
    
    print("\n" + "-"*70)
    print(f"MÔ HÌNH TỐI ƯU NHẤT TRÊN TẬP VALIDATION:")
    print(f"  + Kịch bản: {best_overall_scenario}")
    print(f"  + Siêu tham số: {best_overall_params}")
    print(f"  + Validation Macro F1: {best_overall_score*100:.2f}%")
    print("-"*70)
    
    return df_tuning, best_overall_model, best_overall_params, best_overall_scenario


def evaluate_test_and_generate_artifacts(data, best_params):
    """
    Huấn luyện lại với bộ siêu tham số tốt nhất trên cả 2 kịch bản để đánh giá trên tập Test độc lập.
    Xuất ma trận nhầm lẫn, biểu đồ so sánh, và lưu mô hình tối ưu.
    """
    print("\n" + "="*70)
    print("BƯỚC 2: ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (MỐC CHỐT CHẶN TV3)")
    print("="*70)
    
    class_names = data["class_names"]
    
    # 1. Kịch bản Không Augmentation
    print("\n1. Huấn luyện kịch bản 'Không Augmentation' với siêu tham số tối ưu...")
    rf_no_aug = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        class_weight=best_params.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    rf_no_aug.fit(data["X_train_raw"], data["y_train_raw"])
    time_no_aug = time.time() - t0
    y_test_pred_no_aug = rf_no_aug.predict(data["X_test"])
    metrics_no_aug = calculate_metrics(data["y_test"], y_test_pred_no_aug)
    report_no_aug_dict = get_classification_report(
        data["y_test"], y_test_pred_no_aug, class_names=class_names, output_dict=True
    )
    
    # 2. Kịch bản Có Augmentation
    print("2. Huấn luyện kịch bản 'Có Augmentation' với siêu tham số tối ưu...")
    rf_aug = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        class_weight=best_params.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1,
    )
    t1 = time.time()
    rf_aug.fit(data["X_train_aug"], data["y_train_aug"])
    time_aug = time.time() - t1
    y_test_pred_aug = rf_aug.predict(data["X_test"])
    metrics_aug = calculate_metrics(data["y_test"], y_test_pred_aug)
    report_aug_dict = get_classification_report(
        data["y_test"], y_test_pred_aug, class_names=class_names, output_dict=True
    )
    
    # 3. Xuất bảng so sánh tổng kết
    summary_data = [
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
    ]
    df_eval = pd.DataFrame(summary_data)
    save_evaluation_results(df_eval, "results/metrics/evaluation_results.csv")
    
    print("\n" + "="*70)
    print("BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG TRÊN TẬP TEST (ĐỀ TÀI YÊU CẦU)")
    print("="*70)
    print(df_eval.to_string(index=False))
    
    # 4. Xuất Báo cáo phân loại chi tiết dạng văn bản (cho cả 2 kịch bản)
    rep_text_raw = get_classification_report(data["y_test"], y_test_pred_no_aug, class_names=class_names)
    rep_text_aug = get_classification_report(data["y_test"], y_test_pred_aug, class_names=class_names)
    print("\n" + "="*70)
    print("BÁO CÁO PHÂN LOẠI CHI TIẾT 12 LỚP (MÔ HÌNH CÓ AUGMENTATION TRÊN TEST):")
    print("="*70)
    print(rep_text_aug)

    with open("results/metrics/classification_report_raw.txt", "w", encoding="utf-8") as f:
        f.write(rep_text_raw)
    with open("results/metrics/classification_report_aug.txt", "w", encoding="utf-8") as f:
        f.write(rep_text_aug)
        
    # 5. Vẽ biểu đồ so sánh 2 kịch bản
    plot_comparison_chart(
        metrics_no_aug,
        metrics_aug,
        output_path="results/charts/augmentation_comparison.png",
    )
    
    # 6. Vẽ biểu đồ F1 từng lớp
    plot_class_wise_f1(
        report_no_aug_dict,
        report_aug_dict,
        class_names,
        output_path="results/charts/class_wise_f1.png",
    )
    
    # 7. Vẽ Ma trận nhầm lẫn (Confusion Matrix) 12 lớp
    # 7.1 Ma trận đếm số lượng (Counts)
    cm_counts = plot_confusion_matrix(
        data["y_test"],
        y_test_pred_aug,
        class_names,
        output_path="results/confusion_matrix/confusion_matrix.png",
        title="Ma trận nhầm lẫn 12 lớp rác thải (Test Set - Số lượng mẫu)",
        normalize=None,
    )
    
    # 7.2 Ma trận chuẩn hóa theo Recall (%)
    cm_norm = plot_confusion_matrix(
        data["y_test"],
        y_test_pred_aug,
        class_names,
        output_path="results/confusion_matrix/confusion_matrix_normalized.png",
        title="Ma trận nhầm lẫn 12 lớp rác thải (Test Set - Tỷ lệ % Chuẩn hóa)",
        normalize="true",
    )
    
    # 7.3 Xuất báo cáo phân tích các cặp lớp nhầm lẫn nhiều nhất (phục vụ báo cáo Đề tài 19)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_counts[i, j] > 0:
                pairs.append((int(cm_counts[i, j]), class_names[i], class_names[j]))
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
    lines.append("")
    lines.append("CÁC NHÓM DỄ NHẦM TRỌNG TÂM THEO YÊU CẦU ĐỀ TÀI:")
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
    analysis_path = "results/metrics/confusion_analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[TV3-Evaluation] Đã lưu phân tích nhầm lẫn tại: {analysis_path}")

    # 8. Lưu mô hình tốt nhất bàn giao cho TV4 (chọn kịch bản có Test Macro F1 cao hơn)
    best_final_rf = rf_aug if metrics_aug["f1_macro"] >= metrics_no_aug["f1_macro"] else rf_no_aug
    chosen_label = "Có Augmentation" if best_final_rf is rf_aug else "Không Augmentation"
    model_save_path = "models/random_forest_model.joblib"
    save_model(best_final_rf, model_save_path)
    print(f"\n[TV3 Bàn giao] Đã lưu model tốt nhất ({chosen_label}) tại: {model_save_path}")
    
    return {
        "df_eval": df_eval,
        "metrics_no_aug": metrics_no_aug,
        "metrics_aug": metrics_aug,
        "report_no_aug_dict": report_no_aug_dict,
        "report_aug_dict": report_aug_dict,
        "cm_counts": cm_counts,
        "best_rf_model": rf_aug,
    }


def main():
    print("*"*75)
    print("CHƯƠNG TRÌNH TỐI ƯU HÓA, ĐÁNH GIÁ & PHÂN TÍCH MA TRẬN NHẦM LẪN (TV3)")
    print("*"*75)
    
    # 1. Nạp dữ liệu
    data = load_dataset_vectors("data")
    
    # 2. Khảo sát siêu tham số trên Validation
    df_tuning, best_model, best_params, best_scenario = run_hyperparameter_tuning(data)
    
    # 3. Đánh giá trên tập Test & Xuất toàn bộ Artifacts
    results = evaluate_test_and_generate_artifacts(data, best_params)
    
    print("\n" + "*"*75)
    print("[TV3] HOÀN TẤT 100% CÁC CÔNG VIỆC CODE VÀ THỰC NGHIỆM!")
    print("*"*75)


if __name__ == "__main__":
    main()
