"""
Module: evaluation.py
Phụ trách: Thành viên 3 (TV3 - Tối ưu hóa, Đánh giá & Phân tích Ma trận nhầm lẫn)
Mục đích:
  - Cung cấp các hàm đo lường chỉ số đánh giá (Accuracy, Precision, Recall, Macro F1, Weighted F1).
  - Xuất báo cáo phân loại chi tiết (Classification Report).
  - Vẽ và xuất hình Ma trận nhầm lẫn (Confusion Matrix) 12 lớp chất lượng cao.
  - Vẽ biểu đồ so sánh 2 kịch bản (Không Augmentation vs Có Augmentation).
  - Xuất kết quả đánh giá ra định dạng CSV.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Thiết lập phong cách hiển thị biểu đồ
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_metrics(y_true, y_pred):
    """
    Tính toán các chỉ số đánh giá tổng quát cho bài toán phân loại đa lớp.
    
    Args:
        y_true (array-like): Nhãn thực tế (ground truth).
        y_pred (array-like): Nhãn dự đoán từ mô hình.
        
    Returns:
        dict: Chứa Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Macro/Weighted).
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
    }


def get_classification_report(y_true, y_pred, class_names=None, output_dict=False):
    """
    Tạo báo cáo phân loại chi tiết từng lớp theo scikit-learn.
    
    Args:
        y_true (array-like): Nhãn thực tế.
        y_pred (array-like): Nhãn dự đoán.
        class_names (list, optional): Danh sách tên các lớp.
        output_dict (bool): Nếu True, trả về dictionary; nếu False, trả về string.
        
    Returns:
        str hoặc dict: Nội dung báo cáo phân loại.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=output_dict,
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path="results/confusion_matrix/confusion_matrix.png",
    title="Ma trận nhầm lẫn 12 lớp rác thải (Confusion Matrix)",
    normalize=None,
    figsize=(11, 9),
    dpi=300,
):
    """
    Vẽ và lưu biểu đồ ma trận nhầm lẫn (Confusion Matrix) 12 lớp bằng Seaborn heatmap.
    
    Args:
        y_true (array-like): Nhãn thực tế.
        y_pred (array-like): Nhãn dự đoán.
        class_names (list): Danh sách 12 tên lớp.
        output_path (str): Đường dẫn lưu file ảnh.
        title (str): Tiêu đề biểu đồ.
        normalize (str, optional): 'true' để chuẩn hóa theo hàng, 'pred' theo cột, None giữ nguyên số lượng đếm.
        figsize (tuple): Kích thước hình vẽ (width, height).
        dpi (int): Độ phân giải ảnh xuất ra.
        
    Returns:
        np.ndarray: Ma trận nhầm lẫn số học.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    if normalize == "true":
        fmt = ".2%"
        vmax = 1.0
        vmin = 0.0
        cbar_label = "Tỷ lệ chuẩn hóa (Recall)"
    else:
        fmt = "d"
        vmin = 0
        vmax = int(cm.max())
        cbar_label = "Số lượng mẫu ảnh"
        
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        linewidths=0.5,
        linecolor="#e0e0e0",
        square=True,
    )
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Nhãn thực tế (True Ground Truth)", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[TV3-Evaluation] Đã lưu Confusion Matrix tại: {output_path}")
    return cm


def plot_comparison_chart(
    metrics_no_aug,
    metrics_aug,
    output_path="results/charts/augmentation_comparison.png",
    dpi=300,
):
    """
    Vẽ biểu đồ cột so sánh hiệu năng giữa kịch bản 'Không Augmentation' và 'Có Augmentation'.
    
    Args:
        metrics_no_aug (dict): Kết quả metrics của kịch bản Không Augmentation.
        metrics_aug (dict): Kết quả metrics của kịch bản Có Augmentation.
        output_path (str): Đường dẫn lưu ảnh.
        dpi (int): Độ phân giải ảnh.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labels = ["Độ chính xác (Accuracy)", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    
    vals_no_aug = [metrics_no_aug[k] * 100 for k in keys]
    vals_aug = [metrics_aug[k] * 100 for k in keys]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, vals_no_aug, width, label="Không Augmentation (Baseline)", color="#4A90E2", edgecolor="white")
    rects2 = ax.bar(x + width/2, vals_aug, width, label="Có Augmentation (Tăng cường dữ liệu)", color="#50E3C2", edgecolor="white")
    
    ax.set_ylabel("Giá trị (%)", fontsize=12, fontweight="bold")
    ax.set_title("So sánh hiệu năng: Không Augmentation vs Có Augmentation trên tập Test", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 105)
    
    def autolabel(rects, is_aug=False, base_vals=None):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            label = f"{height:.2f}%"
            if is_aug and base_vals is not None:
                diff = height - base_vals[idx]
                if diff > 0:
                    label += f"\n(+{diff:.2f}%)"
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if is_aug else "normal",
            )
            
    autolabel(rects1)
    autolabel(rects2, is_aug=True, base_vals=vals_no_aug)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[TV3-Evaluation] Đã lưu biểu đồ so sánh Augmentation tại: {output_path}")


def plot_class_wise_f1(
    report_no_aug,
    report_aug,
    class_names,
    output_path="results/charts/class_wise_f1.png",
    dpi=300,
):
    """
    Vẽ biểu đồ F1-score chi tiết của từng lớp trong 12 lớp rác thải để phân tích sâu lớp nào cải thiện tốt nhất.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    f1_no_aug = [report_no_aug[c]["f1-score"] * 100 for c in class_names]
    f1_aug = [report_aug[c]["f1-score"] * 100 for c in class_names]
    
    y = np.arange(len(class_names))
    height = 0.38
    
    fig, ax = plt.subplots(figsize=(10, 8))
    rects1 = ax.barh(y - height/2, f1_no_aug, height, label="Không Augmentation", color="#90CAF9", edgecolor="white")
    rects2 = ax.barh(y + height/2, f1_aug, height, label="Có Augmentation", color="#2E7D32", edgecolor="white")
    
    ax.set_xlabel("F1-score (%)", fontsize=12, fontweight="bold")
    ax.set_title("So sánh F1-Score chi tiết từng lớp rác thải (12 lớp)", fontsize=13, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[TV3-Evaluation] Đã lưu biểu đồ F1 từng lớp tại: {output_path}")


def save_evaluation_results(df_results, output_path="results/metrics/evaluation_results.csv"):
    """
    Lưu bảng tổng hợp kết quả đánh giá ra file CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[TV3-Evaluation] Đã lưu kết quả đánh giá CSV tại: {output_path}")


if __name__ == "__main__":
    print("[TV3-Evaluation] Chạy thử nghiệm tự kiểm tra (Self-test) evaluation.py với dữ liệu giả lập...")
    
    # 1. Tải danh sách lớp
    class_names_path = os.path.join("data", "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = [f"Class_{i}" for i in range(12)]
        
    np.random.seed(42)
    n_samples = 380
    y_true_mock = np.random.randint(0, len(classes), size=n_samples)
    
    # Tạo dự đoán có tỉ lệ chính xác khoảng 82% và nhầm lẫn nhẹ giữa plastic (8) và metal (6)
    y_pred_mock = y_true_mock.copy()
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.18), replace=False)
    for idx in noise_indices:
        if y_true_mock[idx] == 8: # plastic -> nhầm sang metal (6)
            y_pred_mock[idx] = 6
        elif y_true_mock[idx] == 6: # metal -> nhầm sang plastic (8)
            y_pred_mock[idx] = 8
        elif y_true_mock[idx] in [2, 5, 11]: # glass group confusion
            y_pred_mock[idx] = np.random.choice([2, 5, 11])
        else:
            y_pred_mock[idx] = np.random.randint(0, len(classes))
            
    metrics = calculate_metrics(y_true_mock, y_pred_mock)
    print("\nKết quả chỉ số cơ bản:", json.dumps(metrics, indent=2))
    
    print("\nBáo cáo phân loại (Classification Report):")
    rep = get_classification_report(y_true_mock, y_pred_mock, class_names=classes)
    print(rep)
    
    test_cm_path = os.path.join("results", "confusion_matrix", "test_self_check_cm.png")
    plot_confusion_matrix(y_true_mock, y_pred_mock, classes, output_path=test_cm_path)
    print("Kiểm thử module evaluation.py thành công 100%!")
