"""
Module: evaluation.py (Tầng 3 - Đánh giá & Trực quan hóa | Phụ trách: TV3 Nguyễn Văn Khương)
Mục đích:
  - Tính toán các chỉ số: Accuracy, Macro/Weighted Precision, Recall, F1-Score.
  - Xuất báo cáo phân loại chi tiết 12 lớp (Classification Report).
  - Vẽ Ma trận nhầm lẫn (Confusion Matrix) và biểu đồ so sánh 2 kịch bản Data Augmentation.
"""

import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHARTS_DIR, CM_DIR, EVAL_CSV_PATH

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_metrics(y_true, y_pred) -> dict:
    """Tính đầy đủ các chỉ số đánh giá đa lớp (Accuracy, Macro & Weighted P/R/F1)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def get_classification_report(y_true, y_pred, class_names=None, output_dict: bool = False):
    """Tạo bảng Classification Report chi tiết cho 12 lớp."""
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
    output_path: Union[str, Path] = CM_DIR / "confusion_matrix.png",
    title: str = "Ma trận nhầm lẫn 12 lớp rác thải (Confusion Matrix)",
    normalize=None,
    figsize=(11, 9),
    dpi: int = 300,
) -> np.ndarray:
    """Vẽ và lưu Ma trận nhầm lẫn 12 lớp (số lượng mẫu hoặc tỷ lệ % chuẩn hóa)."""
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    fmt = ".2%" if normalize == "true" else "d"
    cbar_label = "Tỷ lệ chuẩn hóa (Recall)" if normalize == "true" else "Số lượng mẫu ảnh"

    sns.heatmap(
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
    print(f"[Evaluation] Đã lưu Confusion Matrix: {output_path}")
    return cm


def plot_comparison_chart(
    metrics_no_aug: dict,
    metrics_aug: dict,
    output_path: Union[str, Path] = CHARTS_DIR / "augmentation_comparison.png",
    dpi: int = 300,
) -> None:
    """Vẽ biểu đồ cột so sánh Không Augmentation vs Có Augmentation trên tập Test."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Độ chính xác (Accuracy)", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    vals_no_aug = [metrics_no_aug[k] * 100 for k in keys]
    vals_aug = [metrics_aug[k] * 100 for k in keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, vals_no_aug, width, label="Không Augmentation (Baseline)", color="#4A90E2")
    rects2 = ax.bar(x + width / 2, vals_aug, width, label="Có Augmentation (Tăng cường)", color="#50E3C2")

    ax.set_ylabel("Giá trị (%)", fontsize=12, fontweight="bold")
    ax.set_title("So sánh hiệu năng: Không Augmentation vs Có Augmentation (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 105)

    for idx, rect in enumerate(rects1):
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for idx, rect in enumerate(rects2):
        h = rect.get_height()
        diff = h - vals_no_aug[idx]
        lbl = f"{h:.2f}%\n(+{diff:.2f}%)" if diff > 0 else f"{h:.2f}%"
        ax.annotate(lbl, xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Đã lưu biểu đồ so sánh: {output_path}")


def plot_class_wise_f1(
    report_no_aug: dict,
    report_aug: dict,
    class_names,
    output_path: Union[str, Path] = CHARTS_DIR / "class_wise_f1.png",
    dpi: int = 300,
) -> None:
    """Vẽ biểu đồ ngang so sánh F1-score của từng lớp giữa 2 kịch bản."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f1_no_aug = [report_no_aug[c]["f1-score"] * 100 for c in class_names]
    f1_aug = [report_aug[c]["f1-score"] * 100 for c in class_names]
    y = np.arange(len(class_names))
    height = 0.38

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y - height / 2, f1_no_aug, height, label="Không Augmentation", color="#90CAF9")
    ax.barh(y + height / 2, f1_aug, height, label="Có Augmentation", color="#2E7D32")
    ax.set_xlabel("F1-score (%)", fontsize=12, fontweight="bold")
    ax.set_title("So sánh F1-Score chi tiết 12 lớp rác thải", fontsize=13, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Đã lưu biểu đồ F1 từng lớp: {output_path}")


def save_evaluation_results(df_results, output_path: Union[str, Path] = EVAL_CSV_PATH) -> None:
    """Lưu bảng kết quả đánh giá tổng hợp ra file CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[Evaluation] Đã lưu CSV kết quả: {output_path}")
