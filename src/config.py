"""
Module: config.py (Tầng 0 - Cấu hình nền móng dùng chung toàn dự án)
Mục đích: Tập trung toàn bộ đường dẫn và hằng số kỹ thuật tại một nơi duy nhất.
"""

from pathlib import Path

# 1. Đường dẫn thư mục gốc và các thư mục thành phần
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "garbage_classification"
AUG_DATA_DIR = DATA_DIR / "augmented"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
CM_DIR = RESULTS_DIR / "confusion_matrix"
METRICS_DIR = RESULTS_DIR / "metrics"
DEMO_DIR = PROJECT_ROOT / "demo_samples"

# 2. Đường dẫn các file quan trọng
CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
RF_MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
EVAL_CSV_PATH = METRICS_DIR / "evaluation_results.csv"
TUNING_CSV_PATH = METRICS_DIR / "hyperparameter_tuning.csv"
BASELINE_CSV_PATH = METRICS_DIR / "baseline_results.csv"
CM_ANALYSIS_PATH = METRICS_DIR / "confusion_analysis.txt"
DEMO_CSV_PATH = DEMO_DIR / "real_test_cases.csv"

# 3. Hằng số kỹ thuật thống nhất toàn nhóm
SEED = 42
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FEATURE_DIM = 1280
N_AUG = 2
VALID_EXTS = (".jpg", ".jpeg", ".png")
LOW_CONFIDENCE_THRESHOLD = 55.0
