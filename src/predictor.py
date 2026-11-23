"""
Module: predictor.py (Tầng 4 - Cầu nối Suy luận | Phụ trách: TV1 Phúc An & TV4 Quí)
Mục đích:
  Đóng gói trọn vẹn luồng nhận diện 1 bức ảnh (từ Tiền xử lý -> MobileNetV2 -> Random Forest -> Ánh xạ 4 nhóm)
  thành một lớp WastePredictor duy nhất, giúp tách biệt hoàn toàn logic AI khỏi giao diện Streamlit.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES_PATH, LOW_CONFIDENCE_THRESHOLD, RF_MODEL_PATH
from src.mapping import get_mapping_info, get_vi_name
from src.mobilenet_feature_extractor import MobileNetV2FeatureExtractor
from src.preprocessing import load_class_names
from src.random_forest import load_model


class WastePredictor:
    """Bộ dự đoán end-to-end: Nhận ảnh đầu vào -> Trả về kết quả phân loại chuẩn hóa."""

    def __init__(
        self,
        model_path: Union[str, Path] = RF_MODEL_PATH,
        class_names_path: Union[str, Path] = CLASS_NAMES_PATH,
    ):
        self.class_names = load_class_names(class_names_path)
        self.extractor = MobileNetV2FeatureExtractor()
        self.rf_model = load_model(model_path)

    def predict(self, image_input: Union[str, Path, Image.Image], top_k: int = 5) -> Dict[str, Any]:
        """
        Phân loại 1 ảnh rác thải và trả về dictionary kết quả đầy đủ cho giao diện hoặc script kiểm thử.
        """
        # 1. Tiền xử lý & trích xuất vector 1.280 chiều qua MobileNetV2
        feat_vector = self.extractor.extract_from_image(image_input)

        # 2. Dự đoán xác suất 12 lớp bằng Random Forest (ánh xạ chuẩn qua rf_model.classes_)
        probs = self.rf_model.predict_proba([feat_vector])[0]
        model_classes = [int(c) for c in self.rf_model.classes_]

        best_pos = int(np.argmax(probs))
        class_id = model_classes[best_pos]
        pred_class = self.class_names[class_id]
        confidence = float(probs[best_pos]) * 100.0

        # 3. Ánh xạ sang 4 nhóm xử lý & thông tin hiển thị
        info = get_mapping_info(pred_class)

        # 4. Lấy Top-K lớp có xác suất cao nhất
        top_positions = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [
            {
                "class_id": model_classes[pos],
                "class_name": self.class_names[model_classes[pos]],
                "vi_name": get_vi_name(self.class_names[model_classes[pos]]),
                "prob": float(probs[pos]) * 100.0,
            }
            for pos in top_positions
        ]

        return {
            "class_id": class_id,
            "class_name": pred_class,
            "vi_name": info["vi_name"],
            "group": info["group"],
            "confidence": confidence,
            "is_low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            "color": info["color"],
            "badge_bg": info["badge_bg"],
            "icon": info["icon"],
            "guide": info["guide"],
            "top_k": top_predictions,
        }


if __name__ == "__main__":
    predictor = WastePredictor()
    dummy = Image.new("RGB", (224, 224), color=(120, 160, 90))
    res = predictor.predict(dummy)
    print(f"[Predictor] Kiểm thử OK! Dự đoán: {res['class_name']} ({res['group']}) - {res['confidence']:.2f}%")
