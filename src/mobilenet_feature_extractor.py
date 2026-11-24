"""
Module: mobilenet_feature_extractor.py (Tầng 2 - Mô hình CNN | Phụ trách: TV1 Lê Lý Phúc An)
Mục đích:
  - Sử dụng MobileNetV2 pretrained ImageNet, thay classifier bằng nn.Identity().
  - Trích xuất vector đặc trưng 1.280 chiều từ tầng Global Average Pooling (GAP).
  - Chạy cố định ở chế độ eval() và torch.no_grad(), dùng chung transform với src/preprocessing.py.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES_PATH, DATA_DIR, FEATURE_DIM
from src.preprocessing import eval_transform, load_and_preprocess_image, load_class_names


class MobileNetV2FeatureExtractor(nn.Module):
    """Bộ trích xuất vector đặc trưng 1.280 chiều từ MobileNetV2 (ImageNet)."""

    def __init__(self, device: Optional[str] = None, weights: str = "DEFAULT"):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights_param = MobileNet_V2_Weights.DEFAULT if weights in ("DEFAULT", True) else None

        base_model = mobilenet_v2(weights=weights_param)
        base_model.classifier = nn.Identity()  # Bỏ lớp Linear cuối, giữ vector 1.280 chiều sau GAP

        self.model = base_model.to(self.device)
        self.model.eval()
        self.transform = eval_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_from_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Trích xuất vector 1D (1280,) từ 1 ảnh đầu vào (đường dẫn, PIL Image hoặc mảng numpy)."""
        if isinstance(image_input, np.ndarray):
            image_input = Image.fromarray(image_input)
        tensor = load_and_preprocess_image(image_input).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_from_batch(self, tensors: torch.Tensor) -> np.ndarray:
        """Trích xuất ma trận đặc trưng 2D (B, 1280) từ 1 batch tensor (B, 3, 224, 224)."""
        tensors = tensors.to(self.device)
        with torch.no_grad():
            features = self.model(tensors)
        return features.cpu().numpy().astype(np.float32)

    def extract_from_directory(
        self,
        root_dir: Union[str, Path],
        batch_size: int = 64,
        num_workers: int = 0,
        transform=None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Trích xuất toàn bộ thư mục ảnh theo cấu trúc ImageFolder (căn chuẩn nhãn theo class_names.json)."""
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {root_dir}")

        dataset = ImageFolder(root=str(root_dir), transform=transform or self.transform)

        # Đồng bộ chỉ số nhãn (0..11) khớp 100% với data/class_names.json
        if CLASS_NAMES_PATH.exists():
            ref_classes = load_class_names(CLASS_NAMES_PATH)
            unknown = [c for c in dataset.classes if c not in ref_classes]
            if unknown:
                raise ValueError(f"Thư mục chứa lớp lạ không có trong class_names.json: {unknown}")
            remap = {dataset.class_to_idx[c]: ref_classes.index(c) for c in dataset.classes}
            dataset.targets = [remap[t] for t in dataset.targets]
            dataset.samples = [(p, remap[t]) for p, t in dataset.samples]
            dataset.class_to_idx = {c: ref_classes.index(c) for c in dataset.classes}
            dataset.classes = ref_classes

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"[MobileNetV2] Đang trích xuất '{root_dir.name}' ({len(dataset)} ảnh)...")

        all_features, all_labels = [], []
        t0 = time.time()
        for idx, (batch_imgs, batch_lbls) in enumerate(dataloader):
            all_features.append(self.extract_from_batch(batch_imgs))
            all_labels.append(batch_lbls.numpy())
            if (idx + 1) % 20 == 0 or (idx + 1) == len(dataloader):
                done = min((idx + 1) * batch_size, len(dataset))
                print(f"  -> Tiến độ: {done}/{len(dataset)} ({done / len(dataset) * 100:.1f}%)")

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        print(f"[MobileNetV2] Hoàn tất trong {time.time() - t0:.2f}s | Shape: {features.shape}")
        return features, labels, dataset.classes


def save_feature_vectors(
    features: np.ndarray,
    labels: np.ndarray,
    save_dir: Union[str, Path] = DATA_DIR,
    prefix: str = "train_raw",
) -> Tuple[str, str]:
    """Lưu ma trận đặc trưng X_<prefix>.npy và nhãn y_<prefix>.npy."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    feat_path = save_dir / f"X_{prefix}.npy"
    label_path = save_dir / f"y_{prefix}.npy"
    np.save(feat_path, features)
    np.save(label_path, labels)
    print(f"[Storage] Đã lưu: {feat_path.name} {features.shape} & {label_path.name} {labels.shape}")
    return str(feat_path), str(label_path)


def load_feature_vectors(
    save_dir: Union[str, Path] = DATA_DIR,
    prefix: str = "train_raw",
) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc cặp file X_<prefix>.npy và y_<prefix>.npy từ thư mục data/."""
    save_dir = Path(save_dir)
    feat_path = save_dir / f"X_{prefix}.npy"
    label_path = save_dir / f"y_{prefix}.npy"
    if not feat_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {feat_path} hoặc {label_path}")
    return np.load(feat_path), np.load(label_path)


if __name__ == "__main__":
    extractor = MobileNetV2FeatureExtractor()
    dummy_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    vec = extractor.extract_from_image(dummy_img)
    batch_vec = extractor.extract_from_batch(torch.randn(4, 3, 224, 224))
    assert vec.shape == (FEATURE_DIM,) and batch_vec.shape == (4, FEATURE_DIM)
    print(f"[MobileNetV2] Kiểm thử OK! Single: {vec.shape}, Batch: {batch_vec.shape}")
