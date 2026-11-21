"""
Module: mobilenet_feature_extractor.py
Tác giả: Thành viên 1 - Phúc An (Nhóm trưởng)
Mục đích:
  - Khởi tạo kiến trúc MobileNetV2 pretrained ImageNet từ torchvision.models.
  - Thay thế classifier bằng nn.Identity() để trích xuất vector đặc trưng 1.280 chiều từ tầng Global Average Pooling (GAP).
  - Đảm bảo model luôn chạy ở chế độ eval (model.eval()) và không tính gradient (torch.no_grad()) để tối ưu tốc độ và bộ nhớ.
  - Cung cấp các hàm trích xuất:
      + Cho 1 ảnh đơn lẻ (PIL Image, đường dẫn ảnh, mảng numpy).
      + Cho 1 batch tensor (N, 3, 224, 224) -> (N, 1280).
      + Cho tập dữ liệu (PyTorch Dataset / DataLoader / Thư mục ảnh chia theo lớp) và lưu ra file .npy.
  - Phục vụ tái sử dụng trực tiếp cho:
      + TV1: Trích xuất đặc trưng tập Train (gốc & augmentation), Val, Test ra file .npy.
      + TV3: Kiểm thử pipeline học máy với Random Forest.
      + TV4: Tích hợp trực tiếp vào ứng dụng web Streamlit (app/app.py) để trích xuất ảnh người dùng upload.
"""

import os
import time
from pathlib import Path
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image


def get_mobilenet_transform():
    """
    Tạo pipeline tiền xử lý ảnh chuẩn cho MobileNetV2:
      1. Resize ảnh về kích thước (224, 224)
      2. Chuyển đổi thành Tensor [0, 1]
      3. Chuẩn hóa (Normalize) theo mean và std của ImageNet:
         mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class MobileNetV2FeatureExtractor(nn.Module):
    """
    Bộ trích xuất đặc trưng MobileNetV2 Pretrained ImageNet.
    Lớp Classifier cuối cùng được thay thế bằng nn.Identity() để trả về vector 1.280 chiều.
    """
    def __init__(self, device: Optional[str] = None, weights: str = "DEFAULT"):
        super().__init__()
        
        # Tự động nhận diện phần cứng (CUDA nếu có GPU, ngược lại dùng CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Nạp weights MobileNetV2 chuẩn
        if weights == "DEFAULT" or weights is True:
            weights_param = MobileNet_V2_Weights.DEFAULT
        elif weights is None or weights is False:
            weights_param = None
        else:
            weights_param = weights
            
        print(f"[MobileNetV2] Đang nạp mô hình MobileNetV2 (weights={weights}, device={self.device})...")
        base_model = mobilenet_v2(weights=weights_param)
        
        # Bỏ lớp phân loại (Linear) cuối cùng, giữ lại tầng Global Average Pooling
        # Đầu ra sau Pooling sẽ là vector 1.280 chiều
        base_model.classifier = nn.Identity()
        
        self.model = base_model.to(self.device)
        self.model.eval()  # Đặt chế độ eval cố định (tắt Dropout, cố định BatchNorm)
        
        # Chuẩn hóa tiền xử lý đồng nhất toàn nhóm
        self.transform = get_mobilenet_transform()
        print("[MobileNetV2] Khởi tạo bộ trích xuất đặc trưng thành công (Đầu ra: 1.280 chiều).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward qua mô hình (chỉ suy luận, không tính gradient).
        """
        return self.model(x)

    def extract_from_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Trích xuất vector 1.280 chiều từ 1 ảnh đầu vào.
        
        Tham số:
          image_input: Đường dẫn ảnh (str/Path), đối tượng PIL.Image, hoặc mảng numpy.
          
        Trả về:
          Vector đặc trưng 1D numpy array kích thước (1280,).
        """
        # 1. Chuyển đổi định dạng đầu vào về PIL Image
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Định dạng ảnh không hỗ trợ: {type(image_input)}")

        # 2. Bắt buộc ép về hệ màu RGB (loại bỏ kênh Alpha nếu là ảnh RGBA/PNG)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 3. Tiền xử lý & tạo batch dimension (1, 3, 224, 224)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 4. Trích xuất đặc trưng với torch.no_grad()
        with torch.no_grad():
            features = self.model(tensor)
            
        # 5. Chuyển về mảng numpy 1D (1280,)
        vector = features.squeeze(0).cpu().numpy().astype(np.float32)
        return vector

    def extract_from_batch(self, tensors: torch.Tensor) -> np.ndarray:
        """
        Trích xuất đặc trưng từ một batch tensor (B, 3, 224, 224).
        
        Trả về:
          Ma trận đặc trưng 2D numpy array kích thước (B, 1280).
        """
        tensors = tensors.to(self.device)
        with torch.no_grad():
            features = self.model(tensors)
        return features.cpu().numpy().astype(np.float32)

    def extract_from_directory(
        self,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Trích xuất đặc trưng từ thư mục ảnh chia theo lớp (cấu trúc torchvision ImageFolder):
        root_dir/
          ├── battery/
          ├── plastic/
          └── ...
          
        Trả về:
          (features, labels, class_names)
          features: ndarray (N, 1280)
          labels: ndarray (N,)
          class_names: danh sách tên các lớp (12 lớp)
        """
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {root_dir}")

        dataset = ImageFolder(root=root_dir, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        class_names = dataset.classes
        print(f"[MobileNetV2] Bắt đầu trích xuất từ '{root_dir}' | Tổng mẫu: {len(dataset)} | Số lớp: {len(class_names)}")

        all_features = []
        all_labels = []

        start_time = time.time()
        for idx, (batch_imgs, batch_lbls) in enumerate(dataloader):
            feats = self.extract_from_batch(batch_imgs)
            all_features.append(feats)
            all_labels.append(batch_lbls.numpy())
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(dataloader):
                processed = min((idx + 1) * batch_size, len(dataset))
                print(f"  -> Tiến độ: {processed}/{len(dataset)} ảnh ({(processed/len(dataset)*100):.1f}%)")

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        elapsed = time.time() - start_time
        print(f"[MobileNetV2] Hoàn thành trích xuất trong {elapsed:.2f}s | Shape: {features.shape}")
        
        return features, labels, class_names


def save_feature_vectors(
    features: np.ndarray,
    labels: np.ndarray,
    save_dir: str = "data",
    prefix: str = "train_raw"
) -> Tuple[str, str]:
    """
    Lưu ma trận vector và nhãn tương ứng ra file .npy.
    Ví dụ:
      X_train_raw.npy, y_train_raw.npy
      X_train_aug.npy, y_train_aug.npy
      X_val.npy, y_val.npy
      X_test.npy, y_test.npy
    """
    os.makedirs(save_dir, exist_ok=True)
    feat_path = os.path.join(save_dir, f"X_{prefix}.npy")
    label_path = os.path.join(save_dir, f"y_{prefix}.npy")
    
    np.save(feat_path, features)
    np.save(label_path, labels)
    
    print(f"[Storage] Đã lưu vector: {feat_path} (Shape: {features.shape})")
    print(f"[Storage] Đã lưu nhãn:   {label_path} (Shape: {labels.shape})")
    return feat_path, label_path


def load_feature_vectors(
    save_dir: str = "data",
    prefix: str = "train_raw"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nạp ma trận vector và nhãn từ file .npy.
    """
    feat_path = os.path.join(save_dir, f"X_{prefix}.npy")
    label_path = os.path.join(save_dir, f"y_{prefix}.npy")
    
    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Không tìm thấy file: {feat_path} hoặc {label_path}")
        
    features = np.load(feat_path)
    labels = np.load(label_path)
    print(f"[Storage] Đã nạp thành công: {feat_path} ({features.shape}), {label_path} ({labels.shape})")
    return features, labels


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM THỬ MODULE TRÍCH XUẤT ĐẶC TRƯNG MOBILENETV2 (TV1 - PHÚC AN)")
    print("=" * 60)
    
    # 1. Khởi tạo mô hình
    extractor = MobileNetV2FeatureExtractor()
    
    # 2. Tạo ảnh mẫu giả lập (3 kênh màu RGB, kích thước 224x224)
    dummy_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    
    # 3. Trích xuất thử nghiệm
    feat = extractor.extract_from_image(dummy_img)
    print(f"Kiểm tra kích thước vector ảnh đơn: {feat.shape} (Kỳ vọng: (1280,))")
    assert feat.shape == (1280,), "Lỗi: Kích thước vector không phải 1280!"
    
    # 4. Kiểm tra trích xuất batch giả lập (4 ảnh)
    dummy_batch = torch.randn(4, 3, 224, 224)
    batch_feats = extractor.extract_from_batch(dummy_batch)
    print(f"Kiểm tra kích thước vector batch (4 mẫu): {batch_feats.shape} (Kỳ vọng: (4, 1280))")
    assert batch_feats.shape == (4, 1280), "Lỗi: Kích thước batch không phải (4, 1280)!"
    
    print("\n[THÀNH CÔNG] Module MobileNetV2FeatureExtractor hoạt động hoàn hảo và sẵn sàng phục vụ toàn nhóm!")
    print("=" * 60)
