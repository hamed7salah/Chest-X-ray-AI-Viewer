import os
import numpy as np
from ..utils.dicom import dicom_to_numpy
from typing import Dict

# Heavy ML libraries are imported lazily only when needed (PREDICTOR_MODE=pytorch)

MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "")


class Predictor:
    def __init__(self, mode: str = "baseline"):
        self.mode = os.environ.get("PREDICTOR_MODE", mode)
        self.device = None
        self.model = None
        if self.mode == "pytorch":
            self._load_model()

    def _load_model(self):
        # Import heavy ML libs lazily
        import torch
        from torchvision import models

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use pretrained ResNet backbone as feature extractor + single output head
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, 1)
        self.model = backbone.to(self.device)
        if MODEL_CHECKPOINT and os.path.exists(MODEL_CHECKPOINT):
            self.model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=self.device))
        self.model.eval()

    def predict(self, dicom_path: str) -> Dict:
        img = dicom_to_numpy(dicom_path)
        if self.mode == "pytorch" and self.model is not None:
            x = self._preprocess(img)
            import torch
            with torch.no_grad():
                out = self.model(x.to(self.device))
                prob = float(torch.sigmoid(out).item())
            label = "pneumonia" if prob > 0.5 else "normal"
            return {"label": label, "score": prob}

        # Baseline heuristic: mean intensity
        mean = img.mean() / 255.0
        score = float(max(0.0, min(1.0, (0.5 - mean) * 2 + 0.5)))
        label = "pneumonia" if score > 0.5 else "normal"
        return {"label": label, "score": score}

    def _preprocess(self, img: np.ndarray):
        # Resize and normalize for ResNet (requires torchvision)
        from torchvision import transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = transform(img)
        x = x.unsqueeze(0)
        return x

    def gradcam(self, dicom_path: str, png_preview_path: str = None) -> str:
        # Simple fallback heatmap for baseline mode
        img = dicom_to_numpy(dicom_path)
        h, w = img.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)
        # Emphasize darker regions (potential pathology)
        gray = np.mean(img, axis=2) / 255.0
        heat = 1.0 - gray
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

        import cv2
        cmap = cv2.applyColorMap((heat * 255).astype('uint8'), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cmap, 0.5, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, 0)
        out_path = png_preview_path or f"static/{os.path.basename(dicom_path)}.heat.png"
        cv2.imwrite(out_path, overlay)
        return out_path
