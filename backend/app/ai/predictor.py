import os
import numpy as np
import cv2
from ..utils.dicom import load_image
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
        img = load_image(dicom_path)
        if self.mode == "pytorch" and self.model is not None:
            x = self._preprocess(img)
            import torch
            with torch.no_grad():
                out = self.model(x.to(self.device))
                prob = float(torch.sigmoid(out).item())
            label = "pneumonia" if prob > 0.5 else "normal"
            return {"label": label, "score": prob}

        return self._heuristic_pneumonia_score(img)

    def _heuristic_pneumonia_score(self, img: np.ndarray) -> Dict:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        cropped = gray[int(h * 0.15):int(h * 0.95), int(w * 0.10):int(w * 0.90)]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cropped)
        blur = cv2.GaussianBlur(enhanced, (7, 7), 0)

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dense_pixels = np.mean(blur > otsu * 0.8)

        local_mean = cv2.blur(blur.astype(np.float32), (31, 31))
        local_sq = cv2.blur(np.square(blur.astype(np.float32)), (31, 31))
        local_var = np.maximum(0.0, local_sq - np.square(local_mean))
        var_score = float(np.percentile(local_var, 90) / 255.0)

        lower_region = blur[int(blur.shape[0] * 0.45) :, :]
        _, lower_otsu = cv2.threshold(lower_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_dense = np.mean(lower_region > lower_otsu * 0.8)

        edges = cv2.Canny(blur, 30, 120)
        edge_ratio = float(np.mean(edges > 0))

        score = 0.28 * dense_pixels + 0.28 * lower_dense + 0.24 * var_score + 0.20 * edge_ratio
        score = float(max(0.0, min(1.0, score)))
        label = "pneumonia" if score > 0.42 else "normal"
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
        img = load_image(dicom_path)
        h, w = img.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)
        # Emphasize darker regions (potential pathology)
        gray = np.mean(img, axis=2) / 255.0
        heat = 1.0 - gray
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

        import cv2
        cmap = cv2.applyColorMap((heat * 255).astype('uint8'), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cmap, 0.5, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, 0)
        if png_preview_path:
            out_path = png_preview_path.replace(".png", ".heat.png")
        else:
            out_path = f"static/{os.path.basename(dicom_path)}.heat.png"
        cv2.imwrite(out_path, overlay)
        return out_path
