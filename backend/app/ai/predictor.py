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
        img = self._validate_image(img)
        if self.mode == "pytorch" and self.model is not None:
            x = self._preprocess(img)
            import torch
            with torch.no_grad():
                out = self.model(x.to(self.device))
                prob = float(torch.sigmoid(out).item())
            label = "pneumonia" if prob > 0.5 else "normal"
            return {
                "label": label,
                "score": prob,
                "ensemble_score": prob,
                "uncertainty": 0.05,
            }

        return self._ensemble_pneumonia_score(img)

    def _validate_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Invalid image data")
        if img.ndim == 2:
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2RGB)
        h, w = img.shape[:2]
        if h < 32 or w < 32:
            img = cv2.resize(img, (max(128, w), max(128, h)), interpolation=cv2.INTER_LINEAR)
        return img

    def _heuristic_pneumonia_score(self, img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        y1, y2 = int(h * 0.15), int(h * 0.95)
        x1, x2 = int(w * 0.10), int(w * 0.90)
        if y2 <= y1 or x2 <= x1:
            cropped = gray
        else:
            cropped = gray[y1:y2, x1:x2]

        if cropped.size == 0:
            cropped = gray

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cropped)
        blur = cv2.GaussianBlur(enhanced, (7, 7), 0)

        if blur.size == 0:
            return 0.0

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dense_pixels = np.mean(blur > otsu * 0.8)

        local_mean = cv2.blur(blur.astype(np.float32), (31, 31))
        local_sq = cv2.blur(np.square(blur.astype(np.float32)), (31, 31))
        local_var = np.maximum(0.0, local_sq - np.square(local_mean))
        var_score = float(np.percentile(local_var, 90) / 255.0)

        lower_region = blur[int(blur.shape[0] * 0.45) :, :]
        if lower_region.size == 0:
            lower_dense = 0.0
        else:
            _, lower_otsu = cv2.threshold(lower_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lower_dense = np.mean(lower_region > lower_otsu * 0.8)

        edges = cv2.Canny(blur, 30, 120)
        edge_ratio = float(np.mean(edges > 0)) if edges.size else 0.0

        score = 0.28 * dense_pixels + 0.28 * lower_dense + 0.24 * var_score + 0.20 * edge_ratio
        return float(max(0.0, min(1.0, score)))

    def _calibrate_score(self, score: float) -> float:
        # Simple logistic-based calibration to improve interpretability
        calibrated = 1 / (1 + np.exp(-12 * (score - 0.35)))
        return float(max(0.0, min(1.0, calibrated)))

    def _ensemble_pneumonia_score(self, img: np.ndarray) -> Dict:
        scores = [
            self._heuristic_pneumonia_score(img),
            self._histogram_opacity_score(img),
            self._texture_variance_score(img),
        ]
        ensemble_score = float(np.mean(scores))
        uncertainty = float(np.std(scores))
        calibrated = self._calibrate_score(ensemble_score)
        label = "pneumonia" if calibrated > 0.5 else "normal"
        return {
            "label": label,
            "score": calibrated,
            "ensemble_score": ensemble_score,
            "uncertainty": uncertainty,
        }

    def _histogram_opacity_score(self, img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = np.sum(hist)
        if total == 0:
            return 0.0
        bright_count = np.sum(hist[160:])
        score = float(bright_count / total)
        return float(max(0.0, min(1.0, 1.0 - score)))

    def _texture_variance_score(self, img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if gray.size == 0:
            return 0.0
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        local_var = cv2.Laplacian(blurred, cv2.CV_32F)
        score = float(np.mean(np.abs(local_var)) / 50.0)
        return float(max(0.0, min(1.0, score)))

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
