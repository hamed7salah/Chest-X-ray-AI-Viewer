import os
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import pydicom


def dicom_to_numpy(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    if 'PixelData' not in ds:
        raise ValueError("No PixelData in DICOM")

    arr = ds.pixel_array.astype(np.float32)

    # Photometric Interpretation handling
    if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
        arr = np.max(arr) - arr

    return normalize_and_rgb(arr)


def image_to_numpy(path: str) -> np.ndarray:
    image = Image.open(path)
    image = image.convert('RGB')
    arr = np.asarray(image).astype(np.float32)
    return normalize_and_rgb(arr)


def load_image(path: str) -> np.ndarray:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}:
        return image_to_numpy(path)

    try:
        return dicom_to_numpy(path)
    except Exception:
        return image_to_numpy(path)


def normalize_and_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.float32)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype(np.float32)

    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    img = (arr * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def read_metadata(path: str) -> dict:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        metadata = {
            "PatientID": str(getattr(ds, "PatientID", "")),
            "PatientName": str(getattr(ds, "PatientName", "")),
            "StudyDate": str(getattr(ds, "StudyDate", "")),
            "Modality": str(getattr(ds, "Modality", "")),
            "StudyDescription": str(getattr(ds, "StudyDescription", "")),
        }
        return metadata
    except Exception:
        return {}


def save_png_from_array(arr, out_path: str):
    # arr expected to be HxWx3 uint8 or HxW
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype('uint8'), cv2.COLOR_GRAY2RGB)
    cv2.imwrite(out_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
