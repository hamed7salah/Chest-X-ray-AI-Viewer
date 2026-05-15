import pydicom
import numpy as np
import cv2


def dicom_to_numpy(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    if 'PixelData' not in ds:
        raise ValueError("No PixelData in DICOM")

    arr = ds.pixel_array.astype(np.float32)

    # Photometric Interpretation handling
    if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
        arr = np.max(arr) - arr

    # Normalize to 0-1
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()

    # Convert to 3-channel for downstream models/visualization
    img = (arr * 255).astype(np.uint8)
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img3


def save_png_from_array(arr, out_path: str):
    # arr expected to be HxWx3 uint8 or HxW
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype('uint8'), cv2.COLOR_GRAY2RGB)
    cv2.imwrite(out_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
