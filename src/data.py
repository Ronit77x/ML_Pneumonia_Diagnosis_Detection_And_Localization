import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom

from config import IMG_SIZE


def read_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return img


def make_lung_mask(img):
    """Lightweight fallback ROI mask.

    This is intentionally not presented as the U-Net used in the reference
    paper. It makes the student implementation runnable without a second
    lung-mask dataset.
    """
    u8 = (img * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Prefer the darker central lung regions.
    inv = 255 - th
    kernel = np.ones((7, 7), np.uint8)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    inv = cv2.resize(inv, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (inv > 0).astype(np.float32)


class RSNAPneumoniaDataset(Dataset):
    def __init__(self, frame, image_dir, split="train", augment=False):
        self.df = frame.reset_index(drop=True)
        self.image_dir = str(image_dir)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        patient_id = str(r["patientId"])
        path = os.path.join(self.image_dir, patient_id + ".dcm")
        img = read_dicom(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() < 0.25:
                angle = np.random.uniform(-7, 7)
                m = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1)
                img = cv2.warpAffine(img, m, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)

        mean, std = img.mean(), img.std() + 1e-6
        img = (img - mean) / std
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(int(r["Target"]), dtype=torch.long)

        meta = {
            "patientId": patient_id,
            "x": float(r["x"]) if "x" in r and pd.notna(r["x"]) else np.nan,
            "y": float(r["y"]) if "y" in r and pd.notna(r["y"]) else np.nan,
            "width": float(r["width"]) if "width" in r and pd.notna(r["width"]) else np.nan,
            "height": float(r["height"]) if "height" in r and pd.notna(r["height"]) else np.nan,
            "orig_h": int(read_dicom(path).shape[0]),
            "orig_w": int(read_dicom(path).shape[1]),
        }
        return x, y, meta
