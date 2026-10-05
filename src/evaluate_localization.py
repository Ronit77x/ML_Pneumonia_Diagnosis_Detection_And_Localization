import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch

from cam_localizer import make_cam, localize
from iou import iou
from model import WeakPneumoniaCNN


IMG_SIZE = 128


def read_dicom(path):
    ds = pydicom.dcmread(path)

    img = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    lo, hi = np.percentile(img, (1, 99))

    img = np.clip(
        (img - lo) / max(hi - lo, 1e-6),
        0,
        1,
    )

    return img


def prepare_image(path):
    img = read_dicom(path)

    small = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    mean = small.mean()
    std = small.std() + 1e-6

    norm = (small - mean) / std

    x = torch.tensor(norm, dtype=torch.float32)[None, None]

    return x, small


def scale_gt_box(row, orig_w, orig_h):
    x = float(row["x"])
    y = float(row["y"])
    w = float(row["width"])
    h = float(row["height"])

    sx = IMG_SIZE / orig_w
    sy = IMG_SIZE / orig_h

    x1 = x * sx
    y1 = y * sy
    x2 = (x + w) * sx
    y2 = (y + h) * sy

    return (x1, y1, x2, y2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="data/split.csv")
    parser.add_argument("--image-dir", default="data/stage_2_train_images")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--classification-threshold", type=float, default=0.35)
    parser.add_argument("--cam-threshold", type=float, default=0.20)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("split:", args.split)
    print("classification threshold:", args.classification_threshold)
    print("cam threshold:", args.cam_threshold)

    df = pd.read_csv(args.csv)
    split_df = df[df["split"] == args.split].copy()

    positive = split_df[split_df["Target"] == 1].copy()
    patient_ids = positive["patientId"].drop_duplicates().tolist()

    print(f"positive {args.split} images:", len(patient_ids))

    model = WeakPneumoniaCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ious = []
    localized_count = 0
    missed_count = 0
    no_box_count = 0

    for idx, patient_id in enumerate(patient_ids):
        patient_id = str(patient_id)
        image_path = os.path.join(args.image_dir, patient_id + ".dcm")

        if not os.path.exists(image_path):
            print("missing:", image_path)
            continue

        patient_rows = positive[positive["patientId"].astype(str) == patient_id]

        x, small = prepare_image(image_path)
        x = x.to(device)

        with torch.no_grad():
            cam, probs = make_cam(model, x, class_idx=1)

        pneumonia_prob = float(probs[1])

        if pneumonia_prob < args.classification_threshold:
            current_iou = 0.0
            missed_count += 1
            ious.append(current_iou)
            continue

        boxes = localize(cam, threshold=args.cam_threshold)

        if len(boxes) == 0:
            current_iou = 0.0
            no_box_count += 1
            ious.append(current_iou)
            continue

        localized_count += 1
        pred = boxes[0][:4]

        ds = pydicom.dcmread(image_path)
        orig_h, orig_w = ds.pixel_array.shape[:2]

        gt_boxes = [scale_gt_box(gt_row, orig_w, orig_h) for _, gt_row in patient_rows.iterrows()]
        gt_ious = [iou(pred, gt_box) for gt_box in gt_boxes]
        current_iou = max(gt_ious) if gt_ious else 0.0

        ious.append(current_iou)

        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1}/{len(patient_ids)}")

    ious = np.array(ious, dtype=np.float32)

    mean_iou = float(np.mean(ious))
    median_iou = float(np.median(ious))
    pct_03 = float(np.mean(ious >= 0.3) * 100)
    pct_05 = float(np.mean(ious >= 0.5) * 100)

    print()
    print("========================================")
    print("LOCALIZATION RESULTS")
    print("========================================")

    print(f"split              : {args.split}")
    print(f"images evaluated   : {len(ious)}")
    print(f"mean IoU           : {mean_iou:.4f}")
    print(f"median IoU         : {median_iou:.4f}")
    print(f"IoU >= 0.3         : {pct_03:.2f}%")
    print(f"IoU >= 0.5         : {pct_05:.2f}%")
    print()
    print("classification threshold:", args.classification_threshold)
    print("CAM threshold:", args.cam_threshold)
    print()
    print(f"classification misses    : {missed_count}")
    print(f"images with CAM boxes    : {localized_count}")
    print(f"images without CAM boxes : {no_box_count}")


if __name__ == "__main__":
    main()
