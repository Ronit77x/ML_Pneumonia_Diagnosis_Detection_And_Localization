import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from matplotlib.patches import Rectangle

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

    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    return img


def prepare_image(path):
    img = read_dicom(path)
    small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
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

    return (x * sx, y * sy, (x + w) * sx, (y + h) * sy)


def draw_box(ax, box, label, linewidth=2):
    x1, y1, x2, y2 = box
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=linewidth)
    ax.add_patch(rect)
    ax.text(x1, max(y1 - 2, 2), label, fontsize=8, backgroundcolor="white")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="data/split.csv")
    parser.add_argument("--image-dir", default="data/stage_2_train_images")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--classification-threshold", type=float, default=0.35)
    parser.add_argument("--cam-threshold", type=float, default=0.20)
    parser.add_argument("--num-cases", type=int, default=6)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("classification threshold:", args.classification_threshold)
    print("cam threshold:", args.cam_threshold)

    df = pd.read_csv(args.csv)
    test = df[(df["split"] == "test") & (df["Target"] == 1)].copy().drop_duplicates(subset=["patientId"]).reset_index(drop=True)

    model = WeakPneumoniaCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_cases = []

    print()
    print("evaluating cases...")

    for idx, row in test.iterrows():
        patient_id = str(row["patientId"])
        image_path = os.path.join(args.image_dir, patient_id + ".dcm")

        if not os.path.exists(image_path):
            continue

        x, small = prepare_image(image_path)
        x = x.to(device)

        with torch.no_grad():
            cam, probs = make_cam(model, x, class_idx=1)

        probability = float(probs[1])
        boxes = localize(cam, threshold=args.cam_threshold)

        ds = pydicom.dcmread(image_path)
        orig_h, orig_w = ds.pixel_array.shape[:2]
        gt = scale_gt_box(row, orig_w, orig_h)

        if probability >= args.classification_threshold and boxes:
            pred = boxes[0][:4]
            current_iou = iou(pred, gt)
        else:
            pred = None
            current_iou = 0.0

        all_cases.append({
            "patientId": patient_id,
            "probability": probability,
            "iou": current_iou,
            "gt": gt,
            "pred": pred,
            "cam": cam,
            "image": small,
        })

        if (idx + 1) % 100 == 0:
            print(f"processed {idx + 1}/{len(test)}")

    best_cases = sorted(all_cases, key=lambda z: z["iou"], reverse=True)[:args.num_cases]
    failed_cases = sorted(all_cases, key=lambda z: z["iou"])[:args.num_cases]

    os.makedirs("outputs/inspection", exist_ok=True)

    for i, case in enumerate(best_cases):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        image = case["image"]
        cam = case["cam"]

        axes[0].imshow(image, cmap="gray")
        draw_box(axes[0], case["gt"], "ground truth")
        axes[0].set_title("X-ray + Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(cam, cmap="jet", alpha=0.45)
        axes[1].set_title(f"CAM\nP={case['probability']:.3f}")
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        if case["pred"] is not None:
            draw_box(axes[2], case["pred"], "CAM box")
        axes[2].set_title(f"Prediction\nIoU={case['iou']:.3f}")
        axes[2].axis("off")

        axes[3].imshow(image, cmap="gray")
        draw_box(axes[3], case["gt"], "GT")
        if case["pred"] is not None:
            draw_box(axes[3], case["pred"], "CAM")
        axes[3].set_title("GT vs CAM")
        axes[3].axis("off")

        fig.suptitle(f"BEST CASE {i + 1} | {case['patientId']}")
        plt.tight_layout()
        output = f"outputs/inspection/best_{i + 1}.png"
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close()
        print("saved:", output)

    for i, case in enumerate(failed_cases):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        image = case["image"]
        cam = case["cam"]

        axes[0].imshow(image, cmap="gray")
        draw_box(axes[0], case["gt"], "ground truth")
        axes[0].set_title("X-ray + Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(cam, cmap="jet", alpha=0.45)
        axes[1].set_title(f"CAM\nP={case['probability']:.3f}")
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        if case["pred"] is not None:
            draw_box(axes[2], case["pred"], "CAM box")
        axes[2].set_title(f"Prediction\nIoU={case['iou']:.3f}")
        axes[2].axis("off")

        axes[3].imshow(image, cmap="gray")
        draw_box(axes[3], case["gt"], "GT")
        if case["pred"] is not None:
            draw_box(axes[3], case["pred"], "CAM")
        axes[3].set_title("GT vs CAM")
        axes[3].axis("off")

        fig.suptitle(f"FAILURE CASE {i + 1} | {case['patientId']}")
        plt.tight_layout()
        output = f"outputs/inspection/failure_{i + 1}.png"
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close()
        print("saved:", output)

    print()
    print("inspection complete.")
    print("images saved in: outputs/inspection/")


if __name__ == "__main__":
    main()
