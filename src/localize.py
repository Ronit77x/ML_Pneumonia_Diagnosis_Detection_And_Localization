import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch

from cam_localizer import make_cam, localize as find_boxes
from model import WeakPneumoniaCNN


def read(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--out", default="outputs/localization.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = read(args.image)
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    norm = (small - small.mean()) / (small.std() + 1e-6)
    x = torch.tensor(norm, dtype=torch.float32)[None, None].to(device)

    model = WeakPneumoniaCNN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cam, probs = make_cam(model, x, class_idx=1)
    boxes = find_boxes(cam, args.threshold)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(small, cmap="gray")
    plt.axis("off")
    plt.title("X-ray")

    plt.subplot(1, 3, 2)
    plt.imshow(small, cmap="gray")
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(f"CAM P={probs[1]:.2f}")

    plt.subplot(1, 3, 3)
    plt.imshow(small, cmap="gray")
    for x1, y1, x2, y2, _ in boxes:
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))
    plt.axis("off")
    plt.title(f"Boxes: {len(boxes)}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print("Pneumonia probability:", float(probs[1]))
    print("Boxes:", boxes)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
