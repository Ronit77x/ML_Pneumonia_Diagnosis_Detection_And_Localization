import cv2
import numpy as np
import torch


def make_cam(model, x, class_idx=1):
    model.eval()
    with torch.no_grad():
        logits, fmap = model(x, return_features=True)
        weights = model.fc.weight[class_idx].detach().cpu().numpy()
        fmap_np = fmap[0].detach().cpu().numpy()
        cam = np.tensordot(weights, fmap_np, axes=(0, 0))
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_LINEAR)
    return cam, logits.softmax(dim=1)[0].detach().cpu().numpy()


def connected_components_boxes(binary):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= 4:
            boxes.append((int(x), int(y), int(x + w), int(y + h), int(area)))
    return sorted(boxes, key=lambda z: z[4], reverse=True)


def localize(cam, threshold=0.20):
    mask = (cam >= threshold).astype(np.uint8)

    # Conservative chest-region constraint.
    # This removes extreme image borders and the lowest part
    # of the image where irrelevant CAM activations can occur.
    h, w = mask.shape

    roi = np.zeros_like(mask)

    x1 = int(0.10 * w)
    x2 = int(0.90 * w)
    y1 = int(0.08 * h)
    y2 = int(0.88 * h)

    roi[y1:y2, x1:x2] = 1

    mask = mask * roi

    return connected_components_boxes(mask)


def draw_boxes(image, boxes):
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor((np.clip(out, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out
