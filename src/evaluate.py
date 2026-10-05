import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from data import RSNAPneumoniaDataset
from model import WeakPneumoniaCNN


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--checkpoint", required=True)

    # Dataset split to evaluate
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )

    # Classification probability threshold
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Probability threshold for predicting pneumonia",
    )

    args = ap.parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Split:", args.split)
    print("Classification threshold:", args.threshold)

    # --------------------------------------------------
    # Load CSV
    # --------------------------------------------------

    df = pd.read_csv(args.csv)

    split_df = df[df["split"] == args.split].copy()

    print("Number of samples:", len(split_df))

    # --------------------------------------------------
    # Dataset and DataLoader
    # --------------------------------------------------

    ds = RSNAPneumoniaDataset(
        split_df,
        args.image_dir,
        split=args.split,
        augment=False,
    )

    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = WeakPneumoniaCNN().to(device)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------

    ys = []
    probabilities = []

    with torch.no_grad():

        for x, y, _ in dl:

            x = x.to(device)

            logits = model(x)

            # Probability of pneumonia class
            probs = torch.softmax(logits, dim=1)[:, 1]

            probabilities.extend(
                probs.cpu().numpy()
            )

            ys.extend(
                y.numpy()
            )

    # --------------------------------------------------
    # Apply classification threshold
    # --------------------------------------------------

    ps = [
        1 if p >= args.threshold else 0
        for p in probabilities
    ]

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    accuracy = accuracy_score(ys, ps)

    precision = precision_score(
        ys,
        ps,
        zero_division=0,
    )

    recall = recall_score(
        ys,
        ps,
        zero_division=0,
    )

    f1 = f1_score(
        ys,
        ps,
        zero_division=0,
    )

    cm = confusion_matrix(ys, ps)

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)

    print(f"Split     : {args.split}")
    print(f"Threshold : {args.threshold:.2f}")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print("=" * 50)


if __name__ == "__main__":
    main()