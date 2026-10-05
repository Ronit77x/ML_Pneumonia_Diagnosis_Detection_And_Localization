import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", default="data/split.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # RSNA has multiple rows for images with multiple boxes. Collapse to one
    # image-level record while preserving the first box for simple evaluation.
    agg = df.groupby("patientId", as_index=False).agg({
        "x": "first", "y": "first", "width": "first", "height": "first",
        "Target": "max"
    })

    # Keep only the two classes used by the supplied paper.
    agg = agg[agg["Target"].isin([0, 1])].copy()

    train, temp = train_test_split(
        agg, test_size=0.30, random_state=42, stratify=agg["Target"]
    )
    val, test = train_test_split(
        temp, test_size=1/3, random_state=42, stratify=temp["Target"]
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    out = pd.concat([train, val, test], ignore_index=True)
    out.to_csv(args.out, index=False)

    print(out.groupby(["split", "Target"]).size())
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
