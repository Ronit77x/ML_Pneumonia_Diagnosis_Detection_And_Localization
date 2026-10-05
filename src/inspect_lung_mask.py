import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from data import make_lung_mask, read_dicom


IMG_SIZE = 128


def main():
    image_dir = "data/stage_2_train_images"
    csv_path = "data/split.csv"

    df = pd.read_csv(csv_path)
    test = df[(df["split"] == "test") & (df["Target"] == 1)].drop_duplicates("patientId")

    patient_ids = [
        "8f8feef4-b211-4d70-83eb-cef152569c34",
        "855d0926-1ae6-403e-a08d-85bd8bae4941",
    ]

    os.makedirs("outputs/inspection", exist_ok=True)

    for patient_id in patient_ids:
        path = os.path.join(image_dir, patient_id + ".dcm")

        img = read_dicom(path)
        small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = make_lung_mask(small)
        masked = small * mask

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(small, cmap="gray")
        axes[0].set_title("X-ray")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Estimated Lung Mask")
        axes[1].axis("off")

        axes[2].imshow(masked, cmap="gray")
        axes[2].set_title("X-ray × Lung Mask")
        axes[2].axis("off")

        fig.suptitle(patient_id)
        plt.tight_layout()

        output = f"outputs/inspection/lung_mask_{patient_id}.png"
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close()

        print("saved:", output)


if __name__ == "__main__":
    main()
