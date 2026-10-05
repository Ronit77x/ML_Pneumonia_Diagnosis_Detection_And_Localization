# Weakly Supervised Pneumonia Localization

## Machine Learning Mini-Project — UE24CS352A

This project implements a weakly supervised approach for Pneumonia classification and localization from chest X-ray images.

The model is trained using image-level Pneumonia/No-Pneumonia labels. Class Activation Mapping (CAM) is then used to obtain an approximate localization region without using pneumonia bounding boxes as direct training targets.

## Team

- PES1UG24CS549 — Ronit Prasad
- PES1UG24CS512 — Vaibhava L

---

## 1. Project Overview

The project consists of two main stages:

1. Pneumonia classification using a convolutional neural network.
2. Pneumonia localization using Class Activation Mapping (CAM).

The overall pipeline is:

DICOM image → preprocessing → CNN → pneumonia probability + feature maps → CAM → thresholding → chest-region constraint → connected components → predicted localization box → IoU evaluation

---

## 2. Requirements

The project was developed using:

- Python 3.13
- PyTorch
- OpenCV
- NumPy
- Pandas
- pydicom
- scikit-learn
- Matplotlib

A CUDA-compatible NVIDIA GPU can be used for faster training and inference, but the code can also be executed on CPU.

---

## 3. Setup Instructions

### Step 1: Clone the repository

```powershell
git clone https://github.com/Ronit77x/ML_Pneumonia_Diagnosis_Detection_And_Localization.git
cd ML_Pneumonia_Diagnosis_Detection_And_Localization
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3: Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install the required Python packages

```powershell
pip install torch torchvision torchaudio
pip install numpy pandas opencv-python pydicom scikit-learn matplotlib
```

If using an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build according to the installed PyTorch/CUDA configuration.

---

## 4. Dataset Setup

The project uses the RSNA Pneumonia Detection Challenge dataset.

The dataset is not included in this repository.

After downloading the dataset, place the DICOM images and label CSV in the local dataset directory expected by `prepare_data.py`.

Run:

```powershell
python src/prepare_data.py
```

This prepares the train, validation and test split used by the project.

### Dataset Split

The prepared project split contains 26,684 unique images:

| Split | Total | No Pneumonia | Pneumonia |
|---|---:|---:|---:|
| Train | 18,678 | 14,470 | 4,208 |
| Validation | 5,337 | 4,134 | 1,203 |
| Test | 2,669 | 2,068 | 601 |

The original RSNA annotations can contain multiple bounding-box rows for the same patient/image. In this implementation, annotations are collapsed by `patientId` using `Target=max`, and the first bounding box is retained.

---

## 5. Preprocessing

The DICOM images are:

- Read using pydicom.
- Corrected for MONOCHROME1 images when required.
- Percentile normalized.
- Resized from 1024 × 1024 to 128 × 128.
- Standardized using the image mean and standard deviation.

Training data may also use image augmentation.

---

## 6. Model

The classifier is a CAM-compatible convolutional neural network consisting of:

- 10 convolutional layers.
- 3 × 3 convolution filters.
- ReLU activations.
- Selected 2 × 2 max-pooling layers.
- Global Average Pooling.
- One final fully connected layer with two output classes.

The two classes are:

- No Pneumonia
- Pneumonia

Global Average Pooling and the final classification layer allow the final convolutional feature maps to be used for CAM generation.

---

## 7. Training

The implemented training configuration is:

- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 20
- Loss: CrossEntropyLoss
- Random seed: 42
- Input size: 128 × 128

To train the model, run:

```powershell
python src/train.py
```

The best validation checkpoint is saved as:

```text
outputs/checkpoints/best.pt
```

The trained checkpoint is not included in the GitHub repository.

---

## 8. Classification Evaluation

After training, classification performance can be evaluated using:

```powershell
python src/evaluate.py
```

This reports:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The final classification threshold selected using the validation set is:

```text
0.35
```

---

## 9. CAM-Based Localization

CAM is generated using the final convolutional feature maps and the Pneumonia-class fully connected weights.

The localization process is:

1. Generate the CAM.
2. Normalize the CAM.
3. Apply the selected CAM threshold.
4. Restrict the activation to a conservative chest region.
5. Find connected components.
6. Generate the predicted localization bounding box.
7. Compare the prediction with the ground-truth box using IoU.

The final CAM threshold selected using the validation set is:

```text
0.20
```

---

## 10. Localization Evaluation

To evaluate localization on the test set, run:

```powershell
python src/evaluate_localization.py
```

The evaluation uses:

```text
classification threshold = 0.35
CAM threshold = 0.20
```

The script reports:

- Mean IoU
- Median IoU
- Percentage of images with IoU ≥ 0.3
- Percentage of images with IoU ≥ 0.5
- Classification misses
- Images producing CAM boxes

---

## 11. Single Image Demo

To run the localization pipeline on an individual DICOM image:

```powershell
python src/localize.py --image "PATH_TO_A_DICOM_IMAGE" --checkpoint outputs/checkpoints/best.pt --threshold 0.20 --out outputs/demo_localization.png
```

Replace `PATH_TO_A_DICOM_IMAGE` with the path to an actual RSNA DICOM image.

The generated output contains:

1. Original X-ray.
2. CAM heatmap.
3. Predicted localization bounding box.

---

## 12. Qualitative Inspection

To inspect successful and failed localization examples:

```powershell
python src/inspect_localization.py
```

The generated inspection images can be found in the corresponding outputs directory.

---

## 13. Final Test Results

### Classification

| Metric | Result |
|---|---:|
| Accuracy | 78.57% |
| Precision | 51.85% |
| Recall | 67.72% |
| F1 Score | 58.73% |

### Localization

| Metric | Result |
|---|---:|
| Mean IoU | 0.1025 |
| Median IoU | 0.0646 |
| IoU ≥ 0.3 | 7.82% |
| IoU ≥ 0.5 | 0.50% |

### Confusion Matrix

| Actual \ Predicted | No Pneumonia | Pneumonia |
|---|---:|---:|
| No Pneumonia | 1690 | 378 |
| Pneumonia | 194 | 407 |

Therefore:

- TN = 1690
- FP = 378
- FN = 194
- TP = 407

---

## 14. Project Structure

```text
ML_Pneumonia_Diagnosis_Detection_And_Localization/
│
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cam_localizer.py
│   ├── evaluate_localization.py
│   ├── localize.py
│   ├── iou.py
│   ├── inspect_localization.py
│   ├── inspect_lung_mask.py
│   └── demo.py
│
├── docs/
│   ├── two_page_writeup.pdf
│   └── presentation.pptx
│
└── README.md
```

---

## 15. Reference Paper Difference

The reference paper uses U-Net lung segmentation and combines original and segmented images as input.

This implementation does not train a U-Net because a lung-mask training dataset was not available. Instead, the classifier operates on the normalized X-ray and a conservative chest-region constraint is applied during CAM post-processing.

---

## 16. Limitations

The main limitations of the implementation are:

- Weak image-level supervision.
- 128 × 128 input resolution.
- Approximate CAM localization.
- Classification false negatives.
- Only the first ground-truth box is retained when multiple RSNA annotations exist for one image.

---

## 17. Future Work

Possible improvements include:

- Higher-resolution input images.
- Transfer learning.
- Dedicated lung segmentation.
- Improved CAM post-processing.
- Multi-region localization.
- Better handling of multiple ground-truth pneumonia boxes.