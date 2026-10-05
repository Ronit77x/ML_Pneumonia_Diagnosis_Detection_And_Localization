from pathlib import Path

IMG_SIZE = 128
NUM_CLASSES = 2
CLASS_NAMES = ["No Pneumonia", "Pneumonia"]

DEFAULT_SEED = 42
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
