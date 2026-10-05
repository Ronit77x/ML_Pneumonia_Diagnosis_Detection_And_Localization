import argparse, os, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from data import RSNAPneumoniaDataset
from model import WeakPneumoniaCNN


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/checkpoints")
    args = ap.parse_args()

    seed_everything()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = RSNAPneumoniaDataset(df[df.split=="train"], args.image_dir, augment=True)
    val_ds = RSNAPneumoniaDataset(df[df.split=="val"], args.image_dir, augment=False)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    model = WeakPneumoniaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best = -1
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, None, device)
        print(f"epoch={epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "val_acc": va_acc}, os.path.join(args.out, "best.pt"))

    print("Best validation accuracy:", best)


if __name__ == "__main__":
    main()
