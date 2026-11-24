"""Entry point: run one transfer-learning approach end to end."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from data import get_dataloaders
from models import build_model, freeze_backbone, count_trainable_params
from train import train_model, evaluate
from utils import set_seed, get_device, save_json, save_csv
from features import run_feature_extraction

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = {"full": 30, "head": 15}
LEARNING_RATES = {"full": 1e-4, "head": 1e-3}   # one learning rate per approach
RESULTS_DIR = Path("results")                    # where to save metrics and checkpoints


def parse_args():
    """Read the approach to run from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["full", "head", "features"], required=True)
    return parser.parse_args()

def run_finetuning(approach):
    """Approaches 1 (full) and 2 (head): train the network with PyTorch."""
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    train_loader, train_eval_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=BATCH_SIZE)

    model = build_model()
    if approach == "head":
        freeze_backbone(model)                                    
    model = model.to(device)

    n_trainable = count_trainable_params(model)
    print("Trainable parameters:", n_trainable)

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=LEARNING_RATES[approach]) 

    out_dir = RESULTS_DIR / approach              # results/full or results/head

    # 1. Training with model selection on the validation set
    history, best_epoch, best_val_acc, train_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS[approach], out_dir
    )
    save_csv(history, out_dir / "history.csv")

    # 2. Train accuracy under test conditions (no augmentation, eval mode), for the train/test gap
    y_tr, p_tr = evaluate(model, train_eval_loader, device)
    train_acc_eval = float((y_tr == p_tr).mean())
    print(f"Train accuracy (eval mode): {train_acc_eval:.4f}")

    # 3. Configuration and selected checkpoint (required for reproducibility)
    config = {
        "approach": approach,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS[approach],
        "learning_rate": LEARNING_RATES[approach],
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "trainable_params": n_trainable,
        "train_acc_eval": train_acc_eval,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_time_s": train_time,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    save_json(config, out_dir / "config.json")

    # 4. Single evaluation on the held-out test set
    y_test, p_test = evaluate(model, test_loader, device)
    predictions = [{"true": int(t), "pred": int(p)} for t, p in zip(y_test, p_test)]
    save_csv(predictions, out_dir / "predictions.csv")
    print(f"Best epoch: {best_epoch}  val_acc={best_val_acc:.4f}  time={train_time:.0f}s")



if __name__ == "__main__":
    args = parse_args()
    print("Approach:", args.approach)
    set_seed(SEED)
    if args.approach in ("full", "head"):
        run_finetuning(args.approach)
    else:
        run_feature_extraction(RESULTS_DIR / "features", batch_size=BATCH_SIZE)