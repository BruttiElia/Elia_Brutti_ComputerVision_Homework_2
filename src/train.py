"""Training loop, evaluation and model selection for approaches 1 and 2."""

import torch
from pathlib import Path
import time


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch; return mean loss and accuracy on the training data."""
    model.train()   # training mode: dropout active, BatchNorm updates its statistics
    if not any(p.requires_grad for p in model.features.parameters()):
        model.features.eval()   # frozen backbone: keep its BatchNorm statistics fixed
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                   # reset the gradients of the previous batch
        outputs = model(images)                 # scores for the 37 breeds
        loss = criterion(outputs, labels)
        loss.backward()                         # compute gradients
        optimizer.step()                        # update the trainable weights

        total_loss += loss.item() * labels.size(0)     # loss is a batch mean: weight it by the batch size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)

    return total_loss / n, correct / n

@torch.no_grad() # no gradients: evaluation only
def evaluate(model, loader, device):
    """Predict on a loader without training; return true and predicted labels."""
    model.eval()                 # evaluation mode: no dropout, BatchNorm uses stored statistics      
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        all_preds.append(outputs.argmax(dim=1).cpu()) # predicted breed = highest score
        all_labels.append(labels)

    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, out_dir):
    """Train for num_epochs, keep the checkpoint with the best validation accuracy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"

    history = []         # one row per epoch, saved to history.csv
    best_val_acc = -1.0
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        y_val, p_val = evaluate(model, val_loader, device)
        val_acc = float((y_val == p_val).mean())                    

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc})
        print(f"Epoch {epoch}/{num_epochs}  train_loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:                # strict ">": on a tie the earlier epoch is kept                        
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)      # save the weights of the best epoch so far

    train_time = time.perf_counter() - start

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # reload the best epoch, not the last one
    return history, best_epoch, best_val_acc, train_time 