"""Approach 3: frozen MobileNetV3 features + a scikit-learn classifier."""

import time
import torch
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import get_dataloaders
from models import build_feature_extractor
from utils import get_device, save_csv, save_json

C_VALUES = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10]      # regularization strengths tried on the validation set

# no gradients: the network is only used for inference
@torch.no_grad()
def extract_features(extractor, loader, device):
    """Pass every image through the frozen extractor; return features (N x 576) and labels (N)."""
    extractor.eval()    # evaluation mode: BatchNorm uses the stored ImageNet statistics
    all_feats, all_labels = [], []

    for images, labels in loader:
        feats = extractor(images.to(device))    # (batch, 576) vectors after global average pooling
        all_feats.append(feats.cpu())           # back to CPU: scikit-learn works on NumPy arrays   
        all_labels.append(labels)

    return torch.cat(all_feats).numpy(), torch.cat(all_labels).numpy()


def run_feature_extraction(out_dir, batch_size=32):
    """Approach 3: extract frozen features, pick C on validation, evaluate once on test."""
    device = get_device()
    _, train_eval_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=batch_size)     # augmented train loader not needed

    extractor = build_feature_extractor().to(device)

    # Feature extraction (same deterministic preprocessing for all splits)
    start = time.perf_counter()
    X_train, y_train = extract_features(extractor, train_eval_loader, device)
    X_val, y_val = extract_features(extractor, val_loader, device)
    X_test, y_test = extract_features(extractor, test_loader, device)
    extraction_time = time.perf_counter() - start
    print("Feature shapes:", X_train.shape, X_val.shape, X_test.shape)

    # Choose C on the validation set (scaler fitted on training data only, inside the pipeline)
    start = time.perf_counter()
    best_c, best_val_acc, best_clf = None, -1.0, None
    val_scores = []
      
    # NumPy on macOS can emit spurious floating-point warnings in matmul;
    # features were checked to be finite, so these warnings are silenced
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        for c in C_VALUES:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))   # standardize features, then logistic regression
            clf.fit(X_train, y_train)          # scaler and classifier fitted on training features only
            val_acc = clf.score(X_val, y_val)  # accuracy on the validation set       
            val_scores.append({"C": c, "val_acc": val_acc})
            print(f"C={c}  val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:         # strict ">": on a tie the C tried first is kept             
                best_c, best_val_acc, best_clf = c, val_acc, clf
        classifier_time = time.perf_counter() - start

        # Single evaluation on the held-out test set with the selected classifier
        p_test = best_clf.predict(X_test)

        # Train accuracy (same conditions as test), for the train/test gap
        train_acc_eval = float(best_clf.score(X_train, y_train))
    print(f"Best C: {best_c}  val_acc={best_val_acc:.4f}  train_acc={train_acc_eval:.4f}")

    # Save validation scores, configuration and test predictions
    save_csv(val_scores, out_dir / "val_scores.csv")
    config = {
        "approach": "features",
        "classifier": "LogisticRegression",
        "scaler": "StandardScaler (fitted on training features only)",
        "feature_dim": int(X_train.shape[1]),
        "c_values": C_VALUES,
        "best_c": best_c,
        "best_val_acc": best_val_acc,
        "train_acc_eval": train_acc_eval,
        "extraction_time_s": extraction_time,
        "classifier_time_s": classifier_time,
        "trainable_params": 0,     # no network parameter is trained
    }
    save_json(config, out_dir / "config.json")

    # One row per test image (true and predicted label), read by evaluate.py
    predictions = [{"true": int(t), "pred": int(p)} for t, p in zip(y_test, p_test)]
    save_csv(predictions, out_dir / "predictions.csv")