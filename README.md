# Computer Vision – Homework 2: Image Classification with Transfer Learning
 
**Author:** Elia Brutti mat.VR553705
 
Classification of the 37 breeds of the **Oxford-IIIT Pet** dataset with a pre-trained
**MobileNetV3-Small**, comparing three transfer-learning strategies:
 
1. **Full fine-tuning** – the whole network is trained.
2. **Classifier-head training** – the convolutional feature extractor is frozen; only the
   two fully connected layers of the classifier (`classifier[0]`, 576→1024, and the new
   `classifier[3]`, 1024→37) are trained.
3. **Feature extraction + scikit-learn** – the network is fully frozen; 576-d features taken
   after the global average pooling are classified with a logistic regression.
   
## Environment
 
| Component | Version |
|---|---|
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| TorchVision | 0.23.0 |
| scikit-learn | 1.6.1 |
| NumPy | 2.0.2 |
| Matplotlib | 3.9.4 |
 
Experiments were run on a MacBook Pro with Apple Silicon GPU (MPS backend).
The code selects the device automatically (CUDA → MPS → CPU).
 
### macOS / Linux

(program tested on macOS)

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install the dependencies
pip install -r requirements.txt
```

### Windows

(not tested on Windows; standard commands for a virtual environment)

```bat
:: create and activate a virtual environment (Command Prompt)
python -m venv .venv
.venv\Scripts\activate

:: install the dependencies
pip install -r requirements.txt
```

If you use **PowerShell**, activate the environment with `.venv\Scripts\Activate.ps1` instead.
 
## Dataset
 
**The Oxford-IIIT Pet Dataset** – O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
*Cats and Dogs*, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
Official page: https://www.robots.ox.ac.uk/~vgg/data/pets/
 
License: the dataset is available for commercial/research purposes under a
**Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**
(https://creativecommons.org/licenses/by-sa/4.0/). The copyright remains with the
original owners of the images (as stated on the official dataset page).
 
The dataset is **not included** in this repository. It is downloaded automatically by
`torchvision.datasets.OxfordIIITPet` (about 800 MB) into `data/` on the first run.
 
Partitions:
- the official `trainval` partition (3680 images) is split into **train (85%, 3128 images)**
  and **validation (15%, 552 images)**, stratified by breed;
- the official `test` partition (3669 images) is used only once, for the final evaluation.

The split is saved in `splits/train.csv` and `splits/val.csv` (index, file name, label, breed)
and is re-used by all approaches. The number of images per breed in each split is in
`results/class_counts.csv`.
 
## File structure
 
```
Elia_Brutti_ComputerVision_Homework_2/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py               # entry point: runs one approach (--approach full | head | features), holds the hyperparameters
│   ├── data.py               # datasets, preprocessing, stratified train/val split, DataLoaders
│   ├── models.py             # MobileNetV3-Small with a 37-class head, freezing, feature extractor
│   ├── train.py              # approaches 1 and 2: training loop, evaluation, checkpoint selection
│   ├── features.py           # approach 3: feature extraction and logistic regression
│   ├── evaluate.py           # test metrics, confusion matrices, most confused pairs, examples
│   ├── make_report.py        # summary table and figures
│   └── utils.py              # seeds, device, saving JSON/CSV
├── data/                     # not included: downloaded automatically by torchvision
├── splits/
│   ├── train.csv             # generated: training images (index, file name, label, breed)
│   └── val.csv               # generated: validation images
└── results/
    ├── class_counts.csv      # generated: images per breed in each split
    ├── summary.csv           # generated: final comparison table
    ├── full/                 # generated: approach 1
    │   ├── config.json       #   hyperparameters, selected epoch, times, versions
    │   ├── history.csv       #   train/validation accuracy per epoch
    │   ├── best.pt           #   selected checkpoint
    │   ├── predictions.csv   #   test predictions
    │   ├── metrics.json      #   test accuracy and macro F1
    │   ├── confusion_matrix.png
    │   └── confused_pairs.csv
    ├── head/                 # generated: approach 2 (same files as full/)
    ├── features/             # generated: approach 3
    │   ├── config.json       #   classifier, selected C, times
    │   ├── val_scores.csv    #   validation accuracy for each C
    │   ├── predictions.csv
    │   ├── metrics.json
    │   ├── confusion_matrix.png
    │   ├── confused_pairs.csv
    │   └── examples.png      #   correct and wrong test predictions
    └── figures/
        ├── training_curves.png   # generated: approaches 1 and 2
        └── features_c_curve.png  # generated: approach 3, validation accuracy vs C
```
 
## How to reproduce
 
Run all commands from the repository root, with the virtual environment active.
 
```bash
# 1. Train / fit the three approaches (each saves its results in results/<approach>/)
python src/main.py --approach full
python src/main.py --approach head
python src/main.py --approach features
 
# 2. Test metrics, confusion matrices, confused pairs, example predictions
python src/evaluate.py
 
# 3. Summary table and figures
python src/make_report.py
```
 
`python src/data.py` alone (re)creates the split files and the class counts without training.
 
## Setup and hyperparameters
 
| | Full fine-tuning | Classifier head | Features + LogReg |
|---|---|---|---|
| Trainable parameters | 1,555,781 | 628,773 | 0 (network) + 21,349 (classifier) |
| Optimizer / loss | Adam / cross-entropy | Adam / cross-entropy | L-BFGS (scikit-learn default) / logistic loss |
| Learning rate | 1e-4 | 1e-3 | – |
| Batch size | 32 | 32 | 32 (feature extraction) |
| Epochs | 30 | 15 | – |
| Model selection | best validation accuracy | best validation accuracy | C ∈ {1e-4 … 10} by validation accuracy |
| Selected | epoch 26 | epoch 5 | C = 0.01 |
 
Preprocessing: validation and test images use the official preprocessing of the
pre-trained weights (`MobileNet_V3_Small_Weights.DEFAULT.transforms()`: resize to 256,
center crop 224, ImageNet normalization). Training images (approaches 1 and 2) use
`RandomResizedCrop(224)` and `RandomHorizontalFlip()` with the same normalization.
For approach 3 all splits use the deterministic evaluation preprocessing, because
features are extracted only once. The `StandardScaler` is fitted on training features only.
 
The test set is never used for training, early stopping, checkpoint selection or
hyperparameter choice.
 
## Results (test set)
 
| Approach | Test accuracy | Test macro F1 | Time |
|---|---|---|---|
| Full fine-tuning | 0.8457 | 0.8434 | 368 s |
| Classifier head | 0.8430 | 0.8401 | 118 s |
| Features + LogReg | 0.8577 | 0.8557 | 20 s |
 
Full details in `results/summary.csv`.
 
## Reproducibility notes
 
- Seeds are fixed to **42** for Python (`random`), NumPy and PyTorch (`src/utils.py`);
  the train/validation split uses `random_state=42` and is saved to file.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False` are set (effective on CUDA).
- Two runs with the same seed on the same Mac (MPS) gave identical per-epoch results.
  Results may still differ slightly on other hardware (e.g. CUDA vs MPS) or with
  different library versions.
- Training time includes the per-epoch validation pass.
- On macOS, NumPy with Apple Accelerate can emit spurious
  `RuntimeWarning: ... encountered in matmul` warnings in the logistic regression.
  The extracted features were checked to be finite, and only these warnings are
  suppressed in `src/features.py`.
 