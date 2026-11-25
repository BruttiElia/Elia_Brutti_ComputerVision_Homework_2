"""MobileNetV3-Small with a 37-class head, and freezing for the three approaches."""

import torch.nn as nn
import torch
from torchvision import models

NUM_CLASSES = 37


def build_model(num_classes=NUM_CLASSES):
    """Pre-trained MobileNetV3-Small with the last layer replaced by a num_classes output."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT      # ImageNet-1K pre-trained weights
    model = models.mobilenet_v3_small(weights=weights)

    # Replace the last classifier layer (1024 -> 1000) with a new one (1024 -> num_classes)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)   # new layer, randomly initialized

    return model

def count_trainable_params(model):
    """Number of parameters that will be updated during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_backbone(model):
    """Freeze the convolutional feature extractor; only the classifier stays trainable."""
    for param in model.features.parameters():
        param.requires_grad = False # no gradient: these weights are not updated
    return model

def build_feature_extractor():
    """Frozen pre-trained backbone (features + avgpool) that maps an image to a 576-d vector."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    extractor = nn.Sequential(
        model.features, # convolutional blocks: [576, 7, 7]
        model.avgpool,  # global average pooling: [576, 1, 1]
        nn.Flatten(),   # -> 576-d vector
    )

    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval() # evaluation mode: BatchNorm uses the stored statistics

    return extractor


if __name__ == "__main__":
    # Quick check: trainable parameters (expected 1,555,781 and 628,773) and feature shape (2, 576)
    model = build_model()
    print(model.classifier)
    print("Trainable parameters (full):", count_trainable_params(model))

    freeze_backbone(model)
    print("Trainable parameters (head only):", count_trainable_params(model))

    extractor = build_feature_extractor()
    x = torch.rand(2, 3, 224, 224)
    print(extractor(x).shape)