import os
import math
import torch
import numpy as np
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score,
    classification_report,
)
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import nn, optim
from PIL import Image
import warnings
import logging
import random
import time
import copy
from collections import Counter

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ── Configuration ─────────────────────────────────────────────────────────────
PRETRAIN_SIZE = (224, 224)   # FER-Autism + FERAC are resized to this
AUGMENT_VAL   = True         # augment the val / early-stopping split
AUGMENT_PRETRAIN = False     # Stage-1 pretraining augmentation (off by default)
SVM_DOWNSCALE = None         # e.g. (48, 48) to make the linear SVM tractable
N_EPOCHS = 200
PATIENCE = 50

# ── Output directories ────────────────────────────────────────────────────────
for d in [
    "output224",
    "output224/WithAugmentation",
    "output224/confusion_matrices",
    "output224/convergence_curves",
    "output224/pretrained_weights",
    "output224/gradcam",
]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename="output224/asd_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

base_dir = os.path.dirname(os.path.abspath(__file__))


# ═════════════════════════════════════════════════════════════════════════════
# AUGMENTATION  —  8 transformations
# ═════════════════════════════════════════════════════════════════════════════
#  1. Horizontal flip                    (PIL)      p = 0.5
#  2. Random rotation  [-20, +20] deg    (PIL)      always
#  3. Colour jitter (brightness/contrast/saturation) (PIL)  always
#  4. Random perspective (corner perturbation)       (PIL)  p = 0.5
#  5. Gaussian blur, random kernel       (tensor)   p = 0.3
#  6. Flip blending, alpha in [0.1,0.3]  (tensor)   p = 0.3
#  7. Illumination gradient              (tensor)   p = 0.4
#  8. Gaussian noise, sigma = 15/255     (tensor)   p = 0.5, applied LAST
#
#  NOTE: every image passes through all eight STAGES, but six of them fire
#  probabilistically, so a typical output has 3-4 active transformations.
#
#  NOTE: saturation jitter is a mathematical no-op on single-channel
#  greyscale input; it is kept only for parity with the RGB pipeline.
# ═════════════════════════════════════════════════════════════════════════════

class RandomGaussianBlur:
    """(5) Gaussian smoothing with a randomly selected kernel size."""

    def __init__(self, kernel_sizes=(3, 5, 7), sigma=(0.1, 1.5), p=0.3):
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        k = random.choice(self.kernel_sizes)
        s = random.uniform(*self.sigma)
        return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[s, s])

    def __repr__(self):
        return f"RandomGaussianBlur(kernels={self.kernel_sizes}, p={self.p})"


class RandomFlipBlend:
    """(6) Blend the image with its horizontally flipped copy."""

    def __init__(self, alpha_range=(0.1, 0.3), p=0.3):
        self.alpha_range = alpha_range
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        a = random.uniform(*self.alpha_range)
        blended = (1.0 - a) * x + a * torch.flip(x, dims=[-1])
        return torch.clamp(blended, 0.0, 1.0)

    def __repr__(self):
        return f"RandomFlipBlend(alpha={self.alpha_range}, p={self.p})"


class IlluminationGradient:
    """(7) Smooth horizontal or vertical illumination ramp."""

    def __init__(self, strength=(0.15, 0.35), p=0.4):
        self.strength = strength
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        _, H, W = x.shape
        s = random.uniform(*self.strength)
        horizontal = random.random() < 0.5
        n = W if horizontal else H
        ramp = torch.linspace(1.0 - s, 1.0 + s, n, dtype=x.dtype,
                              device=x.device)
        if random.random() < 0.5:                 # randomise ramp direction
            ramp = torch.flip(ramp, dims=[0])
        ramp = ramp.view(1, 1, W) if horizontal else ramp.view(1, H, 1)
        return torch.clamp(x * ramp, 0.0, 1.0)

    def __repr__(self):
        return f"IlluminationGradient(strength={self.strength}, p={self.p})"


class AddGaussianNoise:
    """(8) Zero-mean Gaussian noise; sigma is given on the 0-255 scale."""

    def __init__(self, sigma=15.0, p=0.5):
        self.sigma_255 = sigma
        self.sigma = sigma / 255.0
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        return torch.clamp(x + torch.randn_like(x) * self.sigma, 0.0, 1.0)

    def __repr__(self):
        return f"AddGaussianNoise(sigma={self.sigma_255}/255, p={self.p})"


augment_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),                                # 1
    transforms.RandomRotation(20, fill=0),                                 # 2
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 3
    transforms.RandomPerspective(distortion_scale=0.15, p=0.5, fill=0),    # 4
    transforms.ToTensor(),
    RandomGaussianBlur(kernel_sizes=(3, 5, 7), p=0.3),                     # 5
    RandomFlipBlend(alpha_range=(0.1, 0.3), p=0.3),                        # 6
    IlluminationGradient(strength=(0.15, 0.35), p=0.4),                    # 7
    AddGaussianNoise(sigma=15.0, p=0.5),                                   # 8
])

base_transform = transforms.Compose([
    transforms.ToTensor(),
])

pretrain_transform = augment_transform if AUGMENT_PRETRAIN else base_transform
val_transform      = augment_transform if AUGMENT_VAL      else base_transform


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(y_true, y_pred, class_names, model_name,
                          save_dir="output224/confusion_matrices"):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Confusion Matrix - {model_name}", fontsize=14,
                 fontweight="bold", y=1.02)

    def _plot_heatmap(ax, data, fmt, title):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ticks = range(len(class_names))
        ax.set_xticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(class_names)
        thresh = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=9)

    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[0])
        axes[0].set_title("Raw Counts")
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[1])
        axes[1].set_title("Normalised (Row %)")
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("True Label")
    else:
        _plot_heatmap(axes[0], cm,      "d",   "Raw Counts")
        _plot_heatmap(axes[1], cm_norm, ".2f", "Normalised (Row %)")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved confusion matrix: {path}")


def save_emotion_confusion_matrix(y_true, y_pred, class_names, model_name,
                                  save_dir="output224/confusion_matrices"):
    """Per-emotion breakdown: one subplot per true class."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Emotion-level Confusion - {model_name}", fontsize=13,
                 fontweight="bold")

    for i, (ax, true_cls) in enumerate(zip(axes, class_names)):
        row   = cm[i]
        total = row.sum()
        bars  = ax.bar(class_names, row / total if total else row,
                       color=["#4CAF50" if j == i else "#F44336"
                              for j in range(n)])
        ax.set_title(f"True: {true_cls}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        for bar, cnt in zip(bars, row):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    str(int(cnt)), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_emotion_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved emotion confusion matrix: {path}")


def save_convergence_curves(history_dict, model_name,
                            save_dir="output224/convergence_curves"):
    """
    Two separate figures per model, matching the paper layout:
        {model_name}_loss.png       "Final Training Loss"
        {model_name}_accuracy.png   "Final Training Accuracy"
    Blue = training series, orange = validation series.
    """
    epochs = range(1, len(history_dict["train_loss"]) + 1)

    style = {
        "font.family":       "sans-serif",
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "axes.grid":         False,
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "lines.linewidth":   1.0,
    }

    panels = [
        ("loss", "train_loss", "val_loss", "Loss",
         "Final Training Loss", "loss"),
        ("acc",  "train_acc",  "val_acc",  "Accuracy",
         "Final Training Accuracy", "accuracy"),
    ]

    with plt.rc_context(style):
        for _, tr_key, va_key, ylabel, title, suffix in panels:
            train = history_dict.get(tr_key)
            if not train:
                continue

            fig, ax = plt.subplots(figsize=(5.2, 2.6))
            ax.plot(epochs, train, color="#1f77b4",
                    label=f"Train {ylabel}")

            val = history_dict.get(va_key)
            if val:
                ax.plot(range(1, len(val) + 1), val, color="#ff7f0e",
                        label=f"Validation {ylabel}")

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.margins(x=0.01)
            ax.legend(loc="best")

            path = os.path.join(save_dir, f"{model_name}_{suffix}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
            plt.close(fig)
            logging.info(f"Saved convergence curve: {path}")


MODEL_PRETTY = {
    "cnn":       "CNN",
    "resnet34":  "ResNet-34",
    "resnet50":  "ResNet-50",
    "mobilenet": "MobileNetV2",
    "alexnet":   "AlexNet",
}


def save_all_convergence_combined(all_histories,
                                  tag="fold",
                                  title_prefix="Fine-tune Convergence (Last Fold)",
                                  series="train",
                                  save_dir="output224/convergence_curves"):
    """
    Two figures, every model overlaid on one axes:
        all_models_{tag}_loss.png       "<title_prefix> - Loss"
        all_models_{tag}_accuracy.png   "<title_prefix> - Accuracy"

    series : "train" (default), "val", or "both".
             "both" draws the validation series dashed, since colour alone
             can no longer separate ten lines.
    """
    model_names = [k for k in all_histories if all_histories[k]]
    if not model_names:
        logging.warning(f"  Combined convergence [{tag}]: no histories, "
                        f"skipping.")
        return

    colors = ["#1f77b4", "#d62728", "#2ca02c",
              "#ff7f0e", "#9467bd", "#17becf"]
    color_map = {n: colors[i % len(colors)]
                 for i, n in enumerate(model_names)}

    style = {
        "font.family":       "sans-serif",
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "axes.grid":         False,
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "lines.linewidth":   1.0,
    }

    panels = [
        ("train_loss", "val_loss", "Loss",     "loss"),
        ("train_acc",  "val_acc",  "Accuracy", "accuracy"),
    ]

    with plt.rc_context(style):
        for tr_key, va_key, ylabel, suffix in panels:
            fig, ax = plt.subplots(figsize=(5.6, 2.8))
            drew = False

            for name in model_names:
                hist  = all_histories[name]
                c     = color_map[name]
                label = MODEL_PRETTY.get(name, name)

                if series in ("train", "both") and hist.get(tr_key):
                    y = hist[tr_key]
                    ax.plot(range(1, len(y) + 1), y, color=c,
                            label=f"{label} (train)" if series == "both"
                                  else label)
                    drew = True

                if series in ("val", "both") and hist.get(va_key):
                    y = hist[va_key]
                    ax.plot(range(1, len(y) + 1), y, color=c,
                            linestyle="--" if series == "both" else "-",
                            label=f"{label} (val)" if series == "both"
                                  else label)
                    drew = True

            if not drew:
                plt.close(fig)
                continue

            ax.set_title(f"{title_prefix} - {ylabel}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.margins(x=0.01)
            ax.legend(loc="best", ncol=1)

            path = os.path.join(save_dir,
                                f"all_models_{tag}_{suffix}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
            plt.close(fig)
            logging.info(f"Saved combined convergence plot: {path}")


def save_augmentation_examples(images, n=4,
                               save_dir="output224/WithAugmentation"):
    """Visual sanity check: original vs. 8-stage pipeline output."""
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    idxs = np.random.choice(len(images), size=n, replace=False)
    for r, i in enumerate(idxs):
        axes[r, 0].imshow(images[i], cmap="gray")
        axes[r, 0].set_title("Original", fontsize=8)
        axes[r, 0].axis("off")
        for c in range(1, 5):
            aug = augment_transform(images[i]).squeeze().numpy()
            axes[r, c].imshow(aug, cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"Augmented {c}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle("Augmentation pipeline (8 stochastic stages)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "augmentation_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved augmentation examples: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═════════════════════════════════════════════════════════════════════════════

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_images_filtered(data_path, allowed_classes, target_size=PRETRAIN_SIZE):
    """
    Pretrain corpus subset: ALWAYS resized to 224x224.

    Folder names are compared and returned lower-cased, so on-disk casing
    (e.g. 'Natural' vs 'natural') does not matter.
    """
    images, labels = [], []
    for class_name in sorted(os.listdir(data_path)):
        if class_name.lower() not in allowed_classes:
            continue
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                image = Image.open(image_path).convert("L").resize(
                    target_size, Image.BILINEAR)
                images.append(np.array(image))
                labels.append(class_name.lower())
            except Exception as e:
                logging.error(f"Error loading {image_path}: {e}")
    return np.array(images), np.array(labels)


def load_images_and_labels(data_path):
    """
    ASD corpus: images kept at their NATIVE resolution (no resize).

    A uniform spatial size is still required for batching. If the folder
    contains mixed sizes the images are conformed to the modal size and a
    warning is logged, because np.array() would otherwise produce an object
    array and DataLoader collation would fail.
    """
    raw, labels, sizes = [], [], []

    for class_name in sorted(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                image = Image.open(image_path).convert("L")
                raw.append(image)
                sizes.append(image.size)          # (W, H)
                labels.append(class_name)
            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")

    size_counts = Counter(sizes)
    modal_size, modal_n = size_counts.most_common(1)[0]

    if len(size_counts) > 1:
        logging.warning(
            f"ASD dataset contains {len(size_counts)} distinct image sizes; "
            f"conforming all images to the modal size {modal_size} (W,H) "
            f"({modal_n}/{len(raw)} already match). "
            f"Batching requires a uniform tensor shape.")
        for s, c in size_counts.most_common(10):
            logging.warning(f"    size {s}: {c} images")
        images = [np.array(im.resize(modal_size, Image.BILINEAR)
                           if im.size != modal_size else im) for im in raw]
    else:
        logging.info(f"ASD dataset image size is uniform: {modal_size} (W,H) "
                     f"- kept at native resolution, no resize applied.")
        images = [np.array(im) for im in raw]

    for im in raw:
        im.close()

    return np.array(images), np.array(labels)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD ASD DATA  (native resolution)
#
# This MUST come before the pretrain block: the pretrain label space is
# validated against `class_names`, which is defined here.
# ═════════════════════════════════════════════════════════════════════════════

dataset_path = os.path.join(
    base_dir, "Dataset/Kaggle_HandAnnotatedData_majorityVote_cropped")

all_images, all_labels = load_images_and_labels(dataset_path)

label_encoder      = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)
class_names        = list(label_encoder.classes_)
n_asd_classes      = len(class_names)

# (H, W) - required by SimpleCNN to size its first fully-connected layer
ASD_INPUT_SIZE = (all_images.shape[1], all_images.shape[2])

logging.info("\n" + "=" * 60)
logging.info("ASD DATASET")
logging.info("=" * 60)
logging.info(f"ASD classes ({n_asd_classes}): {class_names}")
logging.info(f"ASD total samples : {len(all_images)}")
logging.info(f"ASD image size    : {ASD_INPUT_SIZE} (H, W) - native")
unique, counts = np.unique(all_labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    logging.info(f"  {cls}: {cnt}")

# Scale-shift and architecture-minimum warnings
_h, _w = ASD_INPUT_SIZE
if (_h, _w) != PRETRAIN_SIZE:
    logging.warning(
        f"Pretraining resolution {PRETRAIN_SIZE} differs from the ASD "
        f"fine-tuning resolution {ASD_INPUT_SIZE}. The architectures tolerate "
        f"this (global / adaptive pooling), but the learned features were "
        f"calibrated at a different object scale, which may cost accuracy.")
if min(_h, _w) < 63:
    logging.warning(
        f"ASD images are {ASD_INPUT_SIZE}; AlexNet needs roughly >=63 px per "
        f"side before its adaptive pool degenerates. Results for AlexNet "
        f"should be treated with caution.")

save_augmentation_examples(all_images, n=4)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD & COMBINE FER-Autism + FERAC  (unified pretrain dataset @ 224x224)
#
# FER-Autism contributes all six emotion classes; FERAC contributes the four
# it shares. Both corpora are mapped onto the ASD label space so the two
# stages share an identical class vocabulary AND an identical index order.
# ═════════════════════════════════════════════════════════════════════════════

fer_autism_train_path = os.path.join(base_dir, "Dataset/FER_Autism/train")
ferac_train_path      = os.path.join(base_dir, "Dataset/FERAC/train")

# Folder names to keep (compared lower-cased; on-disk casing may differ)
FER_AUTISM_CLASSES = {"anger", "fear", "joy", "natural", "sadness", "surprise"}
FERAC_CLASSES      = {"anger", "fear", "joy", "natural"}

PRETRAIN_TO_ASD = {
    "anger":    "angry",
    "fear":     "fear",
    "joy":      "happy",
    "natural":  "neutral",
    "sadness":  "sad",
    "surprise": "surprise",
}

fer_images, fer_labels     = load_images_filtered(fer_autism_train_path,
                                                  FER_AUTISM_CLASSES)
ferac_images, ferac_labels = load_images_filtered(ferac_train_path,
                                                  FERAC_CLASSES)

combined_images = np.concatenate([fer_images, ferac_images], axis=0)
combined_labels = np.concatenate([fer_labels, ferac_labels], axis=0)
combined_labels = np.array([PRETRAIN_TO_ASD[l] for l in combined_labels])

assert sorted(set(combined_labels)) == sorted(class_names), (
    f"Pretrain label space {sorted(set(combined_labels))} does not match "
    f"ASD label space {sorted(class_names)}")

combined_label_encoder  = LabelEncoder()
combined_labels_encoded = combined_label_encoder.fit_transform(combined_labels)
n_combined_classes      = len(combined_label_encoder.classes_)

assert list(combined_label_encoder.classes_) == class_names, (
    "Encoder index order differs between pretrain and ASD label spaces")

combined_dataset = ImageDataset(combined_images, combined_labels_encoded,
                                transform=pretrain_transform)
combined_loader  = DataLoader(combined_dataset, batch_size=32, shuffle=True)

logging.info("\n" + "=" * 60)
logging.info("COMBINED PRETRAIN DATASET  (FER-Autism + FERAC)")
logging.info("=" * 60)
logging.info(f"Pretrain image size : {PRETRAIN_SIZE} (all images resized)")
logging.info(f"FER-Autism samples  : {len(fer_images)}  (6 classes)")
logging.info(f"FERAC samples       : {len(ferac_images)}  (4 classes)")
logging.info(f"Combined total      : {len(combined_images)}")
logging.info(f"Pretrain augmentation: {AUGMENT_PRETRAIN}")
logging.info(f"Label space aligned with ASD ({n_combined_classes}): {class_names}")
unique, counts = np.unique(combined_labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    logging.info(f"  {cls}: {cnt}")


# ── Stratified 10-set split ───────────────────────────────────────────────────
sets     = []
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for _, set_idx in splitter.split(np.arange(len(all_images)), all_labels_encoded):
    sets.append(set_idx)

val_indices_per_fold = [sets[(fold + 1) % 10] for fold in range(10)]
for i in range(10):
    for j in range(i + 1, 10):
        overlap = np.intersect1d(val_indices_per_fold[i],
                                 val_indices_per_fold[j])
        assert len(overlap) == 0, (
            f"Val overlap between fold {i+1} and {j+1}: {len(overlap)} samples")
logging.info("Val-set overlap check passed - all 10 val sets are disjoint.")


# ═════════════════════════════════════════════════════════════════════════════
# MODELS
# ═════════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_size):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            x = torch.zeros(1, 1, input_size[0], input_size[1])
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def get_cam_target_layer(self):
        return self.conv2


def _to_single_channel(conv):
    """
    Convert an RGB conv layer to single-channel while PRESERVING the
    ImageNet weights. For a greyscale input g, the RGB-equivalent response is
    sum_c W_c * g, so summing across the channel dimension is the correct
    reduction.
    """
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _load_backbone(name, pretrained):
    """torchvision version-tolerant loader."""
    ctor = getattr(models, name)
    try:
        weights = "IMAGENET1K_V1" if pretrained else None
        return ctor(weights=weights)
    except TypeError:
        return ctor(pretrained=pretrained)


def create_model(model_type, num_classes, pretrained=True):
    if model_type == "resnet34":
        model = _load_backbone("resnet34", pretrained)
        model.conv1 = _to_single_channel(model.conv1)
        model.fc    = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnet50":
        model = _load_backbone("resnet50", pretrained)
        model.conv1 = _to_single_channel(model.conv1)
        model.fc    = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "mobilenet":
        model = _load_backbone("mobilenet_v2", pretrained)
        model.features[0][0] = _to_single_channel(model.features[0][0])
        model.classifier[1]  = nn.Linear(
            model.classifier[1].in_features, num_classes)
    elif model_type == "alexnet":
        model = _load_backbone("alexnet", pretrained)
        model.features[0]   = _to_single_channel(model.features[0])
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def replace_head(model, model_type, num_classes):
    if model_type in ["resnet34", "resnet50"]:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "mobilenet":
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)
    elif model_type == "alexnet":
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes)
    return model


def freeze_backbone(model, model_type):
    for param in model.parameters():
        param.requires_grad = False
    if model_type in ["resnet34", "resnet50"]:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_type == "mobilenet":
        for param in model.classifier[1].parameters():
            param.requires_grad = True
    elif model_type == "alexnet":
        for param in model.classifier[6].parameters():
            param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def get_cam_target_layer(model, model_type):
    """Return the last conv layer suitable for GradCAM."""
    if model_type == "resnet34":
        return model.layer4[-1].conv2
    elif model_type == "resnet50":
        return model.layer4[-1].conv3
    elif model_type == "mobilenet":
        return model.features[-1][0]
    elif model_type == "alexnet":
        return model.features[-1]
    elif model_type == "cnn":
        return model.conv2
    return None


# ═════════════════════════════════════════════════════════════════════════════
# GRADCAM
# ═════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._hooks.append(target_layer.register_forward_hook(self._save_activation))
        try:
            self._hooks.append(
                target_layer.register_full_backward_hook(self._save_gradient))
        except AttributeError:
            self._hooks.append(
                target_layer.register_backward_hook(self._save_gradient))

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            self.model.zero_grad()
            logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


def save_gradcam_grid(model, model_type_or_name, loader, class_names,
                      device, n_samples=8,
                      save_dir="output224/gradcam", tag=""):
    target_layer = get_cam_target_layer(model, model_type_or_name)
    if target_layer is None:
        logging.warning(
            f"  GradCAM: no target layer for {model_type_or_name}, skipping.")
        return

    gradcam = GradCAM(model, target_layer)
    model.eval()

    images_shown, fig_rows = 0, []
    for imgs, lbls in loader:
        for i in range(imgs.size(0)):
            if images_shown >= n_samples:
                break
            img_t  = imgs[i:i + 1].to(device)
            true_c = lbls[i].item()
            cam, pred_c = gradcam(img_t)

            img_np = imgs[i].squeeze().cpu().numpy()
            H, W   = img_np.shape
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (W, H), Image.BILINEAR)) / 255.0

            fig_rows.append((img_np, cam_resized,
                             class_names[true_c], class_names[pred_c]))
            images_shown += 1
        if images_shown >= n_samples:
            break

    gradcam.remove_hooks()
    if not fig_rows:
        return

    ncols = 4
    nrows = int(np.ceil(len(fig_rows) / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    cmap_heat = plt.get_cmap("jet")   # FIX: matplotlib.cm.get_cmap removed >=3.9

    for idx, (img_np, cam_r, true_lbl, pred_lbl) in enumerate(fig_rows):
        row, pair = divmod(idx, 2)
        c_orig, c_cam = pair * 2, pair * 2 + 1

        axes[row, c_orig].imshow(img_np, cmap="gray")
        axes[row, c_orig].set_title(f"True: {true_lbl}\nPred: {pred_lbl}",
                                    fontsize=7)
        axes[row, c_orig].axis("off")

        overlay = np.stack([img_np] * 3, axis=-1)
        heat    = cmap_heat(cam_r)[:, :, :3]
        blended = np.clip(0.55 * overlay + 0.45 * heat, 0, 1)
        axes[row, c_cam].imshow(blended)
        axes[row, c_cam].set_title("GradCAM", fontsize=7)
        axes[row, c_cam].axis("off")

    for idx in range(len(fig_rows), nrows * 2):
        row, pair = divmod(idx, 2)
        axes[row, pair * 2].axis("off")
        axes[row, pair * 2 + 1].axis("off")

    model_label = (model_type_or_name if isinstance(model_type_or_name, str)
                   else type(model).__name__)
    fig.suptitle(f"GradCAM - {model_label} {tag}", fontsize=11,
                 fontweight="bold")
    plt.tight_layout()
    fname = f"{model_label}{('_' + tag) if tag else ''}_gradcam.png"
    path  = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved GradCAM: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def train_torch_model(model, train_loader, device,
                      n_epochs=200, lr=0.001,
                      optimizer_override=None,
                      val_loader=None, patience=50):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_override or optim.Adam(model.parameters(), lr=lr)
    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    t0            = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            _, preds  = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        history["train_loss"].append(run_loss / total if total else 0)
        history["train_acc"].append(correct / total if total else 0)

        if val_loader is not None:
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    if images.size(0) == 0:
                        continue
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    v_loss += loss.item() * images.size(0)
                    _, preds   = torch.max(outputs, 1)
                    v_correct += (preds == labels).sum().item()
                    v_total   += labels.size(0)

            v_loss_epoch = v_loss / v_total if v_total else 0
            v_acc_epoch  = v_correct / v_total if v_total else 0
            history["val_loss"].append(v_loss_epoch)
            history["val_acc"].append(v_acc_epoch)

            if v_loss_epoch < best_val_loss:
                best_val_loss = v_loss_epoch
                best_state    = copy.deepcopy(model.state_dict())
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logging.info(
                        f"    Early stop at epoch {epoch+1} "
                        f"(patience={patience}, "
                        f"best_val_loss={best_val_loss:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, time.perf_counter() - t0


def validate_torch_model(model, val_loader, device):
    model.eval()
    targets, preds = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, labels in val_loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            _, p = torch.max(model(images), 1)
            preds.extend(p.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return targets, preds, time.perf_counter() - t0


def _svm_features(imgs):
    """Flatten (optionally downscaled) images into an SVM feature matrix."""
    if SVM_DOWNSCALE is not None:
        imgs = torch.nn.functional.interpolate(
            imgs, size=SVM_DOWNSCALE, mode="bilinear", align_corners=False)
    return imgs.reshape(imgs.size(0), -1).cpu().numpy()


def train_svm(train_images, train_labels):
    flat = _svm_features(train_images)
    if flat.shape[1] > 20000:
        logging.warning(
            f"SVM feature dimension is {flat.shape[1]:,} raw pixels for "
            f"{flat.shape[0]:,} samples. Training will be slow and memory "
            f"heavy; consider setting SVM_DOWNSCALE=(48, 48).")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flat)
    clf    = svm.SVC(kernel='linear', C=1)
    clf.fit(scaled, train_labels.cpu().numpy())
    return clf, scaler


def svm_predict_loader(loader, clf, scaler):
    imgs = torch.cat([i for i, _ in loader if i.size(0) > 0])
    lbls = torch.cat([l for _, l in loader if l.size(0) > 0])
    scaled = scaler.transform(_svm_features(imgs))
    t0     = time.perf_counter()
    preds  = clf.predict(scaled)
    return lbls.cpu().numpy(), preds, time.perf_counter() - t0


# ═════════════════════════════════════════════════════════════════════════════
# RICH METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(targets, preds, class_names_list):
    targets = np.array(targets)
    preds   = np.array(preds)

    overall = {
        "accuracy":           accuracy_score(targets, preds),
        "balanced_accuracy":  balanced_accuracy_score(targets, preds),
        "macro_precision":    precision_score(targets, preds,
                                              average="macro", zero_division=0),
        "macro_recall":       recall_score(targets, preds,
                                           average="macro", zero_division=0),
        "macro_f1":           f1_score(targets, preds,
                                       average="macro", zero_division=0),
        "weighted_precision": precision_score(targets, preds,
                                              average="weighted",
                                              zero_division=0),
        "weighted_recall":    recall_score(targets, preds,
                                           average="weighted", zero_division=0),
        "weighted_f1":        f1_score(targets, preds,
                                       average="weighted", zero_division=0),
    }

    labels_all = list(range(len(class_names_list)))
    p_arr = precision_score(targets, preds, average=None,
                            labels=labels_all, zero_division=0)
    r_arr = recall_score(targets, preds, average=None,
                         labels=labels_all, zero_division=0)
    f_arr = f1_score(targets, preds, average=None,
                     labels=labels_all, zero_division=0)
    support = np.bincount(targets, minlength=len(class_names_list))

    per_class = {}
    for i, cname in enumerate(class_names_list):
        per_class[cname] = {
            "precision": float(p_arr[i]) if i < len(p_arr) else 0.0,
            "recall":    float(r_arr[i]) if i < len(r_arr) else 0.0,
            "f1":        float(f_arr[i]) if i < len(f_arr) else 0.0,
            "support":   int(support[i]),
        }

    return {"overall": overall, "per_class": per_class}


def log_metrics_block(label, metrics_dict):
    sep = "-" * 70
    logging.info(f"\n{sep}")
    logging.info(f"  Metrics - {label}")
    logging.info(sep)
    o = metrics_dict["overall"]
    for k, pretty in [
        ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_precision", "Macro Precision"), ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"), ("weighted_precision", "Weighted Precision"),
        ("weighted_recall", "Weighted Recall"), ("weighted_f1", "Weighted F1"),
    ]:
        logging.info(f"  {pretty:<28}: {o[k]:.4f}")
    logging.info(f"\n  {'Class':<18} {'Precision':>10} {'Recall':>10} "
                 f"{'F1':>10} {'Support':>10}")
    logging.info(f"  {'-'*60}")
    for cname, cv in metrics_dict["per_class"].items():
        logging.info(f"  {cname:<18} {cv['precision']:>10.4f} "
                     f"{cv['recall']:>10.4f} {cv['f1']:>10.4f} "
                     f"{cv['support']:>10d}")
    logging.info(sep)


# ═════════════════════════════════════════════════════════════════════════════
# STRATIFICATION & SPLIT UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def is_stratified(indices, labels_encoded, class_names_list, tol=0.05):
    global_counts = np.bincount(labels_encoded, minlength=len(class_names_list))
    global_props  = global_counts / global_counts.sum()
    split_counts  = np.bincount(labels_encoded[indices],
                                minlength=len(class_names_list))
    split_props   = split_counts / split_counts.sum()
    for i, (gp, sp) in enumerate(zip(global_props, split_props)):
        if abs(gp - sp) > tol:
            return False, i, gp, sp
    return True, -1, 0, 0


def round_half_up(n):
    """Round-half-up, not Python's banker's rounding. 75.5 -> 76."""
    return int(math.floor(n + 0.5))


def custom_stratified_split(indices_pool, labels_encoded, class_names_list,
                            test_fraction=0.10, seed=42):
    rng = np.random.RandomState(seed)
    train_idx_all, test_idx_all = [], []

    for class_idx in range(len(class_names_list)):
        class_mask = indices_pool[labels_encoded[indices_pool] == class_idx]
        n_test     = round_half_up(len(class_mask) * test_fraction)
        shuffled   = rng.permutation(class_mask)
        test_idx_all.extend(shuffled[:n_test])
        train_idx_all.extend(shuffled[n_test:])

    return np.array(train_idx_all), np.array(test_idx_all)


def log_split_summary(fold_label, train_idx, val_idx, test_idx,
                      labels_encoded, class_names_list):
    sep = "-" * 55
    logging.info(f"\n{sep}")
    logging.info(f"  Split summary - {fold_label}")
    logging.info(sep)
    for split_name, split_idx in [("train", train_idx),
                                  ("val",   val_idx),
                                  ("test",  test_idx)]:
        if split_idx is None:
            continue
        u, c = np.unique(labels_encoded[split_idx], return_counts=True)
        class_counts = ", ".join(
            f"{class_names_list[ci]}={cnt}" for ci, cnt in zip(u, c))
        logging.info(f"  {split_name:<6} : {len(split_idx):>5} samples  "
                     f"[ {class_counts} ]")
    logging.info(sep)


# ═════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL COST UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)


def log_cost(label, model_obj, train_time_s, infer_time_s,
             n_train, n_infer, extra_info=None):
    sep = "-" * 55
    logging.info(f"\n{sep}")
    logging.info(f"  Cost - {label}")
    logging.info(sep)
    logging.info(f"  Train time        : {train_time_s:.3f} s")
    logging.info(f"  Inference time    : {infer_time_s:.4f} s")
    if train_time_s > 0:
        logging.info(f"  Train throughput  : {n_train/train_time_s:.1f} samples/s")
    if infer_time_s > 0:
        logging.info(f"  Infer throughput  : {n_infer/infer_time_s:.1f} samples/s")
    if model_obj is not None:
        total_p, train_p = count_parameters(model_obj)
        logging.info(f"  Total params      : {total_p:,}")
        logging.info(f"  Trainable params  : {train_p:,}")
        logging.info(f"  Model size (fp32) : {get_model_size_mb(model_obj):.2f} MB")
    if extra_info:
        for k, v in extra_info.items():
            logging.info(f"  {k:<18}: {v}")
    if PSUTIL_AVAILABLE:
        ram = psutil.virtual_memory()
        logging.info(f"  CPU %             : {psutil.cpu_percent(interval=None):.1f}%")
        logging.info(f"  RAM used/total    : "
                     f"{ram.used/1e9:.2f} / {ram.total/1e9:.2f} GB")
    if torch.cuda.is_available():
        logging.info(f"  GPU mem alloc     : "
                     f"{torch.cuda.memory_allocated()/1e6:.1f} MB")
        logging.info(f"  GPU mem reserved  : "
                     f"{torch.cuda.memory_reserved()/1e6:.1f} MB")
    logging.info(sep)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 - Pretrain on combined FER-Autism + FERAC @ 224x224
# ═════════════════════════════════════════════════════════════════════════════

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEEP_MODELS = ["resnet34", "resnet50", "mobilenet", "alexnet"]

logging.info("\n" + "=" * 60)
logging.info("STAGE 1: Pretrain on COMBINED dataset (FER-Autism + FERAC)")
logging.info(f"  Input size: {PRETRAIN_SIZE}, Classes: {n_combined_classes}, "
             f"Epochs: 30, LR: 0.001")
logging.info("=" * 60)

for model_type in DEEP_MODELS:
    pt_path = (f"output224/pretrained_weights/"
               f"{model_type}_combined_pretrained.pth")
    if os.path.exists(pt_path):
        logging.info(f"  {model_type}: combined pretrain weights found, skipping.")
        continue
    logging.info(f"  Pretraining {model_type} on combined dataset...")
    model = create_model(model_type, num_classes=n_combined_classes,
                         pretrained=True)
    model, pt_hist, train_t = train_torch_model(
        model, combined_loader, device, n_epochs=30, lr=0.001)
    torch.save(model.state_dict(), pt_path)
    save_convergence_curves(pt_hist, f"{model_type}_stage1_combined")
    log_cost(f"{model_type} | Stage1 combined pretrain",
             model, train_t, 0.0, len(combined_images), 0)
    logging.info(f"  {model_type}: saved to {pt_path}")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 - Stratified 10-fold fine-tuning on ASD (native resolution)
#
# Per fold i (0-indexed):
#   test  = sets[i]               NOT augmented  (10 %)
#   val   = sets[(i+1) % 10]      augmented per AUGMENT_VAL   (10 %)
#   train = remaining 8 sets      augmented      (80 %)
# ═════════════════════════════════════════════════════════════════════════════

logging.info("\n" + "=" * 60)
logging.info("STAGE 2: Stratified 10-Fold Fine-tuning on ASD Dataset")
logging.info(f"  Input size: {ASD_INPUT_SIZE} (native, no resize)")
logging.info(f"  Max epochs: {N_EPOCHS}, Early-stop patience: {PATIENCE}")
logging.info(f"  Augment val/early-stop split: {AUGMENT_VAL}")
logging.info("=" * 60)

MODEL_KEYS = ["cnn", "svm", "resnet34", "resnet50", "mobilenet", "alexnet"]

fold_val_results  = {k: [] for k in MODEL_KEYS}
fold_test_results = {k: [] for k in MODEL_KEYS}
all_test_targets  = {k: [] for k in MODEL_KEYS}
all_test_preds    = {k: [] for k in MODEL_KEYS}
all_val_targets   = {k: [] for k in MODEL_KEYS}
all_val_preds     = {k: [] for k in MODEL_KEYS}

last_histories    = {k: None for k in
                     ["cnn", "resnet34", "resnet50", "mobilenet", "alexnet"]}
total_train_times = {k: 0.0 for k in MODEL_KEYS}


def record_metrics(key, fold, val_t, val_p, test_t, test_p):
    vm = compute_metrics(val_t,  val_p,  class_names)
    tm = compute_metrics(test_t, test_p, class_names)
    fold_val_results[key].append(vm)
    fold_test_results[key].append(tm)
    all_val_targets[key].extend(val_t)
    all_val_preds[key].extend(val_p)
    all_test_targets[key].extend(test_t)
    all_test_preds[key].extend(test_p)
    log_metrics_block(f"{key} | Fold {fold+1} - VAL",  vm)
    log_metrics_block(f"{key} | Fold {fold+1} - TEST", tm)


for fold in range(10):

    test_idx  = sets[fold]
    val_idx   = sets[(fold + 1) % 10]
    train_idx = np.concatenate(
        [sets[i] for i in range(10) if i != fold and i != (fold + 1) % 10])

    assert len(np.intersect1d(train_idx, val_idx))  == 0
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.intersect1d(val_idx,   test_idx)) == 0

    for split_name, split_idx in [("train", train_idx),
                                  ("val",   val_idx),
                                  ("test",  test_idx)]:
        ok, cls_idx, expected, got = is_stratified(
            split_idx, all_labels_encoded, class_names)
        assert ok, (f"Fold {fold+1}: {split_name} NOT stratified - class "
                    f"'{class_names[cls_idx]}' expected {expected:.3f}, "
                    f"got {got:.3f}")
        logging.info(f"  Fold {fold+1}: {split_name} stratification OK.")

    log_split_summary(f"Fold {fold+1}", train_idx, val_idx, test_idx,
                      all_labels_encoded, class_names)

    logging.info(f"\nFold {fold+1} | test=S{fold+1} ({len(test_idx)}) | "
                 f"val=S{(fold+1) % 10 + 1} ({len(val_idx)}) | "
                 f"train={len(train_idx)}")

    train_dataset = ImageDataset(all_images[train_idx],
                                 all_labels_encoded[train_idx],
                                 transform=augment_transform)
    val_dataset   = ImageDataset(all_images[val_idx],
                                 all_labels_encoded[val_idx],
                                 transform=val_transform)
    test_dataset  = ImageDataset(all_images[test_idx],
                                 all_labels_encoded[test_idx],
                                 transform=base_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # ── CNN ───────────────────────────────────────────────────────────────────
    cnn_model, cnn_hist, cnn_train_t = train_torch_model(
        SimpleCNN(n_asd_classes, ASD_INPUT_SIZE), train_loader, device,
        n_epochs=N_EPOCHS, lr=0.001,
        val_loader=val_loader, patience=PATIENCE)

    cnn_val_t,  cnn_val_p,  _           = validate_torch_model(
        cnn_model, val_loader,  device)
    cnn_test_t, cnn_test_p, cnn_infer_t = validate_torch_model(
        cnn_model, test_loader, device)

    record_metrics("cnn", fold, cnn_val_t, cnn_val_p, cnn_test_t, cnn_test_p)
    last_histories["cnn"] = cnn_hist
    log_cost(f"CNN | Fold {fold+1}", cnn_model,
             cnn_train_t, cnn_infer_t, len(train_idx), len(test_idx))
    total_train_times["cnn"] += cnn_train_t
    save_convergence_curves(cnn_hist, f"CNN_fold{fold+1}")

    # ── SVM ───────────────────────────────────────────────────────────────────
    svm_imgs = torch.cat([i for i, _ in train_loader if i.size(0) > 0])
    svm_lbls = torch.cat([l for _, l in train_loader if l.size(0) > 0])

    svm_t0 = time.perf_counter()
    svm_clf, svm_scaler = train_svm(svm_imgs, svm_lbls)
    svm_train_t = time.perf_counter() - svm_t0

    svm_val_t,  svm_val_p,  _           = svm_predict_loader(
        val_loader,  svm_clf, svm_scaler)
    svm_test_t, svm_test_p, svm_infer_t = svm_predict_loader(
        test_loader, svm_clf, svm_scaler)

    record_metrics("svm", fold, svm_val_t, svm_val_p, svm_test_t, svm_test_p)
    log_cost(f"SVM | Fold {fold+1}", None, svm_train_t, svm_infer_t,
             len(svm_imgs), len(test_idx),
             extra_info={
                 "Kernel":          "linear",
                 "Support vectors": len(svm_clf.support_vectors_),
                 "Feature dim":     int(svm_imgs.shape[-1] * svm_imgs.shape[-2]),
             })
    total_train_times["svm"] += svm_train_t

    # ── Deep models ───────────────────────────────────────────────────────────
    deep_models_this_fold = {}

    for model_type, key in [("resnet34",  "resnet34"),
                            ("resnet50",  "resnet50"),
                            ("mobilenet", "mobilenet"),
                            ("alexnet",   "alexnet")]:
        pt_path = (f"output224/pretrained_weights/"
                   f"{model_type}_combined_pretrained.pth")

        # pretrained=False: the Stage-1 checkpoint overwrites everything anyway
        model = create_model(model_type, num_classes=n_combined_classes,
                             pretrained=False)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        model = replace_head(model, model_type, n_asd_classes)

        freeze_backbone(model, model_type)
        head_opt = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
        model, _, warmup_t = train_torch_model(
            model, train_loader, device, n_epochs=5,
            optimizer_override=head_opt)

        unfreeze_all(model)
        model, hist, finetune_t = train_torch_model(
            model, train_loader, device,
            n_epochs=N_EPOCHS, lr=0.0001,
            val_loader=val_loader, patience=PATIENCE)

        val_t,  val_p,  _       = validate_torch_model(model, val_loader,  device)
        test_t, test_p, infer_t = validate_torch_model(model, test_loader, device)

        record_metrics(key, fold, val_t, val_p, test_t, test_p)
        last_histories[key] = hist
        log_cost(f"{model_type} | Fold {fold+1}", model,
                 warmup_t + finetune_t, infer_t, len(train_idx), len(test_idx))
        total_train_times[key] += warmup_t + finetune_t
        deep_models_this_fold[model_type] = model
        save_convergence_curves(hist, f"{model_type}_fold{fold+1}")

    torch.save(cnn_model.state_dict(), f"output224/cnn_fold{fold+1}_weights.pth")
    for mt in DEEP_MODELS:
        torch.save(deep_models_this_fold[mt].state_dict(),
                   f"output224/{mt}_fold{fold+1}_weights.pth")

    logging.info(f"  Generating GradCAM for fold {fold+1}...")
    save_gradcam_grid(cnn_model, "cnn", test_loader, class_names, device,
                      n_samples=8, tag=f"fold{fold+1}")
    for mt in DEEP_MODELS:
        save_gradcam_grid(deep_models_this_fold[mt], mt, test_loader,
                          class_names, device, n_samples=8,
                          tag=f"fold{fold+1}")


# ═════════════════════════════════════════════════════════════════════════════
# MEAN +/- STD ACROSS ALL 10 FOLDS
# ═════════════════════════════════════════════════════════════════════════════

def mean_std_metrics(results_list):
    metric_keys = list(results_list[0]["overall"].keys())
    agg_overall = {}
    for mk in metric_keys:
        vals = [r["overall"][mk] for r in results_list]
        agg_overall[mk] = {"mean": float(np.mean(vals)),
                           "std":  float(np.std(vals))}

    agg_per_class = {}
    for cname in results_list[0]["per_class"]:
        agg_per_class[cname] = {}
        for pm in ["precision", "recall", "f1"]:
            vals = [r["per_class"][cname][pm] for r in results_list]
            agg_per_class[cname][pm] = {"mean": float(np.mean(vals)),
                                        "std":  float(np.std(vals))}

    return {"overall": agg_overall, "per_class": agg_per_class}


def log_agg_metrics(label, agg):
    sep = "-" * 75
    logging.info(f"\n{sep}")
    logging.info(f"  Aggregate Metrics (mean +/- std) - {label}")
    logging.info(sep)
    o = agg["overall"]
    for mk in o:
        logging.info(f"  {mk:<28}: {o[mk]['mean']:.4f} +/- {o[mk]['std']:.4f}")
    logging.info(f"\n  {'Class':<18} {'Precision':>20} {'Recall':>20} {'F1':>20}")
    logging.info(f"  {'-'*78}")
    for cname, cv in agg["per_class"].items():
        p = f"{cv['precision']['mean']:.4f} +/- {cv['precision']['std']:.4f}"
        r = f"{cv['recall']['mean']:.4f} +/- {cv['recall']['std']:.4f}"
        f = f"{cv['f1']['mean']:.4f} +/- {cv['f1']['std']:.4f}"
        logging.info(f"  {cname:<18} {p:>20} {r:>20} {f:>20}")
    logging.info(sep)


logging.info("\n" + "=" * 70)
logging.info("MEAN +/- STD ACROSS 10 FOLDS")
logging.info("=" * 70)

avg_results = {}
for key in MODEL_KEYS:
    for split, results in [("val",  fold_val_results[key]),
                           ("test", fold_test_results[key])]:
        agg = mean_std_metrics(results)
        avg_results[f"{key}_{split}"] = agg
        log_agg_metrics(f"{key.upper()} | {split.upper()}", agg)

np.save("output224/WithAugmentation/aggregate_10fold_results.npy", avg_results)
for key in MODEL_KEYS:
    np.save(f"output224/WithAugmentation/{key}_fold_val_results.npy",
            fold_val_results[key])
    np.save(f"output224/WithAugmentation/{key}_fold_test_results.npy",
            fold_test_results[key])

logging.info("\n  Cumulative training time across 10 folds:")
for key in MODEL_KEYS:
    logging.info(f"    {key:<12}: {total_train_times[key]:>10.1f} s")


# ═════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRICES - aggregated over all 10 folds
# ═════════════════════════════════════════════════════════════════════════════

logging.info("\n" + "=" * 70)
logging.info("CONFUSION MATRICES  (aggregated across all 10 folds)")
logging.info("=" * 70)

MODEL_LABEL_MAP = [
    ("cnn",       "CNN"),
    ("svm",       "SVM"),
    ("resnet34",  "ResNet34"),
    ("resnet50",  "ResNet50"),
    ("mobilenet", "MobileNet"),
    ("alexnet",   "AlexNet"),
]

for key, label in MODEL_LABEL_MAP:
    for split, t_dict, p_dict in [
        ("test", all_test_targets, all_test_preds),
        ("val",  all_val_targets,  all_val_preds),
    ]:
        tag = f"{label}_all_folds_{split}"
        save_confusion_matrix(t_dict[key], p_dict[key], class_names, tag)
        save_emotion_confusion_matrix(t_dict[key], p_dict[key],
                                      class_names, tag)

#save_all_convergence_combined(last_histories)
save_all_convergence_combined(last_histories, tag="fold",
                              title_prefix="Fine-tune Convergence (Last Fold)")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION
#
#   • 90 % train  : presented TWICE per epoch - once original, once augmented
#   • 10 % test   : fixed held-out set, NO augmentation, evaluated ONCE
#   • Early stop  : internal hold-out carved from the 90 % pool, NEVER reported
# ═════════════════════════════════════════════════════════════════════════════

logging.info("\n" + "=" * 70)
logging.info("FINAL EVALUATION")
logging.info("  Train : 90% of dataset  (original + augmented, both passed)")
logging.info("  Test  : 10% held-out, NO augmentation")
logging.info("  Early stopping: internal hold-out from 90% pool (NOT reported)")
logging.info(f"  Max epochs : {N_EPOCHS}   |   Patience : {PATIENCE}")
logging.info("=" * 70)

# Deterministic per-class 10% allocation via round-half-up (not StratifiedKFold,
# whose remainder handling depends on shuffled fold order).
final_trainpool_idx, final_test_idx = custom_stratified_split(
    np.arange(len(all_images)), all_labels_encoded, class_names,
    test_fraction=0.10, seed=seed + 1)

final_train_idx, final_earlystop_idx = custom_stratified_split(
    final_trainpool_idx, all_labels_encoded, class_names,
    test_fraction=0.10, seed=seed + 2)

final_trainpool_idx  = np.sort(final_trainpool_idx)
final_test_idx       = np.sort(final_test_idx)
final_train_idx      = np.sort(final_train_idx)
final_earlystop_idx  = np.sort(final_earlystop_idx)

assert len(np.intersect1d(final_train_idx,     final_earlystop_idx)) == 0, \
    "Final eval: train/early-stop overlap"
assert len(np.intersect1d(final_train_idx,     final_test_idx))      == 0, \
    "Final eval: train/test overlap"
assert len(np.intersect1d(final_earlystop_idx, final_test_idx))      == 0, \
    "Final eval: early-stop/test overlap"

for split_name, split_idx in [
    ("train (90 % pool)",          final_train_idx),
    ("early-stop hold (internal)", final_earlystop_idx),
    ("test  (held-out)",           final_test_idx),
]:
    ok, cls_idx, expected, got = is_stratified(
        split_idx, all_labels_encoded, class_names)
    assert ok, (f"Final eval: '{split_name}' NOT stratified - class "
                f"'{class_names[cls_idx]}' expected {expected:.3f}, "
                f"got {got:.3f}")

sep70 = "-" * 70
logging.info(f"\n{sep70}")
logging.info("  FINAL EVALUATION - Data split summary")
logging.info(sep70)

_total = len(all_images)
for _name, _idx in [
    ("Train (orig+aug, 90% pool)", final_train_idx),
    ("Early-stop hold (internal)", final_earlystop_idx),
    ("Test  (held-out, no aug)",   final_test_idx),
]:
    _n   = len(_idx)
    _pct = 100.0 * _n / _total
    u, c = np.unique(all_labels_encoded[_idx], return_counts=True)
    _breakdown = "  ".join(f"{class_names[ci]}={cnt}" for ci, cnt in zip(u, c))
    logging.info(f"  {_name:<35} : {_n:>5} samples ({_pct:.1f}%)")
    logging.info(f"    Per-class support : {_breakdown}")

logging.info(f"\n  CNN / Deep models train set : {len(final_train_idx)} "
             f"originals + {len(final_train_idx)} augmented "
             f"= {2*len(final_train_idx)} total samples per epoch")
logging.info(f"  SVM train set               : {len(final_train_idx)} "
             f"original images only")
logging.info(sep70)

_train_orig = ImageDataset(all_images[final_train_idx],
                           all_labels_encoded[final_train_idx],
                           transform=base_transform)
_train_aug  = ImageDataset(all_images[final_train_idx],
                           all_labels_encoded[final_train_idx],
                           transform=augment_transform)
final_train_full_dataset = ConcatDataset([_train_orig, _train_aug])

# Early-stopping monitor follows AUGMENT_VAL (currently augmented)
final_earlystop_dataset = ImageDataset(all_images[final_earlystop_idx],
                                       all_labels_encoded[final_earlystop_idx],
                                       transform=val_transform)
final_test_dataset      = ImageDataset(all_images[final_test_idx],
                                       all_labels_encoded[final_test_idx],
                                       transform=base_transform)

final_train_loader     = DataLoader(final_train_full_dataset,
                                    batch_size=32, shuffle=True)
final_earlystop_loader = DataLoader(final_earlystop_dataset,
                                    batch_size=32, shuffle=False)
final_test_loader      = DataLoader(final_test_dataset,
                                    batch_size=32, shuffle=False)

_svm_orig_dataset = ImageDataset(all_images[final_train_idx],
                                 all_labels_encoded[final_train_idx],
                                 transform=base_transform)
_svm_loader = DataLoader(_svm_orig_dataset, batch_size=32, shuffle=False)

final_results = {}


def log_final_metrics(label, metrics_dict, n_train, n_test):
    sep = "-" * 70
    logging.info(f"\n{sep}")
    logging.info(f"  FINAL EVAL - {label}")
    logging.info(f"  Train samples used : {n_train:,}  "
                 f"(originals x2 for CNN/deep, x1 for SVM)")
    logging.info(f"  Test  samples      : {n_test:,}  (held-out, no aug)")
    logging.info(sep)
    o = metrics_dict["overall"]
    for k, pretty in [
        ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_precision", "Macro Precision"), ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"), ("weighted_precision", "Weighted Precision"),
        ("weighted_recall", "Weighted Recall"), ("weighted_f1", "Weighted F1"),
    ]:
        logging.info(f"  {pretty:<28}: {o[k]:.4f}")
    logging.info(f"\n  {'Class':<18} {'Precision':>10} {'Recall':>10} "
                 f"{'F1':>10} {'Support':>10}")
    logging.info(f"  {'-'*58}")
    for cname, cv in metrics_dict["per_class"].items():
        logging.info(f"  {cname:<18} {cv['precision']:>10.4f} "
                     f"{cv['recall']:>10.4f} {cv['f1']:>10.4f} "
                     f"{cv['support']:>10d}")
    logging.info(sep)


# ── CNN ───────────────────────────────────────────────────────────────────────
logging.info("\n  -- CNN (final eval) --")
final_cnn, final_cnn_hist, final_cnn_train_t = train_torch_model(
    SimpleCNN(n_asd_classes, ASD_INPUT_SIZE), final_train_loader, device,
    n_epochs=N_EPOCHS, lr=0.001,
    val_loader=final_earlystop_loader, patience=PATIENCE)

final_cnn_t, final_cnn_p, final_cnn_inf_t = validate_torch_model(
    final_cnn, final_test_loader, device)

final_results["cnn"] = compute_metrics(final_cnn_t, final_cnn_p, class_names)
log_final_metrics("CNN", final_results["cnn"],
                  2 * len(final_train_idx), len(final_test_idx))
log_cost("CNN | Final eval", final_cnn, final_cnn_train_t, final_cnn_inf_t,
         2 * len(final_train_idx), len(final_test_idx))
save_confusion_matrix(final_cnn_t, final_cnn_p, class_names, "CNN_final_eval_test")
save_emotion_confusion_matrix(final_cnn_t, final_cnn_p, class_names,
                              "CNN_final_eval_test")
save_convergence_curves(final_cnn_hist, "CNN_final_eval")
torch.save(final_cnn.state_dict(), "output224/cnn_final_eval_weights.pth")
save_gradcam_grid(final_cnn, "cnn", final_test_loader, class_names, device,
                  n_samples=16, tag="final_eval")

# ── SVM ───────────────────────────────────────────────────────────────────────
logging.info("\n  -- SVM (final eval) --")
svm_imgs_f = torch.cat([i for i, _ in _svm_loader if i.size(0) > 0])
svm_lbls_f = torch.cat([l for _, l in _svm_loader if l.size(0) > 0])
svm_t0 = time.perf_counter()
final_svm_clf, final_svm_scaler = train_svm(svm_imgs_f, svm_lbls_f)
final_svm_train_t = time.perf_counter() - svm_t0

final_svm_t, final_svm_p, final_svm_inf = svm_predict_loader(
    final_test_loader, final_svm_clf, final_svm_scaler)

final_results["svm"] = compute_metrics(final_svm_t, final_svm_p, class_names)
log_final_metrics("SVM", final_results["svm"],
                  len(final_train_idx), len(final_test_idx))
log_cost("SVM | Final eval", None, final_svm_train_t, final_svm_inf,
         len(svm_imgs_f), len(final_test_idx),
         extra_info={
             "Kernel":          "linear",
             "Support vectors": len(final_svm_clf.support_vectors_),
             "Feature dim":     int(svm_imgs_f.shape[-1] * svm_imgs_f.shape[-2]),
         })
save_confusion_matrix(final_svm_t, final_svm_p, class_names, "SVM_final_eval_test")
save_emotion_confusion_matrix(final_svm_t, final_svm_p, class_names,
                              "SVM_final_eval_test")

# ── Deep models ───────────────────────────────────────────────────────────────
final_deep_histories = {}

for model_type, key in [("resnet34",  "resnet34"),
                        ("resnet50",  "resnet50"),
                        ("mobilenet", "mobilenet"),
                        ("alexnet",   "alexnet")]:
    logging.info(f"\n  -- {model_type} (final eval) --")
    pt_path = (f"output224/pretrained_weights/"
               f"{model_type}_combined_pretrained.pth")

    model = create_model(model_type, num_classes=n_combined_classes,
                         pretrained=False)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model = replace_head(model, model_type, n_asd_classes)

    freeze_backbone(model, model_type)
    head_opt = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    model, _, warmup_t = train_torch_model(
        model, final_train_loader, device, n_epochs=5,
        optimizer_override=head_opt)

    unfreeze_all(model)
    model, hist, finetune_t = train_torch_model(
        model, final_train_loader, device,
        n_epochs=N_EPOCHS, lr=0.0001,
        val_loader=final_earlystop_loader, patience=PATIENCE)

    test_t, test_p, infer_t = validate_torch_model(
        model, final_test_loader, device)

    final_results[key] = compute_metrics(test_t, test_p, class_names)
    final_deep_histories[key] = hist

    log_final_metrics(model_type.upper(), final_results[key],
                      2 * len(final_train_idx), len(final_test_idx))
    log_cost(f"{model_type} | Final eval", model,
             warmup_t + finetune_t, infer_t,
             2 * len(final_train_idx), len(final_test_idx))
    save_confusion_matrix(test_t, test_p, class_names,
                          f"{model_type}_final_eval_test")
    save_emotion_confusion_matrix(test_t, test_p, class_names,
                                  f"{model_type}_final_eval_test")
    save_convergence_curves(hist, f"{model_type}_final_eval")
    torch.save(model.state_dict(),
               f"output224/{model_type}_final_eval_weights.pth")
    save_gradcam_grid(model, model_type, final_test_loader, class_names,
                      device, n_samples=16, tag="final_eval")
# ── Combined convergence across all models (final eval) ──────────────────────
final_all_histories = {"cnn": final_cnn_hist, **final_deep_histories}

save_all_convergence_combined(final_all_histories, tag="final_eval",
                                  title_prefix="Final Training")

np.save("output224/WithAugmentation/final_eval_histories.npy",
            final_all_histories)
logging.info("Saved final_eval_histories.npy")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

logging.info("\n" + "=" * 105)
logging.info(f"FINAL EVALUATION SUMMARY - TEST SET  "
             f"({len(final_test_idx):,} held-out images, NO augmentation)")
logging.info(f"  Train : {len(final_train_idx):,} originals + "
             f"{len(final_train_idx):,} augmented "
             f"= {2*len(final_train_idx):,} samples/epoch  (CNN / deep models)")
logging.info(f"  Train : {len(final_train_idx):,} original images  (SVM)")
logging.info("=" * 105)

hdr = ("  {:<12}  {:>8}  {:>10}  {:>8}  {:>8}  {:>8}"
       "  {:>10}  {:>9}  {:>9}")
logging.info(hdr.format("Model", "Acc", "Bal.Acc", "Mac-P", "Mac-R", "Mac-F1",
                        "Wgt-P", "Wgt-R", "Wgt-F1"))
logging.info("  " + "-" * 103)

for key, label in MODEL_LABEL_MAP:
    o = final_results[key]["overall"]
    logging.info(hdr.format(
        label,
        f"{o['accuracy']:.4f}", f"{o['balanced_accuracy']:.4f}",
        f"{o['macro_precision']:.4f}", f"{o['macro_recall']:.4f}",
        f"{o['macro_f1']:.4f}", f"{o['weighted_precision']:.4f}",
        f"{o['weighted_recall']:.4f}", f"{o['weighted_f1']:.4f}"))

logging.info("  " + "=" * 103)

logging.info("\n  -- Per-class breakdown (TEST SET) --")
pc_hdr = "  {:<12}  {:<18}  {:>10}  {:>10}  {:>10}  {:>10}"
for key, label in MODEL_LABEL_MAP:
    logging.info(f"\n  {label}  [test n={len(final_test_idx):,}]")
    logging.info(pc_hdr.format("", "Class", "Precision", "Recall",
                               "F1", "Support"))
    logging.info("  " + "-" * 70)
    for cname, cv in final_results[key]["per_class"].items():
        logging.info(pc_hdr.format("", cname,
                                   f"{cv['precision']:.4f}",
                                   f"{cv['recall']:.4f}",
                                   f"{cv['f1']:.4f}",
                                   str(cv['support'])))

np.save("output224/WithAugmentation/final_eval_results.npy", final_results)
logging.info("\nFinal eval results saved.")
logging.info("\nAll plots saved. Training complete.")
print("Done. Check output224/ for logs, confusion matrices, "
      "convergence curves, and GradCAM visualisations.")
