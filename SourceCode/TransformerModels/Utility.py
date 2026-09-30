import time
import torch
import torch.nn as nn
import os
import cv2
import math
import random
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import DrawGraph as dg
# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
USE_LINUX = False #False #True
USE_CPU_ONLY = False #True  # toggle here

USE_ASUS_LAPTOP = True # False #False  # toggle here

if USE_CPU_ONLY:
    DEVICE = torch.device("cpu")
    device_type = 'cpu'
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    MULTI_GPU = torch.cuda.device_count()
else:
    MULTI_GPU = 1    

# must be alphabetci sort oder: during dataset load from folder, it auto set in this way
# so to map it, we have to assign this way.
#EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise'] 
EMOTIONS = ['angry','fear','happy','neutral','sad','surprise'] 
NUM_CLASSES = len(EMOTIONS)

if USE_ASUS_LAPTOP:    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PRETRAIN_PATH = BASE_DIR / "Pretrain_DB"
    #TRAIN_DATA_PATH = BASE_DIR / "Kaggle_AugmentedTrain_Test_Dataset_MajorityVote" / "Train_Augmented"
    #TEST_DATA_PATH  = BASE_DIR / "Kaggle_AugmentedTrain_Test_Dataset_MajorityVote" / "Test"
    print("USE_ASUS_LAPTOP")
    TRAIN_DATA_PATH = BASE_DIR / "Kaggle_HandAnnotatedData_majorityVote_cropped"
    AUG_DATA_PATH   = BASE_DIR / "Kaggle_HandAnnotatedData_majorityVote_cropped_augmented"
    
    Pretrain_model_path = BASE_DIR / "Project" / "CV_80_10_10" / "hybrid_pretrained_model2.pth"
    Hybrid_Final_Model_Path = BASE_DIR / "Project" / "CV_80_10_10" / "Final_Hybrid_Model2.pth"
    Swin_Final_Model_Path = BASE_DIR / "Project" / "CV_80_10_10" / "Final_Swin_Model2.pth"
    Hybrid_PlotsDir = BASE_DIR / "Project" / "CV_80_10_10" / "HybridPlots2"
    Swin_PlotsDir = BASE_DIR / "Project" / "CV_80_10_10" / "SwinPlots2"
    Hybrid_LogFile = BASE_DIR / "Project" / "CV_80_10_10" / "results_hybrid2.txt"
    Swin_LogFile = BASE_DIR / "Project" / "CV_80_10_10" / "results_Swin2.txt"

    # Paths (mirror your existing Swin_ paths)
    ViT_LogFile          = BASE_DIR / "Project" / "CV_80_10_10" / "results_ViT.txt"
    ViT_PlotsDir         = BASE_DIR / "Project" / "CV_80_10_10" / "plots_ViT/"
    ViT_Final_Model_Path = BASE_DIR / "Project" / "CV_80_10_10" / "vit_final_model.pt"

    print("\nHelllooooooooooooooooooooooooooooooooooooooooooooooo USE_ASUS_LAPTOP")
    print(PRETRAIN_PATH)
    print(TRAIN_DATA_PATH)
    #print(TEST_DATA_PATH)    
    NUM_WORKER = 0
else:
    if USE_LINUX:
        BASE_DIR = "/home/bd_NBR/EM"          # For 221
        BASE_CV_DIR = BASE_DIR + "/CV_80_10_10"
        print("USE_LINUX............................................")

        Pretrain_model_path      = BASE_CV_DIR + "/hybrid_pretrained_model2.pth"
        Hybrid_Final_Model_Path  = BASE_CV_DIR + "/Final_Hybrid_Model2.pth"
        Swin_Final_Model_Path    = BASE_CV_DIR + "/Final_Swin_Model2.pth"

        Hybrid_PlotsDir = BASE_CV_DIR + "/HybridPlots2"
        Swin_PlotsDir   = BASE_CV_DIR + "/SwinPlots2"

        Hybrid_LogFile = BASE_CV_DIR + "/results_hybrid2.txt"
        Swin_LogFile   = BASE_CV_DIR + "/results_Swin2.txt"

        ViT_LogFile          = BASE_CV_DIR + "/results_ViT.txt"
        ViT_PlotsDir         = BASE_CV_DIR + "/plots_ViT"
        ViT_Final_Model_Path = BASE_CV_DIR + "/vit_final_model.pt"

        PRETRAIN_PATH = BASE_DIR + "/Pretrain_DB"
        TRAIN_DATA_PATH = BASE_DIR + "/Kaggle_HandAnnotatedData_majorityVote_cropped"
        AUG_DATA_PATH = BASE_DIR + "/Kaggle_HandAnnotatedData_majorityVote_cropped_augmented"

        NUM_WORKER = 4

IMG_SIZE = 224

os.makedirs(Hybrid_PlotsDir, exist_ok=True)
os.makedirs(Swin_PlotsDir, exist_ok=True)
os.makedirs(ViT_PlotsDir, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

class FoldStat:
    def __init__(self):
        self.val_acc_best = 0; self.val_precision_best = 0; self.val_recall_best = 0; self.val_f1_best = 0; self.macro_f1_best = 0; self.bal_acc_best = 0;
        self.test_acc_best = 0; self.test_precision_best = 0; self.test_recall_best = 0; self.test_f1_best = 0; self.test_macro_f1_best = 0; self.test_bal_acc_best = 0;
        self.best_epoch = 0;
        self.history = dg.InitFoldHistory()
    
    def UpdateEpoch(self, epoch):
        self.best_epoch = epoch

    def UpdateValStat(self, val_acc, val_precision, val_recall, val_f1, macro_f1, bal_acc):
        if(val_acc > self.val_acc_best):
            self.val_acc_best = val_acc
        if(val_precision > self.val_precision_best):
            self.val_precision_best = val_precision
        if(val_recall > self.val_recall_best):
            self.val_recall_best = val_recall
        if(val_f1 > self.val_f1_best):
            self.val_f1_best = val_f1
        if(macro_f1 > self.macro_f1_best):
            self.macro_f1_best = macro_f1
        if(bal_acc > self.bal_acc_best):
            self.bal_acc_best = bal_acc

    def UpdateTestStat(self, test_acc, test_precision, test_recall, test_f1, test_bal_acc, test_macro_f1):
        if(test_acc > self.test_acc_best):
            self.test_acc_best = test_acc
        if(test_precision > self.test_precision_best):
            self.test_precision_best = test_precision
        if(test_recall > self.test_recall_best):
            self.test_recall_best = test_recall
        if(test_f1 > self.test_f1_best):
            self.test_f1_best = test_f1
        if(test_bal_acc > self.test_bal_acc_best):
            self.test_bal_acc_best = test_bal_acc
        if(test_macro_f1 > self.test_macro_f1_best):
            self.test_macro_f1_best = test_macro_f1

class TrainingStat:
    def __init__(self):        
        self.fold_acc = []
        self.fold_precision = []
        self.fold_recall = []
        self.fold_f1 = []
        self.fold_macro_f1 = []
        self.fold_bal_acc = []

        self.fold_test_acc = []
        self.fold_test_precision = []
        self.fold_test_recall = []
        self.fold_test_f1 = []
        self.fold_test_macro_f1 = []
        self.fold_test_bal_acc = []

        self.best_epochs = []

    def UpdateEpochs(self, best_epoch):
        self.best_epochs.append(best_epoch)

    def UpdateFoldValidateStatistics(self, Hyb_fStat: FoldStat): #val_acc_best, val_precision_best, val_recall_best, val_f1_best, macro_f1_best, bal_acc_best):
        self.fold_acc.append(Hyb_fStat.val_acc_best)
        self.fold_precision.append(Hyb_fStat.val_precision_best)
        self.fold_recall.append(Hyb_fStat.val_recall_best)
        self.fold_f1.append(Hyb_fStat.val_f1_best)
        self.fold_macro_f1.append(Hyb_fStat.macro_f1_best)
        self.fold_bal_acc.append(Hyb_fStat.bal_acc_best)

    def UpdateFoldTestStatistics(self, Hyb_fStat: FoldStat): #test_acc_best, test_precision_best, test_recall_best, test_f1_best, test_macro_f1_best, test_bal_acc_best):
        self.fold_test_acc.append(Hyb_fStat.test_acc_best)
        self.fold_test_precision.append(Hyb_fStat.test_precision_best)
        self.fold_test_recall.append(Hyb_fStat.test_recall_best)
        self.fold_test_f1.append(Hyb_fStat.test_f1_best)
        self.fold_test_macro_f1.append(Hyb_fStat.test_macro_f1_best)
        self.fold_test_bal_acc.append(Hyb_fStat.test_bal_acc_best)

def WriteFoldSummary(filename, fold, fstate: FoldStat):
    with open(filename, "a") as f:   # 'a' = append
        #######Table-style output
        f.write(f"\n{'='*80}\n")
        f.write(f"FOLD: {fold} | Best Epoch: {fstate.best_epoch}\n")
        f.write(f"{'='*80}\n")

        # Header
        f.write(
            f"{'Dataset':<15} {'Accuracy':<10} {'Bal Acc':<10} "
            f"{'Precision':<10} {'Recall':<10} {'F1 (W)':<10} {'F1 (M)':<10}\n"
        )
        f.write(f"{'-'*80}\n")

        # Validation row
        f.write(
            f"{'Validation':<15} "
            f"{fstate.val_acc_best:<10.4f} {fstate.bal_acc_best:<10.4f} "
            f"{fstate.val_precision_best:<10.4f} {fstate.val_recall_best:<10.4f} "
            f"{fstate.val_f1_best:<10.4f} {fstate.macro_f1_best:<10.4f}\n"
        )

        # Test row
        f.write(
            f"{'Test Set':<15} "
            f"{fstate.test_acc_best:<10.4f} {fstate.test_bal_acc_best:<10.4f} "
            f"{fstate.test_precision_best:<10.4f} {fstate.test_recall_best:<10.4f} "
            f"{fstate.test_f1_best:<10.4f} {fstate.test_macro_f1_best:<10.4f}\n"
        )

        f.write(f"{'='*80}\n")

def WriteCrossValidationSummary(filename, tstat: TrainingStat, elapsed_time, KFOLD):
    ######Table-style cross-validation summary
    with open(filename, "a") as f:
        f.write(f"\n{'='*90}\n")
        f.write(f"{KFOLD}-FOLD CROSS VALIDATION RESULTS (MEAN)\n")
        f.write(f"{'='*90}\n")

        # Header
        f.write(
            f"{'Dataset':<20} {'Accuracy':<12} {'Bal Acc':<12} "
            f"{'Precision':<12} {'Recall':<12} {'F1 (W)':<12} {'F1 (M)':<12}\n"
        )
        f.write(f"{'-'*90}\n")

        # Validation mean row
        f.write(
            f"{'Validation (Mean)':<20} "
            f"{np.mean(tstat.fold_acc):<12.4f} {np.mean(tstat.fold_bal_acc):<12.4f} "
            f"{np.mean(tstat.fold_precision):<12.4f} {np.mean(tstat.fold_recall):<12.4f} "
            f"{np.mean(tstat.fold_f1):<12.4f} {np.mean(tstat.fold_macro_f1):<12.4f}\n"
        )

        # Held-out mean row
        f.write(
            f"{'Test Set (Mean)':<20} "
            f"{np.mean(tstat.fold_test_acc):<12.4f} {np.mean(tstat.fold_test_bal_acc):<12.4f} "
            f"{np.mean(tstat.fold_test_precision):<12.4f} {np.mean(tstat.fold_test_recall):<12.4f} "
            f"{np.mean(tstat.fold_test_f1):<12.4f} {np.mean(tstat.fold_test_macro_f1):<12.4f}\n"
        )

        f.write(f"{'-'*90}\n")
        f.write(f"{'Total Time:':<20} {format_time(elapsed_time)}\n")
        f.write(f"{'='*90}\n")

def is_parallel(model):
    """Check if model is wrapped in DataParallel or DDP."""
    return isinstance(model, (nn.DataParallel, 
                               nn.parallel.DistributedDataParallel))

def unwrap_model(model):
    """Always get the raw model, regardless of GPU wrapper."""
    return model.module if is_parallel(model) else model

#second idea based helper functions
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

import re

def extract_base_id(filepath):
    """
    Handles both patterns:
    '47.jpg'            → '47'
    'Autistic.1001.jpg' → 'Autistic.1001'
    """
    filename = os.path.basename(filepath)          # '47.jpg' or 'Autistic.1001.jpg'
    name     = os.path.splitext(filename)[0]       # '47'     or 'Autistic.1001'
    return name

class AugmentedSubset(Dataset):
    def __init__(self, dataset, indices, aug_map, transform=None):
        self.dataset = dataset
        self.samples = []
        self.transform = transform

        for idx in indices:
            path, label = dataset.samples[idx]
            self.samples.append((path, label))

            base_id = dataset.base_ids[idx]

            if base_id in aug_map:
                for aug_path in aug_map[base_id]:
                    self.samples.append((aug_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.dataset.loader_fn(path)
        if self.transform:
            img = self.transform(img)       # apply THIS subset's transform        
        return img, label

class CleanSubset(Dataset):
    """For val/test — deterministic transform only, no aug_map."""
    def __init__(self, dataset, indices, transform=None):
        self.dataset   = dataset
        self.transform = transform
        self.samples   = [dataset.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.dataset.loader_fn(path)
        if self.transform:
            img = self.transform(img)
        return img, label        
        
def BuildAugmentationMap(aug_root, classes):
    aug_map = {}

    for emotion in classes:
        folder = os.path.join(aug_root, emotion)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if "_aug" not in fname:
                continue

            base_id = fname.split('_aug')[0]

            if base_id not in aug_map:
                aug_map[base_id] = []

            aug_map[base_id].append(os.path.join(folder, fname))

    return aug_map
# transforms/pipelines.py
def get_train_transformSwin(img_size=IMG_SIZE):
    return transforms.Compose([
        # ── PIL space (must be before ToTensor) ──────────────
        transforms.Resize((img_size, img_size)),   # deterministic
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #transforms.RandomRotation(5),
        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        # ── convert once ─────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
# Transform — same as Swin (ViT-Base also expects 224x224, ImageNet normalisation)
def get_train_transformViT():
    return get_train_transformSwin()   # identical — reuse existing

def get_train_transformHybrid(img_size=IMG_SIZE):
    return transforms.Compose([
        # ── PIL space (must be before ToTensor) ──────────────
        transforms.Resize((img_size, img_size)),   # deterministic
        #transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # replaces Resize+Crop
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        # ── convert once ─────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),

        # ── Tensor space only ─────────────────────────────────
        #transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

# def get_train_transform(img_size=IMG_SIZE):
#     return transforms.Compose([
#         # ── PIL space (must be before ToTensor) ──────────────
#         transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),  # replaces Resize+Crop
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.RandomRotation(5),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),

#         # ── convert once ─────────────────────────────────────
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std =[0.229, 0.224, 0.225]
#         ),

#         # ── Tensor space only ─────────────────────────────────
#         transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
#     ])

def get_val_test_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),   # deterministic
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
def LoadInputRawImage(path):
    # NO transform here — transform assigned per subset later
    dataset = ImageFolder(path, transform=None)
    dataset.targets  = [s[1] for s in dataset.samples]
    dataset.paths    = [s[0] for s in dataset.samples]
    dataset.base_ids = [extract_base_id(p) for p in dataset.paths]
    # dataset.base_ids = [
    #     os.path.basename(p).split('.')[0] for p in dataset.paths
    # ]
    dataset.loader_fn = dataset.loader
    return dataset

def MergeDatasets(dataset_a, dataset_b):
    """
    Merge two ImageFolder-style datasets loaded via LoadInputRawImage.
    Preserves all attributes needed for fold splitting.
    """
    import copy

    # verify both have same classes in same order
    assert dataset_a.classes == dataset_b.classes, \
        f"Class mismatch!\nA: {dataset_a.classes}\nB: {dataset_b.classes}"

    merged = copy.copy(dataset_a)           # shallow copy base structure

    # merge all attributes your pipeline uses
    merged.samples  = dataset_a.samples  + dataset_b.samples
    merged.targets  = dataset_a.targets  + dataset_b.targets
    merged.paths    = dataset_a.paths    + dataset_b.paths
    merged.base_ids = dataset_a.base_ids + dataset_b.base_ids

    merged.transform  = None               # always set per-subset later
    merged.loader_fn  = dataset_a.loader_fn

    print(f"Dataset A : {len(dataset_a):>6} images")
    print(f"Dataset B : {len(dataset_b):>6} images")
    print(f"Merged    : {len(merged):>6} images")
    print(f"Classes   : {merged.classes}")

    return merged

from PIL import Image, ImageDraw, ImageFont

def render_summary_to_png(text, image_filename, font_size=16, padding=24, line_spacing=6,
                           bg_color=(255, 255, 255), text_color=(15, 23, 42)):
    """
    Renders a block of plain (monospace-aligned) text as a PNG image,
    auto-sizing the canvas to fit every line.
    """
    lines = text.split("\n")

    # Try a few common monospace fonts, falling back to PIL's default if none exist
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
        "C:/Windows/Fonts/consola.ttf",                          # Windows (Consolas)
        "/System/Library/Fonts/Menlo.ttc",                       # macOS
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    scratch = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(scratch)
    bbox = font.getbbox("Hg")
    line_height = (bbox[3] - bbox[1]) + line_spacing
    max_width = max((draw.textlength(line, font=font) for line in lines), default=0)

    img_w = int(max_width) + padding * 2
    img_h = int(line_height * len(lines)) + padding * 2

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=text_color)
        y += line_height

    img.save(image_filename)
    print("Model summary image saved:", image_filename)
