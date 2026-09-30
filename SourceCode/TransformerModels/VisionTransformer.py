'''
 Vision Transformer — ViT-Base/16 (google/vit-base-patch16-224-in21k)
 =====================================================================

 Model: google/vit-base-patch16-224-in21k
 Source: https://huggingface.co/google/vit-base-patch16-224-in21k

 Architecture:
   Patch size  : 16 x 16 pixels
   Image size  : 224 x 224
   Num patches : (224/16)^2 = 196 patch tokens
   CLS token   : 1
   Total tokens: 197
   Hidden dim  : 768
   Depth       : 12 transformer encoder layers
   Num heads   : 12
   MLP ratio   : 4x  (FFN hidden dim = 3072)
   Parameters  : ~86M

 Pretraining:
   Dataset     : ImageNet-21K (14M images, 21,841 classes)
   Advantage   : Richer semantic features vs ImageNet-1K
                 Better zero-shot transfer to specialist domains (faces, medical)

 Output shapes:
   backbone(pixel_values=x).last_hidden_state → [B, 197, 768]
     Index 0     : CLS token   — global image representation
     Index 1-196 : Patch tokens — local spatial features (14x14 grid)

 Comparison vs Swin-T (swin_tiny_patch4_window7_224):
   Swin-T  → [B, 768]         (globally pooled, hierarchical windows)
             → unsqueeze(1) → [B, 1, 768]  (single token context for decoder)
   ViT-B   → [B, 197, 768]   (all patch tokens preserved)
             → NO unsqueeze  (197-token rich spatial context for decoder)

 All downstream modules (EmotionQueryTokens, EmotionDecoder,
 ContrastiveModule, EmotionClassifier) are IDENTICAL to Swin-T version.
 hidden_size = 768 = Swin-T final_dim → zero changes to head architecture.

 REQUIRE:
   pip install transformers
 '''
import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Use the 4th GPU
import cv2
import math
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

#OLD
#from torch.amp import autocast, GradScaler #gpu
#from torch.cuda.amp import autocast, GradScaler #cpu
#NEW
import torch
from torch import autocast
from torch.cuda.amp import GradScaler

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import copy

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pathlib import Path
import torch.nn.functional as F
from torchinfo import summary

#-------------------
# Plot related
#------------------
from DrawGraph import *
import time

#import other python local files
import Utility as util

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------

IMG_SIZE = util.IMG_SIZE
BATCH_SIZE = 64*util.MULTI_GPU #16

EPOCHS_FINETUNE = 300
KFOLD = 10
Global_Patience = 25   # For early stoopping
Embedding_Dimension = 768  # ViT-Base hidden_size = 768 (same as Swin-T final_dim)
LABEL_SMOOTHING = 0.02 # base 0.02
LOSS_CNTS = 0.4 #0.5 #base 0.5

# ----------------------------------------------------------
# STEP 1 FACE DETECTION + ALIGNMENT
# ----------------------------------------------------------

def FaceDetectionAndAlignment(image):
    # simplified placeholder
    # real implementation can use RetinaFace or MTCNN
    img = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    return img
 # ----------------------------------------------------------
 # STEP 2 EMOTION REGION ENHANCEMENT MODULE
 # ----------------------------------------------------------
class EmotionRegionEnhancement(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv = nn.Sequential(
             nn.Conv2d(channels,16,3,padding=1), #Extract facial features: learn spatial patterns
             nn.ReLU(), #Non-linear activation: improve representation
             nn.Conv2d(16,channels,1), #Generate attention map: region importance
             nn.Sigmoid() #Normalize to 0-1: attention weighting
        )

    def forward(self,x):
 
        attention = self.conv(x)
        enhanced = x * attention #Enhance important regions: emphasize emotional cues
        return enhanced

# ----------------------------------------------------------
# EMOTION QUERY TOKENS
# ----------------------------------------------------------
class EmotionQueryTokens(nn.Module):
    def __init__(self, dim, num_queries=util.NUM_CLASSES):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.base_queries = nn.Parameter(torch.randn(num_queries, dim))
        self.feature_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.3)        # ADD: prevent gate from overfitting

    def forward(self, features):
        B, N, D = features.shape
        global_feat = features.mean(dim=1)
        context = self.feature_proj(global_feat)
        context = self.dropout(context)        # ADD: drop here before gating
        gate = self.gate(context).unsqueeze(1)
        queries = self.base_queries.unsqueeze(0).repeat(B, 1, 1)
        queries = queries + gate * context.unsqueeze(1)
        queries = self.dropout(queries)        # ADD: drop adapted queries too
        return queries
class EmotionDecoder(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.3):   # ADD dropout param
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dropout=dropout,         # ADD: this controls attn + FFN dropout
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2
        )
        self.post_dropout = nn.Dropout(dropout)      # ADD: after full decoder

    def forward(self, queries, features):
        output = self.decoder(queries, features)
        output = self.post_dropout(output)            # ADD
        return output
class EmotionClassifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)               # ADD: stabilize pooled features
        self.fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),                               # CHANGE: GELU > ReLU for transformers
            nn.Dropout(0.4),                         # INCREASE: 0.3 -> 0.4
            nn.Linear(128, util.NUM_CLASSES)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.norm(x)                            # ADD: normalize before FC
        return self.fc(x)
class ContrastiveModule(nn.Module):
    def __init__(self, input_dim, proj_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)         # ADD
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, proj_dim)
        )

    def forward(self, features):
        pooled = features.mean(dim=1)
        pooled = self.norm(pooled)                  # ADD: before projection
        proj = self.projector(pooled)
        proj = nn.functional.normalize(proj, dim=1)
        return proj                        
# ViT with pretrained ImageNet-21k weights
# ----------------------------------------------------------
# FULL VISION TRANSFORMER (ViT-Base) WITH IMAGENET-21K PRETRAINED WEIGHTS
# ----------------------------------------------------------
# Model: google/vit-base-patch16-224-in21k
# Source: https://huggingface.co/google/vit-base-patch16-224-in21k
#
# Architecture:
#   Patch size   : 16x16
#   Image size   : 224x224
#   Patches      : (224/16)^2 = 196 patch tokens + 1 CLS token = 197 tokens
#   Hidden dim   : 768
#   Depth        : 12 transformer encoder layers
#   Heads        : 12
#   MLP ratio    : 4x (FFN dim = 3072)
#   Parameters   : ~86M
#
# OUTPUT SHAPE COMPARISON vs Swin-T:
#   Swin-T  → backbone(x) = [B, 768]        (globally pooled, no token structure)
#             then unsqueeze(1) → [B, 1, 768]
#
#   ViT-B   → backbone(x) = [B, 197, 768]   (197 tokens: CLS + 196 patches)
#             RICHER token structure — feeds EmotionQueryTokens and EmotionDecoder DIRECTLY
#             No unsqueeze needed — all other modules unchanged
#
# REQUIRE:
# pip install transformers
# ----------------------------------------------------------
import torch
import torch.nn as nn
from transformers import ViTModel

#import logging
#logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ViT_BACKBONE_NAME is the HuggingFace model identifier
# Can be swapped to any compatible ViT variant, e.g.:
#   'google/vit-base-patch16-224'          (ImageNet-1K fine-tuned)
#   'google/vit-base-patch16-224-in21k'    (ImageNet-21K pretrained — recommended)
#   'google/vit-large-patch16-224-in21k'   (larger, ~307M params)
VIT_BACKBONE_NAME = 'google/vit-base-patch16-224-in21k'

# Feature dimension of ViT-Base hidden states
# hidden_size = 768 — same as Swin-T final_dim, so all downstream modules are unchanged
VIT_HIDDEN_DIM = 768

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # Emotion Region Enhancement
        # (unchanged — same as Swin-T pipeline)
        # --------------------------------------------------
        self.region = EmotionRegionEnhancement()

        # --------------------------------------------------
        # OFFICIAL ViT-BASE BACKBONE
        # Pretrained: ImageNet-21K (google/vit-base-patch16-224-in21k)
        #
        # Input  : [B, 3, 224, 224]
        # Output : last_hidden_state = [B, 197, 768]
        #          197 = 1 CLS token + 196 patch tokens (14x14 grid)
        #          768 = hidden dimension (same as Swin-T final_dim)
        #
        # WHY ViT-21K over ViT-1K:
        #   Pretrained on 14M images (21K classes) vs 1.28M (1K classes)
        #   Better feature diversity for small/specialist datasets like ASD
        #   More robust to distribution shift between ImageNet and facial images
        # --------------------------------------------------
        self.backbone = ViTModel.from_pretrained(
            VIT_BACKBONE_NAME,
            add_pooling_layer=False,   # keep all 197 tokens, skip default CLS pooler
        )

        # --------------------------------------------------
        # Feature dimension from ViT-Base
        # hidden_size = 768 — identical to Swin-T final_dim
        # All downstream modules (EmotionQueryTokens, EmotionDecoder,
        # ContrastiveModule, EmotionClassifier) are UNCHANGED
        # --------------------------------------------------
        final_dim = VIT_HIDDEN_DIM   # 768

        # --------------------------------------------------
        # YOUR ORIGINAL MODULES — ALL UNCHANGED
        # --------------------------------------------------
        self.query      = EmotionQueryTokens(dim=final_dim)
        self.decoder    = EmotionDecoder(dim=final_dim)
        self.contrast   = ContrastiveModule(input_dim=final_dim)
        self.classifier = EmotionClassifier(dim=final_dim)

    def forward(self, x):
        # --------------------------------------------------
        # Step 1: Emotion region enhancement
        # Input/output: [B, 3, 224, 224]  (unchanged)
        # --------------------------------------------------
        x = self.region(x)

        # --------------------------------------------------
        # Step 2: ViT-Base backbone forward pass
        #
        # HuggingFace ViTModel expects pixel_values argument
        # Returns BaseModelOutput with:
        #   last_hidden_state : [B, 197, 768]
        #     index 0   → CLS token  (global representation)
        #     index 1-196 → patch tokens (local spatial features)
        #
        # DIFFERENCE vs Swin-T:
        #   Swin-T produced [B, 768] → needed unsqueeze(1) → [B, 1, 768]
        #   ViT produces    [B, 197, 768] directly → richer token context
        #   EmotionQueryTokens and EmotionDecoder receive 197 tokens instead of 1
        #   This gives the decoder MORE spatial context to cross-attend over
        # --------------------------------------------------
        vit_output = self.backbone(pixel_values=x)
        features = vit_output.last_hidden_state   # [B, 197, 768]

        # --------------------------------------------------
        # Step 3: YOUR ORIGINAL PIPELINE — UNCHANGED
        # features shape [B, 197, 768] feeds directly into EmotionQueryTokens
        # (the pipeline already handles [B, N, D] — no modification needed)
        # --------------------------------------------------
        queries    = self.query(features)          # [B, 6, 768]
        decoded    = self.decoder(queries, features)  # [B, 6, 768]
        embeddings = self.contrast(decoded)        # [B, 64]
        logits     = self.classifier(decoded)      # [B, 6]
        return logits, embeddings

# ----------------------------------------------------------
# Early Stopping
# ----------------------------------------------------------
EPS = 1e-3
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.loss = None
        self.acc = None
        self.f1 = None
        self.counter = 0
        self.improvemnet = True

    def stop(self, _loss, _f1, _acc):
        if self.loss is None:
            self.loss = _loss
            self.acc = _acc
            self.f1 = _f1
            self.improvemnet = True
            return False, True

        #if (self.loss > _loss) or (self.acc < _acc) or (self.f1 < _f1):
        ##if (self.loss - _loss > EPS) or (_acc - self.acc > EPS) or (_f1 - self.f1 > EPS):
        if (self.loss - _loss > EPS):
            self.loss = _loss
            self.f1 = _f1
            self.counter = 0
            # if (self.acc < _acc):
            #     self.acc = _acc
            #     self.improvemnet = True
            # else:
            #     self.improvemnet = False    
        else:
            self.counter += 1
            self.improvemnet = False
            print(f"No Loss improvement. Patience: {self.counter}/{self.patience}")

        if (self.acc < _acc):
            self.acc = _acc
            self.improvemnet = True
        else:
            self.improvemnet = False    

        should_stop = self.counter >= self.patience
        return should_stop, self.improvemnet

# -----------------------------------------------------------
# Create a Metrics Function
# -----------------------------------------------------------
def ComputeMetrics(labels, preds):
    acc = accuracy_score(labels, preds)
    # weighted = weighted by class size
    # macro = equal weight per emotion
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds) # * 100.0
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    #print(classification_report(labels,preds))
    return acc, precision, recall, f1, macro_f1, bal_acc

# ----------------------------------------------------------
# DATA LOADER FUNCTION
# ----------------------------------------------------------
def compute_class_weights(dataset, num_classes, device):
    counts  = Counter(dataset.targets)
    total   = sum(counts.values())
    weights = [total / (num_classes * counts[i])
               for i in range(num_classes)]
    return torch.FloatTensor(weights).to(device)

       
def contrastive_loss(embeddings, labels, temperature=0.08): # was 0.07, 0.05
    embeddings = F.normalize(embeddings, dim=1)  # 
    similarity = torch.matmul(embeddings, embeddings.T)
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    ######
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(embeddings.shape[0]).view(-1, 1).to(embeddings.device),
        0
    )
    mask = mask * logits_mask  # Now only true positives, no self-positives
    #######
    logits = similarity / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # stability
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    # guard against empty positives
    valid_mask  = mask.sum(1) > 0
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos[valid_mask].mean()
    return loss
# ----------------------------------------------------------
# TRAIN FUNCTION
# ----------------------------------------------------------

def TrainEpoch(model, loader, optimizer, criterion, scaler, epoch, device=util.DEVICE):
    model.train()

    total_loss = 0.0
    preds = []
    labels = []

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # ---- Forward (Mixed Precision) ----
        #OLD # with autocast(device_type=util.device_type):
        #NEW
        with autocast( device_type=util.device_type, enabled=(util.device_type == "cuda")):
            logits, embeddings = model(x)
            loss_cls = criterion(logits, y)
            loss_con = contrastive_loss(embeddings, y)
            #loss = loss_cls + LOSS_CNTS * loss_con
            # Or anneal it over time:
            lambda_contrast = max(0.05, LOSS_CNTS * (0.99 ** epoch))  # decay each epoch
            loss = loss_cls + lambda_contrast * loss_con

        # ---- Backward (Scaled) ----
        scaler.scale(loss).backward()
        # ---- Gradient Clipping (IMPORTANT) ----
        scaler.unscale_(optimizer)  # required before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # ---- Optimizer Step ----
        scaler.step(optimizer)
        scaler.update()
        # ---- Metrics ----
        total_loss += loss.item()
        preds.append(logits.argmax(1).detach().cpu())
        labels.append(y.detach().cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = accuracy_score(labels, preds)
    return total_loss / len(loader), acc

def ValidateEpoch(model, loader, criterion, epoch, device=util.DEVICE):
    model.eval()

    total_loss = 0.0
    preds = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validate", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- Forward (Mixed Precision for speed) ----
            #OLD # with autocast(device_type=util.device_type):
            #NEW
            with autocast( device_type=util.device_type, enabled=(util.device_type == "cuda")):
                logits, embeddings = model(x)
                loss_cls = criterion(logits, y)
                loss_con = contrastive_loss(embeddings, y)
                #loss = loss_cls + LOSS_CNTS * loss_con
                # Or anneal it over time:
                lambda_contrast = max(0.05, LOSS_CNTS * (0.99 ** epoch))  # decay each epoch
                loss = loss_cls + lambda_contrast * loss_con

            total_loss += loss.item()

            # ---- Collect predictions ----
            preds.append(logits.argmax(1).detach().cpu())
            labels.append(y.detach().cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc, precision, recall, f1, macro_f1, bal_acc = ComputeMetrics(labels, preds)
    return (total_loss / len(loader), acc, precision, recall, f1, macro_f1, bal_acc, preds, labels )

# ----------------------------------------------------------
# Final model create
# ----------------------------------------------------------
def TrainForFinalModel(): #train_data
    print("\n========== FINAL TRAINING ON FULL TRAIN DATA ==========")
    train_data = util.LoadInputRawImage(util.TRAIN_DATA_PATH)
    print(train_data.class_to_idx)

    train_transform    = util.get_train_transformViT()
    val_test_transform = util.get_val_test_transform()    
    train_idx, val_idx = train_test_split(
        range(len(train_data)),
        test_size=0.1,
        stratify=train_data.targets,
        random_state=42
    )
    # build once
    aug_map = util.BuildAugmentationMap(util.AUG_DATA_PATH, train_data.classes)
    train_dataset = util.AugmentedSubset(train_data, train_idx, aug_map, transform=train_transform)
    test_dataset = util.CleanSubset( train_data, val_idx, transform=val_test_transform )
    """
    train_dataset = util.CleanSubset( train_data, train_idx, transform=train_transform )
    test_dataset = util.CleanSubset( train_data, val_idx, transform=val_test_transform )
    """
    print(f"Total samples in train_data : {len(train_dataset)}")
    print(f"Total samples in train_data : {len(test_dataset)}")

    full_train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
        #prefetch_factor=2,              # each worker prefetches 2 batches ahead
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,             # no shuffle for testing
        num_workers=0,
        pin_memory=True,

        #prefetch_factor=2,              # each worker prefetches 2 batches ahead
    )
    # Fresh model
    final_model = VisionTransformer().to(util.DEVICE)
    if util.MULTI_GPU > 1:
        final_model = torch.nn.DataParallel(final_model)

    # ── Always use unwrap_model for ANY attribute access ──────────
    raw_model = util.unwrap_model(final_model)   # gets .module if DataParallel

    # ── Freeze ViT backbone initially ─────────────────────────────
    for param in raw_model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': raw_model.backbone.parameters(),   'lr': 1e-6,  'weight_decay': 1e-4},
        {'params': raw_model.region.parameters(),     'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.query.parameters(),      'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.decoder.parameters(),    'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.contrast.parameters(),   'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.classifier.parameters(), 'lr': 5e-5,  'weight_decay': 1e-3},
    ])

    warmupepoch = 25
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmupepoch)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE-warmupepoch, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmupepoch])   

   
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    early_stopper = EarlyStopping(patience=Global_Patience)

    #OLD
    #scaler = GradScaler(util.device_type)
    #New
    use_amp = (util.device_type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # Train on full training data
    best_final_model = None
    best_final_f1 = 0
    #for Draw Graph
    final_history = InitFinalHistory()
    start = time.perf_counter()
    for epoch in range(EPOCHS_FINETUNE):
        # ── Gradual backbone unfreeze (ViT-specific) ──────────────
        unfreeze_backbone_gradually(raw_model, epoch)

        train_loss, train_acc = TrainEpoch(
            final_model,
            full_train_loader,
            optimizer,
            criterion,
            scaler,
            epoch
        )
        print(f"[FINAL TRAIN] Epoch {epoch} Loss {train_loss} Acc {train_acc}")
        test_loss, test_acc, test_precision, test_recall, test_f1, macro_f1, bal_acc, preds, labels = ValidateEpoch(
            final_model,
            test_loader,
            criterion,
            epoch
        )
        scheduler.step()

        #for Draw Graph
        UpdateFinalHistory(final_history, train_loss, test_loss, train_acc, test_acc)
        #print("Final Train Model: Test Acc",test_acc, "Precision",test_precision, "Recall",test_recall, "F1",test_f1)
        shouldstop, improvemnet = early_stopper.stop(test_loss, test_f1, test_acc)
        print("Final Train Model: Test Acc",test_acc, "Precision",test_precision, "Recall",test_recall, "F1",test_f1, "BestACC", early_stopper.acc)
        if improvemnet:
            best_final_model = copy.deepcopy(final_model)
            optimal_epochs = epoch
        if shouldstop:
            print(f"[FINAL] Early stopping triggered at epoch = {epoch}")
            break                    
    
    # Final evaluation on test set (20%)
    #print("\n========== FINAL TEST SET EVALUATION ==========")
    
    final_test_loss, final_test_acc, final_test_precision, final_test_recall, final_test_f1, final_macro_f1, final_bal_acc, final_preds, final_labels = ValidateEpoch(
        best_final_model,
        test_loader,
        criterion,
        epoch
    )
    end = time.perf_counter()
    elapsed = end - start
    with open(util.ViT_LogFile, "a") as f:
        f.write(f"\n{'='*90}\n")
        f.write("FINAL MODEL 90-10 EVALUATION REPORT (ViT-Base)\n")
        f.write(f"{'='*90}\n")

        # Header
        f.write(
            f"{'Metric':<25} {'Value':<15}\n"
        )
        f.write(f"{'-'*90}\n")

        # Metrics
        f.write(f"{'Accuracy':<25} {final_test_acc:<15.4f}\n")
        f.write(f"{'Balanced Accuracy':<25} {final_bal_acc:<15.4f}\n")
        f.write(f"{'Precision':<25} {final_test_precision:<15.4f}\n")
        f.write(f"{'Recall':<25} {final_test_recall:<15.4f}\n")
        f.write(f"{'F1-score (Weighted)':<25} {final_test_f1:<15.4f}\n")
        f.write(f"{'F1-score (Macro)':<25} {final_macro_f1:<15.4f}\n")

        f.write(f"{'-'*90}\n")
        f.write(f"{'Total Time':<25} {util.format_time(elapsed)}\n")
        f.write(f"{'='*90}\n\n")

        # Classification report
        report = classification_report(final_labels, final_preds, target_names=util.EMOTIONS, digits=4)

        f.write("CLASS-WISE CLASSIFICATION REPORT\n")
        f.write(f"{'-'*90}\n")
        f.write(report)
        f.write(f"{'='*90}\n")    
    #for draw confusion matix
    DrawFinalGraphs(util.ViT_PlotsDir, final_history)
    DrawEmotionGraphs(util.ViT_PlotsDir, final_labels,final_preds,util.EMOTIONS)
    return best_final_model
# ----------------------------------------------------------
# GRADUAL BACKBONE UNFREEZE — ViT-Base specific
# ----------------------------------------------------------
# Gradual unfreeze adapted for ViT
# ViT has no hierarchical stages like Swin's .layers[0-3]. It has a flat .encoder.layer[0-11]:
# epoch 25 → unfreeze encoder layers 8-11  (deepest, most task-specific)
# epoch 40 → unfreeze encoder layers 4-7   (middle)
# epoch 55 → unfreeze encoder layers 0-3   (shallow, general)
# epoch 70 → unfreeze patch embeddings

# ViT-Base has 12 encoder layers (0..11) in backbone.encoder.layer
# Strategy: unfreeze from deepest (layer 11) to shallowest (layer 0)
# This mirrors the Swin-T stage-by-stage unfreeze adapted for ViT's
# flat transformer structure.
#
# Swin-T had:  backbone.layers[3], [2], [1], patch_embed
# ViT-B has :  backbone.encoder.layer[11..8], [7..4], [3..0], embeddings
# For ViT: in my One system (a) it works by backbone.encoder.layer
#       But in other system (b) it works for backbone.layers
# ----------------------------------------------------------


def unfreeze_backbone_gradually(raw_model, epoch):
    backbone = raw_model.backbone
    if epoch == 25:
        # Unfreeze top 4 encoder layers (layers 8-11 — deepest, most task-specific)
        for i in range(8, 12):
            #for param in backbone.encoder.layer[i].parameters():
            for param in backbone.layers[i].parameters():
                param.requires_grad = True
        print(f"  [Epoch {epoch}] Unfroze ViT encoder layers 8-11")

    elif epoch == 40:
        # Unfreeze middle 4 encoder layers (layers 4-7)
        for i in range(4, 8):
            #for param in backbone.encoder.layer[i].parameters():
            for param in backbone.layers[i].parameters():
                param.requires_grad = True
        print(f"  [Epoch {epoch}] Unfroze ViT encoder layers 4-7")

    elif epoch == 55:
        # Unfreeze bottom 4 encoder layers (layers 0-3 — most general features)
        for i in range(0, 4):
            #for param in backbone.encoder.layer[i].parameters():
            for param in backbone.layers[i].parameters():
                param.requires_grad = True
        print(f"  [Epoch {epoch}] Unfroze ViT encoder layers 0-3")

    elif epoch == 70:
        # Unfreeze patch embeddings (position + patch projection)
        for param in backbone.embeddings.parameters():
            param.requires_grad = True
        print(f"  [Epoch {epoch}] Unfroze ViT patch embeddings")

# ----------------------------------------------------------
# FINETUNE WITH 10 FOLD
# ----------------------------------------------------------

def ExecuteFold(train_loader, val_loader, test_loader, fold):

    model = VisionTransformer().to(util.DEVICE)
    if util.MULTI_GPU > 1:
        model = torch.nn.DataParallel(model)

    # ── Always use unwrap_model for ANY attribute access ──────────
    raw_model = util.unwrap_model(model)   # gets .module if DataParallel

    # ── Freeze ViT backbone initially — only custom heads train ───
    # ViT-Base has 86M params; freezing prevents catastrophic forgetting
    # of ImageNet-21K features during early epochs
    for param in raw_model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': raw_model.backbone.parameters(),   'lr': 1e-6,  'weight_decay': 1e-4},
        {'params': raw_model.region.parameters(),     'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.query.parameters(),      'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.decoder.parameters(),    'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.contrast.parameters(),   'lr': 5e-5,  'weight_decay': 1e-3},
        {'params': raw_model.classifier.parameters(), 'lr': 5e-5,  'weight_decay': 1e-3},
    ])

    warmupepoch = 25 #15
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmupepoch)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE-warmupepoch, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmupepoch])   

    early_stopper = EarlyStopping(patience=Global_Patience)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    Fstat = util.FoldStat()

    #OLD
    #scaler = GradScaler(util.device_type)
    #New
    use_amp = (util.device_type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(EPOCHS_FINETUNE):

        # ── Gradual backbone unfreeze (ViT-specific) ──────────────
        unfreeze_backbone_gradually(raw_model, epoch)

        train_loss,train_acc = TrainEpoch(model,train_loader,optimizer,criterion,scaler, epoch)
        val_loss,val_acc,val_precision,val_recall,val_f1, macro_f1, bal_acc, pr,lab = ValidateEpoch(
            model,val_loader,criterion,epoch
        )
        test_loss,test_acc,test_precision,test_recall,test_f1, test_macro_f1, test_bal_acc, preds,labels = ValidateEpoch(
                model,test_loader,criterion,epoch
            )
        scheduler.step()            

        #For draw graph
        UpdateFoldHistory(Fstat.history, train_loss,train_acc, val_loss,val_acc,val_f1)
        Fstat.UpdateValStat(val_acc, val_precision, val_recall, val_f1, macro_f1, bal_acc)
        Fstat.UpdateTestStat(test_acc, test_precision, test_recall, test_f1, test_bal_acc, test_macro_f1)


        print("ViT-FOLD:", fold, "Epoch",epoch)
        print("Train Loss",train_loss,"Train Acc",train_acc)
        print("Val Acc",val_acc, "Precision",val_precision, "Recall",val_recall, "F1",val_f1, "Val loss", val_loss)
        print("Test Acc",test_acc, "Precision",test_precision, "Recall",test_recall, "F1",test_f1)

        print("So far BEST:::best_epoch:", Fstat.best_epoch, "val_acc_best", Fstat.val_acc_best,"val_f1_best", Fstat.val_f1_best, "macro_f1_best", Fstat.macro_f1_best, "bal_acc_best", Fstat.bal_acc_best);
        print("So far BEST:::best_epoch:", Fstat.best_epoch, "test_acc_best", Fstat.test_acc_best,"test_f1_best", Fstat.test_f1_best, "test_macro_f1_best", Fstat.test_macro_f1_best, "test_bal_acc_best", Fstat.test_bal_acc_best);

        shouldstop, improvemnet = early_stopper.stop(val_loss, val_f1, val_acc)
        if improvemnet:
            Fstat.UpdateEpoch(epoch)
        if shouldstop:
            print(f"[FOLD {fold}] Early stopping triggered at epoch = {epoch}")
            break     
        if epoch > 5:
             DrawFoldGraphs(util.ViT_PlotsDir, Fstat.history,fold)
    return Fstat


def FinetuneModel():

    train_transform    = util.get_train_transformViT()
    val_test_transform = util.get_val_test_transform()
    train_data = util.LoadInputRawImage(util.TRAIN_DATA_PATH)
    print(train_data.class_to_idx)
    # build once
    aug_map = util.BuildAugmentationMap(util.AUG_DATA_PATH, train_data.classes)

    # ── prepare arrays for sklearn ──────────────────────────────
    labels  = np.array(train_data.targets)    # [0,2,1,0,...] class per sample
    groups  = np.array(train_data.base_ids)   # group id per sample (prevents leakage)
    indices = np.arange(len(train_data))      # [0,1,2,...,N]

    # ── generate ALL folds first, store as list ─────────────────
    splitter = StratifiedGroupKFold(n_splits=KFOLD, shuffle=True, random_state=42)

    folds = []
    for train_val_ids, test_ids in splitter.split(indices, labels, groups):
        # splitter gives (rest, test) — store test fold indices only
        folds.append(test_ids.tolist())
    # print(f"Total samples in train_data : {len(train_data)}")
    # print(f"Total labels                : {len(labels)}")
    # print(f"Each fold should be ~        {len(train_data) // KFOLD}")

    # # Add this right after fold split to verify
    # total = sum(len(f) for f in folds)
    # print(f"Sum of all fold sizes       : {total}")
    # print(f"Individual fold sizes       : {[len(f) for f in folds]}")
    ViT_tStat = util.TrainingStat()
    start = time.perf_counter()
    for fold in range(KFOLD):
        print(f"\n{'='*40}")
        print(f"FOLD {fold}")
        print(f"{'='*40}")
        test_ids = folds[fold]
        val_ids  = folds[(fold + 1) % KFOLD]

        train_ids = []
        for i in range(KFOLD):
            if i != fold and i != (fold + 1) % KFOLD:
                train_ids.extend(folds[i])
        # print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

        # datasets
        train_dataset = util.AugmentedSubset(train_data, train_ids, aug_map, transform=train_transform)
        val_dataset   = util.AugmentedSubset(train_data, val_ids, aug_map, transform=val_test_transform) # orig images only, clean transform
        #val_dataset   = util.CleanSubset(train_data, val_ids, transform=val_test_transform) # orig images only, clean transform
        test_dataset = util.CleanSubset( train_data, test_ids, transform=val_test_transform )

        # loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=util.NUM_WORKER,
            pin_memory=True,

        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=util.NUM_WORKER,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=util.NUM_WORKER,
            pin_memory=True,

        )

        # return FoldStat
        ViT_fStat = ExecuteFold(train_loader, val_loader, test_loader, fold) 
        #for Draw Graph
        DrawFoldGraphs(util.ViT_PlotsDir, ViT_fStat.history,fold)
        ViT_tStat.UpdateFoldValidateStatistics(ViT_fStat) #val_acc_best, val_precision_best, val_recall_best, val_f1_best, macro_f1_best, bal_acc_best)
        ViT_tStat.UpdateFoldTestStatistics(ViT_fStat) #test_acc_best, test_precision_best, test_recall_best, test_f1_best, test_macro_f1_best, test_bal_acc_best)
        ViT_tStat.UpdateEpochs(ViT_fStat.best_epoch)
        util.WriteFoldSummary(util.ViT_LogFile, fold, ViT_fStat)

        print("FOLD:",fold, "best_epoch:", ViT_fStat.best_epoch, "val_acc_best", ViT_fStat.val_acc_best,"val_f1_best", ViT_fStat.val_f1_best, "macro_f1_best", ViT_fStat.macro_f1_best, "bal_acc_best", ViT_fStat.bal_acc_best)
        print("FOLD:",fold, "best_epoch:", ViT_fStat.best_epoch, "test_acc_best", ViT_fStat.test_acc_best,"test_f1_best", ViT_fStat.test_f1_best, "test_macro_f1_best", ViT_fStat.test_macro_f1_best, "test_bal_acc_best", ViT_fStat.test_bal_acc_best)
        

    end = time.perf_counter()
    elapsed = end - start
    util.WriteCrossValidationSummary(util.ViT_LogFile, ViT_tStat, elapsed, KFOLD)
    #for Draw Graph
    #DrawKFoldGraph(util.ViT_PlotsDir, ViT_tStat.fold_f1)
    DrawKFoldF1Graph(util.ViT_PlotsDir, ViT_tStat.fold_f1)
    DrawKFoldAccGraph(util.ViT_PlotsDir, ViT_tStat.fold_acc)

    optimal_epochs = int(np.mean(ViT_tStat.best_epochs))
    #best_final_model = TrainForFinalModel(train_data)
    #return best_final_model, optimal_epochs
    return optimal_epochs
# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def SaveViTModel(model, optimal_epochs):
    model = util.unwrap_model(model)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': util.NUM_CLASSES,
        'emotions': util.EMOTIONS,
        'epoch': optimal_epochs,
        }, util.ViT_Final_Model_Path)
    print("ViT Model saved:", util.ViT_Final_Model_Path)

def TrainVisionTransformer():

    print("START Vision Transformer (ViT-Base)")
    optimal_epochs = FinetuneModel()
    
    #optimal_epochs = 100
    """
    train_data = util.LoadInputRawImage(util.TRAIN_DATA_PATH)
    aug_data = util.LoadInputRawImage(util.AUG_DATA_PATH)
    all_data = util.MergeDatasets(train_data, aug_data)

    #final_model = TrainForFinalModel(all_data)
    """
    final_model = TrainForFinalModel()
    SaveViTModel(final_model, optimal_epochs)

def SaveModelSummary():
    filename="ViT_CV_80_10_10_Summary.txt"
    png_filename = "ViT_CV_80_10_10_Summary.png"
    model = VisionTransformer().to(util.DEVICE)
    summary_str = summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=0,
        depth=4,

        #input_size=(128, 3, 224, 224),
        #verbose=0
    )
    # replace unicode box chars with ASCII equivalents
    clean = (str(summary_str)
        .replace("├", "+")
        .replace("─", "-")
        .replace("│", "|")
        .replace("└", "+")
        .replace("┼", "+")
        .replace("┤", "+")
        .replace("╠", "+")
        .replace("═", "=")
    )
    with open(filename, "w", encoding="utf-8") as f:
        print(model, file=f)
        print("\n\nSummary**********\n", file=f)
        f.write(clean)
    print("Model summary saved:", filename)
    util.render_summary_to_png(str(summary_str), png_filename)    #options to give clean too

def PrintModelDetails():
    model = VisionTransformer().to(util.DEVICE)
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"Total parameters: {total_params:,}")
    #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Trainable parameters: {trainable_params:,}")
    
    #print(model)
    print(model.backbone)

    print("Summary**********\n")
    summary(model, input_size=(128, 3, 224, 224))