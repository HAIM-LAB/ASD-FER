## HybridT Transformer Model.

import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Use the 1st GPU
import cv2
import math
import random
import numpy as np
#from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

#OLD
#from torch.amp import autocast, GradScaler
#from torch.cuda.amp import autocast, GradScaler
#NEW
import torch
from torch import autocast
from torch.cuda.amp import GradScaler

from collections import Counter

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import copy

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchinfo import summary

import torch.nn.functional as F
from pathlib import Path
#-------------------
# Plot related
#------------------
from DrawGraph import *
import time
#import other python local files
import Utility as util

print("DEVICE:", util.DEVICE)
print("DEVICE TYPE:", util.DEVICE.type)


IMG_SIZE = util.IMG_SIZE
PRE_TRAIN_BATCH_SIZE = 128
BATCH_SIZE = 128*util.MULTI_GPU #128*util.MULTI_GPU
BATCH_SIZE_FINAL_MODEL = BATCH_SIZE #256*util.MULTI_GPU #128*util.MULTI_GPU
EPOCHS_PRETRAIN = 200
EPOCHS_FINETUNE = 300
KFOLD = 10
Global_Patience = 25   # For early stoopping

# if Embedding_Dimension is 64 then EPOCHS_FINETUNE is 150 good
# But if Embedding_Dimension is 128 then EPOCHS_FINETUNE is 100 is enough
Embedding_Dimension = 128 #64 #128
LABEL_SMOOTHING = 0.03 #base 0.03
LOSS_CNTS = 0.4 #0.2 #base 0.2


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
# STEP 3A CNN STEM
# ----------------------------------------------------------

class CNNStem(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(3,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,Embedding_Dimension,3,stride=2,padding=1),
            nn.BatchNorm2d(Embedding_Dimension),
            nn.ReLU()

        )

    def forward(self,x):
        return self.layers(x)

# ----------------------------------------------------------
# STEP 3B SWIN TRANSFORMER ENCODER (simplified)
# ----------------------------------------------------------

class SimpleTransformerEncoder(nn.Module):

    def __init__(self, embed_dim=Embedding_Dimension, num_heads=2): #4 to 2 for speed up

        super().__init__()

        #self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.pool = nn.AdaptiveAvgPool2d((9,9)) # higher is slow but expecting better accuracy
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2 # For small datasets 2 is enough otherwise 4/6 is required.
        )

    def forward(self,x):
        
        x = self.pool(x)
        B,C,H,W = x.shape
        #x = x.view(B,C,H*W).permute(0,2,1)
        x = x.view(B,C,H*W).permute(0,2,1).contiguous()

        x = self.encoder(x)

        return x

# ----------------------------------------------------------
# STEP 4 EMOTION QUERY TOKENS
# ----------------------------------------------------------
class EmotionQueryTokens(nn.Module):
    def __init__(self, dim=Embedding_Dimension, num_queries=util.NUM_CLASSES):
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
    def __init__(self, dim=Embedding_Dimension, heads=4, dropout=0.3):   # ADD dropout param
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
    def __init__(self, dim=Embedding_Dimension):
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
    def __init__(self, input_dim=Embedding_Dimension, proj_dim=64):
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
# FULL MODEL
# ----------------------------------------------------------
class AETFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.region = EmotionRegionEnhancement()
        self.cnn = CNNStem()
        self.encoder = SimpleTransformerEncoder()
        self.query = EmotionQueryTokens()
        self.decoder = EmotionDecoder()    
        self.contrast = ContrastiveModule()
        self.classifier = EmotionClassifier()

    def forward(self,x):
        x = self.region(x)
        x = self.cnn(x)
        features = self.encoder(x)
        queries = self.query(features)  # EmotionQueryTokens
        decoded = self.decoder(queries,features)
        embeddings = self.contrast(decoded)
        logits = self.classifier(decoded)

        return logits, embeddings
# ----------------------------------------------------------
# DATA LOADER FUNCTION
# ----------------------------------------------------------
def LoadInputPreTrainImage(path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(path,transform=transform)
    return dataset
# ----------------------------------------------------------
# LOSS FUNCTION
# ----------------------------------------------------------
def compute_class_weights(dataset, num_classes, device):
    counts  = Counter(dataset.targets)
    total   = sum(counts.values())
    weights = [total / (num_classes * counts[i])
               for i in range(num_classes)]
    return torch.FloatTensor(weights).to(device)

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=None, label_smoothing=LABEL_SMOOTHING, reduction='mean'):
        super().__init__()
        self.gamma = 2.0
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        # Compute label-smoothed cross entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none',
                                  weight=self.alpha, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

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
def compute_total_loss(ce_loss, cont_loss, lambda_contrast):
    """
    Normalize contrastive to same scale as CE before combining.
    Prevents contrastive from dominating regardless of temperature.
    """
    # detach to avoid gradient through normalization
    ce_scale   = ce_loss.detach().abs() + 1e-8
    cont_scale = cont_loss.detach().abs() + 1e-8

    # normalize contrastive to CE scale
    cont_normalized = cont_loss * (ce_scale / cont_scale)

    total = ce_loss + lambda_contrast * cont_normalized
    return total

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
        print("Devicetype:", util.device_type)
        print("acbdabcd#################")
        
        #OLD # with autocast(device_type=util.device_type):
        #NEW
        with autocast( device_type=util.device_type, enabled=(util.device_type == "cuda")):
            logits, embeddings = model(x)
            loss_cls = criterion(logits, y)
            loss_con = contrastive_loss(embeddings, y)
            #loss = loss_cls + LOSS_CNTS * loss_con
            # Or anneal it over time:
            lambda_contrast = max(0.05, LOSS_CNTS * (0.99 ** epoch))  # decay each epoch
            #loss = compute_total_loss(loss_cls, loss_con, LOSS_CNTS)
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
            print("Devicetype:", util.device_type)
            print("acbdabcd#################")
            #OLD #with autocast(device_type=util.device_type):
            #NEW
            with autocast( device_type=util.device_type, enabled=(util.device_type == "cuda")):
                logits, embeddings = model(x)
                loss_cls = criterion(logits, y)
                loss_con = contrastive_loss(embeddings, y)
                #loss = loss_cls + LOSS_CNTS * loss_con
                # Or anneal it over time:
                lambda_contrast = max(0.05, LOSS_CNTS * (0.99 ** epoch))  # decay each epoch
                loss = loss_cls + lambda_contrast * loss_con
                #loss = compute_total_loss(loss_cls, loss_con, LOSS_CNTS)

            total_loss += loss.item()

            # ---- Collect predictions ----
            preds.append(logits.argmax(1).detach().cpu())
            labels.append(y.detach().cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc, precision, recall, f1, macro_f1, bal_acc = ComputeMetrics(labels, preds)
    return (total_loss / len(loader), acc, precision, recall, f1, macro_f1, bal_acc, preds, labels )

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
        #if (self.loss - _loss > EPS) or (_acc - self.acc > EPS) or (_f1 - self.f1 > EPS):
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
# ----------------------------------------------------------
# PRETRAIN STAGE
# ----------------------------------------------------------
def PretrainModel():
    dataset = LoadInputPreTrainImage(util.PRETRAIN_PATH)
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.1,
        stratify=dataset.targets,
        random_state=42,
    )
    train_loader = DataLoader(
        Subset(dataset,train_idx),
        batch_size=PRE_TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=util.NUM_WORKER,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,              # each worker prefetches 2 batches ahead
    )

    val_loader = DataLoader(
        Subset(dataset,val_idx),
        batch_size=PRE_TRAIN_BATCH_SIZE,
        num_workers=util.NUM_WORKER,
        pin_memory=True, #ForCUDA
        persistent_workers=True,
        prefetch_factor=2,              # each worker prefetches 2 batches ahead
    )

    model = AETFormer().to(util.DEVICE)
    # if MULTI_GPU > 1:
    #     model = torch.nn.DataParallel(model)

    # compile model (PyTorch 2.x optimization)
    #model = torch.compile(model) #speed up
    optimizer = optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_val_acc = 0.0  # track best accuracy
    best_model_wts = None   # store best weights in memory
    best_val_f1 = 0.0  # track best accuracy  
    '''
    Standard training and  stable convergence
    A cosine annealing learning rate scheduler is used to gradually reduce the learning rate, enabling stable convergence and fine-grained
    optimization, while early stopping prevents overfitting by terminating training when validation performance no longer improves.
    '''
    warmup_epochs = 15
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=EPOCHS_PRETRAIN - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #       optimizer,
    #        T_max=EPOCHS_PRETRAIN
    #    )
    patience = 15   # For early stoopping
    early_stopper = EarlyStopping(patience=patience)

    #OLD
    #scaler = GradScaler(util.device_type)
    #New
    use_amp = (util.device_type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(EPOCHS_PRETRAIN):
        train_loss,train_acc = TrainEpoch(model,train_loader,optimizer,criterion,scaler, epoch)
        val_loss,val_acc,val_precision,val_recall,val_f1,macro_f1, bal_acc,all_preds,all_labels = ValidateEpoch(model,val_loader,criterion, epoch)
        print("PRETRAIN EPOCH",epoch)
        print("Train Loss",train_loss,"Train Acc",train_acc)
        print("Val Loss",val_loss,"Val Acc",val_acc, "Val F1", val_f1)
        scheduler.step()
        # Keep best model in memory (NOT saving to disk yet)
        shouldstop, improvemnet = early_stopper.stop(val_loss, val_f1, val_acc)
        if improvemnet:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            raw_model = util.unwrap_model(model) ##
            best_model_wts = copy.deepcopy(raw_model.state_dict())
            print("***********Best model updated (in memory):", best_val_acc)
        if shouldstop:        
            print(f"Early stopping triggered  at epoch = {epoch}")
            break           
    torch.save(best_model_wts, util.Pretrain_model_path)
    return best_model_wts


# ----------------------------------------------------------
# Final model create
# ----------------------------------------------------------

def TrainForFinalModel(pretrained_model): #,train_data
    print("\n========== FINAL TRAINING ON FULL TRAIN DATA ==========")
    train_data = util.LoadInputRawImage(util.TRAIN_DATA_PATH)
    print(train_data.class_to_idx)    

    train_transform    = util.get_train_transformHybrid()
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
    print(f"Total samples in train_dataset : {len(train_dataset)}")
    print(f"Total samples in test_dataset : {len(test_dataset)}")

    full_train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_FINAL_MODEL,
        shuffle=True,
        num_workers=util.NUM_WORKER,
        persistent_workers=True,
        prefetch_factor=2,              # each worker prefetches 2 batches ahead
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_FINAL_MODEL,
        shuffle=False,             # no shuffle for testing
        num_workers=util.NUM_WORKER,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,              # each worker prefetches 2 batches ahead
    )

    # Fresh model
    final_model = AETFormer().to(util.DEVICE)
    if util.MULTI_GPU > 1:
        final_model = torch.nn.DataParallel(final_model)
        state_dict = pretrained_model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict["module." + k] = v
        final_model.load_state_dict(new_state_dict)
    else:
        final_model.load_state_dict(pretrained_model.state_dict())

    #optimizer = optim.AdamW(final_model.parameters(), lr=5e-5, weight_decay=1e-3)    
    optimizer = optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=5e-3)
    warmupepoch = 25    
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmupepoch)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE-warmupepoch, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmupepoch])   

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    #class_weights = compute_class_weights( train_dataset, util.NUM_CLASSES, util.DEVICE)
    #criterion = FocalLossWithLabelSmoothing( alpha=class_weights)    

    early_stopper = EarlyStopping(patience=Global_Patience)

    #OLD
    #scaler = GradScaler(util.device_type)
    #New
    use_amp = (util.device_type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    
    # Train on full training data
    best_final_model = None
    #best_final_f1 = 0
    #for Draw Graph
    final_history = InitFinalHistory()
    start = time.perf_counter()
    ttaBest = 0.0
    for epoch in range(EPOCHS_FINETUNE):
        train_loss, train_acc = TrainEpoch( final_model, full_train_loader, optimizer, criterion, scaler, epoch )
        print(f"[FINAL TRAIN] Epoch {epoch} Loss {train_loss} Acc {train_acc}")
        test_loss, test_acc, test_precision, test_recall, test_f1, macro_f1, bal_acc, preds, labels = ValidateEpoch(
            final_model,
            test_loader,
            criterion,
            epoch
        )
        #for Draw Graph
        UpdateFinalHistory(final_history, train_loss, test_loss, train_acc, test_acc)

        #print("Final Train Model: Test Acc",test_acc, "Precision",test_precision, "Recall",test_recall, "F1",test_f1)
              
        scheduler.step()
        shouldstop, improvemnet = early_stopper.stop(test_loss, test_f1, test_acc)
        print("Final Train Model: Test Acc",test_acc, "Precision",test_precision, "Recall",test_recall, "F1",test_f1, "BestACC", early_stopper.acc)

        #if(best_final_f1 < test_f1):
        if improvemnet:
            #best_final_f1 = test_f1
            best_final_model = copy.deepcopy(final_model)
            optimal_epochs = epoch
        if shouldstop:
            print(f"[FINAL] Early stopping triggered at epoch = {epoch}")
            break                    
    
    # Final evaluation on test set (20%)
   
    final_test_loss, final_test_acc, final_test_precision, final_test_recall, final_test_f1, final_macro_f1, final_bal_acc, final_preds, final_labels = ValidateEpoch(
        best_final_model,
        test_loader,
        criterion,
        epoch
    )

    end = time.perf_counter()
    elapsed = end - start

    with open(util.Hybrid_LogFile, "a") as f:
        f.write(f"\n{'='*90}\n")
        f.write("FINAL MODEL 90-10 EVALUATION REPORT\n")
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
    DrawFinalGraphs(util.Hybrid_PlotsDir, final_history)
    DrawEmotionGraphs(util.Hybrid_PlotsDir, final_labels,final_preds,util.EMOTIONS)
    return best_final_model

# ----------------------------------------------------------
# FINETUNE WITH 10 FOLD
# ----------------------------------------------------------
def ExecuteFold(pretrained_model, train_loader, val_loader, test_loader, fold):
    model = AETFormer().to(util.DEVICE)
    if util.MULTI_GPU > 1:
        model = torch.nn.DataParallel(model)
        state_dict = pretrained_model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict["module." + k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(pretrained_model.state_dict())

    #optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
    #warmupepoch = 15
    warmupepoch = 25    
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmupepoch)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE-warmupepoch, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmupepoch])   

    early_stopper = EarlyStopping(patience=Global_Patience)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    #class_weights = compute_class_weights( train_dataset, util.NUM_CLASSES, util.DEVICE)
    #criterion = FocalLossWithLabelSmoothing( alpha=class_weights)    

    Fstat = util.FoldStat()
    #OLD
    #scaler = GradScaler(util.device_type)
    #New
    use_amp = (util.device_type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(EPOCHS_FINETUNE):

        train_loss,train_acc = TrainEpoch(model,train_loader,optimizer,criterion,scaler, epoch)
        val_loss,val_acc,val_precision,val_recall,val_f1, macro_f1, bal_acc, pr,lab = ValidateEpoch(
            model,val_loader,criterion,epoch
        )
        test_loss,test_acc,test_precision,test_recall,test_f1, test_macro_f1, test_bal_acc, preds,labels = ValidateEpoch(
                model,test_loader,criterion,epoch
            )
        #For draw graph
        UpdateFoldHistory(Fstat.history, train_loss,train_acc, val_loss,val_acc,val_f1)
        Fstat.UpdateValStat(val_acc, val_precision, val_recall, val_f1, macro_f1, bal_acc)
        Fstat.UpdateTestStat(test_acc, test_precision, test_recall, test_f1, test_bal_acc, test_macro_f1)


        print("Hybrid-FOLD:", fold, "Epoch",epoch)
        print("Train Loss",train_loss,"Train Acc",train_acc)
        print("Val Acc",val_acc, "Precision",val_precision, "Recall",val_recall, "F1",val_f1, "Val loss", val_loss)
        print("Test Acc",test_acc, "Precision",test_precision, "Recall",test_recall, "F1",test_f1)

        print("So far BEST:::best_epoch:", Fstat.best_epoch, "val_acc_best", Fstat.val_acc_best,"val_f1_best", Fstat.val_f1_best, "macro_f1_best", Fstat.macro_f1_best, "bal_acc_best", Fstat.bal_acc_best);
        print("So far BEST:::best_epoch:", Fstat.best_epoch, "test_acc_best", Fstat.test_acc_best,"test_f1_best", Fstat.test_f1_best, "test_macro_f1_best", Fstat.test_macro_f1_best, "test_bal_acc_best", Fstat.test_bal_acc_best);

        scheduler.step()
        shouldstop, improvemnet = early_stopper.stop(val_loss, val_f1, val_acc)
        if improvemnet:
            Fstat.UpdateEpoch(epoch)
        if shouldstop:
            print(f"[FOLD {fold}] Early stopping triggered at epoch = {epoch}")
            break    
        if epoch > 5:
            DrawFoldGraphs(util.Hybrid_PlotsDir, Fstat.history,fold) 
    return Fstat


def FinetuneModel(pretrained_model):

    train_transform    = util.get_train_transformHybrid()
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
    Hyb_tStat = util.TrainingStat()
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
        print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

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
            persistent_workers=True,
            prefetch_factor=2,              # each worker prefetches 2 batches ahead
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=util.NUM_WORKER,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,              # each worker prefetches 2 batches ahead
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=util.NUM_WORKER,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,              # each worker prefetches 2 batches ahead
        )
        # return FoldStat
        Hyb_fStat = ExecuteFold(pretrained_model, train_loader, val_loader, test_loader, fold) 
        #for Draw Graph
        DrawFoldGraphs(util.Hybrid_PlotsDir, Hyb_fStat.history,fold)
        Hyb_tStat.UpdateFoldValidateStatistics(Hyb_fStat) #.val_acc_best, Hyb_fStat.val_precision_best, val_recall_best, val_f1_best, macro_f1_best, bal_acc_best)
        Hyb_tStat.UpdateFoldTestStatistics(Hyb_fStat) #.test_acc_best, test_precision_best, test_recall_best, test_f1_best, test_macro_f1_best, test_bal_acc_best)
        Hyb_tStat.UpdateEpochs(Hyb_fStat.best_epoch)
        util.WriteFoldSummary(util.Hybrid_LogFile, fold, Hyb_fStat)

        print("FOLD:",fold, "best_epoch:", Hyb_fStat.best_epoch, "val_acc_best", Hyb_fStat.val_acc_best,"val_f1_best", Hyb_fStat.val_f1_best, "macro_f1_best", Hyb_fStat.macro_f1_best, "bal_acc_best", Hyb_fStat.bal_acc_best)
        print("FOLD:",fold, "best_epoch:", Hyb_fStat.best_epoch, "test_acc_best", Hyb_fStat.test_acc_best,"test_f1_best", Hyb_fStat.test_f1_best, "test_macro_f1_best", Hyb_fStat.test_macro_f1_best, "test_bal_acc_best", Hyb_fStat.test_bal_acc_best)
        

    end = time.perf_counter()
    elapsed = end - start
    util.WriteCrossValidationSummary(util.Hybrid_LogFile, Hyb_tStat, elapsed, KFOLD)
    #for Draw Graph
    DrawKFoldF1Graph(util.Hybrid_PlotsDir, Hyb_tStat.fold_f1)
    DrawKFoldAccGraph(util.Hybrid_PlotsDir, Hyb_tStat.fold_acc)
    optimal_epochs = int(np.mean(Hyb_tStat.best_epochs))
    #best_final_model = TrainForFinalModel(pretrained_model, train_data)
    #return best_final_model, optimal_epochs
    return optimal_epochs
# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def LoadPretrainedModel(path):
    model = AETFormer().to(util.DEVICE)
    model.load_state_dict(torch.load(path, map_location=util.DEVICE))
    model.eval()   # optional (safe default)
    return model

def PretrainHybrid():

    print("START Hybrid PRETRAIN")
    pretrained_model = PretrainModel()
    print("Load hybrid PRETRAIN")
    pretrained_model = LoadPretrainedModel(util.Pretrain_model_path)
    print(next(pretrained_model.parameters()).mean())
    return pretrained_model

def SaveHybridModel(model, optimal_epochs):
    model = util.unwrap_model(model)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': util.NUM_CLASSES,
        'emotions': util.EMOTIONS,
        'epoch': optimal_epochs,
        }, util.Hybrid_Final_Model_Path)
    print("Hybrid Model saved:", util.Hybrid_Final_Model_Path)

def TrainAETFormer(pretrained_model):

    #print(sum(p.numel() for p in pretrained_model.parameters()))
    #freeze the backbone during first epochs: Train only decoder + classifier first.
    #This often improves FER fine-tuning stability.

    print("START Hybrid FINETUNE")

    optimal_epochs = FinetuneModel(pretrained_model)
    #optimal_epochs = 100
    """
    train_data = util.LoadInputRawImage(util.TRAIN_DATA_PATH)
    aug_data = util.LoadInputRawImage(util.AUG_DATA_PATH)
    all_data = util.MergeDatasets(train_data, aug_data)
    final_model = TrainForFinalModel(pretrained_model, all_data)
    """
    final_model = TrainForFinalModel(pretrained_model)
    SaveHybridModel(final_model, optimal_epochs)

def SaveModelSummaryUtf8():
    filename="AETFormer2_Summary.txt"
    model = AETFormer().to(util.DEVICE)
    with open(filename, "w") as f:
        # print model architecture
        print(model, file=f)
        print("\n\nSummary**********\n", file=f)
        # redirect stdout temporarily
        old_stdout = sys.stdout
        sys.stdout = f
        summary(model, input_size=(128, 3, 224, 224))
        # restore stdout
        sys.stdout = old_stdout

    print("Model summary saved:", filename)



def SaveModelSummary():
    txt_filename = "Hybrid_CV_80_10_10_Summary.txt"
    png_filename = "Hybrid_CV_80_10_10_Summary.png"

    model = AETFormer().to(util.DEVICE)
    summary_str = summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=0,
        depth=4,
    )
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

    with open(txt_filename, "w", encoding="utf-8") as f:
        #print(model, file=f)
        #print("\n\nSummary**********\n", file=f)
        f.write(clean)
    print("Model summary saved:", txt_filename)

    util.render_summary_to_png(str(summary_str), png_filename)    #options to give clean too
def SaveModelSummary_Mine():
    filename = "Hybrid_CV_80_10_10_Summary.txt"
    model = AETFormer().to(util.DEVICE)
    summary_str = summary(
        model,
        #input_size=(128, 3, 224, 224),
        #verbose=0
        input_size=(1, 3, 224, 224),   # matches your training input_size=(128,3,224,224), batch=1 for the diagram
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=0,
        depth=4,
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
        #print(model, file=f)
        #print("\n\nSummary**********\n", file=f)
        f.write(clean)
    print("Model summary saved:", filename)

def PrintModelDetails():
    model = AETFormer().to(util.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(model)

    print("Summary**********\n")
    summary(model, input_size=(128, 3, 224, 224))

'''
After 10-fold cross-validation, the best practice is to train the final model on the full dataset. 
The purpose of cross-validation is to evaluate how well a model generalizes, not to produce the final 
model itself. Once you have identified the best hyperparameters or the best algorithm through 10-fold CV, 
you retrain that model on all available data to utilize its maximum predictive power.

1. Split: Separate your data into a final test set (e.g., 10%) and a training/validation set (90%).
2. Cross-Validation: Perform 10-fold cross-validation only on the 90% training/validation set to tune hyperparameters and select the best model.
3. Final Training: Train the chosen model with the best hyperparameter settings on the entire 90% training/validation set.
4. Final Evaluation: Test this model on the 10% holdout test set to get an unbiased performance estimate. 
'''