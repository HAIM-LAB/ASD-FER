import os
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

import sys
print(sys.executable)
print("abcd\n")

import cv2
print(cv2.__version__)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import copy

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch.nn.functional as F
from pathlib import Path
#-------------------
# Plot related
#------------------
from DrawGraph import *
import time
from torchinfo import summary

#import other python local files
import Utility as util
import AETFormer as hybrid
import SwinTransformer as Swin
import VisionTransformer as ViT


if __name__ == "__main__":
    print("Device is now: ",util.DEVICE)
    util.set_seed(42)
    
    # model = AETFormer().to(DEVICE)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable parameters: {trainable_params:,}")
    
    # print(model)

    # print("Summary**********\n")
    # summary(model, input_size=(128, 3, 224, 224))

    #HybridT model
    if 1:          
        pretrain_hybrid_model = hybrid.PretrainHybrid()
        hybrid.TrainAETFormer(pretrain_hybrid_model)
        hybrid.SaveModelSummary()

    #SwinT model
    if 1:
        Swin.TrainSwinTransFormer()
        Swin.SaveModelSummary()

    #ViT.PrintModelDetails()
    if 1:
        ViT.TrainVisionTransformer()
        ViT.SaveModelSummary()




    print("Completed")
