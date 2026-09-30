'''
Gradient-weighted Class Activation Mapping
Gradient-weighted Class Activation Mapping (Grad-CAM) is a visualization method used in
explainable artificial intelligence to identify which regions of an input image most influence
a convolutional neural network's (CNN's) decision. It produces a color-coded heatmap overlay that
highlights class-discriminative areas supporting a particular prediction.

Grad-CAM visualization shows that red and yellow regions represent
highly influential areas for emotion prediction, while green areas
indicate moderate importance and blue regions contribute minimally
to the final classification decision.

* Explainable AI (XAI) means showing why the model made a prediction.
* Grad-CAM highlights which image regions influence the prediction.
* Grad-CAM is an XAI library (python -m pip install grad-cam)
* "Grad-CAM visualization shows that the model focuses on eyes and mouth regions,
  which aligns with psychological emotion indicators."

------------------------------------------------------------
MODEL_TYPE options:
  0 => Swin-T         (SwinTransformer)   — uses backbone.layers[-1].blocks[-1].norm1
  1 => AETFormer      (Hybrid CNN-Transformer) — uses cnn.layers[3]
  2 => Vision Transformer (VisionTransformer ViT-Base) — uses backbone.vit.encoder.layer[-1].layernorm_before
------------------------------------------------------------

NOTE on GradCAM for ViT (MODEL_TYPE=2):
  Pure transformers have NO convolutional feature maps, so standard GradCAM
  cannot be applied directly.  The pytorch-grad-cam library provides
  GradCAM support for ViT via a reshape_transform that converts the
  [B, 197, 768] token sequence back to a [B, 768, 14, 14] spatial map,
  treating the 196 patch tokens as a 14x14 feature grid.
  The target layer is the LayerNorm BEFORE the last attention block
  (backbone.vit.encoder.layer[-1].layernorm_before), which carries the
  richest spatial gradient signal in ViT-Base.

  HuggingFace ViTModel internal path:
    backbone          ? ViTModel (your saved attribute name)
    backbone.vit      ? ViTModel's internal ViT trunk
    backbone.vit.encoder.layer   ? list of 12 ViTLayer blocks (0..11)
    backbone.vit.encoder.layer[11].layernorm_before  ? GradCAM target
    backbone.vit.encoder.layer[11].attention         ? alternative target
'''

import os
import torch
import cv2
import numpy as np
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ------------------------------------------------------------
# MODEL_TYPE SELECTION
# 0 => Swin-T Transformer
# 1 => AETFormer (Hybrid CNN-Transformer)
# 2 => Vision Transformer (ViT-Base)
# ------------------------------------------------------------
MODEL_TYPE = 2

# if MODEL_TYPE == 1:
#     from AETFormer import *
# elif MODEL_TYPE == 2:
#     from VisionTransformer import *
# else:
#     from SwinTransformer import *

from AETFormer import *
from VisionTransformer import *
from SwinTransformer import *

from pathlib import Path


# ------------------------------------------------------------
# Model label string — used in legend overlay
# ------------------------------------------------------------
MODEL_LABEL = {
    0: "'SwinT-MODEL'",
    1: "'HybridT-MODEL'",
    2: "'VIT-MODEL'",
}

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
def GetDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
def LoadModel(model_path, model_class, device):
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    emotions    = checkpoint['emotions']
    model       = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, emotions


# ------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------
def GetTransform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


def LoadImage(image_path, transform, device):
    img     = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor  = transform(img_rgb).unsqueeze(0).to(device)
    return img, img_rgb, tensor


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
def PredictEmotion(model, tensor, emotions):
    with torch.no_grad():
        logits, _ = model(tensor)
        probs      = torch.softmax(logits, dim=1)
        pred       = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    emotion = emotions[pred]
    return pred, emotion, confidence


# ============================================================
# GradCAM Layer Selection
# ============================================================
# Each model type requires a different target layer because
# GradCAM needs the LAST spatially-meaningful feature map.
#
# Swin-T:   backbone.layers[-1].blocks[-1].norm1
#           ? last window-attention block LayerNorm
#           ? output shape [B, H, W, C] (spatial Windows)
#
# AETFormer: cnn.layers[3]
#           ? last BatchNorm layer in CNNStem
#           ? output shape [B, C, H, W] (standard conv feature map)
#
# ViT-Base: backbone.vit.encoder.layer[-1].layernorm_before
#           ? LayerNorm BEFORE final attention block
#           ? output shape [B, 197, 768] (tokens)
#           ? reshaped to [B, 768, 14, 14] by vit_reshape_transform
# ============================================================
def GetGradCamLayer(model):
    if MODEL_TYPE == 1:
        # AETFormer: last BN in CNNStem (index 3 = BatchNorm2d after 2nd conv)
        return model.cnn.layers[3]

    elif MODEL_TYPE == 2:
        # ViT-Base: LayerNorm before the last (12th) attention block
        #
        # Confirmed from ViT_CV_Summary.txt model printout:
        #   (backbone): ViTModel
        #     (layers): ModuleList
        #       (0-11): 12 x ViTLayer
        #         (layernorm_before): LayerNorm  ? target
        #
        # Path: model.backbone        ? ViTModel
        #       .layers               ? ModuleList of 12 ViTLayer blocks
        #       [-1]                  ? last ViTLayer (index 11)
        #       .layernorm_before     ? LayerNorm before self-attention
        #
        # NOTE: This HuggingFace version exposes .layers directly on ViTModel
        #       (NOT .vit.encoder.layer — that path is for older HF versions)
        return model.backbone.layers[-1].layernorm_before

    else:
        # Swin-T: LayerNorm in last attention block of last stage
        return model.backbone.layers[-1].blocks[-1].norm1


# ============================================================
# reshape_transform functions
# Each handles the non-standard output shape of the target layer.
# ============================================================

def swin_reshape_transform(tensor):
    '''
    Swin-T target layer outputs [B, H, W, C] (channels last).
    GradCAM expects [B, C, H, W] — permute only.
    '''
    # [B, H, W, C] ? [B, C, H, W]
    return tensor.permute(0, 3, 1, 2)


def vit_reshape_transform(tensor):
    '''
    ViT target layer (layernorm_before) outputs [B, 197, 768].
    197 = 1 CLS token + 196 patch tokens (14x14 grid).

    Steps:
    1. Remove CLS token ? [B, 196, 768]
    2. Reshape to [B, 14, 14, 768]
    3. Permute to [B, 768, 14, 14]  ? standard [B, C, H, W] for GradCAM

    14 = sqrt(196) = 224 / patch_size(16)
    '''
    # Step 1: drop CLS token at index 0
    patch_tokens = tensor[:, 1:, :]          # [B, 196, 768]

    B, N, C = patch_tokens.shape
    h = w = int(N ** 0.5)                    # h = w = 14

    # Step 2 + 3: reshape and permute to spatial map
    result = patch_tokens.reshape(B, h, w, C)   # [B, 14, 14, 768]
    result = result.permute(0, 3, 1, 2)          # [B, 768, 14, 14]
    return result


# ============================================================
# GradCAM Wrapper Classes
# ============================================================

class SwinGradCAMWrapper(torch.nn.Module):
    '''
    Swin-T and AETFormer return (logits, embeddings).
    GradCAM expects a model that returns only logits (a single tensor).
    This wrapper strips the second output.
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


class ViTGradCAMWrapper(torch.nn.Module):
    '''
    ViT version of the wrapper.
    Identical logic — strips contrastive embeddings, returns logits only.

    Additional note: GradCAM hooks into backbone.vit.encoder.layer[-1].layernorm_before
    which is INSIDE this wrapper's self.model, so gradients still flow correctly
    through the unwrapped backbone.
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


# ============================================================
# GradCAM Execution
# ============================================================
def RunGradCAM(wrapped_model, tensor, target_layer, pred_class):
    '''
    Runs GradCAM with the correct reshape_transform for each model type.

    MODEL_TYPE 0 (Swin-T):   reshape swin [B,H,W,C] ? [B,C,H,W]
    MODEL_TYPE 1 (AETFormer): no reshape needed (standard conv [B,C,H,W])
    MODEL_TYPE 2 (ViT-Base):  reshape vit  [B,197,C] ? [B,C,14,14]
    '''
    if MODEL_TYPE == 2:
        cam = GradCAM(
            model=wrapped_model,
            target_layers=[target_layer],
            reshape_transform=vit_reshape_transform   # ? ViT-specific
        )
    elif MODEL_TYPE == 1:
        cam = GradCAM(
            model=wrapped_model,
            target_layers=[target_layer]
            # no reshape needed — AETFormer CNNStem is standard [B,C,H,W]
        )
    else:
        cam = GradCAM(
            model=wrapped_model,
            target_layers=[target_layer],
            reshape_transform=swin_reshape_transform  # ? Swin-specific
        )

    targets         = [ClassifierOutputTarget(pred_class)]
    grayscale_cam   = cam(input_tensor=tensor, targets=targets)
    return grayscale_cam[0]


# ============================================================
# Region Extraction and Highlighting
# ============================================================
def ExtractEmotionRegions(cam_map, threshold=0.2):
    cam_uint8 = np.uint8(255 * cam_map)
    _, mask   = cv2.threshold(cam_uint8, int(255 * threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        if area > 224 * 224 * 0.2:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes


def HighlightEmotionRegions(overlay_img, boxes):
    img = overlay_img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return img


def AddBottomLabel(img, label):
    h, w, _      = img.shape
    label_height = 40
    panel        = np.ones((label_height, w, 3), dtype=np.uint8) * 255
    text_size    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x       = (w - text_size[0]) // 2
    cv2.putText(panel, label, (text_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return np.vstack((img, panel))


# ============================================================
# Visualization
# ============================================================
def CreateVisualization(img_rgb, cam_map, emotion, confidence):

    h, w, _    = img_rgb.shape
    img_float  = img_rgb.astype(np.float32) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = show_cam_on_image(img_float, cam_map, use_rgb=True)

    boxes           = ExtractEmotionRegions(cam_map)
    overlay_regions = HighlightEmotionRegions(overlay, boxes)

    # Prediction panel (right side)
    explanation = np.ones((h, 350, 3), dtype=np.uint8) * 255
    cv2.putText(explanation, "Prediction:", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(explanation, emotion, (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(explanation, f"Confidence: {confidence:.2f}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def ArrowBlock(height):
        block = np.ones((height, 120, 3), dtype=np.uint8) * 255
        cv2.putText(block, "=>", (25, int(height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        return block

    original_panel = AddBottomLabel(img_rgb,        "Original")
    heatmap_panel  = AddBottomLabel(heatmap,         "Heatmap")
    overlay_panel  = AddBottomLabel(overlay,         "Overlay")
    regions_panel  = AddBottomLabel(overlay_regions, "Emotion Regions")
    score_panel    = AddBottomLabel(explanation,     "Emotion + Score")

    panel_height = original_panel.shape[0]
    arrow        = ArrowBlock(panel_height)

    combined = np.hstack((
        original_panel, arrow,
        heatmap_panel,  arrow,
        overlay_panel,  arrow,
        regions_panel,  arrow,
        score_panel
    ))

    # ?? Legend panel ??????????????????????????????????????????
    legend = np.ones((90, combined.shape[1], 3), dtype=np.uint8) * 255

    cv2.putText(legend,
        "GradCAM Heatmap: Red/Yellow=High | Green=Medium | Blue=Low",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Model label badge (colour-coded per model)
    label_colors = {0: (255, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)}
    cv2.putText(legend,
        MODEL_LABEL[MODEL_TYPE],
        (820, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_colors[MODEL_TYPE], 2)

    cv2.putText(legend,
        "Yellow boxes highlight facial regions most responsible for emotion prediction",
        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(legend,
        "Regions are extracted automatically from GradCAM activation",
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    final_img = np.vstack((legend, combined))
    return final_img


# ------------------------------------------------------------
# Save Result
# ------------------------------------------------------------
def SaveOutput(image, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("Saved:", path)


# ============================================================
# XAI Pipeline
# ============================================================
def RunXAI(model_path, model_class, image_path):

    device = GetDevice()

    # ?? Load model ????????????????????????????????????????????
    model, emotions = LoadModel(model_path, model_class, device)

    # ?? Transform + image load ????????????????????????????????
    transform            = GetTransform()
    img_bgr, img_rgb, tensor = LoadImage(image_path, transform, device)

    # ?? Predict ???????????????????????????????????????????????
    pred_class, emotion, confidence = PredictEmotion(model, tensor, emotions)
    print(f"  Prediction : {emotion}")
    print(f"  Confidence : {confidence:.4f}")

    # ?? GradCAM target layer ??????????????????????????????????
    target_layer = GetGradCamLayer(model)

    # ?? Wrap model (strip contrastive output for GradCAM) ?????
    if MODEL_TYPE == 2:
        wrapped_model = ViTGradCAMWrapper(model)
    else:
        wrapped_model = SwinGradCAMWrapper(model)

    # ?? Run GradCAM ???????????????????????????????????????????
    cam_map = RunGradCAM(wrapped_model, tensor, target_layer, pred_class)

    # ?? Visualise and save ????????????????????????????????????
    visualization = CreateVisualization(img_rgb, cam_map, emotion, confidence)
    fileName      = os.path.basename(str(image_path))
    SaveOutput(visualization, "xai_output", fileName)


# ============================================================
# Image Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

IMG_PATH1 =  BASE_DIR / "Project" / "CV_80_10_10" / "Kaggle_HandAnnotatedData_majorityVote_cropped\angry\15.jpg"
IMG_PATH2 =  BASE_DIR / "Project" / "CV_80_10_10" / "Kaggle_HandAnnotatedData_majorityVote_cropped\fear\0202.jpg"
IMG_PATH3 =  BASE_DIR / "Project" / "CV_80_10_10" / "Kaggle_HandAnnotatedData_majorityVote_cropped\happy\01.jpg"
IMG_PATH4 =  BASE_DIR / "Project" / "CV_80_10_10" / "Kaggle_HandAnnotatedData_majorityVote_cropped\neutral\0016.jpg"
IMG_PATH5 =  BASE_DIR / "Project" / "CV_80_10_10" / "Kaggle_HandAnnotatedData_majorityVote_cropped\sad\28.jpg"
IMG_PATH6 =  BASE_DIR / "Project" / "CV_80_10_10" / "Kaggle_HandAnnotatedData_majorityVote_cropped\surprise\0041.jpg"

ALL_IMAGES = [IMG_PATH1, IMG_PATH2, IMG_PATH3, IMG_PATH4, IMG_PATH5, IMG_PATH6]


# ============================================================
# Main
# ============================================================
def main():

    # ?? Model path per type ???????????????????????????????????
    model_paths = {
        0: "Final_Swin_Model2.pth",
        1: "Final_Hybrid_Model2.pth",
        2: "vit_final_model.pt",        # ? ViT model file
    }

    # ?? Model class per type ??????????????????????????????????
    model_classes = {
        0: SwinTransformer,
        1: AETFormer,
        2: VisionTransformer,           # ? ViT class from VisionTransformer.py
    }

    MODEL_PATH  = model_paths[MODEL_TYPE]
    model_class = model_classes[MODEL_TYPE]

    print(f"\nRunning XAI with MODEL_TYPE={MODEL_TYPE} ? {MODEL_LABEL[MODEL_TYPE]}")
    print(f"Model path : {MODEL_PATH}\n")

    for img_path in ALL_IMAGES:
        print(f"Image: {os.path.basename(img_path)}")
        RunXAI(MODEL_PATH, model_class, img_path)
        print()


if __name__ == "__main__":
    main()
