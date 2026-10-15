# ASD-FER: An Annotated Multi-Class Benchmark for Facial Emotion Recognition in Children with Autism Spectrum Disorder

## Overview

**ASD-FER** is a facial emotion recognition benchmark developed to support research on emotion classification in facial images of children with Autism Spectrum Disorder (ASD).

The benchmark contains **2,779 facial images** organized into six emotion categories. It extends a publicly available facial image dataset originally developed for ASD-related image classification by introducing multi-class emotion annotations.

The benchmark also supports comparative evaluation of deep learning and Transformer-based architectures for facial emotion recognition.

## Emotion Categories

The benchmark includes six emotion classes:

- **Angry**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

## Dataset Summary

| Property | Description |
|---|---|
| Dataset name | ASD-FER |
| Task | Multi-class facial emotion recognition |
| Total images | 2,779 |
| Number of emotion classes | 6 |
| Image modality | Static facial images |
| Primary research focus | Emotion classification in images associated with children with ASD |

## Class Distribution

The dataset has an imbalanced class distribution.

| Emotion | Number of Images |
|---|---:|
| Happy | 1,717 |
| Neutral | 758 |
| Angry | 112 |
| Fear | 82 |
| Sad | 51 |
| Surprise | 59 |
| **Total** | **2,779** |

This class imbalance should be considered when training and evaluating models. We recommend reporting class-sensitive metrics, including macro-averaged F1-score and balanced accuracy, alongside overall accuracy.

## Research Framework

The associated study introduces **EQFormer**, an emotion recognition framework designed to combine spatial feature enhancement with emotion-query-based Transformer decoding.

The framework includes:

- **Emotion Region Enhancement (ERE):** Enhances facial feature representations before the backbone.
- **Backbone feature extraction:** Extracts visual representations using CNN–Transformer hybrid and Transformer-based architectures.
- **Emotion query tokens:** Uses six emotion-specific query tokens.
- **Transformer decoder:** Processes emotion queries against the extracted feature representations.
- **Classification head:** Produces predictions for the six emotion categories.
- **Supervised contrastive learning:** Provides an auxiliary training objective to improve emotion representation learning.

The framework is evaluated with HybridT, Swin Transformer (Swin-T), and Vision Transformer (ViT-Base) backbone configurations.

For the complete architecture, experimental settings, evaluation protocol, and results, please refer to the associated research paper.

## Intended Use

ASD-FER is intended for research and educational purposes, including:

- Multi-class facial emotion recognition.
- Comparative evaluation of CNN and Transformer-based models.
- Investigation of emotion representation learning.
- Development and evaluation of methods for imbalanced facial image classification.
- Research into visual emotion analysis involving images associated with children with ASD.

**Important:** This benchmark is a research resource. Emotion predictions should not be interpreted as definitive measurements of a child's internal emotional state, diagnostic evidence, or clinical assessments.

## Dataset Source and Attribution

The source facial images were obtained from the publicly available Kaggle dataset:

**Facial_data_ASD** — uploaded by `shafi420`.

- Dataset: https://www.kaggle.com/datasets/shafi420/facial
- Reference: shafi420, *Facial_data_asd: Dataset for ASD and non-ASD facial image classification*, Kaggle (2022).

ASD-FER introduces six-class emotion annotations for research purposes. Please cite the original dataset and the associated ASD-FER research paper when using this benchmark.

**Image reuse and redistribution:** Please review the original dataset's license and terms before downloading, redistributing, or republishing the facial images. Public availability does not by itself establish unrestricted reuse rights. If this repository distributes only annotations, scripts, or metadata, users may need to obtain the images directly from the original source, subject to its terms.

## Reproducibility

For reproducible experiments, researchers should document:

- Training, validation, and test split procedures.
- Random seeds and model initialization.
- Image preprocessing and augmentation settings.
- Class weighting or sampling strategies used to address class imbalance.
- Evaluation metrics and checkpoint selection criteria.
- Whether augmented versions of the same original image are kept within the same data split.

Avoid placing augmented versions of the same original image across training and evaluation splits, as this can lead to data leakage and overly optimistic performance estimates.

## Citation

If you use ASD-FER, please cite the associated research paper. Replace the placeholder below with the final publication details when available.

```bibtex
@article{asdfer_benchmark,
  title   = {ASD-FER: An Annotated Multi-Class Benchmark and Comparative Study for Facial Emotion Recognition in Children with Autism Spectrum Disorder},
  author  = {Chowdhury, Shahana Yasmin and others},
  journal = {Machine Learning with Applications},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please also cite the original source dataset:

```bibtex
@misc{shafi420_facial_asd_2022,
  author       = {shafi420},
  title        = {Facial_data_asd: Dataset for ASD and non-ASD facial image classification},
  year         = {2022},
  howpublished = {Kaggle},
  url          = {https://www.kaggle.com/datasets/shafi420/facial}
}
```

## Contact

For questions about the benchmark, annotations, experimental protocol, or associated research, please contact the corresponding author through the contact information provided in the research paper.

## Disclaimer

ASD-FER is intended solely for research and educational purposes. It is not a diagnostic tool and should not be used to make clinical, educational, or other high-stakes decisions about individual children.
