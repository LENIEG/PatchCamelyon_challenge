# PatchCamelyon_challenge Performance Report
Medical data patches binary classification task.
This project builds and compares two deep-learning pipelines for PatchCamelyon histopathology classification: 

- ConvNeXt-Tiny (ImageNet-pretrained)
- ViT-Small (DINO-pretrained).

It includes training, inference, checkpointing, metric logging (including AUC), and side-by-side performance comparison.
The goal is robust binary detection of tumor vs non-tumor tissue patches with reproducible evaluation outputs.

[The challenge](https://github.com/basveeling/pcam/blob/master/README.md)


## Dataset

The project uses PatchCamelyon: 96×96 RGB histopathology patch images with binary labels (tumor=1, non-tumor=0).
Data is stored in HDF5 files for train/validation/test plus metadata CSVs (coordinates and slide IDs).
Data handling includes RGB preprocessing, geometric/color augmentations during training.  
[Source](https://github.com/basveeling/pcam/blob/master/README.md)

__Dataset Splits and Leakage Control__  
Train/validation/test splits are leakage-safe: patches come from different original WSIs (slides), so no slide overlaps across splits.
Sample counts are:

Train: 262,144  
Validation: 32,768  
Test: 32,768  

## ConvNeXt-Tiny

Binary classifier based on torchvision ConvNeXt-Tiny pretrained on ImageNet.
Training uses Adam optimizer, BCE-based objective, batch-wise loss logging, checkpointing (latest + best), and auto-resume.
Augmentations include patch-safe geometric transforms (translation, flips), with normalized RGB input.

#### __Latest Configuration__  
```  
'data_dir':        './data',
'batch_size':      32,
'num_workers':     0,          # keep 0 for debugging to avoid multiprocessing issues
'freeze_backbone': False,      # True = freeze backbone, only train head
'num_epochs':      16,          # 2 - low number so a debug run finishes quickly
'backbone_learning_rate': 1e-4,
'head_learning_rate': 1e-5,
'weight_decay':    1e-5,
'lr_factor':       0.5,
'lr_patience':     3,
'auto_resume':     False,      # disable auto-resume for debug runs to start fresh
'resume_checkpoint_path': '',
'checkpoint_dir':  './checkpoints/convnext_tiny',

```

#### Model Architecture (Main Blocks)

Input stem: initial patch embedding/downsampling from RGB image.  
ConvNeXt stages: hierarchical convolutional blocks with depthwise conv + pointwise MLP-style  transforms.  
Feature aggregation: global average pooling.  
Classification head: final linear layer for binary output.  


## ViT-Small (DINO)

Binary classifier using timm ViT-Small (vit_small_patch16_224.dino) with a dropout + linear head.
Training uses AdamW with separate backbone/head learning rates, warmup + cosine schedule, AMP, AUC-based early stopping/checkpointing, and auto-resume.
Augmentations are stronger and pathology-oriented: flips, translation, brightness/contrast jitter, saturation/hue jitter, HED stain jitter, random 90° rotation, and mild random crop.

#### __Latest Configuration__  
```  
'batch_size': 32,  
'num_workers': 0,  
'num_epochs': 45,  
'warmup_epochs': 5,  
'backbone_lr': 1e-5,  
'head_lr': 5e-5,  
'weight_decay': 0.05,  
'head_dropout': 0.1,  
'drop_path_rate': 0.1,  
'freeze_backbone': False,  
'use_amp': True,  
'loss_type': 'bce',  
'focal_gamma': 2.0,  
'use_pos_weight': False,  
'auto_pos_weight': True,  
'pos_weight': None,  
'use_safe_augmentations': True,  
'auc_min_delta': 0.003,  
'auc_patience': 5,  
'auto_resume': True,  
```

#### Model Architecture (Main Blocks)

Patch embedding: splits 96×96 RGB image into 16×16 patches and projects to token vectors.  
Token sequence + positional encoding: patch tokens (plus CLS token) form transformer input.  
Transformer encoder stack: repeated self-attention + feed-forward blocks (12 layers, 6 heads).  
Backbone output: CLS/global representation from DINO-pretrained encoder.  
Binary head: dropout + linear layer producing one logit (tumor probability via sigmoid).  

#### Unsupervised Pretraining of the Backbone

The ViT backbone is initialized from DINO self-supervised pretraining (vit_small_patch16_224.dino).
In self-supervised pretraining, the model learns visual representations from unlabeled images by matching different views of the same image, without class labels.
This gives strong generic features before task-specific fine-tuning, which often improves convergence and performance when labeled medical data is limited.


__Augmentations in training__

1. _Horizontal flip_  
Probability: 0.5  
2. _Vertical flip_  
Probability: 0.5  
3. _Translation via RandomAffine_  
Degrees: 0  
Translate: (4 / image_size, 4 / image_size)  
With image_size = 96, this is about ±4 pixels in x and y  
Applied every sample (not wrapped in RandomApply)  
4. _Brightness and contrast jitter_  
Brightness: 0.35  
Contrast: 0.35  
Probability: 0.9  
5. _Saturation and hue jitter_  
Saturation: 0.30  
Hue: 0.08  
Probability: 0.9  
6. _Stain augmentation in HED space (custom StainJitterHED)_  
Scale jitter: 0.12  
Bias jitter: 0.04  
Probability: 0.85  
7. _Random 90-degree rotation (custom RandomRotate90)_  
Chooses k in {1, 2, 3} which corresponds to 90, 180, 270 degrees  
Probability: 0.5  
8. _Mild crop-resize_  
RandomResizedCrop size: image_size (96)  
Scale: (0.90, 1.00)  
Aspect ratio: (0.95, 1.05)  
Probability: 0.30  
9. _Normalization_  
Mean: [0.485, 0.456, 0.406]  
Std: [0.229, 0.224, 0.225]  
Applied every sample

___AUC___  
AUC means “Area Under the ROC Curve.”
Mathematically:  
AUC = Probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

Intuitively:

* It measures how well the model ranks positives above negatives.
* Higher AUC means better separation between classes across all thresholds.
* AUC = 0.5 is random guessing, AUC = 1.0 is perfect ranking.  

How it is calculated:

1. Take model scores (probabilities/logits), not hard labels.
2. Sweep all possible thresholds.
3. For each threshold, compute:
* TPR = TP / (TP + FN)
* FPR = FP / (FP + TN)
4. Plot ROC curve: TPR vs FPR.
5. AUC is the area under that curve (numerically integrated, e.g. trapezoids).  
Equivalent interpretation:

* AUC is the probability that a random positive sample gets a higher score than a random negative sample.


__Importance of AUC criteria in medical patches:__

__AUC‑based training stopping criteria__ 

AUC‑based is the right choice, not loss‑based, ecpecially in medical imaging—and especially in patch‑level pathology.

Loss continues to decrease when the model memorizes easy or noisy samples
AUC degrades once ranking quality deteriorates.
Loss reacts to label noise, AUC reacts to generalization failure.

The stopping rule must reflect ranking stability, not optimization progress.
the principle is to stop training when the model stops improving its ability to rank positives above negatives on unseen data.  
_What we Do NOT use for stopping:_
* Training AUC
* Validation loss
* Accuracy
* F1 (threshold‑dependent)

_Stopping algorithm_
1. Calculate val AUC 
2. if ther is preveous value, calculate the difference
3. if new AUC is higer then preveous by at least 0.003, save as the best checkpoint, continue to the next epoch, reset “no‑improvement” counter
4. Otherwise increment “no‑improvement” counter
5. When the “no‑improvement” counter reaches 5, stop the training.

___What is ranking stability___  
The model consistently orders positive samples above negative samples across training epochs, random seeds, and data perturbations.
If patch A is more likely to be positive than patch B, a stable model will rank A above B reliably, not only in one lucky run.  
So:  
AUC does not care about the predicted probability values themselves
It cares only about the relative ordering
If ordering is consistent → ranking is stable
If ordering changes a lot → ranking is unstable

___How is that different from non-medical use cases:___

___Why “head‑only training” is usually worse (even if epoch 1 looks good)___  
Head-only training:
fixes color/texture filters learned on non‑medical data
prevents stain adaptation
reduces robustness to domain shift

What often happens:
validation AUC looks okay initially, but collapses on new slides / data batches

Fine‑tuning with:
* very low backbone LR
* strong augmentation
* early stopping

is a safer compromise.

___Optimal values:___

___Meaning of non-optimal values:___


### Challenges

1. Overfitting  
validation loss incresed every epoch
2. AUC validation decresed every epoch by mor then 0.003
3. AUC validation changed a lot each time seed was changed.


## Results and comparison

Both models generate aligned per-sample prediction files with the same schema, enabling direct one-to-one comparison.
Evaluation includes accuracy, precision, recall, specificity, F1, balanced accuracy, and AUC, plus agreement/disagreement analysis between models.
A dedicated comparison script summarizes head-to-head performance and exports disagreement cases for error analysis.

Model A: ConvNeXt-Tiny  
  Accuracy: 0.862000 | F1: 0.846232  
Model B: ViT-Small-DINO  
  Accuracy: 0.888519 | F1: 0.881876  
__Winner__ (accuracy): ViT-Small-DINO  
Agreement rate: 0.890839  
Disagreement count: 3577  
Saved JSON: results\comparisons\convnext_vs_vit_comparison.json  
Saved disagreement CSV: results\comparisons\convnext_vs_vit_disagreements.csv  

## How to Run

```
# 1) Create and activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Train ConvNeXt
python code/model_ConvNeXt/training_main.py

# 3) Train ViT-Small
python code/model_ViT‑Small/training_main.py

# 4) Run inference (ConvNeXt)
python code/model_ConvNeXt/inference_main.py

# 5) Run inference (ViT-Small)
python code/model_ViT‑Small/inference_main.py

# 6) Compare model outputs
python code/compare_inference_performance.py
```