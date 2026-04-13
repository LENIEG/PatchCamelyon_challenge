# PatchCamelyon_challenge Performance Report
Medical data patches binary classification task.
Compares performance of 2 models: 
- ConvNeXT
- ViT Small


## ConvNeXT



## ViT Small

__Latest Configuration__  
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

__Augmentations in training___

_Horizontal flip_  
Probability: 0.5  
_Vertical flip_  
Probability: 0.5  
_Translation via RandomAffine_  
Degrees: 0  
Translate: (4 / image_size, 4 / image_size)  
With image_size = 96, this is about ±4 pixels in x and y  
Applied every sample (not wrapped in RandomApply)  
_Brightness and contrast jitter_  
Brightness: 0.35  
Contrast: 0.35  
Probability: 0.9  
_Saturation and hue jitter_  
Saturation: 0.30  
Hue: 0.08  
Probability: 0.9  
_Stain augmentation in HED space (custom StainJitterHED)_  
Scale jitter: 0.12  
Bias jitter: 0.04  
Probability: 0.85  
_Random 90-degree rotation (custom RandomRotate90)_  
Chooses k in {1, 2, 3} which corresponds to 90, 180, 270 degrees  
Probability: 0.5  
_Mild crop-resize_  
RandomResizedCrop size: image_size (96)  
Scale: (0.90, 1.00)  
Aspect ratio: (0.95, 1.05)  
Probability: 0.30  
_Normalization_  
Mean: [0.485, 0.456, 0.406]  
Std: [0.229, 0.224, 0.225]  
Applied every sample

___AUC___  
Mathematically:  
AUC = Probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.


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