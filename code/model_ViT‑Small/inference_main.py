"""
ViT-Small inference on PatchCamelyon test split.
Supports command-line and VS Code debug execution.
Saves per-sample predictions for model-to-model comparison.

Example command-line usage:
python code/model_ViT‑Small/inference_main.py --checkpoint-dir ./checkpoints/vit_small_dino --output-dir ./results/vit_small_dino
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Add parent directory to path for importing downloader
sys.path.insert(0, str(Path(__file__).parent.parent))
from downloader import HDF5Dataset, preprocessing_fn


class ViTSmallBinary(nn.Module):
    """ViT-Small DINO backbone with binary head."""

    def __init__(self, dropout=0.1, drop_path=0.1, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_small_patch16_224.dino',
            pretrained=pretrained,
            img_size=96,
            num_classes=0,
            drop_path_rate=drop_path,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(384, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def build_test_loader(data_dir, batch_size, num_workers):
    """Create test DataLoader from H5 files."""
    data_dir = Path(data_dir)
    test_x_path = data_dir / 'camelyonpatch_level_2_split_test_x.h5' / 'camelyonpatch_level_2_split_test_x.h5'
    test_y_path = data_dir / 'camelyonpatch_level_2_split_test_y.h5' / 'camelyonpatch_level_2_split_test_y.h5'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_dataset = HDF5Dataset(
        x_path=str(test_x_path),
        y_path=str(test_y_path),
        transform=test_transform,
        preprocessing_function=preprocessing_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test samples: {len(test_dataset)}, batches: {len(test_loader)}")
    return test_loader


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)


def resolve_checkpoint_path(config):
    """Resolve checkpoint path from explicit path or default best model."""
    if config['checkpoint_path']:
        candidate = Path(config['checkpoint_path'])
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    checkpoint_dir = Path(config['checkpoint_dir'])
    best_path = checkpoint_dir / 'best_model.pt'
    if best_path.exists():
        return best_path

    candidates = []
    for path in checkpoint_dir.glob('checkpoint_epoch_*.pt'):
        try:
            epoch_num = int(path.stem.split('_')[-1])
            candidates.append((epoch_num, path))
        except ValueError:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. Expected best_model.pt or checkpoint_epoch_*.pt"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def run_inference(config):
    """Run test-set inference and save outputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = resolve_checkpoint_path(config)
    print(f"Using checkpoint: {checkpoint_path}")

    model = ViTSmallBinary(
        dropout=config['head_dropout'],
        drop_path=config['drop_path_rate'],
        pretrained=False,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    test_loader = build_test_loader(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    images_root = output_dir / 'classified_images'
    outcome_dirs = {
        'TP': images_root / 'TP',
        'FP': images_root / 'FP',
        'TN': images_root / 'TN',
        'FN': images_root / 'FN',
    }
    for path in outcome_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    total = 0
    correct = 0
    rows = []
    global_idx = 0

    start_time = time.perf_counter()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Inference', leave=False)
        for images, labels in pbar:
            images = images.to(device).float()
            labels = labels.to(device).float().view(-1, 1)

            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_class_1 = probs
            probs_class_0 = 1.0 - probs
            # For binary-logit models, [-logit, +logit] is a consistent two-logit view.
            logits_class_1 = logits
            logits_class_0 = -logits
            preds = (probs >= 0.5).float()

            batch_correct = (preds == labels).sum().item()
            total += labels.size(0)
            correct += batch_correct
            batch_acc = batch_correct / labels.size(0)
            running_acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({'batch_acc': f'{batch_acc:.4f}', 'running_acc': f'{running_acc:.4f}'})

            for i in range(labels.size(0)):
                y_true = int(labels[i].item())
                y_pred = int(preds[i].item())
                prob_0 = float(probs_class_0[i].item())
                prob_1 = float(probs_class_1[i].item())
                confidence = max(prob_0, prob_1)

                if y_true == 1 and y_pred == 1:
                    outcome = 'TP'
                elif y_true == 0 and y_pred == 1:
                    outcome = 'FP'
                elif y_true == 0 and y_pred == 0:
                    outcome = 'TN'
                else:
                    outcome = 'FN'

                rows.append([
                    global_idx,
                    y_true,
                    y_pred,
                    float(logits_class_0[i].item()),
                    float(logits_class_1[i].item()),
                    prob_0,
                    prob_1,
                ])

                # Convert normalized tensor back to displayable RGB in [0, 1].
                img = images[i].detach().cpu()
                img = img * std + mean
                img = img.clamp(0.0, 1.0)

                image_name = (
                    f'sample_{global_idx:06d}_'
                    f'true_{y_true}_pred_{y_pred}_'
                    f'conf_{confidence:.4f}_p1_{prob_1:.4f}.png'
                )
                save_image(img, outcome_dirs[outcome] / image_name)

                global_idx += 1

    elapsed = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0

    pred_path = output_dir / config['predictions_file']
    with pred_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_index',
            'y_true',
            'y_pred',
            'logit_class_0',
            'logit_class_1',
            'prob_class_0',
            'prob_class_1',
        ])
        writer.writerows(rows)

    summary = {
        'model_name': 'vit_small_patch16_224.dino',
        'predictions_schema': 'sample_index,y_true,y_pred,logit_class_0,logit_class_1,prob_class_0,prob_class_1',
        'checkpoint_path': str(checkpoint_path),
        'num_samples': total,
        'accuracy': accuracy,
        'elapsed_seconds': elapsed,
        'data_dir': config['data_dir'],
        'batch_size': config['batch_size'],
        'classified_images_dir': str(images_root),
    }

    summary_path = output_dir / config['summary_file']
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 60)
    print('Inference complete')
    print('=' * 60)
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Elapsed (sec): {elapsed:.2f}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved classified images: {images_root}")
    print('=' * 60)


def get_default_config():
    """Get default inference configuration."""
    return {
        'data_dir': './data',
        'checkpoint_dir': './checkpoints/vit_small_dino',
        'checkpoint_path': '',
        'output_dir': './results/vit_small_dino',
        'predictions_file': 'test_predictions.csv',
        'summary_file': 'test_summary.json',
        'batch_size': 64,
        'num_workers': 4,
        'head_dropout': 0.1,
        'drop_path_rate': 0.1,
    }


def main():
    """Main inference entry point."""

    # Used when script runs in debugger with no CLI args.
    DEBUG_CONFIG = {
        'data_dir': './data',
        'checkpoint_dir': './checkpoints/vit_small_dino',
        'checkpoint_path': '',
        'output_dir': './results/vit_small_dino',
        'predictions_file': 'test_predictions.csv',
        'summary_file': 'test_summary.json',
        'batch_size': 64,
        'num_workers': 0,
        'head_dropout': 0.1,
        'drop_path_rate': 0.1,
    }

    parser = argparse.ArgumentParser(description='Run ViT-Small inference on PatchCamelyon test split')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/vit_small_dino')
    parser.add_argument('--checkpoint-path', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='./results/vit_small_dino')
    parser.add_argument('--predictions-file', type=str, default='test_predictions.csv')
    parser.add_argument('--summary-file', type=str, default='test_summary.json')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--head-dropout', type=float, default=0.1)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    args = parser.parse_args()

    if len(sys.argv) == 1:
        config = DEBUG_CONFIG
        print('>> Running in DEBUG mode (using DEBUG_CONFIG) <<')
    else:
        config = get_default_config()
        config['data_dir'] = args.data_dir
        config['checkpoint_dir'] = args.checkpoint_dir
        config['checkpoint_path'] = args.checkpoint_path
        config['output_dir'] = args.output_dir
        config['predictions_file'] = args.predictions_file
        config['summary_file'] = args.summary_file
        config['batch_size'] = args.batch_size
        config['num_workers'] = args.num_workers
        config['head_dropout'] = args.head_dropout
        config['drop_path_rate'] = args.drop_path_rate

    print('\n' + '=' * 60)
    print('Inference Configuration:')
    print('=' * 60)
    for key, value in config.items():
        print(f'  {key}: {value}')
    print('=' * 60 + '\n')

    run_inference(config)


if __name__ == '__main__':
    main()
