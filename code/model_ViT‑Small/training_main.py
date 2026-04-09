"""
ViT-Small training on PatchCamelyon (96x96) using DINO-pretrained backbone.

Architecture (requested):
- ViT-Small
- Patch size: 16x16
- Input size: 96x96 RGB
- Tokens: 6x6=36 + CLS
- Embedding dim: 384
- Layers: 12
- Heads: 6
- MLP ratio: 4.0
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path for importing downloader
sys.path.insert(0, str(Path(__file__).parent.parent))
from downloader import HDF5Dataset, preprocessing_fn


class RandomRotate90:
	"""Rotate tensor/image by a random 90-degree multiple."""

	def __call__(self, img):
		k = int(torch.randint(low=1, high=4, size=(1,)).item())
		if isinstance(img, torch.Tensor):
			return torch.rot90(img, k, dims=(1, 2))
		# Fallback for PIL inputs.
		return transforms.functional.rotate(img, angle=90 * k)


def get_pathology_safe_train_transform(image_size=96, enabled=True):
	"""
	Concrete pathology-safe augmentation preset.

	Includes existing downloader-style augmentations plus conservative additions:
	- H/V flips
	- small translation
	- mild color jitter (stain/brightness/contrast proxy)
	- 90-degree random rotation
	- mild random crop that retains most tissue
	"""
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	if not enabled:
		return transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])

	return transforms.Compose([
		transforms.ToTensor(),
		# Existing downloader-style geometric augmentations.
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomVerticalFlip(p=0.5),
		transforms.RandomAffine(degrees=0, translate=(4 / image_size, 4 / image_size)),
		# Additional pathology-safe augmentations.
		transforms.RandomApply([
			transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)
		], p=0.35),
		transforms.RandomApply([
			RandomRotate90()
		], p=0.5),
		transforms.RandomApply([
			transforms.RandomResizedCrop(
				size=image_size,
				scale=(0.90, 1.00),
				ratio=(0.95, 1.05),
				antialias=True,
			)
		], p=0.30),
		transforms.Normalize(mean=mean, std=std),
	])


class FocalLoss(nn.Module):
	"""Binary focal loss with logits."""

	def __init__(self, gamma=2.0, pos_weight=None):
		super().__init__()
		self.gamma = gamma
		self.pos_weight = pos_weight

	def forward(self, logits, targets):
		bce = nn.functional.binary_cross_entropy_with_logits(
			logits,
			targets,
			reduction='none',
			pos_weight=self.pos_weight,
		)
		probs = torch.sigmoid(logits)
		pt = probs * targets + (1 - probs) * (1 - targets)
		focal_factor = (1 - pt).pow(self.gamma)
		loss = focal_factor * bce
		return loss.mean()


class ViTSmallBinary(nn.Module):
	"""ViT-Small DINO-pretrained backbone with dropout head for binary classification."""

	def __init__(self, dropout=0.1, drop_path=0.1, pretrained=True):
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
		features = self.backbone(x)
		logits = self.head(features)
		return logits


class ViTSmallTrainer:
	"""Trainer for ViT-Small binary classification on PatchCamelyon."""

	def __init__(self, config):
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print(f"Using device: {self.device}")

		self.model = ViTSmallBinary(
			dropout=config['head_dropout'],
			drop_path=config['drop_path_rate'],
			pretrained=True,
		).to(self.device)

		# Optional backbone freezing.
		if config['freeze_backbone']:
			for p in self.model.backbone.parameters():
				p.requires_grad = False
			print('Backbone is frozen; only head will train.')

		backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
		head_params = [p for p in self.model.head.parameters() if p.requires_grad]
		param_groups = []
		if backbone_params:
			param_groups.append({'params': backbone_params, 'lr': config['backbone_lr'], 'name': 'backbone'})
		if head_params:
			param_groups.append({'params': head_params, 'lr': config['head_lr'], 'name': 'head'})

		self.optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])

		self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)

		self.scaler = GradScaler(enabled=config['use_amp'])

		pos_weight_tensor = None
		if config['use_pos_weight']:
			pos_weight_value = self._compute_pos_weight() if config['auto_pos_weight'] else config['pos_weight']
			if pos_weight_value is not None and pos_weight_value > 0:
				pos_weight_tensor = torch.tensor([pos_weight_value], device=self.device, dtype=torch.float32)
				print(f"Using pos_weight={pos_weight_value:.6f}")

		if config['loss_type'] == 'focal':
			self.criterion = FocalLoss(gamma=config['focal_gamma'], pos_weight=pos_weight_tensor)
			print(f"Loss: FocalLoss(gamma={config['focal_gamma']})")
		else:
			self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
			print('Loss: BCEWithLogitsLoss')

		self.checkpoint_dir = Path(config['checkpoint_dir'])
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.best_val_acc = 0.0
		self.best_val_loss = float('inf')
		self.batch_loss_history = []
		self.batch_loss_csv_path = self.checkpoint_dir / 'batch_loss_history.csv'
		self.metrics_csv_path = self.checkpoint_dir / 'metrics.csv'
		self.epoch_end_global_batch = {}
		self.val_loss_history = []
		self.val_loss_batch_positions = []

		self.start_epoch = 0
		self._auto_resume_if_available()
		self._initialize_batch_loss_csv()
		self._initialize_validation_history()

		total_params = sum(p.numel() for p in self.model.parameters())
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		print(f"Total params: {total_params:,}")
		print(f"Trainable params: {trainable_params:,}")
		print(f"Backbone LR: {config['backbone_lr']}")
		print(f"Head LR: {config['head_lr']}")

	def _initialize_batch_loss_csv(self):
		"""Initialize or continue per-batch loss CSV for plotting."""
		if self.start_epoch > 0 and self.batch_loss_csv_path.exists():
			kept_rows = []
			total_rows = 0
			with self.batch_loss_csv_path.open('r', newline='') as csv_file:
				reader = csv.DictReader(csv_file)
				for row in reader:
					total_rows += 1
					epoch_num = int(row['epoch'])
					if epoch_num <= self.start_epoch:
						kept_rows.append(row)
						self.batch_loss_history.append(float(row['loss']))
						self.epoch_end_global_batch[epoch_num] = int(row['global_batch'])

			if len(kept_rows) != total_rows:
				with self.batch_loss_csv_path.open('w', newline='') as csv_file:
					writer = csv.writer(csv_file)
					writer.writerow(['epoch', 'batch', 'global_batch', 'loss'])
					for row in kept_rows:
						writer.writerow([row['epoch'], row['batch'], row['global_batch'], row['loss']])
				print(f"Trimmed batch loss CSV to epoch <= {self.start_epoch}")
			print(f"Loaded existing batch loss history: {len(self.batch_loss_history)} rows")
			return

		with self.batch_loss_csv_path.open('w', newline='') as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(['epoch', 'batch', 'global_batch', 'loss'])
		print(f"Initialized batch loss CSV: {self.batch_loss_csv_path}")

	def _initialize_validation_history(self):
		"""Load/truncate validation-loss history to align with resumed epoch."""
		if not self.metrics_csv_path.exists() or self.start_epoch <= 0:
			return

		kept_rows = []
		total_rows = 0
		with self.metrics_csv_path.open('r', newline='') as csv_file:
			reader = csv.DictReader(csv_file)
			for row in reader:
				total_rows += 1
				epoch_num = int(row['epoch'])
				if epoch_num <= self.start_epoch:
					kept_rows.append(row)
					self.val_loss_history.append(float(row['val_loss']))
					if epoch_num in self.epoch_end_global_batch:
						self.val_loss_batch_positions.append(self.epoch_end_global_batch[epoch_num])

		if len(kept_rows) != total_rows:
			with self.metrics_csv_path.open('w', newline='') as csv_file:
				writer = csv.writer(csv_file)
				writer.writerow([
					'epoch',
					'train_loss',
					'train_acc',
					'val_loss',
					'val_acc',
					'backbone_lr',
					'head_lr',
					'epoch_time_sec',
				])
				for row in kept_rows:
					writer.writerow([
						row['epoch'],
						row['train_loss'],
						row['train_acc'],
						row['val_loss'],
						row['val_acc'],
						row['backbone_lr'],
						row['head_lr'],
						row['epoch_time_sec'],
					])
			print(f"Trimmed metrics CSV to epoch <= {self.start_epoch}")

		print(f"Loaded existing validation-loss history: {len(self.val_loss_history)} epochs")

	def _save_batch_losses(self, epoch, batch_losses):
		"""Append one epoch of batch losses to history CSV."""
		with self.batch_loss_csv_path.open('a', newline='') as csv_file:
			writer = csv.writer(csv_file)
			for batch_idx, loss_value in enumerate(batch_losses):
				global_batch = len(self.batch_loss_history)
				self.batch_loss_history.append(loss_value)
				writer.writerow([epoch + 1, batch_idx + 1, global_batch + 1, loss_value])

		if batch_losses:
			self.epoch_end_global_batch[epoch + 1] = len(self.batch_loss_history)

	def plot_training_losses(self):
		"""Save and display train-loss (raw/smoothed) and validation-loss curves."""
		if not self.batch_loss_history:
			print('No batch losses were recorded; skipping plot.')
			return

		smooth_window = 100
		x_values = np.arange(1, len(self.batch_loss_history) + 1)

		fig, ax1 = plt.subplots(figsize=(12, 6))
		ax1.plot(x_values, self.batch_loss_history, linewidth=1.0, alpha=0.5, label='Train Loss (batch raw)', color='tab:blue')

		if len(self.batch_loss_history) >= smooth_window:
			kernel = np.ones(smooth_window) / smooth_window
			smoothed_losses = np.convolve(self.batch_loss_history, kernel, mode='valid')
			smoothed_x = np.arange(smooth_window, len(self.batch_loss_history) + 1)
			ax1.plot(smoothed_x, smoothed_losses, linewidth=2.0, color='tab:red', label='Train Loss (100-batch average)')
		else:
			print(f"Not enough batches for {smooth_window}-batch smoothing; showing raw loss only.")

		ax2 = ax1.twinx()
		if self.val_loss_history and self.val_loss_batch_positions:
			ax2.plot(
				self.val_loss_batch_positions,
				self.val_loss_history,
				marker='o',
				markersize=3,
				linewidth=1.5,
				color='tab:green',
				label='Validation Loss (epoch)',
			)
		else:
			print('No validation-loss history found for plotting.')

		ax1.set_title('Training and Validation Loss')
		ax1.set_xlabel('Global Batch')
		ax1.set_ylabel('Train Loss')
		ax2.set_ylabel('Validation Loss')
		ax1.grid(True, alpha=0.3)

		handles1, labels1 = ax1.get_legend_handles_labels()
		handles2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
		fig.tight_layout()

		plot_path = self.checkpoint_dir / 'training_batch_loss.png'
		plt.savefig(plot_path, dpi=150)
		print(f"Saved batch loss plot: {plot_path}")
		plt.show()

	def _lr_lambda(self, epoch_idx):
		warmup_epochs = self.config['warmup_epochs']
		total_epochs = self.config['num_epochs']
		if epoch_idx < warmup_epochs:
			return float(epoch_idx + 1) / float(max(1, warmup_epochs))

		progress = (epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
		return 0.5 * (1.0 + math.cos(math.pi * progress))

	def _compute_pos_weight(self):
		train_y_path = Path(self.config['data_dir']) / 'camelyonpatch_level_2_split_train_y.h5' / 'camelyonpatch_level_2_split_train_y.h5'
		with h5py.File(train_y_path, 'r') as f:
			y = f['y'][:]
		y = y.reshape(-1)
		pos = float((y == 1).sum())
		neg = float((y == 0).sum())
		if pos == 0:
			return None
		return neg / pos

	def _auto_resume_if_available(self):
		if not self.config['auto_resume']:
			return

		explicit = self.config['resume_checkpoint']
		ckpt_path = None
		if explicit:
			candidate = Path(explicit)
			if candidate.exists():
				ckpt_path = candidate

		if ckpt_path is None:
			best_ckpt = self.checkpoint_dir / 'best_model.pt'
			if best_ckpt.exists():
				ckpt_path = best_ckpt

		if ckpt_path is None:
			epoch_ckpts = []
			for path in self.checkpoint_dir.glob('checkpoint_epoch_*.pt'):
				try:
					epoch_id = int(path.stem.split('_')[-1])
					epoch_ckpts.append((epoch_id, path))
				except ValueError:
					continue
			if epoch_ckpts:
				epoch_ckpts.sort(key=lambda t: t[0])
				ckpt_path = epoch_ckpts[-1][1]

		if ckpt_path is None:
			print('No checkpoint found. Starting from DINO pretrained backbone.')
			return

		checkpoint = torch.load(ckpt_path, map_location=self.device)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		if 'optimizer_state_dict' in checkpoint:
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if 'scheduler_state_dict' in checkpoint:
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.start_epoch = int(checkpoint.get('epoch', -1)) + 1
		self.best_val_acc = float(checkpoint.get('best_val_acc', self.best_val_acc))
		self.best_val_loss = float(checkpoint.get('best_val_loss', self.best_val_loss))
		print(f'Resumed from checkpoint: {ckpt_path}')
		print(f'Starting from epoch index: {self.start_epoch}')

	def _is_val_loss_worsening_last_epochs(self, window=4):
		"""Return True when validation loss strictly worsened over the last N epochs."""
		if len(self.val_loss_history) < window:
			return False
		last_vals = self.val_loss_history[-window:]
		return all(last_vals[i] > last_vals[i - 1] for i in range(1, window))

	def get_dataloaders(self):
		data_dir = Path(self.config['data_dir'])

		train_x = data_dir / 'camelyonpatch_level_2_split_train_x.h5' / 'camelyonpatch_level_2_split_train_x.h5'
		train_y = data_dir / 'camelyonpatch_level_2_split_train_y.h5' / 'camelyonpatch_level_2_split_train_y.h5'
		valid_x = data_dir / 'camelyonpatch_level_2_split_valid_x.h5' / 'camelyonpatch_level_2_split_valid_x.h5'
		valid_y = data_dir / 'camelyonpatch_level_2_split_valid_y.h5' / 'camelyonpatch_level_2_split_valid_y.h5'

		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]

		train_transform = get_pathology_safe_train_transform(
			image_size=96,
			enabled=self.config['use_safe_augmentations'],
		)
		valid_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])

		if self.config['use_safe_augmentations']:
			print('Using pathology-safe augmentation preset (enabled by default).')
		else:
			print('Pathology-safe augmentation preset disabled.')

		train_dataset = HDF5Dataset(
			x_path=str(train_x),
			y_path=str(train_y),
			transform=train_transform,
			preprocessing_function=preprocessing_fn,
		)
		valid_dataset = HDF5Dataset(
			x_path=str(valid_x),
			y_path=str(valid_y),
			transform=valid_transform,
			preprocessing_function=preprocessing_fn,
		)

		train_loader = DataLoader(
			train_dataset,
			batch_size=self.config['batch_size'],
			shuffle=True,
			num_workers=self.config['num_workers'],
			pin_memory=torch.cuda.is_available(),
		)
		valid_loader = DataLoader(
			valid_dataset,
			batch_size=self.config['batch_size'],
			shuffle=False,
			num_workers=self.config['num_workers'],
			pin_memory=torch.cuda.is_available(),
		)

		print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
		print(f"Valid samples: {len(valid_dataset)}, batches: {len(valid_loader)}")
		return train_loader, valid_loader

	def _step_batch(self, images, labels, train_mode=True):
		labels = labels.to(self.device).float().view(-1, 1)
		images = images.to(self.device).float()

		with autocast(device_type='cuda', enabled=self.config['use_amp'] and self.device.type == 'cuda'):
			logits = self.model(images)
			loss = self.criterion(logits, labels)

		if train_mode:
			self.optimizer.zero_grad(set_to_none=True)
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()

		probs = torch.sigmoid(logits)
		preds = (probs >= 0.5).float()
		correct = (preds == labels).sum().item()
		return loss.item(), correct, labels.size(0)

	def train_epoch(self, train_loader):
		self.model.train()
		total_loss = 0.0
		total_correct = 0.0
		total_samples = 0
		batch_losses = []

		pbar = tqdm(train_loader, desc='Training', leave=False)
		for images, labels in pbar:
			loss_val, correct, n = self._step_batch(images, labels, train_mode=True)
			total_loss += loss_val * n
			total_correct += correct
			total_samples += n
			batch_losses.append(loss_val)
			pbar.set_postfix({'loss': f'{loss_val:.4f}'})

		return total_loss / total_samples, total_correct / total_samples, batch_losses

	def validate_epoch(self, valid_loader):
		self.model.eval()
		total_loss = 0.0
		total_correct = 0.0
		total_samples = 0

		with torch.no_grad():
			pbar = tqdm(valid_loader, desc='Validation', leave=False)
			for images, labels in pbar:
				loss_val, correct, n = self._step_batch(images, labels, train_mode=False)
				total_loss += loss_val * n
				total_correct += correct
				total_samples += n
				pbar.set_postfix({'loss': f'{loss_val:.4f}'})

		return total_loss / total_samples, total_correct / total_samples

	def save_checkpoint(self, epoch, is_best=False):
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'best_val_acc': self.best_val_acc,
			'best_val_loss': self.best_val_loss,
			'config': self.config,
		}

		ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
		torch.save(checkpoint, ckpt_path)
		if is_best:
			best_path = self.checkpoint_dir / 'best_model.pt'
			torch.save(checkpoint, best_path)
			print(f'New best model saved: {best_path}')

	def train(self, train_loader, valid_loader):
		metrics_path = self.metrics_csv_path
		write_header = self.start_epoch == 0
		open_mode = 'w' if self.start_epoch == 0 else 'a'
		with metrics_path.open(open_mode, newline='') as f:
			writer = csv.writer(f)
			if write_header:
				writer.writerow([
					'epoch',
					'train_loss',
					'train_acc',
					'val_loss',
					'val_acc',
					'backbone_lr',
					'head_lr',
					'epoch_time_sec',
				])

			for epoch in range(self.start_epoch, self.config['num_epochs']):
				print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
				epoch_start = time.perf_counter()

				train_loss, train_acc, batch_losses = self.train_epoch(train_loader)
				self._save_batch_losses(epoch, batch_losses)
				val_loss, val_acc = self.validate_epoch(valid_loader)

				backbone_lr = self.optimizer.param_groups[0]['lr'] if len(self.optimizer.param_groups) > 1 else 0.0
				head_lr = self.optimizer.param_groups[-1]['lr']

				self.scheduler.step()

				is_best = val_loss < self.best_val_loss
				if is_best:
					self.best_val_loss = val_loss
					self.best_val_acc = val_acc

				self.save_checkpoint(epoch, is_best=is_best)

				elapsed = time.perf_counter() - epoch_start
				writer.writerow([
					epoch + 1,
					f'{train_loss:.6f}',
					f'{train_acc:.6f}',
					f'{val_loss:.6f}',
					f'{val_acc:.6f}',
					f'{backbone_lr:.8f}',
					f'{head_lr:.8f}',
					f'{elapsed:.3f}',
				])
				f.flush()
				self.val_loss_history.append(val_loss)
				self.val_loss_batch_positions.append(len(self.batch_loss_history))

				m, s = divmod(elapsed, 60)
				print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
				print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')
				print(f'Backbone LR: {backbone_lr:.6e} | Head LR: {head_lr:.6e}')
				print(f'Epoch Time: {int(m):02d}:{s:05.2f}')

				if self._is_val_loss_worsening_last_epochs(window=4):
					print('Early stopping: validation loss worsened over the last 4 epochs.')
					break

		print('\nTraining complete')
		print(f'Best Val Acc: {self.best_val_acc:.4f}')
		print(f'Best Val Loss: {self.best_val_loss:.4f}')
		print(f'Metrics saved: {metrics_path}')
		self.plot_training_losses()


def get_default_config():
	return {
		'data_dir': './data',
		'checkpoint_dir': './checkpoints/vit_small_dino',
		'batch_size': 32,
		'num_workers': 4,
		'num_epochs': 45,
		'warmup_epochs': 5,
		'backbone_lr': 1e-4,
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
		'auto_resume': True,
		'resume_checkpoint': '',
	}


def main():
	# Debug config used when launched without CLI args (F5 in VS Code).
	DEBUG_CONFIG = {
		'data_dir': './data',
		'checkpoint_dir': './checkpoints/vit_small_dino',
		'batch_size': 32,
		'num_workers': 0,
		'num_epochs': 45,
		'warmup_epochs': 5,
		'backbone_lr': 1e-4,
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
		'auto_resume': True,
		'resume_checkpoint': '',
	}

	parser = argparse.ArgumentParser(description='Train DINO-pretrained ViT-Small on PatchCamelyon')
	parser.add_argument('--data-dir', type=str, default='./data')
	parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/vit_small_dino')
	parser.add_argument('--batch-size', type=int, default=32)
	parser.add_argument('--num-workers', type=int, default=4)
	parser.add_argument('--epochs', type=int, default=45)
	parser.add_argument('--warmup-epochs', type=int, default=5)
	parser.add_argument('--backbone-lr', type=float, default=1e-4)
	parser.add_argument('--head-lr', type=float, default=5e-5)
	parser.add_argument('--weight-decay', type=float, default=0.05)
	parser.add_argument('--freeze-backbone', action='store_true')
	parser.add_argument('--no-amp', action='store_true', help='Disable AMP fp16 training')
	parser.add_argument('--loss-type', type=str, default='bce', choices=['bce', 'focal'])
	parser.add_argument('--focal-gamma', type=float, default=2.0)
	parser.add_argument('--use-pos-weight', action='store_true')
	parser.add_argument('--pos-weight', type=float, default=None)
	parser.add_argument('--no-safe-augmentations', action='store_true', help='Disable pathology-safe train augmentations')
	parser.add_argument('--no-auto-resume', action='store_true')
	parser.add_argument('--resume-checkpoint', type=str, default='')
	args = parser.parse_args()

	if len(sys.argv) == 1:
		config = DEBUG_CONFIG
		print('>> Running in DEBUG mode (using DEBUG_CONFIG) <<')
	else:
		config = get_default_config()
		config['data_dir'] = args.data_dir
		config['checkpoint_dir'] = args.checkpoint_dir
		config['batch_size'] = args.batch_size
		config['num_workers'] = args.num_workers
		config['num_epochs'] = args.epochs
		config['warmup_epochs'] = args.warmup_epochs
		config['backbone_lr'] = args.backbone_lr
		config['head_lr'] = args.head_lr
		config['weight_decay'] = args.weight_decay
		config['freeze_backbone'] = args.freeze_backbone
		config['use_amp'] = not args.no_amp
		config['loss_type'] = args.loss_type
		config['focal_gamma'] = args.focal_gamma
		config['use_pos_weight'] = args.use_pos_weight
		config['auto_pos_weight'] = args.pos_weight is None
		config['pos_weight'] = args.pos_weight
		config['use_safe_augmentations'] = not args.no_safe_augmentations
		config['auto_resume'] = not args.no_auto_resume
		config['resume_checkpoint'] = args.resume_checkpoint

	print('\n' + '=' * 60)
	print('Configuration:')
	print('=' * 60)
	for k, v in config.items():
		print(f'  {k}: {v}')
	print('=' * 60 + '\n')

	trainer = ViTSmallTrainer(config)
	train_loader, valid_loader = trainer.get_dataloaders()
	trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
	main()