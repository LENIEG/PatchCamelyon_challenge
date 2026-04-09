"""
ConvNeXt-Tiny training script with ImageNet pretrained weights.
Supports fine-tuning with optional backbone weight freezing.
"""

"""
Example usage

# Full fine-tuning (all weights trainable)
python training_main.py --epochs 32 --batch-size 32 --lr 1e-4

# Backbone frozen (fine-tune only head)
python training_main.py --freeze-backbone --epochs 32 --batch-size 32 --lr 1e-4

# Custom settings
python training_main.py --freeze-backbone --epochs 50 --lr 5e-5 --batch-size 64

# To disable auto-resume and start fresh (useful for debugging or new runs)
python code/model_ConvNeXt/training_main.py --no-auto-resume --epochs 32 --batch-size 32 --lr 1e-4

# To resume from a specific checkpoint (overrides latest search)
python code/model_ConvNeXt/training_main.py --resume-checkpoint ./checkpoints/convnext_tiny/checkpoint_epoch_10.pt --epochs 32 --batch-size 32 --lr 1e-4
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import numpy as np
from tqdm import tqdm

# Add parent directory to path for importing downloader
sys.path.insert(0, str(Path(__file__).parent.parent))
from downloader import get_augmented_dataloader, preprocessing_fn


class ConvNeXtTrainer:
    """
    Trainer class for ConvNeXt-Tiny model on Camelyon Patch dataset.
    Supports fine-tuning with optional backbone freezing.
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config (dict): Configuration dictionary with training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._build_optimizer()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['lr_factor'],
            patience=config['lr_patience'],
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.start_epoch = 0
        self.batch_loss_history = []
        self.batch_loss_csv_path = self.checkpoint_dir / 'batch_loss_history.csv'

        self.batch_loss_history = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # Resume from checkpoint if requested and available.
        self._resume_if_available()

        # Initialize or continue batch loss history logging.
        self._initialize_batch_loss_csv()

        # Best metrics
        print(f"✓ Training will start from epoch index: {self.start_epoch}")

    def _find_latest_checkpoint(self):
        """Find latest epoch checkpoint in checkpoint directory."""
        checkpoint_candidates = []
        for path in self.checkpoint_dir.glob('checkpoint_epoch_*.pt'):
            try:
                epoch_num = int(path.stem.split('_')[-1])
                checkpoint_candidates.append((epoch_num, path))
            except ValueError:
                continue

        if not checkpoint_candidates:
            return None

        checkpoint_candidates.sort(key=lambda x: x[0])
        return checkpoint_candidates[-1][1]

    def _load_checkpoint(self, checkpoint_path):
        """Load model/optimizer/scheduler state from checkpoint and set start epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = int(checkpoint.get('epoch', -1)) + 1
        self.best_val_loss = float(checkpoint.get('best_val_loss', self.best_val_loss))
        self.best_val_acc = float(checkpoint.get('best_val_acc', self.best_val_acc))

        print(f"✓ Resumed from checkpoint: {checkpoint_path}")
        print(f"✓ Loaded start epoch: {self.start_epoch}")

    def _resume_if_available(self):
        """Resume training from explicit or latest checkpoint if enabled."""
        if not self.config.get('auto_resume', True):
            print("✓ Auto-resume disabled. Starting from ImageNet initialization.")
            return

        resume_checkpoint = self.config.get('resume_checkpoint_path', '')
        checkpoint_path = None

        if resume_checkpoint:
            candidate = Path(resume_checkpoint)
            if candidate.exists():
                checkpoint_path = candidate
            else:
                print(f"! resume_checkpoint_path not found: {candidate}. Falling back to latest checkpoint search.")

        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None:
            print("✓ No checkpoint found to resume. Starting from ImageNet initialization.")
            return

        self._load_checkpoint(checkpoint_path)

    def _initialize_batch_loss_csv(self):
        """Initialize batch loss CSV, or continue from existing file when resuming."""
        if self.start_epoch > 0 and self.batch_loss_csv_path.exists():
            with self.batch_loss_csv_path.open('r', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    self.batch_loss_history.append(float(row['loss']))
            print(f"✓ Loaded existing batch loss history: {len(self.batch_loss_history)} rows")
            return

        with self.batch_loss_csv_path.open('w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'batch', 'global_batch', 'loss'])
        print(f"✓ Initialized batch loss CSV: {self.batch_loss_csv_path}")

    def _build_optimizer(self):
        """Create optimizer with separate learning rates for backbone and head."""
        head_params = [param for param in self.model.classifier.parameters() if param.requires_grad]
        backbone_params = [param for param in self.model.features.parameters() if param.requires_grad]

        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config['backbone_learning_rate'],
                'name': 'backbone',
            })

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': self.config['head_learning_rate'],
                'name': 'head',
            })

        print(f"✓ Backbone LR: {self.config['backbone_learning_rate']}")
        print(f"✓ Head LR: {self.config['head_learning_rate']}")

        return optim.Adam(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
    
    def _build_model(self):
        """
        Build ConvNeXt-Tiny model with ImageNet pretrained weights.
        
        Returns:
            model: ConvNeXt-Tiny model with modified classification head for binary classification
        """
        # Load pretrained ConvNeXt-Tiny
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights)
        
        print("✓ Loaded ConvNeXt-Tiny with ImageNet1K v1 pretrained weights")
        
        # Freeze backbone if requested
        if self.config['freeze_backbone']:
            print("✓ Freezing backbone weights")
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            print("✓ Backbone weights are trainable (fine-tuning mode)")
        
        # Modify classification head for binary classification (cancer/no-cancer)
        # ConvNeXt-Tiny has 1000 output units for ImageNet, we need 2 for binary
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 2)
        
        print(f"✓ Modified classifier head: {num_features} -> 2 (binary classification)")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Trainable parameters: {trainable_params:,} / Total: {total_params:,}")
        
        return model
    
    def _get_dataloaders(self):
        """
        Create train and validation data loaders.
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        data_dir = self.config['data_dir']
        batch_size = self.config['batch_size']
        
        # Training data loader with augmentation
        train_x_path = f'{data_dir}/camelyonpatch_level_2_split_train_x.h5/camelyonpatch_level_2_split_train_x.h5'
        train_y_path = f'{data_dir}/camelyonpatch_level_2_split_train_y.h5/camelyonpatch_level_2_split_train_y.h5'
        
        train_loader = get_augmented_dataloader(
            x_path=train_x_path,
            y_path=train_y_path,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['num_workers'],
            preprocessing_function=preprocessing_fn
        )
        
        # Validation data loader with minimal augmentation (just normalization)
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        val_x_path = f'{data_dir}/camelyonpatch_level_2_split_valid_x.h5/camelyonpatch_level_2_split_valid_x.h5'
        val_y_path = f'{data_dir}/camelyonpatch_level_2_split_valid_y.h5/camelyonpatch_level_2_split_valid_y.h5'
        
        from downloader import HDF5Dataset
        val_dataset = HDF5Dataset(
            x_path=val_x_path,
            y_path=val_y_path,
            transform=val_transform,
            preprocessing_function=preprocessing_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✓ Data loaders created:")
        print(f"  - Train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
        print(f"  - Val samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            tuple: (avg_loss, avg_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_losses = []
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device).float() #  dtype mismatch: images are coming in as float64 (double precision) but the model weights are float32
            labels = labels.to(self.device).squeeze().long()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            loss_value = loss.item()
            total_loss += loss_value * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            batch_losses.append(loss_value)
            
            pbar.set_postfix({'loss': loss_value})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc, batch_losses
    
    def validate(self, val_loader):
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: (avg_loss, avg_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for images, labels in pbar:
                images = images.to(self.device).float()
                labels = labels.to(self.device).squeeze().long()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Metrics
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved: {best_path}")

    def _save_batch_losses(self, epoch, batch_losses):
        """Append current epoch batch losses to CSV and in-memory history."""
        with self.batch_loss_csv_path.open('a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for batch_idx, loss_value in enumerate(batch_losses):
                global_batch = len(self.batch_loss_history)
                self.batch_loss_history.append(loss_value)
                writer.writerow([epoch + 1, batch_idx + 1, global_batch + 1, loss_value])

    def plot_training_losses(self):
        """Save and display the batch loss curve after training finishes."""
        if not self.batch_loss_history:
            print("No batch losses were recorded; skipping plot.")
            return

        smooth_window = 100
        x_values = np.arange(1, len(self.batch_loss_history) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(x_values, self.batch_loss_history, linewidth=1.0, alpha=0.5, label='Batch Loss (raw)')

        if len(self.batch_loss_history) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed_losses = np.convolve(self.batch_loss_history, kernel, mode='valid')
            smoothed_x = np.arange(smooth_window, len(self.batch_loss_history) + 1)
            plt.plot(smoothed_x, smoothed_losses, linewidth=2.0, color='red', label='Batch Loss (100-batch average)')
        else:
            print(f"Not enough batches for {smooth_window}-batch smoothing; showing raw loss only.")

        plt.title('Training Loss After Each Batch')
        plt.xlabel('Global Batch')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = self.checkpoint_dir / 'training_batch_loss.png'
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Saved batch loss plot: {plot_path}")
        plt.show()
    
    def train(self, train_loader, val_loader):
        """
        Full training loop with validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            epoch_start_time = time.perf_counter()
            
            # Train
            train_loss, train_acc, batch_losses = self.train_epoch(train_loader)
            self._save_batch_losses(epoch, batch_losses)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            epoch_elapsed = time.perf_counter() - epoch_start_time
            epoch_minutes, epoch_seconds = divmod(epoch_elapsed, 60)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Epoch Time: {int(epoch_minutes):02d}:{epoch_seconds:05.2f}")
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            if len(current_lrs) == 1:
                print(f"Head LR: {current_lrs[0]:.6f}")
            else:
                print(f"Backbone LR: {current_lrs[0]:.6f}, Head LR: {current_lrs[1]:.6f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            self.save_checkpoint(epoch, is_best=is_best)
            
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
        
        print("\n" + "="*60)
        print(f"Training complete!")
        print(f"Best Val Acc: {self.best_val_acc:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*60)
        self.plot_training_losses()


def get_default_config():
    """Get default configuration."""
    return {
        # Data
        'data_dir': './data',
        'batch_size': 32,
        'num_workers': 4,
        
        # Model
        'freeze_backbone': False,  # Set to True to freeze backbone weights
        
        # Training
        'num_epochs': 32,
        'backbone_learning_rate': 1e-4,
        'head_learning_rate': 1e-5,
        'weight_decay': 1e-5,
        'lr_factor': 0.5,
        'lr_patience': 3,

        # Resume behavior
        'auto_resume': True,
        'resume_checkpoint_path': '',
        
        # Checkpointing
        'checkpoint_dir': './checkpoints/convnext_tiny',
    }


def main():
    """Main training entry point."""

    # ------------------------------------------------------------------ #
    # DEBUG CONFIG — edit this block to run/debug directly in VS Code     #
    # (press F5 or right-click → Run Python File in Terminal)             #
    # When command-line arguments are supplied this block is ignored.     #
    # ------------------------------------------------------------------ #
    DEBUG_CONFIG = {
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
    }

    parser = argparse.ArgumentParser(description='Train ConvNeXt-Tiny on Camelyon Patch dataset')
    parser.add_argument('--freeze-backbone', action='store_true', 
                        help='Freeze backbone weights (fine-tune only classification head)')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=32, 
                        help='Number of training epochs')
    parser.add_argument('--backbone-lr', type=float, default=1e-4, 
                        help='Learning rate for the ConvNeXt backbone')
    parser.add_argument('--head-lr', type=float, default=1e-5, 
                        help='Learning rate for the classifier head')
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='Number of data loading workers')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='Disable automatic resume from last checkpoint')
    parser.add_argument('--resume-checkpoint', type=str, default='',
                        help='Explicit checkpoint path to resume from (overrides latest search)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/convnext_tiny',
                        help='Directory for saving checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')

    args = parser.parse_args()

    # If no CLI arguments were passed (e.g. launched from VS Code debugger),
    # use DEBUG_CONFIG; otherwise build from parsed arguments.
    if len(sys.argv) == 1:
        config = DEBUG_CONFIG
        print(">> Running in DEBUG mode (using DEBUG_CONFIG) <<")
    else:
        config = get_default_config()
        config['freeze_backbone'] = args.freeze_backbone
        config['batch_size'] = args.batch_size
        config['num_epochs'] = args.epochs
        config['backbone_learning_rate'] = args.backbone_lr
        config['head_learning_rate'] = args.head_lr
        config['num_workers'] = args.num_workers
        config['auto_resume'] = not args.no_auto_resume
        config['resume_checkpoint_path'] = args.resume_checkpoint
        config['checkpoint_dir'] = args.checkpoint_dir
        config['data_dir'] = args.data_dir
    
    # Print config
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Create trainer and train
    trainer = ConvNeXtTrainer(config)
    train_loader, val_loader = trainer._get_dataloaders()
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()