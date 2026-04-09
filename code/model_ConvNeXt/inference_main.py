"""
ConvNeXt-Tiny inference script on Camelyon Patch test split.
Supports running from command line or VS Code debugger.
Saves per-sample predictions for model-to-model comparison.

Example command-line usage:
python code/model_ConvNeXt/inference_main.py --checkpoint-path ./checkpoints/convnext_tiny/best_model.pt --output-dir ./results/convnext_tiny
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import convnext_tiny

# Add parent directory to path for importing downloader
sys.path.insert(0, str(Path(__file__).parent.parent))
from downloader import HDF5Dataset, preprocessing_fn


def build_model(num_classes=2):
	"""Build ConvNeXt-Tiny architecture matching the training script."""
	model = convnext_tiny(weights=None)
	in_features = model.classifier[-1].in_features
	model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
	return model


def build_test_loader(data_dir, batch_size, num_workers):
	"""Create test DataLoader from H5 files with no augmentation."""
	test_x_path = f"{data_dir}/camelyonpatch_level_2_split_test_x.h5/camelyonpatch_level_2_split_test_x.h5"
	test_y_path = f"{data_dir}/camelyonpatch_level_2_split_test_y.h5/camelyonpatch_level_2_split_test_y.h5"

	test_transform = transforms.Compose([
		transforms.ToTensor(),
	])

	test_dataset = HDF5Dataset(
		x_path=test_x_path,
		y_path=test_y_path,
		transform=test_transform,
		preprocessing_function=preprocessing_fn,
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True if torch.cuda.is_available() else False,
	)

	return test_loader


def load_checkpoint_weights(model, checkpoint_path, device):
	"""Load model weights from checkpoint path.

	Supports both:
	- training checkpoint dict with key 'model_state_dict'
	- raw model state_dict
	"""
	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		model.load_state_dict(checkpoint)


def run_inference(config):
	"""Run inference on the test split and save prediction results."""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	checkpoint_path = Path(config['checkpoint_path'])
	output_dir = Path(config['output_dir'])
	output_dir.mkdir(parents=True, exist_ok=True)

	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	print(f"Loading checkpoint: {checkpoint_path}")
	model = build_model(num_classes=2)
	load_checkpoint_weights(model, checkpoint_path, device)
	model.to(device)
	model.eval()

	test_loader = build_test_loader(
		data_dir=config['data_dir'],
		batch_size=config['batch_size'],
		num_workers=config['num_workers'],
	)

	total = 0
	correct = 0
	results_rows = []
	global_index = 0

	start = time.perf_counter()
	with torch.no_grad():
		pbar = tqdm(test_loader, desc='Inference', leave=False)
		for images, labels in pbar:
			images = images.to(device).float()
			labels = labels.to(device).squeeze().long()

			logits = model(images)
			probs = torch.softmax(logits, dim=1)
			preds = torch.argmax(probs, dim=1)

			total += labels.size(0)
			batch_correct = (preds == labels).sum().item()
			correct += batch_correct
			batch_acc = batch_correct / labels.size(0)
			running_acc = correct / total if total > 0 else 0.0
			pbar.set_postfix({'batch_acc': f'{batch_acc:.4f}', 'running_acc': f'{running_acc:.4f}'})

			for i in range(labels.size(0)):
				results_rows.append([
					global_index,
					int(labels[i].item()),
					int(preds[i].item()),
					float(logits[i, 0].item()),
					float(logits[i, 1].item()),
					float(probs[i, 0].item()),
					float(probs[i, 1].item()),
				])
				global_index += 1

	elapsed_sec = time.perf_counter() - start
	accuracy = correct / total if total > 0 else 0.0

	predictions_path = output_dir / config['predictions_file']
	with predictions_path.open('w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow([
			'sample_index',
			'y_true',
			'y_pred',
			'logit_class_0',
			'logit_class_1',
			'prob_class_0',
			'prob_class_1',
		])
		writer.writerows(results_rows)

	summary = {
		'model_name': 'convnext_tiny',
		'predictions_schema': 'sample_index,y_true,y_pred,logit_class_0,logit_class_1,prob_class_0,prob_class_1',
		'checkpoint_path': str(checkpoint_path),
		'num_samples': total,
		'accuracy': accuracy,
		'elapsed_seconds': elapsed_sec,
		'data_dir': config['data_dir'],
		'batch_size': config['batch_size'],
	}

	summary_path = output_dir / config['summary_file']
	with summary_path.open('w', encoding='utf-8') as f:
		json.dump(summary, f, indent=2)

	print('\n' + '=' * 60)
	print('Inference complete')
	print('=' * 60)
	print(f"Test samples: {total}")
	print(f"Accuracy: {accuracy:.6f}")
	print(f"Elapsed time (sec): {elapsed_sec:.2f}")
	print(f"Saved predictions: {predictions_path}")
	print(f"Saved summary: {summary_path}")
	print('=' * 60)


def get_default_config():
	"""Return default inference configuration."""
	return {
		'data_dir': './data',
		'checkpoint_path': './checkpoints/convnext_tiny/best_model.pt',
		'output_dir': './results/convnext_tiny',
		'predictions_file': 'test_predictions.csv',
		'summary_file': 'test_summary.json',
		'batch_size': 64,
		'num_workers': 4,
	}


def main():
	"""Main inference entry point."""

	# ------------------------------------------------------------------ #
	# DEBUG CONFIG — edit this block to run/debug directly in VS Code     #
	# (press F5 or right-click -> Run Python File in Terminal)            #
	# When command-line arguments are supplied this block is ignored.     #
	# ------------------------------------------------------------------ #
	DEBUG_CONFIG = {
		'data_dir': './data',
		'checkpoint_path': './checkpoints/convnext_tiny/best_model.pt',
		'output_dir': './results/convnext_tiny',
		'predictions_file': 'test_predictions.csv',
		'summary_file': 'test_summary.json',
		'batch_size': 64,
		'num_workers': 0,
	}

	parser = argparse.ArgumentParser(description='Run ConvNeXt-Tiny inference on test split')
	parser.add_argument('--data-dir', type=str, default='./data', help='Path to data directory')
	parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/convnext_tiny/best_model.pt',
						help='Path to trained checkpoint file')
	parser.add_argument('--output-dir', type=str, default='./results/convnext_tiny',
						help='Directory for saving inference outputs')
	parser.add_argument('--predictions-file', type=str, default='test_predictions.csv',
						help='Filename for per-sample prediction CSV')
	parser.add_argument('--summary-file', type=str, default='test_summary.json',
						help='Filename for summary JSON')
	parser.add_argument('--batch-size', type=int, default=64, help='Inference batch size')
	parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker count')

	args = parser.parse_args()

	# If no CLI arguments were passed (e.g. launched from VS Code debugger),
	# use DEBUG_CONFIG; otherwise build from parsed arguments.
	if len(sys.argv) == 1:
		config = DEBUG_CONFIG
		print('>> Running in DEBUG mode (using DEBUG_CONFIG) <<')
	else:
		config = get_default_config()
		config['data_dir'] = args.data_dir
		config['checkpoint_path'] = args.checkpoint_path
		config['output_dir'] = args.output_dir
		config['predictions_file'] = args.predictions_file
		config['summary_file'] = args.summary_file
		config['batch_size'] = args.batch_size
		config['num_workers'] = args.num_workers

	print('\n' + '=' * 60)
	print('Inference Configuration:')
	print('=' * 60)
	for key, value in config.items():
		print(f"  {key}: {value}")
	print('=' * 60 + '\n')

	run_inference(config)


if __name__ == '__main__':
	main()
