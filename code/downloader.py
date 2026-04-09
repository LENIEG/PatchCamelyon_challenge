"""
PyTorch-based data loading and augmentation for Camelyon Patch dataset
Equivalent functionality to Keras HDF5Matrix and ImageDataGenerator
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def preprocessing_fn(img):
    """Normalize image values to [0, 1]."""
    if isinstance(img, Image.Image):
        return np.array(img) / 255.0
    return img


class HDF5Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading data from HDF5 files.
    Equivalent to Keras HDF5Matrix functionality.
    """
    
    def __init__(self, x_path, y_path, dataset_key_x='x', dataset_key_y='y', 
                 transform=None, preprocessing_function=None):
        """
        Args:
            x_path (str): Path to HDF5 file containing input images
            y_path (str): Path to HDF5 file containing labels
            dataset_key_x (str): Key name for images in HDF5 file (default: 'x')
            dataset_key_y (str): Key name for labels in HDF5 file (default: 'y')
            transform (callable): Optional torchvision transforms to apply
            preprocessing_function (callable): Preprocessing function (e.g., normalization)
        """
        self.x_file = x_path
        self.y_file = y_path
        self.dataset_key_x = dataset_key_x
        self.dataset_key_y = dataset_key_y
        self.transform = transform
        self.preprocessing_function = preprocessing_function
        
        # Open files to get length
        with h5py.File(x_path, 'r') as f:
            self.length = len(f[dataset_key_x])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load data on-the-fly to avoid loading entire dataset into memory
        with h5py.File(self.x_file, 'r') as fx:
            x_data = fx[self.dataset_key_x][idx]
        
        with h5py.File(self.y_file, 'r') as fy:
            y_data = fy[self.dataset_key_y][idx]
        
        # Convert to PIL Image if needed for transforms
        if isinstance(x_data, np.ndarray):
            # Handle different image formats (uint8, float32, etc.)
            if x_data.dtype == np.uint8:
                x_data = Image.fromarray(x_data)
            else:
                # Assume RGB float image, convert to uint8
                x_data = Image.fromarray((x_data * 255).astype(np.uint8))
        
        # Apply preprocessing function (e.g., normalization)
        if self.preprocessing_function is not None:
            x_data = self.preprocessing_function(x_data)
        
        # Apply transforms (augmentation)
        if self.transform is not None:
            x_data = self.transform(x_data)
        else:
            # Default: convert to tensor
            x_data = transforms.ToTensor()(x_data)
        
        # Convert label to tensor
        y_data = torch.tensor(y_data, dtype=torch.long)
        
        return x_data, y_data


def get_augmented_dataloader(x_path, y_path, batch_size=32, shuffle=True,
                            num_workers=0, preprocessing_function=None):
    """
    Create a PyTorch DataLoader with image augmentation equivalent to Keras ImageDataGenerator.
    
    Augmentations applied:
    - Normalization (preprocessing_function)
    - Random horizontal shifts (width_shift_range=4)
    - Random vertical shifts (height_shift_range=4)
    - Random horizontal flip
    - Random vertical flip
    
    Args:
        x_path (str): Path to HDF5 file with input images
        y_path (str): Path to HDF5 file with labels
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker threads for data loading
        preprocessing_function (callable): Preprocessing function for normalization
    
    Returns:
        DataLoader: PyTorch DataLoader with augmentation
    """
    
    # Define augmentation pipeline equivalent to Keras ImageDataGenerator
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=0,
            translate=(4/96, 4/96)  # width_shift_range=4, height_shift_range=4
            # Assuming 96x96 images (standard Camelyon patch size)
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # horizontal_flip=True
        transforms.RandomVerticalFlip(p=0.5),    # vertical_flip=True
    ])
    
    # Create custom dataset
    dataset = HDF5Dataset(
        x_path=x_path,
        y_path=y_path,
        dataset_key_x='x',
        dataset_key_y='y',
        transform=transform,
        preprocessing_function=preprocessing_function
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def example_usage():
    """
    Example usage of the augmented data loader for model training.
    Equivalent to Keras model.fit_generator() call.
    """
    
    # Configuration
    batch_size = 32
    epochs = 32 # 1024
    data_dir = './data' 
    
    # Create data loader with augmentation
    x_path = f'{data_dir}/camelyonpatch_level_2_split_train_x.h5/camelyonpatch_level_2_split_train_x.h5'
    y_path = f'{data_dir}/camelyonpatch_level_2_split_train_y.h5/camelyonpatch_level_2_split_train_y.h5'
    train_loader = get_augmented_dataloader(
        x_path=x_path,
        y_path=y_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        preprocessing_function=preprocessing_fn
    )

    print("data paths:", f'x:  {x_path}; ', f'y: {y_path}')
    print("data loader created successfully check:   ", {train_loader})
    
    # Example training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assuming you have a model defined
    # model = YourModel().to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    # criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Equivalent to: model.fit_generator(datagen.flow(...), steps_per_epoch=...)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            # output = model(data)
            # loss = criterion(output, target)
            
            # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Shape: {data.shape}, Label0: {target[0].item()}')
                if False:
                    # save random image from batch to look at augmentation
                    rand_ndx = np.random.randint(0, data.shape[0])
                    img = data[rand_ndx].cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                    img = (img * 255).astype(np.uint8)  # Convert back to uint8
                    Image.fromarray(img).save(f'tmp/augmented_image_epoch{epoch}_batch{batch_idx}_{rand_ndx}_label{target[rand_ndx].item()}.png')
        
        print(f'Epoch {epoch + 1}/{epochs} completed')


if __name__ == '__main__':
    # Run example usage
    example_usage()
