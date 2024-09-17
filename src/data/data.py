import numpy as np
import os
import random
import pandas as pd
from typing import Any
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image



def prepare_dataloaders(config):
    """
    Prepare the dataset by loading it from a file, transforming features, encoding labels, 
    and splitting it into training, validation, and test sets. Also creates DataLoaders for PyTorch.
    
    Args:
        config (dict): Configuration dictionary with dataset path, transformer, and data splits.
    
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders for training, validation, and test sets.
    """
    
    # Get the current directory where this script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Move two directories up
    base_directory = os.path.abspath(os.path.join(current_directory, '../../'))
    
    # Build the full path using the relative path from config
    full_path = os.path.join(base_directory, config['path'])
    
    # Build the full path for CIFAR-10 pictures
    picture_path = os.path.join(full_path, 'train')

    # Load the dataset (assuming CSV file for labels)
    df = pd.read_csv(os.path.join(full_path, 'trainLabels.csv'))

    # 1. Extract features (X) and labels (y)
    X = df.iloc[:, :-1].values  # All columns except the last one (features)
    y = df.iloc[:, -1].values   # Last column (target labels)

    # 2. Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Split the data into training, validation, and test sets
    train_size = config['dist']['train']
    val_size = config['dist']['val']
    
    # Determine the number of samples in each set
    total_size = len(X)
    train_end = int(train_size * total_size)
    val_end = train_end + int(val_size * total_size)

    # Split the data
    train_data, val_data, test_data = X[:train_end], X[train_end:val_end], X[val_end:]
    train_labels, val_labels, test_labels = y_encoded[:train_end], y_encoded[train_end:val_end], y_encoded[val_end:]

    print(f"Train data shape: {len(train_data)} images")
    print(f"Validation data shape: {len(val_data)} images")
    print(f"Test data shape: {len(test_data)} images")

    # 4. Create transformations for data augmentation (if enabled)
    if config.get('data_augment', False):
        train_transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.33),  # Flip the image with 50% probability
            transforms.RandomRotation(degrees=10),   # Random rotation within 15 degrees
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),  # Resize and crop
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
            transforms.ToTensor()  # Convert to tensor (this will handle converting PIL image to tensor)
        ])
    else:
        train_transformations = transforms.Compose([
            transforms.ToTensor()  # Only convert to tensor if no augmentation is applied
        ])
    
    # Apply same transformations for validation and test (no augmentation)
    val_test_transformations = transforms.Compose([
        transforms.ToTensor()  # Convert to tensor
    ])
    
    # 5. Create PyTorch datasets (with transformations)
    class CustomImageDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths  # Instead of preloading, store the paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            # Load the image on demand
            img_path = os.path.join(picture_path, f'{self.image_paths[idx][0]}.png')
            image = Image.open(img_path)

            # Apply the transform (if provided)
            if self.transform:
                image = self.transform(image)  # Pass the PIL image directly to the transform

            label = self.labels[idx]
            return image, label

    # Creating datasets with the corresponding transformations
    train_dataset = CustomImageDataset(train_data, train_labels, transform=train_transformations)
    val_dataset = CustomImageDataset(val_data, val_labels, transform=val_test_transformations)
    test_dataset = CustomImageDataset(test_data, test_labels, transform=val_test_transformations)

    # 6. Create DataLoaders for PyTorch
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 7. Create and save the class mapping file
    class_mapping = {i: str(class_name) for i, class_name in enumerate(label_encoder.classes_)}
    mapping_file_path = 'class_mapping.yaml'
    with open(mapping_file_path, 'w') as f:
        yaml.dump({'classes': class_mapping}, f, default_flow_style=False)

    return train_loader, val_loader, test_loader






def transform_data(data, transformer):
    """
    Apply transformation to the input data using a pre-trained transformer.
    
    Args:
        data (array-like): The input data to be transformed.
        transformer (object): The pre-trained transformer object.
    
    Returns:
        transformed_data (array-like): The transformed data.
    """
    # Fit the transformer on the entire dataset if needed; here we fit on the row for this example
    transformed_data = transformer.transform(data.reshape(1, -1))  # Transform the row
    
    return transformed_data.flatten()  # Flatten to return the transformed row in original shape
