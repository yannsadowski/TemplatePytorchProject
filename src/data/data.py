import numpy as np
import os
import random
import pandas as pd
from typing import Any
import importlib
from sklearn.preprocessing import LabelEncoder
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_transformer(class_path: str) -> Any:
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    transformer_class = getattr(module, class_name)
    return transformer_class

# Function to divide files into training, validation and test sets
def split_data(files, dist):
    random.shuffle(files)
    train_end = int(dist['train_size'] * len(files))
    val_end = train_end + int(dist['validation_size'] * len(files))
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    return train_files, val_files, test_files

# Function to create sequences for LSTM
def create_sequences(data, sequence_size):
    sequences = []
    for i in range(len(data) - sequence_size + 1):
        sequences.append(data[i:i + sequence_size])
    return np.array(sequences)

# Dataset class for PyTorch
class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Main function to transform data and create DataLoaders
def prepare_dataloaders(config):
    # Load the configuration file
    # pathRaw = config['path']['path_raw']

    
    train_data, val_data, test_data = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    # HERE 
    # 1. Create a thread pool for parallel processing to allow multiple tasks to run concurrently

    # 2. Submit tasks for each genre (or data category) to process data asynchronously

    # 3. Process data for each genre according to the project needs (e.g., transformations, feature extraction, splitting, etc.)

    # 4. Store processed data and any associated results or labels for future use

    # 5. Handle errors during processing without stopping the entire operation, logging any issues and continuing other tasks

    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Create PyTorch datasets
    train_dataset = Dataset(train_data, train_labels)
    val_dataset = Dataset(val_data, val_labels)
    test_dataset = Dataset(test_data, test_labels)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and save the class mapping file
    # Need it to be able to found the class after inference
    class_mapping = {i: str(genre) for i, genre in enumerate(label_encoder.classes_)}
    # Specify the path of the class mapping file to root
    mapping_file_path = 'class_mapping.yaml'

    # Save the mapping to a YAML file at root
    with open(mapping_file_path, 'w') as f:
        yaml.dump({'classes': class_mapping}, f, default_flow_style=False)
    
    return train_loader, val_loader, test_loader, len(genres), train_data.shape[2]





