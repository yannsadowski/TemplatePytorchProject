import numpy as np
import os
import random
import pandas as pd
from typing import Any
import importlib
import torch
from torch.utils.data import Dataset, DataLoader


def get_transformer(class_path: str) -> Any:
    """
    Dynamically load a transformer class from a string path.
    
    Args:
        class_path (str): The full path to the transformer class as a string.
    
    Returns:
        transformer_class (class): Loaded transformer class.
    """
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    transformer_class = getattr(module, class_name)
    return transformer_class


# Dataset class for PyTorch
class Dataset(Dataset):
    """
    Custom PyTorch Dataset class to handle data and labels.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Function to create sequences for LSTM
def create_sequences(data, sequence_size):
    sequences = []
    for i in range(len(data) - sequence_size + 1):
        sequences.append(data[i:i + sequence_size])
    return np.array(sequences)

# Main function to transform data and create DataLoaders
def prepare_dataloaders(config):
    """
    Prepare the dataset by loading it from a file, transforming features, calculating the target, 
    and splitting it into training, validation, and test sets. Also creates DataLoaders for PyTorch.
    
    Args:
        config (dict): Configuration dictionary with dataset path, transformer, and data splits.
    
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders for training, validation, and test sets.
        input_size (int): The size of the input features.
    """
    # Get the current directory where this script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Move two directories up
    base_directory = os.path.abspath(os.path.join(current_directory, '../../'))

    # Build the full path using the relative path from config
    full_path = os.path.join(base_directory, config['path'])

    # Load the Iris dataset from CSV file
    df = pd.read_csv(full_path)
    
    # Using pandas' str accessor for splitting and selecting the month part
    df['month'] = df['month'].str.split('-').str[1].astype(int) / 12

    # Calculate the percentage change in total passengers and shift the values to align with future targets, then remove any NaN values
    df['Target'] = (df['total_passengers'].pct_change().shift(-1)) * 100
    df = df.dropna()

    # 1. Extract features (X) and Target value (y)
    X =  df.drop(columns=['Target'])  # All columns expect target
    y = df['Target']

    X = create_sequences(X, config['sequences_size'])
    y = y[:len(y) - config['sequences_size'] + 1]

    
    # 2. Apply the transformation defined in config['transform'] (e.g., a scaler)
    if 'transform' in config and config['transform']:
        # Get the transformer class (e.g., StandardScaler)
        transformer_class = get_transformer(config['transform'])
        transformer = transformer_class()
        
        # Extract the second feature (index 1) across all sequences
        second_feature = X[:, :, 1]  # Shape: (num_sequences, sequence_size)
        
        # Apply transformation to the second feature
        second_feature_transformed = transformer.fit_transform(second_feature)
        
        # Reassign the transformed second feature back into X
        X[:, :, 1] = second_feature_transformed


    # 4. Split the data into training, validation, and test sets
    train_size = config['dist']['train']
    val_size = config['dist']['val']
    
    print("len(X): ",len(X))
    print("len(y): ",len(y))
    
    # Determine the number of samples in each set
    total_size = len(X)
    train_end = int(train_size * total_size)
    val_end = train_end + int(val_size * total_size)

    # Split the data
    train_data, val_data, test_data = X[:train_end], X[train_end:val_end], X[val_end:]
    train_labels, val_labels, test_labels = y[:train_end], y[train_end:val_end], y[val_end:]

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    print(f"Train target shape: {train_labels.shape}")
    print(f"Validation target shape: {val_labels.shape}")
    print(f"Test target shape: {test_labels.shape}")

    # 5. Create PyTorch datasets
    train_dataset = Dataset(train_data, train_labels)
    val_dataset = Dataset(val_data, val_labels)
    test_dataset = Dataset(test_data, test_labels)

    # 6. Create DataLoaders for PyTorch
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.shape[-1]


def transform_data(data, transformer_name):
    """
    Apply transformation to the input data using a transformer.
    
    Args:
        data (array-like): The input data to be transformed.
        transformer_name (object): Name of the transformer to import it.
    
    Returns:
        data (array-like): The transformed data.
    """
    # Extract the second feature (index 1) across all sequences
    second_feature = data[:, 1]  # Shape: (sequence_size,num_features)
    
    # Get transformer
    transformer_class = get_transformer(transformer_name)
    transformer = transformer_class()
   
    # Apply transformation to the second feature
    second_feature_transformed = transformer.fit_transform(second_feature.reshape(-1, 1))
    
    # Reassign the transformed second feature back into X
    data[:, 1] = second_feature_transformed.flatten() 
    
    
    return data 
