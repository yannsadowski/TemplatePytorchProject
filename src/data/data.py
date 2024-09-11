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
import joblib  


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
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Main function to transform data and create DataLoaders
def prepare_dataloaders(config):
    """
    Prepare the dataset by loading it from a file, transforming features, encoding labels, 
    and splitting it into training, validation, and test sets. Also creates DataLoaders for PyTorch.
    
    Args:
        config (dict): Configuration dictionary with dataset path, transformer, and data splits.
    
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders for training, validation, and test sets.
        num_classes (int): The number of unique classes in the dataset.
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

    # 1. Extract features (X) and labels (y)
    X = df.iloc[:, :-1].values  # All columns except the last one (features)
    y = df.iloc[:, -1].values   # Last column (target labels)

    # 2. Apply the transformation defined in config['transform'] (e.g., a scaler)
    if 'transform' in config and config['transform']:
        transformer_class = get_transformer(config['transform'])
        transformer = transformer_class()
        X = transformer.fit_transform(X)
        
        # Save the transformer with joblib
        joblib.dump(transformer, 'transformer.pkl')

    # 3. Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 4. Split the data into training, validation, and test sets
    train_size = config['dist']['train']
    val_size = config['dist']['val']
    
    # Determine the number of samples in each set
    total_size = len(X)
    train_end = int(train_size * total_size)
    val_end = train_end + int(val_size * total_size)

    # Split the data
    train_data, val_data, test_data = X[:train_end], X[train_end:val_end], X[val_end:]
    train_labels, val_labels, test_labels = y_encoded[:train_end], y_encoded[train_end:val_end], y_encoded[val_end:]

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # 5. Create PyTorch datasets
    train_dataset = Dataset(train_data, train_labels)
    val_dataset = Dataset(val_data, val_labels)
    test_dataset = Dataset(test_data, test_labels)

    # 6. Create DataLoaders for PyTorch
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # 7. Create and save the class mapping file
    class_mapping = {i: str(class_name) for i, class_name in enumerate(label_encoder.classes_)}
    mapping_file_path = 'class_mapping.yaml'
    with open(mapping_file_path, 'w') as f:
        yaml.dump({'classes': class_mapping}, f, default_flow_style=False)

    return train_loader, val_loader, test_loader, len(label_encoder.classes_), train_data.shape[1]


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
