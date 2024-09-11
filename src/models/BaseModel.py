import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm

class BaseModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers_dense, hidden_size_multiplier=1.0, dropout=0.2, norm_type=None):
        super(BaseModel, self).__init__()
        
        # Calculate hidden_size using the multiplier
        hidden_size = int(input_size * hidden_size_multiplier)
        
        self.dropout_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # Dense layers with Dropout
        self.dense_layers = nn.ModuleList()
        dense_input_size = input_size  # Start with the input size of the model
        for i in range(num_layers_dense):
            dense_output_size = output_size if i == num_layers_dense - 1 else hidden_size
            self.dense_layers.append(nn.Linear(dense_input_size, dense_output_size))
            if i < num_layers_dense - 1:
                self.dense_layers.append(nn.ReLU())
                self.dense_layers.append(nn.Dropout(dropout))
            dense_input_size = hidden_size

    def forward(self, x):
        # Passing through Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        return x


