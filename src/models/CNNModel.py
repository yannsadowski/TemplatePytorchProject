import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, num_conv_layers, num_dense_layers, hidden_size_multiplier=1.0, dropout=0.2, kernel_size=3):
        super(CNNModel, self).__init__()

        hidden_size = int(64 * hidden_size_multiplier)

        # Convolutional layers and normalization setup
        self.conv_layers = nn.ModuleList()
        
        conv_input_channels = input_channels

        for i in range(num_conv_layers):
            conv_output_channels = hidden_size
            self.conv_layers.append(nn.Conv2d(conv_input_channels, conv_output_channels, kernel_size=kernel_size, padding=1))
            
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout))

            
            conv_input_channels = hidden_size

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense_layers = nn.ModuleList()
        dense_input_size = hidden_size

        for i in range(num_dense_layers):
            dense_output_size = num_classes if i == num_dense_layers - 1 else hidden_size
            self.dense_layers.append(nn.Linear(dense_input_size, dense_output_size))
            
            if i < num_dense_layers - 1:
                self.dense_layers.append(nn.ReLU())
                self.dense_layers.append(nn.Dropout(dropout))
            
            dense_input_size = hidden_size

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
                
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        for layer in self.dense_layers:
            x = layer(x)

        return x
