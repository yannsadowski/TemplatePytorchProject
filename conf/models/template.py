from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    hidden_size_multiplier: int
    num_layers_dense: int
    dropout: float
    input_channels: int
    num_classes: int
    num_conv_layers: int
    kernel_size: int
    

