from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    hidden_size_multiplier: int
    num_layers_dense: int
    dropout: float
    norm_type: str

