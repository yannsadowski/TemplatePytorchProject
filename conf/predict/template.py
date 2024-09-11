from dataclasses import dataclass


@dataclass
class PredictConfig:
    input_size: int
    output_size: int
    class_mapping_path: str
    model_path: str
    transformer_path: str
