from dataclasses import dataclass


@dataclass
class TrainerConfig:
    learning_rate: float
    epochs: int
    early_stopping: bool
    early_stopping_patience: int
