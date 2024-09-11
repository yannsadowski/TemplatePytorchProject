from dataclasses import dataclass


@dataclass
class TrainerConfig:
    learning_rate: float
    epochs: int
