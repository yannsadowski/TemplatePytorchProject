from dataclasses import dataclass, field
from typing import Any

@dataclass
class DatasetConfig:
    train: float
    val: float
    test: float
    data_augment: bool





