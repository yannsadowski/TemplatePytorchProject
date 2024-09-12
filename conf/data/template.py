from dataclasses import dataclass, field
from typing import Any

@dataclass
class DatasetConfig:
    train: float
    val: float
    test: float
    transform: Any = field(default=None)
    sequences_size: int





