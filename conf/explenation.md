# Time Series Regression Configuration Guide

This guide provides an overview of the configuration setup for the Time Series Regression project. The configuration is organized using YAML files and Python scripts to streamline data processing, model training, prediction, and experiment tracking.

## Directory Structure

The main configuration files and directories are structured as follows:

```
D:\code\TemplatePytorchProject_LSTM\conf
│
├── data
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── models
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── predict
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── trainer
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── wandb
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── __pycache__
├── default.yaml
├── template.py
└── __init__.py
```

### Main Configuration (`template.py`)

The `template.py` file sets up the main configuration class using dataclasses. It imports configuration classes from various submodules.

```python
from dataclasses import dataclass

from .data.template import DatasetConfig
from .models.template import ModelConfig
from .trainer.template import TrainerConfig
from .predict.template import PredictConfig
from .wandb.template import WandbConfig

@dataclass
class MainConfig:
    data: DatasetConfig
    models: ModelConfig
    trainer: TrainerConfig
    predict: PredictConfig
    wandb: WandbConfig
```

### Default Configuration (`default.yaml`)

The `default.yaml` file specifies the default configurations to be used.

```yaml
defaults:
  - _self_
  - data: default
  - models: default
  - trainer: default
  - predict: default
  - wandb: default

hydra:
  job:
    chdir: true
```


## Weights & Biases (Wandb) Configuration

The Wandb configuration is defined in the `wandb/default.yaml` file.

### Wandb Default Configuration (`wandb/default.yaml`)

```yaml
api_key: 
project_name: 
```

## Summary

This setup allows you to manage and customize the configuration for different components of the Time Series Regression project efficiently. By organizing the configuration settings in a structured way, you can easily modify parameters, paths, and other settings to fit your specific needs.