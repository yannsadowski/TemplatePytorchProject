# Template for Pytorch project

This template is made for using the Pytorch framework with Hydra for the configuration file and WandB for logging information. The template is designed to work with the CSV file from the Airline Passenger dataset available on Kaggle: [Airline Passenger Dataset](https://www.kaggle.com/datasets/erogluegemen/airline-passengers).

To make it work directly, you need to download the CSV file and place it in a directory named `data`. You should have the following structure:

**Directory:** `D:\code\TemplatePytorchProject\data`

```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         9/11/2024  11:39 AM           4551 iris.data.csv
```

Ensure the file is named `airline-passanger.csv` or modify the script accordingly to match the name. 

Additionally, configuration parameters (such as dataset path, training parameters, etc.) are managed using Hydra. You can modify these settings by editing the configuration files located in the `conf` directory.

This template is based on the project: [TimeSerieClassifierOnMusic](https://github.com/yannsadowski/TimeSerieClassifierOnMusic).

## Project Structure

The main directories and files in the project are structured as follows:

```
TemplatePytorchProject
│
├── conf
│   ├── data
│   ├── models
│   ├── predict
│   ├── trainer
│   ├── wandb
│   ├── default.yaml
│   ├── template.py
│   └── __init__.py
├── data
├── model
├── src
│   ├── data
│   │   ├── data.py
│   ├── models
│   │   ├── Model.py
│   ├── predict
│   │   ├── predict.py
│   ├── trainer
│   │   ├── train.py
├── sweep_file
│   ├── wandb_sweep.yaml
├── main.py
├── predict.py
└─── README.md
```

## Environment Setup

### Create Virtual Environment

1. Create a virtual environment:

    ```sh
    python -m venv dev_env
    ```

2. Activate the virtual environment:

    ```sh
    ./dev_env/Scripts/Activate
    ```

3. Upgrade `pip` and install necessary packages:

    ```sh
    python.exe -m pip install --upgrade pip
    pip install -U pandas
    pip install -U numpy
    pip install -U wandb
    pip install -U hydra-core
    pip install -U torch --index-url https://download.pytorch.org/whl/cu121
    pip install -U torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -U torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -U torchinfo
    pip install -U scikit-learn
    pip install -U matplotlib
    pip install -U tqdm

    ```

### Update Requirements

If you install new packages, update the `requirements.txt` file:

```sh
pip freeze > requirements.txt
```

## Usage

### Training the Model

Run the `main.py` script to train the model:

```sh
python main.py
```

### Wandb Sweep

Use wandb sweep with `sweep_file\wandb_sweep.yaml` file to make the exploration of hyper-parameter of the model:

```sh
wandb sweep .\sweep_file\wandb_sweep.yaml
```

this command will give you the command line to launch a agent.
It's possible to launch multiple agents at once.

### Predicting

Use `predict.py` for using your save model:

```sh
python predict.py
```


## Configuration

Detailed configuration files are located in the `conf` directory. They are organized as follows:

- `conf/data`
- `conf/models`
- `conf/predict`
- `conf/trainer`
- `conf/wandb`

Refer to the respective YAML files for parameter settings and modify them as needed for your experiments.


## Experiment Tracking

We use [Weights & Biases](https://wandb.ai/) for experiment tracking. Ensure you have your API key configured in `conf/wandb/default.yaml`.
