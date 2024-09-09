# Template for Pytorch project

This template is made for using the Pytorch framework with hydra for the configuration file and wandb for the logging information.

## Project Structure

The main directories and files in the project are structured as follows:

```
TimeSerieClassifier
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
    pip install pandas
    pip install numpy
    pip install wandb
    pip install hydra-core
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install matplotlib
    pip install tqm
    ```

### Start Virtual Environment

Activate the virtual environment whenever you start working on the project:

```sh
./dev_env/Scripts/Activate
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